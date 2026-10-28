"""Masked ULTRA fine tuning with one phenotype graph traversal at a time.

ULTRA queries are (phenotype, gene, inverse(causes_phenotype)). Its native
training-edge remover does not modulo inverse relation IDs, so these utilities
mask both directions explicitly and run eval-mode forwards WITH gradients.
Stage 3 uses an exact two-pass chain rule rather than retaining a full graph
backward tape for every phenotype. No validation/test labels enter objectives.
"""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def deterministic_gradient_mode(model):
    """Disable ULTRA's native edge dropout, not parameter gradients."""
    modes={module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.enable_grad():
            yield
    finally:
        for module, mode in modes.items():
            module.training=mode


def prepare_mask(graph,phenotype_ids,forward_relation,disease_id=None,*,rebuild_relation_graph=False):
    """Label-independent direct-edge removal, optionally remove a disease node.

    Delete ALL causes edges ending at query phenotypes and all corresponding
    inverse edges, irrespective of candidate/positive membership. Stage 3 must
    also supply its current training disease ID to delete every incident edge.
    Default retains the original train-only relation graph, matching upstream
    ULTRA's own edge-removal policy; optional rebuilding changes that policy.
    Original graph tensors and relation graph remain untouched.
    """
    if graph.num_relations%2:
        raise ValueError('Expected an even forward+inverse relation vocabulary')
    half=graph.num_relations//2
    if not 0<=forward_relation<half:
        raise ValueError('Pass the forward causes_phenotype relation ID')
    phenotype_ids=list(map(int,phenotype_ids))
    disease_ids=[] if disease_id is None else ([int(disease_id)] if np.isscalar(disease_id) else list(map(int,disease_id)))
    phenos=torch.as_tensor(phenotype_ids,dtype=torch.long,device=graph.edge_index.device)
    if phenos.numel()==0:
        raise ValueError('Empty phenotype profile')
    h,t=graph.edge_index
    r=graph.edge_type
    remove=((r==forward_relation)&torch.isin(t,phenos)) | ((r==forward_relation+half)&torch.isin(h,phenos))
    direct_removed=int(remove.sum().item())
    if disease_ids:
        disease_nodes=torch.as_tensor(disease_ids,dtype=torch.long,device=h.device)
        remove |= torch.isin(h,disease_nodes) | torch.isin(t,disease_nodes)
    masked=copy.copy(graph)
    masked.edge_index=graph.edge_index[:,~remove]
    masked.edge_type=graph.edge_type[~remove]
    masked.mask_metadata={'forward_relation':int(forward_relation),'inverse_relation':int(forward_relation+half),
                          'phenotype_ids':[int(x) for x in phenotype_ids],
                          'disease_ids':disease_ids,
                          'direct_edges_removed':direct_removed,'all_edges_removed':int(remove.sum().item()),
                          'relation_graph':'rebuilt-masked' if rebuild_relation_graph else 'original-train-only',
                          'mask_policy':'all-query-phenotype-causes-edges-both-directions-and-optional-current-disease'}
    if rebuild_relation_graph:
        from ultra.tasks import build_relation_graph
        masked=build_relation_graph(masked)
    return masked


def _check_mask(graph,phenotypes):
    if not hasattr(graph,'mask_metadata'):
        raise ValueError('Call prepare_mask before training')
    if not set(map(int,phenotypes))<=set(graph.mask_metadata['phenotype_ids']):
        raise ValueError('Scored phenotype was not masked')
    return graph.mask_metadata['inverse_relation']


def _batch_scores(model,graph,phenotype,genes,relation):
    device=graph.edge_index.device
    genes=torch.as_tensor(genes,dtype=torch.long,device=device)
    batch=torch.stack((torch.full_like(genes,int(phenotype)),genes,torch.full_like(genes,relation)),dim=-1).unsqueeze(0)
    scores=model(graph,batch).reshape(-1)
    if scores.numel()!=len(genes) or not torch.isfinite(scores).all():
        raise FloatingPointError('Invalid ULTRA phenotype score vector')
    return scores


def _eligible(positive_gene_ids,candidate_gene_ids):
    positives=np.asarray(list(positive_gene_ids),dtype=np.int64)
    candidates=np.asarray(list(candidate_gene_ids),dtype=np.int64)
    if len(positives)==0 or len(set(positives))!=len(positives):
        raise ValueError('Positives must be nonempty and unique')
    if len(set(candidates))!=len(candidates):
        raise ValueError('Candidate genes must be unique')
    if not set(positives)<=set(candidates):
        raise ValueError('Positive gene outside fixed candidate universe')
    return positives,candidates[~np.isin(candidates,positives)]


def link_loss(scores,num_positives):
    """Balanced BCE, uniform weights within positive and negative groups."""
    if not 0<num_positives<scores.numel():
        raise ValueError('Need positives and negatives')
    return .5*(F.softplus(-scores[:num_positives]).mean()+F.softplus(scores[num_positives:]).mean())


def profile_loss(phenotype_scores,num_positives,pooling_logit,temperature=1.):
    """Uniform-positive sampled-list cross entropy on max/mean profile scores."""
    if temperature<=0 or not np.isfinite(temperature):
        raise ValueError('Temperature must be finite and positive')
    if phenotype_scores.ndim!=2 or phenotype_scores.shape[0]==0:
        raise ValueError('Expected nonempty phenotype-by-gene scores')
    if not 0<num_positives<phenotype_scores.shape[1]:
        raise ValueError('Need positives and negatives')
    weight=torch.sigmoid(pooling_logit)
    pooled=weight*phenotype_scores.max(dim=0).values+(1-weight)*phenotype_scores.mean(dim=0)
    logits=pooled/temperature
    return torch.logsumexp(logits,dim=0)-logits[:num_positives].mean()


def stage2_backward(model,graph,phenotype_id,positive_gene_ids,candidate_gene_ids,rng,*,negative_count=64,scale=1.):
    """One typed-gene link update; caller owns optimizer zero/step.

    Supply ALL known training gene positives for this phenotype, not a single
    sampled positive: the complete set is excluded from negative sampling.
    """
    relation=_check_mask(graph,[phenotype_id])
    positives,eligible=_eligible(positive_gene_ids,candidate_gene_ids)
    if negative_count<1 or len(eligible)<negative_count:
        raise ValueError('Not enough distinct eligible gene negatives')
    negatives=rng.choice(eligible,negative_count,replace=False)
    genes=np.concatenate((positives,negatives))
    with deterministic_gradient_mode(model):
        scores=_batch_scores(model,graph,phenotype_id,genes,relation)
        loss=link_loss(scores,len(positives))
        (loss*scale).backward()
    return {'loss':float(loss.detach()),'positives':len(positives),'negatives':len(negatives),
            'negative_gene_ids':negatives.tolist(),'graph_forwards':1,'mask':graph.mask_metadata}


def profile_backward(model,graph,phenotype_ids,positive_gene_ids,candidate_gene_ids,pooling_logit,rng,*,
                     random_count=32,hard_count=32,proposal_size=256,temperature=1.,scale=1.):
    """Exact streaming encoder gradients for a multi-phenotype ranking loss.

    First pass: score all positives + random negatives + hard proposals without
    graph tapes. Select hard genes by the current pooled score. Differentiate
    the small score matrix and scalar pool; replay one phenotype at a time and
    backpropagate its exact score gradient. Pooling, list composition and mask
    stay fixed throughout replay. This uses at most one phenotype's ULTRA graph
    tape, independently of profile length. No optimizer step occurs here.
    """
    phenotypes=list(map(int,phenotype_ids))  # Preserve order and multiplicity.
    relation=_check_mask(graph,phenotypes)
    positives,eligible=_eligible(positive_gene_ids,candidate_gene_ids)
    if random_count<0 or hard_count<0 or random_count+hard_count<1 or proposal_size<hard_count:
        raise ValueError('Invalid negative sampling budget')
    if len(eligible)<random_count+hard_count:
        raise ValueError('Not enough distinct eligible gene negatives')
    random=rng.choice(eligible,random_count,replace=False)
    remaining=eligible[~np.isin(eligible,random)]
    proposals=rng.choice(remaining,min(proposal_size,len(remaining)),replace=False) if hard_count else np.array([],dtype=np.int64)
    initial=np.concatenate((positives,random,proposals))
    with deterministic_gradient_mode(model):
        with torch.no_grad():
            first=torch.stack([_batch_scores(model,graph,p,initial,relation) for p in phenotypes])
            weight=pooling_logit.detach().sigmoid()
            pooled=weight*first.max(dim=0).values+(1-weight)*first.mean(dim=0)
            proposal_scores=pooled[len(positives)+len(random):].cpu().numpy()
            chosen=np.lexsort((proposals,-proposal_scores))[:hard_count]
        selected=np.concatenate((np.arange(len(positives)+len(random)),len(positives)+len(random)+chosen))
        genes=initial[selected]
        small=first[:,selected].detach().requires_grad_(True)
        loss=profile_loss(small,len(positives),pooling_logit,temperature)
        (loss*scale).backward()
        gradients=small.grad.detach()
        # Do not accumulate graph tapes across phenotype forwards.
        for i,phenotype in enumerate(phenotypes):
            scores=_batch_scores(model,graph,phenotype,genes,relation)
            scores.backward(gradients[i])
    return {'loss':float(loss.detach()),'positives':len(positives),'random_negatives':random_count,
            'hard_negatives':hard_count,'proposals_scored':len(proposals),'phenotype_count':len(phenotypes),
            'selected_gene_ids':genes.tolist(),'pooling_weight':float(pooling_logit.detach().sigmoid()),
            'graph_forwards':2*len(phenotypes),'mask':graph.mask_metadata,
            'gradient_strategy':'exact-two-pass-score-chain-rule-single-phenotype-tape'}


def objective_metadata():
    return {'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'stage2_objective':'balanced-BCE-all-training-phenotype-positive-genes-versus-unique-candidate-gene-negatives',
            'stage3_objective':'equal-positive-sampled-list-cross-entropy-on-sigmoid-mix-max-mean',
            'native_ultra_edge_dropout':False,'forward_mode':'eval-with-gradients-enabled',
            'mask':'all-query-phenotype-causes-edges-both-directions; current-disease-node-incident-edges-stage3',
            'negative_label_source':'training-only','hard_importance_correction':False,
            'training_batch_phenotypes':1,'profile_gradient':'exact-streaming-two-pass'}
