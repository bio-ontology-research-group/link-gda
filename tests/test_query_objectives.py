"""Mask invariance and streaming-gradient parity without ULTRA dependencies."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from phenoquery import objectives as u


def graph():
    # Forward r1 genes->p10, inverse r3 p10->genes, unrelated r0/r2,
    # plus two materialization-source disease nodes 20 and 21.
    return SimpleNamespace(edge_index=torch.tensor([[0,10,1,10,2,11,20,10,21,5],[10,0,10,1,11,2,10,20,5,21]]),
                           edge_type=torch.tensor([1,3,1,3,1,3,0,2,0,2]),num_relations=4,
                           relation_graph=object())


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.2,-.3],[.4,.1]]));self.calls=[]
    def forward(self,g,batch):
        assert not self.training  # Native edge dropout must be disabled.
        self.calls.append(batch.detach().clone())
        h,t,r=batch.unbind(-1)
        x=torch.stack((t.float()/5,torch.sin(t.float()+h.float())),dim=-1)
        feature=x@self.weight
        return feature[...,0]+feature[...,1].tanh()


def test_both_inverse_directions_and_all_source_diseases_removed_without_mutation():
    g=graph();original=g.edge_index.clone()
    masked=u.prepare_mask(g,[10],1,disease_id=[20,21])
    assert torch.equal(g.edge_index,original)
    assert masked.edge_index.tolist()==[[2,11],[11,2]]
    assert masked.mask_metadata['direct_edges_removed']==4
    assert masked.mask_metadata['all_edges_removed']==8
    assert masked.relation_graph is g.relation_graph
    assert masked.mask_metadata['disease_ids']==[20,21]
    with pytest.raises(ValueError,match='forward'):
        u.prepare_mask(g,[10],3)


def test_stage2_filters_all_query_positives_and_enables_grad_inside_no_grad():
    m=Toy();g=u.prepare_mask(graph(),[10],1,disease_id=20)
    with torch.no_grad():
        result=u.stage2_backward(m,g,10,[0,1,2],list(range(8)),np.random.default_rng(3),negative_count=4)
    assert not set(result['negative_gene_ids'])&{0,1,2}
    assert len(set(result['negative_gene_ids']))==4
    assert m.weight.grad.abs().sum()>0
    assert m.training  # Restore caller mode.
    assert (m.calls[-1][...,2]==3).all()


def test_streaming_profile_matches_full_graph_backward_including_pool_scalar():
    m=Toy();reference=Toy();reference.load_state_dict(m.state_dict())
    g=u.prepare_mask(graph(),[10,11],1,disease_id=[20,21])
    pooling=torch.nn.Parameter(torch.tensor(.7));reference_pool=torch.nn.Parameter(torch.tensor(.7))
    # Duplicated phenotype deliberately checks multiplicity and max gradients.
    phenos=[11,10,11]
    result=u.profile_backward(m,g,phenos,[0,1],list(range(10)),pooling,np.random.default_rng(17),
                              random_count=2,hard_count=2,proposal_size=5,scale=.3)
    genes=result['selected_gene_ids']
    with u.deterministic_gradient_mode(reference):
        values=torch.stack([u._batch_scores(reference,g,p,genes,3) for p in phenos])
        loss=u.profile_loss(values,2,reference_pool)
        (loss*.3).backward()
    assert result['loss']==pytest.approx(float(loss.detach()))
    torch.testing.assert_close(m.weight.grad,reference.weight.grad,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(pooling.grad,reference_pool.grad)
    assert len(m.calls)==6 and all(len(c)==1 for c in m.calls)
    assert len(genes)==len(set(genes))==6
    assert genes[:2]==[0,1]


def test_mask_required_for_every_scored_phenotype():
    with pytest.raises(ValueError,match='prepare_mask'):
        u.stage2_backward(Toy(),graph(),10,[0],list(range(8)),np.random.default_rng(0),negative_count=2)
    g=u.prepare_mask(graph(),[10],1)
    with pytest.raises(ValueError,match='not masked'):
        u.profile_backward(Toy(),g,[11],[0],list(range(8)),torch.tensor(0.),np.random.default_rng(0),random_count=2,hard_count=0)


def test_reproducible_hard_sampling_and_random_only_support():
    g=u.prepare_mask(graph(),[10,11],1)
    first=u.profile_backward(Toy(),g,[10,11],[0,1],list(range(10)),torch.tensor(0.),np.random.default_rng(8),random_count=2,hard_count=2,proposal_size=5)
    second=u.profile_backward(Toy(),g,[10,11],[0,1],list(range(10)),torch.tensor(0.),np.random.default_rng(8),random_count=2,hard_count=2,proposal_size=5)
    assert first['selected_gene_ids']==second['selected_gene_ids']
    random=u.profile_backward(Toy(),g,[10],[0],list(range(10)),torch.tensor(0.),np.random.default_rng(8),random_count=4,hard_count=0)
    assert random['proposals_scored']==0 and random['hard_negatives']==0
