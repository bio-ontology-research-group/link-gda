"""Candidate-typed corruption with training-positive rejection.

PyKEEN's SLCWA instances supply inverse-expanded INTERNAL mapped triples.
Only designated forward relations and their following inverse IDs are changed;
other relations retain BasicNegativeSampler's corruption behavior.
"""
import torch
from pykeen.sampling import BasicNegativeSampler


class GenePoolNegativeSampler(BasicNegativeSampler):
    def __init__(self, *, target_relations, gene_pool, mapped_triples, **kwargs):
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        fwd = torch.as_tensor(sorted(set(map(int, target_relations))), dtype=torch.long)
        pool = torch.as_tensor(sorted(set(map(int, gene_pool))), dtype=torch.long)
        if not len(fwd) or len(pool) < 2:
            raise ValueError('Need target relations and at least two candidate genes')
        if torch.isin(fwd, fwd + 1).any():
            raise ValueError('Forward and inverse target IDs overlap')
        if pool.min() < 0 or pool.max() >= self.num_entities:
            raise ValueError('Candidate gene outside entity range')
        self.register_buffer('forward_relations', fwd)
        self.register_buffer('gene_pool', pool)
        triples = mapped_triples.detach().cpu()
        is_forward = torch.isin(triples[:, 1], fwd)
        is_inverse = torch.isin(triples[:, 1], fwd + 1)
        selected = triples[is_forward | is_inverse]
        inverse = torch.isin(selected[:, 1], fwd + 1)
        relations = selected[:, 1] - inverse.long()
        symptoms = torch.where(inverse, selected[:, 0], selected[:, 2])
        genes = torch.where(inverse, selected[:, 2], selected[:, 0])
        context = relations * self.num_entities + symptoms
        keys = torch.unique(context * self.num_entities + genes, sorted=True)
        self.register_buffer('positive_keys', keys)
        # Reject impossible contexts before any rejection-sampling loop.
        candidate_keys = torch.unique((context * self.num_entities + genes)[torch.isin(genes, pool)])
        contexts, counts = torch.unique(candidate_keys // self.num_entities, return_counts=True)
        if (counts >= len(pool)).any():
            raise ValueError('Target relation/symptom has no eligible negative candidate')

    def corrupt_batch(self, positive_batch):
        shape = positive_batch.shape[:-1]
        positives = positive_batch.reshape(-1, 3)
        forward = torch.isin(positives[:, 1], self.forward_relations.to(positives.device))
        inverse = torch.isin(positives[:, 1], (self.forward_relations + 1).to(positives.device))
        target = forward | inverse
        result = positives[:, None, :].expand(-1, self.num_negs_per_pos, -1).clone()
        if (~target).any():
            result[~target] = super().corrupt_batch(positives[~target])
        if target.any():
            chosen = positives[target]
            inv = inverse[target]
            context = (chosen[:, 1] - inv.long()) * self.num_entities + torch.where(inv, chosen[:, 0], chosen[:, 2])
            original = torch.where(inv, chosen[:, 2], chosen[:, 0])
            context = context[:, None].expand(-1, self.num_negs_per_pos).reshape(-1)
            original = original[:, None].expand(-1, self.num_negs_per_pos).reshape(-1)
            pool = self.gene_pool.to(positives.device)
            keys = self.positive_keys.to(positives.device)
            draw = pool[torch.randint(len(pool), (len(context),), device=positives.device)]
            # Vectorized over every pending negative; no per-triple Python loop.
            for _ in range(1024):
                query = context * self.num_entities + draw
                if len(keys):
                    index = torch.searchsorted(keys, query)
                    known = (index < len(keys)) & (keys[index.clamp(max=len(keys)-1)] == query)
                else:
                    known = torch.zeros_like(draw, dtype=torch.bool)
                reject = (draw == original) | known
                if not reject.any():
                    break
                draw[reject] = pool[torch.randint(len(pool), (int(reject.sum()),), device=positives.device)]
            else:
                raise RuntimeError('Candidate rejection exhausted; inspect nearly saturated phenotype contexts')
            typed = result[target]
            positions = torch.where(inv, 2, 0)
            typed.scatter_(2, positions[:, None, None].expand(-1, self.num_negs_per_pos, 1), draw.reshape(-1, self.num_negs_per_pos, 1))
            result[target] = typed
        return result.reshape(*shape, self.num_negs_per_pos, 3)
