"""CLMPT query model for the suite: shared, independent-branch and plain-mean encoders.

All three are built by the pinned original constructor (original_model.QueryModel, official CLMPT layer,
ComplEx initializer) with the packed lookup execution variant installed, so the official parameters are
created in the same order from the same seeded generator. Encoder differences:

shared       the frozen model: one joint profile query (k phenotype atoms into e1, e1 into f), Transformer
             aggregation, cosine score.
independent  ontology rows are encoded exactly as shared. A clinical profile with k phenotypes is encoded as
             k single-phenotype two-hop branches (s0 into e1, e1 into f) by the same shared encoder. The
             profile score of gene g is the mean over branches of cosine(branch_i, g), computed as
             dot(v, unit g) with v the mean of the unit branch vectors; v is not renormalized. Training uses
             one softmax objective per profile over those averaged scores.
mean         the official layer with its Transformer replaced by identity, so each variable's update is the
             mean of its incoming messages and its current state, over the same query graphs and number of
             rounds, for ontology and clinical rows alike. The Transformer is still constructed before being
             removed, so the free/existential embeddings equal those of shared at the same seed.
"""
import torch
from torch import nn
import torch.nn.functional as F

from original_model import QueryModel as OriginalQueryModel
from packed_encode import install


ENCODERS = ('shared', 'independent', 'mean')
SCORE_SEMANTICS = {
    'shared': 'representation = encoder output for the joint profile query; score = cosine(representation, gene embedding)',
    'mean': 'representation = plain-mean encoder output for the joint profile query; score = cosine(representation, gene embedding)',
    'independent': 'representation = mean over phenotype branches of unit-normalized branch vectors, NOT renormalized; '
                   'score = dot(representation, unit gene embedding) = mean branch cosine. Cosine replay of this representation is wrong.',
}


def branch_queries(query):
    """Single-phenotype two-hop branches of one joint clinical profile query, in atom order."""
    assert query['type'] == 'profile'
    atoms = query['atoms']
    last = atoms[-1]
    assert last[0] == 'e1' and last[2] == 'f' and len(atoms) - 1 == len(query['hp_ids'])
    branches = []
    for head, role, tail in atoms[:-1]:
        assert tail == 'e1'
        hp = query['constants'][head]
        branches.append({'type': 'profile', 'hp_ids': [hp], 'constants': {'s0': hp}, 'atoms': [['s0', role, 'e1'], ['e1', last[1], 'f']]})
    assert sorted(branch['hp_ids'][0] for branch in branches) == sorted(query['hp_ids'])
    return branches


class SuiteQueryModel(OriginalQueryModel):
    def __init__(self, pretrained, allowed_hp_ids, encoder, device='cuda', temperature=.05):
        assert encoder in ENCODERS
        super().__init__(pretrained, allowed_hp_ids, device=device, temperature=temperature)
        install(self)
        self.encoder = encoder
        if encoder == 'mean':
            self.layer.transformer = nn.Identity()

    def branch_layout(self, queries):
        """Flattened branches plus the profile-by-branch averaging matrix (rows sum to one)."""
        branches, owners = [], []
        for row, query in enumerate(queries):
            parts = branch_queries(query)
            branches.extend(parts)
            owners.extend([row] * len(parts))
        weights = torch.zeros(len(queries), len(branches), dtype=torch.float32)
        for column, row in enumerate(owners):
            weights[row, column] = 1. / (len(queries[row]['atoms']) - 1)
        return branches, weights.to(self.device)

    def profile_representation(self, queries):
        """Representation whose score semantics are SCORE_SEMANTICS[self.encoder]."""
        if self.encoder != 'independent':
            return self.encode(queries)
        branches, weights = self.branch_layout(queries)
        return weights @ F.normalize(self.encode(branches), dim=-1)

    def scores_from_representation(self, representation, candidates):
        genes = F.normalize(self.nbp._entity_embedding(candidates), dim=-1)
        if self.encoder == 'independent':
            return representation @ genes.T
        return F.normalize(representation, dim=-1) @ genes.T

    def profile_scores(self, queries, candidates):
        return self.scores_from_representation(self.profile_representation(queries), candidates)

    def suite_loss(self, queries, positive, negative, ontology_rows, ontology_loss):
        """Contrastive loss over one update; clinical-only divides the clinical sum by the full batch of 128."""
        assert len(queries) == len(positive) == len(negative)
        if not ontology_loss:
            assert ontology_rows == 0
        if self.encoder != 'independent':
            representation = self.encode(queries)
            target = self.nbp._entity_embedding(positive)
            negatives = self.nbp._entity_embedding(negative)
            positive_score = F.cosine_similarity(representation, target, dim=-1) / self.temperature
            negative_score = F.cosine_similarity(representation[:, None, :], negatives, dim=-1) / self.temperature
            logits = torch.cat([positive_score[:, None], negative_score], dim=1)
        else:
            parts = []
            if ontology_rows:
                representation = self.encode(queries[:ontology_rows])
                target = self.nbp._entity_embedding(positive[:ontology_rows])
                negatives = self.nbp._entity_embedding(negative[:ontology_rows])
                positive_score = F.cosine_similarity(representation, target, dim=-1) / self.temperature
                negative_score = F.cosine_similarity(representation[:, None, :], negatives, dim=-1) / self.temperature
                parts.append(torch.cat([positive_score[:, None], negative_score], dim=1))
            averaged = self.profile_representation(queries[ontology_rows:])
            target = F.normalize(self.nbp._entity_embedding(positive[ontology_rows:]), dim=-1)
            negatives = F.normalize(self.nbp._entity_embedding(negative[ontology_rows:]), dim=-1)
            positive_score = (averaged * target).sum(-1) / self.temperature
            negative_score = torch.einsum('pd,pkd->pk', averaged, negatives) / self.temperature
            parts.append(torch.cat([positive_score[:, None], negative_score], dim=1))
            logits = torch.cat(parts, dim=0)
        labels = torch.zeros(len(queries), device=logits.device, dtype=torch.long)
        if ontology_loss:
            return F.cross_entropy(logits, labels)
        return F.cross_entropy(logits, labels, reduction='sum') / 128.
