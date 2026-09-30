"""Negative sampling that produces the contrast the evaluation actually measures.

The default BasicNegativeSampler corrupts head or tail with an entity drawn uniformly from
every entity in the graph. For a (gene, causes_phenotype, symptom) triple that means the
replacement is another gene only about 1.9% of the time, and only half the positives have
their gene position corrupted at all, so under 1% of positives yield a gene-versus-gene
comparison. The loss therefore only has to separate genes from arbitrary ontology classes,
while the metric ranks 4,399 genes against each other.

GenePoolNegativeSampler corrupts the gene position of the designated relations using a
supplied pool of candidate genes, so every such negative is a gene-versus-gene contrast.
Other relations keep the default uniform head/tail behaviour.

With inverse triples the same relation appears twice: at an even internal id where the gene
is the head, and at the following odd id where the triple is flipped and the gene is the
tail. Both are handled, so the gene position is corrupted either way.
"""

import torch
from pykeen.sampling import BasicNegativeSampler

__all__ = ["GenePoolNegativeSampler"]


class GenePoolNegativeSampler(BasicNegativeSampler):
    """Corrupt the gene side of gene-centric relations from an explicit candidate pool.

    :param target_relations: internal ids of the FORWARD relations whose head is a gene.
        The matching inverse ids (id + 1), where the gene is the tail, are derived.
    :param gene_pool: entity ids to draw replacements from. Pass the evaluation candidate
        pool so training contrasts the same genes the metric ranks; deriving the pool from
        the training triples instead would omit genes that never appear as a head.
    """

    def __init__(self, *, target_relations, gene_pool, **kwargs):
        super().__init__(**kwargs)
        fwd = torch.as_tensor(sorted(set(int(r) for r in target_relations)), dtype=torch.long)
        self.forward_relations = fwd
        self.inverse_relations = fwd + 1
        self.gene_pool = torch.as_tensor(sorted(set(int(e) for e in gene_pool)), dtype=torch.long)
        if self.gene_pool.numel() < 2:
            raise ValueError("gene_pool needs at least two entities to form a contrast")

    def corrupt_batch(self, positive_batch):
        batch_shape = positive_batch.shape[:-1]
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        device = negative_batch.device
        relations = negative_batch[:, 1]
        pool = self.gene_pool.to(device)

        gene_is_head = torch.isin(relations, self.forward_relations.to(device))
        gene_is_tail = torch.isin(relations, self.inverse_relations.to(device))

        for mask, position in ((gene_is_head, 0), (gene_is_tail, 2)):
            count = int(mask.sum())
            if count:
                draw = torch.randint(pool.numel(), size=(count,), device=device)
                negative_batch[mask, position] = pool[draw]

        # everything else keeps the default behaviour: half head-corrupted, half tail-,
        # drawn uniformly from all entities
        rest = (~(gene_is_head | gene_is_tail)).nonzero(as_tuple=False).squeeze(-1)
        if rest.numel():
            half = rest.numel() // 2
            for chunk, position in ((rest[:half], 0), (rest[half:], 2)):
                if chunk.numel():
                    negative_batch[chunk, position] = torch.randint(
                        self.num_entities, size=(chunk.numel(),), device=device
                    )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
