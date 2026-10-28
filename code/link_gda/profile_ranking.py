"""End-to-end disease-profile ranking, using training labels only.

Unlike frozen in-sample stacking, this objective directly updates the encoder on
its training examples. Disease identifiers only group phenotype profiles and
positive labels; they are never supplied to the model as learned entities.
Validation/test diseases must remain absent from the training graph and labels.
The coordinator owns graph exclusions, checkpoint selection and outer evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pool_scores(scores: torch.Tensor, weight=0.5):
    """Combine maximum and mean phenotype evidence; preserve gradients."""
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Expected nonempty gene-by-phenotype scores")
    return weight * scores.max(dim=1).values + (1 - weight) * scores.mean(dim=1)


def multi_positive_loss(scores: torch.Tensor, num_positives: int, temperature=1.0):
    """Cross entropy with uniform mass on all known positives in this profile.

    Positives occupy the first positions. Every positive is included once; none
    can occur in the negative segment. This is a sampled-list objective, not an
    unbiased full-candidate softmax estimate (hard proposals have no correction).
    """
    if scores.ndim != 1 or not 0 < num_positives < scores.numel():
        raise ValueError("Need at least one positive and one negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = scores / temperature
    return torch.logsumexp(logits, dim=0) - logits[:num_positives].mean()


class ProfileRanker(nn.Module):
    """Profile-ranking fine tuning around a differentiable PyKEEN encoder.

    ``training_pairs`` and optional ``validation_pairs`` have Gene/Disease
    columns. Validation labels are used only for a disease-disjointness check.
    ``disease2pheno`` maps IDs to phenotype IDs. No outer-test argument exists.
    The optimizer should use ``ranker.parameters()`` so the optional pooling
    scalar is updated alongside the underlying model. Save ``state_dict()``
    for both the encoder and scalar, plus ``rng_state()`` for exact resumption.
    """

    def __init__(self, model, triples_factory, training_pairs, disease2pheno,
                 eval_genes, *, validation_pairs=None, seed=0,
                 learn_pooling=False, temperature=1.0, random_negatives=32,
                 hard_negatives=32, proposal_size=256, chunk_size=8192, evaluation_chunk_size=100000):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if random_negatives < 0 or hard_negatives < 0 or random_negatives + hard_negatives < 1:
            raise ValueError("At least one negative required")
        if proposal_size < hard_negatives or chunk_size < 1 or evaluation_chunk_size < 1:
            raise ValueError("Proposal size must cover hard negatives; chunk size must be positive")
        self.model = model
        self.learn_pooling = bool(learn_pooling)
        if self.learn_pooling:
            self.pooling_logit = nn.Parameter(torch.tensor(0.0, device=self.device))
        else:
            self.register_buffer("pooling_logit", torch.tensor(0.0, device=self.device))
        self.temperature = float(temperature)
        self.random_negatives = int(random_negatives)
        self.hard_negatives = int(hard_negatives)
        self.proposal_size = int(proposal_size)
        self.chunk_size = int(chunk_size)
        self.evaluation_chunk_size = int(evaluation_chunk_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.epochs = 0
        self.genes = list(eval_genes)
        if len(self.genes) < 2 or len(set(self.genes)) != len(self.genes):
            raise ValueError("Candidates must be unique and nonempty")
        self.entity_to_id = triples_factory.entity_to_id
        missing = set(self.genes) - set(self.entity_to_id)
        if missing:
            raise ValueError(f"Candidates absent from encoder: {len(missing)}")
        self.gene_to_column = {g: i for i, g in enumerate(self.genes)}
        self.gene_ids = np.asarray([self.entity_to_id[g] for g in self.genes], dtype=np.int64)
        external = triples_factory.relation_to_id['causes_phenotype']
        self.inverse_triples = bool(triples_factory.create_inverse_triples)
        self.relation_id = int(external) * (2 if self.inverse_triples else 1)
        pairs = sorted(set(zip(training_pairs.Gene, training_pairs.Disease)))
        if not pairs:
            raise ValueError("Empty training associations")
        self.training_pairs = pairs
        self.diseases = sorted({d for _, d in pairs})
        if validation_pairs is not None and set(self.diseases) & set(validation_pairs.Disease):
            raise ValueError("Training and validation diseases overlap")
        self.positives = {d: [] for d in self.diseases}
        for gene, disease in pairs:
            if gene not in self.gene_to_column:
                raise ValueError(f"Training positive is outside candidate pool: {gene}")
            self.positives[disease].append(self.gene_to_column[gene])
        for d in self.diseases:
            self.positives[d] = np.asarray(self.positives[d], dtype=np.int64)
            if len(self.genes) - len(self.positives[d]) < random_negatives + hard_negatives:
                raise ValueError(f"Not enough distinct nonpositive candidates for disease {d}")
        self.disease2pheno = {d: tuple(ps) for d, ps in disease2pheno.items()}
        for disease in self.diseases:
            self._phenotypes(disease)  # Fail before any optimizer mutation.
        self._all_columns = np.arange(len(self.genes), dtype=np.int64)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def pooling_weight(self):
        return float(self.pooling_logit.detach().sigmoid().cpu())

    def rng_state(self):
        return {"numpy": self.rng.bit_generator.state, "epochs": self.epochs}

    def load_rng_state(self, state):
        self.rng.bit_generator.state = state['numpy']
        self.epochs = int(state['epochs'])

    def _phenotypes(self, disease):
        # Preserve the original evaluator's phenotype order and multiplicity:
        # sorting changes floating-point sums; deduplication changes weighting.
        ids = [self.entity_to_id[p] for p in self.disease2pheno.get(disease, ())
               if p in self.entity_to_id]
        if not ids:
            raise ValueError(f"Disease {disease} has no represented phenotypes")
        return ids

    def _score_columns(self, disease, columns, chunk_size=None):
        """Differentiable internal forward-relation score; chunk triple calls."""
        chunk_size = self.chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size < 1:
            raise ValueError("Chunk size must be positive")
        pheno = torch.as_tensor(self._phenotypes(disease), device=self.device, dtype=torch.long)
        genes = torch.as_tensor(self.gene_ids[np.asarray(columns)], device=self.device, dtype=torch.long)
        # Gene chunks avoid retaining a full candidate x phenotype triple tensor.
        gene_chunk = max(1, chunk_size // len(pheno))
        results = []
        for batch in genes.split(gene_chunk):
            h = batch.repeat_interleave(len(pheno))
            t = pheno.repeat(len(batch))
            triples = torch.stack((h, torch.full_like(h, self.relation_id), t), dim=1)
            raw = torch.cat([self.model.score_hrt(part).reshape(-1)
                             for part in triples.split(chunk_size)])
            results.append(pool_scores(raw.reshape(len(batch), len(pheno)), self.pooling_logit.sigmoid()))
        return torch.cat(results)

    def sample_negatives(self, disease):
        """Unique random negatives plus highest-scoring eligible proposal genes.

        All known training positives are excluded, including multi-gene diseases.
        Random negatives are selected first; hard proposals exclude them. No
        annotations or labels from validation/test participate in filtering.
        """
        mask = np.ones(len(self.genes), dtype=bool)
        mask[self.positives[disease]] = False
        eligible = self._all_columns[mask]
        random = self.rng.choice(eligible, self.random_negatives, replace=False)
        if not self.hard_negatives:
            return random.astype(np.int64), {"proposal_count": 0}
        remaining = eligible[~np.isin(eligible, random)]
        proposal = self.rng.choice(remaining, min(self.proposal_size, len(remaining)), replace=False)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                values = self._score_columns(disease, proposal).detach().cpu().numpy()
        finally:
            self.model.train(was_training)
        # Stable gene-column tie break makes identical scores reproducible.
        chosen = np.lexsort((proposal, -values))[:self.hard_negatives]
        negatives = np.concatenate((random, proposal[chosen])).astype(np.int64)
        assert not set(negatives) & set(self.positives[disease])
        assert len(np.unique(negatives)) == len(negatives)
        return negatives, {"proposal_count": int(len(proposal))}

    def train_epoch(self, optimizer, accumulation_steps=16):
        """One equally disease-weighted epoch, with exact final partial-batch scale."""
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        self.train()
        order = self.rng.permutation(len(self.diseases))
        total_loss = 0.0
        proposals = 0
        updates = 0
        for start in range(0, len(order), accumulation_steps):
            group = order[start:start + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            for di in group:
                disease = self.diseases[int(di)]
                negatives, info = self.sample_negatives(disease)
                columns = np.concatenate((self.positives[disease], negatives))
                scores = self._score_columns(disease, columns)
                loss = multi_positive_loss(scores, len(self.positives[disease]), self.temperature)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Nonfinite disease-profile loss")
                (loss / len(group)).backward()
                total_loss += float(loss.detach().cpu())
                proposals += info['proposal_count']
            optimizer.step()
            # PyKEEN constrained embeddings need the standard post-update hook.
            if hasattr(self.model, 'post_parameter_update'):
                self.model.post_parameter_update()
            updates += 1
        self.epochs += 1
        return {"epoch": self.epochs, "diseases": len(order), "optimizer_steps": updates,
                "mean_loss": total_loss / len(order), "pooling_weight": self.pooling_weight(),
                "hard_proposals_scored": proposals, "negatives_per_disease": self.random_negatives + self.hard_negatives}

    def score_profiles(self, diseases: Iterable[str], candidates=None):
        """Full-candidate held-out profile scoring, no labels or calibration."""
        candidates = self.genes if candidates is None else list(candidates)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Scoring candidates must be nonempty and unique")
        columns = np.asarray([self.gene_to_column[g] for g in candidates], dtype=np.int64)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return {d: self._score_columns(d, columns, chunk_size=self.evaluation_chunk_size).detach().cpu().numpy()
                        for d in dict.fromkeys(diseases)}
        finally:
            self.train(was_training)

    def metadata(self):
        training_profiles = {d: list(self.disease2pheno[d]) for d in self.diseases}
        return {"objective": "uniform-positive sampled-list cross entropy",
                "end_to_end": True, "disease_id_embedding_input": False,
                "pooling": "learned-sigmoid-max-mean" if self.learn_pooling else "fixed-half-max-half-mean",
                "pooling_weight": self.pooling_weight(), "temperature": self.temperature,
                "seed": self.seed, "random_negatives": self.random_negatives,
                "hard_negatives": self.hard_negatives, "proposal_size": self.proposal_size,
                "negative_filter": "all same-disease training positives; unique negatives; random/hard disjoint",
                "hard_selection": "top current raw profile scores among uniform unique proposals; column-order ties",
                "proposal_importance_correction": False,
                "chunk_size": self.chunk_size, "evaluation_chunk_size": self.evaluation_chunk_size,
                "phenotype_order_and_multiplicity": "preserved from input",
                "relation_internal_id": self.relation_id,
                "create_inverse_triples": self.inverse_triples,
                "training_pair_count": len(self.training_pairs), "training_diseases": len(self.diseases),
                "candidate_count": len(self.genes), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_pairs_sha256": _digest(self.training_pairs),
                "training_profiles_sha256": _digest(training_profiles), "candidate_sha256": _digest(self.genes)}
