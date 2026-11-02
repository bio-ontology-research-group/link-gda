"""Prepared-fold loading, one-hop query index and a checkpointable sampler."""
import csv
from pathlib import Path

import numpy as np
import torch

from .common import read_json, sha

REQUIRED_FILES = ('graph.npz', 'vocabulary.json', 'profiles.json', 'candidates.txt', 'training.tsv',
                  'validation.tsv', 'test.tsv')
FORWARD = ('associated_with', 'has_symptom')
INVERSE = ('inverse:has_symptom', 'inverse:associated_with')
READOUTS = ('forward', 'inverse')
SHORTCUT_RELATIONS = ('causes_phenotype', 'inverse:causes_phenotype')


def read_pairs(path):
    with open(path, newline='') as handle:
        return [(row['Gene'], row['Disease']) for row in csv.DictReader(handle, delimiter='\t')]


def recorded_digests(identity):
    """File digests recorded by the preparation step, as ``{name: sha256}``."""
    records = identity.get('outputs', identity.get('files'))
    if not isinstance(records, dict):
        raise ValueError('Prepared identity.json lacks output digests')
    return {name: (value['sha256'] if isinstance(value, dict) else value) for name, value in records.items()}


class PreparedData:
    """A verified prepared fold.

    Every file in ``REQUIRED_FILES`` must be listed in ``identity.json`` with a
    matching SHA-256. Validation and test diseases must be absent from the
    entity vocabulary, the materialised shortcut relation must be absent, and
    every candidate and profile phenotype of a held-out disease must be an entity.
    Test rows are read only for these disjointness checks.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        identity_path = self.directory / 'identity.json'
        self.identity = read_json(identity_path)
        self.identity_sha256 = sha(identity_path)
        digests = recorded_digests(self.identity)
        missing = [name for name in REQUIRED_FILES if name not in digests]
        if missing:
            raise ValueError(f'Prepared identity lacks digests for {missing}')
        for name, digest in digests.items():
            if sha(self.directory / name) != digest:
                raise ValueError(f'Prepared file changed: {name}')
        self.file_digests = {name: digests[name] for name in sorted(digests)}
        with np.load(self.directory / 'graph.npz', allow_pickle=False) as archive:
            self.triples = np.asarray(archive['triples'])
        if self.triples.dtype != np.int64 or self.triples.ndim != 2 or self.triples.shape[1] != 3:
            raise ValueError('graph.npz triples must be int64 with shape [N, 3]')
        vocabulary = read_json(self.directory / 'vocabulary.json')
        self.entities = vocabulary['entities']
        self.relations = vocabulary['relations']
        for table in (self.entities, self.relations):
            if sorted(table.values()) != list(range(len(table))):
                raise ValueError('Vocabulary identifiers must be 0..n-1')
        if len(self.triples) and (self.triples.min() < 0 or self.triples[:, [0, 2]].max() >= len(self.entities)
                                  or self.triples[:, 1].max() >= len(self.relations)):
            raise ValueError('Triple identifiers outside the vocabulary')
        for name in FORWARD + INVERSE:
            if name not in self.relations:
                raise ValueError(f'Relation {name} missing from vocabulary')
        if any(name in self.relations for name in SHORTCUT_RELATIONS):
            raise ValueError('Materialised causes_phenotype relation present in the training graph')
        self.profiles = read_json(self.directory / 'profiles.json')
        self.candidates = (self.directory / 'candidates.txt').read_text().splitlines()
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError('Candidate list has duplicates')
        self.rows = {split: read_pairs(self.directory / f'{split}.tsv') for split in ('training', 'validation', 'test')}
        held_out = {disease for split in ('validation', 'test') for _, disease in self.rows[split]}
        leaked = sorted(held_out & set(self.entities))
        if leaked:
            raise ValueError(f'Held-out diseases appear as entities: {leaked[:5]}')
        missing_candidates = [gene for gene in self.candidates if gene not in self.entities]
        if missing_candidates:
            raise ValueError(f'Candidates missing from vocabulary: {missing_candidates[:5]}')

    @property
    def num_entities(self):
        return len(self.entities)

    @property
    def num_relations(self):
        return len(self.relations)

    def relation_ids(self, readout):
        names = FORWARD if readout == 'forward' else INVERSE
        return [self.relations[name] for name in names]

    def validation_panel(self, count):
        """The first ``count`` validation diseases in sorted order (all when ``count`` is None) and their rows."""
        diseases = sorted({disease for _, disease in self.rows['validation']})
        if count is not None:
            diseases = diseases[:count]
        if len(diseases) < 2:
            raise ValueError('Calibrated selection needs at least two validation diseases')
        chosen = set(diseases)
        rows = [(gene, disease) for gene, disease in self.rows['validation'] if disease in chosen]
        positions = {gene: index for index, gene in enumerate(self.candidates)}
        absent = sorted({gene for gene, _ in rows if gene not in positions})
        if absent:
            raise ValueError(f'Validation genes missing from the frozen candidates: {absent[:5]}')
        for disease in diseases:
            if not self.profiles.get(disease):
                raise ValueError(f'Validation disease {disease} has no phenotype profile')
            unknown = [phenotype for phenotype in self.profiles[disease] if phenotype not in self.entities]
            if unknown:
                raise ValueError(f'Profile phenotypes missing from vocabulary: {unknown[:5]}')
        return diseases, rows


class QueryIndex:
    """All one-hop queries (head, relation) of the graph with their answer sets in CSR form.

    The subsampling weight is computed as KGReasoning ``TrainDataset`` does:
    ``torch.sqrt(1 / torch.Tensor([4 + answers]))`` in float32.
    """

    def __init__(self, triples, num_entities):
        order = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))
        ordered = triples[order]
        unique_triples = np.ones(len(ordered), dtype=bool)
        unique_triples[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        ordered = ordered[unique_triples]
        boundary = np.ones(len(ordered), dtype=bool)
        boundary[1:] = np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)
        self.starts = np.flatnonzero(boundary)
        self.counts = np.diff(np.append(self.starts, len(ordered)))
        self.heads = ordered[self.starts, 0]
        self.relations = ordered[self.starts, 1]
        self.answers = ordered[:, 2]
        self.num_entities = int(num_entities)
        self.weights = torch.sqrt(1 / torch.Tensor((4 + self.counts).tolist())).numpy()

    def __len__(self):
        return len(self.starts)

    def answer_set(self, query):
        start = self.starts[query]
        return self.answers[start:start + self.counts[query]]


class Sampler:
    """Step-indexed one-hop batch sampler whose full position is a small serialisable state.

    Each epoch visits every query once in an order drawn from ``(seed, epoch)``;
    the last batch of an epoch may be smaller, as with a non-dropping DataLoader.
    For every query the positive is one uniform answer and the negatives follow
    KGReasoning ``TrainDataset.__getitem__``: draw 2K uniform entities, discard
    known answers, repeat until K remain, keep the first K. All draws come from
    one saved ``numpy.random.Generator``.
    """

    def __init__(self, index, batch_size, negative_sample_size, seed):
        self.index = index
        self.batch_size = int(batch_size)
        self.negative_sample_size = int(negative_sample_size)
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(1,))))
        self._permutation = None
        self._permutation_epoch = None

    def permutation(self):
        if self._permutation_epoch != self.epoch:
            shuffle = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(0, self.epoch))))
            self._permutation = shuffle.permutation(len(self.index))
            self._permutation_epoch = self.epoch
        return self._permutation

    def state_dict(self):
        return {'seed': self.seed, 'epoch': self.epoch, 'offset': self.offset,
                'batch_size': self.batch_size, 'negative_sample_size': self.negative_sample_size,
                'generator': self.generator.bit_generator.state}

    def load_state_dict(self, state):
        if (state['seed'], state['batch_size'], state['negative_sample_size']) != (
                self.seed, self.batch_size, self.negative_sample_size):
            raise ValueError('Sampler configuration changed')
        self.epoch = int(state['epoch'])
        self.offset = int(state['offset'])
        self.generator.bit_generator.state = state['generator']

    def negatives(self, known):
        collected = []
        size = 0
        while size < self.negative_sample_size:
            draw = self.generator.integers(self.index.num_entities, size=self.negative_sample_size * 2)
            draw = draw[~np.isin(draw, known)]
            collected.append(draw)
            size += draw.size
        return np.concatenate(collected)[:self.negative_sample_size]

    def next_batch(self):
        """Return anchors ``[B]``, relations ``[B, 1]``, positives ``[B]``, negatives ``[B, K]`` and weights ``[B]``."""
        if self.offset >= len(self.index):
            self.epoch += 1
            self.offset = 0
        queries = self.permutation()[self.offset:self.offset + self.batch_size]
        self.offset += len(queries)
        positives = np.empty(len(queries), dtype=np.int64)
        negatives = np.empty((len(queries), self.negative_sample_size), dtype=np.int64)
        for row, query in enumerate(queries):
            known = self.index.answer_set(query)
            positives[row] = known[self.generator.integers(len(known))]
            negatives[row] = self.negatives(known)
        return (self.index.heads[queries].astype(np.int64), self.index.relations[queries].astype(np.int64)[:, None],
                positives, negatives, self.index.weights[queries])


def profile_phenotypes(profiles, diseases):
    return sorted({phenotype for disease in diseases for phenotype in profiles[disease]})
