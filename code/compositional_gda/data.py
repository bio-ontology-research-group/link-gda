"""Prepared-fold loading, query indices and checkpointable samplers.

The one-hop ``QueryIndex`` and ``Sampler`` define the default Track A training
stream. ``PathQueryIndex``, ``RelationQueryView`` and ``MixedSampler`` serve the
separate two-hop training diagnostic and its matched one-hop control.
"""
import csv
import hashlib
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


def subsampling_weights(counts):
    return torch.sqrt(1 / torch.Tensor((4 + np.asarray(counts)).tolist())).numpy()


def expand(starts, counts):
    """Positions ``starts[i] + j`` for ``j < counts[i]``, concatenated over ``i``."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    return np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(int(counts.sum()), dtype=np.int64)


def query_pairs(index):
    """Every (head, answer) pair of an index as two int64 arrays, in query order."""
    return (np.repeat(np.asarray(index.heads, dtype=np.int64), index.counts),
            np.asarray(index.answers, dtype=np.int64)[expand(index.starts, index.counts)])


def relation_pairs(index, relation):
    """The (head, tail) pairs of one relation of a one-hop ``QueryIndex``, sorted by head then tail."""
    return query_pairs(RelationQueryView(index, relation))


def query_digest(index):
    """SHA-256 over an index's heads, relation paths, answer counts, answers and weights."""
    digest = hashlib.sha256()
    _, answers = query_pairs(index)
    relations = np.asarray(index.relations, dtype=np.int64).reshape(len(index), -1)
    for array in (np.asarray(index.heads, dtype='<i8'), relations.astype('<i8'),
                  np.asarray(index.counts, dtype='<i8'), answers.astype('<i8'),
                  np.asarray(index.weights, dtype='<f4')):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class RelationQueryView:
    """The one-hop queries of a single relation, sharing the parent ``QueryIndex`` answers.

    Answer sets are the full known tails of each (head, relation) query, so
    negatives are filtered exactly as in the parent index.
    """

    def __init__(self, index, relation):
        selected = np.flatnonzero(index.relations == relation)
        if not selected.size:
            raise ValueError(f'Relation {relation} has no one-hop queries')
        self.relation = int(relation)
        self.starts = index.starts[selected]
        self.counts = index.counts[selected]
        self.heads = index.heads[selected]
        self.relations = index.relations[selected]
        self.answers = index.answers
        self.weights = index.weights[selected]
        self.num_entities = index.num_entities

    def __len__(self):
        return len(self.starts)

    def answer_set(self, query):
        start = self.starts[query]
        return self.answers[start:start + self.counts[query]]


class PathQueryIndex:
    """Two-hop path queries ``(anchor, [r1, r2])`` composed from a one-hop ``QueryIndex``.

    The answer set of an anchor is the union of the ``r2`` tails over every
    ``r1`` tail of the anchor, the existential reading of the path. Only the
    graph triples held by the one-hop index are read. Attributes mirror
    ``QueryIndex``, with ``relations`` of shape ``[Q, 2]`` and weights computed
    with the same float32 expression. ``intermediates`` are the entities that
    join at least one complete path, and ``paths`` counts (anchor,
    intermediate, answer) paths before the union.
    """

    def __init__(self, index, path):
        path = tuple(int(relation) for relation in path)
        if len(path) != 2:
            raise ValueError('PathQueryIndex composes exactly two relations')
        anchors, middles = relation_pairs(index, path[0])
        second_heads, second_tails = relation_pairs(index, path[1])
        left = np.searchsorted(second_heads, middles, side='left')
        right = np.searchsorted(second_heads, middles, side='right')
        lengths = right - left
        positions = expand(left, lengths)
        path_anchors = np.repeat(anchors, lengths)
        path_middles = np.repeat(middles, lengths)
        path_answers = second_tails[positions]
        if not path_answers.size:
            raise ValueError(f'Relation path {path} has no complete paths')
        self.path = path
        self.paths = int(path_answers.size)
        self.intermediates = np.unique(path_middles)
        order = np.lexsort((path_answers, path_anchors))
        path_anchors, path_answers = path_anchors[order], path_answers[order]
        unique = np.ones(len(path_anchors), dtype=bool)
        unique[1:] = (path_anchors[1:] != path_anchors[:-1]) | (path_answers[1:] != path_answers[:-1])
        path_anchors, path_answers = path_anchors[unique], path_answers[unique]
        boundary = np.ones(len(path_anchors), dtype=bool)
        boundary[1:] = path_anchors[1:] != path_anchors[:-1]
        self.starts = np.flatnonzero(boundary)
        self.counts = np.diff(np.append(self.starts, len(path_anchors)))
        self.heads = path_anchors[self.starts]
        self.relations = np.tile(np.asarray(path, dtype=np.int64), (len(self.starts), 1))
        self.answers = path_answers
        self.num_entities = index.num_entities
        self.weights = subsampling_weights(self.counts)

    def __len__(self):
        return len(self.starts)

    def answer_set(self, query):
        start = self.starts[query]
        return self.answers[start:start + self.counts[query]]


def verify_two_hop(data, index, forward, inverse):
    """Fail closed unless the two-hop indices derive from training diseases of the verified graph only.

    ``index`` is the one-hop ``QueryIndex`` of ``data.triples``. Raises
    ``ValueError`` unless: the ``associated_with`` pairs of the graph equal the
    ``training.tsv`` rows; the ``has_symptom`` pairs equal the ``profiles.json``
    phenotypes of training diseases; every intermediate of either direction is a
    training disease and no validation or test disease; the forward union pairs
    equal the independent join of ``training.tsv`` with those profiles; and the
    inverse union pairs equal the transposed forward pairs. Returns a record of
    the checks and counts for the run identity.
    """
    entities = data.entities
    names = {value: name for name, value in entities.items()}
    training_rows = data.rows['training']
    training = {disease for _, disease in training_rows}
    held_out = {disease for split in ('validation', 'test') for _, disease in data.rows[split]}
    absent = sorted({name for gene, disease in training_rows for name in (gene, disease) if name not in entities})
    if absent:
        raise ValueError(f'Training rows name entities missing from the vocabulary: {absent[:5]}')
    unprofiled = sorted(disease for disease in training if not data.profiles.get(disease))
    if unprofiled:
        raise ValueError(f'Training diseases without a phenotype profile: {unprofiled[:5]}')
    unknown = sorted({phenotype for disease in training for phenotype in data.profiles[disease]
                      if phenotype not in entities})
    if unknown:
        raise ValueError(f'Training profile phenotypes missing from the vocabulary: {unknown[:5]}')
    associated = {(entities[gene], entities[disease]) for gene, disease in training_rows}
    symptoms = {(entities[disease], entities[phenotype]) for disease in training for phenotype in data.profiles[disease]}
    for name, expected in ((FORWARD[0], associated), (FORWARD[1], symptoms)):
        heads, tails = relation_pairs(index, data.relations[name])
        if set(zip(heads.tolist(), tails.tolist())) != expected:
            raise ValueError(f'Graph {name} edges differ from the training rows and profiles')
    record = {'training_diseases': len(training), 'held_out_diseases': len(held_out)}
    for label, paths in (('forward', forward), ('inverse', inverse)):
        middles = {names[int(entity)] for entity in paths.intermediates}
        outside = sorted(middles - training)
        if outside:
            raise ValueError(f'{label} two-hop intermediates outside the training diseases: {outside[:5]}')
        leaked = sorted(middles & held_out)
        if leaked:
            raise ValueError(f'{label} two-hop intermediates include held-out diseases: {leaked[:5]}')
        record[f'{label}_intermediates'] = len(middles)
        record[f'{label}_intermediates_equal_training_diseases'] = middles == training
    joined = {(entities[gene], entities[phenotype]) for gene, disease in training_rows
              for phenotype in data.profiles[disease]}
    heads, answers = query_pairs(forward)
    forward_pairs = set(zip(heads.tolist(), answers.tolist()))
    if forward_pairs != joined:
        raise ValueError('Forward two-hop pairs differ from the join of training.tsv with training profiles')
    heads, answers = query_pairs(inverse)
    if set(zip(answers.tolist(), heads.tolist())) != forward_pairs:
        raise ValueError('Inverse two-hop pairs are not the transposed forward pairs')
    record.update({'graph_associated_with_equals_training_rows': True,
                   'graph_has_symptom_equals_training_profiles': True,
                   'intermediates_within_training_diseases': True,
                   'intermediates_disjoint_from_held_out': True,
                   'forward_pairs_equal_training_join': True,
                   'inverse_pairs_equal_transposed_forward': True,
                   'union_pairs': len(forward_pairs)})
    return record


class Sampler:
    """Step-indexed one-hop batch sampler whose full position is a small serialisable state.

    Each epoch visits every query once in an order drawn from ``(seed, epoch)``;
    the last batch of an epoch may be smaller, as with a non-dropping DataLoader.
    For every query the positive is one uniform answer and the negatives follow
    KGReasoning ``TrainDataset.__getitem__``: draw 2K uniform entities, discard
    known answers, repeat until K remain, keep the first K. All draws come from
    one saved ``numpy.random.Generator``.

    The default ``stream=None`` uses spawn keys ``(1,)`` for the draws and
    ``(0, epoch)`` for the order. A tuple ``stream`` prefixes both keys with
    ``(2, *stream)``, giving an independent stream from the same seed. With
    ``wrap=True`` every batch has exactly ``batch_size`` queries: a batch that
    reaches the end of an epoch continues deterministically into the next
    epoch's order, so a query can appear twice in a batch at an epoch boundary
    or when the index is smaller than the batch.
    """

    def __init__(self, index, batch_size, negative_sample_size, seed, stream=None, wrap=False):
        self.index = index
        self.batch_size = int(batch_size)
        self.negative_sample_size = int(negative_sample_size)
        self.seed = int(seed)
        self.stream = None if stream is None else tuple(int(key) for key in stream)
        self.wrap = bool(wrap)
        if self.wrap and len(index) == 0:
            raise ValueError('A wrapping sampler needs at least one query')
        self.prefix = () if self.stream is None else (2, *self.stream)
        self.epoch = 0
        self.offset = 0
        self.generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            self.seed, spawn_key=self.prefix + (1,))))
        self._permutation = None
        self._permutation_epoch = None

    def permutation(self):
        if self._permutation_epoch != self.epoch:
            shuffle = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                self.seed, spawn_key=self.prefix + (0, self.epoch))))
            self._permutation = shuffle.permutation(len(self.index))
            self._permutation_epoch = self.epoch
        return self._permutation

    def state_dict(self):
        return {'seed': self.seed, 'epoch': self.epoch, 'offset': self.offset,
                'batch_size': self.batch_size, 'negative_sample_size': self.negative_sample_size,
                'stream': None if self.stream is None else list(self.stream), 'wrap': self.wrap,
                'queries': len(self.index), 'generator': self.generator.bit_generator.state}

    def load_state_dict(self, state):
        stream = state.get('stream')
        stream = None if stream is None else tuple(int(key) for key in stream)
        if (state['seed'], state['batch_size'], state['negative_sample_size'], stream, bool(state.get('wrap', False))) != (
                self.seed, self.batch_size, self.negative_sample_size, self.stream, self.wrap):
            raise ValueError('Sampler configuration changed')
        if state.get('queries', len(self.index)) != len(self.index):
            raise ValueError('Sampler query index changed')
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

    def next_queries(self):
        if self.offset >= len(self.index):
            self.epoch += 1
            self.offset = 0
        queries = self.permutation()[self.offset:self.offset + self.batch_size]
        self.offset += len(queries)
        if not self.wrap:
            return queries
        pieces = [queries]
        missing = self.batch_size - len(queries)
        while missing > 0:
            if self.offset >= len(self.index):
                self.epoch += 1
                self.offset = 0
            piece = self.permutation()[self.offset:self.offset + missing]
            self.offset += len(piece)
            missing -= len(piece)
            pieces.append(piece)
        return np.concatenate(pieces)

    def next_batch(self):
        """Return anchors ``[B]``, relations ``[B, H]``, positives ``[B]``, negatives ``[B, K]`` and weights ``[B]``.

        ``H`` is 1 for a one-hop index and the path length for a ``PathQueryIndex``.
        """
        queries = self.next_queries()
        positives = np.empty(len(queries), dtype=np.int64)
        negatives = np.empty((len(queries), self.negative_sample_size), dtype=np.int64)
        for row, query in enumerate(queries):
            known = self.index.answer_set(query)
            positives[row] = known[self.generator.integers(len(known))]
            negatives[row] = self.negatives(known)
        return (self.index.heads[queries].astype(np.int64),
                self.index.relations[queries].astype(np.int64).reshape(len(queries), -1),
                positives, negatives, self.index.weights[queries])


class MixedSampler:
    """Fixed-composition batches from several query groups, each with its own wrapping ``Sampler``.

    ``groups`` is an ordered sequence of ``(name, index, slots, stream)``. Every
    call of ``next_batch`` returns, in that order, one ``(name, anchors,
    relations, positives, negatives, weights)`` tuple per group with exactly
    ``slots`` queries, so each update has ``batch_size = sum(slots)`` queries.
    Groups never share random state, so a group's draws do not depend on the
    other groups. Negatives of every group are filtered against the full answer
    set of their query.
    """

    FORMAT = 'mixed-sampler/1'

    def __init__(self, groups, negative_sample_size, seed):
        groups = list(groups)
        if not groups:
            raise ValueError('MixedSampler needs at least one group')
        names = [name for name, _, _, _ in groups]
        if len(set(names)) != len(names):
            raise ValueError('MixedSampler group names must be unique')
        streams = [None if stream is None else tuple(stream) for _, _, _, stream in groups]
        if len(set(streams)) != len(streams):
            raise ValueError('MixedSampler groups must use distinct streams')
        for name, _, slots, _ in groups:
            if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
                raise ValueError(f'Group {name} needs a positive integer number of slots')
        self.groups = [(name, Sampler(index, slots, negative_sample_size, seed, stream=stream, wrap=True))
                       for name, index, slots, stream in groups]
        self.batch_size = sum(sampler.batch_size for _, sampler in self.groups)

    @property
    def epoch(self):
        return self.groups[0][1].epoch

    @property
    def offset(self):
        return self.groups[0][1].offset

    def positions(self):
        """Epoch and offset of every group's sampler."""
        return {name: {'epoch': sampler.epoch, 'offset': sampler.offset} for name, sampler in self.groups}

    def state_dict(self):
        return {'format': self.FORMAT,
                'groups': [{'name': name, 'sampler': sampler.state_dict()} for name, sampler in self.groups]}

    def load_state_dict(self, state):
        if state.get('format') != self.FORMAT:
            raise ValueError('Sampler state is not a mixed-sampler state')
        if [group['name'] for group in state['groups']] != [name for name, _ in self.groups]:
            raise ValueError('Mixed sampler groups changed')
        for group, (_, sampler) in zip(state['groups'], self.groups):
            sampler.load_state_dict(group['sampler'])

    def next_batch(self):
        return [(name, *sampler.next_batch()) for name, sampler in self.groups]


def profile_phenotypes(profiles, diseases):
    return sorted({phenotype for disease in diseases for phenotype in profiles[disease]})
