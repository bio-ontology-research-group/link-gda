"""Replayable ontology/profile sampler over the axiom-only bank.

Policy 'sqrt_group_count' (the agreed curriculum): each update has 96 ontology slots and 32 clinical TRAIN
profile slots. Each ontology slot picks a pattern with probability proportional to the square root of that
pattern's eligible distinct query-group count, then a group uniformly within the pattern. The full curriculum
lists every pattern with at least one eligible group; the three-pattern subset curriculum lists 1p/pip/ppip
under the same policy (it is not the old equal round robin). A group is eligible iff it has an asserted
answer and its known-positive mask leaves at least 128 typed negatives (the bank's 'sampleable' flag).

Policy 'legacy_core_round_robin_replay' reproduces the frozen sampler (f18bb8d2) draw for draw on the
three-pattern core, for replay qualification only; it is not the agreed curriculum.

Positives are resampled from the full answer set; 128 distinct typed negatives exclude every known positive.
Query index space: ontology_queries, then training_profiles, then additional_ontology_queries, so profile
indices are unchanged from the frozen bank.
"""
from collections import Counter
import hashlib
import json

import numpy as np


POLICIES = ('sqrt_group_count', 'legacy_core_round_robin_replay')
LEGACY_TYPES = ('1p', 'pip', 'ppip')
SEEDS = (('graph', 901), ('positive', 902), ('negative', 903), ('profile_order', 904), ('profile_query', 905), ('pattern', 906))
ONTOLOGY_SLOTS, PROFILE_SLOTS, NEGATIVES = 96, 32, 128


class Sampler:
    def __init__(self, bank, mixture):
        assert mixture['policy'] in POLICIES, mixture
        assert mixture['ontology_slots'] == ONTOLOGY_SLOTS and mixture['profile_slots'] == PROFILE_SLOTS and mixture['negatives'] == NEGATIVES
        self.mixture = mixture
        self.policy = mixture['policy']
        self.graph = bank['ontology_queries']
        self.profiles = bank['training_profiles']
        self.additional = bank.get('additional_ontology_queries', [])
        self.queries = self.graph + self.profiles + self.additional
        self.profile_offset = len(self.graph)
        self.additional_offset = len(self.graph) + len(self.profiles)
        self.pools = {'class': np.asarray(bank['class_ids'], dtype=np.int64), 'gene': np.asarray(bank['candidate_ids'], dtype=np.int64)}
        self.pool_sets = {name: set(map(int, values)) for name, values in self.pools.items()}
        self.known = [set(row['negative_known']) for row in self.queries]
        for i, row in enumerate(self.queries):
            assert set(row['answers']) <= self.known[i]
            assert set(row['answers']) <= self.pool_sets[row['domain']]
            assert (len(self.pool_sets[row['domain']]) - len(self.known[i] & self.pool_sets[row['domain']]) >= NEGATIVES) == row.get('sampleable', True)
        ontology = list(range(self.profile_offset)) + list(range(self.additional_offset, len(self.queries)))
        assert all(self.queries[i]['domain'] == 'class' for i in ontology)
        assert all(row['domain'] == 'gene' and row['type'] == 'profile' for row in self.profiles)
        eligible, ineligible, total = {}, Counter(), Counter()
        for i in ontology:
            row = self.queries[i]
            total[row['type']] += 1
            if row['sampleable'] and row['answers']:
                eligible.setdefault(row['type'], []).append(i)
            else:
                ineligible[(row['type'], 'no_asserted_answer' if not row['answers'] else 'fewer_than_128_typed_negatives')] += 1
        self.total_groups = dict(total)
        self.ineligible = [{'type': kind, 'reason': reason, 'groups': count} for (kind, reason), count in sorted(ineligible.items())]
        self.excluded_patterns = sorted(kind for kind in total if kind not in eligible)
        requested = list(mixture['patterns'])
        assert len(set(requested)) == len(requested) and requested
        assert all(kind in eligible for kind in requested), {'requested_without_eligible_groups': [kind for kind in requested if kind not in eligible]}
        if mixture.get('require_every_eligible_bank_pattern'):
            assert set(requested) == set(eligible), {'missing': sorted(set(eligible) - set(requested))}
            declared = bank.get('ontology_patterns')
            assert declared is not None and {entry['type'] for entry in declared} == set(total)
        if self.policy == 'legacy_core_round_robin_replay':
            assert tuple(requested) == LEGACY_TYPES
            self.types = list(LEGACY_TYPES)
            self.groups = {kind: np.array([i for i, row in enumerate(self.graph) if row['type'] == kind and row['sampleable']], dtype=np.int64) for kind in self.types}
            assert all(len(values) for values in self.groups.values())
            self.probabilities = None
            streams = SEEDS[:5]
        else:
            assert mixture['pattern_weight'] == 'sqrt_eligible_distinct_groups'
            self.types = sorted(requested)
            self.groups = {kind: np.array(eligible[kind], dtype=np.int64) for kind in self.types}
            weights = np.sqrt(np.array([len(self.groups[kind]) for kind in self.types], dtype=np.float64))
            self.probabilities = weights / weights.sum()
            streams = SEEDS
        self.rng = {name: np.random.default_rng(seed) for name, seed in streams}
        self.next_step = 0

    def describe(self):
        patterns = []
        for position, kind in enumerate(self.types):
            groups = int(len(self.groups[kind]))
            row = {'type': kind, 'eligible_groups': groups, 'total_groups': int(self.total_groups.get(kind, 0))}
            if self.probabilities is not None:
                probability = float(self.probabilities[position])
                row.update({'probability': probability, 'expected_slots_per_update': ONTOLOGY_SLOTS * probability,
                            'expected_draws_per_group_per_update': ONTOLOGY_SLOTS * probability / groups})
            else:
                row.update({'slots_per_update': ONTOLOGY_SLOTS // len(self.types)})
            patterns.append(row)
        return {'policy': self.policy, 'mixture': self.mixture, 'seeds': {name: seed for name, seed in SEEDS if name in self.rng},
                'patterns': patterns, 'unselected_bank_patterns': sorted(set(self.total_groups) - set(self.types)),
                'patterns_without_eligible_groups': self.excluded_patterns, 'ineligible_groups': self.ineligible,
                'index_layout': {'ontology_queries': [0, self.profile_offset], 'training_profiles': [self.profile_offset, self.additional_offset],
                                 'additional_ontology_queries': [self.additional_offset, len(self.queries)]},
                'slots': {'ontology': ONTOLOGY_SLOTS, 'profile': PROFILE_SLOTS, 'negatives': NEGATIVES},
                'receipt_fields': ['query_indices', 'positive_ids', 'negative_ids'] + (['ontology_pattern_counts', 'draw_step'] if self.probabilities is not None else []) + ['draw_sha256'],
                'scope': 'square-root pattern weighting balances pattern exposure; it is not claimed to prevent memorization of small patterns'}

    def state(self):
        streams = {name: rng.bit_generator.state for name, rng in self.rng.items()}
        if self.policy == 'legacy_core_round_robin_replay':
            return streams
        return {'streams': streams, 'next_step': self.next_step}

    def restore(self, state):
        if self.policy == 'legacy_core_round_robin_replay':
            streams = state
        else:
            assert set(state) == {'streams', 'next_step'}
            streams = state['streams']
            self.next_step = int(state['next_step'])
        assert set(streams) == set(self.rng)
        for name, value in streams.items():
            self.rng[name].bit_generator.state = value

    def negatives(self, index, domain):
        """Draw distinct eligible IDs without a per-query full-pool allocation."""
        accepted, used = [], set()
        pool = self.pools[domain]
        known = self.known[index]
        while len(accepted) < NEGATIVES:
            proposals = self.rng['negative'].choice(pool, size=min(256, len(pool)), replace=False)
            for raw in proposals:
                value = int(raw)
                if value not in known and value not in used:
                    accepted.append(value)
                    used.add(value)
                    if len(accepted) == NEGATIVES:
                        break
        return accepted

    def draw(self, step):
        if self.policy == 'legacy_core_round_robin_replay':
            indices = [int(self.rng['graph'].choice(self.groups[self.types[(step * ONTOLOGY_SLOTS + i) % 3]])) for i in range(ONTOLOGY_SLOTS)]
        else:
            assert step == self.next_step, {'step': step, 'sampler_next_step': self.next_step}
            choices = self.rng['pattern'].choice(len(self.types), size=ONTOLOGY_SLOTS, p=self.probabilities)
            indices = [int(self.rng['graph'].choice(self.groups[self.types[int(choice)]])) for choice in choices]
        indices += list(map(int, self.profile_offset + self.rng['profile_query'].integers(0, len(self.profiles), size=PROFILE_SLOTS)))
        queries = [self.queries[i] for i in indices]
        positive = [int(self.rng['positive'].choice(row['answers'])) for row in queries]
        negative = [self.negatives(i, row['domain']) for i, row in zip(indices, queries)]
        receipt = {'query_indices': indices, 'positive_ids': positive, 'negative_ids': negative}
        if self.policy != 'legacy_core_round_robin_replay':
            receipt['ontology_pattern_counts'] = dict(sorted(Counter(row['type'] for row in queries[:ONTOLOGY_SLOTS]).items()))
            receipt['draw_step'] = step
            self.next_step = step + 1
        receipt['draw_sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return queries, positive, negative, receipt

    def exposure(self, updates_path):
        """Per-pattern slot counts and per-group exposure recomputed from a trainer updates.jsonl."""
        slots, seen, profile_slots, updates = Counter(), Counter(), 0, 0
        with open(updates_path) as stream:
            for line in stream:
                row = json.loads(line)
                indices = row['query_indices']
                assert len(indices) == ONTOLOGY_SLOTS + PROFILE_SLOTS
                kinds = [self.queries[i]['type'] for i in indices[:ONTOLOGY_SLOTS]]
                assert kinds == row['query_types'][:ONTOLOGY_SLOTS]
                if 'ontology_pattern_counts' in row:
                    assert dict(Counter(kinds)) == row['ontology_pattern_counts']
                assert all(self.profile_offset <= i < self.additional_offset for i in indices[ONTOLOGY_SLOTS:])
                slots.update(kinds)
                seen.update(indices[:ONTOLOGY_SLOTS])
                profile_slots += PROFILE_SLOTS
                updates += 1
        per_type = {}
        for index, count in seen.items():
            entry = per_type.setdefault(self.queries[index]['type'], {'distinct_groups_seen': 0, 'maximum_group_exposure': 0, 'group_exposures': 0})
            entry['distinct_groups_seen'] += 1
            entry['maximum_group_exposure'] = max(entry['maximum_group_exposure'], count)
            entry['group_exposures'] += count
        return {'updates': updates, 'pattern_slots': dict(slots), 'profile_slots': profile_slots, 'per_pattern_groups': per_type,
                'eligible_groups': {kind: int(len(values)) for kind, values in self.groups.items()}}
