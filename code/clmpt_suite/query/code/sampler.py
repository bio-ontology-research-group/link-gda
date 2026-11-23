"""Seeded ontology/clinical sampler with independent named streams for the CLMPT suite.

Adapted from code/clmpt_boundary_queries/query/code/sampler.py (policy 'sqrt_group_count'): each update
draws up to 96 ontology slots and exactly 32 clinical TRAIN profile slots. An ontology slot picks a
pattern with probability proportional to the square root of its eligible distinct group count, then a
group uniformly within it; positives come from the full answer set; 128 distinct typed negatives exclude
every known positive.

Differences from the frozen sampler: every stream is seeded from a configurable sampler seed through
numpy SeedSequence([SUITE_ENTROPY, sampler_seed, stream_code]); ontology and clinical draws use disjoint
streams (pattern, group, positive, negative versus profile, positive, negative), so the clinical rows of
an update depend only on the bank, the sampler seed and the step, and are identical across curricula and
encoders; with ontology_slots 0 (clinical-only) no ontology stream is advanced. The historical 901-906
streams are not replayed, so this is a new scientific sampling recipe.
"""
from collections import Counter
import hashlib
import json

import numpy as np


SUITE_ENTROPY = 0x4C474441
STREAMS = (('ontology_pattern', 1), ('ontology_group', 2), ('ontology_positive', 3), ('ontology_negative', 4),
           ('clinical_profile', 5), ('clinical_positive', 6), ('clinical_negative', 7))
ONTOLOGY_STREAMS = tuple(name for name, _ in STREAMS if name.startswith('ontology_'))
CLINICAL_STREAMS = tuple(name for name, _ in STREAMS if name.startswith('clinical_'))
ONTOLOGY_SLOTS, PROFILE_SLOTS, NEGATIVES, BATCH = 96, 32, 128, 128


def stream_seed(sampler_seed, code):
    return np.random.SeedSequence([SUITE_ENTROPY, int(sampler_seed), int(code)])


def gene_domain(bank):
    """Clinical TRAIN positive/negative gene universe and its identity.

    A bank may carry explicit 'training_gene_ids' (for example an excluded cohort whose TRAIN associations
    include genes outside its evaluation ranking pool); otherwise the universe is 'candidate_ids', which is
    the historical behaviour. Validation and TEST always rank 'candidate_ids' only. Identity is the sha256 of
    the compact JSON of the sorted IDs.
    """
    source = 'training_gene_ids' if 'training_gene_ids' in bank else 'candidate_ids'
    ids = [int(value) for value in bank[source]]
    assert ids and len(set(ids)) == len(ids), 'clinical training gene universe must be non-empty and duplicate-free'
    ordered = sorted(ids)
    candidates = set(map(int, bank['candidate_ids']))
    return {'source': source, 'count': len(ids), 'sha256': hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).hexdigest(),
            'outside_candidate_pool': len(set(ids) - candidates), 'candidates_outside_universe': len(candidates - set(ids)), 'ids': ids}


def domain_identity(domain):
    return {key: domain[key] for key in ('source', 'count', 'sha256', 'outside_candidate_pool', 'candidates_outside_universe')}


class Sampler:
    def __init__(self, bank, mixture, sampler_seed):
        """mixture None means clinical-only: no ontology slots are drawn."""
        assert isinstance(sampler_seed, int) and sampler_seed >= 0
        self.sampler_seed = sampler_seed
        self.mixture = mixture
        self.graph = bank['ontology_queries']
        self.profiles = bank['training_profiles']
        self.additional = bank.get('additional_ontology_queries', [])
        self.queries = self.graph + self.profiles + self.additional
        self.profile_offset = len(self.graph)
        self.additional_offset = len(self.graph) + len(self.profiles)
        self.gene_domain = gene_domain(bank)
        self.pools = {'class': np.asarray(bank['class_ids'], dtype=np.int64), 'gene': np.asarray(self.gene_domain['ids'], dtype=np.int64)}
        self.pool_sets = {name: set(map(int, values)) for name, values in self.pools.items()}
        self.known = [set(row['negative_known']) for row in self.queries]
        for i, row in enumerate(self.queries):
            assert set(row['answers']) <= self.known[i]
            assert set(row['answers']) <= self.pool_sets[row['domain']]
            assert (len(self.pool_sets[row['domain']]) - len(self.known[i] & self.pool_sets[row['domain']]) >= NEGATIVES) == row.get('sampleable', True)
        ontology = list(range(self.profile_offset)) + list(range(self.additional_offset, len(self.queries)))
        assert all(self.queries[i]['domain'] == 'class' for i in ontology)
        assert all(row['domain'] == 'gene' and row['type'] == 'profile' for row in self.profiles)
        assert all(row.get('sampleable', True) and row['answers'] for row in self.profiles)
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
        if mixture is None:
            self.ontology_slots = 0
            self.types, self.groups, self.probabilities = [], {}, None
        else:
            assert mixture['policy'] == 'sqrt_group_count' and mixture['pattern_weight'] == 'sqrt_eligible_distinct_groups'
            assert mixture['ontology_slots'] == ONTOLOGY_SLOTS and mixture['profile_slots'] == PROFILE_SLOTS and mixture['negatives'] == NEGATIVES
            requested = list(mixture['patterns'])
            assert len(set(requested)) == len(requested) and requested
            assert all(kind in eligible for kind in requested), {'requested_without_eligible_groups': [kind for kind in requested if kind not in eligible]}
            if mixture.get('require_every_eligible_bank_pattern'):
                assert set(requested) == set(eligible), {'missing': sorted(set(eligible) - set(requested))}
                declared = bank.get('ontology_patterns')
                assert declared is not None and {entry['type'] for entry in declared} == set(total)
            self.ontology_slots = ONTOLOGY_SLOTS
            self.types = sorted(requested)
            self.groups = {kind: np.array(eligible[kind], dtype=np.int64) for kind in self.types}
            weights = np.sqrt(np.array([len(self.groups[kind]) for kind in self.types], dtype=np.float64))
            self.probabilities = weights / weights.sum()
        self.rng = {name: np.random.default_rng(stream_seed(sampler_seed, code)) for name, code in STREAMS}
        self.next_step = 0

    def describe(self):
        patterns = []
        for position, kind in enumerate(self.types):
            groups = int(len(self.groups[kind]))
            probability = float(self.probabilities[position])
            patterns.append({'type': kind, 'eligible_groups': groups, 'total_groups': int(self.total_groups.get(kind, 0)), 'probability': probability,
                             'expected_slots_per_update': ONTOLOGY_SLOTS * probability, 'expected_draws_per_group_per_update': ONTOLOGY_SLOTS * probability / groups})
        return {'policy': 'sqrt_group_count' if self.mixture else 'clinical_only', 'mixture': self.mixture, 'sampler_seed': self.sampler_seed,
                'clinical_training_gene_domain': domain_identity(self.gene_domain),
                'stream_derivation': 'numpy SeedSequence([%d, sampler_seed, code]) -> PCG64' % SUITE_ENTROPY, 'streams': dict(STREAMS),
                'patterns': patterns, 'unselected_bank_patterns': sorted(set(self.total_groups) - set(self.types)),
                'patterns_without_eligible_groups': self.excluded_patterns, 'ineligible_groups': self.ineligible,
                'index_layout': {'ontology_queries': [0, self.profile_offset], 'training_profiles': [self.profile_offset, self.additional_offset],
                                 'additional_ontology_queries': [self.additional_offset, len(self.queries)]},
                'slots': {'ontology': self.ontology_slots, 'profile': PROFILE_SLOTS, 'negatives': NEGATIVES, 'loss_denominator': BATCH},
                'receipt_fields': ['query_indices', 'positive_ids', 'negative_ids', 'ontology_pattern_counts', 'draw_step', 'clinical_draw_sha256', 'draw_sha256'],
                'scope': 'square-root pattern weighting balances pattern exposure; it is not claimed to prevent memorization of small patterns'}

    def state(self):
        return {'streams': {name: rng.bit_generator.state for name, rng in self.rng.items()}, 'next_step': self.next_step, 'sampler_seed': self.sampler_seed}

    def restore(self, state):
        assert set(state) == {'streams', 'next_step', 'sampler_seed'} and state['sampler_seed'] == self.sampler_seed
        assert set(state['streams']) == set(self.rng)
        for name, value in state['streams'].items():
            self.rng[name].bit_generator.state = value
        self.next_step = int(state['next_step'])

    def negatives(self, index, domain, stream):
        """Draw distinct eligible IDs without a per-query full-pool allocation."""
        accepted, used = [], set()
        pool = self.pools[domain]
        known = self.known[index]
        rng = self.rng[stream]
        while len(accepted) < NEGATIVES:
            proposals = rng.choice(pool, size=min(256, len(pool)), replace=False)
            for raw in proposals:
                value = int(raw)
                if value not in known and value not in used:
                    accepted.append(value)
                    used.add(value)
                    if len(accepted) == NEGATIVES:
                        break
        return accepted

    def draw(self, step):
        """Return (queries, positive_ids, negative_ids, receipt); ontology rows first, then 32 clinical rows."""
        assert step == self.next_step, {'step': step, 'sampler_next_step': self.next_step}
        ontology = []
        if self.ontology_slots:
            choices = self.rng['ontology_pattern'].choice(len(self.types), size=self.ontology_slots, p=self.probabilities)
            ontology = [int(self.rng['ontology_group'].choice(self.groups[self.types[int(choice)]])) for choice in choices]
        clinical = list(map(int, self.profile_offset + self.rng['clinical_profile'].integers(0, len(self.profiles), size=PROFILE_SLOTS)))
        ontology_positive = [int(self.rng['ontology_positive'].choice(self.queries[i]['answers'])) for i in ontology]
        clinical_positive = [int(self.rng['clinical_positive'].choice(self.queries[i]['answers'])) for i in clinical]
        ontology_negative = [self.negatives(i, 'class', 'ontology_negative') for i in ontology]
        clinical_negative = [self.negatives(i, 'gene', 'clinical_negative') for i in clinical]
        indices = ontology + clinical
        queries = [self.queries[i] for i in indices]
        positive = ontology_positive + clinical_positive
        negative = ontology_negative + clinical_negative
        clinical_receipt = {'query_indices': clinical, 'positive_ids': clinical_positive, 'negative_ids': clinical_negative, 'draw_step': step}
        receipt = {'query_indices': indices, 'positive_ids': positive, 'negative_ids': negative,
                   'ontology_pattern_counts': dict(sorted(Counter(self.queries[i]['type'] for i in ontology).items())), 'draw_step': step,
                   'clinical_draw_sha256': hashlib.sha256(json.dumps(clinical_receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        receipt['draw_sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.next_step = step + 1
        return queries, positive, negative, receipt

    def exposure(self, updates_path):
        """Per-pattern slot counts and per-group exposure recomputed from a trainer updates.jsonl."""
        slots, seen, profile_slots, updates = Counter(), Counter(), 0, 0
        with open(updates_path) as stream:
            for line in stream:
                row = json.loads(line)
                indices = row['query_indices']
                assert len(indices) == self.ontology_slots + PROFILE_SLOTS
                kinds = [self.queries[i]['type'] for i in indices[:self.ontology_slots]]
                assert dict(Counter(kinds)) == row['ontology_pattern_counts']
                assert all(self.profile_offset <= i < self.additional_offset for i in indices[self.ontology_slots:])
                slots.update(kinds)
                seen.update(indices[:self.ontology_slots])
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
