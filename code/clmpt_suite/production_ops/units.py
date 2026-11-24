"""The production units: 109 ComplEx pretraining units (Orix) and 185 query units (IBEX), from production.py plan().

The enumeration is the frozen root production.py; this module only normalizes condition names to the worker's canonical
encoder/curriculum strings, attaches the input entry each unit reads, and checks the counts. No unit is added or removed.
"""
import importlib.util

from common import SUITE, check

ENCODERS = ('shared', 'independent', 'mean')
CURRICULA = ('full13', 'core3', 'aux_off')
LEGACY = {'transformer_shared': 'shared', 'transformer_independent': 'independent', 'plain_mean': 'mean', 'clinical_only': 'aux_off'}


def _production():
    spec = importlib.util.spec_from_file_location('suite_production_plan', str(SUITE / 'production.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_key(cohort, fold, modality):
    return 'fold%d-%s' % (fold, modality) if cohort == 'main' else 'excluded-%s' % modality


def plan():
    """Normalized unit plan; raises unless it is exactly 109 pretraining and 185 query units with valid parents."""
    raw = _production().plan()
    pretrain = {}
    for unit in raw['pretrain']:
        pretrain[unit['id']] = {'id': unit['id'], 'kind': 'pretrain', 'site': 'orix', 'stage': unit['stage'], 'cohort': unit['cohort'], 'modality': unit['modality'],
                                'fold': unit['fold'] if unit['cohort'] == 'main' else 'excluded', 'seed': unit['seed'],
                                'entry': entry_key(unit['cohort'], unit['fold'], unit['modality'])}
    query = {}
    for unit in raw['query']:
        encoder, curriculum = LEGACY.get(unit['encoder'], unit['encoder']), LEGACY.get(unit['curriculum'], unit['curriculum'])
        check(encoder in ENCODERS and curriculum in CURRICULA, 'unknown condition in plan: %s' % unit['id'])
        parent = pretrain[unit['depends_on']]
        check((parent['cohort'], parent['modality'], parent['fold']) == (unit['cohort'], unit['modality'], parent['fold']), 'parent mismatch: %s' % unit['id'])
        record = {'id': unit['id'], 'kind': 'query', 'site': 'ibex', 'stage': unit['stage'], 'parent': parent['id'], 'cohort': unit['cohort'], 'modality': unit['modality'],
                  'fold': parent['fold'], 'seed': unit['seed'], 'sampler_seed': unit['seed'], 'condition': unit['condition'], 'encoder': encoder, 'curriculum': curriculum,
                  'entry': parent['entry']}
        for key in ('ranking_pools', 'target_subset', 'hpo_reuse_candidate'):
            if key in unit:
                record[key] = unit[key]
        query[unit['id']] = record
    check(len(pretrain) == 109 and len(query) == 185, 'plan counts differ: %d pretrain, %d query' % (len(pretrain), len(query)))
    check(sum(unit['encoder'] == 'mean' for unit in query.values()) == 19, 'plain-mean count differs')
    pfs = {(unit['fold'], unit['seed']) for unit in query.values() if unit['stage'] == 3}
    check(len(pfs) == 19 and sum(unit['stage'] == 3 for unit in query.values()) == 95, 'stage 3 layout differs')
    check(all(sum(unit['stage'] == stage for unit in query.values()) == 30 for stage in (4, 5, 6)), 'stage 4-6 counts differ')
    check(set(unit['parent'] for unit in query.values()) == set(pretrain), 'a pretraining unit has no query child')
    return {'pretrain': pretrain, 'query': query}


def summaries(units):
    """Separate summary groups: tenfold seed-0 and fold-0 ten-seed replication are never pooled."""
    groups = {'tenfold_seed0': [], 'fold0_ten_seed': [], 'excluded_ten_seed': []}
    for unit in units['query'].values():
        if unit['cohort'] == 'excluded':
            groups['excluded_ten_seed'].append(unit['id'])
        else:
            if unit['seed'] == 0:
                groups['tenfold_seed0'].append(unit['id'])
            if unit['fold'] == 0 and unit['stage'] == 3:
                groups['fold0_ten_seed'].append(unit['id'])
    return {key: sorted(value) for key, value in groups.items()}
