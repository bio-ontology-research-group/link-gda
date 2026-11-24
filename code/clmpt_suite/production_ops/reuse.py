"""FROZEN_HPO_ARTIFACT: narrow reuse of the genuine HPO artifacts for the PFS fold-0 seed-0 units only.

Pretraining unit main-PFS-f0-ps0: the exact ComplEx initializer used by the selected (calibrated common-recipe) HPO
winner is admitted as a trained artifact, not retrained, when every scientific identity matches: the winner config's
pretrained_sha256 equals the HPO manifest's admitted initializer record and its original delivery ADMISSION; the delivery
identity/config match the ADMISSION hashes; graph COMPLETE (hence vocabulary and triples) equals the production fold-0
PFS graph; entity and relation counts, rank, init size, batch, optimizer, dtype, learning rate, N3 weight, update budget
and seed 0 equal the production recipe; the common leaderboard is calibrated. The legacy trainer source and protocol
hashes are KEPT as provenance; nothing is relabelled as v2 training and no optimizer state is continued.

Query unit main-PFS-f0-qs0-full13-shared: the HPO winner's completed 20,000-update trajectory is reused only if, in
addition, its scientific config (recipe, seeds, encoder, curriculum, mixture, numerics, budget, cadence, bank, initializer),
its core worker sources and its reference hashes equal what production would run; otherwise only the initializer is reused
and the query unit trains fresh. No other unit, fold, seed, modality or legacy model is admissible.
"""
import hashlib
import json
from pathlib import Path

PRETRAIN_UNIT = 'main-PFS-f0-ps0'
QUERY_UNIT = 'main-PFS-f0-qs0-full13-shared'
QUERY_SOURCES = ('train.py', 'model.py', 'original_model.py', 'packed_encode.py', 'sampler.py', 'evaluate.py', 'metrics.py', 'state_compare.py')
QUERY_SCIENTIFIC = ('learning_rate', 'temperature', 'seed', 'sampler_seed', 'encoder', 'curriculum', 'ontology_loss', 'mixture_id', 'mixture', 'evaluation_every_updates',
                    'maximum_steps', 'rank', 'batch_size', 'profile_batch_size', 'negative_sample_size', 'optimizer', 'weight_decay', 'transformer_layers', 'heads', 'dropout',
                    'dtype', 'strict_deterministic_algorithms', 'checkpoint_retention', 'log_negative_ids', 'data_complete_sha256', 'pretrained_sha256')
ALLOWED_QUERY_DIFFERENCES = ('experiment_id', 'purpose', 'protocol_path', 'protocol_sha256', 'data_dir', 'pretrained_checkpoint', 'maximum_seconds',
                             'continuation_parent', 'parent_complete_sha256', 'parent_checkpoint_sha256')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha_bytes((json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + '\n').encode())


def audit_pretrain(evidence, bundle_inputs, production_recipe, pretrain_fixed, graph_entry='fold0-PFS'):
    """evidence: {'recipe': recipe.json, 'hpo_manifest': frozen-manifest.json, 'admission': ADMISSION.json, 'admission_sha256',
    'identity': delivery identity.json, 'identity_sha256', 'config': delivery config.json, 'config_sha256', 'export_name'}."""
    checks = {}
    winner = evidence['recipe']['common_winner']
    checks['common_leaderboard_calibrated'] = evidence['recipe']['common_leaderboard'] == 'calibrated' == production_recipe['common_leaderboard']
    checks['winner_recipe_is_production_recipe'] = winner['recipe'] == {key: production_recipe[key] for key in ('initializer_recipe', 'learning_rate', 'temperature')}
    exported = winner['identities']['pretrained_sha256']
    records = [record for record in evidence['hpo_manifest']['initializers'].values() if record['sha256'] == exported]
    checks['one_hpo_initializer_record'] = len(records) == 1
    record = records[0] if records else {}
    recipe = production_recipe['initializer_recipe']
    checks['record_recipe'] = all(record.get(key) == recipe[key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget'))
    checks['record_scope'] = (record.get('taxonomy_mode'), record.get('cohort', 'main'), record.get('modality'), record.get('fold'), record.get('pretrain_seed')) == ('original', 'main', 'PFS', 0, 0)
    admission = record.get('delivery_admission') or {}
    checks['delivery_admission_bound'] = admission.get('sha256') == evidence['admission_sha256']
    files = evidence['admission'].get('files', {})
    checks['admission_lists_export'] = files.get(evidence['export_name']) == exported
    checks['admission_lists_identity_and_config'] = files.get('identity.json') == evidence['identity_sha256'] and files.get('config.json') == evidence['config_sha256']
    config = evidence['identity'].get('config', {})
    checks['identity_config_is_delivered_config'] = evidence['identity'].get('config_sha256') == evidence['config_sha256']
    graph = bundle_inputs[graph_entry]
    checks['graph_identical'] = evidence['identity'].get('data_complete_sha256') == graph['graph']['complete_sha256'] == config.get('data_complete_sha256')
    checks['dimensions'] = config.get('entity_count') == graph['pretraining']['entity_count'] and config.get('relation_count') == graph['pretraining']['relation_count']
    checks['fixed_numerics'] = all(config.get(key) == value for key, value in pretrain_fixed.items())
    checks['learning_rate_and_n3'] = config.get('learning_rate') == recipe['pretrain_learning_rate'] and config.get('regularization') == recipe['pretrain_regularization']
    checks['seed_zero'] = config.get('seed') == 0
    checks['budget_milestone'] = recipe['initializer_update_budget'] in (config.get('milestone_updates') or []) and evidence['export_name'] == 'update-%07d.pt' % recipe['initializer_update_budget']
    checks['original_taxonomy'] = config.get('taxonomy_mode', 'original') == 'original'
    passed = all(checks.values())
    return {'kind': 'FROZEN_HPO_ARTIFACT', 'unit': PRETRAIN_UNIT, 'passed': passed, 'checks': checks,
            'export': {'sha256': exported, 'bytes': record.get('bytes'), 'ibex_path': record.get('path')},
            'provenance': {'legacy_sources': evidence['identity'].get('sources'), 'legacy_protocol_sha256': evidence['identity'].get('protocol_sha256'),
                           'legacy_config_sha256': evidence['config_sha256'], 'delivery_admission_sha256': evidence['admission_sha256'],
                           'delivery_admission_path': admission.get('path'), 'hpo_manifest_initializer_record': record,
                           'note': 'trained artifact reuse; legacy trainer and protocol hashes preserved; no optimizer continuation'}}


def audit_query(trajectory_identity, trajectory_complete, pretrain_audit, expected, production_sources, production_references):
    """expected: the production config values for QUERY_UNIT that do not depend on paths."""
    checks = {'initializer_reused': pretrain_audit['passed']}
    config = trajectory_identity.get('config', {})
    checks['complete_bound_to_identity'] = trajectory_complete.get('identity_sha256') == canonical(trajectory_identity)
    checks['complete_steps'] = trajectory_complete.get('steps') == expected['maximum_steps']
    for key in QUERY_SCIENTIFIC:
        checks['config_%s' % key] = config.get(key) == expected[key]
    checks['initializer_fields'] = config.get('initializer') == expected['initializer']
    checks['core_sources'] = {name: config.get('source_hashes', {}).get(name) for name in QUERY_SOURCES} == {name: production_sources[name] for name in QUERY_SOURCES}
    checks['references'] = config.get('reference_hashes') == production_references
    differing = sorted(key for key in set(config) | set(expected['template_keys']) if key not in QUERY_SCIENTIFIC + ('initializer', 'source_hashes', 'reference_hashes')
                       and key not in ALLOWED_QUERY_DIFFERENCES and config.get(key) != expected['template'].get(key))
    checks['no_other_differences'] = not differing
    return {'kind': 'FROZEN_HPO_ARTIFACT', 'unit': QUERY_UNIT, 'passed': all(checks.values()), 'checks': checks, 'other_differences': differing}
