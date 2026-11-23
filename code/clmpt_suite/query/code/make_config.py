"""Canonical query-stage config generator and validator for the CLMPT suite.

Python API (import from this directory):
    build_config(...) -> dict            canonical config; raises on any contradiction with the protocol
    validate_config(path, ...) -> dict   recomputes every pinned hash and reruns train.check_config/check_bank
    write_config(config, path)           canonical JSON, refuses to overwrite
CLI:
    python make_config.py build    --protocol P --purpose ... --output CONFIG.json
    python make_config.py validate --config CONFIG.json [--skip-pretrained-hash] --output RECEIPT.json

Canonical keys are owned here; callers may add only keys not in CANONICAL_KEYS (for example
campaign_trial_id, schema, initializer_name), and those enter the run identity unchanged.
source_hashes covers exactly train.SOURCE_FILES; reference_hashes exactly train.REFERENCE_FILES.
"""
import argparse
import json
from pathlib import Path

from train import INITIALIZER_KEYS, REFERENCE_FILES, RETENTION, SOURCE_FILES, check_bank, check_config, sha

HERE = Path(__file__).resolve().parent


def fold_value(text):
    """Main folds are integers; the excluded cohort's single split is the string 'excluded'."""
    return int(text) if text.isdigit() else text
CANONICAL_KEYS = ('experiment_id', 'phase', 'purpose', 'protocol_path', 'protocol_sha256', 'source_hashes', 'reference_hashes', 'data_dir',
                  'data_complete_sha256', 'pretrained_checkpoint', 'pretrained_sha256', 'initializer', 'seed', 'sampler_seed', 'encoder', 'curriculum',
                  'ontology_loss', 'mixture_id', 'mixture', 'learning_rate', 'temperature', 'maximum_steps', 'evaluation_every_updates', 'maximum_seconds',
                  'dtype', 'strict_deterministic_algorithms', 'rank', 'batch_size', 'profile_batch_size', 'negative_sample_size', 'optimizer', 'weight_decay',
                  'transformer_layers', 'heads', 'dropout', 'checkpoint_retention', 'log_negative_ids', 'test_evaluation', 'continuation_parent',
                  'parent_complete_sha256', 'parent_checkpoint_sha256')


def fixed_fields(protocol):
    """Numerical and retention settings that are identical in every suite query config."""
    return {'dtype': 'float32', 'strict_deterministic_algorithms': True, 'rank': 400, 'batch_size': 128, 'profile_batch_size': 32, 'negative_sample_size': 128,
            'optimizer': 'AdamW', 'weight_decay': protocol['query_weight_decay'], 'transformer_layers': 2, 'heads': 8, 'dropout': .1,
            'checkpoint_retention': RETENTION, 'log_negative_ids': protocol['log_negative_ids'], 'test_evaluation': False}


def source_hashes():
    return {name: sha(HERE / name) for name in SOURCE_FILES}


def reference_hashes():
    return {name: sha(HERE.parent / 'references' / name) for name in REFERENCE_FILES}


def build_config(protocol, purpose, experiment_id, data_dir, pretrained, initializer, seed, encoder, curriculum, learning_rate, temperature,
                 sampler_seed=None, maximum_steps=None, evaluation_every_updates=None, maximum_seconds=13200, continuation_parent=None,
                 pretrained_sha256=None, extra=None):
    """Return a config that the trainer admits. pretrained_sha256 may be given only from a verified delivery receipt."""
    protocol_path = Path(protocol).resolve()
    rules_protocol = json.loads(protocol_path.read_text())
    assert set(initializer) == set(INITIALIZER_KEYS), {'missing': sorted(set(INITIALIZER_KEYS) - set(initializer)), 'unknown': sorted(set(initializer) - set(INITIALIZER_KEYS))}
    curriculum_rule = rules_protocol['curricula'][curriculum]
    rules = rules_protocol['purposes'][purpose]
    cadence = rules['evaluation_every_updates']
    cadences = cadence if isinstance(cadence, list) else [cadence]
    if evaluation_every_updates is None:
        assert len(cadences) == 1, 'purpose %s admits several evaluation cadences %s; choose one explicitly' % (purpose, cadences)
        evaluation_every_updates = cadences[0]
    if maximum_steps is None:
        assert len(rules['maximum_steps']) == 1, 'purpose %s admits several budgets %s; choose one explicitly' % (purpose, rules['maximum_steps'])
        maximum_steps = rules['maximum_steps'][0]
    data = Path(data_dir).resolve()
    config = {
        'experiment_id': experiment_id, 'phase': 'train', 'purpose': purpose,
        'protocol_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
        'source_hashes': source_hashes(), 'reference_hashes': reference_hashes(),
        'data_dir': str(data), 'data_complete_sha256': sha(data / 'COMPLETE.json'),
        'pretrained_checkpoint': str(Path(pretrained).resolve()), 'pretrained_sha256': pretrained_sha256 or sha(pretrained),
        'initializer': dict(initializer), 'seed': seed, 'sampler_seed': seed if sampler_seed is None else sampler_seed,
        'encoder': encoder, 'curriculum': curriculum, 'ontology_loss': curriculum_rule['ontology_loss'], 'mixture_id': curriculum_rule['mixture_id'],
        'mixture': rules_protocol['mixtures'][curriculum_rule['mixture_id']] if curriculum_rule['mixture_id'] else None,
        'learning_rate': learning_rate, 'temperature': temperature, 'maximum_steps': maximum_steps,
        'evaluation_every_updates': evaluation_every_updates, 'maximum_seconds': maximum_seconds, **fixed_fields(rules_protocol),
    }
    if purpose == 'hpo_continuation':
        assert continuation_parent, 'hpo_continuation requires continuation_parent'
        parent = Path(continuation_parent).resolve()
        progress = json.loads((parent / 'progress.json').read_text())
        config.update({'continuation_parent': str(parent), 'parent_complete_sha256': sha(parent / 'COMPLETE.json'), 'parent_checkpoint_sha256': progress['checkpoint_sha256']})
    else:
        assert continuation_parent is None, 'continuation_parent is only valid for hpo_continuation'
    for key, value in (extra or {}).items():
        assert key not in CANONICAL_KEYS, 'extra key %s would override a canonical key' % key
        config[key] = value
    check_config(config, rules_protocol)
    return config


def validate_config(path, verify_pretrained=True):
    """Recompute every pinned identity of an existing config; raise on the first contradiction."""
    path = Path(path)
    config = json.loads(path.read_text())
    receipt = {'config': str(path.resolve()), 'config_sha256': sha(path), 'checks': {}}
    checks = receipt['checks']
    protocol_path = Path(config['protocol_path'])
    assert sha(protocol_path) == config['protocol_sha256'], 'protocol hash mismatch'
    checks['protocol_sha256'] = True
    protocol = json.loads(protocol_path.read_text())
    check_config(config, protocol)
    checks['check_config'] = True
    assert config['source_hashes'] == source_hashes(), {'config': config['source_hashes'], 'actual': source_hashes()}
    checks['source_hashes_match_this_directory'] = True
    assert config['reference_hashes'] == reference_hashes() == protocol['reference_hashes'], 'reference hash mismatch'
    checks['reference_hashes_match_files_and_protocol'] = True
    data = Path(config['data_dir'])
    assert sha(data / 'COMPLETE.json') == config['data_complete_sha256'], 'bank COMPLETE hash mismatch'
    manifest = json.loads((data / 'COMPLETE.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(data / name) == (digest['sha256'] if isinstance(digest, dict) else digest), name
    checks['bank_files_match_complete'] = True
    bank = json.loads((data / 'bank.json').read_text())
    receipt['clinical_training_gene_domain'] = check_bank(bank, protocol['admitted_banks'][config['data_complete_sha256']], manifest)
    checks['bank_matches_protocol_record'] = True
    if verify_pretrained:
        assert sha(config['pretrained_checkpoint']) == config['pretrained_sha256'], 'pretrained export hash mismatch'
        checks['pretrained_sha256'] = True
    else:
        checks['pretrained_sha256'] = 'skipped'
    if config.get('continuation_parent'):
        parent = Path(config['continuation_parent'])
        assert sha(parent / 'COMPLETE.json') == config['parent_complete_sha256']
        progress = json.loads((parent / 'progress.json').read_text())
        assert sha(parent / progress['checkpoint']) == config['parent_checkpoint_sha256'] == progress['checkpoint_sha256']
        checks['continuation_parent'] = True
    receipt['passed'] = True
    return receipt


def write_config(config, path):
    path = Path(path)
    assert not path.exists(), 'refusing to overwrite %s' % path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--protocol', required=True)
    build.add_argument('--purpose', required=True, choices=['qualification', 'hpo_screen', 'hpo_continuation', 'production'])
    build.add_argument('--experiment-id', required=True)
    build.add_argument('--data-dir', required=True)
    build.add_argument('--pretrained', required=True)
    build.add_argument('--pretrained-sha256', help='only from a verified delivery receipt; validate still rehashes unless told to skip')
    build.add_argument('--taxonomy-mode', default='original')
    build.add_argument('--cohort', required=True, choices=['main', 'excluded'])
    build.add_argument('--modality', required=True, choices=['P', 'F', 'S', 'PF', 'PS', 'FS', 'PFS'])
    build.add_argument('--fold', type=fold_value, required=True, help='0-9, or excluded')
    build.add_argument('--pretrain-seed', type=int, required=True)
    build.add_argument('--pretrain-learning-rate', type=float, required=True)
    build.add_argument('--pretrain-regularization', type=float, required=True)
    build.add_argument('--initializer-update-budget', type=int, required=True)
    build.add_argument('--seed', type=int, required=True)
    build.add_argument('--sampler-seed', type=int)
    build.add_argument('--encoder', required=True, choices=['shared', 'independent', 'mean'])
    build.add_argument('--curriculum', required=True, choices=['full13', 'core3', 'aux_off'])
    build.add_argument('--learning-rate', type=float, required=True)
    build.add_argument('--temperature', type=float, required=True)
    build.add_argument('--maximum-steps', type=int)
    build.add_argument('--evaluation-every-updates', type=int)
    build.add_argument('--maximum-seconds', type=float, default=13200)
    build.add_argument('--continuation-parent')
    build.add_argument('--extra', help='JSON object of non-canonical keys, e.g. {"campaign_trial_id": "..."}')
    build.add_argument('--output', required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--config', required=True)
    validate.add_argument('--skip-pretrained-hash', action='store_true')
    validate.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        initializer = {'taxonomy_mode': args.taxonomy_mode, 'cohort': args.cohort, 'modality': args.modality, 'fold': args.fold, 'pretrain_seed': args.pretrain_seed,
                       'pretrain_learning_rate': args.pretrain_learning_rate, 'pretrain_regularization': args.pretrain_regularization,
                       'initializer_update_budget': args.initializer_update_budget}
        config = build_config(args.protocol, args.purpose, args.experiment_id, args.data_dir, args.pretrained, initializer, args.seed, args.encoder,
                              args.curriculum, args.learning_rate, args.temperature, sampler_seed=args.sampler_seed, maximum_steps=args.maximum_steps,
                              evaluation_every_updates=args.evaluation_every_updates, maximum_seconds=args.maximum_seconds,
                              continuation_parent=args.continuation_parent, pretrained_sha256=args.pretrained_sha256,
                              extra=json.loads(args.extra) if args.extra else None)
        write_config(config, args.output)
        print(json.dumps({'config': args.output, 'experiment_id': config['experiment_id'], 'purpose': config['purpose']}))
    else:
        receipt = validate_config(args.config, verify_pretrained=not args.skip_pretrained_hash)
        output = Path(args.output)
        assert not output.exists()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'receipt': str(output), 'passed': receipt['passed']}))


if __name__ == '__main__':
    main()
