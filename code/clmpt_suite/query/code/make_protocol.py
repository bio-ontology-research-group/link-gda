"""Generate and validate the production protocols (query v2, pretrain v2) of the CLMPT suite.

generate  Reads the frozen HPO protocols (query v1, pretrain v1), the common-recipe receipt written by
          `leaderboard.py freeze`, the selection decision, the data worker's main manifest
          (format clmpt-suite-inputs-v1) and the excluded-cohort entry files. Writes two NEW protocol files;
          refuses to overwrite anything and refuses to write over the v1 inputs.
validate  Re-reads generated protocols and every admitted bank, reruns train.check_bank on each record and
          the trainers' check_config on probe configs: every admitted (bank, modality) in production, the
          HPO primary-modality rule, and refusals for unadmitted modalities, cohorts and folds.

Rules. No new numerical search: the production recipe must lie in the v1 grids and match a v1 pretraining
recipe. One shared bank per main fold admits every modality whose manifest entry names that bank;
`admitted_modalities` is explicit in every record. The primary `modality` of a bank already admitted by
v1 is kept (PFS for the HPO bank, so HPO stays PFS-only); a new bank's primary is the first admitted
modality in MODALITY_ORDER. Cohort comes from the source: manifest entries are 'main', excluded entry
files must say 'excluded'. Bank records carry `entity_count` (the vocabulary, from bank num_entities and
every entry's pretraining entity_count) separately from `candidate_genes` (the ranking pool) and the
clinical TRAIN gene universe. A FROZEN output needs a complete manifest (10 folds x 7 modalities) and
excluded F, S and FS entries, and a non-synthetic receipt; otherwise only --draft writes status DRAFT,
which both trainers refuse.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from make_config import fixed_fields
from sampler import domain_identity, gene_domain
from train import COHORTS, INITIALIZER_KEYS, check_bank, check_config, sha

HERE = Path(__file__).resolve().parent
MODALITY_ORDER = ('PFS', 'FS', 'PS', 'PF', 'S', 'F', 'P')
MAIN_FOLDS = tuple(range(10))
EXCLUDED_MODALITIES = ('F', 'FS', 'S')


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def pretrain_module():
    directory = HERE.parent.parent / 'pretrain'
    sys.path.append(str(directory))
    spec = importlib.util.spec_from_file_location('suite_pretrain_train', directory / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_entries(manifest_path, excluded_paths, root):
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['format'] == 'clmpt-suite-inputs-v1', manifest.get('format')
    entries = []
    for name, entry in sorted(manifest['entries'].items()):
        assert entry.get('cohort', 'main') == 'main' and isinstance(entry['fold'], int), name
        entries.append(dict(entry, cohort='main', name=name))
    for path in excluded_paths:
        entry = json.loads(Path(path).read_text())
        assert entry['cohort'] == 'excluded', path
        entries.append(dict(entry, fold='excluded', name='excluded-' + entry['modality']))
    for entry in entries:
        assert entry['modality'] in MODALITY_ORDER and entry['cohort'] in COHORTS, entry['name']
        entry['bank_dir'] = str(resolve(entry['bank_dir'], root))
    expected = {'fold%d-%s' % (fold, modality) for fold in MAIN_FOLDS for modality in MODALITY_ORDER} | {'excluded-' + m for m in EXCLUDED_MODALITIES}
    present = {entry['name'] for entry in entries}
    assert len(present) == len(entries), 'duplicate entry names'
    return manifest, entries, sorted(expected - present), sorted(present - expected)


def bank_record(bank_sha, members, previous):
    """Protocol record for one shared bank, recomputed from the bank itself and its entries."""
    directory = Path(members[0]['bank_dir'])
    assert all(Path(member['bank_dir']) == directory for member in members), 'one bank hash must have one directory'
    assert sha(directory / 'COMPLETE.json') == bank_sha, directory
    manifest = json.loads((directory / 'COMPLETE.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(directory / name) == (digest['sha256'] if isinstance(digest, dict) else digest), name
    bank = json.loads((directory / 'bank.json').read_text())
    folds, cohorts = {member['fold'] for member in members}, {member['cohort'] for member in members}
    assert len(folds) == 1 and len(cohorts) == 1, {'bank': bank_sha, 'folds': sorted(map(str, folds)), 'cohorts': sorted(cohorts)}
    fold, cohort = folds.pop(), cohorts.pop()
    if 'fold' in manifest:
        assert manifest['fold'] == fold, ('bank COMPLETE fold', manifest['fold'], fold)
    for member in members:
        assert member['pretraining']['entity_count'] == bank['num_entities'], ('initializer vocabulary differs from bank vocabulary', member['name'])
    named = {member['modality'] for member in members} | set(previous['admitted_modalities'] if previous else [])
    admitted = [modality for modality in MODALITY_ORDER if modality in named]
    domain = domain_identity(gene_domain(bank))
    if cohort == 'excluded':
        assert domain['source'] == 'training_gene_ids', 'excluded bank must declare its clinical TRAIN gene universe explicitly'
    record = {'taxonomy_mode': 'original', 'cohort': cohort, 'fold': fold,
              'modality': previous['modality'] if previous else admitted[0], 'admitted_modalities': admitted,
              'entity_count': bank['num_entities'], 'candidate_genes': len(bank['candidate_ids']),
              'validation_cases': len(bank['validation']), 'validation_associations': sum(len(row['gene_ids']) for row in bank['validation']),
              'clinical_training_gene_domain': domain,
              'purpose_admitted': sorted(set(previous['purpose_admitted'] if previous else []) | {'qualification', 'production'}),
              'bank_dir_at_generation': str(directory), 'entries': sorted(member['name'] for member in members)}
    if previous:
        assert previous['modality'] in admitted, 'v1 primary modality must stay admitted'
        for key in ('taxonomy_mode', 'cohort', 'fold', 'entity_count', 'candidate_genes', 'validation_cases', 'validation_associations', 'clinical_training_gene_domain'):
            assert previous[key] == record[key], ('v1 record changed', key, previous[key], record[key])
    check_bank(bank, record, manifest)
    return record


def recipe_from_receipt(receipt, decision, query_v1, pretrain_v1, query_v1_sha, decision_sha):
    assert receipt['command'] == 'freeze' and receipt['test_access'] is False
    assert receipt['protocol_sha256'] == query_v1_sha, 'freeze receipt was not produced under this HPO protocol'
    assert receipt['decision_sha256'] == decision_sha and receipt['decision'] == decision
    recipe = receipt['production_recipe']
    assert recipe['common_leaderboard'] == decision['common_leaderboard']
    initializer = recipe['initializer_recipe']
    assert initializer in query_v1['hpo_initializer_recipes'], 'initializer recipe outside the HPO grid'
    assert recipe['learning_rate'] in query_v1['query_learning_rates'] and recipe['temperature'] in query_v1['temperatures'], 'query setting outside the HPO grid'
    matches = [name for name, value in pretrain_v1['recipes'].items() if value == {'learning_rate': initializer['pretrain_learning_rate'],
               'regularization': initializer['pretrain_regularization'], 'total_updates': initializer['initializer_update_budget']}]
    assert len(matches) == 1, 'production initializer recipe must equal exactly one v1 pretraining recipe'
    return recipe, matches[0]


def generate(args):
    root = Path(args.root).resolve()
    query_v1_path, pretrain_v1_path = Path(args.query_protocol).resolve(), Path(args.pretrain_protocol).resolve()
    outputs = [Path(args.output_query).resolve(), Path(args.output_pretrain).resolve()]
    assert not any(path.exists() for path in outputs) and not {query_v1_path, pretrain_v1_path} & set(outputs), 'refusing to overwrite'
    query_v1, pretrain_v1 = json.loads(query_v1_path.read_text()), json.loads(pretrain_v1_path.read_text())
    assert query_v1['status'] == pretrain_v1['status'] == 'FROZEN' and query_v1['production_recipe'] is None and pretrain_v1['production_recipe_id'] is None
    receipt = json.loads(Path(args.freeze).read_text())
    decision = json.loads(Path(args.decision).read_text())
    recipe, recipe_id = recipe_from_receipt(receipt, decision, query_v1, pretrain_v1, sha(query_v1_path), sha(args.decision))
    manifest, entries, missing, unexpected = load_entries(args.manifest, args.excluded_entry, root)
    assert not unexpected, unexpected
    frozen = not missing and manifest['complete'] is True and not receipt.get('synthetic')
    assert frozen or args.draft, {'missing_entries': len(missing), 'manifest_complete': manifest['complete'], 'synthetic_receipt': bool(receipt.get('synthetic'))}
    banks = {}
    for entry in entries:
        banks.setdefault(entry['bank_complete_sha256'], []).append(entry)
    records = {bank_sha: bank_record(bank_sha, members, query_v1['admitted_banks'].get(bank_sha)) for bank_sha, members in sorted(banks.items())}
    for bank_sha, previous in query_v1['admitted_banks'].items():
        records.setdefault(bank_sha, copy.deepcopy(previous))
    status = 'FROZEN' if frozen else 'DRAFT'
    derived = {'query_v1_sha256': sha(query_v1_path), 'pretrain_v1_sha256': sha(pretrain_v1_path), 'freeze_receipt_sha256': sha(args.freeze),
               'decision_sha256': sha(args.decision), 'manifest_sha256': sha(args.manifest), 'manifest_root_sha256': manifest['root_sha256'],
               'excluded_entries_sha256': {str(path): sha(path) for path in args.excluded_entry}, 'missing_entries': missing}
    query_v2 = copy.deepcopy(query_v1)
    query_v2.update({'status': status, 'protocol_id': 'clmpt-suite-query-v2',
                     'status_note': 'Production query protocol generated by make_protocol.py from the frozen common recipe and the data manifests. '
                                    'HPO purposes, grids, sampler and numerics are unchanged from v1; HPO stays restricted to each bank primary modality.',
                     'admitted_banks': records, 'derived_from': derived,
                     'production_recipe': {'initializer_recipe': recipe['initializer_recipe'], 'learning_rate': recipe['learning_rate'], 'temperature': recipe['temperature'],
                                           'common_leaderboard': recipe['common_leaderboard'], 'source_trajectory': recipe['source_trajectory'],
                                           'source_selected_step': recipe['source_selected_step'], 'pretrain_recipe_id': recipe_id}})
    graphs = copy.deepcopy(pretrain_v1['admitted_graphs'])
    for entry in entries:
        record = {'taxonomy_mode': 'original', 'modality': entry['modality'], 'fold': entry['fold'], 'cohort': entry['cohort'],
                  'training_triples': entry['pretraining']['training_triples'], 'entity_count': entry['pretraining']['entity_count'],
                  'relation_count': entry['pretraining']['relation_count'], 'purpose_admitted': ['production', 'qualification'], 'entry': entry['name']}
        previous = graphs.get(entry['graph_complete_sha256'])
        if previous:
            for key in ('taxonomy_mode', 'modality', 'fold', 'cohort', 'training_triples', 'entity_count', 'relation_count'):
                assert previous[key] == record[key], ('graph record changed', entry['name'], key)
            record['purpose_admitted'] = sorted(set(previous['purpose_admitted']) | set(record['purpose_admitted']))
        graphs[entry['graph_complete_sha256']] = record
    pretrain_v2 = copy.deepcopy(pretrain_v1)
    pretrain_v2.update({'status': status, 'protocol_id': 'clmpt-suite-pretrain-v2', 'production_recipe_id': recipe_id, 'admitted_graphs': graphs, 'derived_from': derived,
                        'status_note': 'Production pretraining protocol generated by make_protocol.py: the frozen common recipe, seeds 0-9, every manifest graph; no new search.'})
    for path, value in zip(outputs, (query_v2, pretrain_v2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(canonical_bytes(value))
    summary = {'status': status, 'query_protocol': str(outputs[0]), 'query_sha256': sha(outputs[0]), 'pretrain_protocol': str(outputs[1]), 'pretrain_sha256': sha(outputs[1]),
               'banks': {bank_sha: {key: record[key] for key in ('cohort', 'fold', 'modality', 'admitted_modalities', 'entity_count', 'candidate_genes', 'validation_cases', 'validation_associations')}
                         | {'gene_domain_source': record['clinical_training_gene_domain']['source'], 'gene_domain_count': record['clinical_training_gene_domain']['count']}
                         for bank_sha, record in records.items()},
               'graphs': len(graphs), 'missing_entries': len(missing), 'production_recipe_id': recipe_id}
    print(json.dumps(summary, indent=2, default=str))
    return summary


def probe(protocol, protocol_sha, bank_sha, modality, purpose, recipe=None):
    record = protocol['admitted_banks'][bank_sha]
    recipe = recipe or protocol['production_recipe']
    initializer = {'taxonomy_mode': record['taxonomy_mode'], 'cohort': record['cohort'], 'modality': modality, 'fold': record['fold'], 'pretrain_seed': 0,
                   **recipe['initializer_recipe']}
    rules = protocol['purposes'][purpose]
    cadence = rules['evaluation_every_updates']
    config = {'experiment_id': 'probe', 'phase': 'train', 'purpose': purpose, 'protocol_sha256': protocol_sha, 'data_complete_sha256': bank_sha,
              'initializer': initializer, 'seed': 0, 'sampler_seed': 0, 'encoder': 'shared', 'curriculum': 'full13', 'ontology_loss': True,
              'mixture_id': 'F13_sqrt', 'mixture': protocol['mixtures']['F13_sqrt'], 'learning_rate': recipe['learning_rate'], 'temperature': recipe['temperature'],
              'maximum_steps': rules['maximum_steps'][0], 'evaluation_every_updates': cadence[-1] if isinstance(cadence, list) else cadence, 'maximum_seconds': 13200,
              **fixed_fields(protocol)}
    if purpose == 'hpo_continuation':
        config['continuation_parent'] = 'probe'
    return config


def refused(function):
    try:
        function()
    except (AssertionError, KeyError):
        return True
    return False


def validate(args):
    query_path, pretrain_path = Path(args.query_protocol), Path(args.pretrain_protocol)
    query, pretrain = json.loads(query_path.read_text()), json.loads(pretrain_path.read_text())
    query_sha = sha(query_path)
    pretrain_train = pretrain_module()
    checks, failures = {}, []
    assert query['status'] == pretrain['status'] and query['status'] in ('FROZEN', 'DRAFT')
    if query['status'] == 'DRAFT':
        frozen_view = dict(query, status='FROZEN')
        frozen_pretrain = dict(pretrain, status='FROZEN')
        checks['draft_refused_by_query_trainer'] = refused(lambda: check_config(probe(query, query_sha, next(iter(query['admitted_banks'])), 'PFS', 'production'), query))
    else:
        frozen_view, frozen_pretrain = query, pretrain
    hpo = {bank_sha: record for bank_sha, record in query['admitted_banks'].items() if 'hpo' in record['purpose_admitted']}
    for bank_sha, record in sorted(query['admitted_banks'].items()):
        directory = Path(record.get('bank_dir_at_generation', ''))
        if args.root_override and record.get('bank_dir_at_generation'):
            directory = Path(args.root_override) / directory.relative_to(args.generation_root)
        if record.get('bank_dir_at_generation'):
            manifest = json.loads((directory / 'COMPLETE.json').read_text())
            assert sha(directory / 'COMPLETE.json') == bank_sha
            check_bank(json.loads((directory / 'bank.json').read_text()), record, manifest)
            checks['bank_record_matches_bank:%s' % bank_sha[:12]] = True
        for modality in MODALITY_ORDER:
            config = probe(frozen_view, query_sha, bank_sha, modality, 'production')
            admitted = modality in record['admitted_modalities'] and 'production' in record['purpose_admitted']
            outcome = not refused(lambda: check_config(config, frozen_view))
            key = 'production_%s_%s:%s' % ('admitted' if admitted else 'refused', modality, bank_sha[:12])
            checks[key] = outcome == admitted
        wrong_cohort = probe(frozen_view, query_sha, bank_sha, record['modality'], 'production')
        wrong_cohort['initializer']['cohort'] = 'main' if record['cohort'] == 'excluded' else 'excluded'
        checks['wrong_cohort_refused:%s' % bank_sha[:12]] = refused(lambda: check_config(wrong_cohort, frozen_view))
        wrong_fold = probe(frozen_view, query_sha, bank_sha, record['modality'], 'production')
        wrong_fold['initializer']['fold'] = 9 if record['fold'] != 9 else 8
        checks['wrong_fold_refused:%s' % bank_sha[:12]] = refused(lambda: check_config(wrong_fold, frozen_view))
        missing_cohort = probe(frozen_view, query_sha, bank_sha, record['modality'], 'production')
        del missing_cohort['initializer']['cohort']
        checks['initializer_without_cohort_refused:%s' % bank_sha[:12]] = refused(lambda: check_config(missing_cohort, frozen_view))
    hpo_recipe = {'initializer_recipe': frozen_view['hpo_initializer_recipes'][0], 'learning_rate': frozen_view['query_learning_rates'][0], 'temperature': frozen_view['temperatures'][0]}
    for bank_sha, record in hpo.items():
        for modality in MODALITY_ORDER:
            outcome = not refused(lambda: check_config(probe(frozen_view, query_sha, bank_sha, modality, 'hpo_screen', hpo_recipe), frozen_view))
            checks['hpo_%s_%s:%s' % ('admitted' if modality == record['modality'] else 'refused', modality, bank_sha[:12])] = outcome == (modality == record['modality'])
    checks['hpo_banks_unchanged_primary_PFS'] = all(record['modality'] == 'PFS' for record in hpo.values())
    recipe = frozen_pretrain['recipes'][frozen_pretrain['production_recipe_id']]
    for graph_sha, graph in sorted(frozen_pretrain['admitted_graphs'].items()):
        if 'production' not in graph['purpose_admitted']:
            continue
        config = {'purpose': 'production', 'seed': 9, 'strict_deterministic_algorithms': True, 'rolling_full_checkpoints': 2, 'checkpoint_every_updates': 50000,
                  'checkpoint_every_seconds': 1800, 'maximum_seconds': 86400, 'data_complete_sha256': graph_sha,
                  'graph': {key: graph[key] for key in ('taxonomy_mode', 'modality', 'fold', 'cohort')}, 'recipe_id': frozen_pretrain['production_recipe_id'],
                  'learning_rate': recipe['learning_rate'], 'regularization': recipe['regularization'], 'maximum_steps': recipe['total_updates'],
                  'milestone_updates': [recipe['total_updates']], 'training_triples': graph['training_triples'], 'entity_count': graph['entity_count'],
                  'relation_count': graph['relation_count'], **frozen_pretrain['fixed']}
        checks['pretrain_production_admitted:%s' % graph_sha[:12]] = not refused(lambda: pretrain_train.check_config(config, frozen_pretrain))
        other = dict(config, recipe_id=next(name for name in frozen_pretrain['recipes'] if name != frozen_pretrain['production_recipe_id']))
        checks['pretrain_other_recipe_refused:%s' % graph_sha[:12]] = refused(lambda: pretrain_train.check_config(other, frozen_pretrain))
    failures = [key for key, value in checks.items() if not value]
    receipt = {'query_protocol': str(query_path.resolve()), 'query_sha256': query_sha, 'pretrain_protocol': str(pretrain_path.resolve()), 'pretrain_sha256': sha(pretrain_path),
               'status': query['status'], 'checks': checks, 'failures': failures, 'passed': not failures,
               'scope': 'CPU-only: protocol records against banks and trainer admission rules on probe configs; no model, graph or GPU.'}
    output = Path(args.output)
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'receipt': str(output), 'passed': receipt['passed'], 'checks': len(checks), 'failures': failures}))
    return 0 if receipt['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--query-protocol', required=True, help='frozen HPO query protocol (v1)')
    g.add_argument('--pretrain-protocol', required=True, help='frozen pretrain protocol (v1)')
    g.add_argument('--freeze', required=True, help='leaderboard.py freeze output')
    g.add_argument('--decision', required=True, help='SELECTION-DECISION.json used by the freeze')
    g.add_argument('--manifest', required=True, help='data worker main manifest (clmpt-suite-inputs-v1)')
    g.add_argument('--excluded-entry', action='append', default=[], help='excluded/entries/excluded-<M>.json; repeat')
    g.add_argument('--root', default='.', help='repository root that relative entry paths resolve against')
    g.add_argument('--draft', action='store_true', help='allow incomplete inputs or a synthetic receipt; writes status DRAFT')
    g.add_argument('--output-query', required=True)
    g.add_argument('--output-pretrain', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--query-protocol', required=True)
    v.add_argument('--pretrain-protocol', required=True)
    v.add_argument('--root-override', help='re-root bank paths recorded at generation (for example on IBEX)')
    v.add_argument('--generation-root', default='/', help='prefix of the recorded bank paths to replace with --root-override')
    v.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args)
        return 0
    return validate(args)


if __name__ == '__main__':
    raise SystemExit(main())
