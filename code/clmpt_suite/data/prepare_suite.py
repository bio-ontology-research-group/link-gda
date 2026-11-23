"""CPU preparation of the CLMPT suite inputs: ten supplied folds by seven native-annotation modalities.

Layout under one write-once ROOT:
  ROOT.json                         pinned identities and the fallback order policy
  bases/fold{k}/                    fold PFS graph, bank and TEST cases (folds 3-9 built by the pinned fold builder)
  bases/fold{k}/BASE.json           for folds 0-2: the adopted existing artifacts, by path and COMPLETE hash
  qualification/                    fold0 qualification and the cross-fold fallback audit
  graphs/fold{k}/{M}/               ComplEx input per fold and modality (triples.npy, vocabulary.json, candidates.txt,
                                    union-audit.json, modality.json for derived arms, COMPLETE.json)
  banks/fold{k}/                    query-stage bank, shared by all modalities of the fold (hardlinks of the base bank)
  tests/fold{k}/                    TEST cases (hardlinks of the base)
  entries/fold{k}-{M}.json          write-once identity record per prepared graph

Nothing is projected or sampled: the ontology bank is the audited fold0 axiom13 bank remapped through IRIs. Vocabulary,
ontology background, TRAIN associated_with edges and TRAIN profiles are never changed by a modality. The PFS graph of
every fold is the fold base itself, linked with an unchanged COMPLETE.json so existing initializer identities hold.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import modalities as mod
from common import (BENCHMARK_INPUTS, CANDIDATES_SHA256, CLASS_POOL, CLASS_POOL_SHA256, DISEASE_PREFIX,
                    EXISTING_BASES, NON_DISEASE_ENTITIES, EXPECTED_FALLBACK_GENES, EXPECTED_FALLBACK_TEST_PAIRS,
                    EXPECTED_OUTER_TEST_PAIRS, FOLD_BUILDER, FOLD_BUILDER_SHA256, FOLDS, HP_PREFIX, IRI_BANK,
                    IRI_BANK_COMPLETE, LEGACY_SHORTCUT, MODALITIES, NATIVE_RELATION, PATTERN_COUNT, PREPARE_SHA256,
                    PREPARE_SOURCE, RELATION_COUNT, REPRESENTED_CLASSES, Refusal, check, link_or_copy, pair_rows,
                    read, repo_path, sha, sha_bytes, verify_complete, write_json)

GRAPH_FILES = ('triples.npy', 'candidates.txt', 'vocabulary.json', 'union-audit.json')
FORMAT = 'clmpt-suite-inputs-v1'


def root_record(root):
    root = pathlib.Path(root)
    check((root / 'ROOT.json').is_file(), 'ROOT.json missing; run init first')
    record = read(root / 'ROOT.json')
    here = pathlib.Path(__file__).resolve()
    check(record['format'] == FORMAT and record['builder_sha256'] == sha(here) and record['modalities_sha256'] == sha(pathlib.Path(mod.__file__).resolve())
          and record['common_sha256'] == sha(here.parent / 'common.py'), 'ROOT.json was written by different builder sources; use a fresh root')
    return record


def fold_paths(fold):
    """Benchmark partition files for FOLD."""
    base = repo_path(BENCHMARK_INPUTS) / 'benchmark' / 'data' / ('fold%d' % fold)
    return {name: base / name for name in ('training.tsv', 'validation.tsv', 'test.tsv', 'profiles.json', 'candidates.txt')} | {'raw': base / 'raw' / 'train.txt'}


def manifest_sha(fold, relative):
    """SHA-256 of a benchmark file as listed in MANIFESTS/benchmark-fold{k}.json."""
    manifest = read(repo_path(BENCHMARK_INPUTS) / 'MANIFESTS' / ('benchmark-fold%d.json' % fold))
    return manifest['files']['benchmark/data/fold%d/%s' % (fold, relative)]['sha256']


def cmd_init(args):
    root = pathlib.Path(args.root)
    check(not root.exists(), 'root exists; choose a fresh directory')
    check(args.order_policy in mod.ORDER_POLICIES, 'unknown order policy')
    if args.order_policy == 'prepared-csv':
        check(args.order_source and args.order_sha256, 'prepared-csv needs --order-source and --order-sha256')
        check(sha(args.order_source) == args.order_sha256, 'order source hash differs from --order-sha256')
        order = {'policy': 'prepared-csv', 'path': str(pathlib.Path(args.order_source).resolve()), 'sha256': args.order_sha256}
    else:
        check(args.order_source is None, 'raw-train-line-order takes each fold raw/train.txt from the benchmark manifest')
        order = {'policy': 'raw-train-line-order', 'path': 'benchmark/data/fold{k}/raw/train.txt', 'sha256': 'per-fold MANIFESTS/benchmark-fold{k}.json'}
    verify_complete(repo_path(IRI_BANK), IRI_BANK_COMPLETE)
    check(sha(repo_path(CLASS_POOL)) == CLASS_POOL_SHA256, 'class pool hash mismatch')
    check(sha(repo_path(FOLD_BUILDER)) == FOLD_BUILDER_SHA256 and sha(repo_path(PREPARE_SOURCE)) == PREPARE_SHA256, 'fold builder or prepare.py differs from the executed sources')
    root.mkdir(parents=True)
    for name in ('bases', 'qualification', 'graphs', 'banks', 'tests', 'entries'):
        (root / name).mkdir()
    record = {'format': FORMAT, 'created_unix': time.time(), 'builder_sha256': sha(pathlib.Path(__file__).resolve()),
              'modalities_sha256': sha(pathlib.Path(mod.__file__).resolve()), 'common_sha256': sha(pathlib.Path(__file__).resolve().parent / 'common.py'),
              'fold_builder': {'path': FOLD_BUILDER, 'sha256': FOLD_BUILDER_SHA256, 'vendored_byte_identical_from': 'notes/clmpt-threefold-pilot-20261116/build_fold_inputs.py'},
              'prepare': {'path': PREPARE_SOURCE, 'sha256': PREPARE_SHA256},
              'existing_bases': {str(k): {part: {'path': p, 'complete_sha256': h} for part, (p, h) in v.items()} for k, v in EXISTING_BASES.items()},
              'iri_bank_complete_sha256': IRI_BANK_COMPLETE, 'class_pool_sha256': CLASS_POOL_SHA256, 'fallback_order': order,
              'protocol': {'folds': list(FOLDS), 'modalities': list(MODALITIES), 'taxonomy': 'original', 'legacy_causes_phenotype': 'absent by fixed protocol',
                           'fallback': 'first represented phenotype in the pinned order for genes with native P support and no active-relation support; decided from native annotations only, before TRAIN associations',
                           'bank': 'one bank per fold shared by every modality; the fold0 axiom13 original bank remapped through IRIs; no new projection or sampled queries'}}
    write_json(root / 'ROOT.json', record)
    print(json.dumps({'root': str(root), 'root_sha256': sha(root / 'ROOT.json')}))


def base_dirs(root, fold):
    """Graph, bank and TEST directories of the fold base, with expected COMPLETE hashes when known."""
    root = pathlib.Path(root)
    if fold in EXISTING_BASES:
        return {part: (repo_path(path), digest) for part, (path, digest) in EXISTING_BASES[fold].items()}
    built = root / 'bases' / ('fold%d' % fold) / 'output'
    receipt = built / 'RECEIPT.json'
    check(receipt.is_file(), 'fold %d base is not built; run base --fold %d' % (fold, fold))
    value = read(receipt)
    return {'graph': (built / ('graph-fold%d' % fold), value['graph_complete_sha256']),
            'bank': (built / ('bank-fold%d' % fold), value['bank_complete_sha256']),
            'test': (built / ('test-fold%d' % fold), value['test_complete_sha256'])}


def load_fold_builder():
    path = repo_path(FOLD_BUILDER)
    check(sha(path) == FOLD_BUILDER_SHA256, 'fold builder hash mismatch')
    spec = importlib.util.spec_from_file_location('threefold_build_fold_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_base(args):
    root = pathlib.Path(args.root)
    root_record(root)
    fold = args.fold
    target = root / 'bases' / ('fold%d' % fold)
    check(not target.exists(), 'base for fold %d already recorded' % fold)
    target.mkdir()
    if fold in EXISTING_BASES:
        record = {'fold': fold, 'source': 'adopted existing artifacts; not rebuilt'}
        for part, (path, digest) in base_dirs(root, fold).items():
            verify_complete(path, digest)
            record[part] = {'path': str(path), 'complete_sha256': digest}
        write_json(target / 'BASE.json', record)
        print(json.dumps(record))
        return
    builder = load_fold_builder()
    fold0 = EXISTING_BASES[0]
    namespace = types.SimpleNamespace(fold=fold, benchmark_inputs=str(repo_path(BENCHMARK_INPUTS)), prepare_source=str(repo_path(PREPARE_SOURCE)),
                                      fold0_graph=str(repo_path(fold0['graph'][0])), fold0_bank=str(repo_path(fold0['bank'][0])),
                                      iri_bank=str(repo_path(IRI_BANK)), class_pool=str(repo_path(CLASS_POOL)),
                                      work=str(root / 'work'), output=str(target / 'output'))
    try:
        builder.build(namespace)
    except builder.Refusal as error:
        raise Refusal('fold builder refused fold %d: %s' % (fold, error))
    receipt = read(target / 'output' / 'RECEIPT.json')
    write_json(target / 'BASE.json', {'fold': fold, 'source': 'built by the pinned fold builder', 'builder_sha256': FOLD_BUILDER_SHA256,
                                      'receipt_sha256': sha(target / 'output' / 'RECEIPT.json'),
                                      'graph': {'complete_sha256': receipt['graph_complete_sha256']}, 'bank': {'complete_sha256': receipt['bank_complete_sha256']},
                                      'test': {'complete_sha256': receipt['test_complete_sha256']}})


def ontology_iri_digest(bank, names, relation_names):
    """Order-sensitive digest of all ontology records with every id replaced by its IRI."""
    digest = hashlib.sha256()
    for listing in ('ontology_queries', 'additional_ontology_queries'):
        for row in bank[listing]:
            value = [listing, row['source_id'], row['type'], row['sampleable'],
                     sorted((k, names[v]) for k, v in row['constants'].items()),
                     [[a, relation_names[r], b] for a, r, b in row['atoms']],
                     sorted(names[v] for v in row['answers']), sorted(names[v] for v in row['asserted_answers']),
                     sorted(names[v] for v in row['negative_known']),
                     None if row['taxonomy_target'] is None else names[row['taxonomy_target']]]
            digest.update(json.dumps(value, separators=(',', ':')).encode())
    return digest.hexdigest()


def verify_base(root, fold):
    """Independent integrity checks of one fold base; returns counts and digests."""
    dirs = base_dirs(root, fold)
    for part, (path, digest) in dirs.items():
        verify_complete(path, digest)
    graph_dir, bank_dir, test_dir = dirs['graph'][0], dirs['bank'][0], dirs['test'][0]
    files = fold_paths(fold)
    vocab = read(graph_dir / 'vocabulary.json')
    entities, relations = vocab['entities'], vocab['relations']
    names = {v: n for n, v in entities.items()}
    relation_names = {v: n for n, v in relations.items()}
    non_disease = sum(not n.startswith(DISEASE_PREFIX) for n in entities)
    check(non_disease == NON_DISEASE_ENTITIES and len(relations) == RELATION_COUNT, 'fold %d shared non-disease vocabulary or relation table differs' % fold)
    check(sha(graph_dir / 'candidates.txt') == CANDIDATES_SHA256, 'candidate list differs from the 13,625 main pool')
    check(not any(LEGACY_SHORTCUT in n for n in relations), 'legacy causes_phenotype relation present')
    triples = np.load(graph_dir / 'triples.npy', mmap_mode='r')
    ids = mod.relation_ids(vocab)
    training, validation, test = pair_rows(files['training.tsv']), pair_rows(files['validation.tsv']), pair_rows(files['test.tsv'])
    profiles = read(files['profiles.json'])
    train_d, val_d, test_d = {d for _, d in training}, {d for _, d in validation}, {d for _, d in test}
    check(not (train_d & val_d or train_d & test_d or val_d & test_d), 'fold %d partitions share diseases' % fold)
    vocab_diseases = {n for n in entities if n.startswith(DISEASE_PREFIX)}
    check(vocab_diseases == train_d and len(entities) == NON_DISEASE_ENTITIES + len(train_d), 'fold %d vocabulary diseases differ from TRAIN diseases' % fold)
    check(not ((val_d | test_d) & entities.keys()), 'held-out disease in fold %d vocabulary' % fold)
    association = np.asarray(triples[triples[:, 1] == ids['associated_with'][0]])
    check({(int(h), int(t)) for h, _, t in association.tolist()} == {(entities[g], entities[d]) for g, d in training}, 'fold %d associated_with differs from training.tsv' % fold)
    symptom = np.asarray(triples[triples[:, 1] == ids['has_symptom'][0]])
    check({(int(h), int(t)) for h, _, t in symptom.tolist()} == {(entities[d], entities[p]) for d in train_d for p in profiles[d]}, 'fold %d has_symptom differs from TRAIN profiles' % fold)
    gene_hp = 0
    for key, name in NATIVE_RELATION.items():
        rows = np.asarray(triples[triples[:, 1] == ids[name][0]])
        gene_hp += sum(names[t].startswith(HP_PREFIX) for t in np.unique(rows[:, 2]).tolist())
    check(gene_hp == 0, 'native gene annotation points to an HP class')
    bank = read(bank_dir / 'bank.json')
    check(bank.get('num_entities', len(entities)) == len(entities), 'fold %d bank num_entities differs from its vocabulary' % fold)
    disease_ids = {entities[d] for d in train_d}
    candidates = (graph_dir / 'candidates.txt').read_text().splitlines()
    check(bank['candidate_ids'] == [entities[g] for g in candidates], 'bank candidate order differs from candidates.txt')
    class_ids = set(bank['class_ids'])
    check(len(class_ids) == REPRESENTED_CLASSES and not (class_ids & disease_ids) and not any(mod.is_gene(names[c]) for c in class_ids), 'class pool is not the 175,906 represented classes or holds genes/diseases')
    types_seen = set()
    for listing in ('ontology_queries', 'additional_ontology_queries'):
        for row in bank[listing]:
            types_seen.add(row['type'])
            values = set(row['constants'].values()) | set(row['answers']) | set(row['negative_known'])
            check(not (values & disease_ids), 'disease id inside ontology supervision')
            check(set(row['answers']) <= class_ids and set(row['negative_known']) <= class_ids, 'ontology answers outside represented classes')
    check(len(types_seen) == PATTERN_COUNT, 'fold %d bank has %d patterns' % (fold, len(types_seen)))
    hp_ids = set(bank['hp_ids'])
    train_genes = {entities[g] for g, _ in training}
    for row in bank['training_profiles']:
        check(not (set(row['constants'].values()) & disease_ids) and set(row['hp_ids']) <= hp_ids, 'disease id or non-HP in TRAIN profile constants')
        check(set(row['answers']) <= train_genes and set(row['disease_names']) <= train_d, 'TRAIN profile answers outside TRAIN')
    gold = {}
    for name, rows, cases in (('validation', validation, bank['validation']), ('test', test, read(test_dir / 'test-cases.json')['cases'])):
        expected = collections.defaultdict(set)
        for g, d in rows:
            expected[d].add(entities[g])
        check(len({c['disease_name'] for c in cases}) == len(cases), '%s cases are not distinct diseases' % name)
        check({c['disease_name']: set(c['gene_ids']) for c in cases} == dict(expected), 'fold %d %s labels differ from %s.tsv' % (fold, name, name))
        check(all(set(c['hp_ids']) <= hp_ids and not (set(c['hp_ids']) & disease_ids) for c in cases), '%s profile constants contain a disease or non-HP id' % name)
        gold[name] = {'cases': len(cases), 'associations': len(rows)}
    candidate_ids = set(bank['candidate_ids'])
    check(all(set(c['gene_ids']) <= candidate_ids for c in bank['validation']), 'validation target outside candidates')
    return {'fold': fold, 'graph_complete_sha256': dirs['graph'][1], 'bank_complete_sha256': dirs['bank'][1], 'test_complete_sha256': dirs['test'][1],
            'triples': int(len(triples)), 'training_associations': len(training), 'training_diseases': len(train_d), **{k: v for k, v in gold.items()},
            'training_profiles': len(bank['training_profiles']), 'ontology_groups': len(bank['ontology_queries']) + len(bank['additional_ontology_queries']),
            'ontology_iri_digest': ontology_iri_digest(bank, names, relation_names), 'patterns': PATTERN_COUNT,
            'checks': {'associated_with_equals_training_tsv': True, 'has_symptom_equals_train_profiles': True, 'heldout_diseases_absent': True,
                       'no_disease_ids_in_ontology_or_clinical_constants': True, 'validation_and_test_labels_match_partitions': True,
                       'legacy_causes_phenotype_absent': True, 'candidate_order_13625': True}}


def cmd_verify_base(args):
    root = pathlib.Path(args.root)
    root_record(root)
    result = verify_base(root, args.fold)
    reference = root / 'qualification' / 'base-fold0.json'
    if args.fold != 0:
        check(reference.is_file(), 'verify fold 0 first; its ontology IRI digest is the reference')
        check(result['ontology_iri_digest'] == read(reference)['ontology_iri_digest'], 'fold %d ontology bank is not the fold0 bank under IRI remap' % args.fold)
    write_json(root / 'qualification' / ('base-fold%d.json' % args.fold), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))


def fold_order(root, fold):
    record = root_record(root)['fallback_order']
    if record['policy'] == 'prepared-csv':
        return mod.load_order('prepared-csv', record['path'], record['sha256'])
    return mod.load_order('raw-train-line-order', fold_paths(fold)['raw'], manifest_sha(fold, 'raw/train.txt'))


def derive_one(root, fold, modality):
    dirs = base_dirs(root, fold)
    graph_dir = dirs['graph'][0]
    verify_complete(graph_dir, dirs['graph'][1])
    vocab = read(graph_dir / 'vocabulary.json')
    base = np.load(graph_dir / 'triples.npy')
    candidates = (graph_dir / 'candidates.txt').read_text().splitlines()
    support, phenotype = mod.native_support(base, vocab)
    genes = mod.fallback_genes(support, modality)
    chosen, provenance = {}, None
    if genes:
        order, provenance = fold_order(root, fold)
        chosen = mod.choose_fallback(genes, phenotype, vocab, order)
    triples, audit = mod.derive(base, vocab, candidates, modality, chosen)
    audit.update({'fold': fold, 'base_graph_complete_sha256': dirs['graph'][1], 'fallback_order': provenance})
    return triples, audit, vocab


def cmd_qualify(args):
    """Fold0 base checks, all six derived fold0 modalities as metadata, and the cross-fold fallback TEST audit."""
    root = pathlib.Path(args.root)
    root_record(root)
    started = time.time()
    if not (root / 'qualification' / 'base-fold0.json').is_file():
        write_json(root / 'qualification' / 'base-fold0.json', verify_base(root, 0))
    tests = [(k, pair_rows(fold_paths(k)['test.tsv'])) for k in FOLDS]
    check(sum(len(rows) for _, rows in tests) == EXPECTED_OUTER_TEST_PAIRS, 'outer TEST does not total 6,571 rows')
    check(sum(manifest_sha(k, 'test.tsv') == sha(fold_paths(k)['test.tsv']) for k in FOLDS) == len(FOLDS), 'TEST files differ from the benchmark manifest')
    summary = {}
    for modality in MODALITIES:
        if modality == 'PFS':
            continue
        triples, audit, vocab = derive_one(root, 0, modality)
        fallback_names = {g for g, _ in audit['fallback_edges']}
        audit['fallback_test_pairs'] = mod.fallback_test_pairs(fallback_names, tests)
        if modality in EXPECTED_FALLBACK_GENES:
            check(audit['fallback_genes'] == EXPECTED_FALLBACK_GENES[modality], '%s fallback genes %d, expected %d' % (modality, audit['fallback_genes'], EXPECTED_FALLBACK_GENES[modality]))
            check(audit['fallback_test_pairs']['total'] == EXPECTED_FALLBACK_TEST_PAIRS[modality], '%s fallback TEST pairs %d, expected %d' % (modality, audit['fallback_test_pairs']['total'], EXPECTED_FALLBACK_TEST_PAIRS[modality]))
        else:
            check(audit['fallback_genes'] == 0, '%s must not need a fallback' % modality)
        audit['triples_sha256_in_memory'] = sha_bytes(np.ascontiguousarray(triples).tobytes())
        audit['triples_npy_bytes_estimate'] = int(triples.nbytes + 128)
        write_json(root / 'qualification' / ('fold0-%s.json' % modality), audit)
        summary[modality] = {k: audit[k] for k in ('triples', 'fallback_genes', 'fallback_genes_with_train_association', 'fallback_edges_sha256', 'isolated_candidates',
                                                   'candidates_without_active_or_fallback_annotation', 'triples_npy_bytes_estimate')} | {'fallback_test_pairs': audit['fallback_test_pairs']}
    write_json(root / 'qualification' / 'QUALIFICATION.json', {'fold': 0, 'modalities': summary, 'expected_fallback_genes': EXPECTED_FALLBACK_GENES,
                                                              'expected_fallback_test_pairs': EXPECTED_FALLBACK_TEST_PAIRS, 'outer_test_pairs': EXPECTED_OUTER_TEST_PAIRS,
                                                              'elapsed_seconds': time.time() - started, 'scope': 'metadata only; no derived triples written'})
    print(json.dumps(summary))


def cmd_modality(args):
    root = pathlib.Path(args.root)
    root_record(root)
    fold, modality = args.fold, args.modality
    check((root / 'qualification' / 'QUALIFICATION.json').is_file(), 'run qualify first')
    check((root / 'qualification' / ('base-fold%d.json' % fold)).is_file(), 'run verify-base --fold %d first' % fold)
    entry = root / 'entries' / ('fold%d-%s.json' % (fold, modality))
    check(not entry.exists(), 'fold %d %s already prepared' % (fold, modality))
    target = root / 'graphs' / ('fold%d' % fold) / modality
    check(not target.exists(), 'graph directory exists without an entry; inspect %s by hand' % target)
    dirs = base_dirs(root, fold)
    graph_dir, graph_complete = dirs['graph']
    verify_complete(graph_dir, graph_complete)
    target.mkdir(parents=True)
    links = {}
    if modality == 'PFS':
        for name in GRAPH_FILES + ('COMPLETE.json',):
            links[name] = link_or_copy(graph_dir / name, target / name)
        complete = sha(target / 'COMPLETE.json')
        check(complete == graph_complete, 'PFS COMPLETE.json changed while linking')
        record = {'fold': fold, 'modality': 'PFS', 'graph_dir': str(target), 'graph_complete_sha256': complete, 'base_graph_complete_sha256': graph_complete,
                  'triples': int(np.load(target / 'triples.npy', mmap_mode='r').shape[0]), 'fallback_genes': 0, 'links': links}
    else:
        triples, audit, vocab = derive_one(root, fold, modality)
        reference = read(root / 'qualification' / ('fold0-%s.json' % modality))
        check(audit['fallback_edges_sha256'] == reference['fallback_edges_sha256'], 'fold %d %s fallback edges differ from fold0 qualification' % (fold, modality))
        np.save(target / 'triples.npy', triples)
        for name in ('candidates.txt', 'vocabulary.json', 'union-audit.json'):
            links[name] = link_or_copy(graph_dir / name, target / name)
        write_json(target / 'modality.json', audit)
        files = {name: sha(target / name) for name in GRAPH_FILES + ('modality.json',)}
        write_json(target / 'COMPLETE.json', {'files': files, 'fold': fold, 'modality': modality, 'format': FORMAT, 'base_graph_complete_sha256': graph_complete,
                                             'builder_sha256': sha(pathlib.Path(__file__).resolve()), 'modalities_sha256': sha(pathlib.Path(mod.__file__).resolve())})
        record = {'fold': fold, 'modality': modality, 'graph_dir': str(target), 'graph_complete_sha256': sha(target / 'COMPLETE.json'), 'base_graph_complete_sha256': graph_complete,
                  'triples': audit['triples'], 'fallback_genes': audit['fallback_genes'], 'fallback_edges_sha256': audit['fallback_edges_sha256'], 'links': links}
    for part, folder in (('bank', 'banks'), ('test', 'tests')):
        source, digest = dirs[part]
        destination = root / folder / ('fold%d' % fold)
        if not destination.exists():
            verify_complete(source, digest)
            destination.mkdir(parents=True)
            for name in read(source / 'COMPLETE.json')['files']:
                link_or_copy(source / name, destination / name)
            link_or_copy(source / 'COMPLETE.json', destination / 'COMPLETE.json')
        check(sha(destination / 'COMPLETE.json') == digest, '%s link for fold %d differs' % (part, fold))
        record[part + '_dir'] = str(destination)
        record[part + '_complete_sha256'] = digest
    graph_vocab = read(target / 'vocabulary.json')
    record['cohort'] = 'main'
    record['pretraining'] = {'training_triples': record['triples'], 'entity_count': len(graph_vocab['entities']), 'relation_count': len(graph_vocab['relations']),
                             'steps_per_epoch_at_batch_1000': (record['triples'] + 999) // 1000,
                             'budget_rule': 'fixed optimizer-update budget across modalities, not equal epochs'}
    write_json(entry, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'links'}))


def cmd_plan(args):
    """Counts, reuse and storage estimate from the fold0 PFS graph and qualification records; no graph is written."""
    root = pathlib.Path(args.root)
    root_record(root)
    graph_dir = repo_path(EXISTING_BASES[0]['graph'][0])
    base = np.load(graph_dir / 'triples.npy', mmap_mode='r')
    rows = []
    for fold in FOLDS:
        for modality in MODALITIES:
            existing = fold in EXISTING_BASES
            if modality == 'PFS':
                new_bytes = 0
                status = 'adopt existing base' if existing else 'built with the fold base'
            else:
                q = root / 'qualification' / ('fold0-%s.json' % modality)
                new_bytes = read(q)['triples_npy_bytes_estimate'] if q.is_file() else None
                status = 'derive'
            rows.append({'fold': fold, 'modality': modality, 'status': status, 'new_triples_npy_bytes_estimate': new_bytes})
    known = [r['new_triples_npy_bytes_estimate'] for r in rows if r['new_triples_npy_bytes_estimate']]
    base_bytes = int(sum((graph_dir / n).stat().st_size for n in GRAPH_FILES))
    bank_bytes = int((repo_path(EXISTING_BASES[0]['bank'][0]) / 'bank.json').stat().st_size)
    plan = {'graphs': rows, 'fold0_base_triples': int(base.shape[0]),
            'new_fold_bases': [k for k in FOLDS if k not in EXISTING_BASES],
            'storage_estimate_bytes': {'derived_modality_triples': int(sum(known)) if len(known) == 60 else None,
                                       'new_fold_bases_graph_bank_test': len([k for k in FOLDS if k not in EXISTING_BASES]) * (base_bytes + bank_bytes),
                                       'prepare_work_per_new_fold_graph_npz': int(base.nbytes),
                                       'hardlinked_not_counted': 'vocabulary.json, candidates.txt, union-audit.json, banks, tests and PFS graphs'},
            'pretraining_configurations': {'distinct_graphs': len(FOLDS) * len(MODALITIES), 'note': 'graph identities, not trajectories; seeds multiply trajectories, not graphs'}}
    write_json(pathlib.Path(args.output), plan)
    print(json.dumps(plan['storage_estimate_bytes']))


def cmd_manifest(args):
    root = pathlib.Path(args.root)
    record = root_record(root)
    entries = {p.stem: read(p) for p in sorted((root / 'entries').glob('fold*-*.json'))}
    bases = {p.parent.name: read(p) for p in sorted((root / 'bases').glob('fold*/BASE.json'))}
    checks = {p.stem: read(p)['ontology_iri_digest'] for p in sorted((root / 'qualification').glob('base-fold*.json'))}
    check(len(set(checks.values())) <= 1, 'ontology IRI digests differ across verified folds')
    expected = {'fold%d-%s' % (k, m) for k in FOLDS for m in MODALITIES}
    value = {'format': FORMAT, 'root_sha256': sha(root / 'ROOT.json'), 'fallback_order': record['fallback_order'], 'bases': bases, 'entries': entries,
             'verified_folds': sorted(checks), 'ontology_iri_digest': next(iter(checks.values()), None),
             'missing_entries': sorted(expected - set(entries)), 'complete': expected == set(entries)}
    output = pathlib.Path(args.output)
    write_json(output, value)
    print(json.dumps({'manifest': str(output), 'sha256': sha(output), 'entries': len(entries), 'missing': len(value['missing_entries'])}))


def cmd_all(args):
    """Resumable full preparation: fold bases, base verification, qualification and every fold/modality graph."""
    root = pathlib.Path(args.root)
    root_record(root)
    for fold in FOLDS:
        if not (root / 'bases' / ('fold%d' % fold) / 'BASE.json').is_file():
            cmd_base(types.SimpleNamespace(root=str(root), fold=fold))
        if fold == 0 and not (root / 'qualification' / 'QUALIFICATION.json').is_file():
            cmd_qualify(types.SimpleNamespace(root=str(root)))
        if not (root / 'qualification' / ('base-fold%d.json' % fold)).is_file():
            cmd_verify_base(types.SimpleNamespace(root=str(root), fold=fold))
    for fold in FOLDS:
        for modality in MODALITIES:
            if not (root / 'entries' / ('fold%d-%s.json' % (fold, modality))).is_file():
                started = time.time()
                cmd_modality(types.SimpleNamespace(root=str(root), fold=fold, modality=modality))
                print(json.dumps({'fold': fold, 'modality': modality, 'elapsed_seconds': time.time() - started}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('all', help='resumable: every base, verification, qualification and fold/modality graph')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_all)
    p = sub.add_parser('init', help='create a write-once root and pin the fallback order source')
    p.add_argument('--root', required=True)
    p.add_argument('--order-policy', required=True, choices=mod.ORDER_POLICIES)
    p.add_argument('--order-source')
    p.add_argument('--order-sha256')
    p.set_defaults(func=cmd_init)
    p = sub.add_parser('base', help='adopt (folds 0-2) or build (folds 3-9) one fold base')
    p.add_argument('--root', required=True)
    p.add_argument('--fold', type=int, required=True, choices=FOLDS)
    p.set_defaults(func=cmd_base)
    p = sub.add_parser('verify-base', help='independent integrity checks of one fold base')
    p.add_argument('--root', required=True)
    p.add_argument('--fold', type=int, required=True, choices=FOLDS)
    p.set_defaults(func=cmd_verify_base)
    p = sub.add_parser('qualify', help='fold0 metadata for every derived modality plus the cross-fold fallback TEST audit')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_qualify)
    p = sub.add_parser('modality', help='write one fold/modality ComplEx input directory')
    p.add_argument('--root', required=True)
    p.add_argument('--fold', type=int, required=True, choices=FOLDS)
    p.add_argument('--modality', required=True, choices=MODALITIES)
    p.set_defaults(func=cmd_modality)
    p = sub.add_parser('plan', help='counts, reuse and storage estimate')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser('manifest', help='aggregate write-once entries into one manifest')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_manifest)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'command': args.command, 'reason': str(error)}), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
