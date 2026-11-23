"""Frozen no-MGI (excluded) cohort: export, graph, bank, TEST cases and F/S/FS modality graphs.

Fixed inputs (hash-pinned, never resampled):
  train.csv         archived outer TRAIN of the excluded benchmark (the 6,571 main-benchmark associations)
  test.csv          425 excluded TEST associations (417 diseases, 364 genes)
  baseline.metadata.json  reference_diseases: the 5,454 archived TRAIN diseases of the seed-0 split
  disease_phenotypes.csv  disease HP profiles; only phenotypes represented in the shared vocabulary are used
  pool-primary.txt, pool-secondary.txt  5,696 GO genes without MGI phenotypes; all 18,904 GO genes
  gene_phenotypes.csv     prepared MGI table, the pinned first-eligible fallback order

TRAIN rows are the train.csv rows whose disease is a reference disease. Every other train.csv row is validation
and stays held out: its diseases never enter the graph. Checkpoint selection uses the validation rows whose gene is
in the full GO pool and has no TRAIN association (expected 333 pairs, 324 diseases), ranked against the full GO pool.

The graph reuses the shared preprojected background of the main fold0 graph (ontology triples and native gene
annotations, unchanged) and replaces only the clinical edges with the exact excluded TRAIN associations and TRAIN
profiles. The vocabulary keeps every non-disease name of the main vocabulary with the same ordering rule (sorted
prepared names, then the 49 appended union classes) and swaps in the excluded TRAIN diseases. The ontology bank is
the audited fold0 axiom13 bank remapped through IRIs. Modality arms F, S and FS remove native annotations as in the
main suite; main MGI genes without active support keep their first eligible phenotype; TEST genes have no MGI
phenotype and receive none.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import modalities as mod
from common import (CLASS_POOL, CLASS_POOL_SHA256, DISEASE_PREFIX, EXISTING_BASES, FOLD_BUILDER, FOLD_BUILDER_SHA256,
                    HP_PREFIX, NATIVE_RELATION, Refusal, check, link_or_copy, pair_rows, read, repo_path, sha, sha_bytes,
                    verify_complete, write_json)

SOURCE_DIR = 'data/clmpt-suite-20261123/immutable-sources/excluded-reference'
SOURCES = {'train.csv': '7c7d82ebd509eedb0ac13085df5947286a5ee1784ef256946b547c0cb42654c8',
           'test.csv': '69845550d7259249e531feafea79320a74ba0c3dd651158467b4578348cb906e',
           'disease_phenotypes.csv': '869eb72ffb93a48b13ac2fd891c274f4a782dcda3a4d951b733283cfaa86ef99',
           'pool-primary.txt': 'a53df2be63322261d781fb147db3e0491cb09b6c8d61a2683e3e7aaed4f979fd',
           'pool-secondary.txt': 'a4ba8f57ecf023493dda69a106ef91751cd48f7e1858baa916149afab55ec1b8',
           'baseline.metadata.json': 'c30502aaafcd3410423d91e4da54ec8f086b7cd25d00839e2769a6fc403c151f'}
MGI_TABLE = 'data/clmpt-suite-20261123/immutable-sources/gene_phenotypes.csv'
MGI_SHA256 = '94829ab611a3f02c4c33d3686028d8eded187c256f852d4e2cecb2f548b4d0b6'
EXPECTED = {'train_rows': 6571, 'reference_diseases': 5454, 'test_rows': 425, 'test_diseases': 417, 'test_genes': 364,
            'primary_pool': 5696, 'secondary_pool': 18904, 'selection_pairs': 333, 'selection_diseases': 324}
ARMS = ('F', 'S', 'FS')
SHUFFLE_SEED = 20261123
NEGATIVES = 128
FORMAT = 'clmpt-suite-excluded-v1'
TRUTH_SCOPE = 'Empirical observed TRAIN same-input gene union, not complete logical or clinical truth.'


def load_fold_builder():
    path = repo_path(FOLD_BUILDER)
    check(sha(path) == FOLD_BUILDER_SHA256, 'fold builder hash mismatch')
    spec = importlib.util.spec_from_file_location('threefold_build_fold_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_profiles(path):
    profiles = collections.defaultdict(list)
    with pathlib.Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == ['Disease', 'Phenotype'], 'unexpected disease_phenotypes header %s' % reader.fieldnames)
        for row in reader:
            profiles[row['Disease']].append(row['Phenotype'])
    return dict(profiles)


def historical_split(rows, seed=0, ratio=0.1):
    """Re-derivation of link_gda.data.create_train_val_split for comparison only; never used to assign rows."""
    order = list(dict.fromkeys(d for _, d in rows))
    np.random.seed(seed)
    np.random.shuffle(order)
    held = int(len(order) * ratio)
    return set(order[:len(order) - held])


def cohort(source):
    """Fixed partitions, selector and pools from the pinned sources, with identity checks."""
    source = pathlib.Path(source)
    for name, digest in SOURCES.items():
        check(sha(source / name) == digest, 'source %s hash differs from the pinned value' % name)
    train = pair_rows(source / 'train.csv')
    test = pair_rows(source / 'test.csv')
    metadata = read(source / 'baseline.metadata.json')
    reference = metadata['reference_diseases']
    check(len(reference) == len(set(reference)) == metadata['unique_diseases'], 'reference diseases are not unique')
    reference = set(reference)
    primary = (source / 'pool-primary.txt').read_text().split()
    secondary = (source / 'pool-secondary.txt').read_text().split()
    train_d = {d for _, d in train}
    check(reference <= train_d, 'a reference disease is absent from train.csv')
    training = [row for row in train if row[1] in reference]
    validation = [row for row in train if row[1] not in reference]
    training_genes = {g for g, _ in training}
    secondary_set, primary_set = set(secondary), set(primary)
    selection = [row for row in validation if row[0] in secondary_set and row[0] not in training_genes]
    test_d, test_g = {d for _, d in test}, {g for g, _ in test}
    counts = {'train_rows': len(train), 'reference_diseases': len(reference), 'test_rows': len(test), 'test_diseases': len(test_d),
              'test_genes': len(test_g), 'primary_pool': len(primary_set), 'secondary_pool': len(secondary_set),
              'selection_pairs': len(selection), 'selection_diseases': len({d for _, d in selection})}
    check(counts == EXPECTED, 'cohort counts %s differ from expected %s' % (counts, EXPECTED))
    check(len(primary) == len(primary_set) and len(secondary) == len(secondary_set), 'duplicate pool genes')
    check(test_g <= primary_set <= secondary_set, 'TEST genes not inside primary inside secondary pool')
    check(not (test_d & train_d), 'TEST disease in train.csv')
    main_rows = set()
    for fold in range(10):
        base = repo_path('data/reproduction-20261102/collections/inputs-20261028-v1/benchmark/data/fold%d' % fold)
        main_rows.update(pair_rows(base / 'test.tsv'))
    row_ids = {row: 'train.csv:%d' % (index + 2) for index, row in enumerate(train)}
    row_ids.update({row: 'test.csv:%d' % (index + 2) for index, row in enumerate(test)})
    return {'train': train, 'training': training, 'validation': validation, 'selection': selection, 'test': test, 'row_ids': row_ids,
            'primary': primary, 'secondary': secondary, 'reference': reference,
            'audit': {'counts': counts, 'training_rows': len(training), 'training_diseases': len({d for _, d in training}),
                      'validation_rows': len(validation), 'validation_diseases': len({d for _, d in validation}),
                      'validation_rows_held_out_not_selected': len(validation) - len(selection),
                      'train_csv_equals_main_outer_test_union': set(train) == main_rows,
                      'historical_split_seed0_reproduces_reference': historical_split(train) == reference,
                      'training_genes': len(training_genes), 'training_genes_outside_secondary_pool': len(training_genes - secondary_set),
                      'training_rows_with_gene_outside_secondary_pool': sum(g not in secondary_set for g, _ in training),
                      'test_genes_in_training': len(test_g & training_genes)}}


def build_vocabulary(main_vocab, appended, training_diseases):
    names = set(main_vocab['entities'])
    check(set(appended) <= names, 'appended union classes absent from main vocabulary')
    base = {n for n in names if not n.startswith(DISEASE_PREFIX)} - set(appended)
    check(not (base & training_diseases), 'excluded TRAIN disease collides with a non-disease name')
    ordered = sorted(base | training_diseases) + sorted(appended, key=lambda n: main_vocab['entities'][n])
    return {'entities': {n: i for i, n in enumerate(ordered)}, 'relations': dict(main_vocab['relations'])}


def build_graph(data, main_triples, main_vocab, vocab, profiles):
    ids = mod.relation_ids(main_vocab)
    clinical = [i for name in ('associated_with', 'has_symptom') for i in ids[name]]
    native = np.asarray(main_triples[~np.isin(main_triples[:, 1], clinical)])
    old_to_new = np.full(len(main_vocab['entities']), -1, dtype=np.int64)
    for name, value in main_vocab['entities'].items():
        if name in vocab['entities']:
            old_to_new[value] = vocab['entities'][name]
    mapped = np.stack([old_to_new[native[:, 0]], native[:, 1], old_to_new[native[:, 2]]], axis=1)
    check((mapped >= 0).all(), 'a native triple references a disease')
    entities = vocab['entities']
    a, ai = ids['associated_with']
    h, hi = ids['has_symptom']
    rows = []
    for gene, disease in data['training']:
        rows.append((entities[gene], a, entities[disease]))
        rows.append((entities[disease], ai, entities[gene]))
    for disease in sorted({d for _, d in data['training']}):
        for phenotype in profiles[disease]:
            rows.append((entities[disease], h, entities[phenotype]))
            rows.append((entities[phenotype], hi, entities[disease]))
    clinical_rows = np.unique(np.array(rows, dtype=np.int64), axis=0)
    triples = np.unique(np.concatenate([mapped, clinical_rows]), axis=0)
    check(len(triples) == len(mapped) + len(clinical_rows), 'clinical rows collide with native rows')
    return np.ascontiguousarray(triples), {'native_triples': int(len(mapped)), 'clinical_triples': int(len(clinical_rows)), 'triples': int(len(triples)),
                                           'order': 'unique rows sorted lexicographically by head, relation, tail'}


def profile_groups(data, vocab, profiles, training_gene_ids):
    entities = vocab['entities']
    relations = vocab['relations']
    inv_symptom, inv_assoc = relations['inverse:has_symptom'], relations['inverse:associated_with']
    genes_of, rows_of = collections.defaultdict(set), collections.defaultdict(list)
    for index, (gene, disease) in enumerate(data['train']):
        if disease in data['reference']:
            genes_of[disease].add(entities[gene])
            rows_of[disease].append('train.csv:%d' % (index + 2))
    grouped = collections.defaultdict(list)
    for disease in sorted(genes_of):
        hp = tuple(sorted({entities[p] for p in profiles[disease]}))
        check(hp, 'TRAIN disease %s has no represented phenotype' % disease)
        grouped[hp].append(disease)
    hp_genes = collections.defaultdict(set)
    for disease, genes in genes_of.items():
        for p in profiles[disease]:
            hp_genes[entities[p]].update(genes)
    universe = set(training_gene_ids)
    out = []
    for index, (hp, diseases) in enumerate(sorted(grouped.items())):
        genes = sorted(set().union(*(genes_of[d] for d in diseases)))
        check(set(genes) <= universe, 'TRAIN answer outside training_gene_ids')
        known = set(genes)
        for value in hp:
            known.update(hp_genes[value])
        known = sorted(known)
        check(len(universe - set(known)) >= NEGATIVES, 'fewer than 128 typed negatives for a TRAIN profile')
        out.append({'type': 'profile', 'full_train_index': index, 'hp_ids': list(hp), 'constants': {'s' + str(i): v for i, v in enumerate(hp)},
                    'atoms': [['s' + str(i), inv_symptom, 'e1'] for i in range(len(hp))] + [['e1', inv_assoc, 'f']], 'answers': genes,
                    'negative_known': known, 'negative_known_sha256': hashlib.sha256(np.asarray(known, dtype='<i8').tobytes()).hexdigest(),
                    'domain': 'gene', 'disease_ids': [entities[d] for d in diseases], 'disease_names': diseases, 'truth_scope': TRUTH_SCOPE,
                    'source_case_ids': sorted(r for d in diseases for r in rows_of[d])})
    return out


def training_gene_domain(training_gene_ids, candidate_ids):
    """Declared clinical TRAIN gene universe, in the identity form computed by the query sampler."""
    ordered = sorted(int(v) for v in training_gene_ids)
    candidates = set(int(v) for v in candidate_ids)
    return {'source': 'training_gene_ids', 'count': len(ordered), 'sha256': sha_bytes(json.dumps(ordered, separators=(',', ':')).encode()),
            'outside_candidate_pool': len(set(ordered) - candidates), 'candidates_outside_universe': len(candidates - set(ordered))}


def cases(rows, vocab, profiles, row_ids, s_supported=None):
    entities = vocab['entities']
    genes, sources = collections.defaultdict(set), collections.defaultdict(list)
    for gene, disease in rows:
        genes[disease].add(entities[gene])
        sources[disease].append(row_ids[(gene, disease)])
    out = []
    for disease in sorted(genes):
        hp = sorted({entities[p] for p in profiles[disease]})
        check(hp, 'held-out disease %s has no represented phenotype' % disease)
        case = {'disease_name': disease, 'hp_ids': hp, 'gene_ids': sorted(genes[disease]), 'source_rows': sources[disease]}
        if s_supported is not None:
            case['s_supported_gene_ids'] = sorted(g for g in genes[disease] if g in s_supported)
        out.append(case)
    return out


def cmd_build(args):
    import prepare_suite
    root = pathlib.Path(args.root)
    prepare_suite.root_record(root)
    check((root / 'qualification' / 'fold0-FS.json').is_file(), 'run the main qualification first')
    out = root / 'excluded'
    check(not out.exists(), 'excluded directory exists; use a fresh root or inspect by hand')
    started = time.time()
    data = cohort(repo_path(SOURCE_DIR))
    check(sha(repo_path(MGI_TABLE)) == MGI_SHA256, 'MGI table hash mismatch')
    graph_path, graph_complete = EXISTING_BASES[0]['graph']
    bank_path, bank_complete = EXISTING_BASES[0]['bank']
    main_graph, main_bank_dir = repo_path(graph_path), repo_path(bank_path)
    verify_complete(main_graph, graph_complete)
    verify_complete(main_bank_dir, bank_complete)
    main_vocab = read(main_graph / 'vocabulary.json')
    appended = list(read(main_graph / 'union-audit.json')['appended_entities'])
    main_triples = np.load(main_graph / 'triples.npy')
    raw_profiles = read_profiles(repo_path(SOURCE_DIR) / 'disease_phenotypes.csv')
    represented = {n for n in main_vocab['entities'] if n.startswith(HP_PREFIX)}
    split = {d for _, d in data['train']} | {d for _, d in data['test']}
    check(split <= raw_profiles.keys(), '%d split diseases lack profiles' % len(split - raw_profiles.keys()))
    profiles = {d: list(dict.fromkeys(p for p in raw_profiles[d] if p in represented)) for d in split}
    dropped = sum(len(raw_profiles[d]) - len([p for p in raw_profiles[d] if p in represented]) for d in split)
    main_profiles = read(repo_path('data/reproduction-20261102/collections/inputs-20261028-v1/benchmark/data/fold0/profiles.json'))
    shared = [d for d in split if d in main_profiles]
    profile_mismatch = [d for d in shared if set(main_profiles[d]) != set(profiles[d])]
    check(not profile_mismatch, '%d diseases have profiles differing from the main benchmark, e.g. %s' % (len(profile_mismatch), profile_mismatch[:3]))
    training_diseases = {d for _, d in data['training']}
    vocab = build_vocabulary(main_vocab, appended, training_diseases)
    entities = vocab['entities']
    for label, names in (('secondary pool', data['secondary']), ('TRAIN genes', {g for g, _ in data['training']}),
                         ('validation genes', {g for g, _ in data['validation']}), ('TEST genes', {g for g, _ in data['test']})):
        missing = set(names) - entities.keys()
        check(not missing, '%d %s absent from the shared vocabulary, e.g. %s' % (len(missing), label, sorted(missing)[:3]))
    heldout = {d for _, d in data['validation']} | {d for _, d in data['test']}
    check(not (heldout & entities.keys()), 'held-out disease in excluded vocabulary')
    triples, graph_counts = build_graph(data, main_triples, main_vocab, vocab, profiles)
    support, _ = mod.native_support(triples, vocab)
    names = {v: n for n, v in entities.items()}
    supported = {k: {names[i] for i in v} for k, v in support.items()}
    test_genes = {g for g, _ in data['test']}
    check(not (test_genes & supported['P']), 'an excluded TEST gene has a native phenotype edge')
    check(test_genes <= supported['F'], 'an excluded TEST gene lacks GO support')
    check(set(data['secondary']) <= supported['F'], 'a full-GO pool gene lacks a has_function edge')
    s_supported = {entities[g] for g in supported['S']}

    builder = load_fold_builder()
    bank0 = read(main_bank_dir / 'bank.json')
    check(bank0['relations'] == {'associated_with': vocab['relations']['associated_with'], 'has_symptom': vocab['relations']['has_symptom'],
                                 'inverse_associated_with': vocab['relations']['inverse:associated_with'], 'inverse_has_symptom': vocab['relations']['inverse:has_symptom']},
          'clinical relation ids differ from the fold0 bank')
    try:
        ontology, class_ids = builder.remap_ontology(bank0, main_vocab, vocab, read(repo_path(CLASS_POOL)))
    except builder.Refusal as error:
        raise Refusal('ontology remap refused: %s' % error)
    check(sha(repo_path(CLASS_POOL)) == CLASS_POOL_SHA256, 'class pool hash mismatch')
    candidate_ids = [entities[g] for g in data['secondary']]
    primary_ids = [entities[g] for g in data['primary']]
    training_gene_ids = sorted(set(candidate_ids) | {entities[g] for g, _ in data['training']})
    profiles_bank = profile_groups(data, vocab, profiles, training_gene_ids)
    validation = cases(data['selection'], vocab, profiles, data['row_ids'])
    shuffle, attempts = builder.derangement(validation, SHUFFLE_SEED)
    hp_ids = sorted(v for n, v in entities.items() if n.startswith(HP_PREFIX))
    check(all(set(c['hp_ids']) <= set(hp_ids) for c in validation), 'validation profile outside hp_ids')
    test_cases = cases(data['test'], vocab, profiles, data['row_ids'], s_supported)

    out.mkdir()
    export = out / 'cohort'
    export.mkdir()
    for name, rows in (('training.tsv', data['training']), ('validation-heldout.tsv', data['validation']),
                       ('selection-validation.tsv', data['selection']), ('test.tsv', data['test'])):
        (export / name).write_text('Gene\tDisease\n' + ''.join('%s\t%s\n' % row for row in rows))
    write_json(export / 'profiles.json', {d: profiles[d] for d in sorted(profiles)})
    for name in ('pool-primary.txt', 'pool-secondary.txt'):
        link_or_copy(repo_path(SOURCE_DIR) / name, export / name)
    audit = {**data['audit'], 'profile_phenotypes_dropped_unrepresented': dropped, 'profiles_matching_main_benchmark': len(shared),
             'sources': {name: {'path': SOURCE_DIR + '/' + name, 'sha256': digest} for name, digest in SOURCES.items()},
             'rule': 'TRAIN = train.csv rows of the 5,454 archived reference diseases; every other row is held-out validation; selection = validation rows with a full-GO gene lacking any TRAIN association'}
    write_json(export / 'COHORT-AUDIT.json', audit)
    files = ('training.tsv', 'validation-heldout.tsv', 'selection-validation.tsv', 'test.tsv', 'profiles.json', 'pool-primary.txt', 'pool-secondary.txt', 'COHORT-AUDIT.json')
    write_json(export / 'COMPLETE.json', {'files': {n: sha(export / n) for n in files}, 'format': FORMAT, 'builder_sha256': sha(pathlib.Path(__file__).resolve())})

    base = out / 'base'
    base.mkdir()
    np.save(base / 'triples.npy', triples)
    write_json(base / 'vocabulary.json', vocab)
    (base / 'candidates.txt').write_text(''.join(g + '\n' for g in data['secondary']))
    write_json(base / 'union-audit.json', {'entities': len(entities), 'relations': len(vocab['relations']), 'appended_entities': {n: entities[n] for n in appended},
                                           'rule': 'main non-disease vocabulary unchanged in content and ordering rule; excluded TRAIN diseases replace main TRAIN diseases',
                                           'main_graph_complete_sha256': graph_complete, **graph_counts})
    write_json(base / 'COMPLETE.json', {'files': {n: sha(base / n) for n in ('triples.npy', 'vocabulary.json', 'candidates.txt', 'union-audit.json')},
                                        'format': FORMAT, 'role': 'PFS derivation base for the excluded F/S/FS arms; not itself an arm',
                                        'cohort_complete_sha256': sha(export / 'COMPLETE.json'), 'builder_sha256': sha(pathlib.Path(__file__).resolve())})
    base_complete = sha(base / 'COMPLETE.json')

    bank = {key: value for key, value in bank0.items()}
    bank.update({'fold': 'excluded', 'num_entities': len(entities), 'num_relations': len(vocab['relations']),
                 'ontology_queries': ontology['ontology_queries'], 'additional_ontology_queries': ontology['additional_ontology_queries'],
                 'class_ids': class_ids, 'candidate_ids': candidate_ids, 'training_gene_ids': training_gene_ids, 'hp_ids': hp_ids,
                 'relations': bank0['relations'], 'training_profiles': profiles_bank, 'validation': validation, 'validation_shuffle': shuffle,
                 'probe_queries': [],
                 'clinical_counts': {'validation_cases': len(validation), 'validation_associations': sum(len(c['gene_ids']) for c in validation)},
                 'axiom_bank': {**bank0['axiom_bank'], 'excluded_remap': {'source_bank_complete_sha256': bank_complete, 'graph_complete_sha256': base_complete,
                                'rule': 'fold0 axiom-only bank ids mapped through IRIs into the excluded vocabulary',
                                'candidate_ids': 'full-GO pool (18,904) for validation and TEST ranking',
                                'training_gene_ids': 'typed clinical answer and negative universe: full-GO pool plus every TRAIN answer gene; a sampler that ignores this key reproduces the main-bank behaviour only when every TRAIN gene is a candidate',
                                'validation_shuffle': 'seeded profile-changing derangement (seed %d), descriptive control only' % SHUFFLE_SEED}}})
    bank_dir = out / 'bank'
    bank_dir.mkdir()
    write_json(bank_dir / 'bank.json', bank, compact=True)
    write_json(bank_dir / 'audit.json', {'training_profiles': len(profiles_bank), 'validation_cases': len(validation),
                                         'validation_associations': bank['clinical_counts']['validation_associations'], 'shuffle_attempts': attempts,
                                         'candidate_ids': len(candidate_ids), 'training_gene_ids': len(training_gene_ids),
                                         'training_genes_outside_candidates': len(set(training_gene_ids) - set(candidate_ids)),
                                         'ontology_groups': len(ontology['ontology_queries']) + len(ontology['additional_ontology_queries']),
                                         'class_ids': len(class_ids), 'graph_complete_sha256': base_complete})
    write_json(bank_dir / 'COMPLETE.json', {'files': {n: sha(bank_dir / n) for n in ('bank.json', 'audit.json')}, 'format': bank0['axiom_bank']['format'],
                                            'fold': 'excluded', 'training_gene_domain': training_gene_domain(training_gene_ids, candidate_ids), 'graph_complete_sha256': base_complete, 'builder_sha256': sha(pathlib.Path(__file__).resolve())})

    test_dir = out / 'test'
    test_dir.mkdir()
    write_json(test_dir / 'test-cases.json', {'fold': 'excluded', 'cases': test_cases, 'candidate_ids': candidate_ids, 'candidate_views': {'secondary_full_go': candidate_ids, 'primary_no_mgi': primary_ids},
                                              'relations': bank0['relations'], 'graph_complete_sha256': base_complete, 'bank_complete_sha256': sha(bank_dir / 'COMPLETE.json'),
                                              's_rule': 's_supported_gene_ids marks targets with a native expressed_in edge; candidate pools are unchanged',
                                              'scope': 'TEST, read only after validation selection'})
    s_rows = sum(len(c['s_supported_gene_ids']) for c in test_cases)
    write_json(test_dir / 'COMPLETE.json', {'files': {'test-cases.json': sha(test_dir / 'test-cases.json')}, 'test_cases': len(test_cases),
                                            'test_associations': sum(len(c['gene_ids']) for c in test_cases), 's_supported_associations': s_rows,
                                            's_supported_cases': sum(bool(c['s_supported_gene_ids']) for c in test_cases),
                                            'bank_complete_sha256': sha(bank_dir / 'COMPLETE.json'), 'builder_sha256': sha(pathlib.Path(__file__).resolve())})
    (out / 'entries').mkdir()
    summary = {'cohort_complete_sha256': sha(export / 'COMPLETE.json'), 'base_graph_complete_sha256': base_complete,
               'bank_complete_sha256': sha(bank_dir / 'COMPLETE.json'), 'test_complete_sha256': sha(test_dir / 'COMPLETE.json'),
               'cohort': audit, 'graph': graph_counts, 'entities': len(entities), 'training_profiles': len(profiles_bank),
               'validation_cases': len(validation), 'training_gene_ids': len(training_gene_ids), 'test_s_supported_associations': s_rows,
               'elapsed_seconds': time.time() - started}
    write_json(out / 'BUILD.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cohort'}))


def cmd_modality(args):
    root = pathlib.Path(args.root)
    out = root / 'excluded'
    check((out / 'BUILD.json').is_file(), 'run build first')
    modality = args.modality
    entry = out / 'entries' / ('excluded-%s.json' % modality)
    target = out / 'graphs' / modality
    check(not entry.exists() and not target.exists(), 'excluded %s already prepared' % modality)
    build = read(out / 'BUILD.json')
    base = out / 'base'
    verify_complete(base, build['base_graph_complete_sha256'])
    check(read(base / 'COMPLETE.json')['builder_sha256'] == sha(pathlib.Path(__file__).resolve()), 'excluded base was built by a different source; use a fresh root')
    check((root / 'qualification' / ('fold0-%s.json' % modality)).is_file(), 'main qualification for %s missing' % modality)
    vocab = read(base / 'vocabulary.json')
    triples = np.load(base / 'triples.npy')
    support, phenotype = mod.native_support(triples, vocab)
    genes = mod.fallback_genes(support, modality)
    order, provenance = mod.load_order('prepared-csv', repo_path(MGI_TABLE), MGI_SHA256)
    chosen = mod.choose_fallback(genes, phenotype, vocab, order)
    main_candidates = repo_path(EXISTING_BASES[0]['graph'][0]) / 'candidates.txt'
    derived, audit = mod.derive(triples, vocab, main_candidates.read_text().splitlines(), modality, chosen)
    reference = root / 'qualification' / ('fold0-%s.json' % modality)
    check(audit['fallback_edges_sha256'] == read(reference)['fallback_edges_sha256'], 'excluded %s fallback edges differ from main fold0' % modality)
    pool = np.array([vocab['entities'][g] for g in (base / 'candidates.txt').read_text().split()], dtype=np.int64)
    degree = np.bincount(derived[:, 0], minlength=len(vocab['entities']))
    test = read(out / 'test' / 'test-cases.json')
    test_genes = {g for c in test['cases'] for g in c['gene_ids']}
    audit.update({'cohort': 'excluded', 'base_graph_complete_sha256': build['base_graph_complete_sha256'], 'fallback_order': provenance,
                  'candidate_semantics': 'isolated_candidates and fallback checks refer to the 13,625 main MGI genes; ranking uses the full-GO pool',
                  'full_go_pool_isolated': int((degree[pool] == 0).sum()), 'test_genes_isolated': int(sum(degree[g] == 0 for g in test_genes)),
                  'test_genes_with_fallback': len(test_genes & set(chosen))})
    check(audit['test_genes_with_fallback'] == 0, 'an excluded TEST gene received a fallback edge')
    target.mkdir(parents=True)
    np.save(target / 'triples.npy', derived)
    for name in ('candidates.txt', 'vocabulary.json', 'union-audit.json'):
        link_or_copy(base / name, target / name)
    write_json(target / 'modality.json', audit)
    files = ('triples.npy', 'candidates.txt', 'vocabulary.json', 'union-audit.json', 'modality.json')
    write_json(target / 'COMPLETE.json', {'files': {n: sha(target / n) for n in files}, 'cohort': 'excluded', 'modality': modality, 'format': FORMAT,
                                          'base_graph_complete_sha256': build['base_graph_complete_sha256'], 'builder_sha256': sha(pathlib.Path(__file__).resolve()),
                                          'modalities_sha256': sha(pathlib.Path(mod.__file__).resolve())})
    record = {'cohort': 'excluded', 'fold': 'excluded', 'modality': modality, 'graph_dir': str(target), 'graph_complete_sha256': sha(target / 'COMPLETE.json'),
              'bank_dir': str(out / 'bank'), 'bank_complete_sha256': build['bank_complete_sha256'], 'test_dir': str(out / 'test'),
              'test_complete_sha256': build['test_complete_sha256'], 'triples': audit['triples'], 'fallback_genes': audit['fallback_genes'],
              'full_go_pool_isolated': audit['full_go_pool_isolated'], 'test_genes_isolated': audit['test_genes_isolated'],
              'pretraining': {'training_triples': audit['triples'], 'entity_count': len(vocab['entities']), 'relation_count': len(vocab['relations']),
                              'steps_per_epoch_at_batch_1000': (audit['triples'] + 999) // 1000}}
    write_json(entry, record)
    print(json.dumps(record))


def cmd_verify(args):
    """Independent re-read of the written excluded artifacts against the exported cohort and the fold0 bank digest."""
    import prepare_suite
    root = pathlib.Path(args.root)
    out = root / 'excluded'
    build = read(out / 'BUILD.json')
    verify_complete(out / 'cohort', build['cohort_complete_sha256'])
    verify_complete(out / 'base', build['base_graph_complete_sha256'])
    verify_complete(out / 'bank', build['bank_complete_sha256'])
    verify_complete(out / 'test', build['test_complete_sha256'])
    vocab = read(out / 'base' / 'vocabulary.json')
    entities = vocab['entities']
    names = {v: n for n, v in entities.items()}
    relation_names = {v: n for n, v in vocab['relations'].items()}
    triples = np.load(out / 'base' / 'triples.npy', mmap_mode='r')
    ids = mod.relation_ids(vocab)
    training = pair_rows(out / 'cohort' / 'training.tsv')
    heldout = pair_rows(out / 'cohort' / 'validation-heldout.tsv')
    selection = pair_rows(out / 'cohort' / 'selection-validation.tsv')
    test = pair_rows(out / 'cohort' / 'test.tsv')
    profiles = read(out / 'cohort' / 'profiles.json')
    train_d = {d for _, d in training}
    association = np.asarray(triples[triples[:, 1] == ids['associated_with'][0]])
    check({(int(h), int(t)) for h, _, t in association.tolist()} == {(entities[g], entities[d]) for g, d in training}, 'associated_with differs from training.tsv')
    symptom = np.asarray(triples[triples[:, 1] == ids['has_symptom'][0]])
    check({(int(h), int(t)) for h, _, t in symptom.tolist()} == {(entities[d], entities[p]) for d in train_d for p in profiles[d]}, 'has_symptom differs from TRAIN profiles')
    check({n for n in entities if n.startswith(DISEASE_PREFIX)} == train_d, 'vocabulary diseases differ from TRAIN diseases')
    check(not (({d for _, d in heldout} | {d for _, d in test}) & entities.keys()), 'held-out disease in vocabulary')
    check(set(selection) <= set(heldout) and not (set(heldout) & set(training)), 'selection or held-out rows overlap TRAIN')
    bank = read(out / 'bank' / 'bank.json')
    disease_ids = {entities[d] for d in train_d}
    for listing in ('ontology_queries', 'additional_ontology_queries'):
        for row in bank[listing]:
            check(not ((set(row['constants'].values()) | set(row['answers']) | set(row['negative_known'])) & disease_ids), 'disease id in ontology supervision')
    for row in bank['training_profiles']:
        check(not (set(row['constants'].values()) & disease_ids) and set(row['disease_names']) <= train_d, 'TRAIN profile constants or diseases invalid')
    expected = collections.defaultdict(set)
    for g, d in selection:
        expected[d].add(entities[g])
    check({c['disease_name']: set(c['gene_ids']) for c in bank['validation']} == dict(expected), 'bank validation differs from selection-validation.tsv')
    check(bank['candidate_ids'] == [entities[g] for g in (out / 'cohort' / 'pool-secondary.txt').read_text().split()], 'candidate order differs from the full-GO pool')
    check(set(bank['candidate_ids']) <= set(bank['training_gene_ids']), 'training_gene_ids omits a candidate')
    tests = read(out / 'test' / 'test-cases.json')
    check(tests['candidate_ids'] == bank['candidate_ids'] == tests['candidate_views']['secondary_full_go'], 'TEST candidate_ids differ from the bank')
    check(tests['bank_complete_sha256'] == build['bank_complete_sha256'], 'TEST cases bind a different bank')
    declared = read(out / 'bank' / 'COMPLETE.json')['training_gene_domain']
    check(declared == training_gene_domain(bank['training_gene_ids'], bank['candidate_ids']) and declared['outside_candidate_pool'] == 20, 'declared training gene domain differs')
    check(all(c['source_rows'] for c in tests['cases'] + bank['validation']), 'case without source rows')
    check(tests['candidate_views']['primary_no_mgi'] == [entities[g] for g in (out / 'cohort' / 'pool-primary.txt').read_text().split()], 'primary view differs')
    expected = collections.defaultdict(set)
    for g, d in test:
        expected[d].add(entities[g])
    check({c['disease_name']: set(c['gene_ids']) for c in tests['cases']} == dict(expected), 'TEST cases differ from test.tsv')
    digest = prepare_suite.ontology_iri_digest(bank, names, relation_names)
    reference = root / 'qualification' / 'base-fold0.json'
    check(reference.is_file() and digest == read(reference)['ontology_iri_digest'], 'excluded ontology bank is not the fold0 bank under IRI remap')
    result = {'verified': True, 'ontology_iri_digest': digest, 'training_rows': len(training), 'heldout_rows': len(heldout), 'selection_rows': len(selection),
              'test_rows': len(test), 'validation_cases': len(bank['validation']), 'test_cases': len(tests['cases'])}
    write_json(out / 'VERIFY.json', result)
    print(json.dumps(result))


def cmd_all(args):
    """Resumable: build, verify and every excluded arm."""
    out = pathlib.Path(args.root) / 'excluded'
    if not (out / 'BUILD.json').is_file():
        cmd_build(args)
    if not (out / 'VERIFY.json').is_file():
        cmd_verify(args)
    for modality in ARMS:
        if not (out / 'entries' / ('excluded-%s.json' % modality)).is_file():
            cmd_modality(argparse.Namespace(root=args.root, modality=modality))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('all', help='resumable: build, verify and the F, S and FS arms')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_all)
    p = sub.add_parser('verify', help='independent re-read checks of the excluded artifacts')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser('build', help='frozen cohort export, excluded base graph, bank and TEST cases')
    p.add_argument('--root', required=True)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser('modality', help='one excluded F, S or FS graph')
    p.add_argument('--root', required=True)
    p.add_argument('--modality', required=True, choices=ARMS)
    p.set_defaults(func=cmd_modality)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'command': args.command, 'reason': str(error)}), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
