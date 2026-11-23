"""Fold-specific TRAIN graph, fold-remapped axiom bank and held-out case files for the three-fold CLMPT pilot.

Chain generalized from the executed fold0 builders:
  code/compositional_gda/prepare.py (0a77cf2f, imported byte-exact)
  orix-query-adapters-20261110 prepare.py export (7bf7a80f, fold0 count asserts replaced by fold-generic checks)
  clmpt-two-stage-search query/code/prepare_union.py (eece4c5b, same appended names and id rule)
  orix-query-pilots clmpt prepare.py / prepare_full.py (hp_ids and TRAIN-profile any-HP negative mask)
  clmpt-two-stage-search query/code/prepare_bank.py (0db36050, IRI clinical cross-checks)
The 13-shape axiom bank is not rebuilt: every class/relation id of the audited fold0 bank (8f34c9d2) is
mapped through its IRI into the fold vocabulary. The non-disease vocabulary, relation table and native
ontology triples must be identical to fold0, otherwise the build refuses.
Fold 0 is a reproduction gate: every derived artifact must equal the executed fold0 artifact, and only
the TEST case file is written.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np

PREPARE_SHA256 = '0a77cf2f819d830c5ad46d86127f1ff38a3759dfcc910d6ec3bf9cf0035329a3'
FOLD0_GRAPH_COMPLETE = 'af68a9a3230fc1f05a105e1d57f02200129879c0bea5097c9094cf0feb0201ed'
FOLD0_BANK_COMPLETE = '8f34c9d29efa42d07353d6b51ee190b24901da9f7ba738368c5d75543c5521ec'
IRI_COMPLETE = '4ebbee88742472a6ffa866a71903305d4434fdbd3e480c709fa5cb2451d0cc43'
CLASS_POOL_SHA256 = 'd6813b738c6baaa77e9595783668f8cf5e4c01f70d833f5d0fb01dbb5ed168f0'
FOLD0_PREPARED_OUTPUTS = {'graph.npz': '4d4ceb5a0abae85e23d09a1c7aee308e6d1fecfd7249fba65544bcc2b034544e',
                          'vocabulary.json': '70da2723b06c3f67c6dee33f5a147bad8e45575bf2e037b2fbb83982ea05e680'}
HP_PREFIX = 'http://purl.obolibrary.org/obo/HP_'
DISEASE_PREFIX = 'http://mowl.borg/OMIM_'
CLINICAL_ROLES = ('associated_with', 'has_symptom')
TRUTH_SCOPE = 'Empirical observed TRAIN same-input gene union, not complete logical or clinical truth.'
SHUFFLE_SEED = 20261116


class Refusal(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise Refusal(message)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def verify_complete(directory, expected=None):
    directory = pathlib.Path(directory)
    if expected is not None:
        check(sha(directory / 'COMPLETE.json') == expected, 'COMPLETE.json hash mismatch in %s' % directory)
    value = read(directory / 'COMPLETE.json')
    for name, record in value['files'].items():
        check(sha(directory / name) == (record['sha256'] if isinstance(record, dict) else record), 'file hash mismatch %s/%s' % (directory, name))
    return value


def write_json(path, value, compact=False):
    path = pathlib.Path(path)
    temporary = path.with_name(path.name + '.tmp')
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) if compact else json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
    with temporary.open('w') as stream:
        stream.write(text + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pairs(path):
    with pathlib.Path(path).open(newline='') as stream:
        rows = [(row['Gene'], row['Disease']) for row in csv.DictReader(stream, delimiter='\t')]
    check(len(rows) == len(set(rows)), 'duplicate rows in %s' % path)
    return rows


def sorted_rows(x):
    return x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]


def load_prepare(source):
    check(sha(source) == PREPARE_SHA256, 'prepare.py is not the executed fold0 source 0a77cf2f')
    spec = importlib.util.spec_from_file_location('compositional_gda_prepare', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_fold(prepare_source, inputs, work, fold):
    module = load_prepare(prepare_source)
    status, target = module.prepare_fold(pathlib.Path(inputs), pathlib.Path(work) / 'prepared', fold)
    identity = read(target / 'identity.json')
    check(identity['source']['sha256'] == PREPARE_SHA256 and identity['fold'] == fold, 'prepared identity mismatch')
    return target, identity, status


def export_fold(prepared):
    """Executed 7bf7a80f export transform with fold0 constants replaced by structural checks."""
    vocabulary = read(prepared / 'vocabulary.json')
    entities, relations = vocabulary['entities'], vocabulary['relations']
    count = len(relations)
    check(count % 2 == 0 and set(relations.values()) == set(range(count)), 'relation ids are not dense')
    half = count // 2
    for name, value in relations.items():
        if value < half:
            check(not name.startswith('inverse:') and relations['inverse:' + name] == value + half, 'forward/inverse layout broken')
        else:
            check(name.startswith('inverse:') and relations[name[8:]] == value - half, 'forward/inverse layout broken')
    old_to_new = np.array([2 * (i % half) + (i // half) for i in range(count)], dtype=np.int64)
    inverse = {i: (i + half) % count for i in range(count)}
    check(np.array_equal(old_to_new[np.array([inverse[i] for i in range(count)])], old_to_new ^ 1), 'relation interleave broken')
    graph = np.load(prepared / 'graph.npz')['triples']
    check(graph.ndim == 2 and graph.shape[1] == 3 and graph.dtype == np.int64, 'graph dtype/shape')
    forward = graph[graph[:, 1] < half]
    backward = graph[graph[:, 1] >= half]
    expected = forward[:, [2, 1, 0]].copy()
    expected[:, 1] += half
    check(np.array_equal(sorted_rows(expected), sorted_rows(backward)), 'inverse mirrors not exact')
    check(all('causes_phenotype' not in name for name in relations), 'materialized causes_phenotype present')
    train, validation, test = pairs(prepared / 'training.tsv'), pairs(prepared / 'validation.tsv'), pairs(prepared / 'test.tsv')
    train_d, val_d, test_d = {d for _, d in train}, {d for _, d in validation}, {d for _, d in test}
    check(not (train_d & val_d) and not (train_d & test_d) and not (val_d & test_d), 'split diseases overlap')
    check(not ((val_d | test_d) & entities.keys()), 'held-out disease in vocabulary')
    check({n for n in entities if n.startswith(DISEASE_PREFIX)} == train_d, 'graph diseases differ from TRAIN diseases')
    candidate_names = (prepared / 'candidates.txt').read_text().splitlines()
    candidates = [entities[n] for n in candidate_names]
    check(len(candidates) == len(set(candidates)), 'duplicate candidates')
    check({(entities[g], entities[d]) for g, d in train} == {(int(h), int(t)) for h, r, t in graph[graph[:, 1] == relations['associated_with']]}, 'associated_with differs from training.tsv')
    profiles = read(prepared / 'profiles.json')
    by_disease = collections.defaultdict(set)
    for gene, disease in train:
        by_disease[disease].add(entities[gene])
    grouped = collections.defaultdict(list)
    for disease in sorted(train_d):
        hp = tuple(sorted({entities[h] for h in profiles[disease]}))
        check(hp and all('HP_' in h for h in set(profiles[disease])), 'TRAIN profile empty or non-HP')
        grouped[hp].append(disease)
    check({(entities[d], entities[h]) for d in train_d for h in set(profiles[d])} == {(int(h), int(t)) for h, r, t in graph[graph[:, 1] == relations['has_symptom']]}, 'has_symptom differs from TRAIN profiles')
    groups = [{'hp_ids': list(hp), 'gene_ids': sorted(set().union(*(by_disease[d] for d in diseases))), 'disease_ids': [entities[d] for d in diseases], 'disease_names': diseases}
              for hp, diseases in sorted(grouped.items())]

    candidate_set = set(candidates)

    def heldout(rows):
        genes = collections.defaultdict(set)
        for gene, disease in rows:
            check(gene in entities and entities[gene] in candidate_set, 'held-out gene not a candidate')
            genes[disease].add(entities[gene])
        cases = [{'disease_name': d, 'hp_ids': sorted({entities[h] for h in profiles[d]}), 'gene_ids': sorted(genes[d])} for d in sorted(genes)]
        check(all(case['hp_ids'] for case in cases), 'held-out case without phenotypes')
        return cases
    converted = graph.copy()
    converted[:, 1] = old_to_new[graph[:, 1]]
    vocab = {'entities': entities, 'relations': {n: int(old_to_new[i]) for n, i in relations.items()}}
    counts = {'triples': int(len(graph)), 'forward_triples': int(len(forward)), 'entities': len(entities), 'relations': count, 'training_diseases': len(train_d),
              'profile_groups': len(groups), 'query_gene_pairs': sum(len(g['gene_ids']) for g in groups), 'validation_diseases': len(val_d),
              'validation_associations': len(validation), 'test_diseases': len(test_d), 'test_associations': len(test)}
    return {'triples': converted, 'vocab': vocab, 'groups': groups, 'validation': heldout(validation), 'test': heldout(test),
            'candidates': candidates, 'candidate_names': candidate_names, 'train_diseases': train_d, 'counts': counts, 'profiles': profiles}


def union(export, coverage, fold0_audit):
    """Executed eece4c5b append rule: sorted missing classes, then sorted missing roles as forward/inverse pairs."""
    vocab = {'entities': dict(export['vocab']['entities']), 'relations': dict(export['vocab']['relations'])}
    check(sorted(vocab['entities'].values()) == list(range(len(vocab['entities']))), 'entity ids not dense')
    check(sorted(vocab['relations'].values()) == list(range(len(vocab['relations']))), 'relation ids not dense')
    added_entities, added_relations = {}, {}
    for name in sorted(coverage['missing_classes']):
        check(name not in vocab['entities'], 'coverage class already in fold vocabulary: %s' % name)
        vocab['entities'][name] = len(vocab['entities'])
        added_entities[name] = vocab['entities'][name]
    for name in sorted(coverage['missing_roles']):
        check(name not in vocab['relations'] and 'inverse:' + name not in vocab['relations'], 'coverage role already present')
        role = len(vocab['relations'])
        check(role % 2 == 0, 'odd role id')
        vocab['relations'][name] = role
        vocab['relations']['inverse:' + name] = role + 1
        added_relations[name] = role
        added_relations['inverse:' + name] = role + 1
    check(set(added_entities) == set(fold0_audit['appended_entities']) and set(added_relations) == set(fold0_audit['appended_relations']), 'appended names differ from fold0 union')
    return vocab, added_entities, added_relations


def shared_background(vocab, triples, train_diseases, fold0_vocab, fold0_triples, fold0_train_diseases):
    names_k = set(vocab['entities']) - train_diseases
    names_0 = set(fold0_vocab['entities']) - fold0_train_diseases
    check(names_k == names_0, 'non-disease vocabulary differs from fold0 (%d vs %d)' % (len(names_k), len(names_0)))
    check(vocab['relations'] == fold0_vocab['relations'], 'relation table differs from fold0')
    clinical = {vocab['relations'][r] for r in CLINICAL_ROLES} | {vocab['relations']['inverse:' + r] for r in CLINICAL_ROLES}
    to_fold0 = np.full(len(vocab['entities']), -1, dtype=np.int64)
    for name, value in vocab['entities'].items():
        if name not in train_diseases:
            to_fold0[value] = fold0_vocab['entities'][name]
    native_k = triples[~np.isin(triples[:, 1], list(clinical))]
    native_0 = fold0_triples[~np.isin(fold0_triples[:, 1], list(clinical))]
    mapped = np.stack([to_fold0[native_k[:, 0]], native_k[:, 1], to_fold0[native_k[:, 2]]], axis=1)
    check((mapped >= 0).all(), 'disease entity under a native relation')
    check(len(mapped) == len(native_0) and np.array_equal(sorted_rows(mapped), sorted_rows(np.asarray(native_0))), 'native ontology triples differ from fold0')
    return {'non_disease_entities': len(names_k), 'native_triples': int(len(native_0)), 'relations': len(vocab['relations']), 'identical_to_fold0': True}


def backend_key(constants, atoms):
    def endpoint(name):
        return 'C:' + str(constants[name]) if name in constants else 'V:' + name
    graph = sorted(set((endpoint(h), r, endpoint(t)) for h, r, t in atoms))
    return hashlib.sha256(json.dumps(graph, separators=(',', ':')).encode()).hexdigest()


def remap_ontology(bank, fold0_vocab, vocab, class_pool):
    names0 = {value: name for name, value in fold0_vocab['entities'].items()}
    mapping = np.full(len(names0), -1, dtype=np.int64)
    for value, name in names0.items():
        if not name.startswith(DISEASE_PREFIX):
            mapping[value] = vocab['entities'][name]

    def one(value):
        result = int(mapping[int(value)])
        check(result >= 0, 'ontology record references a disease id')
        return result

    def many(values):
        return sorted(one(v) for v in values)
    class_ids = sorted(vocab['entities'][name] for name in class_pool['class_iris'])
    check(len(class_ids) == len(set(class_ids)) == len(bank['class_ids']) and class_ids == many(bank['class_ids']), 'class pool remap mismatch')
    out, partition = {}, collections.defaultdict(set)
    for listing in ('ontology_queries', 'additional_ontology_queries'):
        rows = []
        for row in bank[listing]:
            new = dict(row)
            new['constants'] = {k: one(v) for k, v in row['constants'].items()}
            new['answers'] = many(row['answers'])
            new['asserted_answers'] = many(row['asserted_answers'])
            new['negative_known'] = many(row['negative_known'])
            new['taxonomy_target'] = None if row['taxonomy_target'] is None else one(row['taxonomy_target'])
            check(backend_key(row['constants'], row['atoms']) == row['backend_key'], 'fold0 backend key not reproducible')
            new['backend_key'] = backend_key(new['constants'], row['atoms'])
            partition[(row['backend_key'], new['backend_key'])].add(listing + ':' + str(row['index']))
            check(set(new['answers']) <= set(new['negative_known']), 'answer outside mask')
            check((len(class_ids) - len(new['negative_known']) >= 128) == row['sampleable'], 'sampleable flag changed')
            rows.append(new)
        out[listing] = rows
    check(len({a for a, b in partition}) == len({b for a, b in partition}) == len(partition), 'backend alias partition changed under remap')
    return out, class_ids


def clinical(export, vocab, relations, iri, fold):
    entities = vocab['entities']
    genes_of = collections.defaultdict(set)
    hp_of = collections.defaultdict(set)
    triples = export['triples']
    for head, _, tail in triples[triples[:, 1] == relations['associated_with']]:
        genes_of[int(tail)].add(int(head))
    for head, _, tail in triples[triples[:, 1] == relations['has_symptom']]:
        hp_of[int(head)].add(int(tail))
    hp_ids = sorted(value for name, value in entities.items() if name.startswith(HP_PREFIX))
    hp_set = set(hp_ids)
    candidates = export['candidates']
    candidate_set = set(candidates)
    diseases, join = set(), set()
    for group in export['groups']:
        check(not diseases.intersection(group['disease_ids']), 'disease in two profile groups')
        diseases.update(group['disease_ids'])
        check(all(sorted(hp_of[d]) == group['hp_ids'] for d in group['disease_ids']), 'group hp mismatch')
        check(sorted(set().union(*(genes_of[d] for d in group['disease_ids']))) == group['gene_ids'], 'group genes mismatch')
        join.update((g, h) for g in group['gene_ids'] for h in group['hp_ids'])
    native_join = {(g, h) for d in diseases for g in genes_of[d] for h in hp_of[d]}
    check(diseases == set(genes_of) == set(hp_of) and native_join == join, 'native join differs from grouped join')
    hp_genes = collections.defaultdict(set)
    for g, h in native_join:
        hp_genes[h].add(g)

    def hp_names(node):
        if node['op'] == 'Anchor':
            return [node['iri']]
        if node['op'] == 'P':
            return hp_names(node['child'])
        return [name for child in node['children'] for name in hp_names(child)]
    iri_train = read(iri / ('fold%d' % fold) / 'train-profile-groups.json')
    by_hp = {tuple(sorted(entities[n] for n in hp_names(row['query']))): row for row in iri_train}
    check(len(by_hp) == len(iri_train) == len(export['groups']), 'IRI TRAIN profile count differs')
    inv_symptom, inv_assoc = relations['inverse_has_symptom'], relations['inverse_associated_with']
    profiles = []
    for index, group in enumerate(export['groups']):
        hp, genes = group['hp_ids'], group['gene_ids']
        check(set(hp) <= hp_set and set(genes) <= candidate_set, 'profile outside typed pools')
        known = set(genes)
        for h in hp:
            known.update(hp_genes[h])
        known = sorted(known)
        check(len(candidate_set - set(known)) >= 128, 'fewer than 128 typed negatives for a profile')
        origin = by_hp[tuple(hp)]
        check(sorted(entities[n] for n in origin['train_answers']) == genes, 'IRI TRAIN answers differ')
        profiles.append({'type': 'profile', 'full_train_index': index, 'hp_ids': hp, 'constants': {'s' + str(i): v for i, v in enumerate(hp)},
                         'atoms': [['s' + str(i), inv_symptom, 'e1'] for i in range(len(hp))] + [['e1', inv_assoc, 'f']], 'answers': genes, 'negative_known': known,
                         'negative_known_sha256': hashlib.sha256(np.asarray(known, dtype='<i8').tobytes()).hexdigest(), 'domain': 'gene',
                         'disease_ids': group['disease_ids'], 'disease_names': group['disease_names'], 'truth_scope': TRUTH_SCOPE, 'source_case_ids': origin['source_case_ids']})

    def attach(cases, split):
        origin = {row['source_disease']: row for row in (json.loads(line) for line in (iri / ('fold%d' % fold) / ('%s-cases.jsonl' % split)).open())}
        check(len(origin) == len(cases), '%s IRI case count differs' % split)
        out = []
        for case in cases:
            row = origin[case['disease_name']]
            check(case['gene_ids'] == sorted(entities[n] for n in row['answers']), '%s IRI answers differ' % split)
            check(case['hp_ids'] == sorted({entities[n] for n in row['original_profile']}), '%s IRI profile differs' % split)
            check(set(case['hp_ids']) <= hp_set, '%s phenotype outside hp_ids' % split)
            out.append({**case, 'source_case_id': row['case_id'], 'source_rows': row['source_rows']})
        check(len({c['disease_name'] for c in out}) == len(out), '%s cases are not distinct diseases' % split)
        return out
    return {'hp_ids': hp_ids, 'candidate_ids': candidates, 'training_profiles': profiles,
            'validation': attach(export['validation'], 'validation'), 'test': attach(export['test'], 'test')}


def derangement(cases, seed):
    rng = np.random.default_rng(seed)
    for attempt in range(10000):
        order = rng.permutation(len(cases))
        if all(i != int(j) and cases[i]['hp_ids'] != cases[int(j)]['hp_ids'] for i, j in enumerate(order)):
            return [int(j) for j in order], attempt
    raise Refusal('no profile-changing derangement found')


def build(args):
    started = time.time()
    output = pathlib.Path(args.output)
    check(not output.exists(), 'output exists; preserve it and choose a fresh directory')
    work = pathlib.Path(args.work)
    fold = args.fold
    verify_complete(args.fold0_graph, FOLD0_GRAPH_COMPLETE)
    fold0_bank_complete = verify_complete(args.fold0_bank, FOLD0_BANK_COMPLETE)
    verify_complete(args.iri_bank, IRI_COMPLETE)
    check(sha(args.class_pool) == CLASS_POOL_SHA256, 'class pool hash mismatch')
    iri = pathlib.Path(args.iri_bank)
    coverage = read(iri / 'ontology-coverage.json')
    fold0_graph = pathlib.Path(args.fold0_graph)
    fold0_vocab = read(fold0_graph / 'vocabulary.json')
    fold0_audit = read(fold0_graph / 'union-audit.json')
    fold0_triples = np.load(fold0_graph / 'triples.npy', mmap_mode='r')
    prepared, identity, status = prepared_fold(args.prepare_source, args.benchmark_inputs, work, fold)
    if fold == 0:
        check({k: identity['outputs'][k]['sha256'] for k in FOLD0_PREPARED_OUTPUTS} == FOLD0_PREPARED_OUTPUTS, 'fold0 prepared graph not reproduced')
    export = export_fold(prepared)
    vocab, added_entities, added_relations = union(export, coverage, fold0_audit)
    fold0_train = {n for n in fold0_vocab['entities'] if n.startswith(DISEASE_PREFIX)}
    background = shared_background(vocab, export['triples'], export['train_diseases'], fold0_vocab, np.asarray(fold0_triples), fold0_train)
    relations = {'associated_with': vocab['relations']['associated_with'], 'has_symptom': vocab['relations']['has_symptom'],
                 'inverse_associated_with': vocab['relations']['inverse:associated_with'], 'inverse_has_symptom': vocab['relations']['inverse:has_symptom']}
    bank0 = read(pathlib.Path(args.fold0_bank) / 'bank.json')
    check(relations == bank0['relations'], 'clinical relation ids differ from fold0')
    class_pool = read(args.class_pool)
    ontology, class_ids = remap_ontology(bank0, fold0_vocab, vocab, class_pool)
    clin = clinical(export, vocab, relations, iri, fold)
    triples_bytes = None
    output.mkdir(parents=True)
    receipt = {'fold': fold, 'builder_sha256': sha(__file__), 'prepare_sha256': PREPARE_SHA256, 'prepared_status': status, 'prepared_identity_sha256': sha(prepared / 'identity.json'),
               'prepared_inputs': identity['inputs'], 'export_counts': export['counts'], 'shared_background': background, 'appended_entities': len(added_entities),
               'appended_relations': len(added_relations), 'fold0_graph_complete_sha256': FOLD0_GRAPH_COMPLETE, 'fold0_bank_complete_sha256': FOLD0_BANK_COMPLETE,
               'iri_bank_complete_sha256': IRI_COMPLETE, 'class_pool_sha256': CLASS_POOL_SHA256, 'materialized_causes_phenotype': False,
               'heldout_disease_edges_in_graph': False, 'graph_sampled_queries': False}
    if fold == 0:
        temporary = output / 'reproduction-triples.npy'
        np.save(temporary, export['triples'])
        triples_bytes = sha(temporary)
        temporary.unlink()
        gate = {'triples.npy': triples_bytes == read(fold0_graph / 'COMPLETE.json')['files']['triples.npy'],
                'vocabulary.json': hashlib.sha256((json.dumps(vocab, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest() == read(fold0_graph / 'COMPLETE.json')['files']['vocabulary.json'],
                'ontology_queries': ontology['ontology_queries'] == bank0['ontology_queries'],
                'additional_ontology_queries': ontology['additional_ontology_queries'] == bank0['additional_ontology_queries'],
                'class_ids': class_ids == bank0['class_ids'], 'hp_ids': clin['hp_ids'] == bank0['hp_ids'], 'candidate_ids': clin['candidate_ids'] == bank0['candidate_ids'],
                'training_profiles': clin['training_profiles'] == bank0['training_profiles'], 'validation': clin['validation'] == bank0['validation']}
        receipt['fold0_reproduction'] = gate
        check(all(gate.values()), 'fold0 reproduction gate failed: %s' % sorted(k for k, v in gate.items() if not v))
        bank_complete = sha(pathlib.Path(args.fold0_bank) / 'COMPLETE.json')
        graph_complete = FOLD0_GRAPH_COMPLETE
    else:
        graph_dir = output / ('graph-fold%d' % fold)
        graph_dir.mkdir()
        np.save(graph_dir / 'triples.npy', export['triples'])
        shutil.copyfile(prepared / 'candidates.txt', graph_dir / 'candidates.txt')
        write_json(graph_dir / 'vocabulary.json', vocab)
        write_json(graph_dir / 'union-audit.json', {'fold': fold, 'entities': len(vocab['entities']), 'relations': len(vocab['relations']), 'appended_entities': added_entities,
                                                   'appended_relations': added_relations, 'existing_ids_unchanged': True, 'shared_background': background,
                                                   'edge_policy': 'No fabricated edges or materialized gene-phenotype shortcuts; held-out diseases absent', 'source_sha256': sha(__file__)})
        write_json(graph_dir / 'COMPLETE.json', {'files': {n: sha(graph_dir / n) for n in ('triples.npy', 'candidates.txt', 'vocabulary.json', 'union-audit.json')},
                                                'source_sha256': sha(__file__), 'fold': fold, 'prepared_identity_sha256': receipt['prepared_identity_sha256']})
        graph_complete = sha(graph_dir / 'COMPLETE.json')
        shuffle, attempts = derangement(clin['validation'], SHUFFLE_SEED + fold)
        bank = {key: value for key, value in bank0.items()}
        bank.update({'fold': fold, 'num_entities': len(vocab['entities']), 'num_relations': len(vocab['relations']),
                     'ontology_queries': ontology['ontology_queries'], 'additional_ontology_queries': ontology['additional_ontology_queries'],
                     'class_ids': class_ids, 'candidate_ids': clin['candidate_ids'], 'hp_ids': clin['hp_ids'], 'relations': relations,
                     'training_profiles': clin['training_profiles'], 'validation': clin['validation'], 'validation_shuffle': shuffle, 'probe_queries': [],
                     'clinical_counts': {'validation_cases': len(clin['validation']), 'validation_associations': sum(len(c['gene_ids']) for c in clin['validation'])},
                     'axiom_bank': {**bank0['axiom_bank'], 'fold_remap': {'fold': fold, 'source_bank_complete_sha256': FOLD0_BANK_COMPLETE, 'graph_complete_sha256': graph_complete,
                                    'rule': 'fold0 axiom-only bank ids mapped through IRIs; identical non-disease vocabulary, relations and native triples verified',
                                    'validation_shuffle': 'seeded profile-changing derangement (seed %d); descriptive shuffled control only, never used for selection' % (SHUFFLE_SEED + fold),
                                    'probe_queries': 'empty: fold0 probes are fold0-TRAIN specific and unused by trainer and evaluator',
                                    'provenance_files': 'fold0 provenance.jsonl/axioms.jsonl apply unchanged by source_id and index'}}})
        bank_dir = output / ('bank-fold%d' % fold)
        bank_dir.mkdir()
        write_json(bank_dir / 'bank.json', bank, compact=True)
        write_json(bank_dir / 'audit.json', {**receipt, 'graph_complete_sha256': graph_complete, 'training_profiles': len(clin['training_profiles']),
                                            'validation_cases': len(clin['validation']), 'validation_associations': bank['clinical_counts']['validation_associations'],
                                            'shuffle_attempts': attempts, 'ontology_groups': len(ontology['ontology_queries']) + len(ontology['additional_ontology_queries'])})
        write_json(bank_dir / 'COMPLETE.json', {'files': {n: sha(bank_dir / n) for n in ('bank.json', 'audit.json')}, 'fold': fold, 'mode': 'original', 'taxonomy_policy': 'explicit',
                                               'format': bank0['format'], 'builder_sha256': sha(__file__), 'fold0_bank_complete_sha256': FOLD0_BANK_COMPLETE, 'graph_complete_sha256': graph_complete})
        bank_complete = sha(bank_dir / 'COMPLETE.json')
    test_dir = output / ('test-fold%d' % fold)
    test_dir.mkdir()
    write_json(test_dir / 'test-cases.json', {'fold': fold, 'cases': clin['test'], 'candidate_ids': clin['candidate_ids'], 'relations': relations,
                                             'graph_complete_sha256': graph_complete, 'bank_complete_sha256': bank_complete,
                                             'scope': 'TEST diseases, one row per distinct disease; read only by test_eval.py after validation selection'})
    write_json(test_dir / 'COMPLETE.json', {'files': {'test-cases.json': sha(test_dir / 'test-cases.json')}, 'fold': fold, 'test_cases': len(clin['test']),
                                           'test_associations': sum(len(c['gene_ids']) for c in clin['test']), 'bank_complete_sha256': bank_complete, 'builder_sha256': sha(__file__)})
    receipt.update({'graph_complete_sha256': graph_complete, 'bank_complete_sha256': bank_complete, 'test_complete_sha256': sha(test_dir / 'COMPLETE.json'),
                    'validation_cases': len(clin['validation']), 'test_cases': len(clin['test']), 'elapsed_seconds': time.time() - started})
    write_json(output / 'RECEIPT.json', receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--benchmark-inputs', required=True)
    parser.add_argument('--prepare-source', required=True)
    parser.add_argument('--fold0-graph', required=True)
    parser.add_argument('--fold0-bank', required=True)
    parser.add_argument('--iri-bank', required=True)
    parser.add_argument('--class-pool', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    try:
        build(parser.parse_args())
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}), flush=True)
        sys.exit(2)
