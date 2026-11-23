"""Native-annotation modality graphs derived from a fold's PFS graph, with the historical phenotype fallback.

A modality keeps the native gene annotations it names (P has_phenotype, F has_function, S expressed_in) and removes
the others together with their inverse edges. Ontology background, vocabulary, TRAIN associated_with edges and TRAIN
has_symptom profiles are kept unchanged. Arms without P (F, S, FS) give every gene that has a native phenotype edge
but no edge of an active relation exactly one phenotype edge: the first row of a pinned order source whose
phenotype is represented. Support is decided from native annotation edges only, before and independently of TRAIN
associated_with edges, as in the historical construction (kge_transd.py 252-265 at 1d5c0a12).
"""
import collections
import csv
import pathlib

import numpy as np

from common import (CANDIDATE_COUNT, CLINICAL_RELATIONS, GENE_PREFIX, LEGACY_SHORTCUT, MODALITIES, MP_PREFIX,
                    NATIVE_RELATION, check, sha, sha_bytes)

ORDER_POLICIES = ('prepared-csv', 'raw-train-line-order')


def is_gene(name):
    """True for a numeric benchmark gene IRI."""
    return name.startswith(GENE_PREFIX) and name[len(GENE_PREFIX):].isdigit()


def relation_ids(vocab):
    """Forward and inverse ids of every native and clinical relation used here."""
    relations = vocab['relations']
    check(not any(LEGACY_SHORTCUT in name for name in relations), 'legacy causes_phenotype relation present in vocabulary')
    out = {}
    for name in list(NATIVE_RELATION.values()) + list(CLINICAL_RELATIONS):
        check(name in relations and 'inverse:' + name in relations, 'relation %s missing' % name)
        out[name] = (int(relations[name]), int(relations['inverse:' + name]))
    return out


def _normalize_gene(value):
    value = value.strip()
    return GENE_PREFIX + value if value.isdigit() else value


def _normalize_phenotype(value):
    value = value.strip()
    if value.startswith('MP:'):
        return MP_PREFIX + value[3:]
    if value.startswith('MP_'):
        return MP_PREFIX + value[3:]
    return value


def load_order(policy, path, expected_sha256):
    """Ordered phenotype lists per gene from a pinned order source.

    prepared-csv: a CSV with Gene and Phenotype columns in the historical prepared-annotation row order.
    raw-train-line-order: has_phenotype rows of a fold's raw/train.txt in file order.
    Returns (gene -> [phenotype IRI, ...], provenance).
    """
    check(policy in ORDER_POLICIES, 'unknown fallback order policy %s' % policy)
    path = pathlib.Path(path)
    digest = sha(path)
    check(expected_sha256 is not None and digest == expected_sha256, 'fallback order source hash %s differs from the pinned %s' % (digest, expected_sha256))
    order = collections.defaultdict(list)
    rows = 0
    if policy == 'prepared-csv':
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            check(reader.fieldnames is not None and 'Gene' in reader.fieldnames and 'Phenotype' in reader.fieldnames,
                  'prepared CSV lacks Gene/Phenotype columns: %s' % reader.fieldnames)
            for row in reader:
                rows += 1
                order[_normalize_gene(row['Gene'])].append(_normalize_phenotype(row['Phenotype']))
    else:
        with path.open() as stream:
            for line in stream:
                head, relation, tail = line.rstrip('\n').split('\t')
                if relation == NATIVE_RELATION['P']:
                    rows += 1
                    order[head].append(tail)
    return dict(order), {'policy': policy, 'path': str(path), 'sha256': digest, 'rows': rows, 'genes': len(order)}


def native_support(triples, vocab):
    """Gene ids with at least one forward edge per native relation, and the forward phenotype edge set."""
    ids = relation_ids(vocab)
    support = {}
    for key, name in NATIVE_RELATION.items():
        rows = triples[triples[:, 1] == ids[name][0]]
        support[key] = set(np.unique(rows[:, 0]).tolist())
    phenotype = triples[triples[:, 1] == ids[NATIVE_RELATION['P']][0]]
    return support, phenotype


def fallback_genes(support, modality):
    """Genes that keep one phenotype edge in MODALITY: native phenotype support and no active-relation support."""
    check(modality in MODALITIES, 'unknown modality %s' % modality)
    if 'P' in modality:
        return set()
    active = set()
    for key in modality:
        active |= support[key]
    return support['P'] - active


def choose_fallback(genes, phenotype_edges, vocab, order):
    """First represented phenotype per gene in the pinned order; refuses any order/graph disagreement.

    A row is eligible when its phenotype is a vocabulary entity (represented). Every eligible row met before the chosen
    one must also be a native graph edge for that gene; otherwise the order source is not the one that built the graph.
    """
    entities = vocab['entities']
    names = {value: name for name, value in entities.items()}
    edges = collections.defaultdict(set)
    for head, _, tail in phenotype_edges.tolist():
        edges[head].add(tail)
    chosen, disagreements, missing = {}, [], []
    for gene in sorted(genes):
        gene_name = names[gene]
        selected = None
        for phenotype in order.get(gene_name, []):
            if phenotype not in entities:
                continue
            value = entities[phenotype]
            if value not in edges[gene]:
                disagreements.append([gene_name, phenotype])
                break
            selected = value
            break
        if selected is None:
            missing.append(gene_name)
        else:
            chosen[gene] = selected
    check(not disagreements, 'order source lists a represented phenotype that is not a graph edge for %d genes, e.g. %s' % (len(disagreements), disagreements[:3]))
    check(not missing, '%d fallback genes have no eligible phenotype in the order source, e.g. %s' % (len(missing), missing[:3]))
    return chosen


def _sorted_rows(x):
    return x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]


def derive(base, vocab, candidate_names, modality, chosen):
    """Modality triples from BASE plus an audit; CHOSEN maps fallback gene id to phenotype id.

    Row order: the base rows that survive, in base order, then the fallback forward and inverse rows sorted
    lexicographically by head, relation, tail.
    """
    ids = relation_ids(vocab)
    removed = [key for key in NATIVE_RELATION if key not in modality]
    removed_ids = [i for key in removed for i in ids[NATIVE_RELATION[key]]]
    keep = ~np.isin(base[:, 1], removed_ids)
    kept = base[keep]
    forward_p, inverse_p = ids[NATIVE_RELATION['P']]
    extra = np.empty((0, 3), dtype=np.int64)
    if chosen:
        check('P' not in modality, 'fallback rows offered to an arm that keeps P')
        genes = np.array(sorted(chosen), dtype=np.int64)
        phenotypes = np.array([chosen[g] for g in sorted(chosen)], dtype=np.int64)
        extra = np.concatenate([np.stack([genes, np.full_like(genes, forward_p), phenotypes], axis=1),
                                np.stack([phenotypes, np.full_like(genes, inverse_p), genes], axis=1)])
        extra = _sorted_rows(extra)
    triples = np.ascontiguousarray(np.concatenate([kept, extra]), dtype=np.int64)
    check(len(np.unique(triples, axis=0)) == len(triples), 'duplicate triples after derivation')
    audit = _audit(base, triples, vocab, candidate_names, modality, chosen, ids, removed)
    return triples, audit


def _audit(base, triples, vocab, candidate_names, modality, chosen, ids, removed):
    names = {value: name for name, value in vocab['entities'].items()}
    forward_ids = {pair[0] for pair in ids.values()}
    inverse_of = {pair[0]: pair[1] for pair in ids.values()}
    for relation_forward, relation_inverse in ids.values():
        forward = triples[triples[:, 1] == relation_forward]
        inverse = triples[triples[:, 1] == relation_inverse]
        mirrored = forward[:, [2, 1, 0]].copy()
        mirrored[:, 1] = relation_inverse
        check(len(forward) == len(inverse) and np.array_equal(_sorted_rows(mirrored), _sorted_rows(inverse)), 'inverse edges do not mirror relation %d' % relation_forward)
    for name in CLINICAL_RELATIONS:
        for value in ids[name]:
            check(np.array_equal(_sorted_rows(base[base[:, 1] == value]), _sorted_rows(triples[triples[:, 1] == value])), 'clinical relation %s changed' % name)
    native_ids = {i for pair in ids.values() for i in pair}
    other_base = base[~np.isin(base[:, 1], list(native_ids))]
    other_new = triples[~np.isin(triples[:, 1], list(native_ids))]
    check(len(other_base) == len(other_new) and np.array_equal(other_base, other_new), 'ontology background changed')
    counts = {}
    for key, name in NATIVE_RELATION.items():
        counts[name] = {'base_forward': int((base[:, 1] == ids[name][0]).sum()), 'forward': int((triples[:, 1] == ids[name][0]).sum())}
    for key in removed:
        name = NATIVE_RELATION[key]
        expected = len(chosen) if key == 'P' else 0
        check(counts[name]['forward'] == expected, 'removed relation %s keeps %d forward edges' % (name, counts[name]['forward']))
    candidates = np.array([vocab['entities'][n] for n in candidate_names], dtype=np.int64)
    check(len(candidates) == CANDIDATE_COUNT == len(set(candidates.tolist())), 'candidate list is not the 13,625 main pool')
    degree = np.bincount(triples[:, 0], minlength=len(names))
    isolated_candidates = int((degree[candidates] == 0).sum())
    association_genes = set(triples[triples[:, 1] == ids['associated_with'][0]][:, 0].tolist())
    annotated = set()
    for key in modality:
        annotated |= set(triples[triples[:, 1] == ids[NATIVE_RELATION[key]][0]][:, 0].tolist())
    candidate_set = set(candidates.tolist())
    fallback = sorted(chosen)
    check(set(fallback) <= candidate_set, 'fallback gene outside the main candidate pool')
    fallback_listing = ''.join('%s\t%s\n' % (names[g], names[chosen[g]]) for g in fallback).encode()
    return {
        'modality': modality,
        'removed_native_relations': [NATIVE_RELATION[k] for k in removed],
        'base_triples': int(len(base)),
        'triples': int(len(triples)),
        'removed_rows': int(len(base) - len(triples) + 2 * len(chosen)),
        'fallback_rows_added': int(2 * len(chosen)),
        'native_forward_counts': counts,
        'fallback_genes': len(fallback),
        'fallback_genes_with_train_association': len(set(fallback) & association_genes),
        'fallback_edges_sha256': sha_bytes(fallback_listing),
        'fallback_edges': [[names[g], names[chosen[g]]] for g in fallback],
        'candidates_without_active_or_fallback_annotation': len(candidate_set - annotated - set(fallback)),
        'isolated_candidates': isolated_candidates,
        'checks': {'inverse_mirrors_exact': True, 'clinical_edges_unchanged': True, 'ontology_background_unchanged': True,
                   'legacy_causes_phenotype_absent': True, 'fallback_subset_of_candidates': True},
        'forward_relation_ids_checked': sorted(forward_ids),
        'inverse_relation_ids_checked': sorted(inverse_of.values()),
    }


def fallback_test_pairs(fallback_names, test_tables):
    """TEST rows whose gene receives a fallback edge, per fold and in total, from (fold, rows) pairs."""
    per_fold, genes = {}, set()
    for fold, rows in test_tables:
        hit = [row for row in rows if row[0] in fallback_names]
        per_fold[fold] = len(hit)
        genes.update(row[0] for row in hit)
    return {'per_fold': per_fold, 'total': sum(per_fold.values()), 'target_genes': len(genes)}
