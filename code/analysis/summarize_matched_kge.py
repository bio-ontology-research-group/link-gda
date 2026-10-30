"""Verify primary matched KGE ranks and apply the fixed ten-contrast analysis."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.evaluate_stepwise_bias import std_loo_cal
from analysis.summarize_controlled_query import cluster_intervals
from analysis.summarize_query_tables import metadata
from link_gda.score_io import load_scores
from phenoquery.common import ranks, read, sha, write
from training.run_matched_kge import GRAPHS, MODELS, PRIMARY

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reproducibility/protocols/matched-kge-analysis-20261029.md'
CONFIDENCE, REPLICATES, SEED = .995, 10000, 20261029
CHOICES = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
INPUTS = ('raw/train.txt', 'candidates.txt', 'profiles.json', 'training.tsv', 'validation.tsv', 'test.tsv')


def statistics(values):
    r = np.asarray(values, dtype=float)
    return dict(h1=float(np.mean(r <= 1) * 100), h10=float(np.mean(r <= 10) * 100),
                mrr=float(np.mean(1 / r)), mr=float(np.mean(r)), med=float(np.median(r)), n=len(r))


def primary_spec():
    return {family: dict(choice=PRIMARY[family],
        scoring='calibrated' if PRIMARY[family].startswith('calibrated') else 'raw',
        **{graph: f'MatchedKGE ({family}, {graph}, {PRIMARY[family]})' for graph in GRAPHS})
        for family in MODELS}


def summarize(manifest_path, report_path, ranks_dir, data_root, expected_candidates=13625):
    manifest_path, report_path, ranks_dir, data_root = map(Path, (manifest_path, report_path, ranks_dir, data_root))
    manifest, report = read(manifest_path), read(report_path)
    meta_path = report_path.with_suffix('.metadata.json'); meta = read(meta_path)
    expected_names = {f'MatchedKGE ({family}, {graph}, {choice})'
                      for family in MODELS for graph in GRAPHS for choice in CHOICES}
    expected_runs = {f'fold{fold}/{graph}/{family}' for fold in range(10) for graph in GRAPHS for family in MODELS}
    if (manifest['training_runs'] != 100 or manifest['score_files'] != 400
            or manifest['folds'] != list(range(10)) or meta['folds'] != list(range(10))
            or len(manifest['models']) != 40 or {m['name'] for m in manifest['models']} != expected_names
            or set(report) != expected_names or set(meta['models']) != expected_names
            or any(len(v) != 42 for v in report.values()) or set(manifest['evidence']) != expected_runs):
        raise ValueError('Require the complete matched campaign and evaluation')
    if manifest['primary_comparisons'] != primary_spec():
        raise ValueError('Primary choices or scoring modes differ from the fixed protocol')
    if meta['score_manifest_sha256'] != sha(manifest_path):
        raise ValueError('Evaluation used a different score manifest')
    if manifest['protocol_sha256'] != sha(ROOT / 'reproducibility/protocols/matched-kge-20261029.md'):
        raise ValueError('Training protocol differs')
    if meta['evaluation_sha256'] != sha(Path(sys.modules[std_loo_cal.__module__].__file__)):
        raise ValueError('Rank replay requires the same evaluator source')
    models = {m['name']: m for m in manifest['models']}
    if any(set(m['folds']) != {str(f) for f in range(10)} for m in models.values()):
        raise ValueError('Missing score folds')
    collected, profiles_all, all_rows, seen_diseases, checksums = defaultdict(list), [], [], set(), {}
    for fold in range(10):
        data = data_root / f'fold{fold}'
        for name in INPUTS:
            if sha(data / name) != manifest['benchmark'][str(fold)][name]:
                raise ValueError(f'Benchmark input changed: fold {fold}/{name}')
        genes, rows, seen, candidate_seen, strata = metadata(data)
        diseases = sorted({d for _, d in rows})
        if len(genes) != expected_candidates or set(diseases) & seen_diseases:
            raise ValueError('Candidate count differs or test diseases repeat across folds')
        seen_diseases.update(diseases)
        profiles = {d: tuple(sorted(set(p))) for d, p in read(data / 'profiles.json').items()}
        di, gi = {d: i for i, d in enumerate(diseases)}, {g: i for i, g in enumerate(genes)}
        expected_rows = [dict(fold=fold, g=g, d=d, gi=gi[g], di=di[d], seen=bool(seen[j]),
            j_same_0=bool(strata['seen_zero'][j] or strata['unseen'][j]),
            not_dup_prof=not bool(strata['training_profile_match'][j]),
            prof_tuple=list(profiles[d]), multi_gene=bool(strata['multi_gene'][j])) for j, (g, d) in enumerate(rows)]
        row_path, rank_path = ranks_dir / f'fold{fold}-rows.json', ranks_dir / f'fold{fold}-ranks.npz'
        if read(row_path) != expected_rows:
            raise ValueError('Saved rank rows differ from frozen associations or profiles')
        checksums[f'fold{fold}'] = dict(rows_sha256=sha(row_path), ranks_sha256=sha(rank_path), scores={})
        all_rows.extend(expected_rows); profiles_all.extend(profiles[d] for _, d in rows)
        with np.load(rank_path, allow_pickle=False) as saved:
            for family, spec in primary_spec().items():
                for graph in GRAPHS:
                    name = spec[graph]; entry = models[name]['folds'][str(fold)]
                    scores = load_scores(manifest_path.parent / entry['path'], genes, diseases,
                                         expected_sha256=entry['sha256'])
                    if spec['scoring'] == 'calibrated': scores = std_loo_cal(scores)
                    actual = ranks(scores, diseases, genes, rows)
                    key = name + ('::r_s1' if spec['scoring'] == 'calibrated' else '::r_s0')
                    if key not in saved or not np.array_equal(actual, saved[key]):
                        raise ValueError('Saved primary ranks differ from actual score files')
                    collected[(family, graph)].extend(actual)
                    checksums[f'fold{fold}']['scores'][name] = entry['sha256']
        print(f'Verified matched primary ranks for fold {fold}', flush=True)
    combined = [row for row in all_rows if row['j_same_0'] and row['not_dup_prof']]
    cohorts = dict(pairs=len(all_rows), diseases=len(seen_diseases),
        seen_pairs=sum(r['seen'] for r in all_rows), zero_overlap_pairs=sum(r['j_same_0'] for r in all_rows),
        training_profile_matches=sum(not r['not_dup_prof'] for r in all_rows),
        training_profile_matches_in_zero_overlap=sum(r['j_same_0'] and not r['not_dup_prof'] for r in all_rows),
        combined_pairs=len(combined), combined_diseases=len({r['d'] for r in combined}),
        combined_profiles=len({tuple(r['prof_tuple']) for r in combined}))
    if meta['cohorts'] != cohorts:
        raise ValueError('Reconstructed cohorts differ from evaluation metadata')
    if expected_candidates == 13625 and cohorts != dict(pairs=6571, diseases=6059, seen_pairs=3085,
        zero_overlap_pairs=4431, training_profile_matches=210, training_profile_matches_in_zero_overlap=166,
        combined_pairs=4265, combined_diseases=4017, combined_profiles=4011):
        raise ValueError('Production summary requires the fixed paper benchmark cohorts')
    primary, contrasts, differences = {}, [], []
    for family, spec in primary_spec().items():
        primary[family] = dict(choice=spec['choice'], scoring=spec['scoring'], graphs={})
        for graph in GRAPHS:
            values = np.asarray(collected[(family, graph)])
            stats = statistics(values)
            condition = 'S1_Cal_All' if spec['scoring'] == 'calibrated' else 'S0_Raw_All'
            if report[spec[graph]][condition] != stats:
                raise ValueError('Primary aggregate metrics differ from recomputed ranks')
            evidence = [manifest['evidence'][f'fold{f}/{graph}/{family}'] for f in range(10)]
            primary[family]['graphs'][graph] = dict(metrics=stats,
                selected_epochs=[e['selected'][spec['choice']]['epoch'] for e in evidence],
                nodes=[e['nodes'] for e in evidence], forward_triples=[e['forward_triples'] for e in evidence],
                epochs=[e['epochs'] for e in evidence], dimension=manifest['settings'][family]['dimension'])
        left, right = (np.asarray(collected[(family, graph)]) for graph in GRAPHS)
        differences.extend([(right <= 10).astype(float) * 100 - (left <= 10) * 100, right - left])
        contrasts.extend([dict(family=family, endpoint='hits@10', units='percentage points'),
                          dict(family=family, endpoint='mr', units='rank units')])
    bootstrap = cluster_intervals(np.stack(differences, axis=1), profiles_all,
                                  repeats=REPLICATES, seed=SEED, confidence=CONFIDENCE)
    for i, contrast in enumerate(contrasts):
        contrast.update(difference=bootstrap['point'][i], lower=bootstrap['lower'][i], upper=bootstrap['upper'][i])
    return dict(schema=1, training_runs=100, score_files=400, verified_primary_score_files=100,
        candidates=expected_candidates, associations=len(all_rows), profile_clusters=bootstrap['profile_clusters'],
        primary=primary, contrasts=contrasts, cohorts=cohorts,
        inference=dict(confidence_level=CONFIDENCE, replicates=REPLICATES, seed=SEED,
                       comparisons=10, direction='nobridge minus original', cluster='unique phenotype profile',
                       scope='Benchmark sampling variation conditional on fitted single-seed models; reused benchmark'),
        provenance=dict(manifest_sha256=sha(manifest_path), report_sha256=sha(report_path),
            metadata_sha256=sha(meta_path), freeze_sha256=manifest['freeze_sha256'],
            analysis_protocol_sha256=sha(PROTOCOL), training_protocol_sha256=manifest['protocol_sha256'],
            summary_source_sha256=sha(__file__), evaluation_source_sha256=meta['evaluation_sha256'],
            bootstrap_source_sha256=sha(Path(sys.modules[cluster_intervals.__module__].__file__))),
        checksums=checksums)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('score-manifest', 'report-json', 'ranks-dir', 'data-root', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.score_manifest, a.report_json, a.ranks_dir, a.data_root)
    if a.output.exists() and read(a.output) != result:
        raise ValueError('Existing summary differs; choose a new output path')
    write(a.output, result)


if __name__ == '__main__': main()
