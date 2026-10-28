#!/usr/bin/env python3
"""Verify all controlled query outputs and apply the fixed eight-contrast analysis."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phenoquery.common import cal, metrics, pairs, ranks, read, sha, write
from phenoquery.campaign import verify_global
from analysis.summarize_query_tables import metadata

ARMS = ['bridge_pool', 'plain_pool', 'plain_and', 'typed_and']
INITS = ['ultra_4g', 'ultraquery']
SEEDS = [0, 1, 2]
CONFIDENCE = 1 - .05 / 8
BOOTSTRAP_SEED = 20261028
REPLICATES = 10000


def checked_file(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def cluster_intervals(differences, profiles, repeats=REPLICATES, seed=BOOTSTRAP_SEED,
                      confidence=CONFIDENCE):
    """Resample whole profiles; preserve each association's original weight."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(profiles) or not len(values):
        raise ValueError('Expected an association-by-contrast matrix and profile identities')
    if not np.isfinite(values).all() or repeats < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap inputs')
    lookup = {}
    cluster_ids = []
    for profile in profiles:
        cluster_ids.append(lookup.setdefault(profile, len(lookup)))
    cluster_ids = np.asarray(cluster_ids)
    sizes = np.bincount(cluster_ids)
    sums = np.zeros((len(sizes), values.shape[1]), dtype=np.float64)
    np.add.at(sums, cluster_ids, values)
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, values.shape[1]), dtype=np.float64)
    for start in range(0, repeats, 100):
        selected = rng.integers(0, len(sizes), size=(min(100, repeats-start), len(sizes)))
        boot[start:start+len(selected)] = sums[selected].sum(1) / sizes[selected].sum(1)[:, None]
    alpha = (1-confidence)/2
    bounds = np.quantile(boot, [alpha, 1-alpha], axis=0)
    return {'point': values.mean(0).tolist(), 'lower': bounds[0].tolist(),
            'upper': bounds[1].tolist(), 'confidence_level': confidence,
            'replicates': repeats, 'seed': seed, 'profile_clusters': len(sizes),
            'associations': len(values)}


def verify_plan(root):
    plan = read(root/'CONTROLLED.json')
    expected = {'arms': ARMS, 'initializations': INITS, 'seeds': SEEDS,
                'expected_runs': 240, 'runs_per_campaign': 40,
                'campaigns': [f'{init}-seed{seed}' for init in INITS for seed in SEEDS]}
    for key, value in expected.items():
        if plan[key] != value:
            raise ValueError(f'Controlled design differs from fixed protocol: {key}')
    protocol = Path(__file__).resolve().parents[2]/'reproducibility/protocols/controlled-query-20261028.md'
    checked_file(protocol, plan['protocol_sha256'])
    global_sha = sha(root/'GLOBAL-SELECTION.json')
    for name in plan['campaigns']:
        child = root/name
        verify_global(child, global_sha)
        checked_file(child/'source.sha256', plan['source_sha256'])
        checked_file(child/'PROTOCOL.md', plan['protocol_sha256'])
        selection = read(child/'SELECTION.json')
        if selection['protocol_sha256'] != plan['protocol_sha256']:
            raise ValueError('Selection protocol changed')
        expected_runs = {f'{arm}-fold{fold}' for fold in range(10) for arm in ARMS}
        if set(selection['runs']) != expected_runs:
            raise ValueError('Incomplete or unexpected training selections')
    return plan, global_sha


def verify_validation_choice(progress, trained):
    curve = progress['curve']
    if progress['step'] != 500 or [x['step'] for x in curve] != list(range(0,501,100)):
        raise ValueError('Incomplete validation trajectory')
    baseline = curve[0]['both_modes']['calibrated']
    best = None
    for item in curve:
        current = item['both_modes']['calibrated']
        for mode in item['both_modes'].values():
            if not all(np.isfinite(v) for v in mode.values()):
                raise ValueError('Nonfinite validation metrics')
        eligible = current['hits@1'] >= baseline['hits@1'] and current['hits@10'] >= baseline['hits@10']
        key = lambda x: (x['hits@10'],x['hits@1'],x['mrr'],-x['mr'])
        chosen = best is None or (eligible and key(current) > key(best['metrics']))
        if item['eligible'] != eligible or item['selected'] != chosen:
            raise ValueError('Validation trajectory violates fixed selection rule')
        if chosen: best = {'step':item['step'],'metrics':current,'both_modes':item['both_modes']}
    if best != trained['best'] or best != progress['best'] or baseline != trained['baseline'] or baseline != progress['baseline']:
        raise ValueError('Frozen choice differs from validation trajectory')


def load_bundle(child, fold, arm, kind, init, seed, genes, diseases, rows, seen, data_hashes):
    selection = read(child/'SELECTION.json')
    run = f'{arm}-fold{fold}'
    parent = child/'training'/run
    checked_file(parent/'COMPLETE.json', selection['runs'][run]['completion_sha256'])
    trained = read(parent/'COMPLETE.json')
    checked_file(parent/'identity.json', trained['identity_sha256'])
    identity = read(parent/'identity.json')
    config = identity['config']
    expected = {'arm': arm, 'fold': fold, 'steps': 500, 'pilot': False,
                'seed': seed, 'initialization': init, 'global_selection_required': True}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Training configuration mismatch: {run}/{key}')
    if identity['seed'] != seed or identity['initialization'] != init:
        raise ValueError('Training seed/initialization mismatch')
    if trained['steps'] != 500 or trained['pilot'] or trained['arm'] != arm or trained['fold'] != fold:
        raise ValueError('Incomplete training')
    for name, digest in data_hashes.items():
        if identity['identity']['files'].get(name) != digest:
            raise ValueError(f'Training/evaluation data mismatch: {name}')
    for record in [identity, trained]:
        if record['protocol_sha256'] != sha(child/'PROTOCOL.md') or record['source_manifest_sha256'] != sha(child/'source.sha256'):
            raise ValueError('Training source/protocol mismatch')
    if trained['best'] != selection['runs'][run]['best'] or trained['checkpoint_sha256'] != selection['runs'][run]['checkpoint_sha256']:
        raise ValueError('Training choice differs from frozen selection')
    if trained['best']['step'] not in range(0, 501, 100):
        raise ValueError('Selected step is not a scheduled validation')
    verify_validation_choice(read(parent/'progress.json'), trained)
    name = arm if kind == 'selected' else arm+'_frozen'
    out = child/'test'/f'fold{fold}'/name
    complete = read(out/'COMPLETE.json')
    key = 'checkpoint_sha256' if kind == 'selected' else 'initial_checkpoint_sha256'
    wanted = {'arm': arm, 'fold': fold, 'checkpoint_kind': kind, 'initialization': init,
              'seed': seed, 'checkpoint_sha256': trained[key],
              'selection_sha256': sha(child/'SELECTION.json'),
              'selected_step': trained['best']['step'] if kind == 'selected' else 0}
    for field, value in wanted.items():
        if complete[field] != value:
            raise ValueError(f'Score identity mismatch: {name}/{field}')
    if set(complete['files']) != {'scores.npz', 'ranks.npz'}:
        raise ValueError('Unexpected score manifest')
    for filename, digest in complete['files'].items():
        checked_file(out/filename, digest)
    with np.load(out/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['genes'].astype(str), genes)
        np.testing.assert_array_equal(z['diseases'].astype(str), diseases)
        values = z['scores'].copy()
        if values.shape != (len(diseases), len(genes)) or not np.isfinite(values).all():
            raise ValueError('Invalid score matrix')
        reference_name = 'validation.npz' if kind == 'selected' else 'initial-validation.npz'
        reference_key = 'validation_sha256' if kind == 'selected' else 'initial_validation_sha256'
        checked_file(parent/reference_name, trained[reference_key])
        with np.load(parent/reference_name, allow_pickle=False) as reference:
            np.testing.assert_array_equal(reference['genes'].astype(str), genes)
            validation_diseases = sorted({d for g,d in pairs(child/'data'/f'fold{fold}'/'validation.tsv')})
            np.testing.assert_array_equal(reference['diseases'].astype(str), validation_diseases)
            if reference['scores'].shape != (len(validation_diseases), len(genes)):
                raise ValueError('Validation score dimensions changed')
            np.testing.assert_array_equal(reference['scores'], z['reference_scores'])
            np.testing.assert_array_equal(reference['diseases'], z['reference_diseases'])
            if not np.isfinite(reference['scores']).all():
                raise ValueError('Nonfinite validation scores')
            validation_rows = pairs(child/'data'/f'fold{fold}'/'validation.tsv')
            validation_metrics = {m:metrics(ranks(x,validation_diseases,genes,validation_rows)) for m,x in
                                  [('raw',reference['scores']),('calibrated',cal(reference['scores']))]}
            if kind == 'selected' and validation_metrics != trained['best']['both_modes']:
                raise ValueError('Selected validation scores disagree with frozen metrics')
            if kind == 'initial' and validation_metrics['calibrated'] != trained['baseline']:
                raise ValueError('Initial validation scores disagree with baseline metrics')
    result = {}
    with np.load(out/'ranks.npz', allow_pickle=False) as saved:
        for field, expected_axis in [('genes', [g for g,d in rows]), ('diseases', [d for g,d in rows]), ('seen', seen)]:
            np.testing.assert_array_equal(saved[field], expected_axis)
        for mode, scores in [('raw', values), ('calibrated', cal(values))]:
            result[mode] = ranks(scores, diseases, genes, rows)
            np.testing.assert_array_equal(result[mode], saved[mode])
            if metrics(result[mode]) != complete['metrics'][mode]:
                raise ValueError('Saved metrics differ from recomputed ranks')
    return result, {'completion_sha256': sha(out/'COMPLETE.json'), **complete['files'],
                    'training_completion_sha256':sha(parent/'COMPLETE.json'),
                    'training_identity_sha256':sha(parent/'identity.json'),
                    'validation_trajectory_sha256':sha(parent/'progress.json'),
                    'validation_scores_sha256':sha(parent/reference_name)}, wanted['selected_step']


def summarize(root, expected_candidates=13625):
    plan, global_sha = verify_plan(root)
    collected = defaultdict(list)
    strata_parts = defaultdict(list)
    profiles_all, association_ids, fold_counts = [], [], []
    checksums, selections = {}, {}
    previous_test = set()
    for fold in range(10):
        canonical = root/plan['campaigns'][0]/'data'/f'fold{fold}'
        genes, rows, seen, candidate_seen, strata = metadata(canonical)
        diseases = sorted({d for _,d in rows})
        if len(genes) != expected_candidates or previous_test.intersection(diseases):
            raise ValueError('Candidate count or outer test-fold identity mismatch')
        previous_test.update(diseases)
        profiles = read(canonical/'profiles.json')
        profiles_all.extend(tuple(sorted(set(profiles[d]))) for g,d in rows)
        association_ids.extend((fold,g,d) for g,d in rows)
        strata.update(seen=seen, unseen=~seen, zero_direct_overlap=strata['seen_zero'] | ~seen,
                      no_training_profile_match=~strata['training_profile_match'])
        for key, mask in strata.items(): strata_parts[key].append(mask)
        data_hashes = {name: sha(canonical/name) for name in ['training.tsv','validation.tsv','test.tsv','profiles.json','candidates.txt']}
        checksums[f'data/fold{fold}'] = data_hashes
        fold_counts.append({'fold': fold, 'associations': len(rows), 'diseases': len(diseases), 'candidates': len(genes)})
        for init in INITS:
            for seed in SEEDS:
                child = root/f'{init}-seed{seed}'
                for name,digest in data_hashes.items(): checked_file(child/'data'/f'fold{fold}'/name, digest)
                for arm in ARMS:
                    for kind in ['selected','initial']:
                        rr, record, step = load_bundle(child,fold,arm,kind,init,seed,genes,diseases,rows,seen,data_hashes)
                        for mode, values in rr.items(): collected[(init,seed,arm,kind,mode)].append(values)
                        key = f'{child.name}/fold{fold}/{arm}/{kind}'
                        checksums[key] = record
                        if kind == 'selected': selections[key] = step
                    if selections[f'{child.name}/fold{fold}/{arm}/selected'] == 0:
                        for mode in ['raw','calibrated']:
                            np.testing.assert_array_equal(
                                collected[(init,seed,arm,'selected',mode)][-1],
                                collected[(init,seed,arm,'initial',mode)][-1],
                                err_msg='Step-zero selection differs from initial checkpoint ranks')
        print('VERIFIED_CONTROLLED_FOLD', fold, 48, flush=True)
    arrays = {key:np.concatenate(parts) for key,parts in collected.items()}
    strata = {key:np.concatenate(parts) for key,parts in strata_parts.items()}
    conditions = {}
    for init in INITS:
        for arm in ARMS:
            for kind in ['selected','initial']:
                for mode in ['raw','calibrated']:
                    rr = np.stack([arrays[(init,s,arm,kind,mode)] for s in SEEDS])
                    per_seed = {str(s):metrics(rr[s]) for s in SEEDS}
                    conditions[f'{init}/{arm}/{kind}/{mode}'] = {
                        'per_seed': per_seed,
                        'mean_over_seeds': {m:float(np.mean([x[m] for x in per_seed.values()])) for m in per_seed['0']},
                        'sd_across_seeds': {m:float(np.std([x[m] for x in per_seed.values()],ddof=1)) for m in per_seed['0']},
                        'range_across_seeds': {m:[min(x[m] for x in per_seed.values()),max(x[m] for x in per_seed.values())] for m in per_seed['0']},
                        'strata_per_seed': {k:{str(s):metrics(rr[s,mask]) if mask.any() else None for s in SEEDS} for k,mask in strata.items()}}
    differences, contrasts = [], []
    for init in INITS:
        for name,left,right in [
            ('disease_bridges',('plain_pool','selected'),('bridge_pool','selected')),
            ('conjunction',('plain_and','selected'),('plain_pool','selected')),
            ('typed_EQ',('typed_and','selected'),('plain_and','selected')),
            ('adaptation',('typed_and','selected'),('typed_and','initial'))]:
            deltas = np.stack([(arrays[(init,s,*left,'calibrated')]<=10).astype(float) -
                               (arrays[(init,s,*right,'calibrated')]<=10).astype(float) for s in SEEDS])
            differences.append(deltas.mean(0))
            contrasts.append({'initialization':init,'contrast':name,'left':list(left),'right':list(right),
                              'per_seed_percentage_points':(100*deltas.mean(1)).tolist()})
    intervals = cluster_intervals(np.stack(differences,axis=1),profiles_all)
    for j,contrast in enumerate(contrasts):
        contrast.update(difference_percentage_points=100*intervals['point'][j],
                        lower_percentage_points=100*intervals['lower'][j],
                        upper_percentage_points=100*intervals['upper'][j])
    return {'schema':1,'scope':'Controlled reused-benchmark analysis; no independent external validation',
            'plan_sha256':sha(root/'CONTROLLED.json'),'global_selection_sha256':global_sha,
            'protocol_sha256':plan['protocol_sha256'],'training_runs':240,'score_bundles':480,
            'analysis_source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('summarize_query_tables.py'),Path(__file__).resolve().parents[1]/'phenoquery/common.py',Path(__file__).resolve().parents[1]/'phenoquery/campaign.py']},
            'verification_scope':'Score hashes, full axes, retained ranks, metrics and frozen selection/checkpoint identities; not independent checkpoint inference',
            'input_sha256':checksums,'folds':fold_counts,'associations':len(association_ids),
            'profile_clusters':intervals['profile_clusters'],'stratum_counts':{k:int(v.sum()) for k,v in strata.items()},
            'selected_step_counts':dict(sorted(Counter(selections.values()).items())),
            'selected_steps':selections,'conditions':conditions,'primary_contrasts':contrasts,
            'inference':{k:v for k,v in intervals.items() if k not in ['point','lower','upper']},
            'inference_description':'Eight association-pooled calibrated Hits@10 contrasts; metric contributions averaged across seeds before paired phenotype-profile-cluster resampling. Intervals describe benchmark sampling variation, not all training/selection uncertainty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Refusing to overwrite existing report')
    write(args.output,summarize(args.root))


if __name__ == '__main__': main()
