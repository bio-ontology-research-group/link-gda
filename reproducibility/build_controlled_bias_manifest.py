"""Connect every controlled query variant to the shared bias-sensitivity evaluator."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code'))
import numpy as np
from analysis.summarize_controlled_query import summarize, INITS, SEEDS, ARMS
from analysis.summarize_query_tables import metadata
from phenoquery.common import read, sha, write


def build(root, report_json, output, expected_candidates=13625):
    root, report_json, output = map(lambda p: Path(p).resolve(), (root, report_json, output))
    report = read(report_json)
    actual = json.loads(json.dumps(summarize(root, expected_candidates=expected_candidates)))
    if report != actual:
        raise ValueError('Controlled reference report differs from complete score replay')
    models = []
    for initialization in INITS:
        for seed in SEEDS:
            for arm in ARMS:
                for kind in ('initial', 'selected'):
                    campaign = f'{initialization}-seed{seed}'
                    score_arm = arm if kind == 'selected' else arm+'_frozen'
                    folds = {}
                    for fold in range(10):
                        folder = root/campaign/'test'/f'fold{fold}'/score_arm
                        evidence = report['input_sha256'][f'{campaign}/fold{fold}/{arm}/{kind}']
                        folds[str(fold)] = dict(path=os.path.relpath(folder/'scores.npz', output.parent),
                            sha256=evidence['scores.npz'],
                            ranks_path=os.path.relpath(folder/'ranks.npz', output.parent),
                            ranks_sha256=evidence['ranks.npz'])
                    models.append(dict(name=f'ControlledQuery ({initialization}, seed{seed}, {arm}, {kind})',
                        initialization=initialization, seed=seed, arm=arm, checkpoint_kind=kind, folds=folds))
    return dict(schema=1, training_runs=240, score_files=480, folds=list(range(10)),
        protocol_sha256=report['protocol_sha256'], global_selection_sha256=report['global_selection_sha256'],
        controlled_report_sha256=sha(report_json), builder_sha256=sha(__file__),
        data_root=os.path.relpath(root/f'{INITS[0]}-seed{SEEDS[0]}'/'data', output.parent),
        analysis_scope='Descriptive bias sensitivities for all arms, seeds and checkpoint types; preserves the eight prespecified primary contrasts',
        benchmark={str(f):report['input_sha256'][f'data/fold{f}'] for f in range(10)}, models=models)


def verify(manifest_path, controlled_report, bias_report, ranks_dir):
    manifest_path, controlled_report, bias_report, ranks_dir = map(Path,
        (manifest_path, controlled_report, bias_report, ranks_dir))
    manifest, controlled, bias = read(manifest_path), read(controlled_report), read(bias_report)
    meta_path = bias_report.with_suffix('.metadata.json'); meta = read(meta_path)
    expected = {(i,s,a,k) for i in INITS for s in SEEDS for a in ARMS for k in ('initial','selected')}
    keys = [(m['initialization'],m['seed'],m['arm'],m['checkpoint_kind']) for m in manifest['models']]
    if (len(keys) != 48 or set(keys) != expected or manifest['score_files'] != 480
            or manifest['training_runs'] != 240 or manifest['folds'] != list(range(10))):
        raise ValueError('Require all 48 controlled variants and 480 score files')
    if manifest['controlled_report_sha256'] != sha(controlled_report):
        raise ValueError('Controlled report changed')
    if (manifest['protocol_sha256'] != controlled['protocol_sha256']
            or manifest['global_selection_sha256'] != controlled['global_selection_sha256']):
        raise ValueError('Controlled protocol or selection differs')
    names = {m['name'] for m in manifest['models']}
    if (len(names) != 48 or set(bias) != names or set(meta['models']) != names
            or len(meta['models']) != 48
            or meta['folds'] != list(range(10)) or any(len(x) != 42 for x in bias.values())
            or meta['score_manifest_sha256'] != sha(manifest_path)
            or meta['evaluation_sha256'] != sha(HERE.parent/'code/analysis/evaluate_stepwise_bias.py')):
        raise ValueError('Bias evaluation does not cover this complete score manifest')
    conditions = set(next(iter(bias.values())))
    required = {'S0_Raw_All','S1_Cal_All','S1_ReferenceDedup_All','S1_PositiveFilter_All',
                'S1_EqualProfile_All','S1_NoTrainingProfileMatch','S4_Cumulative_All4'}
    if not required <= conditions or any(set(values) != conditions for values in bias.values()):
        raise ValueError('Bias sensitivity conditions are missing or inconsistent')
    hashes = {}
    for fold in range(10):
        data = manifest_path.parent/manifest['data_root']/f'fold{fold}'
        inputs = controlled['input_sha256'][f'data/fold{fold}']
        if manifest['benchmark'][str(fold)] != inputs or any(sha(data/name) != digest for name,digest in inputs.items()):
            raise ValueError('Controlled benchmark changed')
        genes, pairs, seen, _, strata = metadata(data)
        diseases = sorted({d for g,d in pairs})
        gi, di = {g:i for i,g in enumerate(genes)}, {d:i for i,d in enumerate(diseases)}
        profiles = read(data/'profiles.json')
        expected_rows = [dict(fold=fold,g=g,d=d,gi=gi[g],di=di[d],seen=bool(seen[j]),
            j_same_0=bool(strata['seen_zero'][j] or strata['unseen'][j]),
            not_dup_prof=not bool(strata['training_profile_match'][j]),
            prof_tuple=sorted(set(profiles[d])),multi_gene=bool(strata['multi_gene'][j])) for j,(g,d) in enumerate(pairs)]
        path = ranks_dir/f'fold{fold}-ranks.npz'
        row_path = ranks_dir/f'fold{fold}-rows.json'
        rows = read(row_path)
        if rows != expected_rows:
            raise ValueError('Bias evaluation rows or phenotype profiles differ from the controlled benchmark')
        with np.load(path, allow_pickle=False) as evaluated:
            for model in manifest['models']:
                if set(model['folds']) != {str(f) for f in range(10)}:
                    raise ValueError('Missing controlled score fold')
                entry = model['folds'][str(fold)]
                key = f"{model['initialization']}-seed{model['seed']}/fold{fold}/{model['arm']}/{model['checkpoint_kind']}"
                evidence = controlled['input_sha256'][key]
                if (entry['sha256'] != evidence['scores.npz'] or entry['ranks_sha256'] != evidence['ranks.npz']
                        or sha(manifest_path.parent/entry['path']) != entry['sha256']):
                    raise ValueError('Bias score manifest differs from controlled score evidence')
                retained_path = manifest_path.parent/entry['ranks_path']
                if sha(retained_path) != entry['ranks_sha256']:
                    raise ValueError('Controlled retained ranks changed')
                # The shared evaluator must preserve the controlled calibration's
                # actual ranks, including numerical ties, before adding sensitivities.
                with np.load(retained_path, allow_pickle=False) as retained:
                    if (retained['genes'].astype(str).tolist() != [r['g'] for r in rows]
                            or retained['diseases'].astype(str).tolist() != [r['d'] for r in rows]
                            or not np.array_equal(retained['seen'], [r['seen'] for r in rows])):
                        raise ValueError('Bias and controlled evaluation row identities differ')
                    for mode, suffix in [('raw','r_s0'),('calibrated','r_s1')]:
                        if not np.array_equal(retained[mode], evaluated[f"{model['name']}::{suffix}"]):
                            raise ValueError('Bias evaluator changes controlled raw/calibrated ranks')
        hashes[str(fold)] = dict(ranks_sha256=sha(path), rows_sha256=sha(row_path))
    for model in manifest['models']:
        for mode, condition in [('raw','S0_Raw_All'),('calibrated','S1_Cal_All')]:
            key = f"{model['initialization']}/{model['arm']}/{model['checkpoint_kind']}/{mode}"
            reference = controlled['conditions'][key]['per_seed'][str(model['seed'])]
            actual = bias[model['name']][condition]
            expected_metrics = dict(h1=reference['hits@1']*100, h10=reference['hits@10']*100,
                                    mrr=reference['mrr'], mr=reference['mr'], n=controlled['associations'])
            if any(actual[k] != v for k,v in expected_metrics.items()):
                raise ValueError('Bias and controlled primary aggregates differ')
    return dict(status='passed', models=48, score_files=480, conditions_per_model=42,
        rank_arrays_equal=960, scope='Shared bias sensitivities retain every controlled raw/calibrated rank and aggregate; primary contrast family is unchanged',
        score_manifest_sha256=sha(manifest_path), controlled_report_sha256=sha(controlled_report),
        bias_report_sha256=sha(bias_report), metadata_sha256=sha(meta_path), ranks=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('build','verify'):
        p = sub.add_parser(action)
        p.add_argument('--controlled-report', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if action == 'build':
            p.add_argument('--root', type=Path, required=True)
        else:
            for name in ('score-manifest','bias-report','ranks-dir'):
                p.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output path')
    result = build(args.root, args.controlled_report, args.output) if args.action == 'build' else verify(
        args.score_manifest, args.controlled_report, args.bias_report, args.ranks_dir)
    write(args.output, result)


if __name__ == '__main__': main()
