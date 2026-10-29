import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reproducibility'))
spec = importlib.util.spec_from_file_location('production_bias_tables', ROOT/'reproducibility/generate_production_bias_tables.py')
tables = importlib.util.module_from_spec(spec); spec.loader.exec_module(tables)


def inputs(kind):
    models = []
    if kind == 'controlled':
        for initialization in ('ultra_4g','ultraquery'):
            for seed in range(3):
                for arm in ('bridge_pool','plain_pool','plain_and','typed_and'):
                    for checkpoint in ('initial','selected'):
                        models.append(dict(name=f'ControlledQuery ({initialization}, seed{seed}, {arm}, {checkpoint})',
                            initialization=initialization, seed=seed, arm=arm, checkpoint_kind=checkpoint))
        runs, scores, protocol = 240, 480, 'controlled-query-20261028.md'
    else:
        for family, primary in [('U1','calibrated_mr'),('T1','raw_head'),('INDIGENA','raw_mr'),
                                ('CONVKB','calibrated_mr'),('PPOOL','raw_head')]:
            for graph in ('original','nobridge'):
                for choice in ('raw_mr','raw_head','calibrated_mr','calibrated_head'):
                    models.append(dict(name=f'MatchedKGE ({family}, {graph}, {choice})',
                        family=family, graph=graph, validation_choice=choice, primary=choice == primary))
        runs, scores, protocol = 100, 400, 'matched-kge-20261029.md'
    for model in models: model['folds'] = {str(f):{} for f in range(10)}
    manifest = dict(schema=1, training_runs=runs, score_files=scores, models=models,
        folds=list(range(10)), protocol_sha256=tables.sha((ROOT/'reproducibility/protocols'/protocol).read_bytes()))
    old = json.loads((ROOT/'reproducibility/results/stepwise-bias-report-v4.json').read_text())
    populations = {condition:value['n'] for condition,value in next(iter(old.values())).items()}
    # Real benchmark schema/populations, distinctive synthetic values. Never paper evidence.
    report = {m['name']:{condition:dict(h1=.1,h10=index+.125,mrr=.04,mr=1234+index,
        med=987+index,n=n) for condition,n in populations.items()} for index,m in enumerate(models)}
    meta = dict(models=list(report), folds=list(range(10)), cohorts=dict(tables.COHORTS),
        score_manifest_sha256='fixture-manifest',
        evaluation_sha256=tables.sha((ROOT/'code/analysis/evaluate_stepwise_bias.py').read_bytes()))
    return report, meta, manifest


@pytest.mark.parametrize('kind,count,panels',[('controlled',48,6),('matched',40,5)])
def test_every_seed_choice_condition_and_population_is_preserved(kind,count,panels):
    report, meta, manifest = inputs(kind)
    actual_kind, outputs = tables.render(report,meta,manifest,'fixture-manifest','fixture-report')
    assert actual_kind == kind and len(outputs) == panels+1
    rows = list(csv.DictReader(io.StringIO(outputs[f'{kind}-bias-metrics.csv'])))
    assert len(rows) == count*42
    assert {(r['model'],r['condition']) for r in rows} == {
        (name,condition) for name,values in report.items() for condition in values}
    models = {m['name']:m for m in manifest['models']}
    for row in rows:
        assert {metric:float(row[metric]) for metric in tables.METRICS} == report[row['model']][row['condition']]
        assert row['population_unit'] == ('profiles' if row['condition'] in tables.PROFILE_CONDITIONS else 'association_pairs')
        keys = ('initialization','seed','arm','checkpoint_kind') if kind == 'controlled' else ('family','graph','validation_choice','primary')
        assert all(row[key] == str(models[row['model']][key]) for key in keys)
    text = '\n'.join(value for name,value in outputs.items() if name.endswith('.tex'))
    assert text.count('Panel A: scoring') == panels and text.count('Panel B: individual') == panels
    assert text.count('Report SHA256: fixture-report') == panels
    for index in range(count):
        assert text.count(f'{index+.125:.2f} / {1234+index:.1f}') == 11
    if kind == 'controlled':
        assert all(f'seed {seed}:' in text for seed in range(3))
    else:
        assert text.count(r'($\ast$)') == 20  # two primary graphs in both displayed panels


@pytest.mark.parametrize('kind',['controlled','matched'])
@pytest.mark.parametrize('problem',['missing-model','duplicate-model','identity','fold','manifest',
    'evaluator','protocol','cohort','condition','population-unit','nonfinite','range','metric'])
def test_invalid_production_reports_are_rejected(kind,problem):
    report, meta, manifest = inputs(kind); name = next(iter(report))
    if problem == 'missing-model': report.pop(name)
    elif problem == 'duplicate-model': manifest['models'][-1] = manifest['models'][0]
    elif problem == 'identity':
        manifest['models'][0]['seed' if kind=='controlled' else 'primary'] = 'wrong'
    elif problem == 'fold': manifest['models'][0]['folds'].pop('9')
    elif problem == 'manifest': meta['score_manifest_sha256'] = 'different'
    elif problem == 'evaluator': meta['evaluation_sha256'] = 'different'
    elif problem == 'protocol': manifest['protocol_sha256'] = 'different'
    elif problem == 'cohort': meta['cohorts']['pairs'] -= 1
    elif problem == 'condition': report[name]['unsupported'] = report[name].pop('S3_EqualProfile')
    elif problem == 'population-unit': report[name]['S4_Cumulative_All4']['n'] = 4265
    elif problem == 'nonfinite': report[name]['S0_Raw_All']['mr'] = float('nan')
    elif problem == 'range': report[name]['S0_Raw_All']['h10'] = 101
    else: report[name]['S0_Raw_All'].pop('med')
    with pytest.raises(ValueError): tables.render(report,meta,manifest,'fixture-manifest','fixture-report')


@pytest.mark.parametrize('kind',['controlled','matched'])
def test_cli_hashes_idempotence_and_overwrite_guard(tmp_path,kind):
    report,meta,manifest = inputs(kind)
    report_path, manifest_path = tmp_path/'report.json', tmp_path/'manifest.json'
    report_path.write_text(json.dumps(report)); manifest_path.write_text(json.dumps(manifest))
    meta['score_manifest_sha256'] = tables.sha(manifest_path.read_bytes())
    report_path.with_suffix('.metadata.json').write_text(json.dumps(meta))
    out = tmp_path/'output'
    command = [sys.executable,str(ROOT/'reproducibility/generate_production_bias_tables.py'),
        '--report-json',str(report_path),'--score-manifest',str(manifest_path),'--out-dir',str(out)]
    subprocess.run(command,check=True,capture_output=True)
    snapshot = {p.name:p.read_bytes() for p in out.iterdir()}
    proof = json.loads(snapshot[f'{kind}-bias-export.json'])
    assert proof['report_sha256'] == tables.sha(report_path.read_bytes())
    assert proof['metadata_sha256'] == tables.sha(report_path.with_suffix('.metadata.json').read_bytes())
    assert proof['manifest_sha256'] == tables.sha(manifest_path.read_bytes())
    assert all(tables.sha(snapshot[name]) == digest for name,digest in proof['outputs'].items())
    subprocess.run(command,check=True,capture_output=True)
    assert {p.name:p.read_bytes() for p in out.iterdir()} == snapshot
    changed = out/f'{kind}-bias-metrics.csv'; changed.write_text('different result')
    failed = subprocess.run(command,capture_output=True)
    assert failed.returncode != 0 and b'Existing output differs' in failed.stderr
    assert changed.read_text() == 'different result'
    assert all((out/name).read_bytes() == data for name,data in snapshot.items() if name != changed.name)
