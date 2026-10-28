import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_baseline_manifest import fixture as baseline_fixture, manifest as builder

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_tables', ROOT / 'reproducibility/generate_baseline_tables.py')
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def inputs(tmp_path):
    data, roots = baseline_fixture(tmp_path)
    manifest = builder.build(data, roots, tmp_path / 'manifest.json')
    old = json.loads((ROOT / 'reproducibility/results/stepwise-bias-report-v4.json').read_text())
    # Actual evaluation schema/populations, synthetic performance values. Never paper evidence.
    template = next(iter(old.values()))
    report = {}
    for index, model in enumerate(manifest['models']):
        report[model['name']] = {condition: dict(h1=.25, h10=index + .125, mrr=.04,
            mr=1234 + index, med=987 + index, n=value['n']) for condition, value in template.items()}
    # First model's H@1 must not exceed its deliberately distinctive H@10.
    for value in next(iter(report.values())).values():
        value['h1'] = .1
    metadata = dict(models=list(report), folds=list(range(10)), cohorts=tables.COHORTS,
                    score_manifest_sha256='fixture-manifest')
    return report, metadata, manifest


def test_every_completed_variant_and_condition_is_exported(tmp_path):
    report, meta, manifest = inputs(tmp_path)
    outputs = tables.render(report, meta, manifest, 'fixture-manifest', 'fixture-report')
    rows = list(csv.DictReader(io.StringIO(outputs['final-baseline-metrics.csv'])))
    assert len(rows) == 26 * 42
    assert {(r['model'], r['condition']) for r in rows} == {
        (model, condition) for model, values in report.items() for condition in values}
    for row in rows:
        assert {metric: float(row[metric]) for metric in tables.METRICS} == report[row['model']][row['condition']]
    phrank = outputs['final-baseline-phrank.tex']
    phenorank = outputs['final-baseline-phenorank.tex']
    assert 'Native HPO, direct & 0.12 / 1234.0' in phrank
    assert 'Native HPO, disease-max' in phrank and 'Human-only UPheno' in phrank
    assert 'Native 2016, simulation' in phenorank and 'Native 2016, continuous' in phenorank
    assert phrank.count(' & ') == 13 * 6
    assert phenorank.count(' & ') == 15 * 6
    assert 'Report SHA256: fixture-report' in phrank


@pytest.mark.parametrize('problem', ['partial', 'missing-model', 'old-label', 'fold',
    'manifest-hash', 'cohort', 'condition', 'nan', 'population', 'metric-range'])
def test_rejects_incomplete_mismatched_or_invalid_results(tmp_path, problem):
    report, meta, manifest = inputs(tmp_path)
    key = next(iter(report))
    if problem == 'partial': manifest['all_baseline_components'] = False
    elif problem == 'missing-model': report.pop(key)
    elif problem == 'old-label': report['PhenoRank (Static 2016 1000-Sim)'] = report.pop(key)
    elif problem == 'fold': manifest['models'][0]['folds'].pop('9')
    elif problem == 'manifest-hash': meta['score_manifest_sha256'] = 'different'
    elif problem == 'cohort': meta['cohorts'] = {**meta['cohorts'], 'pairs': 6500}
    elif problem == 'condition': report[key].pop('S1_EqualProfile_All')
    elif problem == 'nan': report[key]['S0_Raw_All']['mr'] = float('nan')
    elif problem == 'population': report[key]['S4_Cumulative_All4']['n'] = 4265
    elif problem == 'metric-range': report[key]['S0_Raw_All']['h10'] = 101
    with pytest.raises(ValueError):
        tables.render(report, meta, manifest, 'fixture-manifest', 'fixture-report')


def test_cli_records_hashes_and_rejects_changed_existing_outputs(tmp_path):
    report, meta, manifest = inputs(tmp_path)
    manifest_path, report_path = tmp_path / 'manifest.json', tmp_path / 'report.json'
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(report))
    meta['score_manifest_sha256'] = tables.sha(manifest_path.read_bytes())
    report_path.with_suffix('.metadata.json').write_text(json.dumps(meta))
    output = tmp_path / 'tables'
    command = [sys.executable, str(ROOT / 'reproducibility/generate_baseline_tables.py'),
               '--score-manifest', str(manifest_path), '--report-json', str(report_path),
               '--out-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run(['bash', str(ROOT / 'reproducibility/run_pipeline.sh'),
                    'generate-baseline-tables', *command[2:]], check=True, capture_output=True,
                   env={**os.environ, 'PYTHON_BIN': sys.executable})
    proof = json.loads((output / 'final-baseline-export.json').read_text())
    assert proof['report_sha256'] == tables.sha(report_path.read_bytes())
    for file, digest in proof['outputs'].items():
        assert tables.sha((output / file).read_bytes()) == digest
    subprocess.run(command, check=True, capture_output=True)
    target = output / 'final-baseline-phrank.tex'
    target.write_text('existing different result')
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode != 0 and b'Existing output differs' in failed.stderr
    assert target.read_text() == 'existing different result'
