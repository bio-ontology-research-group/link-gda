"""Primary statistics must survive inference replay, not just table rounding."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from analysis.summarize_controlled_query import summarize
from phenoquery.common import read, sha, write
from test_controlled_query_report import make_campaign

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reproducibility'))
from compare_controlled_query_inference import compare as compare_scores
from compare_controlled_query_reports import compare


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp('primary-replay')
    original = make_campaign(root)
    replay = root/'replay'
    shutil.copytree(original,replay)
    # Independent archives may encode identical arrays differently. Recompute
    # their completion record and both reports using the real verification code.
    bundle = replay/'ultraquery-seed2/test/fold9/typed_and'
    for name in ('scores.npz','ranks.npz'):
        with np.load(bundle/name,allow_pickle=False) as source:
            arrays = {k:source[k].copy() for k in source.files}
        np.savez(bundle/name,**arrays)
    complete = read(bundle/'COMPLETE.json')
    complete['files'] = {n:sha(bundle/n) for n in ('scores.npz','ranks.npz')}
    write(bundle/'COMPLETE.json',complete)
    reports = [summarize(p,expected_candidates=17) for p in (original,replay)]
    proof = compare_scores(original,replay,expected_candidates=17)
    return reports,proof


@pytest.fixture
def files(evidence,tmp_path):
    reports,proof = copy.deepcopy(evidence)
    paths = [tmp_path/n for n in ('reference.json','replay.json','inference.json')]
    for path,record in zip(paths,[*reports,proof]): write(path,record)
    return paths


def check(paths):
    return compare(*paths,reference_sha256=sha(paths[0]))


def test_recomputed_reports_allow_only_bound_output_digest_changes(files):
    result = check(files)
    assert result['status'] == 'passed' and result['report_differences'] == []
    assert result['conditions'] == 32 and result['primary_contrasts'] == 8
    assert {d['field'] for d in result['verified_output_digest_changes']} == {
        'scores.npz','ranks.npz','completion_sha256'}
    assert len(result['verified_output_digest_changes']) == 3


@pytest.mark.parametrize('mutation',['bootstrap-endpoint','bootstrap-seed','stratum-metric',
                                    'seed-dispersion','selection','missing-metric'])
def test_unrounded_or_unprinted_changes_require_review(files,mutation):
    report = read(files[1])
    condition = report['conditions']['ultra_4g/typed_and/selected/calibrated']
    if mutation == 'bootstrap-endpoint': report['primary_contrasts'][7]['lower_percentage_points'] += 1e-9
    elif mutation == 'bootstrap-seed': report['inference']['seed'] += 1
    elif mutation == 'stratum-metric': condition['strata_per_seed']['unseen']['2']['mr'] += 1e-8
    elif mutation == 'seed-dispersion': condition['sd_across_seeds']['mrr'] += 1e-8
    elif mutation == 'selection': report['selected_steps']['ultraquery-seed2/fold9/typed_and/selected'] = 200
    else: condition['per_seed']['2'].pop('mrr')
    write(files[1],report)
    result = check(files)
    assert result['status'] == 'requires_review' and len(result['report_differences']) == 1


@pytest.mark.parametrize('mutation',['unbound-score','changed-training','benchmark','missing-condition',
                                    'missing-contrast','duplicate-bundle','failed-score-proof','hidden-failed-bundle',
                                    'source-drift','nonfinite'])
def test_rejects_broken_provenance_and_incomplete_coverage(files,mutation):
    report,proof = read(files[1]),read(files[2])
    key = 'ultraquery-seed2/fold9/typed_and/selected'
    if mutation == 'unbound-score': report['input_sha256'][key]['scores.npz'] = 'a'*64
    elif mutation == 'changed-training':
        report['input_sha256'][key]['training_identity_sha256'] = 'a'*64
        next(r for r in proof['records'] if r['bundle']=='ultraquery-seed2/test/fold9/typed_and')['replay_evidence']['training_identity_sha256'] = 'a'*64
    elif mutation == 'benchmark': report['input_sha256']['data/fold9']['test.tsv'] = 'a'*64
    elif mutation == 'missing-condition': report['conditions'].pop('ultraquery/typed_and/initial/raw')
    elif mutation == 'missing-contrast': report['primary_contrasts'].pop()
    elif mutation == 'duplicate-bundle': proof['records'][-1] = copy.deepcopy(proof['records'][0])
    elif mutation == 'failed-score-proof': proof['status'] = 'requires_review'
    elif mutation == 'hidden-failed-bundle': proof['records'][-1]['ranks']['raw']['changed_ranks'] = 1
    elif mutation == 'source-drift': proof['comparator_sha256'] = 'a'*64
    else: report['primary_contrasts'][0]['lower_percentage_points'] = float('nan')
    # Deliberately bypass the normal writer's NaN rejection to test malformed input.
    files[1].write_text(json.dumps(report)); write(files[2],proof)
    with pytest.raises(ValueError): check(files)


def test_verified_reference_digest_is_required(files):
    digest = sha(files[0])
    report = read(files[0]); report['primary_contrasts'][0]['difference_percentage_points'] += .1
    write(files[0],report)
    with pytest.raises(ValueError,match='verified analysis digest'):
        compare(*files,reference_sha256=digest)


def test_rejects_same_or_hardlinked_report(files):
    with pytest.raises(ValueError,match='separate'):
        compare(files[0],files[0],files[2],sha(files[0]))
    files[1].unlink(); files[1].hardlink_to(files[0])
    with pytest.raises(ValueError,match='separate'): check(files)


def test_cli_retains_failed_comparison_and_refuses_overwrite(files):
    report = read(files[1]); report['primary_contrasts'][0]['upper_percentage_points'] += 1e-8
    write(files[1],report)
    output = files[0].parent/'result.json'
    command = [sys.executable,str(REPO/'reproducibility/compare_controlled_query_reports.py'),
               '--reference-report',str(files[0]),'--replay-report',str(files[1]),
               '--inference-comparison',str(files[2]),'--reference-sha256',sha(files[0]),
               '--output',str(output)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 1 and read(output)['status'] == 'requires_review'
    digest = sha(output)
    again = subprocess.run(command,capture_output=True,text=True)
    assert again.returncode != 0 and sha(output) == digest
