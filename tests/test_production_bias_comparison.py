import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from phenoquery.common import read, sha, write
from test_production_bias_tables import inputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bias_comparison',ROOT/'reproducibility/compare_production_bias_replays.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture(scope='module',params=['matched','controlled'])
def collection(tmp_path_factory,request):
    root = tmp_path_factory.mktemp('bias-comparison-'+request.param)
    report,metadata,manifest = inputs(request.param)
    metadata['schema_version'] = 1
    reference = root/'reference'; reference.mkdir()
    write(reference/'manifest.json',manifest)
    metadata['score_manifest_sha256'] = sha(reference/'manifest.json')
    write(reference/'evaluation.json',report); write(reference/'evaluation.metadata.json',metadata)
    ranks = reference/'ranks'; ranks.mkdir()
    # Complete production schema with synthetic ranks. This checks comparison,
    # not scientific correctness or inference; no fixture is paper evidence.
    for fold in range(10):
        count = 658 if fold == 0 else 657
        rows = [dict(fold=fold,g=f'g{i}',d=f'd{fold}-{i}',gi=i,di=i,seen=False,
                     j_same_0=True,not_dup_prof=True,prof_tuple=['fixture'],multi_gene=False)
                for i in range(count)]
        write(ranks/f'fold{fold}-rows.json',rows)
        arrays = {name+'::'+key:np.arange(count,dtype=np.float64)+1
                  for name in report for key in m.RANK_KEYS}
        np.savez_compressed(ranks/f'fold{fold}-ranks.npz',**arrays)
    replica = root/'replay'; shutil.copytree(reference,replica)
    manifest['fixture_location'] = 'independently regenerated score collection'
    write(replica/'manifest.json',manifest)
    metadata['score_manifest_sha256'] = sha(replica/'manifest.json')
    write(replica/'evaluation.metadata.json',metadata)
    return root,request.param


def compare(root):
    return m.compare(root/'reference/manifest.json',root/'replay/manifest.json',
        root/'reference/evaluation.json',root/'replay/evaluation.json',
        root/'reference/ranks',root/'replay/ranks')


def alter_rank(path,change):
    with np.load(path) as z: arrays = {n:z[n] for n in z.files}
    name = next(n for n in arrays if n.endswith('::r_s2_positive_filter'))
    if change == 'rank': arrays[name][0] += .5
    elif change == 'dtype': arrays[name] = arrays[name].astype(np.float32)
    elif change == 'shape': arrays[name] = arrays[name][:,None]
    elif change == 'missing-array': arrays.pop(name)
    elif change == 'nonfinite': arrays[name][0] = np.nan
    np.savez_compressed(path,**arrays)


def test_complete_comparison_allows_distinct_manifest_hashes(collection):
    root,kind = collection; result = compare(root)
    assert result['status'] == 'passed' and result['kind'] == kind
    assert result['variants'] == (40 if kind == 'matched' else 48)
    assert result['rank_arrays'] == result['variants']*14*10
    assert result['rank_values'] == result['variants']*14*6571
    assert result['conditions_per_variant'] == 42 and result['association_pairs'] == 6571
    assert result['changed_ranks'] == 0 and result['metric_differences'] == []


def test_recompressed_and_reordered_npz_members_are_accepted(collection,tmp_path):
    source,_ = collection; root = tmp_path/'copy'; shutil.copytree(source,root)
    path = root/'replay/ranks/fold0-ranks.npz'
    with zipfile.ZipFile(path) as z: entries = [(n,z.read(n)) for n in z.namelist()]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for n,payload in reversed(entries): z.writestr(n,payload)
    assert compare(root)['status'] == 'passed'


@pytest.mark.parametrize('change',['rank','metric'])
def test_nonprimary_drift_requires_review(collection,tmp_path,change):
    source,_ = collection; root = tmp_path/'copy'; shutil.copytree(source,root)
    if change == 'rank': alter_rank(root/'replay/ranks/fold0-ranks.npz','rank')
    else:
        path = root/'replay/evaluation.json'; report = read(path)
        report[next(iter(report))]['S4_Cumulative_All4']['h10'] += .001
        write(path,report)
    result = compare(root)
    assert result['status'] == 'requires_review'
    assert result['changed_ranks'] == (1 if change == 'rank' else 0)
    assert len(result['metric_differences']) == (1 if change == 'metric' else 0)


@pytest.mark.parametrize('change',['dtype','shape','missing-array','nonfinite','missing-fold',
                                  'row-order','manifest','evaluator','population','shared-ranks'])
def test_incomplete_or_inconsistent_artifacts_are_rejected(collection,tmp_path,change):
    source,_ = collection; root = tmp_path/'copy'; shutil.copytree(source,root)
    path = root/'replay/ranks/fold0-ranks.npz'
    if change in ('dtype','shape','missing-array','nonfinite'):
        alter_rank(path,change)
        if change == 'missing-array':
            # A common omission in both sides must not silently pass.
            alter_rank(root/'reference/ranks/fold0-ranks.npz',change)
    elif change == 'missing-fold': path.unlink()
    elif change == 'row-order':
        path = root/'replay/ranks/fold0-rows.json'; rows = read(path)
        rows[0],rows[1] = rows[1],rows[0]; write(path,rows)
    elif change in ('manifest','evaluator'):
        path = root/'replay/evaluation.metadata.json'; metadata = read(path)
        metadata['score_manifest_sha256' if change == 'manifest' else 'evaluation_sha256'] = '0'*64
        write(path,metadata)
    elif change == 'population':
        path = root/'replay/evaluation.json'; report = read(path)
        report[next(iter(report))]['S4_Cumulative_All4']['n'] += 1; write(path,report)
    else:
        path.unlink(); path.hardlink_to(root/'reference/ranks/fold0-ranks.npz')
    with pytest.raises(ValueError): compare(root)


def test_same_collection_is_rejected(collection):
    root,_ = collection; reference = root/'reference'
    with pytest.raises(ValueError,match='separate'):
        m.compare(reference/'manifest.json',reference/'manifest.json',reference/'evaluation.json',
            reference/'evaluation.json',reference/'ranks',reference/'ranks')


def test_cli_retains_drift_report_and_rejects_overwrite(collection,tmp_path):
    source,_ = collection; root = tmp_path/'copy'; shutil.copytree(source,root)
    alter_rank(root/'replay/ranks/fold0-ranks.npz','rank')
    output = root/'comparison.json'
    command = [sys.executable,str(ROOT/'reproducibility/compare_production_bias_replays.py')]
    for side in ('reference','replay'):
        for flag,name in [('manifest','manifest.json'),('report','evaluation.json'),('ranks','ranks')]:
            command += ['--'+side+'-'+flag,str(root/side/name)]
    command += ['--output',str(output)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 1 and read(output)['status'] == 'requires_review'
    before = output.read_bytes()
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0 and 'Choose a new comparison output path' in result.stderr
    assert output.read_bytes() == before
