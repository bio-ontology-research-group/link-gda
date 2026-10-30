import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from phenoquery.common import read, sha, write
from analysis.summarize_controlled_query import summarize
from test_controlled_query_report import make_campaign

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('controlled_bias', ROOT/'reproducibility/build_controlled_bias_manifest.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    base = tmp_path_factory.mktemp('controlled-bias')
    root = make_campaign(base)
    write(base/'controlled.json', summarize(root, expected_candidates=17))
    manifest = root/'bias-manifest.json'
    write(manifest, m.build(root, base/'controlled.json', manifest, expected_candidates=17))
    result = subprocess.run([sys.executable,str(ROOT/'code/analysis/evaluate_stepwise_bias.py'),
        '--base-dir',str(root/'ultra_4g-seed0'), '--score-manifest',str(manifest),
        '--out-json',str(base/'bias.json'), '--ranks-dir',str(base/'bias-ranks'), '--workers','2'],
        capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    return base


def verify(base):
    return m.verify(base/'campaign/bias-manifest.json',base/'controlled.json',base/'bias.json',base/'bias-ranks')


def test_all_variants_all_conditions_and_rank_parity(evaluated):
    proof = verify(evaluated)
    assert proof['models'] == 48 and proof['score_files'] == 480
    assert proof['conditions_per_model'] == 42 and proof['rank_arrays_equal'] == 960
    report = read(evaluated/'bias.json')
    for model in report.values():
        assert {'S1_ReferenceDedup_All','S1_PositiveFilter_All','S1_EqualProfile_All',
                'S1_NoTrainingProfileMatch','S4_Cumulative_All4'} <= set(model)


def test_relocation_preserves_manifest_and_checks(evaluated,tmp_path):
    original = tmp_path/'original'; shutil.copytree(evaluated,original)
    restored = tmp_path/'restored'; shutil.copytree(original,restored)
    shutil.rmtree(original)
    manifest = restored/'campaign/bias-manifest.json'
    assert read(manifest) == m.build(restored/'campaign',restored/'controlled.json',manifest,expected_candidates=17)
    assert verify(restored) == verify(evaluated)


@pytest.mark.parametrize('problem',['missing-model','condition','replaced-condition','score','benchmark','row-profile','rank-pairing','aggregate','manifest-score'])
def test_inconsistent_bias_analysis_is_rejected(evaluated,tmp_path,problem):
    base = tmp_path/'copy'; shutil.copytree(evaluated,base)
    manifest_path = base/'campaign/bias-manifest.json'; manifest = read(manifest_path)
    first = manifest['models'][0]
    if problem == 'missing-model':
        manifest['models'].pop(); write(manifest_path,manifest)
    elif problem in ('condition','replaced-condition','aggregate'):
        path=base/'bias.json';report=read(path)
        if problem=='condition':report[first['name']].pop('S1_ReferenceDedup_All')
        elif problem=='replaced-condition':
            report[first['name']]['unsupported-condition']=report[first['name']].pop('S1_ReferenceDedup_All')
        else:report[first['name']]['S0_Raw_All']['mr']+=1
        write(path,report)
    elif problem=='score':
        (manifest_path.parent/first['folds']['0']['path']).write_bytes(b'changed')
    elif problem=='benchmark':
        path=manifest_path.parent/manifest['data_root']/'fold0/profiles.json'
        profiles=read(path);profiles['test0a']=['different'];write(path,profiles)
    elif problem=='row-profile':
        path=base/'bias-ranks/fold0-rows.json';rows=read(path);rows[0]['prof_tuple']=['different'];write(path,rows)
    elif problem=='rank-pairing':
        path=base/'bias-ranks/fold0-ranks.npz'
        with np.load(path) as z: arrays={k:z[k] for k in z.files}
        ranks=arrays[first['name']+'::r_s0'];index=np.flatnonzero(ranks!=ranks[0])[0]
        ranks[0],ranks[index]=ranks[index],ranks[0]
        np.savez_compressed(path,**arrays)
    else:
        first['folds']['0']['sha256']='incorrect';write(manifest_path,manifest)
        path=base/'bias.metadata.json';meta=read(path);meta['score_manifest_sha256']=sha(manifest_path);write(path,meta)
    with pytest.raises(ValueError):verify(base)


def test_build_requires_verified_complete_controlled_report(evaluated,tmp_path):
    base=tmp_path/'copy';shutil.copytree(evaluated,base)
    report=read(base/'controlled.json');report['associations']+=1;write(base/'controlled.json',report)
    with pytest.raises(ValueError,match='reference report differs'):
        m.build(base/'campaign',base/'controlled.json',base/'new.json',expected_candidates=17)
