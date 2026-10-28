import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from analysis import summarize_matched_kge as summary
from phenoquery.common import read, sha, write


@pytest.fixture(scope='module')
def evaluated_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('matched-kge-evaluated')
    models = [dict(name=f'MatchedKGE ({family}, {graph}, {choice})', family=family,
                   graph=graph, choice=choice, folds={}) for family in summary.MODELS
              for graph in summary.GRAPHS for choice in summary.CHOICES]
    manifest = dict(training_runs=100, score_files=400, folds=list(range(10)), models=models,
        primary_comparisons=summary.primary_spec(), evidence={}, benchmark={}, settings={},
        protocol_sha256=sha(summary.ROOT / 'reproducibility/protocols/matched-kge-20261029.md'),
        freeze_sha256='synthetic frozen campaign; not scientific results')
    rng = np.random.default_rng(127)
    genes = [f'g{i}' for i in range(17)]
    for fold in range(10):
        data = root / f'workspace/data/fold{fold}'; (data / 'raw').mkdir(parents=True)
        splits = dict(training=[('g0',f'train{fold}a'),('g1',f'train{fold}b')],
                      validation=[('g0',f'val{fold}a'),('g3',f'val{fold}b')],
                      test=[('g0',f'test{fold}a'),('g5',f'test{fold}a'),('g2',f'test{fold}b'),('g3',f'test{fold}c')])
        for split, rows in splits.items():
            (data / (split + '.tsv')).write_text('Gene\tDisease\n' + ''.join(f'{g}\t{d}\n' for g,d in rows))
        (data / 'candidates.txt').write_text('\n'.join(genes)+'\n')
        (data / 'raw/train.txt').write_text('synthetic graph input\n')
        profiles = {f'train{fold}a':['p0'], f'train{fold}b':['p1'], f'val{fold}a':['p3'],
                    f'val{fold}b':['p4'], f'test{fold}a':['p0'], f'test{fold}b':['p6'], f'test{fold}c':['p7']}
        write(data / 'profiles.json', profiles)
        manifest['benchmark'][str(fold)] = {name:sha(data/name) for name in summary.INPUTS}
        for model in models:
            path = root / f"scores/fold{fold}/{model['family']}-{model['graph']}-{model['choice']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, scores=rng.integers(0,6,(3,17)).astype(float),
                                genes=genes, diseases=[f'test{fold}{s}' for s in 'abc'])
            model['folds'][str(fold)] = dict(path=str(path.relative_to(root)), sha256=sha(path))
        for family in summary.MODELS:
            manifest['settings'][family] = dict(dimension=400 if family == 'INDIGENA' else 200)
            for graph in summary.GRAPHS:
                manifest['evidence'][f'fold{fold}/{graph}/{family}'] = dict(
                    selected={choice:dict(epoch=1 if family=='PPOOL' else 20) for choice in summary.CHOICES},
                    nodes=19 if graph=='original' else 17, forward_triples=40 if graph=='original' else 36,
                    epochs=20 if family=='PPOOL' else 1000)
    write(root / 'manifest.json', manifest)
    command = [sys.executable, str(summary.ROOT / 'code/analysis/evaluate_stepwise_bias.py'),
        '--base-dir', str(root/'workspace'), '--score-manifest', str(root/'manifest.json'),
        '--out-json', str(root/'evaluation.json'), '--ranks-dir', str(root/'ranks'), '--workers', '2']
    result = subprocess.run(command, capture_output=True, text=True,
        env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}, timeout=120)
    assert result.returncode==0, result.stdout+'\n'+result.stderr
    return root


def run(root):
    return summary.summarize(root/'manifest.json', root/'evaluation.json', root/'ranks', root/'workspace/data', expected_candidates=17)


def test_real_evaluator_replay_and_independent_paired_bootstrap(evaluated_fixture):
    root=evaluated_fixture; result=run(root)
    assert result['associations']==40 and result['profile_clusters']==3
    assert result['verified_primary_score_files']==100 and len(result['contrasts'])==10
    assert result['inference']['confidence_level']==.995 and result['inference']['replicates']==10000
    arrays=[]
    for family, spec in summary.primary_spec().items():
        sides=[]
        for graph in summary.GRAPHS:
            values=[]
            for fold in range(10):
                with np.load(root/f'ranks/fold{fold}-ranks.npz') as z:
                    values.extend(z[spec[graph]+('::r_s1' if spec['scoring']=='calibrated' else '::r_s0')])
            sides.append(np.asarray(values))
        original,nobridge=sides
        arrays.extend([(nobridge<=10).astype(float)*100-(original<=10)*100,nobridge-original])
    differences=np.stack(arrays,axis=1)
    members=[np.array([i for i in range(40) if i%4 in (0,1)]),np.arange(2,40,4),np.arange(3,40,4)]
    draws=np.random.default_rng(20261029).integers(0,3,size=(10000,3))
    independent=np.stack([differences[np.concatenate([members[i] for i in draw])].mean(0) for draw in draws])
    np.testing.assert_allclose([c['difference'] for c in result['contrasts']],differences.mean(0),rtol=0,atol=0)
    np.testing.assert_allclose([c['lower'] for c in result['contrasts']],np.quantile(independent,.0025,axis=0))
    np.testing.assert_allclose([c['upper'] for c in result['contrasts']],np.quantile(independent,.9975,axis=0))
    assert result['primary']['U1']['choice']=='calibrated_mr'
    assert result['primary']['INDIGENA']['scoring']=='raw'


def test_table_export_preserves_primary_values_and_actual_budgets(evaluated_fixture):
    spec=importlib.util.spec_from_file_location('matched_tables',summary.ROOT/'reproducibility/generate_matched_kge_tables.py')
    table=importlib.util.module_from_spec(spec);spec.loader.exec_module(table)
    result=run(evaluated_fixture)
    outputs=table.render(result,'fixture-summary')
    main=outputs['matched-kge-primary.tex']; design=outputs['matched-kge-design.tex']
    for family, record in result['primary'].items():
        assert f"{table.LABELS[family]} ({record['scoring']})" in main
        for graph in summary.GRAPHS:
            stats=record['graphs'][graph]['metrics']
            assert f"{stats['h10']:.2f}" in main and f"{stats['mr']:.1f}" in main
    assert main.count(' & ')==6*6 and design.count(' & ')==11*6
    assert 'P-pool & Original & 200 & 19 & 40 & 20 & 1' in design
    assert 'U1 & Original & 200 & 19 & 40 & 1,000 & 20' in design
    assert '99.5' in main and 'percentage points' in main
    result['inference']['confidence_level']=.95
    with pytest.raises(ValueError):table.render(result,'wrong-interval')


@pytest.mark.parametrize('problem',['paired-ranks','row-order','score-bytes','primary-choice','manifest-digest',
                                   'aggregate','cohort','test-input','partial-models'])
def test_rejects_bad_pairing_provenance_or_test_selection(evaluated_fixture,tmp_path,problem):
    root=tmp_path/'copy';shutil.copytree(evaluated_fixture,root)
    if problem=='paired-ranks':
        p=root/'ranks/fold0-ranks.npz'
        with np.load(p) as z: arrays={k:z[k] for k in z.files}
        key=summary.primary_spec()['U1']['original']+'::r_s1'
        before=arrays[key].copy()
        other=np.flatnonzero(before!=before[0])[0]
        arrays[key][0],arrays[key][other]=before[other],before[0]
        np.testing.assert_array_equal(np.sort(arrays[key]),np.sort(before))
        np.savez_compressed(p,**arrays)
    elif problem=='row-order':
        p=root/'ranks/fold0-rows.json';rows=read(p);rows[0],rows[1]=rows[1],rows[0];write(p,rows)
    elif problem=='score-bytes':
        m=read(root/'manifest.json');name=summary.primary_spec()['U1']['original']
        model=next(v for v in m['models'] if v['name']==name)
        (root/model['folds']['0']['path']).write_bytes(b'changed')
    elif problem in ('primary-choice','partial-models'):
        p=root/'manifest.json';m=read(p)
        if problem=='primary-choice':m['primary_comparisons']['U1']['scoring']='raw'
        else:m['models'].pop()
        write(p,m)
    elif problem=='aggregate':
        p=root/'evaluation.json';d=read(p);d[summary.primary_spec()['U1']['original']]['S1_Cal_All']['mr']+=1;write(p,d)
    elif problem=='test-input':(root/'workspace/data/fold0/test.tsv').write_text('changed')
    else:
        p=root/'evaluation.metadata.json';d=read(p)
        if problem=='manifest-digest':d['score_manifest_sha256']='wrong'
        else:d['cohorts']['pairs']+=1
        write(p,d)
    with pytest.raises(ValueError):run(root)
