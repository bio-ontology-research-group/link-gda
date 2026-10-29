"""Primary summaries must preserve statistics across independently encoded outputs."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from analysis.summarize_matched_kge import primary_spec, PROTOCOL
from phenoquery.common import read, sha, write
from test_production_bias_tables import inputs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reproducibility'))
from compare_production_bias_replays import compare as compare_bias, RANK_KEYS
from compare_matched_kge_summaries import compare


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp('matched-primary')
    report,metadata,manifest = inputs('matched')
    metadata['schema_version'] = 1
    manifest.update(primary_comparisons=primary_spec(),freeze_sha256='f'*64,
        benchmark={str(f):{'test.tsv':'a'*64} for f in range(10)})
    original = root/'reference'; original.mkdir()
    write(original/'evaluation.json',report)
    (original/'ranks').mkdir()
    for fold in range(10):
        count = 658 if fold == 0 else 657
        write(original/'ranks'/f'fold{fold}-rows.json',[dict(fold=fold,g=f'g{i}',d=f'd{fold}-{i}') for i in range(count)])
        arrays = {name+'::'+key:np.arange(count,dtype=np.float64)+1 for name in report for key in RANK_KEYS}
        np.savez_compressed(original/'ranks'/f'fold{fold}-ranks.npz',**arrays)
    replay = root/'replay'; shutil.copytree(original,replay)
    path = replay/'ranks/fold0-ranks.npz'
    with zipfile.ZipFile(path) as archive: members = [(n,archive.read(n)) for n in archive.namelist()]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,value in reversed(members): archive.writestr(name,value)
    # Production-sized schema/populations with synthetic metrics and score digests.
    # This fixture tests comparison and provenance, not model or summary correctness.
    for side,folder in enumerate((original,replay)):
        local_manifest = copy.deepcopy(manifest)
        for index,model in enumerate(local_manifest['models']):
            model['folds'] = {str(f):dict(path=f'fixture/{index}/{f}.npz',sha256=f'{side*10000+index*10+f:064x}') for f in range(10)}
        write(folder/'manifest.json',local_manifest)
        meta = {**metadata,'score_manifest_sha256':sha(folder/'manifest.json')}
        write(folder/'evaluation.metadata.json',meta)
        models = {m['name']:m for m in local_manifest['models']}
        primary = {}
        for family,spec in primary_spec().items():
            primary[family] = dict(choice=spec['choice'],scoring=spec['scoring'],graphs={})
            for graph in ('original','nobridge'):
                condition = 'S1_Cal_All' if spec['scoring']=='calibrated' else 'S0_Raw_All'
                primary[family]['graphs'][graph] = dict(metrics=report[spec[graph]][condition],
                    selected_epochs=[20]*10,nodes=[1000]*10,forward_triples=[2000]*10,epochs=[1000]*10,dimension=200)
        summary = dict(schema=1,training_runs=100,score_files=400,verified_primary_score_files=100,
            candidates=13625,associations=6571,profile_clusters=5928,primary=primary,cohorts=meta['cohorts'],
            contrasts=[dict(family=f,endpoint=e,units='fixture',difference=.125,lower=-.25,upper=.5)
                for f in primary_spec() for e in ('hits@10','mr')],
            inference=dict(confidence_level=.995,replicates=10000,seed=20261029,comparisons=10),
            provenance=dict(manifest_sha256=sha(folder/'manifest.json'),report_sha256=sha(folder/'evaluation.json'),
                metadata_sha256=sha(folder/'evaluation.metadata.json'),freeze_sha256=manifest['freeze_sha256'],
                analysis_protocol_sha256=sha(PROTOCOL),training_protocol_sha256=manifest['protocol_sha256'],
                summary_source_sha256=sha(REPO/'code/analysis/summarize_matched_kge.py'),
                evaluation_source_sha256=meta['evaluation_sha256'],bootstrap_source_sha256=sha(REPO/'code/analysis/summarize_controlled_query.py')),
            checksums={f'fold{f}':dict(rows_sha256=sha(folder/'ranks'/f'fold{f}-rows.json'),
                ranks_sha256=sha(folder/'ranks'/f'fold{f}-ranks.npz'),scores={spec[g]:models[spec[g]]['folds'][str(f)]['sha256']
                    for spec in primary_spec().values() for g in ('original','nobridge')}) for f in range(10)})
        write(folder/'summary.json',summary)
    proof = compare_bias(original/'manifest.json',replay/'manifest.json',original/'evaluation.json',
        replay/'evaluation.json',original/'ranks',replay/'ranks')
    assert proof['status']=='passed'
    write(root/'bias-comparison.json',proof)
    return root


@pytest.fixture
def files(artifacts,tmp_path):
    for side in ('reference','replay'): shutil.copyfile(artifacts/side/'summary.json',tmp_path/(side+'.json'))
    shutil.copyfile(artifacts/'bias-comparison.json',tmp_path/'bias.json')
    return [tmp_path/'reference.json',tmp_path/'replay.json',artifacts/'reference/manifest.json',
        artifacts/'replay/manifest.json',artifacts/'reference/evaluation.json',artifacts/'replay/evaluation.json',tmp_path/'bias.json']


def check(files): return compare(*files,reference_sha256=sha(files[0]))


def test_bound_manifest_score_and_rank_encoding_changes_pass(files):
    assert read(files[0])['checksums'] != read(files[1])['checksums']
    result = check(files)
    assert result['status']=='passed' and result['summary_differences']==[]
    assert result['primary_contrasts']==10 and result['primary_score_files']==100


@pytest.mark.parametrize('change',['endpoint','bootstrap-seed','metric','epoch','dimension','cohort'])
def test_unrounded_scientific_changes_require_review(files,change):
    result = read(files[1]); graph = result['primary']['U1']['graphs']['original']
    if change=='endpoint': result['contrasts'][9]['lower'] += 1e-9
    elif change=='bootstrap-seed': result['inference']['seed'] += 1
    elif change=='metric': graph['metrics']['mrr'] += 1e-9
    elif change=='epoch': graph['selected_epochs'][9] += 20
    elif change=='dimension': graph['dimension'] += 1
    else: result['cohorts']['zero_overlap_pairs'] += 1
    write(files[1],result)
    proof = check(files)
    assert proof['status']=='requires_review' and len(proof['summary_differences'])==1


@pytest.mark.parametrize('change',['missing-contrast','missing-fold','score-binding','rank-binding','row-binding',
                                 'source','failed-bias','duplicate-fold','hidden-rank-change','artifact-hash','nonfinite'])
def test_missing_coverage_or_unbound_evidence_is_rejected(files,change):
    summary,proof = read(files[1]),read(files[6])
    if change=='missing-contrast': summary['contrasts'].pop()
    elif change=='missing-fold': summary['checksums'].pop('fold9')
    elif change=='score-binding': summary['checksums']['fold9']['scores'][next(iter(summary['checksums']['fold9']['scores']))] = '0'*64
    elif change=='rank-binding': summary['checksums']['fold9']['ranks_sha256'] = '0'*64
    elif change=='row-binding': summary['checksums']['fold9']['rows_sha256'] = '0'*64
    elif change=='source': summary['provenance']['summary_source_sha256'] = '0'*64
    elif change=='failed-bias': proof['status'] = 'requires_review'
    elif change=='duplicate-fold': proof['rank_comparisons'][-1] = proof['rank_comparisons'][0]
    elif change=='hidden-rank-change': proof['rank_comparisons'][-1]['changed_ranks'] = 1
    elif change=='artifact-hash': proof['reference_artifacts'][next(iter(proof['reference_artifacts']))] = '0'*64
    else: summary['contrasts'][0]['lower'] = float('nan')
    files[1].write_text(json.dumps(summary));write(files[6],proof)
    with pytest.raises(ValueError):check(files)


def test_reference_digest_and_separate_summary_files_are_required(files):
    with pytest.raises(ValueError,match='verified analysis digest'):compare(*files,reference_sha256='0'*64)
    files[1].unlink();files[1].hardlink_to(files[0])
    with pytest.raises(ValueError,match='separate'):check(files)


def test_cli_retains_numerical_failure_and_refuses_overwrite(files):
    summary=read(files[1]);summary['contrasts'][0]['upper'] += 1e-8;write(files[1],summary)
    output=files[0].parent/'comparison.json'
    flags=('reference-summary','replay-summary','reference-manifest','replay-manifest','reference-report','replay-report','bias-comparison')
    command=[sys.executable,str(REPO/'reproducibility/compare_matched_kge_summaries.py')]
    for flag,path in zip(flags,files):command.extend(['--'+flag,str(path)])
    command.extend(['--reference-sha256',sha(files[0]),'--output',str(output)])
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==1 and read(output)['status']=='requires_review'
    digest=sha(output);again=subprocess.run(command,capture_output=True,text=True)
    assert again.returncode!=0 and sha(output)==digest
