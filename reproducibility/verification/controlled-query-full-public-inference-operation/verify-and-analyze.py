"""Verify all public-checkpoint query inference and regenerate every reported analysis."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SOURCE_WORK = BASE/'primary-report-comparison-v1'
SOURCE = SOURCE_WORK/'source'
GPU = BASE/'orix-full-query-public-replay-v1'
REFERENCE = BASE/'release-workspace/final-query-score-transfer-v1'
ORIGINAL = BASE/'release-workspace/completed-query-training-v1'
REPLAY = GPU/'workspace/controlled-query'
COMMIT = '1401b5ed155cdf70a3d90679c0bbc1fd666fb0fc'
GLOBAL = 'c9be85acbc90add09acf48484f02ddfb7de727d85106c80fb930e879365efee4'
REPORT = '51630e6e415b917d8a60301075081ceae83ad3a837e3513d1b2862d6fef90adf'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def native_tasks(text, array_job):
    records = {}
    for line in text.strip().splitlines():
        job,raw,state,code = line.strip().split('|')
        if not job.startswith(str(array_job)+'_'):
            raise ValueError('Unexpected native array job record')
        task = int(job.split('_')[1])
        if task not in range(6) or task in records or not raw.isdigit():
            raise ValueError('Duplicate, invalid or unexpected native array task')
        if state != 'COMPLETED' or code != '0:0':
            raise ValueError('Every inference task must complete successfully')
        records[task] = int(raw)
    if set(records) != set(range(6)) or len(set(records.values())) != 6:
        raise ValueError('Require all six independent completed inference tasks')
    return records


def verify_started(proof,campaign,plan_sha256,members):
    if (proof['campaign'] != campaign or proof['inference_plan_sha256'] != plan_sha256
            or not proof['no_test_predictions'] or len(members) != 400
            or proof['member_hashes'] != members):
        raise ValueError('Inference start is not bound to a fresh complete public checkpoint campaign')


def verify_transferred_scores(root,files):
    arms = ('bridge_pool','plain_pool','plain_and','typed_and')
    names = arms+tuple(a+'_frozen' for a in arms)
    expected = {f'workspace/controlled-query/{init}-seed{seed}/test/fold{fold}/{arm}/{name}'
        for init in ('ultra_4g','ultraquery') for seed in range(3) for fold in range(10)
        for arm in names for name in ('COMPLETE.json','scores.npz','ranks.npz')}
    if set(files) != expected:
        raise ValueError('Require source-bound hashes for all 480 score bundles')
    for name,digest in files.items():
        path = root/name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('Transferred score bundle differs from native capture: '+name)


def compare_table_text(original, replayed, original_report, replayed_report,
                       original_manifest, replayed_manifest):
    for prefix,a,b in [('% Report SHA256: ',original_report,replayed_report),
                       ('% Score manifest SHA256: ',original_manifest,replayed_manifest)]:
        before,after = prefix+a,prefix+b
        if original.splitlines().count(before) != 1 or replayed.splitlines().count(after) != 1:
            raise ValueError('Missing or incorrect table provenance comment')
        original = ''.join(after+line[len(before):] if line.rstrip('\r\n') == before else line
                           for line in original.splitlines(keepends=True))
    if original != replayed:
        raise ValueError('Table body or formatting differs')


def verify_exports(directory, inputs):
    primary = {'controlled-query-'+n+'.tex' for n in ('main','full','contrasts')}
    detailed = {f'controlled-bias-{i}-seed{s}.tex' for i in ('ultra_4g','ultraquery') for s in range(3)}
    csv = 'controlled-bias-metrics.csv'
    summary = 'controlled-query-bias-summary.tex'
    expected = {'primary/'+n for n in primary} | {'bias/'+n for n in detailed | {csv,'controlled-bias-export.json'}} | {
        'summary/'+summary,'summary/controlled-query-bias-summary.json'}
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Export inventory differs')
    export = read(directory/'bias/controlled-bias-export.json')
    seed = read(directory/'summary/controlled-query-bias-summary.json')
    for proof in (export,seed):
        for key in ('report_sha256','manifest_sha256','metadata_sha256'):
            if proof[key] != inputs[key]: raise ValueError('Export input provenance differs')
    if export['outputs'] != {n:sha(directory/'bias'/n) for n in detailed | {csv}}:
        raise ValueError('Export file checksums differ')
    if seed['table_sha256'] != sha(directory/'summary'/summary):
        raise ValueError('Summary table checksum differs')
    return dict(hashes={n:sha(directory/n) for n in sorted(expected)},export=export,summary=seed)


def main():
    started = time.time()
    plan = read(ROOT/'plan.json')
    assert plan['script_sha256'] == sha(__file__)
    assert plan['sbatch_sha256'] == sha(ROOT/'verify-and-analyze.sbatch')
    assert not (ROOT/'output').exists() and not (ROOT/'ANALYSIS_COMPLETE.json').exists()
    evidence = {}

    def retain(path, expected=None):
        path = Path(path)
        assert path.is_file() and not path.is_symlink(), path
        digest = sha(path)
        if expected is not None: assert digest == expected, path
        evidence[str(path)] = digest
        return read(path)

    source = retain(SOURCE_WORK/'SOURCE_READY.json',plan['analysis_source_proof_sha256'])
    assert source['status'] == 'passed' and source['source_commit'] == COMMIT
    assert source['tests'] == dict(tests=20,failures=0,errors=0,skipped=0)
    bias_source = retain(BASE/'production-bias-comparison-v1/SOURCE_READY.json',plan['bias_source_proof_sha256'])
    assert bias_source['status'] == 'passed' and bias_source['tests'] == dict(tests=32,failures=0,errors=0,skipped=0)
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    for name,digest in {**source['files_sha256'],**plan['additional_source_sha256']}.items():
        assert sha(SOURCE/name) == digest, name
        evidence[str(SOURCE/name)] = digest
    for name,digest in bias_source['files_sha256'].items(): assert sha(SOURCE/name) == digest, name
    import numpy as np
    assert np.__version__ == source['numpy'] == '2.2.6'
    original_source = retain(GPU/'SOURCE_READY.json',plan['inference_source_proof_sha256'])
    download = retain(GPU/'public-download-proof.json',plan['download_proof_sha256'])
    assert original_source['status'] == download['status'] == 'passed'
    assert download['source_proof_sha256'] == sha(GPU/'SOURCE_READY.json')
    assert sha(GPU/'run-inference.py') == plan['inference_script_sha256']
    assert sha(GPU/'run-inference.sbatch') == plan['inference_sbatch_sha256']
    evidence[str(GPU/'run-inference.py')] = plan['inference_script_sha256']
    evidence[str(GPU/'run-inference.sbatch')] = plan['inference_sbatch_sha256']
    assert sha(REPLAY/'GLOBAL-SELECTION.json') == sha(ORIGINAL/'GLOBAL-SELECTION.json') == GLOBAL
    execution = retain(GPU/'inference-plan.json',plan['inference_plan_sha256'])
    inputs = retain(GPU/'INPUTS_READY.json',execution['inputs_proof_sha256'])
    public = retain(GPU/'release-manifest.json','880e0dfd563f5f6f7a783d2ee669e568f56c969c15998db56ac4d502546a069c')
    for key, wanted in [('script_sha256',plan['inference_script_sha256']),
                        ('sbatch_sha256',plan['inference_sbatch_sha256']),
                        ('source_proof_sha256',plan['inference_source_proof_sha256']),
                        ('download_proof_sha256',plan['download_proof_sha256'])]:
        assert execution[key] == wanted, key
    assert inputs['status'] == 'layout_ready' and inputs['files'] == 2220
    assert inputs['no_test_predictions_before_inference']
    assert inputs['source_proof_sha256'] == execution['source_proof_sha256']
    assert inputs['public_manifest_sha256'] == sha(GPU/'release-manifest.json')
    assert inputs['public_download_proof_sha256'] == execution['download_proof_sha256']
    capture = retain(GPU/'NATIVE_CAPTURE.json',plan['native_capture_sha256'])
    assert capture['status'] == 'native_completed' and capture['cluster'] == 'orix'
    assert capture['inference_array'] == plan['inference_array']
    assert capture['inference_plan_sha256'] == plan['inference_plan_sha256']
    assert capture['capture_script_sha256'] == execution['capture_script_sha256'] == sha(GPU/'capture-native.py')
    evidence[str(GPU/'capture-native.py')] = execution['capture_script_sha256']
    verify_transferred_scores(GPU,capture['score_files_sha256'])
    native_path = GPU/'native-completion.log'
    assert native_path.is_file() and not native_path.is_symlink()
    assert sha(native_path) == capture['native_completion_sha256']
    evidence[str(native_path)] = sha(native_path)
    native = native_path.read_text()
    jobs = native_tasks(native,plan['inference_array'])
    assert {str(k):v for k,v in jobs.items()} == capture['native_task_job_ids']
    with (ROOT/'inference-native-completion.log').open('x') as stream: stream.write(native)
    sample = retain(GPU/'validation-verification.json',execution['validation_proof_sha256'])
    assert sha(GPU/'validation-verification.json') == '5997f94dbf20b893be94e2a8e05aad8849697438d29a0cc4de6862d34c7b2658'
    assert sample['status'] == 'passed' and sample['all_scores_within_tolerance'] and sample['all_validation_ranks_exact']
    assert len(sample['records']) == 16 and sample['job_id'] == 354814
    coverage = set()
    campaigns = {f'{i}-seed{s}' for i in ('ultra_4g','ultraquery') for s in range(3)}
    for task in range(6):
        campaign = f"{('ultra_4g','ultraquery')[task//3]}-seed{task%3}"
        folder = GPU/'gpu-runs'/campaign
        started_proof = retain(folder/'STARTED.json')
        expected_members = {n:r['sha256'] for n,r in public['provenance']['expected_files'].items()
            if Path(n).parts[1] == campaign or Path(n).parts[1] not in campaigns}
        verify_started(started_proof,campaign,plan['inference_plan_sha256'],expected_members)
        coverage.update(expected_members)
        completed = retain(folder/'COMPLETE.json')
        assert completed['status'] == 'inference_completed' and completed['campaign'] == campaign
        assert completed['score_bundles'] == 80
        assert Path(completed['environment_file']).name == completed['environment_file']
        attempt = retain(folder/completed['environment_file'],completed['environment_sha256'])
        wanted = dict(campaign=campaign,task=task,job_id=jobs[task],array_job_id=plan['inference_array'],
            torch='2.1.2+cu121',numpy='1.26.4',torch_geometric='2.4.0',capability=[9,0],cpu_threads=8,
            matmul_tf32=False,cudnn_tf32=True,inputs_proof_sha256=execution['inputs_proof_sha256'],
            started_proof_sha256=sha(folder/'STARTED.json'),
            download_proof_sha256=plan['download_proof_sha256'],operation_script_sha256=plan['inference_script_sha256'],
            launcher_sha256=original_source['analysis_files']['scripts/run_controlled_query_scoring.py'],
            validation_replay_proof_sha256=execution['validation_proof_sha256'])
        assert all(attempt[k] == v for k,v in wanted.items()), campaign
        assert 'H200' in attempt['gpu']
        worker = retain(REPLAY/'scoring-workers'/f'{campaign}.json',completed['worker_proof_sha256'])
        assert worker['campaign'] == campaign and worker['folds'] == list(range(10))
        assert worker['score_bundles'] == 80 and worker['global_selection_sha256'] == GLOBAL
        assert worker['launcher_sha256'] == wanted['launcher_sha256']
    assert len(coverage) == 2220 and coverage == set(public['provenance']['expected_files'])
    production = retain(REFERENCE/'ANALYSIS_COMPLETE.json',plan['reference_analysis_sha256'])
    assert production['status'] == 'passed' and production['global_selection_sha256'] == GLOBAL
    for path,digest in production['outputs'].items():
        assert sha(path) == digest, path
        evidence[path] = digest
    assert sha(REFERENCE/'controlled-report.json') == REPORT
    reference_manifest = ORIGINAL/'controlled-bias-manifest.json'
    reference_bias = REFERENCE/'controlled-bias-report.json'
    reference_proof = retain(REFERENCE/'controlled-bias-verification.json')
    assert reference_proof['status'] == 'passed'
    for fold,record in reference_proof['ranks'].items():
        for suffix,key in [('ranks.npz','ranks_sha256'),('rows.json','rows_sha256')]:
            path = REFERENCE/'bias-ranks'/f'fold{fold}-{suffix}'
            assert sha(path) == record[key]
            evidence[str(path)] = record[key]
    output = ROOT/'output'; output.mkdir()

    def command(relative,*args):
        print('RUN',relative,flush=True)
        subprocess.run([sys.executable,str(SOURCE/relative),*map(str,args)],check=True)

    inference = output/'inference-comparison.json'
    command('reproducibility/compare_controlled_query_inference.py','--reference-root',ORIGINAL,
        '--replay-root',REPLAY,'--output',inference)
    report = output/'controlled-report.json'
    command('code/analysis/summarize_controlled_query.py','--root',REPLAY,'--output',report)
    command('reproducibility/compare_controlled_query_reports.py','--reference-report',REFERENCE/'controlled-report.json',
        '--replay-report',report,'--inference-comparison',inference,'--reference-sha256',REPORT,
        '--output',output/'primary-report-comparison.json')
    manifest,bias,ranks = output/'controlled-bias-manifest.json',output/'controlled-bias-report.json',output/'bias-ranks'
    command('reproducibility/build_controlled_bias_manifest.py','build','--root',REPLAY,
        '--controlled-report',report,'--output',manifest)
    command('code/analysis/evaluate_stepwise_bias.py','--base-dir',REPLAY/'ultra_4g-seed0',
        '--score-manifest',manifest,'--out-json',bias,'--ranks-dir',ranks,'--workers',4)
    command('reproducibility/build_controlled_bias_manifest.py','verify','--controlled-report',report,
        '--score-manifest',manifest,'--bias-report',bias,'--ranks-dir',ranks,'--output',output/'controlled-bias-verification.json')
    command('reproducibility/compare_production_bias_replays.py','--reference-manifest',reference_manifest,
        '--replay-manifest',manifest,'--reference-report',reference_bias,'--replay-report',bias,
        '--reference-ranks',REFERENCE/'bias-ranks','--replay-ranks',ranks,'--output',output/'bias-comparison.json')
    rendered = []
    for name,primary,bias_path,score_manifest in [('reference',REFERENCE/'controlled-report.json',reference_bias,reference_manifest),
                                               ('replay',report,bias,manifest)]:
        destination = output/(name+'-tables')
        command('reproducibility/generate_controlled_query_tables.py','--report-json',primary,'--out-dir',destination/'primary')
        for generator,part in [('generate_production_bias_tables.py','bias'),('generate_controlled_bias_summary.py','summary')]:
            command('reproducibility/'+generator,'--report-json',bias_path,'--score-manifest',score_manifest,'--out-dir',destination/part)
        inputs = dict(report_sha256=sha(bias_path),manifest_sha256=sha(score_manifest),metadata_sha256=sha(bias_path.with_suffix('.metadata.json')))
        rendered.append((destination,inputs,verify_exports(destination,inputs)))
    left,right = rendered
    manuscript_names = {'primary/controlled-query-'+name+'.tex' for name in ('main','full','contrasts')} | {
        f'bias/controlled-bias-{init}-seed{seed}.tex' for init in ('ultra_4g','ultraquery') for seed in range(3)} | {
        'summary/controlled-query-bias-summary.tex'}
    assert set(plan['manuscript_tables_sha256']) == manuscript_names
    for name,digest in plan['manuscript_tables_sha256'].items():
        assert left[2]['hashes'][name] == digest, ('Current manuscript table differs',name)
    for name in left[2]['hashes']:
        if name.endswith('.json'): continue
        a,b = left[0]/name,right[0]/name
        if name.startswith('primary/') or name.endswith('.csv'):
            assert a.read_bytes() == b.read_bytes(), name
        else:
            compare_table_text(a.read_text(),b.read_text(),left[1]['report_sha256'],right[1]['report_sha256'],
                left[1]['manifest_sha256'],right[1]['manifest_sha256'])
    for kind,ignored in [('export',{'report_sha256','manifest_sha256','metadata_sha256','outputs'}),
                         ('summary',{'report_sha256','manifest_sha256','metadata_sha256','table_sha256'})]:
        assert {k:v for k,v in left[2][kind].items() if k not in ignored} == {
            k:v for k,v in right[2][kind].items() if k not in ignored}, kind
    for path,digest in evidence.items(): assert sha(path) == digest, path
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    for name in ('inference-comparison.json','primary-report-comparison.json','controlled-bias-verification.json','bias-comparison.json'):
        assert read(output/name)['status'] == 'passed', name
    assert read(output/'bias-comparison.json')['rank_arrays'] == 6720
    result = dict(status='passed',source_commit=COMMIT,script_sha256=sha(__file__),plan_sha256=sha(ROOT/'plan.json'),
        inference_array=plan['inference_array'],inference_cluster='orix',score_bundles=480,primary_contrasts=8,bias_variants=48,bias_conditions=42,
        bias_rank_arrays=6720,primary_tables=3,detailed_bias_tables=6,seed_summary_tables=1,
        native_completion_sha256=sha(ROOT/'inference-native-completion.log'),input_evidence_sha256=evidence,
        manuscript_tables_sha256=plan['manuscript_tables_sha256'],
        outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        elapsed_seconds=time.time()-started,
        scope='All six ORIX H200 inference tasks from relayed public checkpoints completed. Full 480-bundle score/rank comparison, every primary statistic and eight bootstrap contrasts, all 42 bias conditions and 6720 rank arrays, and current primary/detailed/seed-summary exports agree with the verified production reference. No new training or external-cohort validation claim.')
    with (ROOT/'ANALYSIS_COMPLETE.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_evidence_sha256','outputs')},indent=2),flush=True)


if __name__ == '__main__': main()
