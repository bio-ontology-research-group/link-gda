"""Verify complete independent matched inference and all downstream paper results."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SOURCE_WORK = BASE/'matched-primary-comparison-v1'
SOURCE = SOURCE_WORK/'source'
GPU = BASE/'orix-full-matched-retained-replay-v3'
REFERENCE = BASE/'release-workspace/final-matched-transfer-v1'
ORIGINAL = BASE/'release-workspace/completed-matched-training-v1'
DATA = BASE/'release-workspace/inputs/benchmark/data'
ANNOTATIONS = BASE/'release-workspace/inputs/matched-kge/inputs/gene_phenotypes.csv'
REPLAY = GPU/'workspace/matched-kge'
REPLAY_DATA = GPU/'workspace/benchmark/data'
COMMIT = '193a4c9755393a80a10469ac7ade689dd510df18'
TRAINING = '063efc263446b4d119d3dbf4b97edb8709859f1e'
FREEZE = 'c0044268b876ff8d0a7f03daef733c6d4908bf381c152ada5bee91d13779cee5'
FAMILIES = ('U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL')
CHOICES = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
read = lambda p: json.loads(Path(p).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def retain_input(path, evidence, expected=None, *, parse_json=True):
    """Bind every retained file's bytes; decode only declared JSON payloads."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), path
    digest = sha(path)
    if expected is not None:
        assert digest == expected, path
    evidence[str(path)] = digest
    return read(path) if parse_json else None


def native_tasks(text, array_job):
    records = {}
    if type(array_job) is not int or array_job <= 0: raise ValueError('Invalid array job')
    for line in text.strip().splitlines():
        job, raw, state, code = line.strip().split('|')
        prefix = str(array_job)+'_'
        if not job.startswith(prefix) or not job[len(prefix):].isdigit():
            raise ValueError('Unexpected native array job record')
        task = int(job[len(prefix):])
        if task not in range(3) or task in records or not raw.isdigit() or int(raw) <= 0:
            raise ValueError('Duplicate, invalid or unexpected native array task')
        if state != 'COMPLETED' or code != '0:0':
            raise ValueError('Every inference task must complete successfully')
        records[task] = int(raw)
    if set(records) != set(range(3)) or len(set(records.values())) != 3:
        raise ValueError('Require three completed native allocations covering all ten folds')
    return {fold: records[fold % 3] for fold in range(10)}


def members_for_fold(files, fold):
    if type(fold) is not int or fold not in range(10) or len(files) != 843:
        raise ValueError('Incomplete public member design')
    for name in files:
        p = PurePosixPath(name)
        if (not re.fullmatch(r'(matched-kge|benchmark)/[A-Za-z0-9_./-]+', name)
                or '..' in p.parts or str(p) != name):
            raise ValueError('Unsafe member path')
    shared = {'matched-kge/FROZEN.json', 'matched-kge/inputs/PROVENANCE.json',
              'matched-kge/inputs/gene_phenotypes.csv'}
    own = {n:r['sha256'] for n,r in files.items()
           if n.startswith(f'matched-kge/fold{fold}/')
           or n.startswith(f'benchmark/data/fold{fold}/') or n in shared}
    if len(own) != 87 or not shared <= own.keys():
        raise ValueError('Incomplete fold input inventory')
    if sum(n.startswith(f'matched-kge/fold{fold}/') and n.endswith('.pt') for n in own) != 40:
        raise ValueError('Require forty weights per fold')
    return own


def verify_started(start, fold, plan, execution, members):
    expected = dict(fold=fold, inference_plan_sha256=plan['inference_plan_sha256'],
        staging_proof_sha256=plan['staging_proof_sha256'], freeze_sha256=FREEZE,
        operation_script_sha256=plan['inference_script_sha256'], training_source_commit=TRAINING,
        no_test_predictions=True, member_hashes=members)
    if start != expected or len(members) != 87:
        raise ValueError('Inference start differs from the fresh public fold')


def verify_capture_files(root, capture, public_files):
    members_for_fold(public_files, 0)
    expected = {f'workspace/matched-kge/fold{fold}/{graph}/{family}/{name}'
        for fold in range(10) for graph in ('original', 'nobridge') for family in FAMILIES
        for name in ['SCORED.json', *('scores-'+c+'.npz' for c in CHOICES)]}
    if set(capture['score_files_sha256']) != expected:
        raise ValueError('Require all 400 score files and 100 scorer records')
    binaries = expected - {n for n in expected if n.endswith('/SCORED.json')}
    if set(capture['binary_files']) != binaries:
        raise ValueError('Captured binary inventory differs')
    workspace = root/'workspace'
    paths = list(workspace.rglob('*'))
    locks = {f'matched-kge/fold{fold}/{graph}/{family}/run.lock'
             for fold in range(10) for graph in ('original', 'nobridge') for family in FAMILIES}
    wanted = set(public_files) | {n.removeprefix('workspace/') for n in expected} | locks
    if (workspace.is_symlink() or len(public_files) != 843
            or any(p.is_symlink() for p in paths)
            or {p.relative_to(workspace).as_posix() for p in paths if p.is_file()} != wanted):
        raise ValueError('Destination workspace inventory differs')
    for name in locks:
        if (workspace/name).read_bytes() != b'':
            raise ValueError('Nonempty scorer lock')
    # CPU verification hashes every transferred input and score. No login-node hashing.
    for name, record in public_files.items():
        path = workspace/name
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Public input bytes differ: '+name)
    for name, digest in capture['score_files_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('Transferred scorer output differs: '+name)
        if name in binaries:
            record = capture['binary_files'][name]
            if record['sha256'] != digest or record['bytes'] != (root/name).stat().st_size:
                raise ValueError('Binary capture metadata differs: '+name)
    for name, record in capture['metadata_files'].items():
        p = PurePosixPath(name)
        if (not re.fullmatch(r'[A-Za-z0-9_./-]+', name) or p.is_absolute()
                or '..' in p.parts or str(p) != name):
            raise ValueError('Unsafe captured path')
        path = root/name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Symlink in captured metadata path')
        raw = path.read_bytes()
        if (raw != record['text'].encode() or len(raw) != record['bytes']
                or hashlib.sha256(raw).hexdigest() != record['sha256']):
            raise ValueError('Captured metadata bytes differ: '+name)


def compare_table_text(original, replayed, headers):
    """Allow only the exact, independently bound leading provenance comments."""
    left, right = original.splitlines(keepends=True), replayed.splitlines(keepends=True)
    for index, (prefix, old, new) in enumerate(headers):
        for lines, digest in ((left, old), (right, new)):
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid provenance digest')
            if len(lines) <= index or lines[index] != prefix+digest+'\n':
                raise ValueError('Missing or incorrect leading provenance comment')
            if sum(line.startswith(prefix) for line in lines) != 1:
                raise ValueError('Duplicated provenance comment')
    if left[len(headers):] != right[len(headers):]:
        raise ValueError('Table body or formatting differs')


def verify_exports(directory, inputs):
    primary = {'matched-kge-primary.tex', 'matched-kge-design.tex'}
    detailed = {f'matched-bias-{family}.tex' for family in FAMILIES}
    csv = 'matched-bias-metrics.csv'
    expected = {'primary/'+n for n in primary} | {
        'bias/'+n for n in detailed | {csv, 'matched-bias-export.json'}}
    paths = list(directory.rglob('*'))
    if any(p.is_symlink() for p in paths) or {
            str(p.relative_to(directory)) for p in paths if p.is_file()} != expected:
        raise ValueError('Export inventory differs')
    proof = read(directory/'bias/matched-bias-export.json')
    for key in ('report_sha256', 'manifest_sha256', 'metadata_sha256', 'generator_sha256'):
        if proof[key] != inputs[key]: raise ValueError('Export input provenance differs')
    if proof['variants'] != 40 or proof['conditions'] != 42:
        raise ValueError('Incomplete export design')
    if proof['outputs'] != {n:sha(directory/'bias'/n) for n in detailed | {csv}}:
        raise ValueError('Export file checksums differ')
    return dict(hashes={n:sha(directory/n) for n in sorted(expected)}, export=proof)


def main():
    started = time.time()
    plan = read(ROOT/'plan.json')
    assert plan['script_sha256'] == sha(__file__)
    assert plan['sbatch_sha256'] == sha(ROOT/'verify-and-analyze.sbatch')
    assert not (ROOT/'output').exists() and not (ROOT/'ANALYSIS_COMPLETE.json').exists()
    evidence = {}

    def retain(path, expected=None, *, parse_json=True):
        return retain_input(path, evidence, expected, parse_json=parse_json)

    source = retain(SOURCE_WORK/'SOURCE_READY.json', plan['analysis_source_proof_sha256'])
    assert source['status'] == 'passed' and source['source_commit'] == COMMIT
    assert source['tests'] == dict(tests=20, failures=0, errors=0, skipped=0)
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    for name in source['files_sha256'].keys() & plan['additional_source_sha256'].keys():
        assert source['files_sha256'][name] == plan['additional_source_sha256'][name], name
    for name, digest in {**source['files_sha256'], **plan['additional_source_sha256']}.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts, name
        assert sha(SOURCE/name) == digest, name
        evidence[str(SOURCE/name)] = digest
    import numpy as np
    assert np.__version__ == source['numpy'] == '2.2.6'
    inference_source = retain(GPU/'SOURCE_READY.json', plan['inference_source_proof_sha256'])
    staged = retain(GPU/'STAGED_INPUTS.json', plan['staging_proof_sha256'])
    inference_plan = retain(GPU/'inference-plan.json', plan['inference_plan_sha256'])
    assert inference_source['status'] == staged['status'] == 'passed'
    assert inference_source['commits']['training-source'] == TRAINING
    assert staged['source_proof_sha256'] == plan['inference_source_proof_sha256']
    assert staged['checkpoint_files'] == 400 and staged['checkpoint_collection_files'] == 701
    assert staged['file_count'] == len(staged['files']) == 843
    assert staged['no_predictions_before_inference'] is True
    for key, value in dict(source_proof_sha256=plan['inference_source_proof_sha256'],
            staging_proof_sha256=plan['staging_proof_sha256'], freeze_sha256=FREEZE,
            operation_script_sha256=plan['inference_script_sha256'], sbatch_sha256=plan['inference_sbatch_sha256']).items():
        assert inference_plan[key] == value, key
    for filename, key in [('run-inference.py', 'inference_script_sha256'), ('run-inference.sbatch', 'inference_sbatch_sha256')]:
        assert sha(GPU/filename) == plan[key]
        evidence[str(GPU/filename)] = plan[key]
    assert staged['freeze_sha256'] == sha(REPLAY/'FROZEN.json') == sha(ORIGINAL/'FROZEN.json') == FREEZE
    capture = retain(GPU/'NATIVE_CAPTURE.json', plan['native_capture_sha256'])
    assert capture['status'] == 'native_completed' and capture['cluster'] == 'orix'
    assert capture['inference_array'] == plan['inference_array']
    assert capture['inference_plan_sha256'] == plan['inference_plan_sha256']
    assert capture['capture_script_sha256'] == inference_plan['capture_script_sha256'] == sha(GPU/'capture-native.py')
    evidence[str(GPU/'capture-native.py')] = capture['capture_script_sha256']
    verify_capture_files(GPU, capture, staged['files'])
    native_path = GPU/'native-completion.log'
    assert native_path.is_file() and not native_path.is_symlink()
    assert sha(native_path) == capture['native_completion_sha256']
    evidence[str(native_path)] = sha(native_path)
    native = native_path.read_text()
    jobs = native_tasks(native, plan['inference_array'])
    assert {str(k):v for k,v in jobs.items()} == capture['native_fold_job_ids']
    with (ROOT/'inference-native-completion.log').open('x') as stream: stream.write(native)
    sample = retain(GPU/'validation-verification.json', inference_plan['validation_sample_proof_sha256'])
    assert sha(GPU/'validation-verification.json') == 'b83fb30e8e378babc448dd609803b40c446a006344c685ef56d665fa68fc2e95'
    assert sample['status'] == 'passed' and sample['checkpoint_files'] == 12 and sample['all_validation_metrics_exact']
    assert sample['job_id'] == 355504 and sample['cpu_threads'] == sample['allocated_cpus'] == 8
    assert sample['source_proof_sha256'] == plan['inference_source_proof_sha256']
    assert len(sample['records']) == 12 and all(r['actual'] == r['expected'] for r in sample['records'])
    sample_native = GPU/'validation-native-completion.log'
    assert sha(sample_native) == inference_plan['validation_native_sha256'] == '4389a78e02ad371b2048c3d5fb259f81a5f212b9a973b4f94c52656d1de7f48b'
    evidence[str(sample_native)] = sha(sample_native)
    retain(GPU/'staging-plan.json', staged['staging_plan_sha256'])
    retain(GPU/'stage-inputs.py', staged['stage_script_sha256'], parse_json=False)
    manifest = retain(GPU/'checkpoint-release-manifest.json', staged['manifest_sha256'])
    reference = retain(GPU/'INPUTS_REFERENCE.json', staged['input_reference_sha256'])
    assert staged['transport'] == 'native_hardlinks'
    assert staged['verification'] == 'metadata_only_before_task_hashes'
    assert staged['files'] == {**manifest['provenance']['expected_files'], **reference['files']}
    assert capture['native_allocations'] == 3 and capture['folds'] == 10
    coverage = set()
    for fold in range(10):
        folder = GPU/'gpu-runs'/f'fold{fold}'
        start = retain(folder/'STARTED.json')
        expected_members = members_for_fold(staged['files'], fold)
        verify_started(start, fold, plan, inference_plan, expected_members)
        coverage.update(expected_members)
        complete = retain(folder/'COMPLETE.json')
        assert complete['status'] == 'inference_completed' and complete['fold'] == fold
        assert complete['training_runs'] == 10 and complete['score_files'] == 40
        assert complete['started_proof_sha256'] == sha(folder/'STARTED.json')
        assert Path(complete['environment_file']).name == complete['environment_file']
        attempt = retain(folder/complete['environment_file'], complete['environment_sha256'])
        wanted = dict(job_id=jobs[fold], array_job_id=plan['inference_array'], fold=fold,
            torch='2.10.0+cu128', numpy='2.4.2', pykeen='1.11.1', capability=[9,0],
            cpu_threads=8, allocated_cpus=8, array_task_id=fold % 3, started_proof_sha256=sha(folder/'STARTED.json'),
            matmul_tf32=sample['runtime']['tf32_matmul'], cudnn_tf32=sample['runtime']['tf32_cudnn'])
        assert all(attempt[k] == v for k,v in wanted.items()) and 'H200' in attempt['gpu'], fold
        expected_runs = {f'fold{fold}/{g}/{m}' for g in ('original','nobridge') for m in FAMILIES}
        assert len(complete['records']) == 10 and {r['run'] for r in complete['records']} == expected_runs
        for record in complete['records']:
            run = REPLAY/record['run']
            scored = retain(run/'SCORED.json', record['scored_sha256'])
            assert scored['files'] == record['files']
            assert set(record['files']) == {'scores-'+c+'.npz' for c in CHOICES}
    assert len(coverage) == 843 and coverage == set(staged['files'])
    production = retain(REFERENCE/'ANALYSIS_COMPLETE.json', plan['reference_analysis_sha256'])
    checkpoints = retain(REFERENCE/'CHECKPOINTS_VERIFIED.json', plan['reference_checkpoints_sha256'])
    assert production['status'] == checkpoints['status'] == 'passed'
    assert production['freeze_sha256'] == checkpoints['freeze_sha256'] == FREEZE
    assert production['training_runs'] == checkpoints['training_runs'] == 100
    assert checkpoints['checkpoint_files'] == production['model_checkpoints'] == 400
    assert production['score_files'] == 400 and production['primary_contrasts'] == 10
    for path, digest in production['outputs'].items():
        assert sha(path) == digest, path
        evidence[path] = digest
    reference_manifest = ORIGINAL/'score-manifest.json'
    reference_report, reference_summary = REFERENCE/'evaluation.json', REFERENCE/'summary.json'
    for path in (reference_manifest, reference_report, reference_summary, reference_report.with_suffix('.metadata.json')):
        assert any(Path(p).resolve() == path.resolve() and digest == sha(path)
                   for p,digest in production['outputs'].items()), path
    output = ROOT/'output'; output.mkdir()

    def command(relative, *args):
        assert evidence.get(str(SOURCE/relative)) == sha(SOURCE/relative), relative
        print('RUN', relative, flush=True)
        subprocess.run([sys.executable, str(SOURCE/relative), *map(str,args)], check=True)

    command('reproducibility/compare_matched_kge_inference.py', '--reference-root', ORIGINAL,
        '--replay-root', REPLAY, '--reference-data', DATA, '--replay-data', REPLAY_DATA,
        '--reference-annotations', ANNOTATIONS, '--replay-annotations', REPLAY/'inputs/gene_phenotypes.csv',
        '--output', output/'inference-comparison.json')
    manifest, report, summary, ranks = output/'score-manifest.json', output/'evaluation.json', output/'summary.json', output/'ranks'
    command('reproducibility/build_matched_kge_manifest.py', '--root', REPLAY, '--data-root', REPLAY_DATA,
        '--gene-phenotypes', REPLAY/'inputs/gene_phenotypes.csv', '--output', manifest)
    command('code/analysis/evaluate_stepwise_bias.py', '--base-dir', REPLAY_DATA.parent,
        '--score-manifest', manifest, '--out-json', report, '--ranks-dir', ranks, '--workers', 4)
    command('code/analysis/summarize_matched_kge.py', '--score-manifest', manifest, '--report-json', report,
        '--ranks-dir', ranks, '--data-root', REPLAY_DATA, '--output', summary)
    command('reproducibility/compare_production_bias_replays.py', '--reference-manifest', reference_manifest,
        '--replay-manifest', manifest, '--reference-report', reference_report, '--replay-report', report,
        '--reference-ranks', REFERENCE/'ranks', '--replay-ranks', ranks, '--output', output/'bias-comparison.json')
    command('reproducibility/compare_matched_kge_summaries.py', '--reference-summary', reference_summary,
        '--replay-summary', summary, '--reference-manifest', reference_manifest, '--replay-manifest', manifest,
        '--reference-report', reference_report, '--replay-report', report, '--bias-comparison', output/'bias-comparison.json',
        '--reference-sha256', sha(reference_summary), '--output', output/'primary-summary-comparison.json')
    rendered = []
    for name, primary, bias, scores in [('reference', reference_summary, reference_report, reference_manifest),
                                      ('replay', summary, report, manifest)]:
        destination = output/(name+'-tables')
        command('reproducibility/generate_matched_kge_tables.py', '--report-json', primary, '--out-dir', destination/'primary')
        command('reproducibility/generate_production_bias_tables.py', '--report-json', bias,
            '--score-manifest', scores, '--out-dir', destination/'bias')
        inputs = dict(summary_sha256=sha(primary), report_sha256=sha(bias), manifest_sha256=sha(scores),
            metadata_sha256=sha(bias.with_suffix('.metadata.json')),
            generator_sha256=sha(SOURCE/'reproducibility/generate_production_bias_tables.py'))
        rendered.append((destination, inputs, verify_exports(destination, inputs)))
    left, right = rendered
    # Match the original verified exports as well as the regenerated pair.
    for name, digest in left[2]['hashes'].items():
        part, filename = name.split('/')
        reference = REFERENCE/('tables' if part == 'primary' else 'bias-tables')/filename
        assert sha(reference) == digest, name
        assert any(Path(p).resolve() == reference.resolve() and digest == h
                   for p,h in production['outputs'].items()), name
        a, b = left[0]/name, right[0]/name
        if name.endswith('.json'): continue
        if name.endswith('.csv'):
            assert a.read_bytes() == b.read_bytes(), name
        else:
            bindings = [('Summary', 'summary_sha256')] if part == 'primary' else [
                ('Report', 'report_sha256'), ('Score manifest', 'manifest_sha256')]
            compare_table_text(a.read_text(), b.read_text(), [
                ('% '+label+' SHA256: ', left[1][key], right[1][key]) for label,key in bindings])
    ignored = {'report_sha256', 'manifest_sha256', 'metadata_sha256', 'outputs'}
    assert {k:v for k,v in left[2]['export'].items() if k not in ignored} == {
        k:v for k,v in right[2]['export'].items() if k not in ignored}
    for path, digest in evidence.items(): assert sha(path) == digest, path
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    for name in ('inference-comparison.json', 'primary-summary-comparison.json', 'bias-comparison.json'):
        assert read(output/name)['status'] == 'passed', name
    assert read(output/'bias-comparison.json')['rank_arrays'] == 5600
    result = dict(status='passed', source_commit=COMMIT, script_sha256=sha(__file__), plan_sha256=sha(ROOT/'plan.json'),
        inference_array=plan['inference_array'], training_runs=100, score_files=400, primary_contrasts=10,
        bias_variants=40, bias_conditions=42, bias_rank_arrays=5600, primary_tables=2, detailed_bias_tables=5,
        native_completion_sha256=sha(ROOT/'inference-native-completion.log'), input_evidence_sha256=evidence,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        elapsed_seconds=time.time()-started,
        scope='Three retained-checkpoint allocations completed all ten inference folds; 400 score comparisons, ten primary contrasts, all 40 x 42 bias conditions, 5600 rank arrays, and primary/bias tables agree with verified production. No independent retraining or external-cohort validation claim.')
    with (ROOT/'ANALYSIS_COMPLETE.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_evidence_sha256','outputs')}, indent=2), flush=True)


if __name__ == '__main__': main()
