"""Verify complete independent matched inference and all downstream paper results."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SOURCE_WORK = BASE/'matched-primary-comparison-v1'
SOURCE = SOURCE_WORK/'source'
GPU = BASE/'full-matched-public-replay-v1'
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


def native_tasks(text, array_job):
    records = {}
    if type(array_job) is not int or array_job <= 0: raise ValueError('Invalid array job')
    for line in text.strip().splitlines():
        job, raw, state, code = line.strip().split('|')
        prefix = str(array_job)+'_'
        if not job.startswith(prefix) or not job[len(prefix):].isdigit():
            raise ValueError('Unexpected native array job record')
        task = int(job[len(prefix):])
        if task not in range(10) or task in records or not raw.isdigit() or int(raw) <= 0:
            raise ValueError('Duplicate, invalid or unexpected native array task')
        if state != 'COMPLETED' or code != '0:0':
            raise ValueError('Every inference task must complete successfully')
        records[task] = int(raw)
    if set(records) != set(range(10)) or len(set(records.values())) != 10:
        raise ValueError('Require all ten independent completed inference tasks')
    return records


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

    def retain(path, expected=None):
        path = Path(path)
        assert path.is_file() and not path.is_symlink(), path
        digest = sha(path)
        if expected is not None: assert digest == expected, path
        evidence[str(path)] = digest
        return read(path)

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
    download = retain(GPU/'DOWNLOAD_COMPLETE.json', plan['download_proof_sha256'])
    inference_plan = retain(GPU/'inference-plan.json', plan['inference_plan_sha256'])
    assert inference_source['status'] == download['status'] == 'passed'
    assert inference_source['commits']['training-source'] == TRAINING
    assert download['source_proof_sha256'] == plan['inference_source_proof_sha256']
    assert download['checkpoint_files'] == 400 and download['checkpoint_collection_files'] == 701
    assert download['file_count'] == len(download['files']) == 843
    assert download['no_predictions_before_inference'] is True
    for key, value in dict(source_proof_sha256=plan['inference_source_proof_sha256'],
            download_proof_sha256=plan['download_proof_sha256'], freeze_sha256=FREEZE,
            operation_script_sha256=plan['inference_script_sha256'], sbatch_sha256=plan['inference_sbatch_sha256']).items():
        assert inference_plan[key] == value, key
    for filename, key in [('run-inference.py', 'inference_script_sha256'), ('run-inference.sbatch', 'inference_sbatch_sha256')]:
        assert sha(GPU/filename) == plan[key]
        evidence[str(GPU/filename)] = plan[key]
    assert download['freeze_sha256'] == sha(REPLAY/'FROZEN.json') == sha(ORIGINAL/'FROZEN.json') == FREEZE
    sample_path = BASE/'release-workspace/interim-matched-artifact-check/validation-inference-v1/verification.json'
    sample = retain(sample_path, inference_plan['validation_sample_proof_sha256'])
    assert sample['status'] == 'passed' and sample['checkpoint_files'] == 12 and sample['all_validation_metrics_exact'] is True
    for filename, key in [('matched-environment-verification.json', 'environment_proof_sha256'),
                          ('matched-gpu-54758758.json', 'gpu_environment_proof_sha256')]:
        assert retain(BASE/'inference-environments-v1'/filename, inference_plan[key])['status'] == 'passed'
    native = subprocess.check_output(['sacct', '--array', '-X', '-n', '-P', '-j', str(plan['inference_array']),
        '--format=JobID%40,JobIDRaw%40,State,ExitCode'], text=True)
    jobs = native_tasks(native, plan['inference_array'])
    with (ROOT/'inference-native-completion.log').open('x') as stream: stream.write(native)
    for fold in range(10):
        folder = GPU/'gpu-runs'/f'fold{fold}'
        start = retain(folder/'STARTED.json')
        assert start == dict(fold=fold, inference_plan_sha256=plan['inference_plan_sha256'],
            download_proof_sha256=plan['download_proof_sha256'], freeze_sha256=FREEZE,
            operation_script_sha256=plan['inference_script_sha256'], training_source_commit=TRAINING)
        complete = retain(folder/'COMPLETE.json')
        assert complete['status'] == 'inference_completed' and complete['fold'] == fold
        assert complete['training_runs'] == 10 and complete['score_files'] == 40
        assert complete['started_proof_sha256'] == sha(folder/'STARTED.json')
        assert Path(complete['environment_file']).name == complete['environment_file']
        attempt = retain(folder/complete['environment_file'], complete['environment_sha256'])
        wanted = dict(job_id=jobs[fold], fold=fold, torch='2.10.0+cu128', numpy='2.4.2',
            pykeen='1.11.1', capability=[8,0], cpu_threads=8, started_proof_sha256=sha(folder/'STARTED.json'))
        assert all(attempt[k] == v for k,v in wanted.items()), fold
        expected_runs = {f'fold{fold}/{g}/{m}' for g in ('original','nobridge') for m in FAMILIES}
        assert len(complete['records']) == 10 and {r['run'] for r in complete['records']} == expected_runs
        for record in complete['records']:
            run = REPLAY/record['run']
            scored = retain(run/'SCORED.json', record['scored_sha256'])
            assert scored['files'] == record['files']
            assert set(record['files']) == {'scores-'+c+'.npz' for c in CHOICES}
            # The public inference comparator below checks every prediction and weight byte.
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
        scope='All ten public-checkpoint inference tasks completed; 400 score comparisons, ten primary contrasts, all 40 x 42 bias conditions, 5600 rank arrays, and primary/bias tables agree with verified production. No independent retraining or external-cohort validation claim.')
    with (ROOT/'ANALYSIS_COMPLETE.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_evidence_sha256','outputs')}, indent=2), flush=True)


if __name__ == '__main__': main()
