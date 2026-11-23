"""Run ONE bounded pause/resume qualification condition inside one GPU allocation.

Builds the canonical config with the existing builders (query/code/make_config.py, pretrain/make_config.py),
validates it, then runs the existing `qualify.py resume`, which runs the real worker uninterrupted, paused
and resumed in three child processes on the allocated GPU and writes its own receipt. Nothing here trains,
compares states or computes losses itself.

Storage split. The two trajectories (uninterrupted and paused-resumed), including their large full
optimizer/RNG checkpoints, live under --scratch (node-local, transient; left for node cleanup). Durable
evidence goes to --output (shared): ATTEMPT.json (identity: condition, inputs, source/protocol/reference
hashes, allocation), the absolute config and its validation receipt, child logs, the unmodified qualifier
receipt `resume-<condition>.json` (schema, subcommand and calls as consumed by ops engine_receipts), every
non-checkpoint file of both trajectories (logs, identities, progress, curves, selections, saved score
matrices and their metadata), ONE terminal model-only checkpoint, and RESULT.json with sha256 of every
copied file. The exact terminal-state comparison is the qualifier receipt; the compared optimizer states
themselves are transient. These are qualification fixtures, not results.

Exit 0 only if the qualifier receipt passed and every copied artifact hash verifies. A failed child halts
the run with RESULT.json status FAIL (evidence copied, nothing retried). An existing output directory is
accepted only when it holds a PASS whose identity and hashes all re-verify; any other existing directory
is refused and needs a new attempt path. --dry-run builds and validates the config and prints the command
without touching a GPU; use a separate --output for it.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
QUERY = HERE.parent
PRETRAIN = QUERY.parent / 'pretrain'
QUERY_PROTOCOL = QUERY / 'protocols' / 'suite-query-v1.json'
PRETRAIN_PROTOCOL = PRETRAIN / 'protocols' / 'suite-pretrain-v1.json'
QUERY_SOURCES = ('train.py', 'model.py', 'original_model.py', 'packed_encode.py', 'sampler.py', 'evaluate.py', 'metrics.py', 'state_compare.py',
                 'make_config.py', 'qualify.py', 'run_resume_qualification.py')
PRETRAIN_SOURCES = ('train.py', 'complex_model.py', 'state_compare.py', 'make_config.py')
REFERENCES = ('official-clmpt.py', 'official-complex.py', 'official-transformer.py', 'official-training.py')
INITIALIZER = {'pretrain_learning_rate': 0.4, 'pretrain_regularization': 0.05, 'initializer_update_budget': 633450,
               'sha256': '81792f17a74e14b0bb0b3fbd4d3475221b0e88ff32ab698e121844e5c5280482'}
QUERY_CONDITIONS = {'shared': ('shared', 'full13', 0), 'independent': ('independent', 'full13', 0), 'mean': ('mean', 'full13', 0),
                    'core3': ('shared', 'core3', 0), 'aux_off': ('shared', 'aux_off', 0), 'seed1': ('shared', 'full13', 1)}
QUERY_SCHEDULE = {'maximum_steps': 20, 'evaluation_every_updates': 10, 'pause_at_step': 15, 'learning_rate': 0.0001, 'temperature': 0.05}
PRETRAIN_SCHEDULE = {'recipe_id': 'plr0.4-n3r0.05-u633450', 'seed': 1, 'maximum_steps': 40, 'pause_at_step': 25, 'checkpoint_every_updates': 10, 'checkpoint_every_seconds': 60}
SCRATCH_MINIMUM_FREE = 24 * 1024**3
SHARED_RESERVE = 8 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(args):
    """Everything that must be equal for a completed output to be accepted again."""
    inputs = {'bank': {'path': args.bank, 'complete_sha256': sha(Path(args.bank) / 'COMPLETE.json')},
              'graph': {'path': args.graph, 'complete_sha256': sha(Path(args.graph) / 'COMPLETE.json')},
              'pretrained': {'path': args.pretrained, 'sha256': sha(args.pretrained)}}
    return {'condition': args.condition, 'checkout_query_code': str(HERE), 'interpreter': sys.executable, 'inputs': inputs,
            'query_sources': {name: sha(HERE / name) for name in QUERY_SOURCES},
            'pretrain_sources': {name: sha(PRETRAIN / name) for name in PRETRAIN_SOURCES},
            'references': {name: sha(QUERY / 'references' / name) for name in REFERENCES},
            'protocols': {'query': {'path': str(QUERY_PROTOCOL), 'sha256': sha(QUERY_PROTOCOL)}, 'pretrain': {'path': str(PRETRAIN_PROTOCOL), 'sha256': sha(PRETRAIN_PROTOCOL)}},
            'query_schedule': QUERY_SCHEDULE, 'pretrain_schedule': PRETRAIN_SCHEDULE, 'initializer': INITIALIZER}


def run_logged(command, logs, name, environment=None):
    """Run a child with stdout/stderr captured to shared log files; return its return code."""
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / (name + '.stdout.log')).open('ab') as out, (logs / (name + '.stderr.log')).open('ab') as err:
        out.write(('\n$ %s\n' % ' '.join(command)).encode())
        out.flush()
        return subprocess.run(command, stdout=out, stderr=err, env=environment).returncode


def build_query_config(args, config_path, logs):
    encoder, curriculum, seed = QUERY_CONDITIONS[args.condition]
    command = [sys.executable, str(HERE / 'make_config.py'), 'build', '--protocol', str(QUERY_PROTOCOL), '--purpose', 'qualification',
               '--experiment-id', 'qual-resume-%s' % args.condition, '--data-dir', args.bank, '--pretrained', args.pretrained,
               '--cohort', 'main', '--modality', 'PFS', '--fold', '0', '--pretrain-seed', '0',
               '--pretrain-learning-rate', str(INITIALIZER['pretrain_learning_rate']), '--pretrain-regularization', str(INITIALIZER['pretrain_regularization']),
               '--initializer-update-budget', str(INITIALIZER['initializer_update_budget']), '--seed', str(seed), '--encoder', encoder, '--curriculum', curriculum,
               '--learning-rate', str(QUERY_SCHEDULE['learning_rate']), '--temperature', str(QUERY_SCHEDULE['temperature']),
               '--maximum-steps', str(QUERY_SCHEDULE['maximum_steps']), '--evaluation-every-updates', str(QUERY_SCHEDULE['evaluation_every_updates']),
               '--output', str(config_path)]
    assert run_logged(command, logs, 'build-config') == 0, 'query config build failed; see logs/build-config.*'
    validation = config_path.with_name('config-validation.json')
    command = [sys.executable, str(HERE / 'make_config.py'), 'validate', '--config', str(config_path), '--output', str(validation)]
    assert run_logged(command, logs, 'validate-config') == 0, 'query config validation failed; see logs/validate-config.*'
    receipt = json.loads(validation.read_text())
    assert receipt['passed'] is True and receipt['config_sha256'] == sha(config_path)
    config = json.loads(config_path.read_text())
    assert config['sampler_seed'] == config['seed'] == seed and config['encoder'] == encoder and config['curriculum'] == curriculum
    assert config['purpose'] == 'qualification' and config['pretrained_sha256'] == INITIALIZER['sha256']
    return {'config_sha256': sha(config_path), 'validation': str(validation), 'validation_sha256': sha(validation)}


def pretrain_module():
    sys.path.append(str(PRETRAIN))
    spec = importlib.util.spec_from_file_location('suite_pretrain_train_for_qualification', PRETRAIN / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pretrain_config(args, config_path, logs):
    schedule = PRETRAIN_SCHEDULE
    command = [sys.executable, str(PRETRAIN / 'make_config.py'), '--protocol', str(PRETRAIN_PROTOCOL), '--purpose', 'qualification',
               '--experiment-id', 'qual-resume-pretrain-s%d' % schedule['seed'], '--data-dir', args.graph, '--recipe-id', schedule['recipe_id'],
               '--seed', str(schedule['seed']), '--maximum-steps', str(schedule['maximum_steps']),
               '--checkpoint-every-updates', str(schedule['checkpoint_every_updates']), '--checkpoint-every-seconds', str(schedule['checkpoint_every_seconds']),
               '--output', str(config_path)]
    assert run_logged(command, logs, 'build-config') == 0, 'pretrain config build failed; see logs/build-config.*'
    config = json.loads(config_path.read_text())
    module = pretrain_module()
    protocol = json.loads(Path(config['protocol_path']).read_text())
    checks = {'protocol_sha256': sha(config['protocol_path']) == config['protocol_sha256'],
              'source_hashes': config['source_hashes'] == {name: sha(PRETRAIN / name) for name in module.SOURCE_FILES},
              'graph_complete_sha256': sha(Path(config['data_dir']) / 'COMPLETE.json') == config['data_complete_sha256'],
              'schedule': config['seed'] == schedule['seed'] and config['maximum_steps'] == schedule['maximum_steps'] and config['purpose'] == 'qualification'}
    module.check_config(config, protocol)
    checks['check_config'] = True
    validation = config_path.with_name('config-validation.json')
    dump(validation, {'config': str(config_path), 'config_sha256': sha(config_path), 'checks': checks, 'passed': all(checks.values())})
    assert all(checks.values()), checks
    return {'config_sha256': sha(config_path), 'validation': str(validation), 'validation_sha256': sha(validation)}


def qualifier_command(args, config_path, workdir, receipt):
    pause = PRETRAIN_SCHEDULE['pause_at_step'] if args.condition == 'pretrain' else QUERY_SCHEDULE['pause_at_step']
    return [sys.executable, str(HERE / 'qualify.py'), 'resume', '--engine', 'pretrain' if args.condition == 'pretrain' else 'query',
            '--config', str(config_path), '--pause-at-step', str(pause), '--workdir', str(workdir), '--output', str(receipt)]


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    shutil.copyfile(source, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    digest = sha(target)
    assert digest == sha(source), ('copy mismatch', str(source))
    return digest


def evidence_files(run):
    """Every file of a trajectory except full checkpoints and model exports (transient or copied separately)."""
    return sorted(path for path in run.rglob('*') if path.is_file() and path.suffix not in ('.pt', '.tmp') and path.name != 'writer.lock')


def terminal_model(args, workdir):
    """Path of the paused-resumed run's terminal model state: a model-only export, written to shared output later."""
    run = workdir / 'paused-resumed'
    progress = json.loads((run / 'progress.json').read_text())
    if args.condition == 'pretrain':
        name = 'models/update-%07d.pt' % PRETRAIN_SCHEDULE['maximum_steps']
        assert name in progress['artifacts'] and sha(run / name) == progress['artifacts'][name]
        return run / name, None
    assert progress['step'] == QUERY_SCHEDULE['maximum_steps'] and progress['pending_evaluation'] is None
    full = run / progress['checkpoint']
    assert sha(full) == progress['checkpoint_sha256']
    return full, progress


def export_query_terminal(full, progress, target):
    """Model-only state of the terminal full checkpoint, in the trainer's export format (aliases stored once)."""
    import torch
    state = torch.load(full, map_location='cpu', weights_only=False)
    model = state['model']
    aliases = {key: 'nbp.' + key[len('layer.nbp.'):] for key in model if key.startswith('layer.nbp.')}
    assert all(torch.equal(model[key], model[target_key]) for key, target_key in aliases.items())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    torch.save({'identity_sha256': state['identity_sha256'], 'step': state['step'], 'model': {key: value for key, value in model.items() if key not in aliases},
                'aliases': aliases, 'source_full_checkpoint_sha256': progress['checkpoint_sha256'], 'kind': 'terminal_model_only_qualification_fixture'}, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return sha(target)


def verify_existing(output, current):
    """Accept an existing output only as a fully re-verified PASS of the same identity."""
    result_path, attempt_path = output / 'RESULT.json', output / 'ATTEMPT.json'
    assert result_path.exists() and attempt_path.exists(), 'existing output without a completed RESULT: use a new attempt path'
    result, attempt = json.loads(result_path.read_text()), json.loads(attempt_path.read_text())
    assert result['status'] == 'PASS', 'existing output did not pass (%s): use a new attempt path' % result['status']
    assert canonical(attempt['identity']) == canonical(current) == result['identity_sha256'], 'existing output has a different identity: use a new attempt path'
    assert sha(attempt_path) == result['attempt_sha256']
    for name, digest in result['files'].items():
        assert sha(output / name) == digest, ('existing artifact changed', name)
    receipt = json.loads((output / result['receipt']).read_text())
    assert receipt['passed'] is True and receipt['subcommand'] == 'resume'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--condition', required=True, choices=sorted(QUERY_CONDITIONS) + ['pretrain'])
    parser.add_argument('--bank', required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True, help='shared, durable, per-attempt directory')
    parser.add_argument('--scratch', required=True, help='node-local directory for transient trajectories')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for name in ('bank', 'graph', 'pretrained', 'output', 'scratch'):
        assert Path(getattr(args, name)).is_absolute(), '--%s must be absolute' % name
    output, scratch = Path(args.output), Path(args.scratch)
    current = identity(args)
    assert current['inputs']['pretrained']['sha256'] == INITIALIZER['sha256'], 'pretrained export is not the LR 0.4 / 633450 initializer'
    assert all((QUERY / 'references' / name).is_file() for name in REFERENCES)
    if output.exists():
        assert not args.dry_run, 'dry-run needs a fresh output path'
        result = verify_existing(output, current)
        print(json.dumps({'status': 'ALREADY_PASSED_VERIFIED', 'output': str(output), 'receipt': result['receipt']}))
        return 0
    output.mkdir(parents=True)
    logs = output / 'logs'
    attempt = {'identity': current, 'identity_sha256': canonical(current), 'dry_run': args.dry_run, 'started': time.time(),
               'allocation': {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')},
               'scratch': str(scratch), 'note': 'Trajectories (with full optimizer/RNG states) are transient under scratch; the shared output keeps the proof, evidence and one terminal model-only checkpoint.'}
    dump(output / 'ATTEMPT.json', attempt)
    config_path = output / 'config' / ('%s.json' % args.condition)
    receipt_path = output / ('resume-%s.json' % args.condition)
    workdir = scratch / ('resume-%s-%d' % (args.condition, time.time_ns()))
    result = {'condition': args.condition, 'identity_sha256': attempt['identity_sha256'], 'attempt_sha256': sha(output / 'ATTEMPT.json'),
              'config': str(config_path), 'receipt': receipt_path.name, 'workdir_transient': str(workdir), 'files': {}}
    try:
        built = build_pretrain_config(args, config_path, logs) if args.condition == 'pretrain' else build_query_config(args, config_path, logs)
        result['config_build'] = built
        command = qualifier_command(args, config_path, workdir, receipt_path)
        result['qualifier_command'] = command
        if args.dry_run:
            result.update({'status': 'DRY_RUN', 'gpu_used': False})
            result['files'] = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'RESULT.json'}
            dump(output / 'RESULT.json', result)
            print(json.dumps({'status': 'DRY_RUN', 'config': str(config_path), 'config_sha256': built['config_sha256'], 'command': command}, indent=2))
            return 0
        scratch.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(scratch).free
        assert free >= SCRATCH_MINIMUM_FREE, 'scratch has %d free bytes; need %d' % (free, SCRATCH_MINIMUM_FREE)
        code = run_logged(command, logs, 'qualify-resume')
        result['qualifier_returncode'] = code
        assert receipt_path.exists(), 'qualifier wrote no receipt (return code %d)' % code
        receipt = json.loads(receipt_path.read_text())
        result['receipt_sha256'] = sha(receipt_path)
        copies = []
        for role in ('uninterrupted', 'paused-resumed'):
            run = workdir / role
            if run.exists():
                copies += [(path, Path('trajectories') / role / path.relative_to(run)) for path in evidence_files(run)]
        needed = sum(path.stat().st_size for path, _ in copies)
        passed = code == 0 and receipt.get('passed') is True and receipt.get('subcommand') == 'resume'
        if passed:
            model, progress = terminal_model(args, workdir)
            needed += model.stat().st_size if progress is None else model.stat().st_size // 3
        assert shutil.disk_usage(output).free >= needed + SHARED_RESERVE, 'shared output lacks space for %d bytes plus reserve' % needed
        for source, relative in copies:
            result['files'][str(relative)] = copy_verified(source, output / relative)
        if not passed:
            raise AssertionError('qualifier did not pass (return code %d, passed=%r); evidence copied' % (code, receipt.get('passed')))
        command_config = receipt['calls'][0]['command'][receipt['calls'][0]['command'].index('--config') + 1]
        assert command_config == str(config_path) and sha(config_path) == built['config_sha256']
        target = output / 'terminal-model.pt'
        if progress is None:
            result['files']['terminal-model.pt'] = copy_verified(model, target)
            result['terminal_model_source'] = {'path': str(model), 'kind': 'pretrain milestone model-only export'}
        else:
            result['files']['terminal-model.pt'] = export_query_terminal(model, progress, target)
            result['terminal_model_source'] = {'path': str(model), 'full_checkpoint_sha256': progress['checkpoint_sha256'], 'kind': 'model-only export of the paused-resumed terminal full state'}
        for name in [receipt_path.name, 'ATTEMPT.json'] + [str(path.relative_to(output)) for path in sorted((output / 'config').rglob('*')) if path.is_file()]:
            result['files'][name] = sha(output / name)
        for name, digest in result['files'].items():
            assert sha(output / name) == digest, ('artifact changed after copy', name)
        result.update({'status': 'PASS', 'finished': time.time(), 'shared_bytes': sum((output / name).stat().st_size for name in result['files']),
                       'scope': 'bounded qualification fixture; no accuracy claim; optimizer/RNG states compared by the qualifier and not retained'})
        dump(output / 'RESULT.json', result)
        print(json.dumps({'status': 'PASS', 'output': str(output), 'receipt': receipt_path.name, 'shared_bytes': result['shared_bytes']}))
        return 0
    except BaseException as error:
        result.update({'status': 'FAIL', 'error': repr(error), 'finished': time.time()})
        for name in sorted(path for path in output.rglob('*') if path.is_file() and path.name != 'RESULT.json' and not path.name.endswith('.tmp')):
            result['files'].setdefault(str(name.relative_to(output)), sha(name))
        dump(output / 'RESULT.json', result)
        print(json.dumps({'status': 'FAIL', 'output': str(output), 'error': repr(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
