"""Stage a boundary-expansion pretraining campaign directory (login node, no GPU, no submission).

Writes the frozen protocol copy, one config and runtime file per trajectory, the batch script, the release
template and a manifest that pins every campaign file and every executed source file in the code checkout.
The graph and the parent state must already be in place; their hashes are checked against the protocol.
Refuses to overwrite an existing manifest.
"""
import argparse, hashlib, json, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CODE_FILES = ('train.py', 'complex_model.py', 'state_compare.py', 'controller.py', 'submit.py', 'monitor.py', 'stage.py', 'protocol.json')
PARENT_FILES = ('training/COMPLETE.json', 'training/identity.json', 'training/progress.json', 'training/updates.jsonl')


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def write(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def git(*arguments):
    return subprocess.run(['git', '-C', str(HERE)] + list(arguments), capture_output=True, text=True, timeout=60, check=True).stdout


def verified_checkout(revision):
    """Require the package to be committed, clean and at the pushed upstream revision; return the Git record.

    Only this package directory is checked; edits elsewhere in the working tree are ignored.
    """
    head = git('rev-parse', 'HEAD').strip()
    assert head == revision, {'git_head': head, 'code_revision': revision}
    upstream = git('rev-parse', '@{upstream}').strip()
    assert upstream == head, {'git_head': head, 'upstream': upstream, 'reason': 'staged revision must be the pushed and pulled commit'}
    status = git('status', '--porcelain', '--untracked-files=all', '--ignored', '--', '.')
    unexpected = [line for line in status.splitlines() if not line.endswith('__pycache__/') and '/__pycache__/' not in line]
    assert not unexpected, {'package_not_clean': unexpected}
    tracked = {}
    for line in git('ls-files', '-s', '--', '.').splitlines():
        meta, path = line.split('\t', 1)
        tracked[pathlib.Path(path).name] = meta.split()[1]
    for name in CODE_FILES:
        assert name in tracked, ('executed file not tracked', name)
        assert git('hash-object', str(HERE / name)).strip() == tracked[name], ('working file differs from committed blob', name)
    return {'head': head, 'upstream': upstream, 'branch': git('rev-parse', '--abbrev-ref', 'HEAD').strip(), 'package_status_clean': True,
            'tracked_blob_ids': tracked, 'top_level': git('rev-parse', '--show-toplevel').strip()}


def job_script(python, code_dir, campaign, cpus, memory):
    return '\n'.join(['#!/bin/bash', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=%d' % cpus, '#SBATCH --mem=%s' % memory, 'set -uo pipefail',
                      'export OMP_NUM_THREADS=1', 'export OPENBLAS_NUM_THREADS=1', 'export CUBLAS_WORKSPACE_CONFIG=:4096:8',
                      'case "$1" in %s) ;; *) echo "campaign path mismatch" >&2; exit 2;; esac' % campaign,
                      'exec %s -u %s/controller.py --campaign "$1" --trial "$2" --wave "$3"' % (python, code_dir), ''])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=pathlib.Path, required=True)
    parser.add_argument('--graph-dir', type=pathlib.Path, required=True)
    parser.add_argument('--parent-dir', type=pathlib.Path, required=True, help='parent trajectory directory containing training/')
    parser.add_argument('--python', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--qos', required=True)
    parser.add_argument('--account', default='')
    parser.add_argument('--gpu-request', required=True)
    parser.add_argument('--gpu-name-substring', required=True)
    parser.add_argument('--job-name-prefix', default='lgdabd')
    parser.add_argument('--initial-time', default='03:00:00')
    parser.add_argument('--long-time', default='12:00:00')
    parser.add_argument('--fresh-allocations', type=int, default=8)
    parser.add_argument('--resume-allocations', type=int, default=5)
    parser.add_argument('--required-free-gib', type=int, default=60)
    parser.add_argument('--storage-command', nargs='*', default=[])
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory', default='64G')
    parser.add_argument('--checkpoint-every-seconds', type=int, default=600)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    assert not (campaign / 'frozen-manifest.json').exists(), 'manifest exists: staging is write-once'
    checkout = verified_checkout(args.code_revision)
    protocol_source = HERE / 'protocol.json'
    protocol = read(protocol_source)
    assert protocol['status'] == 'FROZEN'
    protocol_path = campaign / 'config/boundary-protocol.json'
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol_source, protocol_path)
    protocol_sha = sha(protocol_path)
    assert protocol_sha == sha(protocol_source)
    graph = args.graph_dir.resolve()
    assert campaign in graph.parents
    completion = read(graph / 'COMPLETE.json')
    graph_sha = sha(graph / 'COMPLETE.json')
    assert protocol['admitted_graph_complete_sha256'][graph_sha] == 'original'
    for name, record in completion['files'].items():
        assert sha(graph / name) == (record['sha256'] if isinstance(record, dict) else record), name
    parent_dir = args.parent_dir.resolve()
    assert campaign in parent_dir.parents
    admitted = protocol['admitted_import_parents'][0]
    parent_training = parent_dir / 'training'
    assert sha(parent_training / 'COMPLETE.json') == admitted['complete_sha256']
    done = read(parent_training / 'COMPLETE.json')
    assert done['identity_sha256'] == admitted['identity_sha256'] and done['steps'] == admitted['steps']
    progress = read(parent_training / 'progress.json')
    assert progress['checkpoint_sha256'] == admitted['checkpoint_sha256'] and progress['step'] == admitted['steps']
    assert sha(parent_training / progress['checkpoint']) == admitted['checkpoint_sha256'] == done['files'][progress['checkpoint']]
    for name in ('identity.json', 'progress.json', 'updates.jsonl'):
        assert sha(parent_training / name) == done['files'][name], name
    parent_config = read(parent_training / 'identity.json')['config']
    sources = {name: sha(HERE / name) for name in ('train.py', 'complex_model.py', 'state_compare.py')}
    assert sources['complex_model.py'] == read(parent_training / 'identity.json')['sources']['complex_model.py']
    base = {'batch_size': 1000, 'checkpoint_every_updates': 42230, 'checkpoint_every_seconds': args.checkpoint_every_seconds, 'data_complete_sha256': graph_sha,
            'data_dir': str(graph), 'dtype': 'float32', 'entity_count': parent_config['entity_count'], 'init_size': .001, 'maximum_seconds': 43200,
            'maximum_steps': protocol['pretrain_total_updates'], 'milestone_updates': protocol['pretrain_milestone_updates'], 'optimizer': 'Adagrad',
            'protocol_path': str(protocol_path), 'protocol_sha256': protocol_sha, 'rank': 400, 'regularization': .05, 'relation_count': parent_config['relation_count'],
            'rolling_full_checkpoints': 2, 'seed': 0, 'source_hashes': sources, 'steps_per_epoch': protocol['base_steps_per_epoch'], 'strict_deterministic_algorithms': True,
            'taxonomy_mode': 'original', 'training_triples': parent_config['training_triples']}
    for key in ('rank', 'init_size', 'batch_size', 'optimizer', 'seed', 'dtype', 'regularization', 'training_triples', 'entity_count', 'relation_count', 'steps_per_epoch', 'data_complete_sha256'):
        assert base[key] == parent_config[key], key
    trials = {}
    for learning_rate, name in ((.2, 'lr0.2-resume633450'), (.3, 'lr0.3-fresh'), (.4, 'lr0.4-fresh')):
        config = dict(base, learning_rate=learning_rate, campaign_trial_id='BO-original-plr%s-n3r0.05-u%d%s' % (learning_rate, protocol['pretrain_total_updates'], '-from633450' if learning_rate == .2 else ''))
        if learning_rate == .2:
            assert admitted['learning_rate'] == learning_rate
            config['import_parent'] = {'training_dir': str(parent_training), 'complete_sha256': admitted['complete_sha256'], 'checkpoint_sha256': admitted['checkpoint_sha256'],
                                       'identity_sha256': admitted['identity_sha256'], 'steps': admitted['steps']}
        assert [config['learning_rate'], config['regularization']] in protocol['pretrain_grid']
        write(campaign / 'trials' / name / 'config.json', config)
        write(campaign / 'trials' / name / 'runtime.json', {'mode': 'pretrain', 'worker': 'train.py',
                                                             'maximum_allocations': args.resume_allocations if learning_rate == .2 else args.fresh_allocations})
        trials[name] = config['campaign_trial_id']
    (campaign / 'ops').mkdir(exist_ok=True)
    (campaign / 'ops/job.sh').write_text(job_script(args.python, HERE, campaign, args.cpus, args.memory))
    (campaign / 'logs').mkdir(exist_ok=True)
    template = {'slurm_request': {'partition': args.partition, 'qos': args.qos, 'account': args.account, 'time': args.initial_time, 'job_name_prefix': args.job_name_prefix},
                'gpu_request': args.gpu_request, 'gpu_name_substring': args.gpu_name_substring, 'allowed_time_limits': sorted({args.initial_time, args.long_time}),
                'save_margin_seconds': 600, 'exit_margin_seconds': 180, 'hard_stop_margin_seconds': 90, 'stop_budget_seconds': 150, 'worker_cpus': args.cpus,
                'required_free_bytes': args.required_free_gib * 1024**3, 'initial_trials': ['trials/' + name for name in trials], 'site_commands': {'storage': args.storage_command},
                'utilization_policy': {'threshold_percent': 75, 'sample_interval_seconds': 5, 'window_seconds': 3600, 'minimum_window_coverage': .8,
                                       'stop_after_warning_seconds': 1800, 'stall_seconds': 900,
                                       'early_guard': {'after_seconds': 1500, 'window_seconds': 1200, 'threshold_percent': 60},
                                       'long_continuation': {'time_limit': args.long_time, 'operator_file': 'LONG_CONTINUATION.json', 'minimum_allocation_mean_percent': 75,
                                                             'minimum_wall_coverage': .9, 'require_rolling_hour_minimum': True}}}
    write(campaign / 'config/release-template.json', template)
    files = {}
    for path in sorted(p for p in campaign.rglob('*') if p.is_file()):
        relative = path.relative_to(campaign)
        if relative.parts[0] in ('config', 'trials', 'ops', 'inputs'):
            files[str(relative)] = sha(path)
    parent_files = {}
    for name in PARENT_FILES + (str(pathlib.Path('training') / progress['checkpoint']),):
        parent_files[str((parent_dir / name).relative_to(campaign))] = sha(parent_dir / name)
    manifest = {'code_dir': str(HERE), 'code_revision': args.code_revision, 'git': checkout, 'code_files': {name: sha(HERE / name) for name in CODE_FILES},
                'protocol_sha256': protocol_sha, 'graph_complete_sha256': graph_sha, 'trials': trials, 'files': files, 'parent_files': parent_files,
                'parent_files_note': 'verified at staging and again by the worker at import; not rehashed at every allocation start', 'python': args.python, 'argv': sys.argv}
    write(campaign / 'frozen-manifest.json', manifest)
    print(json.dumps({'campaign': str(campaign), 'manifest_sha256': sha(campaign / 'frozen-manifest.json'), 'trials': trials, 'code_revision': args.code_revision}, indent=2))


if __name__ == '__main__':
    main()
