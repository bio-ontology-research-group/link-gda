"""Shared helpers for the boundary query campaign: immutable writes, hashing, manifest checks and trial configs.

The write helpers and the query/continuation config builders are ported from the executed clmpt-axiom13-hpo
materializer (ops/materialize_campaign.py, sha256 313cb962...). Config builders keep the same keys and values so
a new screen differs from the control screen only in its initializer, protocol pin and paths.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
JOB_NAME_PREFIX = 'lgdaq-'
QUERY_SOURCES = ('train.py', 'sampler.py', 'model.py', 'original_model.py', 'packed_encode.py', 'evaluate.py', 'metrics.py', 'state_compare.py')
REFERENCES = ('official-clmpt.py', 'official-complex.py', 'official-training.py', 'official-transformer.py')
CODE_FILES = ('README.md', 'requirements-query.txt', 'protocol.json', 'campaign.py', 'controller.py', 'advance.py', 'stage.py', 'select_boundary.py',
              'bridge.py', 'job-template.sh', 'select-job-template.sh', 'analysis/select_stage.py') + tuple('query/code/' + name for name in QUERY_SOURCES)
WORKER = 'query/code/train.py'
QUERY_FIXED = {'batch_size': 128, 'dropout': 0.1, 'dtype': 'float32', 'heads': 8, 'maximum_seconds': 13200, 'negative_sample_size': 128,
               'optimizer': 'AdamW', 'profile_batch_size': 32, 'rank': 400, 'seed': 0, 'strict_deterministic_algorithms': True, 'transformer_layers': 2}


class Refusal(Exception):
    pass


def check(condition, message):
    if not condition:
        raise Refusal(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inside(campaign, path):
    """Absolute, normalized, symlink-free path inside the campaign."""
    path = Path(os.path.normpath(str(path)))
    check(path.is_absolute(), 'path must be absolute: %s' % path)
    check(str(path) == str(campaign) or str(path).startswith(str(campaign) + os.sep), 'path outside campaign: %s' % path)
    probe = path
    while str(probe) != str(campaign):
        check(not os.path.islink(str(probe)), 'symlink inside campaign: %s' % probe)
        probe = probe.parent
    return path


def write_exclusive(campaign, path, value):
    """Create a file once; an existing file must hold identical bytes. Returns 'created' or 'identical'."""
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else encoded(value)
    if os.path.lexists(str(path)):
        check(path.read_bytes() == data, 'immutable file differs: %s' % path)
        return 'identical'
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(str(temporary), str(path))
    finally:
        os.unlink(str(temporary))
    fsync_directory(path.parent)
    check(path.read_bytes() == data, 'immutable file differs after racing writer: %s' % path)
    return 'created'


def write_replace(campaign, path, value):
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    with temporary.open('wb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_line(campaign, path, value):
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(json.dumps(dict(value, time=time.time(), pid=os.getpid()), sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def campaign_root(argument):
    given = Path(argument)
    check(given.is_absolute() and os.path.realpath(str(given)) == str(given), 'campaign must be an absolute symlink-free path: %s' % given)
    return given


def verify_manifest(campaign, bank_full=False):
    """Frozen-manifest context: campaign files and checkout package files by hash; the bank by COMPLETE hash and sizes.

    Full bank hashing is done at staging and by the worker at every allocation start; bank_full repeats it here.
    """
    manifest_path = campaign / 'frozen-manifest.json'
    check(manifest_path.exists(), 'campaign not staged: frozen-manifest.json missing')
    manifest = read_json(manifest_path)
    check(manifest.get('schema') == 'clmpt-boundary-queries-manifest/1' and manifest['campaign'] == str(campaign), 'manifest does not describe this campaign')
    for name, digest in manifest['files'].items():
        check(sha(inside(campaign, campaign / name)) == digest, 'frozen campaign file changed: %s' % name)
    code_dir = Path(manifest['code_dir'])
    for name, digest in manifest['code_files'].items():
        check(sha(code_dir / name) == digest, 'checkout package file changed: %s' % name)
    check(sha(HERE / 'campaign.py') == manifest['code_files']['campaign.py'] and HERE == code_dir, 'running helper differs from the frozen checkout')
    bank = manifest['bank']
    check(sha(Path(bank['dir']) / 'COMPLETE.json') == bank['complete_sha256'], 'bank COMPLETE changed')
    for name, record in bank['files'].items():
        path = Path(bank['dir']) / name
        check(path.stat().st_size == record['bytes'], 'bank file size changed: %s' % name)
        if bank_full:
            check(sha(path) == record['sha256'], 'bank file hash changed: %s' % name)
    protocol_path = campaign / 'config' / 'protocol.json'
    protocol = read_json(protocol_path)
    check(protocol['status'] == 'FROZEN' and sha(protocol_path) == manifest['protocol_sha256'], 'protocol not frozen or not pinned')
    return {'campaign': campaign, 'manifest': manifest, 'manifest_sha256': sha(manifest_path), 'protocol': protocol, 'protocol_path': str(protocol_path),
            'protocol_sha256': manifest['protocol_sha256'], 'query_sources': dict(protocol['query_source_hashes']), 'bank': bank, 'code_dir': code_dir,
            'python': manifest['python']}


def fmt(value):
    return repr(float(value))


def query_id(stage, mode, plr, n3, budget, qlr, temperature):
    return '%s-%s-plr%s-n3r%s-u%d-qlr%s-t%s' % (stage, mode, fmt(plr), fmt(n3), int(budget), fmt(qlr), fmt(temperature))


def trial_dir(campaign, stage, identifier):
    return Path(campaign) / 'trials' / stage / identifier


def job_name(identifier):
    return JOB_NAME_PREFIX + identifier


def query_config(context, initializer, learning_rate, temperature, steps, mixture_id, identifier):
    """Fresh query trial from an admitted initializer record; same keys and values as the executed materializer."""
    protocol, mode = context['protocol'], initializer['taxonomy_mode']
    check(mode in protocol['taxonomy_modes'], 'initializer taxonomy mode not admitted')
    check(initializer['initializer_update_budget'] in protocol['initializer_update_budgets'], 'initializer budget not admitted')
    check(learning_rate in protocol['query_learning_rates'] and temperature in protocol['temperatures'] and steps in protocol['maximum_steps'], 'query cell outside the frozen grid')
    check(mixture_id in protocol['mixtures'], 'mixture not in the frozen protocol')
    config = dict(QUERY_FIXED)
    config.update({'checkpoint_retention': protocol['checkpoint_retention'], 'data_dir': context['bank']['dir'],
                   'data_complete_sha256': context['bank']['complete_sha256'], 'evaluation_every_updates': protocol['evaluation_every_updates'],
                   'initializer_taxonomy_mode': mode, 'initializer_update_budget': initializer['initializer_update_budget'],
                   'learning_rate': learning_rate, 'log_negative_ids': protocol['log_negative_ids'], 'maximum_steps': steps,
                   'mixture_id': mixture_id, 'mixture': protocol['mixtures'][mixture_id],
                   'pretrain_learning_rate': initializer['pretrain_learning_rate'], 'pretrain_regularization': initializer['pretrain_regularization'],
                   'pretrained_checkpoint': initializer['path'], 'pretrained_sha256': initializer['sha256'],
                   'protocol_path': context['protocol_path'], 'protocol_sha256': context['protocol_sha256'],
                   'source_hashes': dict(context['query_sources']), 'taxonomy_mode': mode, 'temperature': temperature,
                   'weight_decay': protocol['query_weight_decay'], 'campaign_trial_id': identifier})
    return config


def continuation_config(context, parent_config, parent_training_dir, parent_complete_sha256, parent_checkpoint_sha256, identifier):
    protocol = context['protocol']
    check(parent_config['maximum_steps'] == protocol['continuation_parent_steps'] and not parent_config.get('continuation_parent'), 'continuation parent is not a fresh 2500-update trial')
    check(parent_config['protocol_sha256'] == context['protocol_sha256'] and parent_config['source_hashes'] == context['query_sources'], 'continuation parent was configured under other frozen inputs')
    config = dict(parent_config)
    config.update({'maximum_steps': protocol['continuation_total_steps'], 'continuation_parent': str(parent_training_dir),
                   'parent_complete_sha256': parent_complete_sha256, 'parent_checkpoint_sha256': parent_checkpoint_sha256,
                   'campaign_trial_id': identifier})
    return config


def runtime(stage, identifier, maximum_allocations, order_key):
    check(1 <= maximum_allocations <= 16, 'runtime out of range')
    return {'mode': 'query', 'worker': WORKER, 'maximum_allocations': maximum_allocations, 'stage': stage, 'taxonomy_mode': 'original',
            'trial_id': identifier, 'job_name': job_name(identifier), 'order_key': list(order_key)}


def git(directory, *arguments):
    return subprocess.run(['git', '-C', str(directory)] + list(arguments), capture_output=True, text=True, timeout=60, check=True).stdout


def verified_checkout(revision):
    """Require this package to be committed, clean and at the pushed upstream revision; return the Git record.

    Only this package directory is checked; edits elsewhere in the working tree are ignored.
    """
    head = git(HERE, 'rev-parse', 'HEAD').strip()
    check(head == revision, 'git HEAD %s differs from --code-revision %s' % (head, revision))
    upstream = git(HERE, 'rev-parse', '@{upstream}').strip()
    check(upstream == head, 'staged revision must be the pushed and pulled commit: HEAD %s, upstream %s' % (head, upstream))
    status = git(HERE, 'status', '--porcelain', '--untracked-files=all', '--ignored', '--', '.')
    unexpected = [line for line in status.splitlines() if not line.endswith('__pycache__/') and '/__pycache__/' not in line]
    check(not unexpected, 'package not clean: %s' % unexpected)
    tracked = {}
    for line in git(HERE, 'ls-files', '-s', '--', '.').splitlines():
        meta, path = line.split('\t', 1)
        tracked[path] = meta.split()[1]
    for name in CODE_FILES:
        check(name in tracked, 'executed file not tracked: %s' % name)
        check(git(HERE, 'hash-object', str(HERE / name)).strip() == tracked[name], 'working file differs from committed blob: %s' % name)
    return {'head': head, 'upstream': upstream, 'branch': git(HERE, 'rev-parse', '--abbrev-ref', 'HEAD').strip(), 'package_status_clean': True,
            'tracked_blob_ids': tracked, 'top_level': git(HERE, 'rev-parse', '--show-toplevel').strip()}
