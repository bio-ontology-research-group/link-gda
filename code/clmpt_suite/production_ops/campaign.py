"""Host-side production campaign (Orix ComplEx pretraining or IBEX query training). Login node, python3, standard library.

stage           Create the campaign once from a deployment bundle and a host DEPLOY.json: verify the pinned source checkout,
                every input this site reads (COMPLETE hashes at the host paths), copy the worker sources, protocols and (IBEX)
                references by hash, render the job script, write unit folders and a write-once frozen manifest.
release         Live capability checks (partition, GPU type, QoS, overtime grace, storage, sbatch --test-only), then a
                write-once ROOT_RELEASE.json with the exact Slurm arguments. Rehearsal bundles never release for GPU work.
tick            Under a lock: classify every unit, build configs just in time through the workers' own config builders,
                submit eligible units within the active-job cap and the storage budget, resubmit HELD clean pauses, and
                (IBEX) prune full optimizer states of archived units. Idempotent; never revives failed or cancelled units.
admit           IBEX: admit one delivered pretraining export from incoming/.partial-<parent> after hashing every file and
                checking its identity against the bundle; atomic rename, so a partial transfer is never admitted.
inventory       Read-only JSON for the bridge: unit states, completed exports (Orix), admitted parents, delivery slots,
                archivable units (IBEX), storage, halts and attention.
ack-archive     IBEX: record a verified Orix archive receipt for one completed unit (enables pruning).
verify-archive  Orix: verify an archive upload (.partial-<unit>) against the expected hashes from stdin and rename it.
stop            Write a stop marker for one unit (or all) and send TERM to its running controller.
status, gate    Read-only.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (Refusal, canonical, check, exclusive_lock, free_bytes, job_token, read_json, run, seconds, sha, tree_bytes,  # noqa: E402
                    write_exclusive, write_replace, append_line)

WAVE = re.compile(r'wave-(\d{3})')
ACTIVE = ('active', 'unknown', 'suspect')
PREFIX = {'orix': 'lgp-', 'ibex': 'lgq-'}
PRETRAIN_SOURCES = ('train.py', 'complex_model.py', 'state_compare.py')
DELIVERED = ('COMPLETE.json', 'identity.json', 'config.json', 'progress.json')


class Halt(Exception):
    pass


def campaign_root(argument):
    path = Path(argument)
    check(path.is_absolute() and os.path.realpath(str(path)) == str(path), 'campaign must be an absolute symlink-free path')
    return path


def event(campaign, value):
    append_line(campaign / 'state' / 'events.jsonl', value)


def halt(campaign, reason, detail=None):
    path = campaign / 'state' / 'HALT.json'
    if not path.exists():
        write_exclusive(path, {'reason': reason, 'detail': detail, 'time': time.time(), 'resolution': 'a person resolves and moves HALT.json aside'})
    event(campaign, {'event': 'halt', 'reason': reason, 'detail': detail})
    raise Halt(reason)


# ---------------------------------------------------------------- stage


def stage(args):
    campaign = campaign_root(args.campaign)
    check(not campaign.exists() or not any(campaign.iterdir()), 'campaign exists and is not empty')
    bundle_path = Path(args.bundle)
    bundle = read_json(bundle_path)
    check(bundle['schema'] == 'clmpt-suite-production-bundle/1', 'not a production bundle')
    deploy = read_json(args.deploy)
    site = deploy['site']
    check(site in ('orix', 'ibex') and deploy['campaign'] == str(campaign), 'DEPLOY.json does not describe this campaign')
    policy = bundle['site_policy'][site]
    source = Path(deploy['source_checkout'])
    suite = source / 'code' / 'clmpt_suite'
    for name, digest in bundle['source']['files'].items():
        check((suite / name).is_file() and sha(suite / name) == digest, 'source checkout differs from the bundle pin: %s' % name)
    if not bundle['rehearsal']:
        head = run(['git', '-C', str(source), 'rev-parse', 'HEAD'], 60)['stdout'].strip()
        check(head == bundle['source']['revision'], 'checkout HEAD %s differs from the bundle revision' % head)
    check(sha(suite / 'production_ops' / 'site-policy.json') == bundle['site_policy_sha256'], 'site policy differs from the bundle')
    inputs_root = Path(deploy['inputs_root'])
    site_units = bundle['units']['pretrain' if site == 'orix' else 'query']
    kinds = ('graph',) if site == 'orix' else ('bank', 'test')
    verified = {}
    for unit in site_units.values():
        record = bundle['inputs'][unit['entry']]
        for kind in kinds:
            directory = inputs_root / record[kind]['relative']
            key = str(directory)
            if key not in verified:
                check(sha(directory / 'COMPLETE.json') == record[kind]['complete_sha256'], 'input COMPLETE differs at %s' % directory)
                manifest = read_json(directory / 'COMPLETE.json')
                for name, digest in manifest['files'].items():
                    expected = digest['sha256'] if isinstance(digest, dict) else digest
                    check((directory / name).is_file(), 'input file missing: %s/%s' % (directory, name))
                    if kind in ('bank', 'test'):
                        check(sha(directory / name) == expected, '%s file differs: %s/%s' % (kind, directory, name))
                verified[key] = record[kind]['complete_sha256']
    for folder in ('bundle', 'protocols', 'ops', 'units', 'state', 'logs', 'incoming', 'initializers', 'worker'):
        (campaign / folder).mkdir(parents=True, exist_ok=True)
    files = {}

    def place(relative, data):
        write_exclusive(campaign / relative, data)
        files[relative] = sha(campaign / relative)
    place('bundle/BUNDLE.json', bundle_path.read_bytes())
    place('DEPLOY.json', Path(args.deploy).read_bytes())
    for kind in ('query', 'pretrain'):
        name = bundle['protocols'][kind]['name']
        place('protocols/' + name, (bundle_path.parent / name).read_bytes())
        check(files['protocols/' + name] == bundle['protocols'][kind]['sha256'], 'bundle protocol copy differs: %s' % name)
    if site == 'orix':
        for name in PRETRAIN_SOURCES + ('make_config.py',):
            place('worker/' + name, (suite / 'pretrain' / name).read_bytes())
        worker = {name: files['worker/' + name] for name in PRETRAIN_SOURCES}
    else:
        for path in sorted((suite / 'query' / 'code').iterdir()):
            if path.is_file() and path.suffix == '.py':
                place('query/code/' + path.name, path.read_bytes())
        protocol = read_json(campaign / 'protocols' / bundle['protocols']['query']['name'])
        references = Path(deploy['references_dir'])
        for name, digest in protocol['reference_hashes'].items():
            check(sha(references / name) == digest, 'reference differs from the protocol pin: %s' % name)
            place('query/references/' + name, (references / name).read_bytes())
        worker = {name: files['query/code/' + name] for name in ('train.py', 'model.py', 'original_model.py', 'packed_encode.py', 'sampler.py', 'evaluate.py', 'metrics.py', 'state_compare.py')}
    environment = '\n'.join('export %s=%s' % (key, shlex.quote(value)) for key, value in sorted(policy['environment'].items()))
    scripts = [('%s-job.sh' % site, 'ops/job.sh')] + ([('ibex-inference-job.sh', 'ops/inference.sh')] if site == 'ibex' else [])
    for template_name, target in scripts:
        text = (HERE / 'jobs' / template_name).read_text().replace('@ENVIRONMENT@', environment).replace('@PYTHON@', shlex.quote(deploy['worker_python']))
        text = text.replace('@CODE_DIR@', shlex.quote(str(HERE)))
        check('@' not in text.replace('$@', ''), 'unrendered placeholder in %s' % template_name)
        place(target, text.encode())
        os.chmod(str(campaign / target), 0o755)
    for unit in site_units.values():
        record = dict(unit, inputs=bundle['inputs'][unit['entry']])
        place('units/%s/unit.json' % unit['id'], record)
    code_files = {name: sha(HERE / name) for name in ('common.py', 'campaign.py', 'controller.py', 'units.py', 'build_query_config.py', 'site-policy.json', 'reuse.py',
                                                      'inference_job.py', 'export_terminal.py', 'readout.py', 'jobs/%s-job.sh' % site, 'jobs/ibex-inference-job.sh')}
    manifest = {'schema': 'clmpt-suite-production-site/1', 'site': site, 'campaign': str(campaign), 'rehearsal': bool(bundle['rehearsal']),
                'bundle_sha256': files['bundle/BUNDLE.json'], 'code_dir': str(HERE), 'code_files': code_files, 'files': files, 'worker_sources': worker,
                'worker_python': deploy['worker_python'], 'inputs_root': str(inputs_root), 'verified_inputs': verified, 'units': sorted(site_units), 'staged': time.time()}
    write_exclusive(campaign / 'frozen-manifest.json', manifest)
    print(json.dumps({'campaign': str(campaign), 'site': site, 'units': len(site_units), 'inputs_verified': len(verified), 'rehearsal': manifest['rehearsal'],
                      'manifest_sha256': sha(campaign / 'frozen-manifest.json')}, indent=2))
    return 0


def load(argument, require_release=True):
    campaign = campaign_root(argument)
    manifest = read_json(campaign / 'frozen-manifest.json')
    check(manifest['campaign'] == str(campaign), 'manifest describes another campaign')
    for name, digest in manifest['files'].items():
        check(sha(campaign / name) == digest, 'frozen campaign file changed: %s' % name)
    check(Path(manifest['code_dir']) == HERE, 'running campaign.py is not the staged checkout')
    for name, digest in manifest['code_files'].items():
        check(sha(HERE / name) == digest, 'staged code changed: %s' % name)
    bundle = read_json(campaign / 'bundle' / 'BUNDLE.json')
    release = read_json(campaign / 'ROOT_RELEASE.json') if (campaign / 'ROOT_RELEASE.json').exists() else None
    approved = bool(release and release.get('approved') is True and release.get('manifest_sha256') == sha(campaign / 'frozen-manifest.json'))
    if require_release:
        check(approved, 'ROOT_RELEASE.json missing or does not approve this manifest')
    site = manifest['site']
    return {'campaign': campaign, 'manifest': manifest, 'bundle': bundle, 'site': site, 'policy': bundle['site_policy'][site], 'release': release if approved else None,
            'deploy': read_json(campaign / 'DEPLOY.json'), 'recipe': bundle['production_recipe']}


# ---------------------------------------------------------------- Slurm commands


def slurm_arguments(context, partition_choice=None):
    policy, site = context['policy'], context['site']
    request = dict(policy['slurm'])
    if site == 'orix':
        if partition_choice in ('pi', 'freecycle'):
            request.update({key: value for key, value in policy['alternative_partitions'][partition_choice].items() if key in ('partition', 'qos')})
        return ['--partition=' + request['partition'], '--qos=' + request['qos'], '--gpus=' + request['gpu_request'], '--ntasks=1',
                '--cpus-per-task=%d' % request['cpus_per_task'], '--mem=' + request['memory'], '--time=' + request['time']]
    return ['--partition=' + request['partition'], '--constraint=' + request['constraint'], '--gres=' + request['gres'], '--ntasks=1',
            '--cpus-per-task=%d' % request['cpus_per_task'], '--mem=' + request['memory'], '--time=' + request['time']]


def inference_arguments(training_arguments, policy):
    rules = policy['inference']
    replaced = {'--time': rules['time'], '--mem': rules['memory'], '--cpus-per-task': str(rules['cpus_per_task'])}
    return [part.split('=', 1)[0] + '=' + replaced[part.split('=', 1)[0]] if part.split('=', 1)[0] in replaced else part for part in training_arguments]


def sbatch_command(campaign, release, unit, wave, dependency=None, test_only=False):
    """The only form in which a unit allocation is submitted (tick, resume and controller successors)."""
    command = ['sbatch'] + (['--test-only'] if test_only else ['--parsable']) + ['--no-requeue', '--job-name=' + release['job_name_prefix'] + unit,
               '--output=' + str(Path(campaign) / 'logs' / '%x-%j.out')] + list(release['slurm_arguments'])
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    return command + [str(Path(campaign) / 'ops' / 'job.sh'), str(campaign), unit, str(wave)]


def inference_command(campaign, release, unit, attempt):
    """Short GPU readout job for one completed unit; same personal-job cap as training."""
    return ['sbatch', '--parsable', '--no-requeue', '--job-name=' + release['job_name_prefix'] + 'test-' + unit, '--output=' + str(Path(campaign) / 'logs' / '%x-%j.out')] \
        + list(release['inference_arguments']) + [str(Path(campaign) / 'ops' / 'inference.sh'), str(campaign), unit, str(attempt)]


def user_jobs():
    result = run(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%T|%j|%P'], 60)
    if result['returncode'] != 0:
        return None
    jobs = {}
    for line in result['stdout'].splitlines():
        if line.strip():
            identifier, state, name, partition = (line.strip().split('|', 3) + ['', '', ''])[:4]
            jobs[identifier] = {'state': state, 'name': name, 'partition': partition, 'base': identifier.split('_')[0]}
    return jobs


# ---------------------------------------------------------------- release


def release(args):
    context = load(args.campaign, require_release=False)
    campaign, site, policy = context['campaign'], context['site'], context['policy']
    check(args.approve and args.authority.strip(), 'release requires --approve and --authority')
    check(not (campaign / 'ROOT_RELEASE.json').exists(), 'release is write-once')
    check(not context['manifest']['rehearsal'] or args.rehearsal_fake_slurm, 'rehearsal campaign: only a fake-Slurm rehearsal release is possible')
    arguments = slurm_arguments(context, args.partition_choice)
    probes = {'sinfo': run(['sinfo', '-h', '-o', '%P|%G|%f'], 60), 'partition': run(['scontrol', 'show', 'partition', arguments[0].split('=', 1)[1], '-o'], 60),
              'overtime': run(['scontrol', 'show', 'config'], 60)}
    if site == 'orix':
        probes['qos'] = run(['sacctmgr', '-nP', 'show', 'qos', format_value(arguments, '--qos'), 'format=name,maxwall,maxtrespu,maxtrespa,grptres,maxjobspu,maxsubmitpu'], 60)
        probes['storage_usage'] = run(['storage_usage'], 60)
        gres = policy['slurm']['gpu_request'].split(':')[0]
        check(gres in probes['sinfo']['stdout'], 'GPU type %s not offered by sinfo; refusing (verify the live identifier)' % gres)
    else:
        match = re.search(r'^OverTimeLimit\s*=\s*(\d+)', probes['overtime']['stdout'], re.MULTILINE)
        probes['overtime_minutes'] = int(match.group(1)) if match else None
    candidate = {'job_name_prefix': PREFIX[site], 'slurm_arguments': arguments}
    probes['test_only'] = run(sbatch_command(campaign, candidate, 'release-probe', 0, test_only=True), 60)
    check(probes['test_only']['returncode'] == 0, 'scheduler refused the job request: %s' % probes['test_only']['stderr'][-1000:])
    write_replace(campaign / 'state' / 'release-probes.json', probes)
    maximum = args.max_active if args.max_active is not None else (policy['max_active_jobs_default'] if site == 'orix' else policy['user_active_job_cap'])
    if site == 'ibex':
        maximum = min(maximum, policy['user_active_job_cap'])
    value = dict(candidate, approved=True, authority=args.authority, manifest_sha256=sha(campaign / 'frozen-manifest.json'), max_active=int(maximum),
                 quota_command=json.loads(args.quota_command) if args.quota_command else None, rehearsal=bool(context['manifest']['rehearsal']),
                 partition_choice=args.partition_choice, probes_sha256=sha(campaign / 'state' / 'release-probes.json'), time=time.time())
    if site == 'ibex':
        value['inference_arguments'] = inference_arguments(arguments, policy)
    write_exclusive(campaign / 'ROOT_RELEASE.json', value)
    if args.storage_decision:
        record_storage_decision(campaign, args.storage_decision, args.authority)
    print(json.dumps({key: value.get(key) for key in ('max_active', 'slurm_arguments', 'inference_arguments', 'rehearsal')}, indent=2))
    return 0


STORAGE_FIELDS = ('release_new_production_reportables_after_verified_archive', 'archive_current_hpo_reportables', 'release_current_hpo_reportables_after_verified_archive')


def storage_decision(campaign):
    """The recorded explicit storage decision; every permission is False without one."""
    path = Path(campaign) / 'state' / 'STORAGE-DECISION.json'
    value = read_json(path) if path.exists() else {}
    result = {field: value.get(field) is True for field in STORAGE_FIELDS}
    result.update(recorded=path.exists(), sha256=sha(path) if path.exists() else None)
    return result


def record_storage_decision(campaign, decision_path, authority):
    value = read_json(decision_path)
    check(value.get('schema') == 'clmpt-suite-storage-decision/1' and all(isinstance(value.get(field), bool) for field in STORAGE_FIELDS),
          'storage decision must use schema clmpt-suite-storage-decision/1 and give every field as true or false')
    check(str(value.get('decided_by', '')).strip() and str(value.get('date', '')).strip(), 'storage decision must name who decided and when')
    write_exclusive(Path(campaign) / 'state' / 'STORAGE-DECISION.json', Path(decision_path).read_bytes())
    event(Path(campaign), {'event': 'storage_decision_recorded', 'recorded_by': authority, 'decision': {field: value[field] for field in STORAGE_FIELDS}})


def record_storage(args):
    context = load(args.campaign)
    record_storage_decision(context['campaign'], args.decision, args.authority)
    print(json.dumps(storage_decision(context['campaign']), indent=2))
    return 0


def format_value(arguments, flag):
    return next(part.split('=', 1)[1] for part in arguments if part.startswith(flag + '='))


# ---------------------------------------------------------------- classification


def unit_dir(campaign, unit):
    return Path(campaign) / 'units' / unit


def waves(folder):
    allocations = folder / 'allocations'
    if not allocations.is_dir():
        return []
    return sorted((int(WAVE.fullmatch(path.name).group(1)), path) for path in allocations.iterdir() if WAVE.fullmatch(path.name))


def recorded_jobs(folder):
    jobs = set()
    for path in [folder / 'submission-receipt.json'] + sorted((folder / 'resubmissions').glob('wave-*.json') if (folder / 'resubmissions').is_dir() else []):
        if path.exists() and not path.name.endswith('.SUBMITTING.json'):
            token = job_token(read_json(path))
            if token:
                jobs.add(token)
    for _, wave in waves(folder):
        for name in ('allocation.json', 'successor.json'):
            if (wave / name).exists():
                token = str(read_json(wave / name).get('job', ''))
                if token.isdigit():
                    jobs.add(token)
    return jobs


def stop_marker(campaign, folder):
    return (campaign / 'OPERATOR_STOP.json').exists() or (folder / 'STOP.json').exists()


def reuse_record(context, kind):
    """The bundle's passed FROZEN_HPO_ARTIFACT audit for 'pretrain' or 'query', else None."""
    record = (context['bundle'].get('reuse') or {}).get(kind)
    return record if record and record.get('passed') is True and record.get('kind') == 'FROZEN_HPO_ARTIFACT' else None


def training_dir(context, unit):
    reused = unit_dir(context['campaign'], unit) / 'REUSED.json'
    return Path(read_json(reused)['training_dir']) if reused.exists() else unit_dir(context['campaign'], unit) / 'training'


def classify(context, unit, jobs):
    campaign, folder = context['campaign'], unit_dir(context['campaign'], unit)
    if context['site'] == 'orix' and reuse_record(context, 'pretrain') and unit == reuse_record(context, 'pretrain')['unit']:
        return 'reused'
    if (folder / 'REUSED.json').exists():
        return 'complete'
    if context['site'] == 'ibex' and reuse_record(context, 'query') and unit == reuse_record(context, 'query')['unit']:
        return 'reuse_pending'
    if not (folder / 'submission-receipt.json').exists() and not (folder / 'SUBMITTING.json').exists():
        if (folder / 'config.json').exists():
            return 'ready'
        if context['site'] == 'ibex' and not (campaign / 'initializers' / read_json(folder / 'unit.json')['parent'] / 'ADMITTED.json').exists():
            return 'waiting_parent'
        return 'buildable'
    if job_token(read_json(folder / 'submission-receipt.json', {})) is None:
        return 'attention'
    present = waves(folder)
    for _, wave in present:
        if ((wave / 'ATTENTION.json').exists() or (wave / 'AMBIGUOUS_SUCCESSOR.json').exists() or (wave / 'SIGNAL_NOT_PREEMPTION.json').exists()) \
                and not (wave / 'RESUME_AUTHORIZED.json').exists():
            return 'attention'
    last = present[-1][1] if present else None
    if last is not None and (last / 'ALLOCATION_COMPLETE.json').exists() and (folder / 'training' / 'COMPLETE.json').exists():
        return 'complete'
    if jobs is not None and recorded_jobs(folder) & {value['base'] for value in jobs.values()}:
        return 'active'
    following = present[-1][0] + 1 if present else 0
    if (folder / 'resubmissions' / ('wave-%03d.json' % following)).exists():
        return 'unknown' if jobs is None else 'suspect'
    if last is not None and (last / 'STOPPED.json').exists():
        return 'stopped'
    if last is not None and (last / 'HELD.json').exists():
        return 'held'
    if last is not None and (last / 'PREEMPTION_VERIFIED.json').exists():
        return 'preempted'
    if last is not None and (last / 'SIGNAL_PAUSE_CANDIDATE.json').exists():
        return 'preemption_check'
    if jobs is None:
        return 'unknown'
    if stop_marker(campaign, folder):
        return 'stopped'
    return 'suspect'


def classify_all(context, jobs, mutate=True):
    campaign = context['campaign']
    suspects = read_json(campaign / 'state' / 'suspects.json', {})
    states, kept, now = {}, {}, time.time()
    for unit in context['manifest']['units']:
        state = classify(context, unit, jobs)
        if state == 'suspect':
            first = suspects.get(unit, now)
            kept[unit] = first
            if now - first >= 900:
                state = 'attention'
        if state == 'attention' and mutate:
            path = campaign / 'state' / 'attention' / (unit + '.json')
            if not path.exists():
                write_exclusive(path, {'unit': unit, 'first_seen': now, 'automatic_retry': False})
                event(campaign, {'event': 'unit_attention', 'unit': unit})
        states[unit] = state
    if mutate:
        write_replace(campaign / 'state' / 'suspects.json', kept)
    return states


def unacknowledged(campaign):
    folder = campaign / 'state' / 'attention'
    return sorted(path.stem for path in folder.glob('*.json') if not path.name.endswith('.ack.json') and not (folder / (path.stem + '.ack.json')).exists()) if folder.is_dir() else []


# ---------------------------------------------------------------- config building


def build_pretrain_config(context, unit):
    campaign, folder = context['campaign'], unit_dir(context['campaign'], unit)
    record = read_json(folder / 'unit.json')
    policy = context['policy']
    protocol_path = campaign / 'protocols' / context['bundle']['protocols']['pretrain']['name']
    protocol = read_json(protocol_path)
    data = Path(context['manifest']['inputs_root']) / record['inputs']['graph']['relative']
    temporary = folder / 'config.building.json'
    if temporary.exists():
        temporary.unlink()
    command = [context['manifest']['worker_python'], str(campaign / 'worker' / 'make_config.py'), '--protocol', str(protocol_path), '--purpose', 'production',
               '--experiment-id', unit, '--data-dir', str(data), '--recipe-id', protocol['production_recipe_id'], '--seed', str(record['seed']),
               '--checkpoint-every-updates', str(policy['checkpoint_every_updates']), '--checkpoint-every-seconds', str(policy['checkpoint_every_seconds']),
               '--maximum-seconds', str(policy['worker_maximum_seconds']), '--output', str(temporary)]
    result = run(command, 600)
    write_replace(folder / 'config-build.json', result)
    check(result['returncode'] == 0, 'pretrain make_config refused %s: %s' % (unit, result['stderr'][-1500:]))
    config = read_json(temporary)
    check(config['data_complete_sha256'] == record['inputs']['graph']['complete_sha256'] and config['seed'] == record['seed'], 'config input or seed differs: %s' % unit)
    check(config['graph'] == {'taxonomy_mode': 'original', 'modality': record['modality'], 'fold': record['fold'], 'cohort': record['cohort']}, 'config graph differs: %s' % unit)
    check(config['protocol_sha256'] == context['bundle']['protocols']['pretrain']['sha256'] and config['source_hashes'] == context['manifest']['worker_sources'], 'config pins: %s' % unit)
    os.replace(str(temporary), str(folder / 'config.json'))
    event(campaign, {'event': 'config_built', 'unit': unit, 'sha256': sha(folder / 'config.json')})


def build_query_config(context, unit):
    campaign, folder = context['campaign'], unit_dir(context['campaign'], unit)
    record = read_json(folder / 'unit.json')
    admitted = read_json(campaign / 'initializers' / record['parent'] / 'ADMITTED.json')
    recipe = context['recipe']
    parent = context['bundle']['units']['pretrain'][record['parent']]
    request = {'output': str(folder / 'config.building.json'), 'protocol': str(campaign / 'protocols' / context['bundle']['protocols']['query']['name']), 'experiment_id': unit,
               'data_dir': str(Path(context['manifest']['inputs_root']) / record['inputs']['bank']['relative']),
               'pretrained': admitted.get('export_path') or str(campaign / 'initializers' / record['parent'] / admitted['export']), 'pretrained_sha256': admitted['export_sha256'],
               'initializer': {'taxonomy_mode': 'original', 'cohort': record['cohort'], 'modality': record['modality'], 'fold': record['fold'], 'pretrain_seed': parent['seed'],
                               **recipe['initializer_recipe']},
               'seed': record['seed'], 'sampler_seed': record['sampler_seed'], 'encoder': record['encoder'], 'curriculum': record['curriculum'],
               'learning_rate': recipe['learning_rate'], 'temperature': recipe['temperature'], 'maximum_steps': context['bundle']['site_policy']['query_steps'],
               'evaluation_every_updates': context['bundle']['site_policy']['evaluation_every_updates'], 'maximum_seconds': context['policy']['worker_maximum_seconds']}
    for leftover in (folder / 'config.building.json', folder / 'config-request.json', folder / 'config-receipt.json'):
        if leftover.exists():
            leftover.unlink()
    write_replace(folder / 'config-request.json', request)
    result = run([context['manifest']['worker_python'], '-I', str(HERE / 'build_query_config.py'), '--worker-code', str(campaign / 'query' / 'code'),
                  '--request', str(folder / 'config-request.json'), '--receipt', str(folder / 'config-receipt.json')], 1200)
    write_replace(folder / 'config-build.json', result)
    check(result['returncode'] == 0, 'query make_config refused %s: %s' % (unit, result['stderr'][-1500:]))
    config = read_json(folder / 'config.building.json')
    check(config['purpose'] == 'production' and config['pretrained_sha256'] == admitted['export_sha256'] and config['protocol_sha256'] == context['bundle']['protocols']['query']['sha256'],
          'config pins differ: %s' % unit)
    check({name: config['source_hashes'][name] for name in context['manifest']['worker_sources']} == context['manifest']['worker_sources'], 'config sources differ: %s' % unit)
    os.replace(str(folder / 'config.building.json'), str(folder / 'config.json'))
    event(campaign, {'event': 'config_built', 'unit': unit, 'sha256': sha(folder / 'config.json')})


# ---------------------------------------------------------------- storage (IBEX)


def query_peak(policy):
    storage = policy['storage']
    sizes = storage['estimates_until_measured']
    evaluations = 20000 // 500 + 1
    return int((storage['full_states_at_peak'] * sizes['full_state_bytes'] + storage['model_only_at_peak'] * sizes['model_only_bytes']
                + evaluations * sizes['scores_per_evaluation_bytes']) * storage['safety_factor'])


def storage_budget(context, states):
    campaign, policy = context['campaign'], context['policy']
    release = context['release'] or {}
    free, detail = free_bytes(campaign, release.get('quota_command'))
    if context['site'] == 'orix':
        return {'free': free, 'detail': detail, 'reserve': policy['storage_reserve_bytes'], 'owed': 0, 'peak': 0}
    peak = query_peak(policy)
    owed = sum(max(0, peak - tree_bytes(unit_dir(campaign, unit) / 'training')) for unit, state in states.items() if state in ACTIVE + ('held', 'preempted', 'preemption_check'))
    return {'free': free, 'detail': detail, 'reserve': policy['storage']['reserve_bytes'], 'owed': owed, 'peak': peak}


# ---------------------------------------------------------------- verified preemption


def sacct_state(job):
    result = run(['sacct', '-j', job, '-X', '-n', '-P', '-o', 'State'], 60)
    if result['returncode'] != 0 or not result['stdout'].strip():
        return None
    return result['stdout'].strip().splitlines()[0].split()[0].split('|')[0]


def resolve_preemptions(context, states, dry_run):
    """A clean signal pause becomes PREEMPTION_VERIFIED only when sacct reports PREEMPTED for that allocation, no stop
    marker exists and the per-unit resume bound holds; any other terminal state becomes SIGNAL_NOT_PREEMPTION (attention)."""
    campaign, rules = context['campaign'], context['policy']['preemption']
    decided = []
    for unit, state in sorted(states.items()):
        if state != 'preemption_check':
            continue
        folder = unit_dir(campaign, unit)
        wave = waves(folder)[-1][1]
        candidate = read_json(wave / 'SIGNAL_PAUSE_CANDIDATE.json')
        if time.time() - candidate['time'] < rules['sacct_settle_seconds']:
            continue
        terminal = sacct_state(candidate['job'])
        if terminal is None:
            if time.time() - candidate['time'] > 3600 and not dry_run:
                write_exclusive(wave / 'SIGNAL_NOT_PREEMPTION.json', {'job': candidate['job'], 'sacct': None, 'reason': 'sacct gave no terminal state within an hour'})
                decided.append({'unit': unit, 'outcome': 'attention', 'sacct': None})
            continue
        if terminal in ('PENDING', 'RUNNING', 'COMPLETING', 'REQUEUED', 'SUSPENDED'):
            continue
        used = len(list(folder.glob('allocations/wave-*/PREEMPTION_VERIFIED.json')))
        verified = terminal in rules['verified_states'] and not stop_marker(campaign, folder) and used < rules['maximum_resumes_per_unit']
        reason = 'verified scheduler preemption' if verified else ('stop marker present' if stop_marker(campaign, folder) else
                                                                    'resume bound reached' if terminal in rules['verified_states'] else 'terminal state %s is not a preemption' % terminal)
        if not dry_run:
            write_exclusive(wave / ('PREEMPTION_VERIFIED.json' if verified else 'SIGNAL_NOT_PREEMPTION.json'),
                            {'job': candidate['job'], 'sacct': terminal, 'reason': reason, 'resumes_before': used, 'time': time.time()})
            event(campaign, {'event': 'preemption_verified' if verified else 'signal_not_preemption', 'unit': unit, 'sacct': terminal})
            states[unit] = 'preempted' if verified else 'attention'
        decided.append({'unit': unit, 'outcome': 'resume' if verified else 'attention', 'sacct': terminal})
    return decided


# ---------------------------------------------------------------- FROZEN_HPO_ARTIFACT reuse (IBEX)


def admit_reuse(context, dry_run):
    """Admit the audited HPO initializer as the PFS fold-0 seed-0 parent and, if audited, the HPO trajectory as the reused
    query unit. Every pinned hash is re-read at the IBEX path; old COMPLETE and identity files are only read."""
    campaign, done = context['campaign'], []
    pretrain = reuse_record(context, 'pretrain')
    if pretrain:
        final = campaign / 'initializers' / pretrain['unit']
        if not (final / 'ADMITTED.json').exists():
            path = Path(pretrain['export']['ibex_path'])
            check(path.is_file() and path.stat().st_size == pretrain['export']['bytes'] and sha(path) == pretrain['export']['sha256'],
                  'FROZEN_HPO_ARTIFACT initializer missing or changed at %s' % path)
            admission = Path(pretrain['provenance']['delivery_admission_path'])
            check(sha(admission) == pretrain['provenance']['delivery_admission_sha256'], 'original delivery ADMISSION changed')
            if not dry_run:
                write_exclusive(final / 'ADMITTED.json', {'parent': pretrain['unit'], 'kind': 'FROZEN_HPO_ARTIFACT', 'export': path.name, 'export_path': str(path),
                                                          'export_sha256': pretrain['export']['sha256'], 'provenance': pretrain['provenance'], 'audit_checks': pretrain['checks'],
                                                          'note': 'trained artifact reuse; not trained under the v2 protocol; legacy source and protocol hashes kept', 'time': time.time()})
                event(campaign, {'event': 'frozen_hpo_artifact_admitted', 'unit': pretrain['unit']})
            done.append(pretrain['unit'])
    query = reuse_record(context, 'query')
    if query and pretrain:
        folder = unit_dir(campaign, query['unit'])
        if not (folder / 'REUSED.json').exists():
            training = Path(query['training_dir'])
            check(sha(training / 'COMPLETE.json') == query['complete_sha256'] and sha(training / 'identity.json') == query['identity_sha256'],
                  'reused HPO trajectory changed at %s' % training)
            if not dry_run:
                write_exclusive(folder / 'REUSED.json', {'unit': query['unit'], 'kind': 'FROZEN_HPO_ARTIFACT', 'training_dir': str(training),
                                                         'complete_sha256': query['complete_sha256'], 'identity_sha256': query['identity_sha256'], 'audit_checks': query['checks'],
                                                         'note': 'completed HPO trajectory reused; never rewritten or pruned by production', 'time': time.time()})
                event(campaign, {'event': 'frozen_hpo_trajectory_reused', 'unit': query['unit']})
            done.append(query['unit'])
    return done


# ---------------------------------------------------------------- released per-unit TEST inference (IBEX)


TEST_SOURCES = ('train.py', 'model.py', 'original_model.py', 'packed_encode.py', 'sampler.py', 'evaluate.py', 'metrics.py', 'state_compare.py', 'test_eval.py')


def test_gate(context, unit, state):
    """(passed, reasons). TEST is released for one unit only after the HPO common recipe is globally frozen and this unit's
    training is complete with its validation selections frozen. Nothing here depends on any TEST result."""
    bundle, reasons = context['bundle'], []
    check_record = bundle.get('hpo_check') or {}
    if bundle['rehearsal']:
        if not (check_record.get('rehearsal') is True and (context['release'] or {}).get('rehearsal') is True):
            reasons.append('rehearsal bundle without a rehearsal HPO check and release')
    elif not (check_record.get('passed') is True and not check_record.get('synthetic') and check_record.get('sha256') == bundle.get('hpo_check_sha256')):
        reasons.append('HPO common recipe not globally frozen by a passed, non-synthetic check')
    if bundle['production_recipe'].get('common_leaderboard') != 'calibrated':
        reasons.append('production recipe is not the recorded calibrated common recipe')
    if state != 'complete':
        reasons.append('training not complete')
        return False, reasons
    directory = training_dir(context, unit)
    complete = read_json(directory / 'COMPLETE.json')
    for name in ('selection.json', 'curve.json'):
        if complete['files'].get(name) != sha(directory / name):
            reasons.append('%s not frozen in COMPLETE' % name)
    if not reasons and complete.get('selection') != read_json(directory / 'selection.json'):
        reasons.append('COMPLETE selection differs from selection.json')
    terminal = complete.get('terminal_checkpoint', {})
    for view in ('raw', 'calibrated'):
        pointer = (terminal.get('retained_best') or {}).get(view)
        if not pointer or pointer['step'] != complete['selection'][view]['step']:
            reasons.append('%s selected checkpoint pointer missing or not the frozen selection' % view)
    return not reasons, reasons


def prepare_inference(context, unit):
    """Freeze the selection, then write the per-unit TEST-RELEASE.json, the test_eval config and the plan (write-once)."""
    campaign, bundle = context['campaign'], context['bundle']
    folder = unit_dir(campaign, unit)
    inference = folder / 'inference'
    record = read_json(folder / 'unit.json')
    directory = training_dir(context, unit)
    complete_sha = sha(directory / 'COMPLETE.json')
    complete = read_json(directory / 'COMPLETE.json')
    write_exclusive(folder / 'SELECTION-FROZEN.json', {'unit': unit, 'training_complete_sha256': complete_sha, 'selection': complete['selection'],
                                                       'selection_sha256': sha(directory / 'selection.json'), 'curve_sha256': sha(directory / 'curve.json'),
                                                       'retained_best': complete['terminal_checkpoint']['retained_best'], 'initial_checkpoint': complete['terminal_checkpoint']['initial_checkpoint']})
    test_dir = Path(context['manifest']['inputs_root']) / record['inputs']['test']['relative']
    bank_dir = Path(context['manifest']['inputs_root']) / record['inputs']['bank']['relative']
    test_sha = record['inputs']['test']['complete_sha256']
    check(sha(test_dir / 'COMPLETE.json') == test_sha, 'TEST COMPLETE changed for %s' % unit)
    write_exclusive(inference / 'TEST-RELEASE.json', {'status': 'RELEASED', 'unit': unit, 'training_complete_sha256': [complete_sha], 'test_complete_sha256': [test_sha],
                                                      'gates': {'hpo_check_sha256': bundle.get('hpo_check_sha256'), 'bundle_sha256': context['manifest']['bundle_sha256'],
                                                                'selection_frozen_sha256': sha(folder / 'SELECTION-FROZEN.json'),
                                                                'common_leaderboard': bundle['production_recipe']['common_leaderboard']},
                                                      'rule': 'released per unit after the global recipe freeze and this unit\'s frozen validation selections; no decision uses TEST'})
    views = list(context['policy']['inference']['views'])
    terminal = directory / complete['terminal_checkpoint']['checkpoint']
    check(terminal.exists(), 'terminal full state already missing for %s: inference must precede pruning' % unit)
    config = {'phase': 'test', 'views': views, 'training_dir': str(directory), 'training_complete_sha256': complete_sha, 'bank_dir': str(bank_dir), 'test_dir': str(test_dir),
              'test_complete_sha256': test_sha, 'release_path': str(inference / 'TEST-RELEASE.json'), 'release_sha256': sha(inference / 'TEST-RELEASE.json'),
              'source_hashes': {name: sha(campaign / 'query' / 'code' / name) for name in TEST_SOURCES}}
    write_exclusive(inference / 'test-config.json', config)
    write_exclusive(inference / 'INFERENCE-PLAN.json', {'unit': unit, 'training_dir': str(directory), 'training_complete_sha256': complete_sha,
                                                        'files': {'TEST-RELEASE.json': sha(inference / 'TEST-RELEASE.json'), 'test-config.json': sha(inference / 'test-config.json')}})
    event(campaign, {'event': 'test_released', 'unit': unit})


def inference_verified(folder):
    complete = folder / 'inference' / 'INFERENCE_COMPLETE.json'
    if not complete.exists():
        return False
    return all(sha(folder / 'inference' / name) == digest for name, digest in read_json(complete)['files'].items())


def inference_state(context, unit, state, jobs):
    folder = unit_dir(context['campaign'], unit)
    inference = folder / 'inference'
    if (inference / 'INFERENCE_COMPLETE.json').exists():
        return 'complete'
    if any(inference.glob('attempt-*/ATTENTION.json')):
        return 'attention'
    receipt = inference / 'submission-receipt.json'
    if receipt.exists():
        token = job_token(read_json(receipt))
        if token is None:
            return 'attention'
        if jobs is None or token in {value['base'] for value in jobs.values()}:
            return 'active'
        return 'attention' if time.time() - read_json(receipt)['time'] > 900 else 'active'
    if (inference / 'INFERENCE-PLAN.json').exists():
        return 'ready'
    return 'gated' if state == 'complete' else 'not_ready'


def submit_inference(context, unit):
    campaign, release = context['campaign'], context['release']
    inference = unit_dir(campaign, unit) / 'inference'
    command = inference_command(campaign, release, unit, 0)
    write_exclusive(inference / 'SUBMITTING.json', {'time': time.time(), 'command': command})
    result = run(command, 60)
    write_exclusive(inference / 'submission-receipt.json', dict(result, time=time.time()))
    token = job_token(result)
    event(campaign, {'event': 'inference_submitted', 'unit': unit, 'job': token})
    if not token:
        halt(campaign, 'inference_submission_ambiguous', {'unit': unit, 'result': result})
    return token


# ---------------------------------------------------------------- submission


def submit(context, unit, wave, reason):
    campaign, release = context['campaign'], context['release']
    folder = unit_dir(campaign, unit)
    check(not stop_marker(campaign, folder), 'stop marker present')
    if wave == 0:
        marker, receipt = folder / 'SUBMITTING.json', folder / 'submission-receipt.json'
    else:
        marker, receipt = folder / 'resubmissions' / ('wave-%03d.SUBMITTING.json' % wave), folder / 'resubmissions' / ('wave-%03d.json' % wave)
    check(not marker.exists() and not receipt.exists(), 'already submitted: %s wave %d' % (unit, wave))
    command = sbatch_command(campaign, release, unit, wave)
    write_exclusive(marker, {'time': time.time(), 'reason': reason, 'command': command})
    result = run(command, 60)
    write_exclusive(receipt, dict(result, time=time.time()))
    token = job_token(result)
    event(campaign, {'event': 'submitted', 'unit': unit, 'wave': wave, 'job': token, 'reason': reason})
    if not token:
        halt(campaign, 'submission_ambiguous', {'unit': unit, 'wave': wave, 'result': result})
    return token


def priority(context, unit):
    record = read_json(unit_dir(context['campaign'], unit) / 'unit.json')
    if context['site'] == 'orix':
        children = sum(1 for child in context['bundle']['units']['query'].values() if child['parent'] == unit)
        return (-children, record['stage'], unit)
    return (record['stage'], unit)


def tick(args):
    context = load(args.campaign)
    campaign = context['campaign']
    with exclusive_lock(campaign / 'state' / 'tick.lock') as locked:
        if not locked:
            print(json.dumps({'status': 'busy'}))
            return 0
        if (campaign / 'state' / 'HALT.json').exists():
            summary = {'status': 'halted', 'halt': read_json(campaign / 'state' / 'HALT.json')}
            print(json.dumps(summary, indent=2, default=str))
            return 1
        if (campaign / 'OPERATOR_STOP.json').exists():
            print(json.dumps({'status': 'operator_stop'}))
            return 0
        summary = {'status': 'ok', 'site': context['site'], 'time': time.time(), 'dry_run': bool(args.dry_run)}
        try:
            jobs = user_jobs()
            summary['scheduler_available'] = jobs is not None
            if context['site'] == 'ibex':
                summary['reuse_admitted'] = admit_reuse(context, args.dry_run)
            states = classify_all(context, jobs, mutate=not args.dry_run)
            if jobs is not None:
                summary['preemption_decisions'] = resolve_preemptions(context, states, args.dry_run)
                if not args.dry_run:
                    states = classify_all(context, jobs, mutate=True)
            tests = {}
            if context['site'] == 'ibex':
                summary['test_blocked'] = {}
                for unit, state in states.items():
                    tests[unit] = inference_state(context, unit, state, jobs)
                    if tests[unit] == 'gated':
                        passed, reasons = test_gate(context, unit, state)
                        if passed and not args.dry_run:
                            prepare_inference(context, unit)
                            tests[unit] = 'ready'
                        elif not passed:
                            summary['test_blocked'][unit] = reasons
                    if tests[unit] == 'attention' and not args.dry_run:
                        path = campaign / 'state' / 'attention' / (unit + '.inference.json')
                        if not path.exists():
                            write_exclusive(path, {'unit': unit, 'phase': 'inference', 'first_seen': time.time(), 'automatic_retry': False})
                summary['test_states'] = count(tests)
            attention = unacknowledged(campaign)
            summary['unacknowledged_attention'] = attention
            if attention and not args.dry_run:
                halt(campaign, 'unit_attention', attention)
            if context['site'] == 'ibex' and not args.dry_run:
                summary['pruned'] = prune_archived(context, states, tests)
            summary['budget'] = budget = storage_budget(context, states)
            summary['storage_decision'] = storage_decision(campaign)
            if jobs is None:
                summary['skipped'] = {'scheduler_unavailable': True}
            else:
                summary['submitted'], summary['skipped'] = schedule(context, states, tests, jobs, budget, args.dry_run)
            summary['states'] = count(states)
            summary['done'] = all(state in ('complete', 'reused') for state in states.values()) and all(value == 'complete' for value in tests.values())
        except Halt as reason:
            summary.update({'status': 'halted', 'reason': str(reason)})
        except Refusal as error:
            if args.dry_run:
                summary.update({'status': 'refused', 'detail': str(error)})
            else:
                try:
                    halt(campaign, 'refusal', str(error))
                except Halt:
                    summary.update({'status': 'halted', 'reason': 'refusal', 'detail': str(error)})
        if not args.dry_run:
            write_replace(campaign / 'state' / 'last-tick.json', summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0 if summary['status'] == 'ok' else 1


def schedule(context, states, tests, jobs, budget, dry_run):
    campaign, release, site = context['campaign'], context['release'], context['site']
    if site == 'orix':
        partition = format_value(release['slurm_arguments'], '--partition')
        used = sum(1 for value in jobs.values() if value['partition'] == partition)
    else:
        used = len(jobs)
    cap = release['max_active']
    submitted, skipped = [], {}
    for unit in sorted(unit for unit, value in tests.items() if value == 'ready'):
        if used + len(submitted) >= cap:
            skipped.setdefault('active_cap_%d_inference' % cap, []).append(unit)
            continue
        submitted.append({'unit': unit, 'inference': True, 'job': None if dry_run else submit_inference(context, unit)})
    resumable = sorted((unit for unit, state in states.items() if state in ('held', 'preempted')), key=lambda unit: priority(context, unit))
    fresh = sorted((unit for unit, state in states.items() if state in ('ready', 'buildable')), key=lambda unit: priority(context, unit))
    available = None if budget['free'] is None else budget['free'] - budget['reserve'] - budget['owed']
    for unit in resumable + fresh:
        folder = unit_dir(campaign, unit)
        if stop_marker(campaign, folder):
            skipped.setdefault('stop_marker', []).append(unit)
            continue
        if used + len(submitted) >= cap:
            skipped.setdefault('active_cap_%d' % cap, []).append(unit)
            continue
        if states[unit] in ('held', 'preempted'):
            reason = 'held clean pause' if states[unit] == 'held' else 'verified scheduler preemption (fresh submission, no dependency on the preempted job)'
            if not dry_run:
                wave = waves(folder)[-1][0] + 1
                check(wave < context['policy']['maximum_allocations'], 'allocation bound exhausted: %s' % unit)
                submitted.append({'unit': unit, 'job': submit(context, unit, wave, reason), 'wave': wave, 'reason': reason})
            else:
                submitted.append({'unit': unit, 'would_resubmit': True, 'reason': reason})
            continue
        if available is None or available < max(budget['peak'], 0):
            skipped.setdefault('storage', []).append(unit)
            continue
        if dry_run:
            submitted.append({'unit': unit, 'would_build_config': states[unit] == 'buildable', 'would_submit': True})
            available -= budget['peak']
            continue
        if states[unit] == 'buildable':
            (build_pretrain_config if site == 'orix' else build_query_config)(context, unit)
        token = submit(context, unit, 0, 'eligible')
        available -= budget['peak']
        submitted.append({'unit': unit, 'job': token})
    if skipped.get('storage') and not any(state in ACTIVE for state in states.values()):
        skipped['storage_stall'] = True
        decision = storage_decision(campaign)
        if site == 'ibex' and not decision['release_new_production_reportables_after_verified_archive']:
            skipped['storage_permission_required'] = {
                'reason': 'no unit can start: completed units keep their reportables on IBEX until an explicit storage decision allows releasing archived copies',
                'decision_recorded': decision['recorded'], 'free_bytes': budget['free'], 'reserve_bytes': budget['reserve'], 'unit_peak_bytes': budget['peak'],
                'action': 'record a clmpt-suite-storage-decision/1 file with campaign.py record-storage-decision, or free quota; nothing is deleted without it'}
    return submitted, skipped


def count(states):
    value = {}
    for state in states.values():
        value[state] = value.get(state, 0) + 1
    return value


# ---------------------------------------------------------------- delivery admission (IBEX)


def admit(args):
    context = load(args.campaign)
    campaign, bundle = context['campaign'], context['bundle']
    check(context['site'] == 'ibex', 'admit runs on the IBEX campaign')
    parent = args.parent
    check(parent in bundle['units']['pretrain'], 'unknown pretraining unit %s' % parent)
    final = campaign / 'initializers' / parent
    if (final / 'ADMITTED.json').exists():
        print(json.dumps({'status': 'already_admitted', 'parent': parent}))
        return 0
    partial = campaign / 'incoming' / ('.partial-' + parent)
    delivery = read_json(partial / 'DELIVERY.json')
    problems = []
    for name, digest in delivery['files'].items():
        if not (partial / name).is_file() or sha(partial / name) != digest:
            problems.append('delivered file missing or differs: %s' % name)
    unit = bundle['units']['pretrain'][parent]
    recipe_id = read_json(campaign / 'protocols' / bundle['protocols']['pretrain']['name'])['production_recipe_id']
    total = bundle['production_recipe']['initializer_recipe']['initializer_update_budget']
    complete = read_json(partial / 'COMPLETE.json') if (partial / 'COMPLETE.json').exists() else {}
    identity = read_json(partial / 'identity.json') if (partial / 'identity.json').exists() else {}
    export = 'models/update-%07d.pt' % total
    sources = {name: bundle['source']['files']['pretrain/' + name] for name in PRETRAIN_SOURCES}
    checks = {
        'complete_files': set(DELIVERED) <= set(delivery['files']) and export.split('/')[-1] in delivery['files'],
        'export_hash_in_complete': complete.get('files', {}).get(export) == delivery['files'].get(export.split('/')[-1]),
        'complete_steps': complete.get('steps') == total and complete.get('recipe_id') == recipe_id and complete.get('seed') == unit['seed'],
        'complete_graph': complete.get('graph') == {'taxonomy_mode': 'original', 'modality': unit['modality'], 'fold': unit['fold'], 'cohort': unit['cohort']},
        'identity_bound': bool(identity) and canonical(identity) == complete.get('identity_sha256'),
        'identity_protocol': identity.get('protocol_sha256') == bundle['protocols']['pretrain']['sha256'],
        'identity_sources': identity.get('sources') == sources,
        'identity_graph': identity.get('data_complete_sha256') == bundle['inputs'][unit['entry']]['graph']['complete_sha256'],
        'config_bound': (partial / 'config.json').exists() and sha(partial / 'config.json') == identity.get('config_sha256'),
        'progress_in_complete': (partial / 'progress.json').exists() and sha(partial / 'progress.json') == complete.get('files', {}).get('progress.json'),
    }
    problems += [name for name, value in checks.items() if not value]
    if problems:
        write_replace(campaign / 'state' / 'admission-refusals' / (parent + '.json'), {'time': time.time(), 'problems': problems, 'checks': checks})
        raise Refusal('delivery of %s refused: %s' % (parent, problems))
    os.replace(str(partial), str(final))
    write_exclusive(final / 'ADMITTED.json', {'parent': parent, 'export': export.split('/')[-1], 'export_sha256': delivery['files'][export.split('/')[-1]],
                                              'complete_sha256': sha(final / 'COMPLETE.json'), 'identity_sha256': complete['identity_sha256'], 'checks': checks,
                                              'delivery_sha256': sha(final / 'DELIVERY.json'), 'source': delivery.get('source'), 'time': time.time()})
    event(campaign, {'event': 'initializer_admitted', 'parent': parent})
    print(json.dumps({'status': 'admitted', 'parent': parent}))
    return 0


# ---------------------------------------------------------------- archive and pruning (IBEX), archive verification (Orix)


def reportable(policy_all, complete):
    prefixes = tuple(name for name in policy_all['reportable_artifacts'] if name.endswith('/'))
    names = set(name for name in policy_all['reportable_artifacts'] if not name.endswith('/'))
    return {name: digest for name, digest in complete['files'].items() if name in names or name.startswith(prefixes)}


def archive_item(context, unit):
    """Reportable training files plus the verified inference outputs of one unit, keyed 'training/...' and 'inference/...'."""
    folder = unit_dir(context['campaign'], unit)
    directory = training_dir(context, unit)
    complete = read_json(directory / 'COMPLETE.json')
    files = {'training/' + name: digest for name, digest in reportable(context['bundle']['site_policy'], complete).items()}
    files['training/COMPLETE.json'] = sha(directory / 'COMPLETE.json')
    for name, digest in read_json(folder / 'inference' / 'INFERENCE_COMPLETE.json')['files'].items():
        files['inference/' + name] = digest
    files['inference/INFERENCE_COMPLETE.json'] = sha(folder / 'inference' / 'INFERENCE_COMPLETE.json')
    for name in ('SELECTION-FROZEN.json', 'unit.json'):
        files['unit/' + name] = sha(folder / name)
    return {'sources': {'training': str(directory), 'inference': str(folder / 'inference'), 'unit': str(folder)}, 'files': files,
            'owned_training': not (folder / 'REUSED.json').exists()}


def ack_archive(args):
    context = load(args.campaign)
    campaign = context['campaign']
    receipt = json.loads(sys.stdin.read())
    check(inference_verified(unit_dir(campaign, args.unit)), 'inference not complete and verified: nothing is archived before both training and inference')
    expected = archive_item(context, args.unit)['files']
    check(receipt.get('unit') == args.unit and receipt.get('verified_on_archive') is True and receipt.get('files') == expected, 'archive receipt does not match the unit')
    write_exclusive(campaign / 'state' / 'archive' / (args.unit + '.json'), dict(receipt, acknowledged=time.time()))
    event(campaign, {'event': 'archive_acknowledged', 'unit': args.unit})
    print(json.dumps({'status': 'acknowledged', 'unit': args.unit}))
    return 0


def prune_archived(context, states, tests):
    """Delete full optimizer states of archived, inference-complete units that production owns; copies of archived
    reportables only when the recorded storage decision allows it. Reused HPO directories are never touched."""
    campaign = context['campaign']
    decision = storage_decision(campaign)
    done = []
    for unit, state in sorted(states.items()):
        ack = campaign / 'state' / 'archive' / (unit + '.json')
        receipt = campaign / 'state' / 'pruned' / (unit + '.json')
        if state != 'complete' or tests.get(unit) != 'complete' or not ack.exists() or receipt.exists():
            continue
        if (unit_dir(campaign, unit) / 'REUSED.json').exists():
            continue
        check(inference_verified(unit_dir(campaign, unit)), 'inference outputs changed before pruning: %s' % unit)
        training = unit_dir(campaign, unit) / 'training'
        complete = read_json(training / 'COMPLETE.json')
        archive = read_json(ack)
        release_copies = decision['release_new_production_reportables_after_verified_archive']
        removable = {name: digest for name, digest in complete['files'].items() if name.startswith('checkpoints/')}
        if release_copies:
            removable.update({name[len('training/'):]: digest for name, digest in archive['files'].items() if name.startswith(('training/fits/', 'training/models/'))})
        for name, digest in removable.items():
            check((training / name).exists() is False or sha(training / name) == digest, 'file changed before pruning: %s/%s' % (unit, name))
        write_exclusive(receipt, {'unit': unit, 'removed': removable, 'archive_receipt_sha256': sha(ack), 'reportables_released': release_copies,
                                  'storage_decision_sha256': decision['sha256'], 'time': time.time()})
        checkpoints = {name: digest for name, digest in removable.items() if name.startswith('checkpoints/')}
        if checkpoints and not (training / 'PRUNED.json').exists():
            write_exclusive(training / 'PRUNED.json', {'identity_sha256': complete['identity_sha256'], 'receipt_sha256': sha(receipt), 'files': checkpoints})
        for name in removable:
            if (training / name).exists():
                (training / name).unlink()
        event(campaign, {'event': 'pruned', 'unit': unit, 'files': len(removable)})
        done.append(unit)
    return done


HPO_REPORTABLE = ('COMPLETE.json', 'identity.json', 'progress.json', 'curve.json', 'selection.json', 'updates.jsonl', 'commits.jsonl', 'sampler-description.json', 'PRUNED.json')


def hpo_campaign_path(context, argument):
    """Only the HPO campaign this production bundle was frozen from; any older campaign is refused."""
    expected = (context['bundle'].get('hpo_check') or {}).get('hpo_campaign')
    path = Path(argument)
    check(expected and str(path) == expected and os.path.realpath(str(path)) == str(path), 'not the HPO campaign of this bundle: %s' % argument)
    return path


def protected_hpo_paths(context):
    """Directories production reuses (FROZEN_HPO_ARTIFACT trajectory and every pointer root it names, the initializer)."""
    protected = set()
    query = reuse_record(context, 'query')
    if query:
        training = Path(query['training_dir'])
        protected.add(str(training))
        complete = read_json(training / 'COMPLETE.json')
        terminal = complete.get('terminal_checkpoint', {})
        for pointer in [terminal.get('initial_checkpoint')] + list((terminal.get('retained_best') or {}).values()):
            if pointer:
                protected.add(str(Path(pointer['root'])))
    pretrain = reuse_record(context, 'pretrain')
    if pretrain:
        protected.add(str(Path(pretrain['export']['ibex_path']).parent))
    return protected


def hpo_archive_list(args):
    context = load(args.campaign)
    check(storage_decision(context['campaign'])['archive_current_hpo_reportables'], 'no recorded storage decision allows archiving the HPO campaign')
    hpo = hpo_campaign_path(context, args.hpo_campaign)
    acked = {path.stem for path in (context['campaign'] / 'state' / 'hpo-archive').glob('*.json')} if (context['campaign'] / 'state' / 'hpo-archive').is_dir() else set()
    items = {}
    records = {}
    for path in sorted(list((hpo / 'decisions').rglob('*')) + list((hpo / 'config').rglob('*')) + [hpo / 'frozen-manifest.json', hpo / 'ROOT_RELEASE.json']):
        if path.is_file():
            records['hpo/' + str(path.relative_to(hpo))] = sha(path)
    if 'campaign-records' not in acked:
        items['campaign-records'] = {'sources': {'hpo': str(hpo)}, 'files': records}
    for training in sorted(hpo.glob('trials/*/*/training')):
        key = '%s-%s' % (training.parent.parent.name, training.parent.name)
        if key in acked or not (training / 'COMPLETE.json').exists():
            continue
        complete = read_json(training / 'COMPLETE.json')
        files = {'hpo/' + name: digest for name, digest in complete['files'].items() if name in HPO_REPORTABLE or name.startswith(('fits/', 'models/'))}
        files['hpo/COMPLETE.json'] = sha(training / 'COMPLETE.json')
        if (training / 'PRUNED.json').exists():
            files['hpo/PRUNED.json'] = sha(training / 'PRUNED.json')
        items[key] = {'sources': {'hpo': str(training)}, 'files': files}
    print(json.dumps({'hpo_campaign': str(hpo), 'items': items, 'acknowledged': sorted(acked)}, indent=2, sort_keys=True))
    return 0


def ack_hpo_archive(args):
    context = load(args.campaign)
    receipt = json.loads(sys.stdin.read())
    check(receipt.get('verified_on_archive') is True and receipt.get('unit') == args.unit and receipt.get('files'), 'HPO archive receipt incomplete')
    write_exclusive(context['campaign'] / 'state' / 'hpo-archive' / (args.unit + '.json'), dict(receipt, acknowledged=time.time()))
    print(json.dumps({'status': 'acknowledged', 'item': args.unit}))
    return 0


def release_hpo_copies(args):
    """Delete IBEX copies of archived HPO score archives and model-only exports, only with an explicit storage decision,
    only for items whose Orix archive was verified, never for directories production reuses, never outside the bundle's
    HPO campaign. --dry-run lists what would be removed."""
    context = load(args.campaign)
    campaign = context['campaign']
    decision = storage_decision(campaign)
    check(decision['release_current_hpo_reportables_after_verified_archive'], 'no recorded storage decision allows releasing HPO copies; nothing removed')
    hpo = hpo_campaign_path(context, args.hpo_campaign)
    protected = protected_hpo_paths(context)
    plan, kept = {}, []
    for ack in sorted((campaign / 'state' / 'hpo-archive').glob('*.json')) if (campaign / 'state' / 'hpo-archive').is_dir() else []:
        record = read_json(ack)
        source = Path(record.get('sources', {}).get('hpo', ''))
        if ack.stem == 'campaign-records' or hpo not in source.parents:
            continue
        if str(source) in protected:
            kept.append(str(source))
            continue
        removable = {name[len('hpo/'):]: digest for name, digest in record['files'].items() if name.startswith(('hpo/fits/', 'hpo/models/'))}
        plan[ack.stem] = {'directory': str(source), 'files': removable}
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'would_remove': {key: len(value['files']) for key, value in plan.items()}, 'protected_kept': kept}, indent=2))
        return 0
    for key, item in plan.items():
        receipt = campaign / 'state' / 'hpo-released' / (key + '.json')
        if receipt.exists():
            continue
        directory = Path(item['directory'])
        for name, digest in item['files'].items():
            check(not (directory / name).exists() or sha(directory / name) == digest, 'HPO file changed before release: %s/%s' % (directory, name))
        write_exclusive(receipt, {'item': key, 'directory': str(directory), 'removed': item['files'], 'storage_decision_sha256': decision['sha256'],
                                  'archive_receipt_sha256': sha(campaign / 'state' / 'hpo-archive' / (key + '.json')), 'authority': args.authority, 'time': time.time()})
        for name in item['files']:
            if (directory / name).exists():
                (directory / name).unlink()
    print(json.dumps({'released': sorted(plan), 'protected_kept': kept}, indent=2))
    return 0


def verify_archive(args):
    """Orix: called by the bridge after an upload to <root>/.partial-<unit>; expected hashes on stdin."""
    root = Path(args.root)
    expected = json.loads(sys.stdin.read())
    partial, final = root / ('.partial-' + args.unit), root / args.unit
    if (final / 'ARCHIVED.json').exists():
        record = read_json(final / 'ARCHIVED.json')
        check(record['files'] == expected['files'], 'an archive with other content exists for %s' % args.unit)
        print(json.dumps({'unit': args.unit, 'verified_on_archive': True, 'files': record['files'], 'already': True}))
        return 0
    problems = [name for name, digest in expected['files'].items() if not (partial / name).is_file() or sha(partial / name) != digest]
    check(not problems, 'archive upload incomplete or corrupted for %s: %s' % (args.unit, problems[:10]))
    os.replace(str(partial), str(final))
    write_exclusive(final / 'ARCHIVED.json', {'unit': args.unit, 'files': expected['files'], 'source': expected.get('source'), 'time': time.time()})
    print(json.dumps({'unit': args.unit, 'verified_on_archive': True, 'files': expected['files']}))
    return 0


# ---------------------------------------------------------------- inventory, stop, status


def inventory(args):
    context = load(args.campaign, require_release=False)
    campaign, site = context['campaign'], context['site']
    jobs = user_jobs()
    states = classify_all(context, jobs, mutate=False)
    value = {'site': site, 'time': time.time(), 'states': states, 'counts': count(states), 'halt': read_json(campaign / 'state' / 'HALT.json', {}) or None,
             'operator_stop': (campaign / 'OPERATOR_STOP.json').exists(), 'attention': unacknowledged(campaign), 'released': context['release'] is not None,
             'last_tick': read_json(campaign / 'state' / 'last-tick.json', {}) or None}
    if site == 'orix':
        exports = {}
        value['reused'] = [unit for unit, state in states.items() if state == 'reused']
        for unit, state in states.items():
            if state == 'complete':
                training = unit_dir(campaign, unit) / 'training'
                complete = read_json(training / 'COMPLETE.json')
                model = complete['initializer_model']
                files = {model.split('/')[-1]: complete['files'][model], 'COMPLETE.json': sha(training / 'COMPLETE.json'), 'identity.json': complete['files']['identity.json'],
                         'progress.json': complete['files']['progress.json'], 'config.json': sha(unit_dir(campaign, unit) / 'config.json')}
                exports[unit] = {'training_dir': str(training), 'export': model, 'files': files, 'paths': {model.split('/')[-1]: model, 'COMPLETE.json': 'COMPLETE.json',
                                 'identity.json': 'identity.json', 'progress.json': 'progress.json', 'config.json': '../config.json'}}
        value['exports'] = exports
        value['utilization_warnings'] = sorted(str(path.relative_to(campaign)) for path in campaign.glob('units/*/allocations/wave-*/UTILIZATION_*.json'))
    else:
        admitted = sorted(path.parent.name for path in (campaign / 'initializers').glob('*/ADMITTED.json'))
        waiting = [unit for unit, state in states.items() if state in ('ready', 'buildable')]
        policy = context['policy']
        budget = storage_budget(context, states)
        room = None if budget['free'] is None else budget['free'] - budget['reserve'] - budget['owed']
        pending_parents = {read_json(unit_dir(campaign, unit) / 'unit.json')['parent'] for unit in waiting}
        slots = max(0, policy['storage']['staged_initializers_ahead'] - len(pending_parents))
        if room is not None and room < policy['storage']['estimates_until_measured']['initializer_bytes'] * 2:
            slots = 0
        archivable, tests = {}, {}
        for unit, state in states.items():
            tests[unit] = inference_state(context, unit, state, jobs)
            if state == 'complete' and tests[unit] == 'complete' and inference_verified(unit_dir(campaign, unit)) \
                    and not (campaign / 'state' / 'archive' / (unit + '.json')).exists():
                archivable[unit] = archive_item(context, unit)
        value.update({'test_states': tests, 'test_counts': count(tests), 'storage_decision': storage_decision(campaign),
                      'reused': sorted(unit for unit in states if (unit_dir(campaign, unit) / 'REUSED.json').exists())})
        value.update({'admitted_parents': admitted, 'delivery_slots': slots, 'archivable': archivable, 'budget': budget,
                      'archived': sorted(path.stem for path in (campaign / 'state' / 'archive').glob('*.json')) if (campaign / 'state' / 'archive').is_dir() else []})
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


def stop(args):
    context = load(args.campaign)
    campaign = context['campaign']
    check(args.authority.strip(), 'stop requires --authority')
    if args.all:
        write_exclusive(campaign / 'OPERATOR_STOP.json', {'authority': args.authority, 'time': time.time()})
        targets = context['manifest']['units']
    else:
        write_exclusive(unit_dir(campaign, args.unit) / 'STOP.json', {'authority': args.authority, 'time': time.time()})
        targets = [args.unit]
    jobs = user_jobs() or {}
    signalled = []
    for unit in targets:
        for job in sorted(recorded_jobs(unit_dir(campaign, unit))):
            states = {value['state'] for value in jobs.values() if value['base'] == job}
            if 'RUNNING' in states:
                signalled.append({'job': job, 'result': run(['scancel', '--batch', '--signal=TERM', job], 30)})
            elif states:
                signalled.append({'job': job, 'cancelled_pending': run(['scancel', job], 30)})
    event(campaign, {'event': 'stop', 'targets': len(targets), 'signalled': signalled})
    print(json.dumps({'status': 'ok', 'signalled': signalled}, indent=2))
    return 0


def status(args):
    return inventory(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('stage')
    p.add_argument('--campaign', required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--deploy', required=True)
    p = sub.add_parser('release')
    p.add_argument('--campaign', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('--approve', action='store_true')
    p.add_argument('--max-active', type=int)
    p.add_argument('--partition-choice', choices=['batch', 'pi', 'freecycle'], default='batch')
    p.add_argument('--quota-command', help='JSON list printing available bytes on its last line')
    p.add_argument('--storage-decision', help='explicit clmpt-suite-storage-decision/1 file; without it every release of reportable copies stays off')
    p.add_argument('--rehearsal-fake-slurm', action='store_true')
    for name in ('tick', 'inventory', 'status', 'gate'):
        p = sub.add_parser(name)
        p.add_argument('--campaign', required=True)
        if name == 'tick':
            p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('admit')
    p.add_argument('--campaign', required=True)
    p.add_argument('--parent', required=True)
    p = sub.add_parser('ack-archive')
    p.add_argument('--campaign', required=True)
    p.add_argument('--unit', required=True)
    p = sub.add_parser('verify-archive')
    p.add_argument('--root', required=True)
    p.add_argument('--unit', required=True)
    p = sub.add_parser('stop')
    p.add_argument('--campaign', required=True)
    p.add_argument('--authority', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--unit')
    group.add_argument('--all', action='store_true')
    p = sub.add_parser('record-storage-decision')
    p.add_argument('--campaign', required=True)
    p.add_argument('--decision', required=True)
    p.add_argument('--authority', required=True)
    p = sub.add_parser('hpo-archive-list')
    p.add_argument('--campaign', required=True)
    p.add_argument('--hpo-campaign', required=True)
    p = sub.add_parser('ack-hpo-archive')
    p.add_argument('--campaign', required=True)
    p.add_argument('--unit', required=True)
    p = sub.add_parser('release-hpo-copies')
    p.add_argument('--campaign', required=True)
    p.add_argument('--hpo-campaign', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    handlers = {'stage': stage, 'release': release, 'tick': tick, 'inventory': inventory, 'status': status, 'gate': status, 'admit': admit,
                'ack-archive': ack_archive, 'verify-archive': verify_archive, 'stop': stop, 'record-storage-decision': record_storage,
                'hpo-archive-list': hpo_archive_list, 'ack-hpo-archive': ack_hpo_archive, 'release-hpo-copies': release_hpo_copies}
    try:
        return handlers[args.command](args)
    except Refusal as error:
        print(json.dumps({'refused': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
