"""Bounded, idempotent query-stage advancer for the boundary sweep (cluster login node or CPU job; never holds a GPU).

Ported from the executed clmpt-axiom13-hpo advancer (ops/advance_stage.py, sha256 1e6d5eab...): same trial
classification, attention/halt semantics, exclusive immutable records, frozen-selector binding and submit-once
receipts. Scope here: one taxonomy mode, one query recipe, no pretraining, no S3/S4/S6 grids, no pruning beyond the
trainer's own retention.

tick    (login-node safe) under a campaign lock: refuse unless the frozen manifest verifies and ROOT_RELEASE approves
        it and neither HALT.json nor OPERATOR_STOP.json exists; classify every trial from its files and one squeue
        snapshot; admit each delivered initializer (bridge delivery + lineage re-verification) and create its screen
        trial; only when all 8 screens are complete and the control verifies, write the screen index and submit one
        CPU selection job; create continuations of the promoted new screens; after they complete, write the final
        index and submit one CPU selection job; submit READY GPU trials once each within the GPU-trial cap.
select  inside its own CPU Slurm job only: run select_boundary.py once for the written index.
release write ROOT_RELEASE.json once, after review, with sbatch --test-only admission of both job scripts.
status  read-only snapshot.
Ambiguity, refusal, trial attention or a failed selection writes HALT.json: nothing further is submitted, selected
or retried until a person resolves it.
"""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def _load(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load('boundary_campaign', 'campaign.py')
K = _load('boundary_controller', 'controller.py')
Refusal, check, sha, read_json = C.Refusal, C.check, C.sha, C.read_json

STATE = 'state'
WAVE = re.compile(r'wave-\d{3}')
ACTIVE_STATES = ('active', 'unknown', 'suspect')
TERMINAL_SACCT = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
STAGE_PRIORITY = ('S2', 'S5')
SELECTION_STAGES = ('screen', 'final')


class Halt(Exception):
    """Campaign-level stop: recorded in HALT.json, never retried automatically."""


def state_dir(context):
    return context['campaign'] / STATE


def event(context, value):
    C.append_line(context['campaign'], state_dir(context) / 'events.jsonl', value)


def halt(context, reason, detail=None):
    path = state_dir(context) / 'HALT.json'
    value = {'reason': reason, 'detail': detail, 'time': time.time(), 'resolution': 'a person inspects, resolves and moves HALT.json aside; the advancer never clears it'}
    if not os.path.lexists(str(path)):
        C.write_exclusive(context['campaign'], path, value)
    event(context, {'event': 'halt', 'reason': reason, 'detail': detail})
    raise Halt(reason)


def load(argument, require_release=True):
    campaign = C.campaign_root(argument)
    context = C.verify_manifest(campaign)
    check(sha(Path(__file__).resolve()) == context['manifest']['code_files']['advance.py'] and HERE == context['code_dir'], 'running advancer differs from the frozen checkout')
    check(sys.flags.optimize == 0, 'refusing python -O: the frozen selector relies on assert')
    release_path = campaign / 'ROOT_RELEASE.json'
    present = release_path.exists()
    release = read_json(release_path) if present else {}
    approved = release.get('approved') is True and release.get('manifest_sha256') == context['manifest_sha256']
    if require_release:
        check(present, 'ROOT_RELEASE.json missing')
        check(approved, 'ROOT_RELEASE.json does not approve this frozen manifest')
    context.update({'release': release if approved else {}, 'release_present': present, 'release_approved': approved,
                    'release_sha256': sha(release_path) if present else None, 'runtime': context['protocol']['runtime']})
    return context


def run(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=K.slurm_env() if command[0] in ('sbatch', 'squeue', 'sacct') else None)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'timeout': False}
    except subprocess.TimeoutExpired as error:
        return {'returncode': None, 'stdout': str(error.stdout), 'stderr': str(error.stderr), 'timeout': True}


def scheduler_jobs():
    result = run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%b|%j'], 60)
    if result['returncode'] != 0:
        return None
    jobs = {}
    for line in result['stdout'].splitlines():
        if line.strip():
            identifier, status, gres, name = line.strip().split('|', 3)
            key = identifier.split('_')[0]
            tasks = jobs[key]['tasks'] + 1 if key in jobs else 1
            jobs[key] = {'state': status, 'name': name, 'gpu': 'gpu' in gres.lower(), 'tasks': tasks}
    return jobs


def receipt_job(trial):
    for name in ('submission-reconciled.json', 'submission-receipt.json'):
        if (trial / name).exists():
            value = read_json(trial / name)
            token = str(value.get('stdout', '')).strip().split(';')[0]
            if value.get('returncode') == 0 and token.isdigit():
                return token
    return None


def recorded_jobs(trial):
    jobs = set()
    token = receipt_job(trial)
    if token:
        jobs.add(token)
    allocations = trial / 'allocations'
    if allocations.is_dir():
        for wave in sorted(path for path in allocations.iterdir() if WAVE.fullmatch(path.name)):
            for name in ('allocation.json', 'successor.json'):
                if (wave / name).exists():
                    token = str(read_json(wave / name).get('job', ''))
                    if token.isdigit():
                        jobs.add(token)
    return jobs


def discover(context):
    root = context['campaign'] / 'trials'
    trials = []
    for stage in STAGE_PRIORITY:
        folder = root / stage
        for trial in sorted(folder.iterdir()) if folder.is_dir() else []:
            if (trial / 'runtime.json').exists() and not trial.is_symlink():
                trials.append(trial)
    check(len(trials) <= 64, 'unexpected trial count')
    return trials


def classify(trial, jobs):
    if not (trial / 'config.json').exists():
        return 'malformed'
    receipt, marker = trial / 'submission-receipt.json', trial / 'INITIAL_SUBMITTING.json'
    if not receipt.exists() and not marker.exists():
        return 'ready'
    if receipt_job(trial) is None:
        return 'submission_ambiguous'
    waves = sorted(path for path in (trial / 'allocations').iterdir() if WAVE.fullmatch(path.name)) if (trial / 'allocations').is_dir() else []
    if any((wave / 'ATTENTION.json').exists() or (wave / 'AMBIGUOUS_SUCCESSOR.json').exists() for wave in waves):
        return 'attention'
    if (trial / 'training' / 'ATTENTION.json').exists():
        return 'attention'
    if waves and (waves[-1] / 'ALLOCATION_COMPLETE.json').exists() and (trial / 'training' / 'COMPLETE.json').exists():
        return 'complete'
    if jobs is None:
        return 'unknown'
    if any(job in jobs for job in recorded_jobs(trial)):
        return 'active'
    return 'suspect'


def reconcile(context, trial, runtime, jobs):
    """Resolve a submit marker without a usable receipt from the scheduler by the trajectory's unique job name; never resubmits."""
    marker = trial / 'INITIAL_SUBMITTING.json'
    if not marker.exists():
        return None
    since = read_json(marker)['time'] - 600
    found = {identifier for identifier, value in (jobs or {}).items() if value['name'] == runtime['job_name']}
    accounting = run(['sacct', '-n', '-X', '-P', '--name', runtime['job_name'], '-S', time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(since)), '-o', 'JobID'], 60)
    if accounting['returncode'] == 0:
        found |= {line.strip() for line in accounting['stdout'].splitlines() if line.strip().isdigit()}
    detail = {'job_name': runtime['job_name'], 'found': sorted(found), 'squeue_available': jobs is not None, 'sacct_returncode': accounting['returncode']}
    if len(found) == 1 and jobs is not None and accounting['returncode'] == 0:
        identifier = found.pop()
        C.write_exclusive(context['campaign'], trial / 'submission-reconciled.json', {'returncode': 0, 'stdout': identifier, 'reconciled_from_scheduler': True,
                                                                                   'job_name': runtime['job_name'], 'time': time.time()})
        event(context, {'event': 'submission_reconciled', 'trial': str(trial), 'job': identifier})
        return 'reconciled'
    if len(found) > 1:
        event(context, {'event': 'submission_duplicates', 'trial': str(trial), 'detail': detail})
        return 'duplicates'
    if jobs is None or accounting['returncode'] != 0 or time.time() - read_json(marker)['time'] < context['runtime']['suspect_grace_seconds']:
        return 'waiting'
    event(context, {'event': 'submission_unresolved', 'trial': str(trial), 'detail': detail})
    return 'unresolved'


def classify_all(context, jobs, now, mutate=True):
    suspects_path = state_dir(context) / 'suspects.json'
    suspects = read_json(suspects_path) if suspects_path.exists() else {}
    states, runtimes, kept = {}, {}, {}
    for trial in discover(context):
        runtime = read_json(trial / 'runtime.json')
        state = classify(trial, jobs)
        if state == 'submission_ambiguous' and mutate:
            outcome = reconcile(context, trial, runtime, jobs)
            if outcome == 'reconciled':
                state = classify(trial, jobs)
            elif outcome == 'waiting':
                state = 'unknown'
            else:
                record_attention(context, trial, runtime, 'submission_ambiguous:%s' % outcome)
        key = str(trial)
        if state == 'suspect':
            first = suspects.get(key, now)
            kept[key] = first
            if now - first >= context['runtime']['suspect_grace_seconds']:
                state = 'attention'
                if mutate:
                    record_attention(context, trial, runtime, 'submitted trial ended without completion or controller marker (cancelled or lost); never revived')
        if state == 'attention' and mutate:
            record_attention(context, trial, runtime, 'attention marker or failed allocation')
        states[key] = state
        runtimes[key] = runtime
    if mutate:
        C.write_replace(context['campaign'], suspects_path, kept)
    return states, runtimes


def record_attention(context, trial, runtime, reason):
    path = state_dir(context) / 'attention' / (runtime['trial_id'] + '.json')
    if not os.path.lexists(str(path)):
        C.write_exclusive(context['campaign'], path, {'trial': str(trial), 'trial_id': runtime['trial_id'], 'stage': runtime['stage'], 'reason': reason,
                                                      'first_seen': time.time(), 'automatic_retry': False})
        event(context, {'event': 'trial_attention', 'trial': str(trial), 'reason': reason})


def delivery_name(expected):
    return '%s-u%07d' % (expected['trajectory'], expected['initializer_update_budget'])


def verify_delivery(context, expected, folder):
    """Initializer record for a delivered milestone, re-verified from the shipped pretraining metadata (port of verify_milestone)."""
    protocol = context['protocol']
    lineage, budget = protocol['pretraining_lineage'], expected['initializer_update_budget']
    admission = read_json(folder / 'ADMISSION.json')
    check(admission.get('schema') == 'clmpt-boundary-delivery/1' and admission['trajectory'] == expected['trajectory']
          and admission['initializer_update_budget'] == budget, 'delivery does not name its expected milestone: %s' % folder)
    model = folder / ('update-%07d.pt' % budget)
    check(set(admission['files']) == {model.name, 'identity.json', 'config.json', 'commits.jsonl'}, 'delivery file set differs: %s' % folder)
    for name, digest in admission['files'].items():
        check(sha(folder / name) == digest, 'delivered file hash differs: %s/%s' % (folder, name))
    model_sha = admission['files'][model.name]
    check(model_sha != protocol['control']['pretrained_sha256'], 'delivered model equals the control initializer')
    identity, config = read_json(folder / 'identity.json'), read_json(folder / 'config.json')
    check(admission['files']['config.json'] == expected['pretrain_config_sha256'] == identity['config_sha256'], 'pretraining config differs from its pin: %s' % folder)
    check(identity['config'] == {key: value for key, value in config.items() if key != 'data_dir'}, 'pretraining identity config differs: %s' % folder)
    check(identity['sources'] == lineage['source_hashes'] and identity['protocol_sha256'] == config['protocol_sha256'] == lineage['protocol_sha256'], 'pretraining sources or protocol differ: %s' % folder)
    check(identity['data_complete_sha256'] == config['data_complete_sha256'] == lineage['graph_complete_sha256'], 'pretraining graph differs: %s' % folder)
    check(config['campaign_trial_id'] == expected['pretrain_campaign_trial_id'] and config['learning_rate'] == expected['pretrain_learning_rate']
          and config['regularization'] == expected['pretrain_regularization'] and budget in config['milestone_updates'], 'pretraining trajectory differs: %s' % folder)
    for key, value in lineage['fixed'].items():
        check(config[key] == value, 'pretraining fixed setting %s differs: %s' % (key, folder))
    identity_sha, artifact, journaled = C.canonical(identity), lineage['milestone_model_name'] % budget, None
    with (folder / 'commits.jsonl').open() as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get('identity_sha256') == identity_sha and entry.get('artifacts', {}).get(artifact) == model_sha and entry.get('step', -1) >= budget:
                journaled = entry
                break
    check(journaled is not None, 'milestone %s not in the shipped commit journal: %s' % (artifact, folder))
    return {'schema': 'clmpt-boundary-initializer/1', 'taxonomy_mode': 'original', 'initializer_trial': expected['pretrain_campaign_trial_id'],
            'trajectory': expected['trajectory'], 'initializer_update_budget': budget, 'pretrain_learning_rate': expected['pretrain_learning_rate'],
            'pretrain_regularization': expected['pretrain_regularization'], 'path': str(model), 'sha256': model_sha,
            'source_identity_sha256': identity_sha, 'source_config_sha256': identity['config_sha256'], 'graph_complete_sha256': identity['data_complete_sha256'],
            'pretrain_protocol_sha256': identity['protocol_sha256'], 'journal_step': journaled['step'], 'journal_reason': journaled.get('reason'),
            'pretrain_torch': identity.get('torch'), 'pretrain_cuda_runtime': identity.get('cuda_runtime'), 'delivery': admission.get('lineage'),
            'delivery_admission_sha256': sha(folder / 'ADMISSION.json'), 'hardware_note': lineage['hardware_note']}


def initializers(context, mutate=True):
    found = {}
    for expected in context['protocol']['expected_initializers']:
        name = delivery_name(expected)
        record_path = state_dir(context) / 'initializers' / (name + '.json')
        if record_path.exists():
            found[name] = read_json(record_path)
            continue
        folder = context['campaign'] / 'initializers' / name
        if not (folder / 'ADMISSION.json').exists():
            continue
        record = verify_delivery(context, expected, C.inside(context['campaign'], folder))
        if mutate:
            C.write_exclusive(context['campaign'], record_path, record)
            event(context, {'event': 'initializer_admitted', 'name': name, 'sha256': record['sha256']})
        found[name] = record
    return found


def screen_id(context, record):
    screen = context['protocol']['stages']['S2_screen']
    return C.query_id('S2', 'original', record['pretrain_learning_rate'], record['pretrain_regularization'], record['initializer_update_budget'],
                      screen['learning_rate'], screen['temperature'])


def create_trial(context, stage, identifier, config, allocations, order_key):
    folder = C.inside(context['campaign'], C.trial_dir(context['campaign'], stage, identifier))
    states = [C.write_exclusive(context['campaign'], folder / 'config.json', config),
              C.write_exclusive(context['campaign'], folder / 'runtime.json', C.runtime(stage, identifier, allocations, order_key))]
    if 'created' in states:
        event(context, {'event': 'trial_created', 'trial': str(folder), 'stage': stage})
    return 'created' in states


def materialize_screens(context, found):
    screen, created = context['protocol']['stages']['S2_screen'], 0
    for record in found.values():
        identifier = screen_id(context, record)
        config = C.query_config(context, record, screen['learning_rate'], screen['temperature'], screen['steps'], screen['mixture_id'], identifier)
        created += create_trial(context, 'S2', identifier, config, context['runtime']['maximum_allocations_query_2500'],
                                [record['initializer_update_budget'], record['pretrain_learning_rate']])
    return created


def verify_control_light(context):
    control = context['manifest']['control']
    check(control['protocol_sha256'] == context['protocol']['control']['protocol_sha256'] and sha(control['protocol_path']) == control['protocol_sha256'], 'control protocol changed')
    for part in ('screen', 'long'):
        check(sha(Path(control[part]['training_dir']) / 'COMPLETE.json') == control[part]['complete_sha256'] == context['protocol']['control'][part]['complete_sha256'],
              'control %s COMPLETE changed' % part)


def completed_entry(folder, expected_steps, protocol_sha256):
    training = folder / 'training'
    complete = read_json(training / 'COMPLETE.json')
    check(complete['steps'] == expected_steps, 'trial %s completed %s steps, expected %d' % (folder, complete['steps'], expected_steps))
    identity = read_json(training / 'identity.json')
    check(C.canonical(identity) == complete['identity_sha256'] and identity['config_sha256'] == sha(folder / 'config.json'), 'trial identity differs from its config: %s' % folder)
    return {'id': read_json(folder / 'runtime.json')['trial_id'], 'training_dir': str(training), 'complete_sha256': sha(training / 'COMPLETE.json'),
            'role': 'new', 'protocol_sha256': protocol_sha256}


def control_entry(context, part):
    value = context['manifest']['control'][part]
    return {'id': value['trial_id'], 'training_dir': value['training_dir'], 'complete_sha256': value['complete_sha256'], 'role': 'control',
            'protocol_sha256': context['manifest']['control']['protocol_sha256']}


def write_index(context, stage, entries):
    ids = [entry['id'] for entry in entries]
    check(len(set(ids)) == len(ids), 'duplicate %s index entries' % stage)
    path = context['campaign'] / 'decisions' / ('%s-index.json' % stage)
    value = {'schema': 'clmpt-boundary-stage-index/1', 'stage': stage, 'protocol_sha256': context['protocol_sha256'], 'complete_planned_grid': True,
             'entries': sorted(entries, key=lambda entry: entry['id'])}
    if C.write_exclusive(context['campaign'], path, value) == 'created':
        event(context, {'event': 'index', 'stage': stage, 'entries': len(entries)})
    return path


def select_command(context, stage, test_only=False):
    command = ['sbatch'] + (['--test-only'] if test_only else ['--parsable']) + ['--no-requeue', '--job-name=' + C.job_name('select-' + stage),
               '--output=' + str(context['campaign'] / 'logs' / '%x-%j.out'), str(context['campaign'] / 'ops' / 'select_job.sh'), str(context['campaign']), stage]
    return command


def selection(context, stage, index_path, jobs):
    """Frozen-selector decision bound to this index, computed once by a CPU job; None while pending."""
    output = context['campaign'] / 'decisions' / ('%s.json' % stage)
    folder = state_dir(context) / 'select' / stage
    if not output.exists():
        marker, receipt = folder / 'SUBMITTING.json', folder / 'submission-receipt.json'
        if not marker.exists():
            if jobs is None:
                return None
            C.write_exclusive(context['campaign'], marker, {'command': select_command(context, stage), 'index_sha256': sha(index_path), 'time': time.time()})
            result = run(select_command(context, stage), 30)
            C.write_exclusive(context['campaign'], receipt, dict(result, time=time.time()))
            token = result['stdout'].strip().split(';')[0] if result['returncode'] == 0 else ''
            if not token.isdigit():
                halt(context, 'selection_submission_ambiguous', {'stage': stage, 'result': result})
            event(context, {'event': 'selection_submitted', 'stage': stage, 'job': token})
            return None
        check(receipt.exists(), 'selection marker without receipt: %s' % stage)
        job = read_json(receipt)['stdout'].strip().split(';')[0]
        if jobs is None or job in jobs:
            return None
        accounting = run(['sacct', '-j', job, '-X', '-n', '-P', '-o', 'State,ExitCode,Elapsed'], 60)
        line = accounting['stdout'].strip().splitlines()[0] if accounting['returncode'] == 0 and accounting['stdout'].strip() else ''
        if not output.exists():
            if line.split('|')[0].split(' ')[0] not in TERMINAL_SACCT:
                wait_path = folder / 'accounting-wait.json'
                if not wait_path.exists():
                    C.write_exclusive(context['campaign'], wait_path, {'job': job, 'first_seen': time.time()})
                if time.time() - read_json(wait_path)['first_seen'] < context['runtime']['suspect_grace_seconds']:
                    return None
            halt(context, 'selection_job_ended_without_decision', {'stage': stage, 'job': job, 'sacct': line, 'retry': 'never automatic'})
    value = read_json(output)
    files = context['manifest']['code_files']
    if not (value.get('stage') == stage and value.get('test_access') is False and value.get('index_sha256') == sha(index_path)
            and value.get('protocol_sha256') == context['protocol_sha256'] and value.get('selector_sha256') == files['analysis/select_stage.py'] == context['protocol']['selector_sha256']
            and value.get('wrapper_sha256') == files['select_boundary.py']):
        halt(context, 'decision_binding_mismatch', {'stage': stage, 'decision': str(output)})
    C.write_exclusive(context['campaign'], output.with_name(stage + '.frozen.json'), {'decision': str(output), 'decision_sha256': sha(output), 'index': str(index_path),
                                                                                    'index_sha256': sha(index_path), 'stage': stage})
    return value


def advance(context, states, found, jobs):
    protocol, campaign = context['protocol'], context['campaign']
    expected = protocol['expected_initializers']
    missing = [delivery_name(row) for row in expected if delivery_name(row) not in found]
    if missing:
        return {'gate': 'waiting_for_initializers', 'missing': missing}
    screens = [C.trial_dir(campaign, 'S2', screen_id(context, found[delivery_name(row)])) for row in expected]
    if not all(states.get(str(folder)) == 'complete' for folder in screens):
        return {'gate': 'S2_running_or_incomplete'}
    verify_control_light(context)
    entries = [completed_entry(folder, protocol['continuation_parent_steps'], context['protocol_sha256']) for folder in screens] + [control_entry(context, 'screen')]
    index = write_index(context, 'screen', entries)
    decision = selection(context, 'screen', index, jobs)
    if decision is None:
        return {'gate': 'screen_selection_pending'}
    continuations = []
    for rank, row in enumerate(sorted(decision['continue'], key=lambda value: value['id'])):
        parent = Path(row['continuation_parent'])
        check(parent.name == 'training' and str(parent.parent) in {str(folder) for folder in screens}, 'continuation parent outside the new screens')
        identifier = 'S5-' + row['id']
        config = C.continuation_config(context, read_json(parent.parent / 'config.json'), row['continuation_parent'], row['parent_complete_sha256'],
                                       row['parent_checkpoint_sha256'], identifier)
        check(config['maximum_steps'] == row['maximum_steps'] == protocol['continuation_total_steps'], 'continuation budget differs')
        continuations.append({'id': identifier, 'parent_id': row['id'], 'trial_dir': str(C.trial_dir(campaign, 'S5', identifier))})
        create_trial(context, 'S5', identifier, config, context['runtime']['maximum_allocations_query_20000'], [rank])
    reuse = decision['reuse_control_long']
    if reuse is not None:
        check(reuse == protocol['control']['long'], 'control reuse differs from the protocol pin')
    C.write_exclusive(campaign, campaign / 'decisions' / 'S5-plan.json', {'schema': 'clmpt-boundary-s5-plan/1', 'screen_decision_sha256': sha(campaign / 'decisions' / 'screen.json'),
                                                                          'continuations': continuations, 'reuse_control_long': reuse})
    if not all(states.get(row['trial_dir']) == 'complete' for row in continuations):
        return {'gate': 'S5_continuations_running_or_incomplete'}
    entries = [completed_entry(Path(row['trial_dir']), protocol['continuation_total_steps'], context['protocol_sha256']) for row in continuations]
    if reuse is not None:
        entries.append(control_entry(context, 'long'))
    index = write_index(context, 'final', entries)
    final = selection(context, 'final', index, jobs)
    if final is None:
        return {'gate': 'final_selection_pending'}
    C.write_exclusive(campaign, state_dir(context) / 'DONE.json', {'final_decision_sha256': sha(campaign / 'decisions' / 'final.json'),
                                                                   'S5_plan_sha256': sha(campaign / 'decisions' / 'S5-plan.json'),
                                                                   'note': 'all planned GPU trials complete and both decisions frozen; reporting is separate'})
    event(context, {'event': 'done'})
    return {'gate': 'done'}


def available_bytes(context):
    result = run(['df', '-B1', '--output=avail', str(context['campaign'])], 60)
    check(result['returncode'] == 0, 'df failed: %s' % result['stderr'])
    return int(result['stdout'].splitlines()[-1])


def submit_trial(context, trial, runtime):
    """Port of the executed ops/submit.py: scheduler admission test, exclusive marker, one sbatch, receipt; never retried."""
    campaign = context['campaign']
    check(not os.path.lexists(str(campaign / 'OPERATOR_STOP.json')), 'operator stop')
    check(not (trial / 'INITIAL_SUBMITTING.json').exists() and not (trial / 'submission-receipt.json').exists(), 'trial already submitted: %s' % trial)
    command = K.sbatch_command(campaign, trial, 0, runtime['job_name'])
    dry = run([command[0], '--test-only'] + [part for part in command[1:] if part != '--parsable'], 30)
    C.write_replace(campaign, trial / 'scheduler-readiness.json', dry)
    if dry['returncode'] != 0:
        return {'submitted': False, 'reason': 'sbatch --test-only refused', 'stderr': dry['stderr'][-1000:]}
    C.write_exclusive(campaign, trial / 'INITIAL_SUBMITTING.json', {'time': time.time(), 'manifest_sha256': context['manifest_sha256'], 'command': command})
    result = run(command, 30)
    receipt = dict(result, time=time.time())
    C.write_exclusive(campaign, trial / 'submission-receipt.json', receipt)
    token = result['stdout'].strip().split(';')[0] if result['returncode'] == 0 else ''
    return {'submitted': token.isdigit(), 'job': token, 'receipt': receipt}


def submit_ready(context, states, runtimes, jobs):
    """Submit READY trials once each; the cap counts this campaign's active trials plus the user's other queued or running GPU jobs."""
    release, runtime_policy = context['release'], context['runtime']
    released = set(release.get('stages', []))
    cap = min(runtime_policy['max_concurrent_gpu_trials'], int(release.get('max_concurrent_gpu_trials', runtime_policy['max_concurrent_gpu_trials'])))
    own = set()
    for key in states:
        own |= recorded_jobs(Path(key))
    foreign = ['%s#%d' % (identifier, task) for identifier, value in sorted(jobs.items()) if value['gpu'] and identifier not in own
               and not value['name'].startswith(C.JOB_NAME_PREFIX) for task in range(value['tasks'])]
    active = [key for key, state in states.items() if state in ACTIVE_STATES] + foreign
    ready = sorted((key for key, state in states.items() if state == 'ready'),
                   key=lambda key: (STAGE_PRIORITY.index(runtimes[key]['stage']), runtimes[key]['order_key'], runtimes[key]['trial_id']))
    submitted, skipped = [], {}
    for key in ready:
        runtime = runtimes[key]
        if runtime['stage'] not in released:
            skipped.setdefault('stage_not_in_ROOT_RELEASE', []).append(runtime['trial_id'])
            continue
        if len(active) + len(submitted) >= cap:
            skipped.setdefault('gpu_trial_cap_%d' % cap, []).append(runtime['trial_id'])
            continue
        free, need = available_bytes(context), max(runtime_policy['required_free_bytes'], int(release.get('required_free_bytes', 0)))
        if free < need:
            skipped.setdefault('storage_reserve', []).append({'trial': runtime['trial_id'], 'free': free, 'required': need})
            break
        outcome = submit_trial(context, Path(key), runtime)
        refusals_path = state_dir(context) / 'submit-refusals.json'
        if 'receipt' not in outcome:
            refusals = read_json(refusals_path) if refusals_path.exists() else {'consecutive': 0, 'last': []}
            refusals = {'consecutive': refusals['consecutive'] + 1, 'last': (refusals['last'] + [{'trial': key, 'reason': outcome['reason'], 'stderr': outcome['stderr'], 'time': time.time()}])[-5:]}
            C.write_replace(context['campaign'], refusals_path, refusals)
            event(context, {'event': 'submit_refused_before_marker', 'trial': key, 'consecutive': refusals['consecutive']})
            if refusals['consecutive'] >= 3:
                halt(context, 'submission_refused_before_sbatch_3_times', refusals)
            skipped.setdefault('submit_refused_before_marker_nothing_submitted', []).append(runtime['trial_id'])
            break
        C.write_exclusive(context['campaign'], state_dir(context) / 'submissions' / (runtime['trial_id'] + '.json'), dict(outcome, trial=key, recorded_at=time.time()))
        C.write_replace(context['campaign'], refusals_path, {'consecutive': 0, 'last': []})
        if not outcome['submitted']:
            event(context, {'event': 'submission_ambiguous', 'trial': key, 'next': 'next tick reconciles by unique job name; never resubmitted'})
            skipped.setdefault('submission_ambiguous_reconcile_next_tick', []).append(runtime['trial_id'])
            break
        event(context, {'event': 'submitted', 'trial': key, 'stage': runtime['stage'], 'job': outcome['job']})
        submitted.append({'trial': runtime['trial_id'], 'job': outcome['job']})
    return submitted, skipped, cap, len(active)


def counts(states, runtimes):
    value = {}
    for key, state in states.items():
        value.setdefault(runtimes[key]['stage'], {}).setdefault(state, 0)
        value[runtimes[key]['stage']][state] += 1
    return value


def tick(argument):
    context = load(argument)
    campaign = context['campaign']
    folder = state_dir(context)
    folder.mkdir(parents=True, exist_ok=True)
    lock = os.open(str(C.inside(campaign, folder / 'advance.lock')), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {'status': 'busy'}
        if (folder / 'HALT.json').exists():
            return {'status': 'halted', 'halt': read_json(folder / 'HALT.json')}
        if os.path.lexists(str(campaign / 'OPERATOR_STOP.json')):
            return {'status': 'operator_stop'}
        if (folder / 'DONE.json').exists():
            return {'status': 'done', 'done': read_json(folder / 'DONE.json')}
        summary = {'status': 'ok', 'time': time.time(), 'code_revision': context['manifest']['code_revision']}
        try:
            jobs = scheduler_jobs()
            summary['scheduler_available'] = jobs is not None
            verify_control_light(context)
            states, runtimes = classify_all(context, jobs, time.time())
            if any(state == 'malformed' for state in states.values()):
                halt(context, 'malformed_trial_directory', sorted(key for key, state in states.items() if state == 'malformed'))
            records = sorted((folder / 'attention').glob('*.json')) if (folder / 'attention').is_dir() else []
            attention = [path.stem for path in records if not path.name.endswith('.ack.json') and not path.with_name(path.stem + '.ack.json').exists()]
            summary['unacknowledged_trial_attention'] = attention
            if len(attention) >= context['runtime']['attention_halt_threshold']:
                halt(context, 'trial_attention_threshold', attention)
            found = initializers(context)
            summary['initializers'] = sorted(found)
            summary['S2_created'] = materialize_screens(context, found)
            states, runtimes = classify_all(context, jobs, time.time(), mutate=False)
            summary['gate'] = advance(context, states, found, jobs)
            states, runtimes = classify_all(context, jobs, time.time(), mutate=False)
            if jobs is None:
                summary['submitted'], summary['skipped'] = [], {'scheduler_unavailable': True}
            else:
                summary['submitted'], summary['skipped'], summary['cap'], summary['active_before'] = submit_ready(context, states, runtimes, jobs)
            summary['trial_states'] = counts(states, runtimes)
        except Halt as reason:
            summary.update({'status': 'halted', 'reason': str(reason)})
        except Refusal as error:
            try:
                halt(context, 'refusal', str(error))
            except Halt:
                summary.update({'status': 'halted', 'reason': 'refusal', 'detail': str(error)})
        C.write_replace(campaign, folder / 'last-tick.json', summary)
        return summary
    finally:
        os.close(lock)


def select(argument, stage):
    context = load(argument)
    check(os.environ.get('SLURM_JOB_ID', '').isdigit() and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'select runs only inside its CPU Slurm job')
    campaign = context['campaign']
    index = campaign / 'decisions' / ('%s-index.json' % stage)
    output = campaign / 'decisions' / ('%s.json' % stage)
    check(index.exists() and not output.exists(), 'selection %s: index missing or decision already present' % stage)
    marker = read_json(state_dir(context) / 'select' / stage / 'SUBMITTING.json')
    check(marker['index_sha256'] == sha(index), 'index changed after the selection job was submitted')
    result = run([sys.executable, '-I', str(HERE / 'select_boundary.py'), '--stage', stage, '--protocol', context['protocol_path'],
                  '--control-protocol', context['manifest']['control']['protocol_path'], '--index', str(index), '--output', str(output)], 3000)
    C.write_exclusive(campaign, campaign / 'decisions' / ('%s.selector-run.json' % stage), dict(result, job=os.environ['SLURM_JOB_ID'], time=time.time()))
    return {'status': 'ok' if result['returncode'] == 0 else 'selector_refused', 'returncode': result['returncode'], 'stderr': result['stderr'][-2000:]}


def release(argument, authority, approve):
    context = load(argument, require_release=False)
    campaign = context['campaign']
    check(approve and authority.strip(), 'release requires --approve and a non-empty --authority')
    check(not context['release_present'], 'ROOT_RELEASE.json exists: release is write-once')
    probes = {'gpu_job': run(['sbatch', '--test-only', '--no-requeue', str(campaign / 'ops' / 'job.sh'), str(campaign), str(campaign / 'trials'), '0'], 30),
              'selection_job': run(select_command(context, 'screen', test_only=True), 30)}
    C.write_replace(campaign, state_dir(context) / 'release-probes.json', probes)
    check(all(probe['returncode'] == 0 for probe in probes.values()), 'scheduler refused a job script: %s' % {key: value['stderr'][-500:] for key, value in probes.items()})
    value = {'approved': True, 'authority': authority, 'manifest_sha256': context['manifest_sha256'], 'protocol_sha256': context['protocol_sha256'],
             'code_revision': context['manifest']['code_revision'], 'stages': list(STAGE_PRIORITY), 'max_concurrent_gpu_trials': context['runtime']['max_concurrent_gpu_trials'],
             'required_free_bytes': context['runtime']['required_free_bytes'], 'probes_sha256': sha(state_dir(context) / 'release-probes.json'), 'time': time.time()}
    C.write_exclusive(campaign, campaign / 'ROOT_RELEASE.json', value)
    return value


def status(argument):
    """Read-only snapshot: no lock, no writes, no subprocess except squeue. Works before ROOT_RELEASE exists."""
    context = load(argument, require_release=False)
    jobs = scheduler_jobs()
    states, runtimes = classify_all(context, jobs, time.time(), mutate=False)
    folder = state_dir(context)
    delivered = sorted(path.name for path in (context['campaign'] / 'initializers').iterdir() if (path / 'ADMISSION.json').exists()) if (context['campaign'] / 'initializers').is_dir() else []
    return {'trial_states': counts(states, runtimes), 'halt': read_json(folder / 'HALT.json') if (folder / 'HALT.json').exists() else None,
            'operator_stop': os.path.lexists(str(context['campaign'] / 'OPERATOR_STOP.json')), 'done': (folder / 'DONE.json').exists(),
            'root_release': {'present': context['release_present'], 'approves_current_manifest': context['release_approved'], 'sha256': context['release_sha256']},
            'delivered_initializers': delivered, 'admitted_initializers': sorted(path.stem for path in (folder / 'initializers').glob('*.json')) if (folder / 'initializers').is_dir() else [],
            'decisions': sorted(path.name for path in (context['campaign'] / 'decisions').glob('*.json')),
            'last_tick': read_json(folder / 'last-tick.json') if (folder / 'last-tick.json').exists() else None}


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('tick', 'status', 'select', 'release'):
        sub = commands.add_parser(name)
        sub.add_argument('--campaign', required=True)
        if name == 'select':
            sub.add_argument('--stage', choices=SELECTION_STAGES, required=True)
        if name == 'release':
            sub.add_argument('--approve', action='store_true')
            sub.add_argument('--authority', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'tick':
            result = tick(args.campaign)
        elif args.command == 'status':
            result = status(args.campaign)
        elif args.command == 'select':
            result = select(args.campaign, args.stage)
        else:
            result = release(args.campaign, args.authority, args.approve)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get('status', 'ok') in ('ok', 'done', 'busy') else 1
    except Refusal as error:
        print(json.dumps({'refused': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
