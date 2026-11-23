"""Bounded, idempotent advancer for the suite HPO campaign (login node or CPU job; never holds a GPU).

Ported from code/clmpt_boundary_queries/advance.py (commit 4518251): same trial classification, attention/halt
semantics, exclusive immutable records, submit-once receipts and scheduler reconciliation by unique job name.

tick             under a campaign lock: refuse unless the frozen manifest verifies, ROOT_RELEASE approves it and no
                 HALT.json/OPERATOR_STOP.json exists. Creates the two qualification trials; once they complete, one
                 CPU job compares them; only a passed (or explicitly waived) qualification creates the 18 screens.
                 Only when all 18 screens are complete, one CPU job selects the continuations; when all
                 continuations are complete, one CPU job selects the per-view winners. The common recipe is frozen
                 only from a recorded selection decision. READY trials and HELD clean pauses are submitted within
                 the user job cap and the storage budget.
status           read-only machine-readable snapshot (also before release).
gate             read-only: qualification, screen, continuation, final and recipe gates only.
release          write ROOT_RELEASE.json once after review, with sbatch --test-only admission of both job scripts.
record-decision  copy an explicit selection decision (common leaderboard raw or calibrated) into the campaign once.
select, qualify  inside their own CPU Slurm job only: run the selector or the qualification comparison once, then
                 prune full-state checkpoints that the just-verified decision makes unnecessary.
resume           submit the next allocation of a STOPPED or HELD trial (or an acknowledged ATTENTION trial with
                 --after-attention) once its stop marker is gone.
stop             write a trial STOP.json (or OPERATOR_STOP.json with --all) and signal its running controller.
ack              acknowledge one trial attention record so the halt threshold no longer counts it.
Ambiguity, refusal, unacknowledged trial attention or a failed decision writes HALT.json; nothing further is
submitted or selected until a person moves it aside.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
import importlib.util


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load('suite_common', 'suite.py')
K = _load('suite_capacity', 'capacity.py')
Refusal, check, sha, read_json = C.Refusal, C.check, C.sha, C.read_json

WAVE = re.compile(r'wave-(\d{3})')
ACTIVE_STATES = ('active', 'unknown', 'suspect')
TERMINAL_SACCT = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
CPU_TARGETS = {'qualify': ('qualification',), 'select': ('screen', 'final', 'recipe')}


class Halt(Exception):
    """Campaign-level stop: recorded in HALT.json, never cleared automatically."""


def state_dir(context):
    return context['campaign'] / 'state'


def event(context, value):
    C.append_line(context['campaign'], state_dir(context) / 'events.jsonl', value)


def halt(context, reason, detail=None):
    path = state_dir(context) / 'HALT.json'
    if not os.path.lexists(str(path)):
        C.write_exclusive(context['campaign'], path, {'reason': reason, 'detail': detail, 'time': time.time(),
                                                      'resolution': 'a person inspects, resolves and moves HALT.json aside; the advancer never clears it'})
    event(context, {'event': 'halt', 'reason': reason, 'detail': detail})
    raise Halt(reason)


def load(argument, require_release=True):
    campaign = C.campaign_root(argument)
    context = C.verify_manifest(campaign)
    check(sha(Path(__file__).resolve()) == context['manifest']['code_files']['advance.py'] and HERE == context['code_dir'], 'running advancer differs from the frozen checkout')
    check(sys.flags.optimize == 0, 'refusing python -O: the selector relies on assert')
    release_path = campaign / 'ROOT_RELEASE.json'
    present = release_path.exists()
    release = read_json(release_path) if present else {}
    approved = release.get('approved') is True and release.get('manifest_sha256') == context['manifest_sha256']
    if require_release:
        check(present and approved, 'ROOT_RELEASE.json missing or does not approve this frozen manifest')
    context.update({'release': release if approved else {}, 'release_present': present, 'release_approved': approved,
                    'release_sha256': sha(release_path) if present else None, 'runtime': context['protocol']['runtime']})
    return context


def waves(trial):
    folder = trial / 'allocations'
    if not folder.is_dir():
        return []
    return sorted((int(WAVE.fullmatch(path.name).group(1)), path) for path in folder.iterdir() if WAVE.fullmatch(path.name))


def resubmission_path(trial, wave):
    return trial / 'resubmissions' / ('wave-%03d.json' % wave)


def receipt_job(trial):
    for name in ('submission-reconciled.json', 'submission-receipt.json'):
        if (trial / name).exists():
            token = C.job_token(read_json(trial / name))
            if token:
                return token
    return None


def recorded_jobs(trial):
    jobs = set()
    token = receipt_job(trial)
    if token:
        jobs.add(token)
    for _, wave in waves(trial):
        for name in ('allocation.json', 'successor.json'):
            if (wave / name).exists():
                token = str(read_json(wave / name).get('job', ''))
                if token.isdigit():
                    jobs.add(token)
    if (trial / 'resubmissions').is_dir():
        for path in (trial / 'resubmissions').glob('wave-*.json'):
            if not path.name.endswith('.SUBMITTING.json'):
                token = C.job_token(read_json(path))
                if token:
                    jobs.add(token)
    return jobs


def stop_marker(context, trial):
    return os.path.lexists(str(context['campaign'] / 'OPERATOR_STOP.json')) or os.path.lexists(str(trial / 'STOP.json'))


def classify(context, trial, jobs):
    if not (trial / 'config.json').exists() or not (trial / 'runtime.json').exists():
        return 'malformed'
    if not (trial / 'submission-receipt.json').exists() and not (trial / 'INITIAL_SUBMITTING.json').exists():
        return 'ready'
    if receipt_job(trial) is None:
        return 'submission_ambiguous'
    if (trial / 'resubmissions').is_dir():
        for marker in (trial / 'resubmissions').glob('*.SUBMITTING.json'):
            receipt = marker.with_name(marker.name.replace('.SUBMITTING.json', '.json'))
            if not receipt.exists() or C.job_token(read_json(receipt)) is None:
                return 'attention'
    present = waves(trial)
    for _, wave in present:
        if ((wave / 'ATTENTION.json').exists() or (wave / 'AMBIGUOUS_SUCCESSOR.json').exists()) and not (wave / 'RESUME_AUTHORIZED.json').exists():
            return 'attention'
    last = present[-1][1] if present else None
    if last is not None and (last / 'ALLOCATION_COMPLETE.json').exists() and (trial / 'training' / 'COMPLETE.json').exists():
        return 'complete'
    following = present[-1][0] + 1 if present else 0
    if jobs is not None and any(job in jobs for job in recorded_jobs(trial)):
        return 'active'
    if resubmission_path(trial, following).exists():
        return 'unknown' if jobs is None else 'suspect'
    if last is not None and (last / 'STOPPED.json').exists():
        return 'stopped'
    if last is not None and (last / 'HELD.json').exists():
        return 'held'
    if jobs is None:
        return 'unknown'
    if stop_marker(context, trial):
        return 'stopped'
    return 'suspect'


def reconcile(context, trial, runtime, jobs):
    """Resolve an initial submit marker without a usable receipt by the trajectory's unique job name; never resubmits."""
    marker = trial / 'INITIAL_SUBMITTING.json'
    if not marker.exists():
        return None
    since = read_json(marker)['time'] - 600
    found = {value['base'] for value in (jobs or {}).values() if value['name'] == runtime['job_name']}
    accounting = C.run(['sacct', '-n', '-X', '-P', '--name', runtime['job_name'], '-S', time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(since)), '-o', 'JobID'], 60)
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


def discover(context):
    trials = []
    for stage in C.STAGES:
        folder = context['campaign'] / 'trials' / stage
        for trial in sorted(folder.iterdir()) if folder.is_dir() else []:
            if trial.is_dir() and not trial.is_symlink():
                trials.append(trial)
    check(len(trials) <= 32, 'unexpected trial count')
    return trials


def record_attention(context, trial, runtime, reason):
    path = state_dir(context) / 'attention' / (runtime['trial_id'] + '.json')
    if not os.path.lexists(str(path)):
        C.write_exclusive(context['campaign'], path, {'trial': str(trial), 'trial_id': runtime['trial_id'], 'stage': runtime['stage'], 'reason': reason,
                                                      'first_seen': time.time(), 'automatic_retry': False})
        event(context, {'event': 'trial_attention', 'trial': str(trial), 'reason': reason})


def classify_all(context, jobs, now, mutate=True):
    suspects_path = state_dir(context) / 'suspects.json'
    suspects = read_json(suspects_path) if suspects_path.exists() else {}
    states, runtimes, kept = {}, {}, {}
    for trial in discover(context):
        state = classify(context, trial, jobs)
        if state == 'malformed':
            states[str(trial)] = state
            continue
        runtime = read_json(trial / 'runtime.json')
        if state == 'submission_ambiguous' and mutate:
            outcome = reconcile(context, trial, runtime, jobs)
            if outcome == 'reconciled':
                state = classify(context, trial, jobs)
            elif outcome == 'waiting':
                state = 'unknown'
            else:
                state = 'attention'
                record_attention(context, trial, runtime, 'submission_ambiguous:%s' % outcome)
        key = str(trial)
        if state == 'suspect':
            first = suspects.get(key, now)
            kept[key] = first
            if now - first >= context['runtime']['suspect_grace_seconds']:
                state = 'attention'
                if mutate:
                    record_attention(context, trial, runtime, 'submitted trial ended without completion or controller marker (cancelled or lost); never revived automatically')
        if state == 'attention' and mutate:
            record_attention(context, trial, runtime, 'attention marker or failed allocation')
        states[key], runtimes[key] = state, runtime
    if mutate:
        C.write_replace(context['campaign'], suspects_path, kept)
    return states, runtimes


def create_trial(context, stage, identifier, config, runtime):
    folder = C.inside(context['campaign'], C.trial_dir(context['campaign'], stage, identifier))
    outcomes = [C.write_exclusive(context['campaign'], folder / 'config.json', config), C.write_exclusive(context['campaign'], folder / 'runtime.json', runtime)]
    if 'created' in outcomes:
        event(context, {'event': 'trial_created', 'trial': str(folder), 'stage': stage})
    return folder


def planned(context, identifier):
    """Config bytes built and validated by the worker's check_config at staging, pinned in the manifest."""
    path = context['campaign'] / 'config' / 'planned' / ('%s.json' % identifier)
    data = path.read_bytes()
    check(C.sha(path) == context['manifest']['planned_configs'][identifier], 'planned config changed: %s' % identifier)
    return data


def qualification_trials(context):
    protocol = context['protocol']
    plan = protocol['qualification']
    folders, created = [], {}
    for number, pair in enumerate(plan['pairs']):
        for role, identifier in (('reference', pair['reference']), ('resumed', pair['resumed'])):
            if identifier not in created:
                resumed = role == 'resumed'
                runtime = C.runtime(protocol, 'Q', identifier, protocol['runtime']['maximum_allocations']['Q'], [len(created)], constraint=plan['constraint'],
                                    forced_pause_after_seconds=pair.get('forced_pause_after_seconds') if resumed else None,
                                    pause_at_step=pair.get('pause_at_step') if resumed else None)
                created[identifier] = create_trial(context, 'Q', identifier, planned(context, identifier), runtime)
            folders.append((role, number, created[identifier]))
    return folders


def screen_trials(context, create=True):
    protocol = context['protocol']
    folders = []
    for number, candidate in enumerate(protocol['initializer_candidates']):
        for learning_rate in protocol['query_learning_rates']:
            for temperature in protocol['temperatures']:
                identifier = C.screen_id(candidate['name'], learning_rate, temperature)
                if create:
                    runtime = C.runtime(protocol, 'S', identifier, protocol['runtime']['maximum_allocations']['S'], [number, learning_rate, temperature])
                    folders.append(create_trial(context, 'S', identifier, planned(context, identifier), runtime))
                else:
                    folders.append(C.trial_dir(context['campaign'], 'S', identifier))
    return folders


def completed_entry(folder, expected_steps):
    training = folder / 'training'
    complete = read_json(training / 'COMPLETE.json')
    check(complete['steps'] == expected_steps, 'trial %s completed %s steps, expected %d' % (folder, complete['steps'], expected_steps))
    identity = read_json(training / 'identity.json')
    check(C.canonical(identity) == complete['identity_sha256'] and identity['config_sha256'] == sha(folder / 'config.json'), 'trial identity differs from its config: %s' % folder)
    return {'id': read_json(folder / 'runtime.json')['trial_id'], 'trial_dir': str(folder), 'training_dir': str(training), 'complete_sha256': sha(training / 'COMPLETE.json')}


def write_index(context, stage, value):
    path = context['campaign'] / 'decisions' / ('%s-index.json' % stage)
    body = dict(value, schema='clmpt-suite-index/1', stage=stage, protocol_sha256=context['protocol_sha256'])
    if C.write_exclusive(context['campaign'], path, body) == 'created':
        event(context, {'event': 'index', 'stage': stage})
    return path


def decision_output(context, target):
    return context['campaign'] / 'decisions' / ('%s.json' % target)


def binding_ok(context, action, target, value, index_path):
    files = context['manifest']['code_files']
    common = value.get('test_access') is False and value.get('index_sha256') == sha(index_path) and value.get('protocol_sha256') == context['protocol_sha256']
    if action == 'qualify':
        return common and value.get('stage') == 'qualification' and value.get('qualifier_sha256') == files['qualify.py']
    agreement = context['campaign'] / 'decisions' / ('%s.crosscheck-agreement.json' % target)
    return (common and value.get('stage') == target and value.get('selector_sha256') == files['select_hpo.py']
            and value.get('worker_protocol_sha256') == context['worker']['protocol_sha256']
            and (target == 'final' or agreement.exists()) and (target != 'recipe' or value.get('calibration_benefit_presumed') is False))


def cpu_decision(context, action, target, index_path, jobs):
    """Decision bound to this index, computed once by a CPU job; None while pending. Never resubmitted automatically."""
    campaign = context['campaign']
    output = decision_output(context, target)
    folder = state_dir(context) / 'cpu' / target
    frozen = output.with_name(target + '.frozen.json')
    if not frozen.exists():
        marker, receipt = folder / 'SUBMITTING.json', folder / 'submission-receipt.json'
        if not marker.exists():
            if output.exists():
                halt(context, 'decision_without_cpu_job', {'target': target})
            if jobs is None:
                return None
            command = C.cpu_command(campaign, action, target)
            C.write_exclusive(campaign, marker, {'command': command, 'index_sha256': sha(index_path), 'time': time.time()})
            result = C.run(command, 30)
            C.write_exclusive(campaign, receipt, dict(result, time=time.time()))
            token = C.job_token(result)
            if not token:
                halt(context, 'cpu_job_submission_ambiguous', {'target': target, 'result': result})
            event(context, {'event': 'cpu_job_submitted', 'action': action, 'target': target, 'job': token})
            return None
        check(receipt.exists(), 'CPU job marker without receipt: %s' % target)
        job = C.job_token(read_json(receipt))
        if jobs is None or job in {value['base'] for value in jobs.values()}:
            return None
        if not output.exists():
            accounting = C.run(['sacct', '-j', job, '-X', '-n', '-P', '-o', 'State,ExitCode,Elapsed'], 60)
            line = accounting['stdout'].strip().splitlines()[0] if accounting['returncode'] == 0 and accounting['stdout'].strip() else ''
            if line.split('|')[0].split(' ')[0] not in TERMINAL_SACCT:
                wait_path = folder / 'accounting-wait.json'
                if not wait_path.exists():
                    C.write_exclusive(campaign, wait_path, {'job': job, 'first_seen': time.time()})
                if time.time() - read_json(wait_path)['first_seen'] < context['runtime']['suspect_grace_seconds']:
                    return None
            halt(context, 'cpu_job_ended_without_decision', {'target': target, 'job': job, 'sacct': line, 'retry': 'never automatic'})
    value = read_json(output)
    if not binding_ok(context, action, target, value, index_path):
        halt(context, 'decision_binding_mismatch', {'target': target, 'decision': str(output)})
    C.write_exclusive(campaign, output.with_name(target + '.frozen.json'), {'decision': str(output), 'decision_sha256': sha(output), 'index': str(index_path),
                                                                         'index_sha256': sha(index_path), 'target': target})
    return value


def qualification_gate(context, states, jobs):
    release = context['release']
    if release.get('waive_qualification'):
        return {'status': 'waived', 'authority': release.get('waiver_authority')}
    trials = qualification_trials(context)
    if not all(states.get(str(folder)) == 'complete' for _, _, folder in trials):
        return {'status': 'qualification_running_or_incomplete'}
    pairs = {}
    for role, number, folder in trials:
        pairs.setdefault(number, {})[role] = completed_entry(folder, context['protocol']['qualification']['steps'])
    index = write_index(context, 'qualification', {'pairs': [pairs[number] for number in sorted(pairs)]})
    decision = cpu_decision(context, 'qualify', 'qualification', index, jobs)
    if decision is None:
        return {'status': 'qualification_comparison_pending'}
    if not decision['passed']:
        halt(context, 'qualification_failed', {'decision': str(decision_output(context, 'qualification'))})
    return {'status': 'passed'}


def continuation_id(screen_identifier):
    check(screen_identifier.startswith('S-'), 'not a screen id: %s' % screen_identifier)
    return 'C-' + screen_identifier[2:]


def build_continuation_configs(context, decision):
    """CPU job only: hpo_continuation configs for the promoted screens, validated by the worker's own check_config."""
    protocol, campaign = context['protocol'], context['campaign']
    screens = {str(folder) for folder in screen_trials(context, create=False)}
    folder = campaign / 'decisions' / 'continuation-configs'
    items, parents = [], {}
    for row in sorted(decision['continue'], key=lambda value: value['id']):
        parent = Path(row['continuation_parent'])
        check(parent.name == 'training' and str(parent.parent) in screens, 'continuation parent outside the screens')
        check(row['maximum_steps'] == protocol['continuation_total_steps'], 'continuation budget differs')
        check(sha(parent / 'COMPLETE.json') == row['parent_complete_sha256'] and read_json(parent / 'progress.json')['checkpoint_sha256'] == row['parent_checkpoint_sha256'],
              'continuation parent changed after selection: %s' % parent)
        identifier = continuation_id(row['id'])
        parents[identifier] = (row['id'], read_json(parent.parent / 'config.json'))
        items.append(C.continuation_item(folder / ('%s.json' % identifier), parents[identifier][1], parent, identifier, protocol['continuation_total_steps']))
    plan = campaign / 'decisions' / 'continuation-plan.json'
    C.write_exclusive(campaign, plan, {'configs': items})
    receipt = campaign / 'decisions' / 'continuation-receipt.json'
    result = C.run(C.worker_configs_command(sys.executable, campaign / 'query' / 'code', plan, receipt), 1800)
    check(result['returncode'] == 0 and receipt.exists(), 'worker make_config refused a continuation config: %s' % result['stderr'][-3000:])
    written = {}
    for identifier, (parent_id, parent_config) in sorted(parents.items()):
        config = read_json(folder / ('%s.json' % identifier))
        differences = C.continuation_differences(parent_config, config)
        check(set(differences) <= set(C.CONTINUATION_KEYS), 'continuation %s changes non-budget settings %s' % (identifier, differences))
        check(config['parent_complete_sha256'] == sha(Path(config['continuation_parent']) / 'COMPLETE.json'), 'continuation parent pin differs: %s' % identifier)
        written[identifier] = {'parent_id': parent_id, 'sha256': sha(folder / ('%s.json' % identifier)), 'differs_from_parent_in': differences}
    C.write_exclusive(campaign, campaign / 'decisions' / 'continuation-configs.json', {'screen_decision_sha256': sha(decision_output(context, 'screen')), 'configs': written,
                                                                                       'receipt_sha256': sha(receipt), 'builder': 'campaign query/code/make_config.build_config + validate_config'})


def continuation_trials(context, decision):
    protocol, campaign, rows = context['protocol'], context['campaign'], []
    summary_path = campaign / 'decisions' / 'continuation-configs.json'
    if not summary_path.exists():
        halt(context, 'continuation_configs_missing', 'the screen selection job did not write validated continuation configs')
    summary = read_json(summary_path)
    check(summary['screen_decision_sha256'] == sha(decision_output(context, 'screen')), 'continuation configs belong to another screen decision')
    check(sorted(summary['configs']) == sorted(continuation_id(row['id']) for row in decision['continue']), 'continuation configs differ from the promoted set')
    for rank, identifier in enumerate(sorted(summary['configs'])):
        path = campaign / 'decisions' / 'continuation-configs' / ('%s.json' % identifier)
        check(sha(path) == summary['configs'][identifier]['sha256'], 'continuation config changed: %s' % identifier)
        runtime = C.runtime(protocol, 'C', identifier, protocol['runtime']['maximum_allocations']['C'], [rank])
        rows.append({'id': identifier, 'parent_id': summary['configs'][identifier]['parent_id'], 'trial_dir': str(create_trial(context, 'C', identifier, path.read_bytes(), runtime))})
    C.write_exclusive(campaign, campaign / 'decisions' / 'C-plan.json', {'schema': 'clmpt-suite-continuation-plan/1', 'screen_decision_sha256': sha(decision_output(context, 'screen')),
                                                                         'continuations': rows})
    return rows


ENGINE_RECORD = 'engine-qualification.json'


def engine_admitted(context):
    """The admitted engine-qualification record, only if it binds this campaign's worker sources and references."""
    path = context['campaign'] / 'decisions' / ENGINE_RECORD
    if not path.exists():
        return None
    record = read_json(path)
    if not (record.get('passed') is True and record.get('worker_source_hashes') == context['worker']['source_hashes']
            and record.get('worker_reference_hashes') == context['worker']['reference_hashes'] and record.get('protocol_sha256') == context['protocol_sha256']):
        halt(context, 'engine_qualification_binding_mismatch', str(path))
    return record


def advance(context, states, jobs):
    campaign, protocol = context['campaign'], context['protocol']
    gate = {'qualification': qualification_gate(context, states, jobs)}
    if gate['qualification']['status'] not in ('passed', 'waived'):
        return dict(gate, gate='qualification')
    engine = engine_admitted(context)
    gate['engine_qualification'] = 'admitted' if engine else 'missing'
    if engine is None:
        return dict(gate, gate='waiting_for_engine_qualification_admission')
    screens = screen_trials(context)
    complete = sum(states.get(str(folder)) == 'complete' for folder in screens)
    gate['screens'] = {'complete': complete, 'total': len(screens)}
    if complete < len(screens):
        return dict(gate, gate='screens_running_or_incomplete')
    entries = [completed_entry(folder, protocol['stages']['S_screen']['steps']) for folder in screens]
    index = write_index(context, 'screen', {'entries': sorted(entries, key=lambda entry: entry['id'])})
    decision = cpu_decision(context, 'select', 'screen', index, jobs)
    if decision is None:
        return dict(gate, gate='screen_selection_pending')
    gate['promoted'] = decision['promoted']
    continuations = continuation_trials(context, decision)
    complete = sum(states.get(row['trial_dir']) == 'complete' for row in continuations)
    gate['continuations'] = {'complete': complete, 'total': len(continuations)}
    if complete < len(continuations):
        return dict(gate, gate='continuations_running_or_incomplete')
    entries = [completed_entry(Path(row['trial_dir']), protocol['continuation_total_steps']) for row in continuations]
    index = write_index(context, 'final', {'entries': sorted(entries, key=lambda entry: entry['id'])})
    final = cpu_decision(context, 'select', 'final', index, jobs)
    if final is None:
        return dict(gate, gate='final_selection_pending')
    gate['winners'] = {view: final['winners'][view]['id'] for view in C.VIEWS}
    selection_decision = campaign / 'decisions' / 'selection-decision.json'
    if not selection_decision.exists():
        return dict(gate, gate='waiting_for_recorded_selection_decision')
    index = write_index(context, 'recipe', {'entries': read_json(campaign / 'decisions' / 'final-index.json')['entries'], 'promotion_sha256': sha(decision_output(context, 'screen')),
                                            'final_sha256': sha(decision_output(context, 'final')), 'decision_sha256': sha(selection_decision)})
    recipe = cpu_decision(context, 'select', 'recipe', index, jobs)
    if recipe is None:
        return dict(gate, gate='recipe_freeze_pending')
    gate['common_leaderboard'] = recipe['common_leaderboard']
    gate['production_recipe'] = recipe['production_recipe']
    C.write_exclusive(campaign, state_dir(context) / 'DONE.json', {'final_decision_sha256': sha(decision_output(context, 'final')), 'recipe_decision_sha256': sha(decision_output(context, 'recipe')),
                                                                   'note': 'HPO complete: all trials complete, decisions and common recipe frozen; production freeze reads decisions/recipe.json production_recipe'})
    event(context, {'event': 'done'})
    return dict(gate, gate='done')


def submit_trial(context, trial, runtime):
    """Scheduler admission test, exclusive marker, one sbatch, receipt; never retried."""
    campaign = context['campaign']
    check(not os.path.lexists(str(campaign / 'OPERATOR_STOP.json')) and not os.path.lexists(str(trial / 'STOP.json')), 'stop marker present')
    check(not (trial / 'INITIAL_SUBMITTING.json').exists() and not (trial / 'submission-receipt.json').exists(), 'trial already submitted: %s' % trial)
    dry = C.run(C.sbatch_command(campaign, trial, 0, runtime, test_only=True), 30)
    C.write_replace(campaign, trial / 'scheduler-readiness.json', dry)
    if dry['returncode'] != 0:
        return {'submitted': False, 'reason': 'sbatch --test-only refused', 'stderr': dry['stderr'][-1000:]}
    command = C.sbatch_command(campaign, trial, 0, runtime)
    C.write_exclusive(campaign, trial / 'INITIAL_SUBMITTING.json', {'time': time.time(), 'manifest_sha256': context['manifest_sha256'], 'command': command})
    result = C.run(command, 30)
    receipt = dict(result, time=time.time())
    C.write_exclusive(campaign, trial / 'submission-receipt.json', receipt)
    token = C.job_token(result)
    return {'submitted': bool(token), 'job': token, 'receipt': receipt}


def resubmit(context, trial, runtime, reason):
    """Next allocation of a clean-paused trial: exclusive marker, one sbatch without dependency, receipt; never retried."""
    campaign = context['campaign']
    check(not os.path.lexists(str(campaign / 'OPERATOR_STOP.json')) and not os.path.lexists(str(trial / 'STOP.json')), 'stop marker present')
    present = waves(trial)
    check(present, 'resubmission needs a previous allocation')
    wave = present[-1][0] + 1
    check(wave < runtime['maximum_allocations'], 'allocation bound exhausted for %s' % runtime['trial_id'])
    receipt_path = resubmission_path(trial, wave)
    marker = receipt_path.with_name(receipt_path.name.replace('.json', '.SUBMITTING.json'))
    check(not marker.exists() and not receipt_path.exists(), 'wave %d already submitted for %s' % (wave, runtime['trial_id']))
    command = C.sbatch_command(campaign, trial, wave, runtime)
    C.write_exclusive(campaign, marker, {'time': time.time(), 'reason': reason, 'command': command, 'manifest_sha256': context['manifest_sha256']})
    result = C.run(command, 30)
    C.write_exclusive(campaign, receipt_path, dict(result, time=time.time(), reason=reason))
    token = C.job_token(result)
    event(context, {'event': 'resubmitted', 'trial': str(trial), 'wave': wave, 'job': token, 'reason': reason})
    return {'submitted': bool(token), 'job': token, 'wave': wave}


def qualification_record(context):
    path = decision_output(context, 'qualification')
    return read_json(path) if (path.with_name('qualification.frozen.json')).exists() else None


def submit_ready(context, states, runtimes, jobs):
    """READY trials (first allocation) and HELD trials (next allocation), within the user job cap and the storage budget."""
    protocol, release = context['protocol'], context['release']
    released = set(release.get('stages', []))
    cap = K.slot_cap(protocol, release)
    used = len(jobs)
    free, free_detail = K.free_bytes(C.run, context['campaign'], release)
    sizes = K.sizes(protocol, qualification_record(context))
    reserve = max(protocol['storage']['reserve_bytes'], int(release.get('reserve_bytes', 0)))
    owed = sum(K.owed_bytes(protocol, sizes, key, read_json(Path(key) / 'config.json')) for key, state in states.items() if state in ACTIVE_STATES + ('held', 'stopped'))
    stage_order = {stage: number for number, stage in enumerate(C.STAGES)}
    candidates = sorted((key for key, state in states.items() if state in ('ready', 'held')),
                        key=lambda key: (stage_order[runtimes[key]['stage']], json.dumps(runtimes[key]['order_key']), runtimes[key]['trial_id']))
    submitted, skipped = [], {}
    budget = {'cap': cap, 'user_jobs_before': used, 'free_bytes': free, 'free_detail': free_detail, 'reserve_bytes': reserve, 'owed_bytes': owed, 'sizes': sizes}
    for key in candidates:
        runtime, trial = runtimes[key], Path(key)
        if runtime['stage'] not in released:
            skipped.setdefault('stage_not_released', []).append(runtime['trial_id'])
            continue
        if stop_marker(context, trial):
            skipped.setdefault('stop_marker', []).append(runtime['trial_id'])
            continue
        if used + len(submitted) >= cap:
            skipped.setdefault('user_job_cap_%d' % cap, []).append(runtime['trial_id'])
            continue
        if states[key] == 'held':
            outcome = resubmit(context, trial, runtime, 'held clean pause released')
            if not outcome['submitted']:
                halt(context, 'resubmission_ambiguous', {'trial': key, 'outcome': outcome})
            submitted.append({'trial': runtime['trial_id'], 'job': outcome['job'], 'wave': outcome['wave']})
            continue
        need = K.peak_bytes(protocol, sizes, read_json(trial / 'config.json'))
        if free is None or free - reserve - owed < need:
            skipped.setdefault('storage', []).append({'trial': runtime['trial_id'], 'need': need, 'free': free, 'reserve': reserve, 'owed': owed})
            continue
        outcome = submit_trial(context, trial, runtime)
        refusals_path = state_dir(context) / 'submit-refusals.json'
        if 'receipt' not in outcome:
            refusals = read_json(refusals_path) if refusals_path.exists() else {'consecutive': 0, 'last': []}
            refusals = {'consecutive': refusals['consecutive'] + 1, 'last': (refusals['last'] + [{'trial': key, 'reason': outcome['reason'], 'stderr': outcome['stderr'], 'time': time.time()}])[-5:]}
            C.write_replace(context['campaign'], refusals_path, refusals)
            if refusals['consecutive'] >= 3:
                halt(context, 'submission_refused_before_sbatch_3_times', refusals)
            skipped.setdefault('refused_before_marker', []).append(runtime['trial_id'])
            break
        C.write_replace(context['campaign'], refusals_path, {'consecutive': 0, 'last': []})
        if not outcome['submitted']:
            event(context, {'event': 'submission_ambiguous', 'trial': key, 'next': 'next tick reconciles by unique job name; never resubmitted'})
            skipped.setdefault('submission_ambiguous', []).append(runtime['trial_id'])
            break
        owed += need
        event(context, {'event': 'submitted', 'trial': key, 'stage': runtime['stage'], 'job': outcome['job']})
        submitted.append({'trial': runtime['trial_id'], 'job': outcome['job']})
    budget['storage_stall'] = bool(skipped.get('storage')) and not submitted and not any(state in ACTIVE_STATES for state in states.values())
    return submitted, skipped, budget


def counts(states, runtimes):
    value = {}
    for key, state in states.items():
        stage = runtimes[key]['stage'] if key in runtimes else 'malformed'
        value.setdefault(stage, {}).setdefault(state, 0)
        value[stage][state] += 1
    return value


def unacknowledged_attention(context):
    folder = state_dir(context) / 'attention'
    records = sorted(folder.glob('*.json')) if folder.is_dir() else []
    return [path.stem for path in records if not path.name.endswith('.ack.json') and not path.with_name(path.stem + '.ack.json').exists()]


def tick(argument):
    context = load(argument)
    campaign, folder = context['campaign'], state_dir(context)
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
            jobs = K.user_jobs(C.run)
            summary['scheduler_available'] = jobs is not None
            states, runtimes = classify_all(context, jobs, time.time())
            if any(state == 'malformed' for state in states.values()):
                halt(context, 'malformed_trial_directory', sorted(key for key, state in states.items() if state == 'malformed'))
            summary['unacknowledged_trial_attention'] = attention = unacknowledged_attention(context)
            if len(attention) >= context['runtime']['attention_halt_threshold']:
                halt(context, 'trial_attention_threshold', attention)
            summary['gate'] = advance(context, states, jobs)
            states, runtimes = classify_all(context, jobs, time.time(), mutate=False)
            if jobs is None:
                summary['submitted'], summary['skipped'] = [], {'scheduler_unavailable': True}
            else:
                summary['submitted'], summary['skipped'], summary['budget'] = submit_ready(context, states, runtimes, jobs)
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


def prune_trial(context, trial, reason, decision_path):
    """Delete only full-state checkpoints of one completed trial after verifying every retained pointer; per-file receipt first."""
    campaign = context['campaign']
    trial = C.inside(campaign, Path(trial))
    training = trial / 'training'
    runtime = read_json(trial / 'runtime.json')
    complete, progress = read_json(training / 'COMPLETE.json'), read_json(training / 'progress.json')
    receipt_path = state_dir(context) / 'pruned' / (runtime['trial_id'] + '.json')
    if receipt_path.exists():
        receipt = read_json(receipt_path)
    else:
        for pointer in [progress['initial_checkpoint']] + list(progress['retained_best'].values()):
            check(sha(Path(pointer['root']) / pointer['path']) == pointer['sha256'], 'retained model-only export missing or changed: %s' % pointer)
        for name in complete['files']:
            if name.startswith('fits/'):
                check((training / name).is_file(), 'saved scores missing before pruning: %s' % name)
        removable = []
        for path in sorted((training / 'checkpoints').glob('*.pt')):
            relative = str(path.relative_to(training))
            expected = complete['files'].get(relative) or progress['artifacts'].get(relative)
            check(expected is not None and sha(path) == expected, 'full state without a matching receipt hash; not pruned: %s' % path)
            removable.append({'path': relative, 'sha256': expected, 'bytes': path.stat().st_size})
        receipt = {'trial_id': runtime['trial_id'], 'training_dir': str(training), 'reason': reason, 'decision': str(decision_path), 'decision_sha256': sha(decision_path),
                   'removed': removable, 'kept': 'model-only exports (initial, raw-best, calibrated-best), fits, curve, logs, COMPLETE and progress records', 'time': time.time()}
        C.write_exclusive(campaign, receipt_path, receipt)
    if receipt['removed']:
        C.write_exclusive(campaign, training / 'PRUNED.json', {'identity_sha256': complete['identity_sha256'], 'receipt_sha256': sha(receipt_path),
                                                              'files': {row['path']: row['sha256'] for row in receipt['removed']}, 'receipt': str(receipt_path)})
    for row in receipt['removed']:
        path = training / row['path']
        if path.exists():
            check(sha(path) == row['sha256'], 'full state changed after the pruning receipt: %s' % path)
            path.unlink()
    C.write_exclusive(campaign, receipt_path.with_name(runtime['trial_id'] + '.done.json'), {'receipt_sha256': sha(receipt_path), 'freed_bytes': sum(row['bytes'] for row in receipt['removed'])})
    event(context, {'event': 'pruned', 'trial': runtime['trial_id'], 'files': len(receipt['removed'])})


def prune_after(context, target, decision_path):
    decision = read_json(decision_path)
    campaign = context['campaign']
    if target == 'qualification':
        if decision['passed']:
            for pair in read_json(campaign / 'decisions' / 'qualification-index.json')['pairs']:
                for role in ('reference', 'resumed'):
                    prune_trial(context, pair[role]['trial_dir'], 'qualification passed; terminal states compared', decision_path)
    elif target == 'screen':
        for identifier in decision['not_promoted']:
            prune_trial(context, C.trial_dir(campaign, 'S', identifier), 'screen not promoted', decision_path)
    elif target == 'final':
        keep = {decision['winners'][view]['id'] for view in C.VIEWS}
        for entry in read_json(campaign / 'decisions' / 'final-index.json')['entries']:
            if entry['id'] not in keep:
                prune_trial(context, entry['trial_dir'], 'continuation is not a per-view winner', decision_path)
        for entry in read_json(campaign / 'decisions' / 'screen-index.json')['entries']:
            if entry['id'] in read_json(campaign / 'decisions' / 'screen.json')['promoted']:
                prune_trial(context, entry['trial_dir'], 'promoted screen whose terminal state its completed continuation imported', decision_path)


def cpu_action(argument, action, target):
    context = load(argument)
    check(target in CPU_TARGETS[action], 'target %s not admitted for %s' % (target, action))
    check(os.environ.get('SLURM_JOB_ID', '').isdigit() and not os.environ.get('CUDA_VISIBLE_DEVICES'), '%s runs only inside its CPU Slurm job' % action)
    campaign = context['campaign']
    index = campaign / 'decisions' / ('%s-index.json' % target)
    output = decision_output(context, target)
    check(index.exists() and not output.exists(), '%s: index missing or decision already present' % target)
    check(read_json(state_dir(context) / 'cpu' / target / 'SUBMITTING.json')['index_sha256'] == sha(index), 'index changed after the CPU job was submitted')
    if action == 'qualify':
        command = [sys.executable, '-I', str(HERE / 'qualify.py'), '--protocol', context['protocol_path'], '--index', str(index), '--output', str(output)]
    else:
        command = [sys.executable, '-I', str(HERE / 'select_hpo.py'), '--stage', target, '--protocol', context['protocol_path'], '--worker-protocol', context['worker_protocol_path'],
                   '--index', str(index), '--output', str(output)]
        if target == 'recipe':
            command += ['--final', str(decision_output(context, 'final')), '--decision', str(campaign / 'decisions' / 'selection-decision.json')]
    result = C.run(command, 3000)
    C.write_exclusive(campaign, campaign / 'decisions' / ('%s.run.json' % target), dict(result, job=os.environ['SLURM_JOB_ID'], time=time.time()))
    if result['returncode'] != 0 or not output.exists():
        return {'status': 'refused', 'returncode': result['returncode'], 'stderr': result['stderr'][-2000:]}
    if action == 'select' and target in ('screen', 'recipe'):
        agreement = cross_check(context, target, index, read_json(output))
        if not agreement:
            return {'status': 'refused', 'reason': 'selector cross-check disagreement; see decisions/%s.crosscheck-mismatch.json' % target}
    check(binding_ok(context, action, target, read_json(output), index), 'decision binding mismatch')
    if action == 'select' and target == 'screen':
        build_continuation_configs(context, read_json(output))
    if target != 'recipe':
        prune_after(context, target, output)
    return {'status': 'ok', 'target': target}


def cross_check(context, target, index, mine):
    """Run the query worker's leaderboard.py on the same index and require identical promotions, orders, winners and recipe."""
    campaign = context['campaign']
    theirs_path = campaign / 'decisions' / ('%s.crosscheck.json' % target)
    command = [sys.executable, '-I', str(campaign / 'query' / 'code' / 'leaderboard.py'), 'promote' if target == 'screen' else 'freeze',
               '--protocol', context['worker_protocol_path'], '--index', str(index), '--output', str(theirs_path)]
    if target == 'recipe':
        command += ['--decision', str(campaign / 'decisions' / 'selection-decision.json')]
    result = C.run(command, 3000)
    C.write_exclusive(campaign, campaign / 'decisions' / ('%s.crosscheck-run.json' % target), dict(result, time=time.time()))
    differences = []
    if result['returncode'] != 0 or not theirs_path.exists():
        differences.append({'leaderboard_refused': result['stderr'][-2000:]})
    else:
        theirs = read_json(theirs_path)
        boards = mine['leaderboards'] if target == 'screen' else read_json(decision_output(context, 'final'))['leaderboards']
        for view in C.VIEWS:
            mine_order = [row['id'] for row in boards[view]]
            if mine_order != [row['id'] for row in theirs['leaderboards'][view]]:
                differences.append({'view': view, 'order_mine': mine_order, 'order_theirs': [row['id'] for row in theirs['leaderboards'][view]]})
        if target == 'screen' and sorted(mine['promoted']) != sorted(theirs['promoted_ids']):
            differences.append({'promoted_mine': mine['promoted'], 'promoted_theirs': theirs['promoted_ids']})
        if target == 'recipe':
            final = read_json(decision_output(context, 'final'))
            for view in C.VIEWS:
                left, right = final['winners'][view], theirs['separately_selected_winners'][view]
                if (left['id'], left['step'], left['recipe']) != (right['id'], right['step'], right['recipe']):
                    differences.append({'winner': view, 'mine': [left['id'], left['step']], 'theirs': [right['id'], right['step']]})
            if mine['production_recipe'] != theirs['production_recipe']:
                differences.append({'production_recipe_mine': mine['production_recipe'], 'production_recipe_theirs': theirs['production_recipe']})
    record = {'target': target, 'index_sha256': sha(index), 'leaderboard_sha256': sha(campaign / 'query' / 'code' / 'leaderboard.py'), 'differences': differences,
              'crosscheck_sha256': sha(theirs_path) if theirs_path.exists() else None}
    C.write_exclusive(campaign, campaign / 'decisions' / ('%s.crosscheck-%s.json' % (target, 'mismatch' if differences else 'agreement')), record)
    return not differences


def release(argument, authority, approve, waive_qualification, user_active_job_cap, free_bytes_command, reserve_bytes):
    context = load(argument, require_release=False)
    campaign = context['campaign']
    check(approve and authority.strip(), 'release requires --approve and a non-empty --authority')
    check(not context['release_present'], 'ROOT_RELEASE.json exists: release is write-once')
    check(context['manifest']['git'].get('package_status_clean') is True, 'campaign was staged without the Git check (local rehearsal); it can never be released')
    probe_runtime = {'job_name': C.job_name('release-probe'), 'constraint': context['protocol']['runtime']['gpu_constraint']}
    probes = {'gpu_job': C.run(C.sbatch_command(campaign, campaign / 'trials', 0, probe_runtime, test_only=True), 30),
              'cpu_job': C.run(C.cpu_command(campaign, 'select', 'screen', test_only=True), 30)}
    C.write_replace(campaign, state_dir(context) / 'release-probes.json', probes)
    check(all(probe['returncode'] == 0 for probe in probes.values()), 'scheduler refused a job script: %s' % {key: value['stderr'][-500:] for key, value in probes.items()})
    value = {'approved': True, 'authority': authority, 'manifest_sha256': context['manifest_sha256'], 'protocol_sha256': context['protocol_sha256'],
             'code_revision': context['manifest']['code_revision'], 'stages': list(C.STAGES), 'probes_sha256': sha(state_dir(context) / 'release-probes.json'),
             'waive_qualification': bool(waive_qualification), 'waiver_authority': authority if waive_qualification else None, 'time': time.time()}
    if user_active_job_cap is not None:
        value['user_active_job_cap'] = int(user_active_job_cap)
    if free_bytes_command:
        command = json.loads(free_bytes_command)
        check(isinstance(command, list) and all(isinstance(part, str) for part in command), '--free-bytes-command must be a JSON list of strings')
        value['free_bytes_command'] = command
    if reserve_bytes is not None:
        value['reserve_bytes'] = int(reserve_bytes)
    C.write_exclusive(campaign, campaign / 'ROOT_RELEASE.json', value)
    return value


def record_decision(argument, decision_file, authority):
    context = load(argument, require_release=False)
    campaign = context['campaign']
    check(authority.strip(), 'record-decision requires a non-empty --authority')
    raw = Path(decision_file).read_bytes()
    decision = json.loads(raw)
    check(decision.get('common_leaderboard') in C.VIEWS, 'decision must name common_leaderboard raw or calibrated')
    check(decision.get('rule') == context['protocol']['selection']['decision_rule_names'], 'decision rule differs from the frozen selection rule')
    check(decision.get('raw_calibration_comparison_required') is True, 'decision must require the same-checkpoint raw/calibrated comparison')
    target = campaign / 'decisions' / 'selection-decision.json'
    C.write_exclusive(campaign, target, raw)
    receipt = {'source': str(Path(decision_file).resolve()), 'sha256': sha(target), 'authority': authority, 'recorded_at': time.time(),
               'recorded_before_final_selection': not decision_output(context, 'final').exists(), 'common_leaderboard': decision['common_leaderboard']}
    C.write_exclusive(campaign, campaign / 'decisions' / 'selection-decision.receipt.json', receipt)
    event(context, {'event': 'selection_decision_recorded', 'common_leaderboard': decision['common_leaderboard']})
    return receipt


def option_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def admit_engine_qualification(argument, receipts_dir, worker_code, authority):
    """Verify the query worker's qualifier receipts and record them once; screens require this record.

    construction and scores receipts carry no source hashes, so they are bound through worker_code: the directory that
    produced them must hold exactly this campaign's worker sources and qualify.py. Resume receipts are also bound through
    the configs they ran, whose pinned source and reference hashes must equal this campaign's.
    """
    context = load(argument, require_release=False)
    campaign = context['campaign']
    check(authority.strip(), 'admission requires --authority')
    worker_code = Path(worker_code).resolve()
    receipts, found, problems = engine_receipts(context['protocol']['engine_qualification'], context['worker'], context['manifest']['files']['query/code/qualify.py'], receipts_dir, worker_code)
    check(not problems, 'engine qualification not admitted: %s' % problems)
    record = {'passed': True, 'authority': authority, 'time': time.time(), 'receipts_dir': str(Path(receipts_dir).resolve()), 'worker_code': str(worker_code),
              'worker_source_hashes': context['worker']['source_hashes'], 'worker_reference_hashes': context['worker']['reference_hashes'],
              'qualify_sha256': sha(worker_code / 'qualify.py'), 'protocol_sha256': context['protocol_sha256'],
              'receipt_sha256': {path: sha(path) for path in sorted(receipts)}, 'found': found,
              'scope': 'construction and score parity on CUDA, exact pause/resume for five conditions plus seed 1 and for pretraining; same device per receipt, no cross-GPU claim'}
    C.write_exclusive(campaign, campaign / 'decisions' / ENGINE_RECORD, record)
    event(context, {'event': 'engine_qualification_admitted', 'receipts': len(receipts)})
    return {'status': 'ok', 'admitted': True, 'found': {key: len(value) for key, value in found.items()}}


def engine_receipts(spec, worker, qualify_sha256, receipts_dir, worker_code):
    """Pure check of qualifier receipts against the campaign worker; returns (receipts, found, problems)."""
    problems = []
    for name, digest in worker['source_hashes'].items():
        if not (worker_code / name).is_file() or sha(worker_code / name) != digest:
            problems.append('receipt-producing worker source differs from the campaign: %s' % name)
    if not (worker_code / 'qualify.py').is_file() or sha(worker_code / 'qualify.py') != qualify_sha256:
        problems.append('receipt-producing qualify.py differs from the campaign copy')
    receipts = {}
    for path in sorted(Path(receipts_dir).glob('*.json')):
        try:
            value = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('subcommand') in ('construction', 'scores', 'resume'):
            receipts[str(path)] = value
    found = {'construction': [], 'scores': [], 'query_resume': [], 'pretrain_resume': []}
    for path, value in receipts.items():
        if not value.get('passed') is True:
            problems.append('receipt did not pass: %s' % path)
            continue
        if value['subcommand'] in ('construction', 'scores'):
            if value.get('device') == spec[value['subcommand']]['device']:
                found[value['subcommand']].append(path)
            else:
                problems.append('receipt not on %s: %s' % (spec[value['subcommand']]['device'], path))
            continue
        command = value['calls'][0]['command']
        if not Path(option_value(command, '--config') or '').is_absolute() or not Path(option_value(command, '--config')).is_file():
            problems.append('resume receipt config path is not an existing absolute path (run qualify.py resume with an absolute --config): %s' % path)
            continue
        config = read_json(option_value(command, '--config'))
        script = Path(command[0]).resolve()
        if value['engine'] == 'query':
            if script.parent != worker_code or config['source_hashes'] != worker['source_hashes'] or config['reference_hashes'] != worker['reference_hashes']:
                problems.append('resume ran other worker sources: %s' % path)
            elif config['purpose'] != spec['resume_purpose']:
                problems.append('resume config purpose is not qualification: %s' % path)
            else:
                found['query_resume'].append({'receipt': path, 'encoder': config['encoder'], 'curriculum': config['curriculum'], 'seed': config['seed'],
                                              'sampler_seed': config['sampler_seed'], 'config_sha256': sha(option_value(command, '--config'))})
        elif value['engine'] == 'pretrain':
            found['pretrain_resume'].append({'receipt': path, 'seed': config.get('seed'), 'worker': str(script), 'worker_sha256': sha(script),
                                             'config_sha256': sha(option_value(command, '--config'))})
    covered = {(row['encoder'], row['curriculum'], row['seed']) for row in found['query_resume']}
    missing = [row for row in spec['query_resumes'] if (row['encoder'], row['curriculum'], row['seed']) not in covered]
    if missing:
        problems.append('query resume conditions without a passed receipt: %s' % missing)
    for name in ('construction', 'scores'):
        if not found[name]:
            problems.append('no passed %s receipt on %s' % (name, spec[name]['device']))
    if not any(row['seed'] == spec['pretrain_resume']['seed'] for row in found['pretrain_resume']):
        problems.append('no passed pretraining resume receipt with seed %s' % spec['pretrain_resume']['seed'])
    return receipts, found, problems


def find_trial(context, identifier):
    for stage in C.STAGES:
        folder = C.trial_dir(context['campaign'], stage, identifier)
        if (folder / 'runtime.json').exists():
            return folder
    raise Refusal('unknown trial %s' % identifier)


def resume(argument, identifier, authority, after_attention):
    context = load(argument)
    check(authority.strip(), 'resume requires --authority')
    check(not (state_dir(context) / 'HALT.json').exists(), 'campaign halted')
    trial = find_trial(context, identifier)
    runtime = read_json(trial / 'runtime.json')
    jobs = K.user_jobs(C.run)
    check(jobs is not None, 'scheduler unavailable')
    state = classify(context, trial, jobs)
    present = waves(trial)
    if after_attention:
        check(state == 'attention' and present, 'trial is not in attention with an allocation')
        check((state_dir(context) / 'attention' / (identifier + '.ack.json')).exists(), 'acknowledge the attention record first')
        progress = read_json(trial / 'training' / 'progress.json')
        check(sha(trial / 'training' / progress['checkpoint']) == progress['checkpoint_sha256'], 'committed state does not verify; not resumable')
        C.write_exclusive(context['campaign'], present[-1][1] / 'RESUME_AUTHORIZED.json', {'authority': authority, 'time': time.time(), 'step': progress['step']})
    else:
        check(state in ('stopped', 'held'), 'trial is %s, not stopped or held' % state)
    return resubmit(context, trial, runtime, 'manual resume: %s' % authority)


def stop(argument, identifier, everything, authority):
    context = load(argument)
    check(authority.strip(), 'stop requires --authority')
    jobs = K.user_jobs(C.run) or {}
    if everything:
        C.write_exclusive(context['campaign'], context['campaign'] / 'OPERATOR_STOP.json', {'authority': authority, 'time': time.time()})
        trials = discover(context)
    else:
        trial = find_trial(context, identifier)
        C.write_exclusive(context['campaign'], trial / 'STOP.json', {'authority': authority, 'time': time.time()})
        trials = [trial]
    signalled = []
    for trial in trials:
        for job in sorted(recorded_jobs(trial)):
            states = {value['state'] for value in jobs.values() if value['base'] == job}
            if 'RUNNING' in states:
                signalled.append({'job': job, 'result': C.run(['scancel', '--batch', '--signal=TERM', job], 30)})
            elif states:
                signalled.append({'job': job, 'pending_cancelled': C.run(['scancel', job], 30)})
    event(context, {'event': 'stop', 'all': everything, 'trial': identifier, 'signalled': signalled})
    return {'status': 'ok', 'signalled': signalled}


def ack(argument, identifier, authority):
    context = load(argument)
    path = state_dir(context) / 'attention' / (identifier + '.json')
    check(path.exists() and authority.strip(), 'no attention record for %s or empty authority' % identifier)
    C.write_exclusive(context['campaign'], path.with_name(identifier + '.ack.json'), {'authority': authority, 'time': time.time(), 'record_sha256': sha(path)})
    return {'status': 'ok', 'acknowledged': identifier}


def gates(context):
    decisions = context['campaign'] / 'decisions'
    value = {}
    for target in ('qualification', 'screen', 'final'):
        frozen = decisions / ('%s.frozen.json' % target)
        record = read_json(decisions / ('%s.json' % target)) if frozen.exists() else None
        value[target] = {'frozen': frozen.exists()}
        if record and target == 'qualification':
            value[target].update({'passed': record['passed'], 'measured': record['measured']})
        if record and target == 'screen':
            value[target].update({'promoted': record['promoted'], 'promoted_by_view': record['promoted_by_view']})
        if record and target == 'final':
            value[target].update({'winners': {view: record['winners'][view]['id'] for view in C.VIEWS}, 'comparison': record['separately_selected_comparison']})
    value['engine_qualification_admitted'] = (decisions / ENGINE_RECORD).exists()
    value['selection_decision_recorded'] = (decisions / 'selection-decision.json').exists()
    if value['selection_decision_recorded']:
        value['selection_decision'] = read_json(decisions / 'selection-decision.receipt.json')
    recipe = decisions / 'recipe.frozen.json'
    record = read_json(decisions / 'recipe.json') if recipe.exists() else None
    value['recipe'] = {'frozen': recipe.exists(), 'production_recipe': record['production_recipe'] if record else None,
                       'other_view_recipe_differs': record['other_view_recipe_differs'] if record else None}
    value['production_freeze_ready'] = recipe.exists()
    for target in ('screen', 'recipe'):
        value['%s_crosscheck' % target] = ('agreement' if (decisions / ('%s.crosscheck-agreement.json' % target)).exists()
                                           else 'mismatch' if (decisions / ('%s.crosscheck-mismatch.json' % target)).exists() else None)
    return value


def status(argument, gate_only=False):
    """Read-only: no lock and no writes; only squeue and df are called."""
    context = load(argument, require_release=False)
    folder = state_dir(context)
    result = {'gates': gates(context), 'halt': read_json(folder / 'HALT.json') if (folder / 'HALT.json').exists() else None,
              'operator_stop': os.path.lexists(str(context['campaign'] / 'OPERATOR_STOP.json')), 'done': (folder / 'DONE.json').exists(),
              'root_release': {'present': context['release_present'], 'approves_current_manifest': context['release_approved'], 'sha256': context['release_sha256']}}
    if gate_only:
        return result
    jobs = K.user_jobs(C.run)
    states, runtimes = classify_all(context, jobs, time.time(), mutate=False)
    result.update({'trial_states': counts(states, runtimes), 'trials': {runtimes[key]['trial_id']: state for key, state in states.items() if key in runtimes},
                   'unacknowledged_trial_attention': unacknowledged_attention(context), 'user_jobs': None if jobs is None else len(jobs),
                   'user_job_cap': K.slot_cap(context['protocol'], context['release']), 'free_bytes': K.free_bytes(C.run, context['campaign'], context['release']),
                   'pruned': sorted(path.name for path in (folder / 'pruned').glob('*.done.json')) if (folder / 'pruned').is_dir() else [],
                   'last_tick': read_json(folder / 'last-tick.json') if (folder / 'last-tick.json').exists() else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('tick', 'status', 'gate', 'release', 'record-decision', 'admit-engine-qualification', 'select', 'qualify', 'resume', 'stop', 'ack'):
        sub = commands.add_parser(name)
        sub.add_argument('--campaign', required=True)
        if name in ('select', 'qualify'):
            sub.add_argument('--target', required=True)
        if name in ('release', 'record-decision', 'admit-engine-qualification', 'resume', 'stop', 'ack'):
            sub.add_argument('--authority', required=True)
        if name == 'admit-engine-qualification':
            sub.add_argument('--receipts-dir', required=True)
            sub.add_argument('--worker-code', required=True, help='the query/code directory that produced the receipts')
        if name == 'release':
            sub.add_argument('--approve', action='store_true')
            sub.add_argument('--waive-qualification', action='store_true')
            sub.add_argument('--user-active-job-cap', type=int)
            sub.add_argument('--free-bytes-command', help='JSON list; prints available bytes as an integer on its last line')
            sub.add_argument('--reserve-bytes', type=int)
        if name == 'record-decision':
            sub.add_argument('--decision-file', required=True)
        if name in ('resume', 'ack'):
            sub.add_argument('--trial', required=True)
        if name == 'resume':
            sub.add_argument('--after-attention', action='store_true')
        if name == 'stop':
            group = sub.add_mutually_exclusive_group(required=True)
            group.add_argument('--trial')
            group.add_argument('--all', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'tick':
            result = tick(args.campaign)
        elif args.command in ('status', 'gate'):
            result = status(args.campaign, gate_only=args.command == 'gate')
        elif args.command == 'release':
            result = release(args.campaign, args.authority, args.approve, args.waive_qualification, args.user_active_job_cap, args.free_bytes_command, args.reserve_bytes)
        elif args.command == 'record-decision':
            result = record_decision(args.campaign, args.decision_file, args.authority)
        elif args.command == 'admit-engine-qualification':
            result = admit_engine_qualification(args.campaign, args.receipts_dir, args.worker_code, args.authority)
        elif args.command in ('select', 'qualify'):
            result = cpu_action(args.campaign, args.command, args.target)
        elif args.command == 'resume':
            result = resume(args.campaign, args.trial, args.authority, args.after_attention)
        elif args.command == 'stop':
            result = stop(args.campaign, args.trial, args.all, args.authority)
        else:
            result = ack(args.campaign, args.trial, args.authority)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get('status', 'ok') in ('ok', 'done', 'busy', 'operator_stop') else 1
    except Refusal as error:
        print(json.dumps({'refused': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
