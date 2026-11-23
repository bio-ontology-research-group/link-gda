"""CPU rehearsal of the in-job grace read and the qualification deadline-safety rule (standard library only).

Exercises suite.read_overtime / deadline_safety, the HPO root controller deadlines and the production_ops controller
deadlines with a fake `scontrol` on PATH: a validated 60-minute grace; a read that fails every bounded attempt (stderr
audited, requested-time fallback); a transient failure followed by success; a partial 20-minute grace (fallback); and
the safety rule refusing an unvalidated 3:45 exit, a finish after the recorded exit deadline, a missing completion
record and a save deadline too close to exit. It also replays the old campaign's five recorded Q allocations (values
copied from its allocation.json files) through the new rule; this is information only and imports no old decision.
Writes one JSON receipt; touches nothing outside --scratch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE = '''#!/bin/sh
state=%(state)s
count=$(cat "$state" 2>/dev/null || echo 0); count=$((count + 1)); echo $count > "$state"
mode=$(cat %(mode)s)
case "$mode" in
  ok) echo "OverTimeLimit           = 60 min"; exit 0 ;;
  twenty) echo "OverTimeLimit           = 20 min"; exit 0 ;;
  fail) echo "scontrol: error: slurm_load_ctl_conf error: Unable to contact slurm controller" >&2; exit 1 ;;
  flaky) if [ "$count" -le 2 ]; then echo "scontrol: error: Socket timed out on send/recv operation" >&2; exit 1; fi; echo "OverTimeLimit           = 60 min"; exit 0 ;;
esac
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scratch = Path(args.scratch).resolve()
    assert not scratch.exists() or not any(scratch.iterdir())
    (scratch / 'bin').mkdir(parents=True)
    mode, counter = scratch / 'mode', scratch / 'count'
    (scratch / 'bin' / 'scontrol').write_text(FAKE % {'state': counter, 'mode': mode})
    os.chmod(str(scratch / 'bin' / 'scontrol'), 0o755)
    os.environ['PATH'] = str(scratch / 'bin') + os.pathsep + os.environ['PATH']
    suite = load('grace_suite', SUITE / 'suite.py')
    root = load('grace_root_controller', SUITE / 'controller.py')
    sys.path.insert(0, str(SUITE / 'production_ops'))
    production = load('grace_production_controller', SUITE / 'production_ops' / 'controller.py')
    hpo = json.loads((SUITE / 'hpo-protocol.json').read_text())['runtime']
    ibex = json.loads((SUITE / 'production_ops' / 'site-policy.json').read_text())['ibex']
    slept = []

    def scenario(name):
        mode.write_text(name)
        counter.write_text('0')
        overtime, record = suite.read_overtime(suite.run, sleep=slept.append)
        return overtime, record

    results, checks = {}, {}
    overtime, record = scenario('ok')
    results['validated_60m'] = {'overtime': overtime, 'record': record, 'root': root.deadlines(hpo, 10800, overtime)}
    counter.write_text('0')
    results['validated_60m']['production'] = production.deadlines('ibex', ibex, 10800, delays=(0,))[:2]
    checks['validated_60m_gives_3h40_save_3h45_exit'] = (record['read'] and overtime == 3600 and tuple(results['validated_60m']['root']) == (13200, 13500, True)
                                                         and tuple(results['validated_60m']['production']) == (13200, 13500))
    slept.clear()
    overtime, record = scenario('fail')
    results['failed_read'] = {'overtime': overtime, 'record': record, 'root': root.deadlines(hpo, 10800, overtime), 'sleeps': list(slept)}
    mode.write_text('fail')
    counter.write_text('0')
    production_failed = production.deadlines('ibex', ibex, 10800, delays=(0, 0))
    checks['failed_read_audited_and_falls_back_inside_request'] = (not record['read'] and len(record['attempts']) == len(suite.GRACE_READ_DELAYS)
                                                                   and all('Unable to contact' in attempt['stderr'] for attempt in record['attempts'])
                                                                   and slept == [delay for delay in suite.GRACE_READ_DELAYS if delay]
                                                                   and tuple(results['failed_read']['root']) == (10200, 10500, False)
                                                                   and tuple(production_failed[:2]) == (10200, 10500) and production_failed[2]['basis'] == 'requested_time_fallback'
                                                                   and len(production_failed[2]['attempts']) == 2)
    slept.clear()
    overtime, record = scenario('flaky')
    results['transient_then_ok'] = {'overtime': overtime, 'attempts': len(record['attempts'])}
    checks['transient_failure_recovered_by_retry'] = record['read'] and len(record['attempts']) == 3 and tuple(root.deadlines(hpo, 10800, overtime)) == (13200, 13500, True)
    overtime, record = scenario('twenty')
    checks['partial_20m_grace_not_used'] = tuple(root.deadlines(hpo, 10800, overtime)) == (10200, 10500, False)
    start = 1_800_000_000.0

    def allocation(validated, exit_after, save_after=None, overtime=3600):
        return {'allocation_start': start, 'time_limit_seconds': 10800, 'overtime_validated': validated, 'overtime_seconds': overtime if validated else 0,
                'save_deadline': start + (save_after if save_after is not None else exit_after - 300), 'exit_deadline': start + exit_after}
    cases = {
        'validated_grace_finished_in_time': (allocation(True, 13500), {'completed_at': start + 13300}, True),
        'fallback_finished_in_time': (allocation(False, 10500), {'completed_at': start + 10400}, True),
        'unvalidated_3h45_exit_refused': (allocation(False, 13500), {'completed_at': start + 11000}, False),
        'finished_after_exit_deadline_refused': (allocation(True, 13500), {'completed_at': start + 13505}, False),
        'missing_completion_record_refused': (allocation(True, 13500), None, False),
        'save_too_close_to_exit_refused': (allocation(False, 10500, save_after=10400), {'completed_at': start + 1000}, False),
        'validated_but_exit_beyond_cap_refused': (allocation(True, 13800), {'completed_at': start + 1000}, False),
    }
    results['safety_cases'] = {}
    for name, (value, terminal, expected) in cases.items():
        verdict = suite.deadline_safety(value, terminal, hpo)
        results['safety_cases'][name] = verdict
        checks['rule_' + name] = verdict['safe'] is expected
    old = {
        'Q-ref wave-000': ({'allocation_start': 1795437922.0, 'time_limit_seconds': 10800, 'overtime_validated': True, 'overtime_seconds': 3600, 'save_deadline': 1795451122.0,
                            'exit_deadline': 1795451422.0}, {'completed_at': 1795440163.4233727}),
        'Q-step-pause wave-000': ({'allocation_start': 1795437922.0, 'time_limit_seconds': 10800, 'overtime_validated': False, 'overtime_seconds': 0,
                                   'save_deadline': 1795448122.0, 'exit_deadline': 1795448422.0}, {'completed_at': 1795438404.0657194}),
        'Q-step-pause wave-001': ({'allocation_start': 1795438658.0, 'time_limit_seconds': 10800, 'overtime_validated': False, 'overtime_seconds': 0,
                                   'save_deadline': 1795448858.0, 'exit_deadline': 1795449158.0}, {'completed_at': 1795440049.7653925}),
        'Q-time-pause wave-000': ({'allocation_start': 1795438219.0, 'time_limit_seconds': 10800, 'overtime_validated': False, 'overtime_seconds': 0,
                                   'save_deadline': 1795438825.5730681, 'exit_deadline': 1795448719.0}, {'completed_at': 1795438859.57773}),
        'Q-time-pause wave-001': ({'allocation_start': 1795439073.0, 'time_limit_seconds': 10800, 'overtime_validated': False, 'overtime_seconds': 0,
                                   'save_deadline': 1795449273.0, 'exit_deadline': 1795449573.0}, {'completed_at': 1795440508.7687442}),
    }
    results['old_campaign_replay_information_only'] = {name: suite.deadline_safety(value, terminal, hpo) for name, (value, terminal) in old.items()}
    checks['old_allocations_would_be_safe_information_only'] = all(value['safe'] for value in results['old_campaign_replay_information_only'].values())
    compiled = {}
    for path in sorted(list(SUITE.glob('*.py')) + list((SUITE / 'ops').glob('*.py')) + list((SUITE / 'production_ops').glob('*.py'))):
        compile(path.read_text(), str(path), 'exec')
        compiled[str(path.relative_to(SUITE))] = True
    results['compiled'] = sorted(compiled)
    checks['all_root_ops_and_production_files_compile'] = len(compiled) > 20
    receipt = {'checks': checks, 'passed': all(checks.values()), 'results': results,
               'scope': 'CPU only, fake scontrol, standard library; no Slurm job, no remote host, no old decision imported'}
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'passed': receipt['passed'], 'checks': checks}, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
