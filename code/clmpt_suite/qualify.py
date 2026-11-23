"""Stage 0 qualification: exact pause/resume equivalence of the suite query worker, plus measured time and sizes.

Runs on CPU inside its own Slurm job. For each pair, the reference trajectory ran uninterrupted and the resumed
trajectory was forced to pause on its first allocation and continued by the controller's automatic successor. Both
use the same configuration (only the trial id differs). The pair passes only if every evaluation's metrics, every
saved score array, every logged update (loss, draw receipt) and the terminal model, optimizer, sampler and RNG states
are exactly equal; the resumed trajectory shows at least one clean forced pause followed by a successor; every
allocation of both ran on the same GPU model and recorded the validated overtime grace. Measured update time,
evaluation time and artifact sizes are written for the capacity controller. TEST is never read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch


IGNORED_STATE_KEYS = ('identity_sha256', 'log_bytes', 'log_sha256', 'initial_checkpoint', 'retained_best', 'pending_checkpoint', 'curve')
IGNORED_LOG_KEYS = ('update_seconds', 'gpu_peak_bytes')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def exact(a, b, where, differences):
    """Recursive exact comparison; tensors and arrays must match dtype, shape and every value bit for bit."""
    if len(differences) >= 20:
        return
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())):
            detail = None
            if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.is_floating_point():
                detail = float((a.cpu().double() - b.cpu().double()).abs().max())
            differences.append({'where': where, 'kind': 'tensor', 'max_abs_difference': detail})
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)):
            differences.append({'where': where, 'kind': 'array'})
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            differences.append({'where': where, 'kind': 'keys', 'only_a': sorted(map(str, set(a) - set(b)))[:5], 'only_b': sorted(map(str, set(b) - set(a)))[:5]})
        for key in sorted(set(a) & set(b), key=str):
            exact(a[key], b[key], '%s/%s' % (where, key), differences)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            differences.append({'where': where, 'kind': 'length', 'a': len(a), 'b': len(b)})
        for number, (left, right) in enumerate(zip(a, b)):
            exact(left, right, '%s[%d]' % (where, number), differences)
    elif type(a) is not type(b) or a != b:
        differences.append({'where': where, 'kind': 'value', 'a': repr(a)[:80], 'b': repr(b)[:80]})


def allocations(trial):
    rows = []
    for wave in sorted((Path(trial) / 'allocations').glob('wave-*')):
        record = {'wave': wave.name, 'files': sorted(path.name for path in wave.iterdir())}
        if (wave / 'allocation.json').exists():
            allocation = read(wave / 'allocation.json')
            record.update({'job': allocation['job'], 'overtime_validated': allocation.get('overtime_validated'), 'forced_pause_after_seconds': allocation.get('forced_pause_after_seconds')})
        if (wave / 'gpu.json').exists():
            record['gpu'] = read(wave / 'gpu.json').get('name')
        if (wave / 'CLEAN_DEADLINE.json').exists():
            record['clean_pause_step'] = read(wave / 'CLEAN_DEADLINE.json')['step']
        rows.append(record)
    return rows


def log_rows(path):
    with Path(path).open() as stream:
        return [{key: value for key, value in json.loads(line).items() if key not in IGNORED_LOG_KEYS} for line in stream if line.strip()]


def measure(training):
    updates = [json.loads(line)['update_seconds'] for line in (training / 'updates.jsonl').open() if line.strip()]
    evaluations = [read(path)['elapsed_seconds'] for path in sorted((training / 'fits').glob('fit-step-*.json'))]
    commits = [json.loads(line) for line in (training / 'commits.jsonl').open() if line.strip()]
    models = [path.stat().st_size for path in (training / 'models').glob('*.pt')]
    scores = [path.stat().st_size for path in (training / 'fits').glob('fit-step-*.npz')]
    return {'update_seconds_median': statistics.median(updates), 'update_seconds_mean': statistics.fmean(updates), 'update_seconds_p95': sorted(updates)[int(.95 * (len(updates) - 1))],
            'evaluation_seconds_max': max(evaluations), 'evaluation_seconds_mean': statistics.fmean(evaluations),
            'full_state_bytes': max(row['checkpoint_bytes'] for row in commits), 'model_only_bytes': max(models), 'scores_per_evaluation_bytes': max(scores)}


def compare_pair(pair, protocol):
    reference, resumed = Path(pair['reference']['training_dir']), Path(pair['resumed']['training_dir'])
    result = {'reference': str(reference), 'resumed': str(resumed), 'differences': {}}
    for side, directory in (('reference', reference), ('resumed', resumed)):
        assert sha(directory / 'COMPLETE.json') == pair[side]['complete_sha256']
    configs = [read(directory / 'identity.json')['config'] for directory in (reference, resumed)]
    differing = sorted(key for key in set(configs[0]) | set(configs[1]) if configs[0].get(key) != configs[1].get(key))
    result['config_keys_differing'] = differing
    checks = {'config_only_trial_id_differs': differing == ['experiment_id']}
    curves = [read(directory / 'curve.json') for directory in (reference, resumed)]
    checks['same_evaluation_steps'] = [row['step'] for row in curves[0]] == [row['step'] for row in curves[1]]
    metric_differences, score_differences = [], []
    for left, right in zip(*curves):
        exact(left['validation'], right['validation'], 'curve/step-%d' % left['step'], metric_differences)
        a, b = np.load(left['scores']), np.load(right['scores'])
        exact({key: a[key] for key in a.files}, {key: b[key] for key in b.files}, 'scores/step-%d' % left['step'], score_differences)
    checks['identical_metrics'] = not metric_differences
    checks['identical_scores'] = not score_differences
    logs = [log_rows(directory / 'updates.jsonl') for directory in (reference, resumed)]
    log_differences = []
    exact(logs[0], logs[1], 'updates', log_differences)
    checks['identical_update_log'] = not log_differences
    progress = [read(directory / 'progress.json') for directory in (reference, resumed)]
    states = [torch.load(directory / row['checkpoint'], map_location='cpu', weights_only=False) for directory, row in zip((reference, resumed), progress)]
    state_differences = []
    exact({key: value for key, value in states[0].items() if key not in IGNORED_STATE_KEYS}, {key: value for key, value in states[1].items() if key not in IGNORED_STATE_KEYS}, 'terminal', state_differences)
    checks['identical_terminal_state'] = not state_differences
    selections = [read(directory / 'selection.json') for directory in (reference, resumed)]
    checks['identical_selection'] = all(selections[0][view]['step'] == selections[1][view]['step'] for view in ('raw', 'calibrated'))
    waves = {'reference': allocations(Path(pair['reference']['trial_dir'])), 'resumed': allocations(Path(pair['resumed']['trial_dir']))}
    checks['resumed_paused_and_chained'] = (len(waves['resumed']) >= 2 and 'clean_pause_step' in waves['resumed'][0] and 'successor.json' in waves['resumed'][0]['files']
                                            and 0 < waves['resumed'][0]['clean_pause_step'] < protocol['qualification']['steps'])
    checks['restore_proof_present'] = any(resumed.glob('restore-proof-*.json'))
    gpus = {row.get('gpu') for side in waves.values() for row in side}
    checks['single_gpu_model'] = len(gpus) == 1 and None not in gpus
    checks['overtime_validated_everywhere'] = all(row.get('overtime_validated') is True for side in waves.values() for row in side)
    result.update({'checks': checks, 'allocations': waves, 'gpu_models': sorted(str(gpu) for gpu in gpus), 'passed': all(checks.values()),
                   'differences': {'metrics': metric_differences, 'scores': score_differences, 'updates': log_differences, 'terminal_state': state_differences},
                   'measured': measure(reference)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert protocol['status'] == 'FROZEN' and protocol['test_access'] is False
    index = read(args.index)
    output = Path(args.output)
    assert not output.exists()
    pairs = [compare_pair(pair, protocol) for pair in index['pairs']]
    measured = {}
    for key in ('full_state_bytes', 'model_only_bytes', 'scores_per_evaluation_bytes', 'update_seconds_p95', 'evaluation_seconds_max'):
        measured[key] = max(pair['measured'][key] for pair in pairs)
    every = protocol['evaluation_every_updates']
    window = protocol['runtime']['exit_by_allocation_seconds'] - protocol['runtime']['save_margin_seconds'] - 600
    per_update = measured['update_seconds_p95'] + measured['evaluation_seconds_max'] / every
    measured['projected_hours'] = {str(steps): steps * per_update / 3600 for steps in (protocol['continuation_parent_steps'], protocol['continuation_total_steps'] - protocol['continuation_parent_steps'])}
    measured['projected_allocations'] = {key: int(-(-hours * 3600 // window)) for key, hours in measured['projected_hours'].items()}
    bounds = protocol['runtime']['maximum_allocations']
    sufficient = (measured['projected_allocations'][str(protocol['continuation_parent_steps'])] < bounds['S']
                  and measured['projected_allocations'][str(protocol['continuation_total_steps'] - protocol['continuation_parent_steps'])] < bounds['C'])
    result = {'stage': 'qualification', 'protocol_sha256': sha(args.protocol), 'index_sha256': sha(args.index), 'qualifier_sha256': sha(__file__), 'test_access': False,
              'pairs': pairs, 'allocation_bounds_sufficient': sufficient, 'passed': sufficient and all(pair['passed'] for pair in pairs), 'measured': measured,
              'scope': 'Exact equality on one GPU model for one configuration; it does not certify cross-hardware bit identity or the untested curricula, encoders and seeds.'}
    temporary = output.with_name('.' + output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')
    temporary.rename(output)
    print(json.dumps({'passed': result['passed'], 'measured': measured}, indent=2))


if __name__ == '__main__':
    main()
