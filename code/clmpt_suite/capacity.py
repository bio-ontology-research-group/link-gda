"""Slot and storage accounting for suite submissions (login-node safe, standard library and Slurm/df only).

Slots: every queued or running job of this user counts against the user cap (24), whatever campaign or project it
belongs to; there is no separate suite limit. Storage: a trial is admitted only if the measured free space, minus
the reserve, minus the growth still owed by every admitted unfinished trial, covers the new trial's peak. Peaks use
measured qualification sizes when a passed qualification exists, the protocol estimates otherwise.
"""
import getpass
import os
from pathlib import Path


def user_jobs(run):
    """All of this user's jobs, array tasks expanded; None when squeue fails."""
    result = run(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%T|%j|%b'], 60)
    if result['returncode'] != 0:
        return None
    jobs = {}
    for line in result['stdout'].splitlines():
        if line.strip():
            identifier, state, name, gres = (line.strip().split('|', 3) + ['', '', ''])[:4]
            jobs[identifier] = {'state': state, 'name': name, 'gpu': 'gpu' in gres.lower(), 'base': identifier.split('_')[0]}
    return jobs


def slot_cap(protocol, release):
    cap = protocol['runtime']['user_active_job_cap']
    if release.get('user_active_job_cap') is not None:
        cap = min(cap, int(release['user_active_job_cap']))
    return cap


def free_bytes(run, campaign, release):
    """Smaller of df available bytes and an optional release-pinned quota command printing one integer on its last line."""
    result = run(['df', '-B1', '--output=avail', str(campaign)], 60)
    if result['returncode'] != 0:
        return None, {'df': result}
    values = {'df_avail': int(result['stdout'].split()[-1])}
    command = release.get('free_bytes_command')
    if command:
        quota = run(list(command), 60)
        token = quota['stdout'].strip().splitlines()[-1].strip() if quota['returncode'] == 0 and quota['stdout'].strip() else ''
        if not token.isdigit():
            return None, dict(values, quota=quota)
        values['quota_avail'] = int(token)
    return min(values.values()), values


def sizes(protocol, qualification):
    """Artifact sizes: measured by a passed qualification when present, else protocol estimates."""
    estimates = dict(protocol['storage']['estimates_until_qualified'])
    if qualification and qualification.get('passed') and qualification.get('measured'):
        measured = qualification['measured']
        for key in estimates:
            if measured.get(key):
                estimates[key] = int(measured[key])
        estimates['source'] = 'qualification'
    else:
        estimates['source'] = 'protocol_estimate'
    return estimates


def peak_bytes(protocol, sizes_, config):
    """Peak on-disk bytes one trajectory can own: full states with an in-flight save, model-only exports and new score archives."""
    storage = protocol['storage']
    start = protocol['continuation_parent_steps'] if config.get('continuation_parent') else 0
    evaluations = (config['maximum_steps'] - start) // config['evaluation_every_updates'] + (0 if start else 1)
    total = storage['full_states_at_peak'] * sizes_['full_state_bytes'] + storage['model_only_at_peak'] * sizes_['model_only_bytes'] + evaluations * sizes_['scores_per_evaluation_bytes']
    return int(total * storage['safety_factor'])


def used_bytes(training):
    total = 0
    if training.is_dir():
        for root, _, files in os.walk(str(training)):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except FileNotFoundError:
                    pass
    return total


def owed_bytes(protocol, sizes_, trial, config):
    return max(0, peak_bytes(protocol, sizes_, config) - used_bytes(Path(trial) / 'training'))
