"""Release a staged campaign once and submit wave 0 of named trajectories once each (login node).

--approve writes ROOT_RELEASE.json exclusively after verifying the manifest, the code checkout, free space,
the absence of queued or running jobs with the campaign's job-name prefix, and the scheduler's admission
(sbatch --test-only, which applies partition and QoS limits) of both the initial request and the conditional
long-continuation request. Admitting the long request does not grant it: the controller still requires the
operator file and the measured utilization criteria at every successor. Without --approve an existing approved release is verified. Each
--trial is submitted only if it has no INITIAL_SUBMITTING marker; an existing marker means reconcile with
squeue/sacct, never resubmit.
"""
import argparse, getpass, json, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from controller import check_release_request, read, sbatch_arguments, sha, submit_initial, verified_release


def queue_rows(prefix):
    rows = subprocess.run(['squeue', '-u', getpass.getuser(), '-h', '-o', '%i|%j|%T|%M|%L|%R'], capture_output=True, text=True, timeout=30)
    assert rows.returncode == 0, rows.stderr
    return [line for line in rows.stdout.splitlines() if line.split('|')[1].startswith(prefix + '-')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=pathlib.Path, required=True)
    parser.add_argument('--approve', action='store_true')
    parser.add_argument('--authority', default='')
    parser.add_argument('--trial', type=pathlib.Path, action='append', default=[])
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    assert not (campaign / 'OPERATOR_STOP.json').exists()
    template = read(campaign / 'config/release-template.json')
    manifest_sha = sha(campaign / 'frozen-manifest.json')
    manifest = read(campaign / 'frozen-manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(campaign / name) == digest, name
    for name, digest in manifest['code_files'].items():
        assert sha(pathlib.Path(manifest['code_dir']) / name) == digest, name
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(campaign)], text=True).splitlines()[-1])
    assert free >= template['required_free_bytes'], {'available_bytes': free, 'required_free_bytes': template['required_free_bytes']}
    (campaign / 'logs').mkdir(exist_ok=True)
    if args.approve:
        assert args.authority, 'state the user authorization in --authority'
        existing = queue_rows(template['slurm_request']['job_name_prefix'])
        assert not existing, {'existing_jobs_with_prefix': existing}
        release = dict(template, approved=True, manifest_sha256=manifest_sha, approved_at=time.time(), authority=args.authority)
        check_release_request(release)
        probe_trial = campaign / release['initial_trials'][0]
        checks = {'sbatch_test_only': {}, 'free_bytes': free, 'queue_before': existing}
        for limit in sorted({release['slurm_request']['time'], release['utilization_policy']['long_continuation']['time_limit']}):
            command = sbatch_arguments(campaign, release, probe_trial, 0, limit)
            command.insert(1, '--test-only')
            command.remove('--parsable')
            dry = subprocess.run(command, capture_output=True, text=True, timeout=30)
            checks['sbatch_test_only'][limit] = {'command': command, 'returncode': dry.returncode, 'stdout': dry.stdout, 'stderr': dry.stderr}
        qos = subprocess.run(['sacctmgr', '-nP', 'show', 'qos', release['slurm_request']['qos'], 'format=name,maxwall,maxtrespu,maxtrespa,grptres,maxjobspu,maxsubmitpu'], capture_output=True, text=True, timeout=60)
        checks['qos'] = {'returncode': qos.returncode, 'stdout': qos.stdout, 'stderr': qos.stderr}
        for name, cmd in [('storage', release['site_commands'].get('storage') or []), ('partition', ['scontrol', 'show', 'partition', release['slurm_request']['partition'], '-o'])]:
            if cmd:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                checks[name] = {'command': cmd, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        (campaign / 'release-checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        rejected = {limit: probe for limit, probe in checks['sbatch_test_only'].items() if probe['returncode'] != 0}
        assert not rejected, {'scheduler_rejected': rejected}
        with (campaign / 'ROOT_RELEASE.json').open('x') as stream:
            json.dump(release, stream, indent=2)
            stream.write('\n')
    release, _ = verified_release(campaign)
    receipts = {}
    for trial in args.trial:
        trial = trial.resolve()
        assert str(trial.relative_to(campaign)) in release['initial_trials'], 'only released initial trials may be submitted by hand'
        receipts[str(trial.relative_to(campaign))] = submit_initial(campaign, release, trial, 'initial wave by submit.py')
    print(json.dumps({'campaign': str(campaign), 'manifest_sha256': manifest_sha, 'jobs': receipts}, indent=2))


if __name__ == '__main__':
    main()
