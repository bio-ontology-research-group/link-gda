import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "compositional"
PILOT = ROOT / "notes" / "compositional-gda-20261102" / "ibex" / "campaign.pilot-a.json"
REJECTED = "sbatch: error: Batch job submission failed: Invalid partition name specified"
SOCKET = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
_spec = importlib.util.spec_from_file_location("compositional_chain", SCRIPTS / "chain.py")
assert _spec is not None and _spec.loader is not None
chain = importlib.util.module_from_spec(_spec)
sys.modules["compositional_chain"] = chain
_spec.loader.exec_module(chain)


class FakeSlurm:
    """In-memory stand-in for sbatch/squeue/sacct."""

    def __init__(self):
        self.next_id = 1000
        self.jobs = {}
        self.calls = []
        self.fail_sbatch = []
        self.accounting = {}
        self.hidden_names = set()
        self.lag = {}
        self.sacct_errors = 0
        self.lookup_errors = 0
        self.job_states = {}
        self.on_sacct = None

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        tool = argv[0]
        if tool == "sbatch":
            behaviour = self.fail_sbatch.pop(0) if self.fail_sbatch else "ok"
            if behaviour == "reject":
                raise subprocess.CalledProcessError(1, argv, stderr=REJECTED)
            if behaviour in ("socket_lost", "timeout_lost"):
                if behaviour == "timeout_lost":
                    raise subprocess.TimeoutExpired(argv, 120)
                raise subprocess.CalledProcessError(1, argv, stderr=SOCKET)
            name = argv[argv.index("--job-name") + 1]
            self.next_id += 1
            job = str(self.next_id)
            self.jobs[job] = {"name": name, "argv": argv}
            if behaviour == "timeout_after_submit":
                raise subprocess.TimeoutExpired(argv, 120)
            if behaviour == "socket_after_submit":
                raise subprocess.CalledProcessError(1, argv, stderr=SOCKET)
            return f"{job}\n"
        if tool == "squeue":
            name = argv[argv.index("--name") + 1]
            if self.lookup_errors:
                self.lookup_errors -= 1
                raise subprocess.CalledProcessError(1, argv, stderr="squeue: error: slurm_load_jobs")
            if name in self.hidden_names:
                return ""
            if self.lag.get(name, 0) > 0:
                self.lag[name] -= 1
                return ""
            return "".join(f"{j}\n" for j, rec in self.jobs.items() if rec["name"] == name)
        if tool == "sacct" and "--name" in argv:
            return ""
        if tool == "sacct":
            job = argv[argv.index("-j") + 1]
            if self.on_sacct is not None:
                self.on_sacct()
            if self.sacct_errors:
                self.sacct_errors -= 1
                raise subprocess.CalledProcessError(1, argv, stderr="sacct: error: slurmdbd unreachable")
            if "--format=JobID,State" in argv:
                state = self.job_states.get(job)
                return f"{job}|{state}\n" if state else ""
            return "".join(f"{job}_{task}|{state}|{code}\n"
                           for task, (state, code) in self.accounting.get(job, {}).items())
        if tool == "nvidia-smi":
            return "Tesla V100-SXM2-32GB\n"
        raise AssertionError(f"unexpected command {argv}")

    def sbatch_calls(self):
        return [c for c in self.calls if c[0] == "sbatch"]


@pytest.fixture
def fake():
    return FakeSlurm()


@pytest.fixture
def slurm(fake):
    return chain.Slurm(runner=fake, sleep=lambda s: None)


def make_campaign(tmp_path, names=("transe", "betae"), max_waves=3, **extra):
    cfg_dir = tmp_path / "spec"
    cfg_dir.mkdir()
    runs = []
    for name in names:
        (cfg_dir / f"{name}.json").write_text(json.dumps({"method": name, "steps": 500}))
        runs.append({"name": name, "config": f"{name}.json"})
    spec = {"code_root": "/code/link-gda", "python": "/env/bin/python", "max_waves": max_waves,
            "array_parallelism": 4, "sbatch_gpu_args": ["--partition=gpu"],
            "sbatch_cpu_args": ["--partition=cpu"], "runs": runs, **extra}
    (cfg_dir / "campaign.json").write_text(json.dumps(spec))
    campaign = tmp_path / "campaign"
    chain.init_campaign(cfg_dir / "campaign.json", campaign)
    return campaign.resolve()


def run_dir(campaign, name):
    path = campaign / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def pause(campaign, name, step, validations=None):
    path = run_dir(campaign, name)
    (path / "resume.pt").write_bytes(b"state")
    progress = {"step": step}
    if validations is not None:
        progress["curve"] = [{}] * validations
    (path / "progress.json").write_text(json.dumps(progress))


def complete(campaign, name):
    path = run_dir(campaign, name)
    (path / "COMPLETE.json").write_text(json.dumps({"step": 500}))
    (path / "progress.json").write_text(json.dumps({"step": 500}))


def wave(campaign, n):
    return json.loads(chain.wave_path(campaign, n).read_text())


def start(campaign, slurm):
    return chain.ensure_wave(campaign, slurm, 0, ["transe", "betae"],
                             reconcile_polls=1, reconcile_delay=0)


def handle(campaign, slurm, n, after):
    return chain.handle_wave(campaign, slurm, n, after, accounting_polls=2, accounting_delay=0,
                             reconcile_polls=1, reconcile_delay=0, lock_timeout=0)


def test_init_freezes_configs_and_refuses_reuse(tmp_path):
    campaign = make_campaign(tmp_path)
    frozen = json.loads((campaign / "campaign.json").read_text())
    assert [r["name"] for r in frozen["runs"]] == ["transe", "betae"]
    assert frozen["deadline_seconds"] == 11400
    for run in frozen["runs"]:
        assert chain.sha256_file(Path(run["config"])) == run["config_sha256"]
    with pytest.raises(chain.ChainError, match="not empty"):
        chain.init_campaign(tmp_path / "spec" / "campaign.json", campaign)


@pytest.mark.parametrize("override", [["--time=01:00:00"], ["--constraint=h100"],
                                      ["--signal=B:USR1@60"], ["--gres=gpu:2"]])
def test_init_refuses_overriding_fixed_resources(tmp_path, override):
    with pytest.raises(chain.ChainError, match="may not set"):
        make_campaign(tmp_path, sbatch_gpu_args=override)


def test_start_submits_array_then_controller_once(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    record = start(campaign, slurm)
    calls = fake.sbatch_calls()
    assert len(calls) == 2
    array, ctl = calls
    assert "--array=0,1%2" in array
    assert array[-1] == "/code/link-gda/scripts/compositional/train-segment.sbatch"
    assert "--partition=gpu" in array
    assert f"--dependency=afterany:{record['array']['job']}" in ctl
    assert ctl[-4:] == ["--wave", "0", "--after", record["array"]["job"]]
    assert record["controller"]["state"] == "submitted"
    start(campaign, slurm)
    assert len(fake.sbatch_calls()) == 2
    assert json.loads((campaign / "status.json").read_text())["state"] == "running"


def test_unknown_submission_is_reconciled_not_resubmitted(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["timeout_after_submit"]
    record = start(campaign, slurm)
    assert len(fake.sbatch_calls()) == 2
    assert record["array"]["reconciled"] is True
    assert record["array"]["attempts"] == 1
    assert record["array"]["job"] in fake.jobs


def test_rejected_submission_retries_with_same_token(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["reject"]
    record = start(campaign, slurm)
    arrays = [c for c in fake.sbatch_calls() if not any(a.startswith("--dependency") for a in c)]
    assert len(arrays) == 2
    assert {c[c.index("--job-name") + 1] for c in arrays} == {record["token"]}
    assert record["array"]["attempts"] == 2


def test_repeated_rejection_stops_at_attempt_cap(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["reject", "reject", "reject"]
    with pytest.raises(chain.ChainError, match="failed 3 times"):
        start(campaign, slurm)
    assert json.loads((campaign / "status.json").read_text())["state"] == "needs_attention"
    assert len(fake.sbatch_calls()) == 3
    with pytest.raises(chain.ChainError, match="failed 3 times"):
        start(campaign, slurm)
    assert len(fake.sbatch_calls()) == 3


def test_crash_between_attempt_and_reply_adopts_existing_job(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    token = "cgda-test-w000-abcdef"
    record = {"campaign_id": json.loads((campaign / "campaign.json").read_text())["campaign_id"],
              "wave": 0, "token": token, "created": chain.now(), "runs": ["transe", "betae"],
              "indices": [0, 1], "start_progress": {"transe": None, "betae": None},
              "array": {"state": "attempting", "job": None, "attempts": 1, "errors": []},
              "controller": {"state": "intent", "job": None, "attempts": 0, "errors": []},
              "outcomes": None, "handled": False, "problem": None}
    chain.write_json(chain.wave_path(campaign, 0), record)
    fake.jobs["777"] = {"name": token, "argv": []}
    armed = chain.ensure_wave(campaign, slurm, 0, ["transe", "betae"], reconcile_polls=1,
                              reconcile_delay=0)
    assert armed["array"]["job"] == "777"
    assert armed["token"] == token
    assert len(fake.sbatch_calls()) == 1
    assert "--dependency=afterany:777" in fake.sbatch_calls()[0]


def test_duplicate_jobs_under_one_token_stop_the_chain(tmp_path, fake):
    campaign = make_campaign(tmp_path)

    def duplicating(argv):
        if argv[0] == "sbatch" and not any(a.startswith("--dependency") for a in argv):
            name = argv[argv.index("--job-name") + 1]
            fake.jobs["555"] = {"name": name, "argv": list(argv)}
            fake.jobs["556"] = {"name": name, "argv": list(argv)}
            raise subprocess.TimeoutExpired(argv, 120)
        return fake(argv)

    dup = chain.Slurm(runner=duplicating, sleep=lambda s: None)
    with pytest.raises(chain.ChainError, match="more than once"):
        chain.ensure_wave(campaign, dup, 0, ["transe", "betae"], reconcile_polls=1,
                          reconcile_delay=0)
    assert wave(campaign, 0)["array"]["state"] == "duplicate"
    assert chain.summarize(campaign)["state"] == "needs_attention"


def test_controller_submission_failure_is_flagged_and_arm_recovers(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["ok", "reject", "reject", "reject"]
    with pytest.raises(chain.ChainError, match="controller submission failed"):
        start(campaign, slurm)
    status = json.loads((campaign / "status.json").read_text())
    assert status["state"] == "needs_attention"
    record = wave(campaign, 0)
    record["controller"].update(state="unknown", attempts=0)
    record["problem"] = None
    chain.write_json(chain.wave_path(campaign, 0), record)
    armed = chain.ensure_wave(campaign, slurm, 0, reconcile_polls=1, reconcile_delay=0)
    assert armed["controller"]["job"] is not None
    assert chain.summarize(campaign)["state"] == "running"


def test_pause_resumes_while_oom_error_cancel_and_missing_checkpoint_do_not(tmp_path, fake, slurm):
    names = ("pause", "timeout", "oom", "error", "cancel", "nockpt", "done", "cudaoom")
    campaign = make_campaign(tmp_path, names=names)
    chain.ensure_wave(campaign, slurm, 0, list(names), reconcile_polls=1, reconcile_delay=0)
    record = wave(campaign, 0)
    array = record["array"]["job"]
    pause(campaign, "pause", 300, 4)
    pause(campaign, "timeout", 250)
    pause(campaign, "oom", 100)
    pause(campaign, "error", 100)
    pause(campaign, "cancel", 100)
    run_dir(campaign, "nockpt")
    complete(campaign, "done")
    pause(campaign, "cudaoom", 100)
    (campaign / "logs" / f"{record['token']}-{array}_7.out").write_text(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate\n")
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("TIMEOUT", "0:15"),
                              2: ("OUT_OF_MEMORY", "0:125"), 3: ("FAILED", "1:0"),
                              4: ("CANCELLED", "0:15"), 5: ("TIMEOUT", "0:15"),
                              6: ("COMPLETED", "0:0"), 7: ("FAILED", "1:0")}
    handled = handle(campaign, slurm, 0, array)
    outcomes = {k: v["outcome"] for k, v in handled["outcomes"].items()}
    assert outcomes == {"pause": "resume", "timeout": "resume", "oom": "failed_oom",
                        "error": "failed", "cancel": "cancelled", "nockpt": "no_checkpoint",
                        "done": "completed", "cudaoom": "failed_oom"}
    nxt = wave(campaign, 1)
    assert nxt["runs"] == ["pause", "timeout"]
    assert nxt["start_progress"]["pause"] == {"step": 300, "validations": 4}
    assert "--array=0,1%2" in [c for c in fake.sbatch_calls() if nxt["token"] in c][0]
    status = chain.summarize(campaign)
    assert status["state"] == "running_with_attention"
    assert status["attention"] == ["cancel", "cudaoom", "error", "nockpt", "oom"]


def test_completed_exit_without_marker_is_a_failure(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    complete(campaign, "transe")
    run_dir(campaign, "betae")
    fake.accounting[array] = {0: ("COMPLETED", "0:0"), 1: ("COMPLETED", "0:0")}
    handled = handle(campaign, slurm, 0, array)
    assert handled["outcomes"]["betae"]["outcome"] == "exit0_without_complete"
    assert not chain.wave_path(campaign, 1).exists()
    assert chain.summarize(campaign)["state"] == "needs_attention"


def test_no_progress_since_previous_wave_is_stalled(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    pause(campaign, "transe", 200, 3)
    pause(campaign, "betae", 200, 3)
    array = start(campaign, slurm)["array"]["job"]
    pause(campaign, "transe", 200, 3)
    pause(campaign, "betae", 200, 4)
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("FAILED", "75:0")}
    handled = handle(campaign, slurm, 0, array)
    assert handled["outcomes"]["transe"]["outcome"] == "stalled"
    assert handled["outcomes"]["betae"]["outcome"] == "resume"
    assert wave(campaign, 1)["runs"] == ["betae"]


def test_unreadable_progress_is_not_resumed(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    for name in ("transe", "betae"):
        (run_dir(campaign, name) / "resume.pt").write_bytes(b"x")
    (campaign / "runs" / "transe" / "progress.json").write_text("{\"stage\": 3}")
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("FAILED", "75:0")}
    outcomes = handle(campaign, slurm, 0, array)["outcomes"]
    assert outcomes["transe"]["outcome"] == "progress_unreadable"
    assert outcomes["betae"]["outcome"] == "progress_unreadable"


def test_chain_cap_stops_resubmission(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path, max_waves=1)
    array = start(campaign, slurm)["array"]["job"]
    pause(campaign, "transe", 100)
    pause(campaign, "betae", 100)
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("TIMEOUT", "0:15")}
    handled = handle(campaign, slurm, 0, array)
    assert {v["outcome"] for v in handled["outcomes"].values()} == {"cap_reached"}
    assert not chain.wave_path(campaign, 1).exists()
    assert chain.summarize(campaign)["state"] == "needs_attention"


def test_controller_is_idempotent_and_recovers_after_crash(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    pause(campaign, "transe", 100)
    pause(campaign, "betae", 100)
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("FAILED", "75:0")}
    handle(campaign, slurm, 0, array)
    submitted = len(fake.sbatch_calls())
    handle(campaign, slurm, 0, array)
    assert len(fake.sbatch_calls()) == submitted
    record = wave(campaign, 0)
    record["handled"] = False
    chain.write_json(chain.wave_path(campaign, 0), record)
    handle(campaign, slurm, 0, array)
    assert len(fake.sbatch_calls()) == submitted
    assert wave(campaign, 0)["handled"] is True


def test_controller_refuses_wrong_array_and_incomplete_accounting(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    with pytest.raises(chain.ChainError, match="does not match"):
        handle(campaign, slurm, 0, "424242")
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("RUNNING", "0:0")}
    pause(campaign, "transe", 100)
    with pytest.raises(chain.ChainError, match="incomplete Slurm accounting"):
        handle(campaign, slurm, 0, array)
    assert wave(campaign, 0)["handled"] is False
    assert wave(campaign, 0)["outcomes"] is None
    assert len(fake.sbatch_calls()) == 2


def test_controller_waits_for_lock_then_gives_up(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    with (campaign / "chain" / "controller.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(chain.ChainError, match="lock"):
            handle(campaign, slurm, 0, array)


def test_exec_task_builds_trainer_command_and_records_segment(tmp_path, fake, slurm, monkeypatch):
    campaign = make_campaign(tmp_path)
    record = start(campaign, slurm)
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", record["array"]["job"])
    argv, code, fd = chain.prepare_task(campaign, 0, 1, runner=fake, python="/env/bin/python")
    try:
        assert code == 0
        assert argv == ["/env/bin/python", "/code/link-gda/code/compositional_gda/train.py",
                        "--config", str(campaign / "configs" / "betae.json"),
                        "--run-dir", str(campaign / "runs" / "betae"),
                        "--device", "cuda", "--deadline-seconds", "11400"]
        assert os.get_inheritable(fd)
        segment = json.loads((campaign / "chain" / "segments" / "betae" /
                              f"{record['array']['job']}_1.json").read_text())
        assert segment["gpu"] == "Tesla V100-SXM2-32GB"
        again, code2, fd2 = chain.prepare_task(campaign, 0, 1, runner=fake)
        assert again is None and code2 == chain.DUPLICATE_EXIT and fd2 is None
    finally:
        os.close(fd)


def test_exec_task_skips_complete_and_refuses_changed_config(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    start(campaign, slurm)
    complete(campaign, "transe")
    assert chain.prepare_task(campaign, 0, 0, runner=fake) == (None, 0, None)
    (campaign / "configs" / "betae.json").write_text("{\"method\": \"betae\", \"steps\": 9}")
    with pytest.raises(chain.ChainError, match="changed after init"):
        chain.prepare_task(campaign, 0, 1, runner=fake)
    with pytest.raises(chain.ChainError, match="not part of wave"):
        chain.prepare_task(campaign, 0, 5, runner=fake)


def test_templates_fix_resources_and_exec_without_site_details():
    gpu = (SCRIPTS / "train-segment.sbatch").read_text()
    cpu = (SCRIPTS / "continue.sbatch").read_text()
    assert "#SBATCH --constraint=v100|a100" in gpu
    assert "#SBATCH --time=03:30:00" in gpu
    assert "#SBATCH --gres=gpu:1" in gpu
    match = re.search(r"#SBATCH --signal=B:USR1@(\d+)", gpu)
    assert match and 0 < int(match.group(1)) < 12600 - chain.DEFAULT_DEADLINE
    assert re.search(r"^exec \"\$CGDA_PYTHON\" .*chain\.py\" exec-task", gpu, re.M)
    assert re.search(r"^exec \"\$CGDA_PYTHON\" .*chain\.py\" controller", cpu, re.M)
    assert "--gres" not in cpu
    text = gpu + cpu + (SCRIPTS / "campaign.example.json").read_text()
    for private in ("zhapacfp", "ibex", "kaust", "--account", "/home/", "miniforge", "embedpvp"):
        assert private not in text.lower()
    for script in ("train-segment.sbatch", "continue.sbatch"):
        subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True)


def test_dry_run_prints_commands_without_submitting(tmp_path, capsys, monkeypatch):
    campaign = make_campaign(tmp_path)
    monkeypatch.setattr(chain, "run_command", lambda argv: pytest.fail("submitted"))
    assert chain.main(["start", "--campaign", str(campaign), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--deadline-seconds 11400" in out
    assert "--dependency=afterany:ARRAY_JOB_ID" in out
    assert not chain.wave_path(campaign, 0).exists()


STUB_TRAINER = '''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
run = Path(args[args.index("--run-dir") + 1])
run.mkdir(parents=True, exist_ok=True)
flag = {"stop": False}
signal.signal(signal.SIGUSR1, lambda *a: flag.update(stop=True))
(run / "started.json").write_text(json.dumps({"pid": os.getpid(), "argv": args}))
limit = time.time() + 30
while not flag["stop"] and time.time() < limit:
    time.sleep(0.05)
(run / "resume.pt").write_bytes(b"state")
(run / "progress.json").write_text(json.dumps({"step": 7}))
sys.exit(75 if flag["stop"] else 1)
'''


def stub_code_root(tmp_path):
    root = tmp_path / "code-root"
    (root / "scripts" / "compositional").mkdir(parents=True)
    (root / "code" / "compositional_gda").mkdir(parents=True)
    for name in ("chain.py", "train-segment.sbatch", "continue.sbatch"):
        (root / "scripts" / "compositional" / name).symlink_to(SCRIPTS / name)
    (root / "code" / "compositional_gda" / "train.py").write_text(STUB_TRAINER)
    return root


def test_gpu_template_execs_trainer_which_receives_usr1_and_holds_lock(tmp_path, fake, slurm):
    import signal
    import time
    root = stub_code_root(tmp_path)
    campaign = make_campaign(tmp_path, code_root=str(root), python=sys.executable)
    record = start(campaign, slurm)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(CGDA_PYTHON=sys.executable, CGDA_CODE_ROOT=str(root), CGDA_CAMPAIGN=str(campaign),
               CGDA_WAVE="0", SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_JOB_ID=record["array"]["job"],
               SLURM_JOB_ID="1", PATH=env.get("PATH", "/usr/bin:/bin"))
    proc = subprocess.Popen(["bash", str(root / "scripts/compositional/train-segment.sbatch")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    started = campaign / "runs" / "transe" / "started.json"
    limit = time.time() + 20
    while not started.exists() and time.time() < limit and proc.poll() is None:
        time.sleep(0.05)
    try:
        info = json.loads(started.read_text())
        assert info["pid"] == proc.pid
        assert info["argv"][-2:] == ["--deadline-seconds", "11400"]
        with (campaign / "chain" / "locks" / "transe.lock").open("a") as probe:
            with pytest.raises(OSError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(proc.pid, signal.SIGUSR1)
        out, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == chain.PAUSE_EXIT, out
    assert "task=0" in out


def test_cli_start_strips_job_environment_from_sbatch(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seen = tmp_path / "seen.txt"
    (bin_dir / "sbatch").write_text(
        "#!/bin/bash\n"
        f"echo \"${{SLURM_JOB_ID:-none}} $*\" >> {seen}\n"
        f"n=$(wc -l < {seen}); echo $((5000 + n))\n")
    for tool in ("squeue", "sacct"):
        (bin_dir / tool).write_text("#!/bin/bash\nexit 0\n")
    for tool in ("sbatch", "squeue", "sacct"):
        (bin_dir / tool).chmod(0o755)
    campaign = make_campaign(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    assert chain.main(["start", "--campaign", str(campaign)]) == 0
    lines = seen.read_text().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("none ") for line in lines)
    assert "--dependency=afterany:5001" in lines[1]
    assert chain.main(["start", "--campaign", str(campaign)]) == 0
    assert len(seen.read_text().splitlines()) == 2
    assert chain.main(["status", "--campaign", str(campaign)]) == 0


def controller(campaign, slurm, n, after, **options):
    options = {"accounting_polls": 2, "accounting_delay": 0, "reconcile_polls": 1,
               "reconcile_delay": 0, "lock_timeout": 0, **options}
    return chain.run_controller(campaign, n, after, slurm=slurm, **options)


def status(campaign):
    return json.loads((campaign / "status.json").read_text())


def assert_needs_attention(campaign, n, text):
    state = status(campaign)
    assert state["state"] == "needs_attention", state
    assert any(p.get("wave") == n and text in (p.get("problem") or "") for p in state["problems"]), state


def test_incomplete_accounting_is_needs_attention_then_recovers(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    complete(campaign, "transe")
    complete(campaign, "betae")
    fake.accounting[array] = {0: ("COMPLETED", "0:0"), 1: ("RUNNING", "0:0")}
    assert controller(campaign, slurm, 0, array) == 1
    assert_needs_attention(campaign, 0, "incomplete Slurm accounting")
    assert wave(campaign, 0)["handled"] is False
    assert len(fake.sbatch_calls()) == 2
    fake.accounting[array][1] = ("COMPLETED", "0:0")
    assert controller(campaign, slurm, 0, array) == 0
    assert wave(campaign, 0)["problem"] is None
    assert status(campaign)["state"] == "completed"


def test_sacct_client_error_is_needs_attention_not_running(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    fake.sacct_errors = 10
    assert controller(campaign, slurm, 0, array) == 1
    assert_needs_attention(campaign, 0, "slurmdbd unreachable")
    assert status(campaign)["state"] != "running"


def test_transient_sacct_error_is_retried_within_the_window(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    pause(campaign, "transe", 100)
    pause(campaign, "betae", 100)
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("FAILED", "75:0")}
    fake.sacct_errors = 1
    assert controller(campaign, slurm, 0, array) == 0
    assert wave(campaign, 0)["handled"] is True
    assert wave(campaign, 1)["controller"]["job"] is not None


def test_lock_timeout_and_wrong_array_are_visible(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    with (campaign / "chain" / "controller.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert chain.run_controller(campaign, 0, array, slurm=slurm, lock_timeout=0) == 1
    assert list((campaign / "chain" / "failures").glob("wave_000-*.json"))
    assert_needs_attention(campaign, 0, "held for more than")
    assert controller(campaign, slurm, 0, "424242") == 1
    assert_needs_attention(campaign, 0, "does not match")


def test_unreadable_wave_record_is_visible(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    chain.wave_path(campaign, 0).write_text("{\"campaign_id\": ")
    assert controller(campaign, slurm, 0, array) == 1
    assert_needs_attention(campaign, 0, "unreadable")


def test_sigterm_at_controller_time_limit_is_recorded(tmp_path, fake, slurm):
    import signal
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    previous = signal.getsignal(signal.SIGTERM)
    fake.on_sacct = lambda: os.kill(os.getpid(), signal.SIGTERM)
    assert controller(campaign, slurm, 0, array) == 1
    assert signal.getsignal(signal.SIGTERM) == previous
    assert_needs_attention(campaign, 0, "ControllerTerminated")
    assert len(fake.sbatch_calls()) == 2


def test_next_wave_client_error_is_recorded(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    array = start(campaign, slurm)["array"]["job"]
    pause(campaign, "transe", 100)
    pause(campaign, "betae", 100)
    fake.accounting[array] = {0: ("FAILED", "75:0"), 1: ("FAILED", "75:0")}
    fake.fail_sbatch = ["socket_lost"]
    assert controller(campaign, slurm, 0, array) == 1
    assert_needs_attention(campaign, 0, "next wave not armed")
    assert_needs_attention(campaign, 1, "unresolved")
    assert wave(campaign, 1)["array"]["attempts"] == 1


def test_status_flags_a_controller_job_that_ended_without_handling(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    record = start(campaign, slurm)
    assert chain.summarize(campaign)["state"] == "running"
    ctl = record["controller"]["job"]
    fake.job_states[ctl] = "RUNNING"
    assert chain.probe_controllers(campaign, slurm) == []
    fake.job_states[ctl] = "TIMEOUT"
    assert [f["job"] for f in chain.probe_controllers(campaign, slurm)] == [ctl]
    assert chain.probe_controllers(campaign, slurm) == []
    chain.write_status(campaign)
    assert_needs_attention(campaign, 0, f"controller job {ctl} ended TIMEOUT")
    array = record["array"]["job"]
    complete(campaign, "transe")
    complete(campaign, "betae")
    fake.accounting[array] = {0: ("COMPLETED", "0:0"), 1: ("COMPLETED", "0:0")}
    assert controller(campaign, slurm, 0, array) == 0
    assert status(campaign)["state"] == "completed"


def test_cli_status_probes_controller_jobs(tmp_path, monkeypatch, fake, slurm):
    campaign = make_campaign(tmp_path)
    record = start(campaign, slurm)
    fake.job_states[record["controller"]["job"]] = "CANCELLED"
    monkeypatch.setattr(chain, "run_command", fake)
    assert chain.main(["status", "--campaign", str(campaign), "--no-slurm"]) == 0
    assert chain.main(["status", "--campaign", str(campaign)]) == 1
    assert_needs_attention(campaign, 0, "ended CANCELLED")


def test_ambiguous_failure_with_empty_lookup_is_not_resubmitted(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["socket_lost"]
    with pytest.raises(chain.ChainError, match="unresolved"):
        chain.ensure_wave(campaign, slurm, 0, ["transe", "betae"], reconcile_polls=3,
                          reconcile_delay=0)
    assert len(fake.sbatch_calls()) == 1
    assert wave(campaign, 0)["array"]["state"] == "unresolved"
    assert_needs_attention(campaign, 0, "unresolved")
    with pytest.raises(chain.ChainError, match="unresolved"):
        chain.ensure_wave(campaign, slurm, 0, reconcile_polls=1, reconcile_delay=0)
    assert len(fake.sbatch_calls()) == 1
    armed = chain.ensure_wave(campaign, slurm, 0, reconcile_polls=1, reconcile_delay=0,
                              confirm_absent=True)
    assert armed["array"]["attempts"] == 2
    assert armed["controller"]["job"] is not None
    assert len(fake.sbatch_calls()) == 3
    assert chain.summarize(campaign)["state"] == "running"


@pytest.mark.parametrize("behaviour", ["timeout_lost", "socket_lost"])
def test_ambiguous_failures_never_retry_on_their_own(tmp_path, fake, slurm, behaviour):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = [behaviour]
    with pytest.raises(chain.ChainError, match="ambiguous"):
        start(campaign, slurm)
    assert len(fake.sbatch_calls()) == 1


def test_lagging_lookup_adopts_an_accepted_job_without_duplicate(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["socket_after_submit"]
    original = fake.__call__

    def lagging(argv):
        if argv[0] == "squeue" and not fake.lag:
            fake.lag[argv[argv.index("--name") + 1]] = 2
        return original(argv)

    lag = chain.Slurm(runner=lagging, sleep=lambda s: None)
    record = chain.ensure_wave(campaign, lag, 0, ["transe", "betae"], reconcile_polls=4,
                               reconcile_delay=0)
    arrays = [c for c in fake.sbatch_calls() if not any(a.startswith("--dependency") for a in c)]
    assert len(arrays) == 1
    assert record["array"]["reconciled"] is True
    assert record["array"]["job"] in fake.jobs


def test_crash_mid_call_with_no_visible_job_is_unresolved(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    record = {"campaign_id": json.loads((campaign / "campaign.json").read_text())["campaign_id"],
              "wave": 0, "token": "cgda-test-w000-000000", "created": chain.now(),
              "runs": ["transe", "betae"], "indices": [0, 1],
              "start_progress": {"transe": None, "betae": None},
              "array": {"state": "attempting", "job": None, "attempts": 1, "errors": []},
              "controller": {"state": "intent", "job": None, "attempts": 0, "errors": []},
              "outcomes": None, "handled": False, "problem": None}
    chain.write_json(chain.wave_path(campaign, 0), record)
    with pytest.raises(chain.ChainError, match="unresolved"):
        chain.ensure_wave(campaign, slurm, 0, reconcile_polls=2, reconcile_delay=0)
    assert fake.sbatch_calls() == []


def test_failed_lookup_after_definite_rejection_is_not_taken_as_absence(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["reject"]
    fake.lookup_errors = 1
    with pytest.raises(chain.ChainError, match="Slurm lookup failed"):
        start(campaign, slurm)
    assert len(fake.sbatch_calls()) == 1
    record = start(campaign, slurm)
    assert record["array"]["attempts"] == 2
    assert record["problem"] is None


def test_confirm_absent_still_adopts_a_job_that_appeared(tmp_path, fake, slurm):
    campaign = make_campaign(tmp_path)
    fake.fail_sbatch = ["socket_after_submit"]
    token = None

    def hidden(argv):
        if argv[0] == "squeue" and token is None:
            return ""
        return fake(argv)

    blind = chain.Slurm(runner=hidden, sleep=lambda s: None)
    with pytest.raises(chain.ChainError, match="unresolved"):
        chain.ensure_wave(campaign, blind, 0, ["transe", "betae"], reconcile_polls=2,
                          reconcile_delay=0)
    token = wave(campaign, 0)["token"]
    armed = chain.ensure_wave(campaign, blind, 0, reconcile_polls=1, reconcile_delay=0,
                              confirm_absent=True)
    arrays = [c for c in fake.sbatch_calls() if not any(a.startswith("--dependency") for a in c)]
    assert len(arrays) == 1
    assert armed["array"]["reconciled"] is True


def test_sbatch_outcome_classification():
    def raising(exc):
        def runner(argv):
            raise exc
        return chain.Slurm(runner=runner, sleep=lambda s: None)

    cases = [
        (subprocess.CalledProcessError(1, ["sbatch"], stderr=REJECTED), chain.SubmissionRejected),
        (subprocess.CalledProcessError(1, ["sbatch"], stderr=SOCKET), chain.SubmissionUnknown),
        (subprocess.CalledProcessError(1, ["sbatch"], stderr="sbatch: error: Invalid feature "
                                       "specification; Unable to contact slurm controller"),
         chain.SubmissionUnknown),
        (subprocess.CalledProcessError(1, ["sbatch"], stderr="something odd"), chain.SubmissionUnknown),
        (subprocess.TimeoutExpired(["sbatch"], 120), chain.SubmissionUnknown),
        (FileNotFoundError("sbatch"), chain.SubmissionRejected),
    ]
    for exc, expected in cases:
        with pytest.raises(chain.ChainError) as raised:
            raising(exc).sbatch(["x"])
        assert type(raised.value) is expected, exc
    with pytest.raises(chain.SubmissionUnknown):
        chain.Slurm(runner=lambda argv: "Submitted batch job\n").sbatch(["x"])


def test_timing_windows_fit_the_controller_time_limit():
    text = (SCRIPTS / "continue.sbatch").read_text()
    hours, minutes, seconds = map(int, re.search(r"--time=(\d+):(\d+):(\d+)", text).groups())
    limit = hours * 3600 + minutes * 60 + seconds
    accounting = (chain.ACCOUNTING_POLLS - 1) * chain.ACCOUNTING_DELAY
    reconcile = (chain.RECONCILE_POLLS - 1) * chain.RECONCILE_DELAY
    assert accounting >= 300
    assert reconcile >= 60
    assert chain.LOCK_TIMEOUT + accounting + 2 * reconcile + 2 * 120 < limit


def test_exclusive_publication_is_complete_or_absent(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    assert chain.create_json_exclusive(target, {"a": 1})
    assert not chain.create_json_exclusive(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}
    other = tmp_path / "crash.json"
    real_fsync = os.fsync

    def crash(fd):
        raise OSError("disk gone")

    monkeypatch.setattr(chain.os, "fsync", crash)
    with pytest.raises(OSError):
        chain.create_json_exclusive(other, {"b": 1})
    monkeypatch.setattr(chain.os, "fsync", real_fsync)
    assert not other.exists()

    def no_links(src, dst):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(chain.os, "link", no_links)
    with pytest.raises(chain.ChainError, match="hard links"):
        chain.create_json_exclusive(other, {"b": 1})
    assert not other.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


def test_exclusive_publication_race_has_one_winner(tmp_path, monkeypatch):
    target = tmp_path / "race.json"
    real_link = os.link

    def racing(src, dst):
        Path(dst).write_text(json.dumps({"winner": "other"}))
        return real_link(src, dst)

    monkeypatch.setattr(chain.os, "link", racing)
    assert chain.create_json_exclusive(target, {"winner": "me"}) is False
    assert json.loads(target.read_text()) == {"winner": "other"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["race.json"]


@pytest.mark.parametrize("deadline,ok", [(11400, True), (11401, False), (12599, False),
                                         (0, False), (11400.0, False)])
def test_deadline_is_capped_at_the_margin(tmp_path, deadline, ok):
    if ok:
        make_campaign(tmp_path, deadline_seconds=deadline)
    else:
        with pytest.raises(chain.ChainError, match="deadline_seconds"):
            make_campaign(tmp_path, deadline_seconds=deadline)


@pytest.mark.skipif(not PILOT.is_file(), reason="local ibex deployment notes absent")
def test_local_pilot_spec_has_five_matched_variants(tmp_path):
    spec = json.loads(PILOT.read_text())
    assert spec["array_parallelism"] == 4
    assert spec["deadline_seconds"] == chain.DEFAULT_DEADLINE
    assert len(spec["runs"]) == 5
    configs = {run["name"]: json.loads((PILOT.parent / run["config"]).read_text())
               for run in spec["runs"]}
    assert sorted(c["method"] for c in configs.values()) == ["betae", "geometre", "geometre",
                                                             "q2b", "transe"]
    geometre = [c for c in configs.values() if c["method"] == "geometre"]
    assert sorted(c["membership_mode"] for c in geometre) == ["original", "sign_corrected"]
    strip = lambda c: {k: v for k, v in c.items() if k not in ("name", "method", "membership_mode")}
    assert len({json.dumps(strip(c), sort_keys=True) for c in configs.values()}) == 1
    assert all(c["name"] == name for name, c in configs.items())
    pytest.importorskip("torch")
    from compositional_gda import train
    for name, config in configs.items():
        resolved, data_dir = train.resolve_config(config, PILOT.parent / "configs" / f"{name}.json")
        assert resolved["dim"] == 100 and resolved["steps"] == 500 and resolved["seed"] == 0
        assert data_dir.is_absolute()
    frozen = chain.init_campaign(PILOT, tmp_path / "campaign")
    assert "--array=0,1,2,3,4%4" in chain.array_argv(frozen, {
        "wave": 0, "token": "t", "indices": [0, 1, 2, 3, 4], "array": {"job": "1"}})


def trainer_workspace(tmp_path):
    pytest.importorskip("torch")
    from test_compositional_models import make_prepared
    from compositional_gda import train
    make_prepared(tmp_path / "fold0")
    config = tmp_path / "transe.json"
    config.write_text(json.dumps({
        "data_dir": "fold0", "method": "transe", "dim": 4, "steps": 6, "eval_every": 2,
        "batch_size": 4, "negative_sample_size": 3, "learning_rate": 0.05,
        "validation_diseases": 3, "query_chunk": 3, "score_chunk_elements": 64,
        "checkpoint_every_steps": 1}))
    return train, config


def run_trainer(train, config, run, *extra):
    return train.main(["--config", str(config), "--run-dir", str(run), "--device", "cpu", *extra])


def checkpoint_step(run):
    import torch
    return torch.load(run / "resume.pt", map_location="cpu", weights_only=False)["step"]


def test_real_trainer_pause_is_resumed_and_progress_matches_checkpoint(tmp_path):
    train, config = trainer_workspace(tmp_path)
    run = tmp_path / "run"
    assert run_trainer(train, config, run, "--stop-after", "3") == chain.PAUSE_EXIT
    progress = json.loads((run / "progress.json").read_text())
    assert progress["step"] == checkpoint_step(run) == 3
    outcome, seen = chain.classify("FAILED", f"{chain.PAUSE_EXIT}:0", run, None)
    assert outcome == "resume" and seen["step"] == 3
    start_progress = chain.read_progress(run)
    assert run_trainer(train, config, run, "--stop-after", "5") == chain.PAUSE_EXIT
    assert chain.read_progress(run)["step"] == checkpoint_step(run) == 5
    assert chain.classify("TIMEOUT", "0:15", run, start_progress)[0] == "resume"
    assert run_trainer(train, config, run) == 0
    assert chain.classify("COMPLETED", "0:0", run, None)[0] == "completed"


def test_real_trainer_commits_checkpoint_before_progress(tmp_path, monkeypatch):
    train, config = trainer_workspace(tmp_path)
    run = tmp_path / "run"
    assert run_trainer(train, config, run, "--stop-after", "3") == chain.PAUSE_EXIT
    start_progress = chain.read_progress(run)
    order = []
    real_save, real_write = train.atomic_torch_save, train.write_json

    def save(path, value):
        order.append(Path(path).name)
        return real_save(path, value)

    def write(path, value):
        if Path(path).name == "progress.json" and order and order[-1] == "resume.pt" \
                and value["step"] == 4:
            raise KeyboardInterrupt("killed between resume.pt and progress.json")
        order.append(Path(path).name)
        return real_write(path, value)

    monkeypatch.setattr(train, "atomic_torch_save", save)
    monkeypatch.setattr(train, "write_json", write)
    with pytest.raises(KeyboardInterrupt):
        run_trainer(train, config, run)
    assert checkpoint_step(run) == 4
    assert chain.read_progress(run)["step"] == 3
    assert chain.classify("NODE_FAIL", "0:0", run, start_progress)[0] == "stalled"
    monkeypatch.setattr(train, "write_json", real_write)
    assert run_trainer(train, config, run, "--stop-after", "5") == chain.PAUSE_EXIT
    assert chain.read_progress(run)["step"] == checkpoint_step(run) == 5


@pytest.mark.parametrize("change", ["source", "versions"])
def test_real_trainer_refuses_resume_under_changed_code_or_packages(tmp_path, monkeypatch, change):
    train, config = trainer_workspace(tmp_path)
    run = tmp_path / "run"
    assert run_trainer(train, config, run, "--stop-after", "3") == chain.PAUSE_EXIT
    before = (run / "resume.pt").read_bytes()
    if change == "source":
        real = train.source_hashes
        monkeypatch.setattr(train, "source_hashes", lambda: {**real(), "train.py": "0" * 64})
    else:
        real = train.installed_versions
        monkeypatch.setattr(train, "installed_versions", lambda: {**real(), "torch": "9.9.9"})
    with pytest.raises(train.IdentityMismatch):
        run_trainer(train, config, run)
    assert (run / "resume.pt").read_bytes() == before
    assert chain.classify("FAILED", "1:0", run, chain.read_progress(run))[0] == "failed"
