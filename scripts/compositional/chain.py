#!/usr/bin/env python3
"""Checkpoint-resume chain controller for compositional GDA training on Slurm.

A campaign is a frozen list of training runs. Each wave is one GPU job array,
followed by a CPU controller job that depends on it with ``afterany``. The
controller classifies every task from Slurm accounting and the run directory,
then submits the next wave only for runs that paused with a checkpoint and made
progress. Out-of-memory failures, ordinary errors, cancellations, stalled runs
and runs that reach the wave cap are reported and never retried.

Submission is idempotent. A wave record is published exclusively before any
``sbatch`` call, an attempt counter is persisted before each call, and an
interrupted or failed call is reconciled against Slurm by the wave's unique
job name before another submission is allowed. Only a definite rejection by
``sbatch`` is retried automatically. A call whose outcome is ambiguous (time
out, lost connection, unparseable reply, crash mid-call) is never resubmitted
by the controller: when no job with the wave's name appears within the
reconciliation window, the wave is marked for attention and an operator
confirms absence with ``arm --confirm-absent``.

Every controller failure is made visible. An exception of any kind, a lock
timeout, a Slurm client error that outlasts the accounting window or a
``SIGTERM`` at the job's time limit is recorded on the wave, and ``status``
also flags an unhandled wave whose controller job ended in Slurm.

Only the standard library is used, so the controller runs in any Python 3.8+.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_TEMPLATE = "train-segment.sbatch"
CPU_TEMPLATE = "continue.sbatch"
TRAINER = Path("code") / "compositional_gda" / "train.py"

PAUSE_EXIT = 75
DUPLICATE_EXIT = 3
DEFAULT_DEADLINE = 11400
TIME_LIMIT_SECONDS = 12600
DEADLINE_MARGIN_SECONDS = 1200
MAX_DEADLINE = TIME_LIMIT_SECONDS - DEADLINE_MARGIN_SECONDS
RECOVERABLE_STATES = frozenset({"TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"})
TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL",
    "OUT_OF_MEMORY", "CANCELLED", "DEADLINE", "REVOKED",
})
RESERVED_SBATCH = ("--time", "-t", "--constraint", "-C", "--gres", "--signal",
                   "--array", "-a", "--job-name", "-J", "--dependency", "-d",
                   "--output", "-o", "--error", "-e", "--export", "--chdir", "-D",
                   "--parsable")
OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED",
                "oom-kill", "Out Of Memory")
MAX_SUBMIT_ATTEMPTS = 3
RECONCILE_POLLS = 6
RECONCILE_DELAY = 20.0
ACCOUNTING_POLLS = 25
ACCOUNTING_DELAY = 20.0
LOCK_TIMEOUT = 180.0
FAILURE_LOCK_TIMEOUT = 30.0
CLIENT_ERRORS = (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError)
DEFINITE_REJECTIONS = (
    "Invalid", "invalid", "Requested node configuration is not available",
    "violates accounting/QOS policy", "QOSMax", "AssocMax", "Unable to open file",
    "unrecognized option", "unknown option",
)
AMBIGUOUS_MARKERS = (
    "timed out", "Timed out", "Socket", "Zero Bytes", "Unable to contact", "temporarily",
    "Connection", "connection", "slurm_receive_msg", "Communication",
)

ATTENTION_OUTCOMES = frozenset({
    "failed", "failed_oom", "cancelled", "stalled", "cap_reached",
    "no_checkpoint", "progress_unreadable", "exit0_without_complete", "missing",
})


class ChainError(RuntimeError):
    """A condition that must stop the controller without submitting anything."""


class SubmissionUnknown(ChainError):
    """An ``sbatch`` call whose outcome could not be established."""


class SubmissionRejected(ChainError):
    """An ``sbatch`` call that Slurm definitely refused, so no job exists."""


class ControllerTerminated(ChainError):
    """The controller received ``SIGTERM``, normally at its Slurm time limit."""


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, data) -> None:
    """Write JSON atomically: temp file, fsync, rename, fsync of the directory."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def create_json_exclusive(path: Path, data) -> bool:
    """Publish ``path`` complete and only if absent. Returns False when it already exists.

    The content is written and fsynced to a private temporary file, which is
    then hard-linked into place. ``link`` fails when the target exists, so the
    publication is exclusive, and readers never see a partly written file.
    """
    path = Path(path)
    if path.exists():
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(tmp), str(path))
        except FileExistsError:
            return False
        except OSError as exc:
            raise ChainError(f"cannot publish {path} exclusively: hard links unsupported ({exc})") from exc
    finally:
        try:
            os.unlink(str(tmp))
        except FileNotFoundError:
            pass
    fsync_dir(path.parent)
    return True


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def read_record(path: Path) -> dict:
    """Read a controller-owned JSON record, turning corruption into a ``ChainError``."""
    try:
        data = read_json(path)
    except FileNotFoundError as exc:
        raise ChainError(f"{path} does not exist") from exc
    except (OSError, ValueError) as exc:
        raise ChainError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ChainError(f"{path} does not hold a JSON object")
    return data


def append_event(campaign: Path, event: str, **fields) -> None:
    record = {"time": now(), "event": event, **fields}
    path = Path(campaign) / "chain" / "events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


class Lock:
    """Exclusive ``flock`` on a file, with a bounded wait."""

    def __init__(self, path: Path, timeout: float = LOCK_TIMEOUT,
                 sleep: Callable[[float], None] = time.sleep):
        self.path = Path(path)
        self.timeout = timeout
        self.sleep = sleep
        self.handle = None

    def __enter__(self):
        handle = self.path.open("a")
        self.handle = handle
        waited = 0.0
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if waited >= self.timeout:
                    handle.close()
                    raise ChainError(f"lock {self.path} held for more than {self.timeout:.0f} s")
                self.sleep(5.0)
                waited += 5.0

    def __exit__(self, *exc):
        handle = self.handle
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return False


def run_command(argv: Sequence[str]) -> str:
    """Run a Slurm client command. ``sbatch`` gets an environment without the
    calling job's ``SLURM_*`` variables so a controller job does not leak its
    own allocation into the jobs it submits."""
    env = None
    if argv and argv[0] == "sbatch":
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    result = subprocess.run(list(argv), check=True, capture_output=True, text=True,
                            timeout=120, env=env)
    return result.stdout


class Slurm:
    """Thin wrapper over ``sbatch``, ``squeue`` and ``sacct``.

    ``runner`` takes an argv and returns stdout or raises. Tests replace it.
    """

    def __init__(self, runner: Optional[Callable[[Sequence[str]], str]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.runner = runner or (lambda argv: run_command(argv))
        self.sleep = sleep

    def sbatch(self, argv: Sequence[str]) -> str:
        """Submit and return the job ID.

        Raises ``SubmissionRejected`` only when no job can exist: ``sbatch``
        could not be executed, or it exited non-zero with a refusal message
        and no sign of a lost connection. Everything else is
        ``SubmissionUnknown``.
        """
        try:
            out = self.runner(["sbatch", "--parsable", *argv])
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or str(exc)).strip()
            if (any(p in detail for p in DEFINITE_REJECTIONS)
                    and not any(m in detail for m in AMBIGUOUS_MARKERS)):
                raise SubmissionRejected(f"sbatch rejected the job: {detail}") from exc
            raise SubmissionUnknown(f"sbatch did not confirm submission: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SubmissionUnknown(f"sbatch did not confirm submission: {exc}") from exc
        except OSError as exc:
            raise SubmissionRejected(f"sbatch could not be executed: {exc}") from exc
        job = out.strip().split(";")[0]
        if not job.isdigit():
            raise SubmissionUnknown(f"unparseable sbatch output: {out!r}")
        return job

    def find_by_name(self, name: str, since: str) -> set:
        jobs = set()
        queue = self.runner(["squeue", "-h", "--name", name, "-o", "%i"])
        accounting = self.runner(["sacct", "-X", "-n", "-P", "--name", name,
                                  "--starttime", since, "--format=JobID,JobName"])
        for line in queue.splitlines():
            if line.strip():
                jobs.add(line.strip().split("_")[0])
        for line in accounting.splitlines():
            cells = line.split("|")
            if len(cells) >= 2 and cells[1] == name and cells[0].strip():
                jobs.add(cells[0].strip().split("_")[0])
        return jobs

    def array_accounting(self, array_job: str) -> Dict[int, Tuple[str, str]]:
        out = self.runner(["sacct", "-j", array_job, "-X", "-n", "-P",
                           "--format=JobID,State,ExitCode"])
        tasks: Dict[int, Tuple[str, str]] = {}
        for line in out.splitlines():
            cells = line.split("|")
            if len(cells) < 3:
                continue
            match = re.fullmatch(rf"{re.escape(array_job)}_(\d+)", cells[0].strip())
            if not match:
                continue
            state = cells[1].split()[0].rstrip("+") if cells[1].strip() else ""
            tasks[int(match.group(1))] = (state, cells[2].strip())
        return tasks

    def job_state(self, job: str) -> Optional[str]:
        """Accounting state of a plain job, or None when Slurm has no record yet."""
        out = self.runner(["sacct", "-j", job, "-X", "-n", "-P", "--format=JobID,State"])
        for line in out.splitlines():
            cells = line.split("|")
            if len(cells) >= 2 and cells[0].strip() == job and cells[1].strip():
                return cells[1].split()[0].rstrip("+")
        return None


def campaign_paths(campaign: Path) -> Dict[str, Path]:
    campaign = Path(campaign)
    return {
        "spec": campaign / "campaign.json",
        "configs": campaign / "configs",
        "runs": campaign / "runs",
        "logs": campaign / "logs",
        "chain": campaign / "chain",
        "locks": campaign / "chain" / "locks",
        "segments": campaign / "chain" / "segments",
        "failures": campaign / "chain" / "failures",
        "lock": campaign / "chain" / "controller.lock",
        "status": campaign / "status.json",
    }


def wave_path(campaign: Path, wave: int) -> Path:
    return campaign_paths(campaign)["chain"] / f"wave_{wave:03d}.json"


def _check_path_value(label: str, value: str) -> None:
    if "," in value or "\n" in value:
        raise ChainError(f"{label} must not contain commas or newlines: {value!r}")


def _check_sbatch_args(label: str, args: Iterable[str]) -> List[str]:
    args = list(args)
    for arg in args:
        key = arg.split("=", 1)[0]
        if key in RESERVED_SBATCH:
            raise ChainError(f"{label} may not set {key}; the controller or template fixes it")
    return args


def init_campaign(spec_path: Path, campaign: Path) -> dict:
    """Freeze a campaign from a spec file into a new, empty campaign directory."""
    spec_path = Path(spec_path).resolve()
    campaign = Path(campaign).resolve()
    spec = read_json(spec_path)
    required = ("code_root", "python", "max_waves", "runs")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ChainError(f"spec lacks {missing}")
    if not isinstance(spec["max_waves"], int) or spec["max_waves"] < 1:
        raise ChainError("max_waves must be a positive integer")
    deadline = spec.get("deadline_seconds", DEFAULT_DEADLINE)
    if not isinstance(deadline, int) or isinstance(deadline, bool) or not 0 < deadline <= MAX_DEADLINE:
        raise ChainError(f"deadline_seconds must be an integer in (0, {MAX_DEADLINE}]: the "
                         f"{TIME_LIMIT_SECONDS} s limit less a {DEADLINE_MARGIN_SECONDS} s margin")
    code_root = Path(spec["code_root"])
    python = spec["python"]
    for label, value in (("campaign", str(campaign)), ("code_root", str(code_root)),
                         ("python", python)):
        _check_path_value(label, value)
    if not code_root.is_absolute():
        raise ChainError("code_root must be an absolute path on the compute cluster")
    gpu_args = _check_sbatch_args("sbatch_gpu_args", spec.get("sbatch_gpu_args", []))
    cpu_args = _check_sbatch_args("sbatch_cpu_args", spec.get("sbatch_cpu_args", []))
    runs = spec["runs"]
    if not runs:
        raise ChainError("spec lists no runs")
    names = [run["name"] for run in runs]
    if len(set(names)) != len(names):
        raise ChainError("run names must be unique")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ChainError(f"run name {name!r} must be a plain file name")
    if campaign.exists() and any(campaign.iterdir()):
        raise ChainError(f"{campaign} exists and is not empty; campaigns are never reused")
    frozen_runs = []
    sources = []
    for index, run in enumerate(runs):
        source = Path(run["config"])
        if not source.is_absolute():
            source = spec_path.parent / source
        if not source.is_file():
            raise ChainError(f"config for {run['name']} not found: {source}")
        read_json(source)
        sources.append(source)
        extra = [str(arg) for arg in run.get("trainer_args", [])]
        for arg in extra:
            if arg.split("=", 1)[0] in ("--config", "--run-dir", "--device", "--deadline-seconds"):
                raise ChainError(f"trainer_args for {run['name']} may not override {arg}")
        frozen_runs.append({"index": index, "name": run["name"], "trainer_args": extra})
    paths = campaign_paths(campaign)
    for key in ("configs", "runs", "logs", "chain", "locks", "segments", "failures"):
        paths[key].mkdir(parents=True, exist_ok=True)
    for run, source in zip(frozen_runs, sources):
        target = paths["configs"] / f"{run['name']}.json"
        shutil.copyfile(source, target)
        run["config"] = str(target)
        run["config_sha256"] = sha256_file(target)
        run["config_source"] = str(source)
        run["run_dir"] = str(paths["runs"] / run["name"])
    frozen = {
        "campaign_id": secrets.token_hex(4),
        "created": now(),
        "campaign": str(campaign),
        "code_root": str(code_root),
        "python": python,
        "device": spec.get("device", "cuda"),
        "deadline_seconds": deadline,
        "max_waves": spec["max_waves"],
        "array_parallelism": int(spec.get("array_parallelism", len(frozen_runs))),
        "sbatch_gpu_args": gpu_args,
        "sbatch_cpu_args": cpu_args,
        "runs": frozen_runs,
        "spec_source": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
    }
    if not create_json_exclusive(paths["spec"], frozen):
        raise ChainError("campaign.json appeared concurrently; refusing to continue")
    append_event(campaign, "init", campaign_id=frozen["campaign_id"], runs=names)
    write_status(campaign)
    return frozen


def load_campaign(campaign: Path) -> dict:
    path = campaign_paths(campaign)["spec"]
    if not path.is_file():
        raise ChainError(f"{campaign} is not an initialised campaign")
    return read_record(path)


def runs_by_name(spec: dict) -> Dict[str, dict]:
    return {run["name"]: run for run in spec["runs"]}


def read_progress(run_dir: Path) -> Optional[dict]:
    """Committed progress of a run, or None when absent or unreadable.

    Uses ``progress.json`` written by the trainer: an integer ``step`` is
    required; validations are counted from ``curve`` when present.
    """
    path = Path(run_dir) / "progress.json"
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    step = data.get("step") if isinstance(data, dict) else None
    if not isinstance(step, int) or isinstance(step, bool):
        return None
    curve = data.get("curve")
    validations = len(curve) if isinstance(curve, list) else None
    return {"step": step, "validations": validations}


def advanced(before: Optional[dict], after: Optional[dict]) -> bool:
    if after is None:
        return False
    if before is None:
        return True
    if after["step"] > before["step"]:
        return True
    if after["step"] == before["step"] and None not in (after["validations"], before["validations"]):
        return after["validations"] > before["validations"]
    return False


def log_mentions_oom(log: Path) -> bool:
    try:
        with Path(log).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 65536))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return False
    return any(pattern in tail for pattern in OOM_PATTERNS)


def classify(state: str, exit_code: str, run_dir: Path, start_progress: Optional[dict],
             log: Optional[Path] = None) -> Tuple[str, Optional[dict]]:
    """Return ``(outcome, progress)`` for one finished array task.

    ``resume`` is returned only for a known pause (exit 75) or a recoverable
    Slurm state, with ``resume.pt`` and readable progress that moved forward.
    """
    run_dir = Path(run_dir)
    progress = read_progress(run_dir)
    if (run_dir / "COMPLETE.json").is_file():
        try:
            read_json(run_dir / "COMPLETE.json")
            return "completed", progress
        except (OSError, ValueError):
            return "failed", progress
    code = exit_code.split(":")[0] if exit_code else ""
    if state == "OUT_OF_MEMORY":
        return "failed_oom", progress
    if state == "CANCELLED":
        return "cancelled", progress
    if state == "COMPLETED":
        return "exit0_without_complete", progress
    paused = state == "FAILED" and code == str(PAUSE_EXIT)
    if paused or state in RECOVERABLE_STATES:
        if not (run_dir / "resume.pt").is_file():
            return "no_checkpoint", progress
        if progress is None:
            return "progress_unreadable", progress
        if not advanced(start_progress, progress):
            return "stalled", progress
        return "resume", progress
    if log is not None and log_mentions_oom(log):
        return "failed_oom", progress
    return "failed", progress


def _since(record: dict) -> str:
    created = _dt.datetime.fromisoformat(record["created"]) - _dt.timedelta(days=1)
    return created.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def array_argv(spec: dict, record: dict) -> List[str]:
    campaign = spec["campaign"]
    indices = ",".join(str(i) for i in record["indices"])
    parallel = max(1, min(spec["array_parallelism"], len(record["indices"])))
    env = (f"ALL,CGDA_PYTHON={spec['python']},CGDA_CODE_ROOT={spec['code_root']},"
           f"CGDA_CAMPAIGN={campaign},CGDA_WAVE={record['wave']}")
    return [
        "--job-name", record["token"],
        f"--array={indices}%{parallel}",
        "--output", f"{campaign}/logs/{record['token']}-%A_%a.out",
        "--chdir", campaign,
        f"--export={env}",
        *spec["sbatch_gpu_args"],
        str(Path(spec["code_root"]) / "scripts" / "compositional" / GPU_TEMPLATE),
    ]


def controller_argv(spec: dict, record: dict) -> List[str]:
    campaign = spec["campaign"]
    name = record["token"] + "-ctl"
    env = (f"ALL,CGDA_PYTHON={spec['python']},CGDA_CODE_ROOT={spec['code_root']},"
           f"CGDA_CAMPAIGN={campaign}")
    return [
        "--job-name", name,
        f"--dependency=afterany:{record['array']['job']}",
        "--output", f"{campaign}/logs/{name}-%j.out",
        "--chdir", campaign,
        f"--export={env}",
        *spec["sbatch_cpu_args"],
        str(Path(spec["code_root"]) / "scripts" / "compositional" / CPU_TEMPLATE),
        "--wave", str(record["wave"]),
        "--after", record["array"]["job"],
    ]


def _lookup(slurm: Slurm, name: str, since: str, polls: int,
             delay: float) -> Tuple[set, List[str]]:
    """Search Slurm for jobs named ``name``. Returns the jobs found and any lookup errors."""
    found: set = set()
    errors: List[str] = []
    for poll in range(max(1, polls)):
        try:
            found = slurm.find_by_name(name, since)
        except CLIENT_ERRORS as exc:
            errors.append(str(getattr(exc, "stderr", None) or exc).strip())
            found = set()
        if found:
            break
        if poll + 1 < polls:
            slurm.sleep(delay)
    return found, errors


def _submit_part(campaign: Path, slurm: Slurm, record: dict, part: str, name: str,
                 argv: List[str], reconcile_polls: int, reconcile_delay: float) -> None:
    """Drive one submission (array or controller) of a wave to a known job ID.

    Before any further ``sbatch``, a previous attempt is reconciled by name.
    A definite rejection is checked once and retried up to the attempt cap.
    An ambiguous attempt is searched for over the whole reconciliation
    window. If it is not found the wave is marked ``unresolved`` and nothing
    is resubmitted, because ``squeue``/``sacct`` may lag a job that does exist.
    """
    entry = record[part]
    path = wave_path(campaign, record["wave"])
    while entry["job"] is None:
        if entry["attempts"] > 0:
            definite = entry["state"] == "rejected"
            polls = 1 if definite else reconcile_polls
            found, errors = _lookup(slurm, name, _since(record), polls, reconcile_delay)
            if len(found) > 1:
                entry["state"] = "duplicate"
                record["problem"] = f"{part} submitted more than once: {sorted(found)}"
                write_json(path, record)
                append_event(campaign, "duplicate_submission", wave=record["wave"], part=part,
                             jobs=sorted(found))
                raise ChainError(record["problem"])
            if found:
                entry["job"] = found.pop()
                entry["state"] = "submitted"
                entry["reconciled"] = True
                write_json(path, record)
                append_event(campaign, "reconciled", wave=record["wave"], part=part, job=entry["job"])
                return
            if errors or not definite:
                waited = (polls - 1) * reconcile_delay
                reason = (f"Slurm lookup failed: {errors[-1]}" if errors and len(errors) == polls
                          else f"the last sbatch outcome was ambiguous and no job named {name} "
                               f"appeared within {waited:.0f} s")
                if not definite:
                    entry["state"] = "unresolved"
                record["problem"] = (f"{part} submission unresolved: {reason}. Check "
                                     f"`squeue --name {name}` and `sacct --name {name}`; then run "
                                     f"`chain.py arm --wave {record['wave']}` (adopts a job that "
                                     f"exists) or add `--confirm-absent` once it is certain no "
                                     f"job exists")
                write_json(path, record)
                append_event(campaign, "submission_unresolved", wave=record["wave"], part=part,
                             lookup_errors=errors)
                raise ChainError(record["problem"])
            if entry["attempts"] >= MAX_SUBMIT_ATTEMPTS:
                entry["state"] = "failed"
                record["problem"] = f"{part} submission failed {entry['attempts']} times"
                write_json(path, record)
                append_event(campaign, "submission_failed", wave=record["wave"], part=part)
                raise ChainError(record["problem"])
        entry["attempts"] += 1
        entry["state"] = "attempting"
        write_json(path, record)
        try:
            job = slurm.sbatch(argv)
        except SubmissionRejected as exc:
            entry["state"] = "rejected"
            entry["errors"].append({"time": now(), "error": str(exc)})
            write_json(path, record)
            append_event(campaign, "submission_rejected", wave=record["wave"], part=part,
                         error=str(exc))
            continue
        except SubmissionUnknown as exc:
            entry["state"] = "unknown"
            entry["errors"].append({"time": now(), "error": str(exc)})
            write_json(path, record)
            append_event(campaign, "submission_unknown", wave=record["wave"], part=part,
                         error=str(exc))
            continue
        entry["job"] = job
        entry["state"] = "submitted"
        write_json(path, record)
        append_event(campaign, "submitted", wave=record["wave"], part=part, job=job, name=name)


def _confirm_absent(campaign: Path, slurm: Slurm, record: dict, part: str, name: str) -> None:
    """Operator override for an ``unresolved`` submission: one more lookup, then allow a retry."""
    entry = record[part]
    if entry["state"] != "unresolved":
        return
    found, errors = _lookup(slurm, name, _since(record), 1, 0.0)
    if errors:
        raise ChainError(f"cannot confirm absence of {name}: Slurm lookup failed: {errors[-1]}")
    if found:
        return
    entry["state"] = "rejected"
    entry["errors"].append({"time": now(), "error": "operator confirmed no job exists"})
    record["problem"] = None
    write_json(wave_path(campaign, record["wave"]), record)
    append_event(campaign, "absence_confirmed", wave=record["wave"], part=part)


def ensure_wave(campaign: Path, slurm: Slurm, wave: int, run_names: Optional[List[str]] = None,
                reconcile_polls: int = RECONCILE_POLLS,
                reconcile_delay: float = RECONCILE_DELAY,
                confirm_absent: bool = False) -> dict:
    """Create (if ``run_names`` is given) and drive a wave to the ``armed`` state.

    Safe to call repeatedly: an existing record is resumed, never duplicated.
    ``confirm_absent`` lets an operator release an ``unresolved`` submission
    after checking Slurm by hand. A job found at that point is still adopted.
    """
    campaign = Path(campaign).resolve()
    spec = load_campaign(campaign)
    path = wave_path(campaign, wave)
    if run_names is not None and not path.exists():
        if wave >= spec["max_waves"]:
            raise ChainError(f"wave {wave} exceeds the cap of {spec['max_waves']} waves")
        by_name = runs_by_name(spec)
        unknown = [name for name in run_names if name not in by_name]
        if unknown or not run_names:
            raise ChainError(f"invalid run list for wave {wave}: {unknown or 'empty'}")
        ordered = [run for run in spec["runs"] if run["name"] in set(run_names)]
        token = f"cgda-{spec['campaign_id']}-w{wave:03d}-{secrets.token_hex(3)}"
        record = {
            "campaign_id": spec["campaign_id"],
            "wave": wave,
            "token": token,
            "created": now(),
            "runs": [run["name"] for run in ordered],
            "indices": [run["index"] for run in ordered],
            "start_progress": {run["name"]: read_progress(Path(run["run_dir"])) for run in ordered},
            "array": {"state": "intent", "job": None, "attempts": 0, "errors": []},
            "controller": {"state": "intent", "job": None, "attempts": 0, "errors": []},
            "outcomes": None,
            "handled": False,
            "problem": None,
        }
        if create_json_exclusive(path, record):
            append_event(campaign, "wave_intent", wave=wave, token=token, runs=record["runs"])
    if not path.exists():
        raise ChainError(f"wave {wave} has no record")
    record = read_record(path)
    if record["campaign_id"] != spec["campaign_id"]:
        raise ChainError("wave record belongs to another campaign")
    if run_names is not None and sorted(record["runs"]) != sorted(run_names):
        raise ChainError(f"wave {wave} already exists with a different run list")
    try:
        for part, name in (("array", record["token"]), ("controller", record["token"] + "-ctl")):
            if record[part]["state"] in ("duplicate", "failed"):
                raise ChainError(record.get("problem") or f"{part} submission needs attention")
            if record[part]["state"] == "unresolved" and confirm_absent:
                _confirm_absent(campaign, slurm, record, part, name)
            argv = array_argv(spec, record) if part == "array" else controller_argv(spec, record)
            _submit_part(campaign, slurm, record, part, name, argv, reconcile_polls, reconcile_delay)
        if (record.get("problem") or "").startswith(("array submission", "controller submission")):
            record["problem"] = None
            write_json(path, record)
            append_event(campaign, "submission_resolved", wave=wave)
    finally:
        write_status(campaign)
    return record


def gather_accounting(slurm: Slurm, array_job: str, indices: List[int],
                      polls: int = ACCOUNTING_POLLS,
                      delay: float = ACCOUNTING_DELAY) -> Dict[int, Tuple[str, str]]:
    """Poll ``sacct`` until every task is terminal. A client error counts as an incomplete poll."""
    accounting: Dict[int, Tuple[str, str]] = {}
    last_error = None
    for poll in range(polls):
        try:
            accounting = slurm.array_accounting(array_job)
        except CLIENT_ERRORS as exc:
            last_error = str(getattr(exc, "stderr", None) or exc).strip()
        else:
            if all(i in accounting and accounting[i][0] in TERMINAL_STATES for i in indices):
                return accounting
        if poll + 1 < polls:
            slurm.sleep(delay)
    missing = [i for i in indices if i not in accounting or accounting[i][0] not in TERMINAL_STATES]
    detail = f"; last sacct error: {last_error}" if last_error else ""
    raise ChainError(f"incomplete Slurm accounting for array {array_job} tasks {missing} after "
                     f"{polls} polls{detail}")


def handle_wave(campaign: Path, slurm: Slurm, wave: int, after: str,
                accounting_polls: int = ACCOUNTING_POLLS,
                accounting_delay: float = ACCOUNTING_DELAY,
                reconcile_polls: int = RECONCILE_POLLS,
                reconcile_delay: float = RECONCILE_DELAY,
                lock_timeout: float = LOCK_TIMEOUT) -> dict:
    """Classify a finished wave and arm the next one. Idempotent under the lock."""
    campaign = Path(campaign).resolve()
    spec = load_campaign(campaign)
    paths = campaign_paths(campaign)
    with Lock(paths["lock"], timeout=lock_timeout, sleep=slurm.sleep):
        path = wave_path(campaign, wave)
        record = read_record(path)
        if record.get("handled"):
            append_event(campaign, "already_handled", wave=wave)
            write_status(campaign)
            return record
        if record["array"]["job"] != str(after):
            raise ChainError(f"array {after} does not match wave {wave} record "
                             f"{record['array']['job']}")
        if record["outcomes"] is None:
            accounting = gather_accounting(slurm, str(after), record["indices"],
                                           accounting_polls, accounting_delay)
            by_name = runs_by_name(spec)
            outcomes = {}
            for name, index in zip(record["runs"], record["indices"]):
                run = by_name[name]
                state, exit_code = accounting[index]
                log = paths["logs"] / f"{record['token']}-{after}_{index}.out"
                outcome, progress = classify(state, exit_code, Path(run["run_dir"]),
                                             record["start_progress"].get(name), log)
                if outcome == "resume" and wave + 1 >= spec["max_waves"]:
                    outcome = "cap_reached"
                outcomes[name] = {"outcome": outcome, "slurm_state": state,
                                  "exit_code": exit_code, "progress": progress,
                                  "log": str(log)}
            record["outcomes"] = outcomes
            record["classified"] = now()
            write_json(path, record)
            append_event(campaign, "classified", wave=wave,
                         outcomes={k: v["outcome"] for k, v in outcomes.items()})
        remaining = [name for name, item in record["outcomes"].items()
                     if item["outcome"] == "resume"]
        if remaining:
            try:
                nxt = ensure_wave(campaign, slurm, wave + 1, remaining,
                                  reconcile_polls, reconcile_delay)
            except Exception as exc:
                record["problem"] = f"next wave not armed: {exc}"
                write_json(path, record)
                write_status(campaign)
                raise
            record["next_wave"] = nxt["wave"]
        record["problem"] = None
        record["handled"] = True
        record["handled_at"] = now()
        write_json(path, record)
        append_event(campaign, "handled", wave=wave, resumed=remaining)
        write_status(campaign)
        return record


def list_waves(campaign: Path) -> Tuple[List[dict], List[dict]]:
    """Readable wave records, and a problem entry for every unreadable one."""
    chain = campaign_paths(campaign)["chain"]
    waves, problems = [], []
    for path in sorted(chain.glob("wave_[0-9][0-9][0-9].json")):
        try:
            waves.append(read_record(path))
        except ChainError as exc:
            problems.append({"wave": int(path.stem.split("_")[1]), "problem": str(exc)})
    return waves, problems


def failure_markers(campaign: Path, wave: int) -> List[dict]:
    """Controller failures recorded outside the wave record (lock unavailable or Slurm probe)."""
    directory = campaign_paths(campaign)["failures"]
    markers = []
    for path in sorted(directory.glob(f"wave_{wave:03d}-*.json")):
        try:
            markers.append(read_record(path))
        except ChainError as exc:
            markers.append({"problem": str(exc)})
    return markers


def write_failure_marker(campaign: Path, wave: int, problem: str, **fields) -> Path:
    directory = campaign_paths(campaign)["failures"]
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"wave_{wave:03d}-{stamp}-{os.getpid()}-{secrets.token_hex(2)}.json"
    write_json(path, {"time": now(), "wave": wave, "problem": problem, **fields})
    return path


def summarize(campaign: Path) -> dict:
    campaign = Path(campaign)
    spec = load_campaign(campaign)
    waves, problems = list_waves(campaign)
    runs = {run["name"]: {"status": "not_submitted", "wave": None} for run in spec["runs"]}
    active = False
    for record in waves:
        if record.get("problem"):
            problems.append({"wave": record["wave"], "problem": record["problem"]})
        if not record.get("handled"):
            for marker in failure_markers(campaign, record["wave"]):
                problems.append({"wave": record["wave"], "problem": marker.get("problem")})
        submitted = record["array"]["job"] is not None and record["controller"]["job"] is not None
        if not record.get("handled"):
            if submitted:
                active = True
            elif not record.get("problem"):
                problems.append({"wave": record["wave"],
                                 "problem": "wave not fully submitted; run `arm`"})
        for name in record["runs"]:
            item = (record.get("outcomes") or {}).get(name)
            runs[name] = {
                "wave": record["wave"],
                "array_job": record["array"]["job"],
                "status": item["outcome"] if item else ("running" if submitted else "submitting"),
                "slurm_state": item["slurm_state"] if item else None,
                "exit_code": item["exit_code"] if item else None,
                "progress": item["progress"] if item else read_progress(
                    Path(runs_by_name(spec)[name]["run_dir"])),
            }
    for name, item in runs.items():
        if item["status"] == "resume" and not active:
            problems.append({"run": name, "problem": "resumable run without an armed wave"})
    attention = sorted(name for name, item in runs.items() if item["status"] in ATTENTION_OUTCOMES)
    if all(item["status"] == "completed" for item in runs.values()):
        state = "completed"
    elif active and not problems:
        state = "running_with_attention" if attention else "running"
    elif not waves and not problems:
        state = "initialised"
    else:
        state = "needs_attention"
    return {"campaign_id": spec["campaign_id"], "state": state, "updated": now(),
            "waves": len(waves), "max_waves": spec["max_waves"], "attention": attention,
            "problems": problems, "runs": runs}


def write_status(campaign: Path) -> dict:
    status = summarize(campaign)
    write_json(campaign_paths(campaign)["status"], status)
    return status


def trainer_argv(spec: dict, run: dict, python: str) -> List[str]:
    return [python, str(Path(spec["code_root"]) / TRAINER),
            "--config", run["config"], "--run-dir", run["run_dir"],
            "--device", spec["device"],
            "--deadline-seconds", str(spec["deadline_seconds"]),
            *run.get("trainer_args", [])]


def gpu_name(runner: Callable[[Sequence[str]], str]) -> Optional[str]:
    try:
        return runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def prepare_task(campaign: Path, wave: int, task: int,
                 runner: Callable[[Sequence[str]], str] = run_command,
                 python: str = sys.executable) -> Tuple[Optional[List[str]], int, Optional[int]]:
    """Validate a GPU array task and return ``(argv, exit_code, lock_fd)``.

    ``argv`` is None when the task must exit with ``exit_code`` instead of
    running the trainer. The returned lock descriptor is inheritable, so the
    per-run lock is held by the trainer after ``exec``.
    """
    campaign = Path(campaign).resolve()
    spec = load_campaign(campaign)
    record = read_record(wave_path(campaign, wave))
    if task not in record["indices"]:
        raise ChainError(f"task {task} is not part of wave {wave}")
    run = spec["runs"][task]
    if run["index"] != task:
        raise ChainError("campaign manifest indices are inconsistent")
    if sha256_file(Path(run["config"])) != run["config_sha256"]:
        raise ChainError(f"frozen config for {run['name']} changed after init")
    run_dir = Path(run["run_dir"])
    if (run_dir / "COMPLETE.json").is_file():
        print(f"{run['name']} already complete; nothing to run", flush=True)
        return None, 0, None
    paths = campaign_paths(campaign)
    fd = os.open(str(paths["locks"] / f"{run['name']}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        print(f"{run['name']} is already running in another job; refusing duplicate", flush=True)
        return None, DUPLICATE_EXIT, None
    os.set_inheritable(fd, True)
    argv = trainer_argv(spec, run, python)
    segment_dir = paths["segments"] / run["name"]
    segment_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    write_json(segment_dir / f"{job}_{task}.json", {
        "run": run["name"], "wave": wave, "token": record["token"], "array_job": job,
        "task": task, "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_name(runner), "started": now(), "start_progress": read_progress(run_dir),
        "config_sha256": run["config_sha256"], "argv": argv,
    })
    print(json.dumps({"run": run["name"], "wave": wave, "argv": argv}), flush=True)
    return argv, 0, fd


def cmd_init(args) -> int:
    frozen = init_campaign(args.spec, args.campaign)
    print(json.dumps({"campaign": frozen["campaign"], "campaign_id": frozen["campaign_id"],
                      "runs": [r["name"] for r in frozen["runs"]]}, indent=2))
    return 0


def cmd_start(args) -> int:
    campaign = Path(args.campaign).resolve()
    spec = load_campaign(campaign)
    names = [run["name"] for run in spec["runs"]]
    if args.dry_run:
        record = {"wave": 0, "token": f"cgda-{spec['campaign_id']}-w000-DRYRUN",
                  "indices": [run["index"] for run in spec["runs"]],
                  "array": {"job": "ARRAY_JOB_ID"}}
        print(" ".join(["sbatch", "--parsable", *array_argv(spec, record)]))
        print(" ".join(["sbatch", "--parsable", *controller_argv(spec, record)]))
        for run in spec["runs"]:
            print(" ".join(trainer_argv(spec, run, spec["python"])))
        return 0
    with Lock(campaign_paths(campaign)["lock"]):
        ensure_wave(campaign, Slurm(), 0, names)
    print(json.dumps(summarize(campaign), indent=2))
    return 0


def cmd_arm(args) -> int:
    campaign = Path(args.campaign).resolve()
    with Lock(campaign_paths(campaign)["lock"]):
        ensure_wave(campaign, Slurm(), args.wave, confirm_absent=args.confirm_absent)
    print(json.dumps(summarize(campaign), indent=2))
    return 0


def record_controller_failure(campaign: Path, wave: int, error: str,
                              sleep: Callable[[float], None] = time.sleep,
                              lock_timeout: float = FAILURE_LOCK_TIMEOUT,
                              message: Optional[str] = None) -> str:
    """Make a failed controller visible in ``status.json``.

    Under the campaign lock the problem is written on the wave record. If the
    lock or the record is unavailable, a failure marker is written instead;
    ``summarize`` reports markers of any unhandled wave. A more specific
    problem already written for the same error (``next wave not armed``) is
    kept. Returns where the failure was recorded.
    """
    problem = f"controller failed: {error}"
    where = "marker"
    try:
        with Lock(campaign_paths(campaign)["lock"], timeout=lock_timeout, sleep=sleep):
            path = wave_path(campaign, wave)
            record = read_record(path)
            if not record.get("handled"):
                if not (message and message in (record.get("problem") or "")):
                    record["problem"] = problem
                record.setdefault("controller_failures", []).append({"time": now(), "error": error})
                write_json(path, record)
            where = "wave"
    except Exception as exc:
        write_failure_marker(campaign, wave, problem, recording_error=str(exc))
    try:
        append_event(campaign, "controller_error", wave=wave, error=error, recorded=where)
    except OSError:
        pass
    write_status(campaign)
    return where


def run_controller(campaign: Path, wave: int, after: str, slurm: Optional[Slurm] = None,
                   **options) -> int:
    """Handle one wave and turn every failure into a visible ``needs_attention`` state.

    ``SIGTERM`` (sent by Slurm at the controller's time limit) is converted to
    an exception so that the failure can still be recorded before ``SIGKILL``.
    """
    campaign = Path(campaign).resolve()
    slurm = slurm or Slurm()

    def terminate(signum, frame):
        raise ControllerTerminated(f"received signal {signum} before the wave was handled")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            handle_wave(campaign, slurm, wave, after, **options)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            record_controller_failure(campaign, wave, error, sleep=slurm.sleep, message=str(exc))
            print(f"controller stopped without submitting further work: {error}", file=sys.stderr)
            return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(summarize(campaign), indent=2))
    return 0


def cmd_controller(args) -> int:
    return run_controller(Path(args.campaign), args.wave, args.after)


def probe_controllers(campaign: Path, slurm: Slurm) -> List[dict]:
    """Flag every unhandled wave whose controller job has ended in Slurm.

    A controller that was killed (time limit, node failure, cancellation)
    cannot record its own failure. Its wave would otherwise look ``running``
    forever. A wave is flagged only when Slurm reports a terminal state for
    the controller job and the record, re-read afterwards, is still unhandled.
    """
    flagged = []
    waves, _ = list_waves(campaign)
    for record in waves:
        job = record["controller"].get("job")
        if record.get("handled") or job is None:
            continue
        state = slurm.job_state(str(job))
        if state not in TERMINAL_STATES:
            continue
        try:
            current = read_record(wave_path(campaign, record["wave"]))
        except ChainError:
            current = record
        if current.get("handled"):
            continue
        if any(m.get("controller_job") == str(job) for m in failure_markers(campaign, record["wave"])):
            continue
        problem = (f"controller job {job} ended {state} without handling wave {record['wave']}; "
                   f"rerun `chain.py controller --wave {record['wave']} --after "
                   f"{record['array']['job']}`")
        write_failure_marker(campaign, record["wave"], problem, controller_job=str(job),
                             slurm_state=state)
        append_event(campaign, "controller_lost", wave=record["wave"], job=str(job), state=state)
        flagged.append({"wave": record["wave"], "job": str(job), "state": state})
    return flagged


def cmd_status(args) -> int:
    campaign = Path(args.campaign).resolve()
    if not args.no_slurm:
        try:
            probe_controllers(campaign, Slurm())
        except Exception as exc:
            print(f"warning: Slurm probe of controller jobs failed: {exc}", file=sys.stderr)
    status = write_status(campaign)
    print(json.dumps(status, indent=2))
    return 0 if status["state"] != "needs_attention" else 1


def cmd_exec_task(args) -> int:
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        raise ChainError("exec-task must run inside a Slurm array task")
    argv, code, _fd = prepare_task(Path(args.campaign), args.wave, int(task))
    if argv is None:
        return code
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(argv[0], argv)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="freeze a campaign spec into a new campaign directory")
    p.add_argument("spec", type=Path)
    p.add_argument("--campaign", type=Path, required=True)
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("start", help="submit wave 0 and its controller (idempotent)")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="print sbatch and trainer commands only")
    p.set_defaults(func=cmd_start)
    p = sub.add_parser("arm", help="finish or reconcile an existing wave's submissions")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--wave", type=int, required=True)
    p.add_argument("--confirm-absent", action="store_true",
                   help="after checking squeue and sacct by hand, allow resubmitting an "
                        "unresolved submission; a job found by name is still adopted")
    p.set_defaults(func=cmd_arm)
    p = sub.add_parser("controller", help="classify a finished wave and arm the next one")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--wave", type=int, required=True)
    p.add_argument("--after", required=True)
    p.set_defaults(func=cmd_controller)
    p = sub.add_parser("status", help="recompute and print status.json")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--no-slurm", action="store_true",
                   help="skip the sacct check for controller jobs that ended without handling")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("exec-task", help="inside a GPU array task: validate and exec the trainer")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--wave", type=int, required=True)
    p.set_defaults(func=cmd_exec_task)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ChainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
