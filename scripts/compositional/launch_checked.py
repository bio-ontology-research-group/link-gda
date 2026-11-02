#!/usr/bin/env python3
"""Gate the compositional pilot campaign on verified GPU tests and prepared data.

The launcher checks at least two JUnit reports and one prepared ``identity.json``,
binds the campaign spec to that prepared fold, and only when every check passes
calls ``chain.py init`` and ``chain.py start`` with the current interpreter. Every
outcome, including refusals, is written atomically to the report file.

Exit status: 0 launched, 1 a gate check failed, 2 the launch was refused or a
chain command failed. Nothing is submitted unless the status is 0.
"""

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional, Sequence

SCHEMA = 1
MIN_TESTS = 5
MIN_REPORTS = 2
CHAIN = Path(__file__).resolve().parent / "chain.py"
PREPARED_OUTPUTS = ("graph.npz", "vocabulary.json", "profiles.json", "candidates.txt",
                    "training.tsv", "validation.tsv", "test.tsv")
JUNIT_COUNTERS = ("tests", "failures", "errors", "skipped")
OUTCOME_TAGS = {"failure": "failures", "error": "errors", "skipped": "skipped"}

EXIT_LAUNCHED = 0
EXIT_GATE_FAILED = 1
EXIT_REFUSED = 2


class Refusal(Exception):
    """The campaign cannot be launched safely from the current state."""


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


def write_report(path: Path, data: dict) -> None:
    """Write the report atomically: temporary file, fsync, rename, directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def read_previous(path: Path):
    """Return the previous report and a note when it exists but cannot be used."""
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"previous report unreadable: {exc}"
    if not isinstance(data, dict):
        return None, "previous report is not a JSON object"
    return data, None


def _count(element, name: str):
    value = element.get(name)
    if value is None:
        return None
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"attribute {name}={value!r} is not an integer")
    if number < 0:
        raise ValueError(f"attribute {name}={value!r} is negative")
    return number


def check_junit(path: Path) -> dict:
    """Count test cases and outcomes from leaf elements, never from aggregate totals."""
    path = Path(path)
    result = {"path": str(path), "ok": False, "problems": []}
    if not path.is_file():
        result["problems"].append("JUnit report missing")
        return result
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        result["problems"].append(f"malformed JUnit XML: {exc}")
        return result
    if root.tag not in ("testsuites", "testsuite"):
        result["problems"].append(f"unexpected JUnit root element {root.tag!r}")
        return result
    leaves = [s for s in root.iter("testsuite") if s.find(".//testsuite") is None]
    counted = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failing = []
    for case in root.iter("testcase"):
        counted["tests"] += 1
        for child in case:
            key = OUTCOME_TAGS.get(child.tag)
            if key:
                counted[key] += 1
                failing.append({"case": f"{case.get('classname', '')}::{case.get('name', '')}",
                                "outcome": child.tag})
    declared = {name: 0 for name in JUNIT_COUNTERS}
    try:
        for suite in leaves:
            for name in JUNIT_COUNTERS:
                value = _count(suite, name)
                if value is not None:
                    declared[name] += value
    except ValueError as exc:
        result["problems"].append(f"malformed JUnit counters: {exc}")
        return result
    result.update({
        "tests": counted["tests"], "failures": counted["failures"], "errors": counted["errors"],
        "skipped": counted["skipped"], "declared_leaf_totals": declared,
        "suites": [{"name": s.get("name"), "hostname": s.get("hostname"),
                    "timestamp": s.get("timestamp")} for s in leaves],
        "non_passing_cases": failing,
    })
    if not leaves:
        result["problems"].append("no testsuite element")
    if counted["tests"] < MIN_TESTS:
        result["problems"].append(f"{counted['tests']} test cases, fewer than {MIN_TESTS}")
    for name in ("failures", "errors", "skipped"):
        if counted[name] or declared[name]:
            result["problems"].append(
                f"{name}: {counted[name]} counted, {declared[name]} declared; zero required")
    if any(s.get("tests") is not None for s in leaves) and declared["tests"] != counted["tests"]:
        result["problems"].append(
            f"declared {declared['tests']} tests but found {counted['tests']} test cases")
    result["ok"] = not result["problems"]
    return result


def check_junit_set(paths: Sequence[Path]) -> dict:
    reports = [check_junit(p) for p in paths]
    problems = []
    if len(paths) < MIN_REPORTS:
        problems.append(f"{len(paths)} JUnit reports given, {MIN_REPORTS} required")
    resolved = [str(Path(p).resolve()) for p in paths]
    if len(set(resolved)) != len(resolved):
        problems.append("the same JUnit report path was given more than once")
    digests = [r["sha256"] for r in reports if "sha256" in r]
    if len(set(digests)) != len(digests):
        problems.append("two JUnit reports have identical bytes; each run must report separately")
    ok = not problems and all(r["ok"] for r in reports)
    return {"ok": ok, "problems": problems, "reports": reports}


def check_prepared(identity_path: Path) -> dict:
    """Verify every recorded prepared output by size and streamed SHA-256."""
    identity_path = Path(identity_path)
    result = {"identity": str(identity_path), "ok": False, "problems": []}
    if not identity_path.is_file():
        result["problems"].append("identity.json missing")
        return result
    result["identity_sha256"] = sha256_file(identity_path)
    try:
        identity = json.loads(identity_path.read_text())
    except (OSError, ValueError) as exc:
        result["problems"].append(f"identity.json unreadable: {exc}")
        return result
    if not isinstance(identity, dict) or not isinstance(identity.get("outputs"), dict):
        result["problems"].append("identity.json lacks an outputs mapping")
        return result
    root = identity_path.resolve().parent
    result["prepared_dir"] = str(root)
    result["fold"] = identity.get("fold")
    result["schema"] = identity.get("schema")
    outputs = identity["outputs"]
    if set(outputs) != set(PREPARED_OUTPUTS):
        result["problems"].append(
            f"outputs listed {sorted(outputs)}; expected {sorted(PREPARED_OUTPUTS)}")
    actual = {}
    for name in sorted(outputs):
        record = outputs[name]
        entry: dict = {"recorded_bytes": None, "recorded_sha256": None}
        actual[name] = entry
        if not isinstance(record, dict) or "/" in name or name in ("", ".", ".."):
            result["problems"].append(f"output record {name!r} malformed")
            continue
        entry["recorded_bytes"] = record.get("bytes")
        entry["recorded_sha256"] = record.get("sha256")
        path = root / name
        if path.is_symlink() or not path.is_file():
            entry["exists"] = False
            result["problems"].append(f"{name} missing or not a regular file")
            continue
        entry["exists"] = True
        entry["bytes"] = path.stat().st_size
        if entry["bytes"] != record.get("bytes"):
            result["problems"].append(
                f"{name} has {entry['bytes']} bytes, identity records {record.get('bytes')}")
            continue
        entry["sha256"] = sha256_file(path)
        if entry["sha256"] != record.get("sha256"):
            result["problems"].append(f"{name} SHA-256 differs from identity record")
    result["outputs"] = actual
    graph = identity.get("graph")
    if not isinstance(graph, dict) or graph.get("file") != "graph.npz":
        result["problems"].append("identity.json lacks a graph record naming graph.npz")
    else:
        shape = graph.get("shape")
        result["graph"] = {"file": graph["file"], "shape": shape,
                           "exists": actual.get("graph.npz", {}).get("exists", False)}
        if not result["graph"]["exists"]:
            result["problems"].append("graph.npz does not exist")
        if not (isinstance(shape, list) and len(shape) == 2 and shape[1] == 3
                and isinstance(shape[0], int) and shape[0] > 0):
            result["problems"].append(f"graph shape {shape!r} is not a non-empty [N, 3]")
    result["ok"] = not result["problems"]
    return result


def check_spec(spec_path: Path, prepared_dir: Optional[str]) -> dict:
    """Hash the spec and its run configs and confirm every run trains on the prepared fold."""
    spec_path = Path(spec_path)
    result = {"path": str(spec_path.resolve()), "ok": False, "problems": [], "runs": []}
    if not spec_path.is_file():
        result["problems"].append("campaign spec missing")
        return result
    result["sha256"] = sha256_file(spec_path)
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, ValueError) as exc:
        result["problems"].append(f"campaign spec unreadable: {exc}")
        return result
    runs = spec.get("runs") if isinstance(spec, dict) else None
    if not isinstance(runs, list) or not runs:
        result["problems"].append("campaign spec lists no runs")
        return result
    for run in runs:
        if not isinstance(run, dict) or "name" not in run or "config" not in run:
            result["problems"].append(f"run entry {run!r} lacks name or config")
            continue
        source = Path(run["config"])
        if not source.is_absolute():
            source = spec_path.resolve().parent / source
        entry = {"name": run["name"], "config": str(source)}
        result["runs"].append(entry)
        if not source.is_file():
            result["problems"].append(f"config for {run['name']} missing: {source}")
            continue
        entry["sha256"] = sha256_file(source)
        try:
            config = json.loads(source.read_text())
        except (OSError, ValueError) as exc:
            result["problems"].append(f"config for {run['name']} unreadable: {exc}")
            continue
        data_dir = config.get("data_dir") if isinstance(config, dict) else None
        if not isinstance(data_dir, str):
            result["problems"].append(f"config for {run['name']} lacks data_dir")
            continue
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = source.resolve().parent / resolved
        entry["data_dir"] = str(Path(os.path.realpath(resolved)))
        if prepared_dir is not None and entry["data_dir"] != prepared_dir:
            result["problems"].append(
                f"{run['name']} trains on {entry['data_dir']}, not the verified {prepared_dir}")
    result["ok"] = not result["problems"]
    return result


def fingerprint(junit: dict, prepared: dict, spec: dict) -> str:
    material = {
        "junit": sorted(r.get("sha256", "") for r in junit["reports"]),
        "identity": prepared.get("identity_sha256"),
        "outputs": {k: v.get("sha256") for k, v in sorted(prepared.get("outputs", {}).items())},
        "spec": spec.get("sha256"),
        "configs": [[r["name"], r.get("sha256")] for r in spec["runs"]],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def frozen_campaign(campaign: Path):
    path = campaign / "campaign.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Refusal(f"{path} unreadable: {exc}")
    if not isinstance(data, dict):
        raise Refusal(f"{path} is not a JSON object")
    return data


def check_frozen_matches(frozen: dict, spec: dict) -> None:
    """Refuse a frozen campaign that was not frozen from exactly this spec and these configs."""
    if frozen.get("spec_sha256") != spec["sha256"]:
        raise Refusal("existing campaign was frozen from a different spec (SHA-256 differs)")
    if frozen.get("spec_source") != spec["path"]:
        raise Refusal(f"existing campaign was frozen from {frozen.get('spec_source')}, "
                      f"not {spec['path']}")
    frozen_runs = frozen.get("runs") or []
    if [r.get("name") for r in frozen_runs] != [r["name"] for r in spec["runs"]]:
        raise Refusal("existing campaign lists different runs")
    for run, current in zip(frozen_runs, spec["runs"]):
        if run.get("config_sha256") != current.get("sha256"):
            raise Refusal(f"frozen config for {run['name']} differs from {current['config']}")
        copy = Path(run.get("config", ""))
        if not copy.is_file() or sha256_file(copy) != run.get("config_sha256"):
            raise Refusal(f"frozen config copy for {run['name']} altered or missing")


def default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True)


def call(runner: Callable, argv: List[str]) -> dict:
    started = now()
    try:
        done = runner(argv)
        record = {"argv": argv, "returncode": done.returncode,
                  "stdout": done.stdout, "stderr": done.stderr}
    except OSError as exc:
        record = {"argv": argv, "returncode": None, "stdout": "", "stderr": f"cannot execute: {exc}"}
    record["started"] = started
    record["finished"] = now()
    try:
        record["result"] = json.loads(record["stdout"]) if record["stdout"] else None
    except ValueError:
        record["result"] = None
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--junit", type=Path, action="append", required=True,
                        help="JUnit XML report of one GPU verification run; give at least two")
    parser.add_argument("--prepared", type=Path, required=True,
                        help="identity.json of the prepared fold the campaign trains on")
    parser.add_argument("--spec", type=Path, required=True, help="campaign spec for chain.py init")
    parser.add_argument("--campaign", type=Path, required=True, help="campaign directory")
    parser.add_argument("--report", type=Path, required=True, help="gate report JSON (rewritten)")
    return parser


def launch(args, runner: Callable) -> int:
    report_path = Path(args.report).resolve()
    campaign = Path(os.path.realpath(args.campaign))
    previous, note = read_previous(report_path)
    binding = previous.get("binding") if previous else None
    report = {
        "schema": SCHEMA,
        "created": now(),
        "updated": now(),
        "state": "checking",
        "problems": [],
        "notes": [note] if note else [],
        "campaign": str(campaign),
        "launcher": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "python": sys.executable,
        "commands": [],
        "binding": binding,
    }

    def publish(state: str, **fields) -> None:
        report["state"] = state
        report["updated"] = now()
        report.update(fields)
        write_report(report_path, report)

    junit = check_junit_set(args.junit)
    prepared = check_prepared(args.prepared)
    spec = check_spec(args.spec, prepared.get("prepared_dir"))
    report["checks"] = {"junit": junit, "prepared": prepared, "spec": spec}
    report["gate_fingerprint"] = fingerprint(junit, prepared, spec)
    for label, check in (("junit", junit), ("prepared", prepared), ("spec", spec)):
        report["problems"] += [f"{label}: {p}" for p in check["problems"]]
        if label == "junit":
            for item in check["reports"]:
                report["problems"] += [f"junit {item['path']}: {p}" for p in item["problems"]]
    if report["problems"]:
        publish("failed", stage="gate")
        return EXIT_GATE_FAILED
    publish("passed", stage="gate")

    try:
        if binding is not None:
            if binding.get("campaign") != str(campaign):
                raise Refusal(f"report already binds campaign {binding.get('campaign')}")
            if binding.get("gate_fingerprint") != report["gate_fingerprint"]:
                raise Refusal("report binds this campaign to a different gate result; "
                              "a changed gate needs a new campaign and report")
        frozen = frozen_campaign(campaign)
        if frozen is None:
            if campaign.exists() and any(campaign.iterdir()):
                raise Refusal(f"{campaign} is non-empty but holds no campaign.json")
            if binding is not None and binding.get("campaign_id"):
                raise Refusal(f"campaign {binding['campaign_id']} was initialised here and is gone; "
                              "its submissions are unknown, so it is not recreated")
        else:
            if binding is None:
                raise Refusal(f"{campaign} already exists and was not initialised by this gate")
            if binding.get("campaign_id") not in (None, frozen.get("campaign_id")):
                raise Refusal(f"campaign id {frozen.get('campaign_id')} differs from bound "
                              f"{binding['campaign_id']}")
            check_frozen_matches(frozen, spec)
    except Refusal as exc:
        report["problems"].append(f"refused: {exc}")
        publish("refused", stage="campaign")
        return EXIT_REFUSED

    chain = {"path": str(CHAIN), "sha256": sha256_file(CHAIN) if CHAIN.is_file() else None}
    report["chain"] = chain
    if chain["sha256"] is None:
        report["problems"].append(f"chain.py not found at {CHAIN}")
        publish("failed", stage="init")
        return EXIT_REFUSED

    if frozen is None:
        report["binding"] = {"campaign": str(campaign), "gate_fingerprint": report["gate_fingerprint"],
                             "spec_sha256": spec["sha256"], "campaign_id": None, "bound": now()}
        publish("initialising", stage="init")
        record = call(runner, [sys.executable, str(CHAIN), "init", spec["path"],
                               "--campaign", str(campaign)])
        report["commands"].append(record)
        if record["returncode"] != 0:
            report["problems"].append(f"chain init exited {record['returncode']}")
            publish("failed", stage="init")
            return EXIT_REFUSED
        try:
            frozen = frozen_campaign(campaign)
            if frozen is None:
                raise Refusal("chain init succeeded but campaign.json is absent")
            check_frozen_matches(frozen, spec)
        except Refusal as exc:
            report["problems"].append(f"refused: {exc}")
            publish("failed", stage="init")
            return EXIT_REFUSED
    else:
        report["notes"].append(f"reusing frozen campaign {frozen.get('campaign_id')}")
    report["binding"]["campaign_id"] = frozen.get("campaign_id")
    report["frozen_campaign"] = {"campaign_id": frozen.get("campaign_id"),
                                 "sha256": sha256_file(campaign / "campaign.json")}
    publish("starting", stage="start")

    record = call(runner, [sys.executable, str(CHAIN), "start", "--campaign", str(campaign)])
    report["commands"].append(record)
    if record["returncode"] != 0:
        report["problems"].append(f"chain start exited {record['returncode']}")
        publish("failed", stage="start")
        return EXIT_REFUSED
    report["binding"]["launched"] = report["binding"].get("launched") or now()
    publish("launched", stage="start", start_result=record["result"])
    return EXIT_LAUNCHED


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Callable] = None) -> int:
    args = build_parser().parse_args(argv)
    lock_path = Path(args.report).resolve().with_name(Path(args.report).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"error: another launcher holds {lock_path}", file=sys.stderr)
            return EXIT_REFUSED
        code = launch(args, runner or default_runner)
    report = json.loads(Path(args.report).resolve().read_text())
    print(json.dumps({"state": report["state"], "report": str(Path(args.report).resolve()),
                      "problems": report["problems"]}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
