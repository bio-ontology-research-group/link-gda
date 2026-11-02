import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "compositional"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chain = _load("compositional_chain", SCRIPTS / "chain.py")
launcher = _load("compositional_launch_checked", SCRIPTS / "launch_checked.py")

OUTPUTS = launcher.PREPARED_OUTPUTS


def junit_xml(cases=6, failures=0, errors=0, skipped=0, wrap=True, declared=None, host="node"):
    body = []
    for i in range(cases):
        inner = ""
        if i < failures:
            inner = '<failure message="boom">trace</failure>'
        elif i < failures + errors:
            inner = '<error message="err">trace</error>'
        elif i < failures + errors + skipped:
            inner = '<skipped message="no cuda" type="pytest.skip">skip</skipped>'
        body.append(f'<testcase classname="tests.test_gpu" name="test_{i}" time="0.1">{inner}</testcase>')
    totals = declared or {"tests": cases, "failures": failures, "errors": errors, "skipped": skipped}
    attrs = " ".join(f'{k}="{v}"' for k, v in totals.items())
    suite = f'<testsuite name="pytest" hostname="{host}" {attrs}>{"".join(body)}</testsuite>'
    if wrap:
        return f'<?xml version="1.0" encoding="utf-8"?><testsuites {attrs}>{suite}</testsuites>'
    return f'<?xml version="1.0" encoding="utf-8"?>{suite}'


class Setup:
    def __init__(self, tmp):
        self.tmp = tmp
        self.junit = [tmp / "gpu" / "v100.xml", tmp / "gpu" / "a100.xml"]
        self.junit[0].parent.mkdir()
        self.junit[0].write_text(junit_xml(host="v100"))
        self.junit[1].write_text(junit_xml(host="a100"))
        self.prepared = tmp / "prepared" / "fold0"
        self.prepared.mkdir(parents=True)
        outputs = {}
        for name in OUTPUTS:
            data = (f"{name}\n" * 3).encode()
            (self.prepared / name).write_bytes(data)
            outputs[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.identity = self.prepared / "identity.json"
        self.identity.write_text(json.dumps({
            "schema": 1, "fold": 0, "outputs": outputs,
            "graph": {"file": "graph.npz", "shape": [12, 3], "dtype": "int64"}}))
        specdir = tmp / "spec"
        (specdir / "configs").mkdir(parents=True)
        runs = []
        for method in ("transe", "betae"):
            config = specdir / "configs" / f"{method}.json"
            config.write_text(json.dumps({"method": method, "data_dir": "../../prepared/fold0",
                                          "steps": 500}))
            runs.append({"name": f"{method}-d100-s0", "config": f"configs/{method}.json"})
        self.spec = specdir / "campaign.json"
        self.spec.write_text(json.dumps({"code_root": str(tmp / "code"), "python": sys.executable,
                                         "max_waves": 2, "runs": runs}))
        self.campaign = tmp / "campaigns" / "pilot"
        self.report = tmp / "gate" / "report.json"

    def argv(self, junit=None):
        argv = []
        for path in (self.junit if junit is None else junit):
            argv += ["--junit", str(path)]
        return argv + ["--prepared", str(self.identity), "--spec", str(self.spec),
                       "--campaign", str(self.campaign), "--report", str(self.report)]

    def read(self):
        return json.loads(self.report.read_text())


class FakeRunner:
    """Runs ``chain.py init`` in process and stands in for ``chain.py start``."""

    def __init__(self, start_codes=(0,), init_code=None):
        self.calls = []
        self.start_codes = list(start_codes)
        self.init_code = init_code

    def __call__(self, argv):
        self.calls.append(list(argv))
        assert argv[0] == sys.executable
        assert Path(argv[1]) == SCRIPTS / "chain.py"
        if argv[2] == "init":
            if self.init_code is not None:
                return subprocess.CompletedProcess(argv, self.init_code, "", "error: refused\n")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = chain.main(argv[2:])
            return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())
        assert argv[2] == "start"
        code = self.start_codes.pop(0) if len(self.start_codes) > 1 else self.start_codes[0]
        stdout = json.dumps({"state": "running"}) if code == 0 else ""
        return subprocess.CompletedProcess(argv, code, stdout, "" if code == 0 else "error: x\n")

    def verbs(self):
        return [call[2] for call in self.calls]


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


def run(setup, runner, argv=None):
    return launcher.main(setup.argv() if argv is None else argv, runner=runner)


def test_all_checks_pass_initialises_then_starts(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    assert runner.verbs() == ["init", "start"]
    report = setup.read()
    frozen = json.loads((setup.campaign / "campaign.json").read_text())
    assert report["state"] == "launched" and report["problems"] == []
    assert report["binding"]["campaign_id"] == frozen["campaign_id"]
    assert report["start_result"] == {"state": "running"}
    assert [c["returncode"] for c in report["commands"]] == [0, 0]
    assert report["checks"]["junit"]["reports"][0]["tests"] == 6
    assert report["checks"]["prepared"]["outputs"]["graph.npz"]["sha256"]
    assert report["chain"]["sha256"] == launcher.sha256_file(SCRIPTS / "chain.py")
    assert not [p for p in setup.report.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("xml,reason", [
    (junit_xml(failures=1), "failures"),
    (junit_xml(errors=1), "errors"),
    (junit_xml(skipped=1), "skipped"),
    (junit_xml(cases=4), "fewer than 5"),
    (junit_xml(cases=6, declared={"tests": 6, "failures": 1, "errors": 0, "skipped": 0}), "failures"),
    (junit_xml(cases=6, declared={"tests": 7, "failures": 0, "errors": 0, "skipped": 0}), "declared 7"),
    ('<testsuites><testsuite name="x"><testcase name="a"></testsuite>', "malformed"),
    ("<html></html>", "unexpected JUnit root"),
    (junit_xml(declared={"tests": "six"}), "malformed JUnit counters"),
])
def test_bad_junit_fails_without_submitting(setup, xml, reason):
    setup.junit[1].write_text(xml)
    runner = FakeRunner()
    assert run(setup, runner) == 1
    report = setup.read()
    assert report["state"] == "failed" and report["stage"] == "gate"
    assert any(reason in p for p in report["problems"]), report["problems"]
    assert runner.calls == [] and not setup.campaign.exists()


def test_missing_junit_and_single_report_fail(setup):
    runner = FakeRunner()
    missing = setup.tmp / "gpu" / "absent.xml"
    assert run(setup, runner, setup.argv([setup.junit[0], missing])) == 1
    assert any("JUnit report missing" in p for p in setup.read()["problems"])
    assert run(setup, runner, setup.argv([setup.junit[0]])) == 1
    assert any("2 required" in p for p in setup.read()["problems"])
    assert run(setup, runner, setup.argv([setup.junit[0], setup.junit[0]])) == 1
    assert runner.calls == []


def test_identical_report_bytes_are_not_two_runs(setup):
    setup.junit[1].write_text(setup.junit[0].read_text())
    assert run(setup, FakeRunner()) == 1
    assert any("identical bytes" in p for p in setup.read()["problems"])


def test_testsuites_totals_are_not_double_counted(setup):
    cases = "".join(f'<testcase classname="c" name="t{i}"/>' for i in range(5))
    setup.junit[0].write_text(
        f'<testsuites tests="10" failures="0"><testsuite name="outer" tests="5">'
        f'<testsuite name="inner" tests="5" failures="0" errors="0" skipped="0">{cases}'
        f'</testsuite></testsuite></testsuites>')
    setup.junit[1].write_text(junit_xml(cases=5, wrap=False))
    assert run(setup, FakeRunner()) == 0
    reports = setup.read()["checks"]["junit"]["reports"]
    assert [r["tests"] for r in reports] == [5, 5]
    assert reports[0]["declared_leaf_totals"]["tests"] == 5


@pytest.mark.parametrize("damage,reason", [
    (lambda p: (p / "graph.npz").write_bytes(b"x" * (p / "graph.npz").stat().st_size), "SHA-256 differs"),
    (lambda p: (p / "vocabulary.json").write_bytes(b"short"), "bytes, identity records"),
    (lambda p: (p / "graph.npz").unlink(), "graph.npz does not exist"),
    (lambda p: (p / "identity.json").write_text("{not json"), "unreadable"),
    (lambda p: (p / "identity.json").unlink(), "identity.json missing"),
])
def test_prepared_identity_mismatch_fails(setup, damage, reason):
    damage(setup.prepared)
    runner = FakeRunner()
    assert run(setup, runner) == 1
    assert any(reason in p for p in setup.read()["problems"]), setup.read()["problems"]
    assert runner.calls == []


def test_identity_must_list_exactly_the_prepared_outputs(setup):
    identity = json.loads(setup.identity.read_text())
    del identity["outputs"]["test.tsv"]
    setup.identity.write_text(json.dumps(identity))
    assert run(setup, FakeRunner()) == 1
    assert any("expected" in p for p in setup.read()["problems"])


def test_config_on_other_data_dir_fails(setup):
    config = setup.spec.parent / "configs" / "betae.json"
    config.write_text(json.dumps({"method": "betae", "data_dir": "/elsewhere/fold0"}))
    assert run(setup, FakeRunner()) == 1
    assert any("not the verified" in p for p in setup.read()["problems"])


def test_retry_with_same_gate_is_idempotent(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    first = setup.read()["binding"]
    assert run(setup, runner) == 0
    report = setup.read()
    assert runner.verbs() == ["init", "start", "start"]
    assert report["binding"]["campaign_id"] == first["campaign_id"]
    assert report["binding"]["launched"] == first["launched"]


def test_start_failure_is_reported_and_retry_skips_init(setup):
    runner = FakeRunner(start_codes=[1, 0])
    assert run(setup, runner) == 2
    report = setup.read()
    assert report["state"] == "failed" and report["stage"] == "start"
    assert "launched" not in report["binding"]
    assert run(setup, runner) == 0
    assert runner.verbs() == ["init", "start", "start"]


def test_init_failure_never_starts(setup):
    runner = FakeRunner(init_code=2)
    assert run(setup, runner) == 2
    report = setup.read()
    assert report["state"] == "failed" and report["stage"] == "init"
    assert runner.verbs() == ["init"]


def test_crash_between_init_and_report_update_is_recoverable(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    report = setup.read()
    report["binding"]["campaign_id"] = None
    report["state"] = "initialising"
    setup.report.write_text(json.dumps(report))
    assert run(setup, runner) == 0
    assert runner.verbs() == ["init", "start", "start"]


def test_existing_campaign_not_made_by_gate_is_refused(setup):
    chain.init_campaign(setup.spec, setup.campaign)
    runner = FakeRunner()
    assert run(setup, runner) == 2
    report = setup.read()
    assert report["state"] == "refused"
    assert any("not initialised by this gate" in p for p in report["problems"])
    assert runner.calls == []


def test_changed_gate_on_bound_campaign_is_refused(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    setup.junit[1].write_text(junit_xml(cases=7, host="a100"))
    assert run(setup, runner) == 2
    assert any("different gate result" in p for p in setup.read()["problems"])
    assert runner.verbs() == ["init", "start"]


def test_altered_frozen_config_is_refused(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    frozen = json.loads((setup.campaign / "campaign.json").read_text())
    Path(frozen["runs"][0]["config"]).write_text("{}")
    assert run(setup, runner) == 2
    assert any("altered or missing" in p for p in setup.read()["problems"])
    assert runner.verbs() == ["init", "start"]


def test_vanished_campaign_is_not_recreated(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    for path in sorted(setup.campaign.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    assert run(setup, runner) == 2
    assert any("is gone" in p for p in setup.read()["problems"])
    assert runner.verbs() == ["init", "start"]


def test_report_bound_to_other_campaign_is_refused(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    setup.campaign = setup.tmp / "campaigns" / "other"
    assert run(setup, runner) == 2
    assert any("already binds campaign" in p for p in setup.read()["problems"])
    assert not setup.campaign.exists()


def test_failed_gate_after_launch_keeps_binding(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    binding = setup.read()["binding"]
    setup.junit[0].write_text(junit_xml(failures=1))
    assert run(setup, runner) == 1
    assert setup.read()["binding"] == binding


def test_unreadable_previous_report_with_existing_campaign_is_refused(setup):
    runner = FakeRunner()
    assert run(setup, runner) == 0
    setup.report.write_text("garbage")
    assert run(setup, runner) == 2
    report = setup.read()
    assert any("previous report unreadable" in n for n in report["notes"])
    assert runner.verbs() == ["init", "start"]


def test_concurrent_launcher_is_refused(setup):
    setup.report.parent.mkdir(parents=True)
    lock = setup.report.with_name(setup.report.name + ".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        runner = FakeRunner()
        assert run(setup, runner) == 2
        assert runner.calls == []
