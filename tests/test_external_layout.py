"""Layout checks for the batch-4 migration of the external-baseline,
ontology-projection, and W&B-extraction tools.

`prepare_ultra_data.py` and `score_ultra.py` load the real ULTRA
checkout only lazily (inside `main`), so `--help` exits before any ULTRA
import and must work from a foreign working directory.

The Scala compiler (`code/projector/compile_projector.sh`) and the
semantic-similarity launcher (`code/baselines/run_all_sem_sim.sh`) are
exercised with FAKE `scalac`/`jar`/`groovy` executables in a temporary
copied checkout whose path contains spaces. The stubs only record their
invocations and verify path wiring (source file present, output written to
the requested location, launcher cwd and driver arguments). They do not
perform any real Scala/Groovy compilation, inference, Exomiser, or W&B API
call, and they never write into the real `build/`, `data/`, or `paper/`
resources of this checkout. `exomiser_eval.py` starts its JVM at import, so
it is deliberately not `--help`-tested here; its relocation was checked by a
static before/after AST parity check and a mocked-JVM resource-path test
in test_external_resource_paths.py.
"""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "code" / "baselines"

@pytest.mark.parametrize("name", ["prepare_ultra_data.py", "score_ultra.py"])
def test_ultra_driver_help_from_foreign_cwd(name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(BASELINES / name), "--help"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def make_fake(directory, name, script):
    path = directory / name
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def copy_into(checkout, rel):
    source = REPO_ROOT / rel
    destination = checkout / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


@pytest.mark.parametrize("override_build", [False, True])
def test_compile_projector_path_wiring_with_fake_toolchain(tmp_path, override_build):
    """compile_projector.sh must locate the Scala source and build output in
    the copied checkout, driven only by fake scalac/jar executables."""
    checkout = tmp_path / "link gda copy"
    copy_into(checkout, "code/projector/compile_projector.sh")
    source = copy_into(
        checkout, "code/projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala")

    fakes = tmp_path / "fake tools"
    fakes.mkdir()
    log = tmp_path / "tool_calls.log"
    make_fake(fakes, "scalac", """\
#!/bin/bash
printf 'scalac' >> "${FAKE_LOG:?}"
printf '\\t%s' "$@" >> "${FAKE_LOG:?}"
printf '\\n' >> "${FAKE_LOG:?}"
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-d" ]; then out="$a"; fi
  prev="$a"
done
[ -n "$out" ] || exit 3
[ -f "${@: -1}" ] || { echo "fake scalac: source missing" >&2; exit 1; }
mkdir -p "$out"
exit 0
""")
    make_fake(fakes, "jar", """\
#!/bin/bash
printf 'jar' >> "${FAKE_LOG:?}"
printf '\\t%s' "$@" >> "${FAKE_LOG:?}"
printf '\\n' >> "${FAKE_LOG:?}"
[ "$1" = "cf" ] || exit 3
out="$2"
[ -d "${@: -2:1}" ] || { echo "fake jar: stage dir missing" >&2; exit 1; }
mkdir -p "$(dirname "$out")"
printf 'fake jar contents\\n' > "$out"
exit 0
""")

    mowl_lib = tmp_path / "mowl lib"
    mowl_lib.mkdir()
    (mowl_lib / "fake-mowl.jar").write_text("fake mowl jar")
    build_dir = tmp_path / "build dir" if override_build else checkout / "build"
    stage_parent = tmp_path / "stage dir"
    stage_parent.mkdir()
    launch_from = tmp_path / "launch from"
    launch_from.mkdir()

    env = {**os.environ,
           "PATH": f"{fakes}{os.pathsep}{os.environ['PATH']}",
           "MOWL_LIB_DIR": str(mowl_lib),
           "TMPDIR": str(stage_parent),
           "FAKE_LOG": str(log)}
    env.pop("JAR_OUT", None)
    env.pop("BUILD_DIR", None)
    if override_build:
        env["BUILD_DIR"] = str(build_dir)
    result = subprocess.run(
        ["bash", str(checkout / "code" / "projector" / "compile_projector.sh")],
        cwd=str(launch_from), env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    jar_out = build_dir / "OWL2VecStarGDAProjector.jar"
    assert jar_out.exists() and jar_out.stat().st_size > 0

    calls = log.read_text().splitlines()
    assert len(calls) == 2, calls
    scalac_args = calls[0].split("\t")[1:]
    jar_args = calls[1].split("\t")[1:]
    assert scalac_args[-1] == str(source)
    stage = Path(scalac_args[scalac_args.index("-d") + 1])
    assert stage.parent == stage_parent
    assert not stage.exists()
    assert str(mowl_lib / "fake-mowl.jar") in scalac_args
    assert jar_args[1] == str(jar_out)


def test_run_all_sem_sim_path_wiring_with_fake_groovy(tmp_path):
    """run_all_sem_sim.sh must cd to the copied repo root and launch all 50
    measure/fold groovy calls with code/baselines driver paths, driven by a
    fake groovy that records cwd and arguments."""
    checkout = tmp_path / "sem sim checkout"
    copy_into(checkout, "code/baselines/run_all_sem_sim.sh")
    copy_into(checkout, "code/baselines/semantic_similarity.groovy")
    copy_into(checkout, "code/baselines/semantic_similarity_simgic.groovy")

    fakes = tmp_path / "fake tools"
    fakes.mkdir()
    log = tmp_path / "groovy_calls.log"
    make_fake(fakes, "groovy", """\
#!/bin/bash
printf '%s\\t%s\\n' "$(pwd)" "$*" >> "${FAKE_LOG:?}"
exit 0
""")

    launch_from = tmp_path / "launch from"
    launch_from.mkdir()
    env = {**os.environ,
           "PATH": f"{fakes}{os.pathsep}{os.environ['PATH']}",
           "FAKE_LOG": str(log)}
    result = subprocess.run(
        ["bash", str(checkout / "code" / "baselines" / "run_all_sem_sim.sh")],
        cwd=str(launch_from), env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    lines = log.read_text().splitlines()
    assert len(lines) == 50, f"expected 50 groovy calls, saw {len(lines)}"
    for line in lines:
        cwd, args = line.split("\t", 1)
        assert cwd == str(checkout), f"groovy launched from {cwd}"
        driver = args.split()[0]
        assert driver in ("code/baselines/semantic_similarity.groovy",
                          "code/baselines/semantic_similarity_simgic.groovy")
        assert (checkout / driver).exists(), driver
    drivers = [line.split("\t", 1)[1].split()[0] for line in lines]
    assert drivers.count("code/baselines/semantic_similarity.groovy") == 40
    assert drivers.count("code/baselines/semantic_similarity_simgic.groovy") == 10

    logs_dir = checkout / "logs" / "sem_sim"
    results_dir = checkout / "data" / "baseline_results"
    assert logs_dir.is_dir() and results_dir.is_dir()
    fold_logs = sorted(p.name for p in logs_dir.iterdir() if "_fold" in p.name)
    assert len(fold_logs) == 50
    masters = [p for p in logs_dir.iterdir() if p.name.endswith(".master.log")]
    assert len(masters) == 5
