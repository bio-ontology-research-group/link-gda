"""Verify relocated resource paths without loading external runtimes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_exomiser_classpath_stays_under_repository_root(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    script = checkout / "code/baselines/exomiser_eval.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "code/baselines/exomiser_eval.py", script)
    resources = checkout / "exomiser/exomiser-cli-14.0.0"
    (resources / "lib").mkdir(parents=True)
    dependency = resources / "lib/dependency.jar"
    dependency.touch()
    probe = """
import json, runpy, sys, types
class StopProbe(Exception):
    pass
jpype = types.ModuleType('jpype')
jpype.getDefaultJVMPath = lambda: 'fake-jvm'
def start(*args, **kwargs):
    print(json.dumps(kwargs['classpath']))
    raise StopProbe()
jpype.startJVM = start
sys.modules['jpype'] = jpype
try:
    runpy.run_path(sys.argv[1], run_name='__main__')
except StopProbe:
    pass
"""
    result = subprocess.run([sys.executable, "-c", probe, str(script)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == {
        str(dependency), str(resources / "exomiser-cli-14.0.0.jar")}


def test_ultra_setup_finds_repository_environment_file(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    script = checkout / "code/baselines/setup_ultra_env.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "code/baselines/setup_ultra_env.sh", script)
    environment = checkout / "environment-ultra.yml"
    environment.write_text("name: fixture\n")
    conda = tmp_path / "fake conda"
    profile = conda / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text('''conda() {
  if [ "$1" = "env" ] && [ "$2" = "create" ]; then
    printf '%s' "$4" > "$FAKE_CAPTURE"
    test -f "$4"
  fi
}
''')
    tools = tmp_path / "fake tools"
    tools.mkdir()
    python = tools / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    capture = tmp_path / "environment-path.txt"
    result = subprocess.run(["bash", str(script)], cwd=tmp_path,
                            env={**os.environ, "CONDA_ROOT": str(conda),
                                 "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                                 "FAKE_CAPTURE": str(capture)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(capture.read_text()).resolve() == environment.resolve()
