#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JAR_OUT="${JAR_OUT:-$BUILD_DIR/OWL2VecStarGDAProjector.jar}"

command -v scalac >/dev/null 2>&1 || { echo "scalac is required; activate the Scala/mOWL build environment." >&2; exit 1; }
command -v jar >/dev/null 2>&1 || { echo "jar is required; install a JDK and ensure jar is on PATH." >&2; exit 1; }

if [ -z "${MOWL_LIB_DIR:-}" ]; then
  MOWL_LIB_DIR="$($PYTHON_BIN - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("mowl")
if spec is None or spec.origin is None:
    raise SystemExit("mOWL is not importable from PYTHON_BIN")
print(Path(spec.origin).parent / "lib")
PY
)" || {
    echo "Could not locate mOWL with $PYTHON_BIN; set MOWL_LIB_DIR explicitly." >&2
    exit 1
  }
fi

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/link-gda-projector.XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT
shopt -s nullglob
mowl_jars=("$MOWL_LIB_DIR"/*.jar)
if [ "${#mowl_jars[@]}" -eq 0 ]; then
  echo "No mOWL jars found in $MOWL_LIB_DIR; set MOWL_LIB_DIR to the active mOWL lib directory." >&2
  exit 1
fi

classpath="$(IFS=:; echo "${mowl_jars[*]}")"
scalac -cp "$classpath" -d "$STAGE_DIR" \
  "$ROOT_DIR/projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala"
mkdir -p "$(dirname -- "$JAR_OUT")"
jar cf "$JAR_OUT" -C "$STAGE_DIR" .
