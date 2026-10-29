"""Verify the locked CPU analysis environment against its captured package identities."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import subprocess


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    if platform.python_version() != capture['python']:
        raise ValueError('Python version differs from the captured analysis environment')
    expected = {canonical(d['name']): d['version'] for d in capture['distributions']}
    distributions = list(importlib.metadata.distributions())
    installed = {canonical(d.metadata['Name']): d.version for d in distributions}
    if len(expected) != len(capture['distributions']) or len(installed) != len(distributions):
        raise ValueError('Duplicate Python distribution identities')
    if installed != expected:
        differences = {name: (expected.get(name), installed.get(name))
            for name in sorted(expected.keys() | installed.keys())
            if expected.get(name) != installed.get(name)}
        raise ValueError('Python distributions differ: ' + str(differences))
    records = {r['name']: r for path in (Path(sys.prefix) / 'conda-meta').glob('*.json')
               for r in [json.loads(path.read_text())]}
    if set(records) != {r['name'] for r in capture['conda']}:
        raise ValueError('Conda package set differs')
    for record in capture['conda']:
        for field in ['version', 'build', 'md5', 'sha256']:
            if record[field] != records[record['name']].get(field):
                raise ValueError('Conda package identity differs: ' + record['name'] + '/' + field)

    requirements_path = Path(__file__).with_name('requirements-embedding-replay.txt')
    requirements = dict(line.split('==') for line in requirements_path.read_text().splitlines()
                        if line.strip() and not line.startswith('#'))
    if requirements != capture['requirements']:
        raise ValueError('Captured requirements differ from the analysis pins')
    if any(installed.get(canonical(name)) != version for name, version in requirements.items()):
        raise ValueError('Installed analysis requirements differ')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    # Import the actual binary-backed analysis stack after verifying all identities.
    import numpy
    import scipy.stats
    import scipy.sparse
    import pandas
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    result = dict(status='passed', python=platform.python_version(),
        conda_packages=len(records), python_distributions=len(installed),
        requirements_verified=len(requirements), all_package_identities_equal=True,
        binary_stack_imports=True,
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requirements_sha256=hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
        scope='Captured package identities, pip dependency check and analysis-stack imports; numerical replay is verified separately.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Choose a new environment verification output path')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
