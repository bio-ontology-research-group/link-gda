"""Recompute validation metrics and both readouts' selections from a run's saved arrays.

Reads only ``identity.json``, ``progress.json``, the prepared ``profiles.json``
and ``validation/step-*.npz``. Exits nonzero when any recomputed aggregate, rank,
metric or selected step differs from what the trainer recorded.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from compositional_gda.common import read_json, sha
from compositional_gda.data import READOUTS
from compositional_gda.evaluate import recompute_from_arrays, replay_selection


def recompute_run(run_dir, data_dir):
    run_dir = Path(run_dir)
    identity = read_json(run_dir / 'identity.json')
    progress = read_json(run_dir / 'progress.json')
    if sha(Path(data_dir) / 'profiles.json') != identity['data']['files']['profiles.json']:
        raise ValueError('profiles.json differs from the run identity')
    profiles = read_json(Path(data_dir) / 'profiles.json')
    weight = identity['config']['aggregation_weight']
    curve = []
    for entry in progress['curve']:
        with np.load(run_dir / entry['validation'], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if arrays['diseases'].tolist() != identity['validation_panel']['diseases']:
            raise ValueError(f'Validation panel differs at step {entry["step"]}')
        recomputed = recompute_from_arrays(arrays, profiles, weight)
        for readout in READOUTS:
            for mode in ('raw', 'calibrated'):
                if recomputed[readout][mode] != entry['readouts'][readout][mode]:
                    raise ValueError(f'{readout} {mode} metrics differ at step {entry["step"]}')
        curve.append((entry['step'], recomputed))
    selected = replay_selection(curve)
    recorded = {readout: progress['best'][readout]['step'] for readout in READOUTS}
    if selected != recorded:
        raise ValueError(f'Selection differs: recomputed {selected}, recorded {recorded}')
    return {'steps': [step for step, _ in curve], 'selected_step': selected,
            'metrics': {str(step): values for step, values in curve}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='Prepared fold used by the run')
    args = parser.parse_args(argv)
    print(json.dumps(recompute_run(args.run_dir, args.data_dir), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
