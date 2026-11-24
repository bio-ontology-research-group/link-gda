"""Write one model-only export of a completed query run's terminal state, bound to the full state it came from.

Runs before any pruning, because the terminal checkpoint is not necessarily a selected one. The export holds the
model state_dict exactly as stored in the terminal full state (no optimizer, RNG or sampler state), its step, the run
identity hash and the source full-state hash. Refuses if the terminal full state is missing or differs from COMPLETE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training, output = Path(args.training_dir), Path(args.output)
    complete = json.loads((training / 'COMPLETE.json').read_text())
    terminal = complete['terminal_checkpoint']
    path = training / terminal['checkpoint']
    if output.exists():
        record = json.loads(output.with_suffix('.json').read_text())
        assert record['sha256'] == sha(output) and record['source_full_checkpoint_sha256'] == terminal['checkpoint_sha256']
        print(json.dumps(record))
        return
    assert path.exists() and sha(path) == terminal['checkpoint_sha256'], 'terminal full state missing or changed'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == complete['steps'] and state['identity_sha256'] == complete['identity_sha256']
    temporary = output.with_suffix('.pt.tmp')
    torch.save({'kind': 'terminal_model_only', 'identity_sha256': complete['identity_sha256'], 'step': state['step'], 'model': state['model'], 'aliases': {},
                'source_full_checkpoint_sha256': terminal['checkpoint_sha256']}, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    record = {'path': output.name, 'sha256': sha(output), 'step': state['step'], 'identity_sha256': complete['identity_sha256'],
              'source_full_checkpoint': terminal['checkpoint'], 'source_full_checkpoint_sha256': terminal['checkpoint_sha256']}
    output.with_suffix('.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
