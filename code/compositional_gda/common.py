"""Hashing, atomic file publication and environment identity for compositional GDA runs."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TRAINER_MODULES = ('__init__.py', 'common.py', 'models.py', 'data.py', 'evaluate.py', 'train.py')


def sha(path):
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    """SHA-256 of a JSON value serialised with sorted keys."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def fsync_directory(directory):
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path, writer):
    """Write through a temporary file, fsync it, rename it over ``path`` and fsync the directory.

    ``writer`` receives an open binary handle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    atomic_write_bytes(path, lambda handle: handle.write(data))


def atomic_torch_save(path, value):
    import torch
    atomic_write_bytes(path, lambda handle: torch.save(value, handle))


def atomic_savez(path, **arrays):
    import numpy as np
    atomic_write_bytes(path, lambda handle: np.savez_compressed(handle, **arrays))


def atomic_copy(source, target, expected_sha):
    """Copy ``source`` to ``target`` atomically after checking its SHA-256."""
    if sha(source) != expected_sha:
        raise ValueError(f'Committed artifact changed: {source}')
    data = Path(source).read_bytes()
    atomic_write_bytes(target, lambda handle: handle.write(data))


def source_hashes():
    """SHA-256 of every module the trainer imports."""
    return {name: sha(PACKAGE_DIR / name) for name in TRAINER_MODULES}


def installed_versions():
    import numpy
    import torch
    return {'python': platform.python_version(), 'python_implementation': platform.python_implementation(),
            'torch': torch.__version__, 'numpy': numpy.__version__, 'torch_cuda': torch.version.cuda,
            'platform': sys.platform}
