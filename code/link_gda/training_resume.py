"""Atomic epoch-boundary snapshots for PyKEEN and validation selection.

The PyKEEN checkpoint preserves model, optimizer and Python/NumPy/PyTorch RNG
states. Each committed generation also preserves callback state and hard links
to the corresponding selected checkpoints. A crash between selection and the
next snapshot therefore cannot replace the previous generation's selections.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import random
import numpy as np
import torch
from importlib.metadata import version
from pykeen.training.callbacks import TrainingCallback
from .checkpoint_identity import graph_identity


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class EpochSnapshots(TrainingCallback):
    def __init__(self, directory, factory, identity, callbacks, *, frequency=10,
                 total_epochs, stop_after=None):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.output = self.directory.parent.resolve()
        self.factory = factory
        self.identity = identity
        self.callbacks = callbacks
        self.frequency = frequency
        self.total_epochs = total_epochs
        self.stop_after = stop_after
        if frequency < 1 or total_epochs < 1:
            raise ValueError('Positive epoch budget and snapshot frequency required')

    def save(self, loop):
        epoch = loop._epoch
        generation = self.directory / f'epoch-{epoch:06d}-{uuid.uuid4().hex}'
        generation.mkdir()
        loop._save_state(path=generation / 'loop.pt', triples_factory=self.factory)
        files = {}
        for callback in self.callbacks:
            for path in callback.checkpoint_files():
                path = Path(path).resolve()
                name = str(path.relative_to(self.output))
                if name in files: continue
                target = f'artifact-{len(files):03d}'
                os.link(path, generation / target)
                files[name] = {'snapshot': target, 'sha256': digest(generation / target)}
        metadata = {'identity': self.identity, 'epoch': epoch,
                    'early_stopped': bool(loop._should_stop),
                    'callbacks': [c.state_dict() for c in self.callbacks],
                    'loop_sha256': digest(generation / 'loop.pt'), 'files': files}
        atomic_json(generation / 'metadata.json', metadata)
        pointer = {'generation': generation.name,
                   'metadata_sha256': digest(generation / 'metadata.json')}
        atomic_json(self.directory / 'CURRENT.json', pointer)
        # Keep two committed generations. Hard links retain selected checkpoints
        # independently of replacement of their working-path directory entries.
        generations = sorted(self.directory.glob('epoch-*'), key=lambda p: p.stat().st_mtime_ns)
        for old in generations[:-2]:
            if old != generation: shutil.rmtree(old)

    def restore(self, loop):
        pointer_path = self.directory / 'CURRENT.json'
        if not pointer_path.exists(): return None
        pointer = json.loads(pointer_path.read_text())
        name = pointer['generation']
        if Path(name).name != name: raise ValueError('Invalid snapshot generation')
        generation = self.directory / name
        if digest(generation / 'metadata.json') != pointer['metadata_sha256']:
            raise ValueError('Snapshot metadata checksum mismatch')
        meta = json.loads((generation / 'metadata.json').read_text())
        if meta['identity'] != self.identity:
            raise ValueError('Resume configuration or scientific inputs changed')
        if digest(generation / 'loop.pt') != meta['loop_sha256']:
            raise ValueError('Training state checksum mismatch')
        if len(meta['callbacks']) != len(self.callbacks):
            raise ValueError('Validation callback configuration changed')
        for relative, item in meta['files'].items():
            destination = (self.output / relative).resolve()
            destination.relative_to(self.output)
            if Path(item['snapshot']).name != item['snapshot']:
                raise ValueError('Invalid snapshot artifact')
            source = generation / item['snapshot']
            if digest(source) != item['sha256']:
                raise ValueError('Selected checkpoint checksum mismatch')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.restore')
            if temporary.exists(): temporary.unlink()
            os.link(source, temporary)
            temporary.replace(destination)
        for callback, state in zip(self.callbacks, meta['callbacks']):
            callback.load_state_dict(state)
        loop._load_state(path=generation / 'loop.pt', triples_factory=self.factory)
        if loop._epoch != meta['epoch']:
            raise ValueError('Training and selection epoch mismatch')
        return meta

    def post_epoch(self, epoch, epoch_loss, **kwargs):
        technical_stop = self.stop_after is not None and epoch >= self.stop_after
        if epoch % self.frequency == 0 or epoch == self.total_epochs or self.training_loop._should_stop or technical_stop:
            self.save(self.training_loop)
        if technical_stop:
            # The committed state records the scientific early-stopping flag,
            # before this deliberately interrupted technical run stops.
            self.training_loop._should_stop = True


def train_resumable(loop, factory, *, directory, identity, callbacks,
                    total_epochs, batch_size, sub_batch_size=None,
                    checkpoint_frequency=10, stop_after=None):
    """Train to a total epoch budget, restoring the last committed boundary."""
    if loop.lr_scheduler is not None:
        raise ValueError('This snapshot callback requires a fixed learning rate')
    if loop.automatic_memory_optimization:
        raise ValueError('Set automatic_memory_optimization=False and specify batch sizes for reproducible resumption')
    if version('pykeen') != '1.11.1':
        raise ValueError('Epoch snapshots require the verified PyKEEN 1.11.1 interface')
    identity = {'experiment': identity, 'graph_sha256': graph_identity(factory),
                'pykeen': version('pykeen'), 'device_type': loop.model.device.type,
                'total_epochs': total_epochs, 'batch_size': batch_size,
                'sub_batch_size': sub_batch_size, 'checkpoint_frequency': checkpoint_frequency}
    snapshots = EpochSnapshots(directory, factory, identity, callbacks,
                               frequency=checkpoint_frequency, total_epochs=total_epochs,
                               stop_after=stop_after)
    state = snapshots.restore(loop)
    if state and (state['early_stopped'] or state['epoch'] >= total_epochs):
        return {'epochs': state['epoch'], 'early_stopped': state['early_stopped'], 'complete': True}
    if state is None:
        existing = [str(path) for c in callbacks for path in c.checkpoint_files()]
        if existing:
            raise ValueError(f'Selected weights cannot substitute for resumable training state: {existing}')
    loop.train(triples_factory=factory, num_epochs=total_epochs,
               batch_size=batch_size, sub_batch_size=sub_batch_size,
               continue_training=state is not None, callbacks=[*callbacks, snapshots],
               use_tqdm=False, num_workers=0, checkpoint_on_failure=False)
    pointer = json.loads((Path(directory) / 'CURRENT.json').read_text())
    meta = json.loads((Path(directory) / pointer['generation'] / 'metadata.json').read_text())
    complete = meta['early_stopped'] or meta['epoch'] >= total_epochs
    return {'epochs': meta['epoch'], 'early_stopped': meta['early_stopped'], 'complete': complete}


class _ProfileState:
    """Expose the same epoch snapshot interface for the profile objective."""
    def __init__(self, ranker, optimizer):
        self.ranker, self.optimizer = ranker, optimizer
        self._epoch, self._should_stop = ranker.epochs, False
        self.history = []

    def _save_state(self, path, triples_factory):
        torch.save({'model': self.ranker.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'ranker_rng': self.ranker.rng_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                    'epoch': self._epoch, 'history': self.history}, path)

    def _load_state(self, path, triples_factory):
        state = torch.load(path, map_location=self.ranker.device, weights_only=False)
        self.ranker.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.ranker.load_rng_state(state['ranker_rng'])
        self._epoch, self.history = state['epoch'], state['history']
        if self.ranker.epochs != self._epoch:
            raise ValueError('Profile sampler and model epoch mismatch')
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None:
            if not torch.cuda.is_available():
                raise ValueError('CUDA profile snapshot requires CUDA')
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])


def train_profile_resumable(ranker, optimizer, factory, *, directory, identity,
                            callback, total_epochs=20, accumulation_steps=16,
                            stop_after=None):
    """Commit the profile model, pooling scalar, optimizer, RNG and selection together."""
    identity = {'experiment': identity, 'graph_sha256': graph_identity(factory),
                'device_type': ranker.device.type, 'total_epochs': total_epochs,
                'accumulation_steps': accumulation_steps, 'ranker': ranker.metadata()}
    loop = _ProfileState(ranker, optimizer)
    snapshots = EpochSnapshots(directory, factory, identity, [callback],
                               frequency=1, total_epochs=total_epochs)
    state = snapshots.restore(loop)
    if state is None and callback.checkpoint_files():
        raise ValueError('Selected weights cannot substitute for resumable profile state')
    for epoch in range(loop._epoch + 1, total_epochs + 1):
        stats = ranker.train_epoch(optimizer, accumulation_steps=accumulation_steps)
        callback.evaluate_epoch(epoch)
        loop._epoch = epoch
        loop.history.append(stats)
        snapshots.save(loop)
        print('PROFILE_EPOCH', epoch, stats, flush=True)
        if stop_after is not None and epoch >= stop_after: break
    return {'epochs': loop._epoch, 'complete': loop._epoch >= total_epochs,
            'history': loop.history}
