"""Train one Track A path model on sampled one-hop queries and select checkpoints on composed validation.

With ``auxiliary`` set, the run is the separate two-hop training diagnostic
(``two_hop``) or its matched one-hop readout control (``one_hop_readout``):
each update keeps ``batch_size`` queries, of which ``auxiliary_slots`` come from
the auxiliary groups and the rest from the default one-hop stream.

Exit codes: 0 when the fixed budget and every scheduled validation are complete
(``COMPLETE.json`` written), 75 when the run paused with a committed
``resume.pt`` (deadline, SIGUSR1/SIGTERM, ``--stop-after`` or
``--stop-before-validation``), 3 when another process holds the run lock.
Anything else is a failure. No test-set scoring exists in this entry point.
"""
import argparse
import fcntl
import json
import math
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path

PROCESS_START = time.monotonic()

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from compositional_gda.common import (atomic_copy, atomic_savez, atomic_torch_save, atomic_write_bytes, canonical_sha,
                                      installed_versions, read_json, sha, source_hashes, write_json)
from compositional_gda.data import (FORWARD, INVERSE, READOUTS, MixedSampler, PathQueryIndex, PreparedData, QueryIndex,
                                    RelationQueryView, Sampler, query_digest, verify_two_hop)
from compositional_gda.evaluate import ValidationInterrupted, eligible, selection_key, validate
from compositional_gda.models import (METHOD_DEFAULTS, METHOD_LABELS, METHODS, MEMBERSHIP_MODES, SOURCES, build_model,
                                      query_loss)

EXIT_COMPLETE = 0
EXIT_PAUSE = 75
EXIT_LOCKED = 3

BASE_DEFAULTS = {
    'seed': 0, 'steps': 500, 'eval_every': 100, 'dim': 100, 'batch_size': 128, 'negative_sample_size': 32,
    'learning_rate': 1e-4, 'validation_diseases': 32, 'aggregation_weight': 0.5, 'query_chunk': 1024,
    'score_chunk_elements': 2 ** 25, 'checkpoint_every_steps': 100, 'checkpoint_every_seconds': 900.0,
    'deterministic': True, 'name': None, 'auxiliary': None, 'auxiliary_slots': 0,
}

AUXILIARY = {
    'two_hop': {'stream': 1, 'groups': (('forward_2p', FORWARD), ('inverse_2p', INVERSE))},
    'one_hop_readout': {'stream': 2, 'groups': tuple((f'readout_1p:{name}', (name,)) for name in FORWARD + INVERSE)},
}
BASE_GROUP = 'base_1p'

PROTOCOL = {
    'training_queries': 'all (head, relation) one-hop queries of graph.npz, inverse relations included',
    'loss': 'KGReasoning train_step: mean of subsampling-weighted -logsigmoid(positive) and -logsigmoid(-negative)',
    'subsampling_weight': 'sqrt(1 / (4 + number of answers))',
    'negatives': 'uniform entities with known answers of the query removed (np.isin, no assume_unique)',
    'optimizer': 'Adam, constant learning rate, no scheduler',
    'reference_schedule_deviation': 'sources divide the learning rate by 5 at max_steps // 2; not applied',
    'readouts': {'forward': 'gene -> associated_with -> has_symptom -> phenotype',
                 'inverse': 'phenotype -> inverse:has_symptom -> inverse:associated_with -> gene'},
    'aggregation': 'weight * max + (1 - weight) * mean over the profile multiset, duplicates kept',
    'selection': 'per readout: calibrated Hits@10, Hits@1, MRR, -MR; step 0 fallback; Hits@1 and Hits@10 >= step 0',
    'ties': 'average rank',
    'geometre_transitive': 'off',
}

AUXILIARY_DESCRIPTIONS = {
    'two_hop': ('two-hop path queries built from the training graph only, one query per anchor with the union answer '
                'set over all training intermediate diseases: forward gene -> associated_with -> has_symptom, '
                'inverse phenotype -> inverse:has_symptom -> inverse:associated_with'),
    'one_hop_readout': ('one-hop queries of each readout relation: associated_with, has_symptom, inverse:has_symptom, '
                        'inverse:associated_with'),
}

PAUSE_REQUEST = {'signal': None}


def protocol(config):
    """The protocol record of a configuration; ``PROTOCOL`` itself when no auxiliary group is set."""
    mode = config['auxiliary']
    if mode is None:
        return PROTOCOL
    groups = [name for name, _ in AUXILIARY[mode]['groups']]
    share = config['auxiliary_slots'] // len(groups)
    base = config['batch_size'] - config['auxiliary_slots']
    return {**PROTOCOL,
            'training_queries': (f'fixed per-update composition: {base} queries from all (head, relation) one-hop '
                                 f'queries of graph.npz, inverse relations included, plus {share} from each of '
                                 f'{groups} ({AUXILIARY_DESCRIPTIONS[mode]})'),
            'auxiliary_mode': mode,
            'auxiliary_sampling': ('each group has its own sampler and random stream; each wraps into its next epoch '
                                   'so every update has exactly batch_size queries; one uniform positive per query '
                                   'from its answer set'),
            'auxiliary_negatives': 'uniform entities with the full answer set of the query removed (union for two-hop)',
            'auxiliary_loss': ('one loss over the concatenated group logits with the source subsampling weights; '
                               'GeometrE membership term computed once per update'),
            'reference_mixture_deviation': ('KGReasoning samples query structures from one proportional pool; this '
                                            'diagnostic uses a fixed quota per group')}


class IdentityMismatch(RuntimeError):
    """The run directory belongs to another configuration, source, data or environment."""


class OrphanedHistory(RuntimeError):
    """The run directory holds training or validation history but no ``resume.pt`` to continue it from."""


def request_pause(signum, frame):
    PAUSE_REQUEST['signal'] = signal.Signals(signum).name


def install_signal_handlers():
    return {number: signal.signal(number, request_pause) for number in (signal.SIGUSR1, signal.SIGTERM)}


def restore_signal_handlers(previous):
    for number, handler in previous.items():
        signal.signal(number, handler)


def resolve_config(raw, config_path):
    """Fill defaults, refuse unknown or inapplicable keys and resolve ``data_dir`` against the config's directory."""
    if 'method' not in raw or raw['method'] not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if 'data_dir' not in raw:
        raise ValueError('data_dir is required')
    method = raw['method']
    allowed = {'method', 'data_dir'} | set(BASE_DEFAULTS) | set(METHOD_DEFAULTS[method])
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f'Unknown or inapplicable configuration keys for {method}: {unknown}')
    config = {**BASE_DEFAULTS, **METHOD_DEFAULTS[method], **raw}
    if method == 'geometre' and config['membership_mode'] not in MEMBERSHIP_MODES:
        raise ValueError(f'GeometrE needs an explicit membership_mode in {MEMBERSHIP_MODES}')
    for name in ('steps', 'eval_every', 'dim', 'batch_size', 'negative_sample_size', 'query_chunk',
                 'score_chunk_elements', 'checkpoint_every_steps'):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if config['validation_diseases'] is not None and config['validation_diseases'] < 2:
        raise ValueError('validation_diseases must be at least 2 or null for all')
    slots = config['auxiliary_slots']
    if not isinstance(slots, int) or isinstance(slots, bool):
        raise ValueError('auxiliary_slots must be an integer')
    if config['auxiliary'] is None:
        if slots != 0:
            raise ValueError('auxiliary_slots must be 0 when auxiliary is null')
    else:
        if config['auxiliary'] not in AUXILIARY:
            raise ValueError(f'auxiliary must be null or one of {sorted(AUXILIARY)}')
        groups = len(AUXILIARY[config['auxiliary']]['groups'])
        if not 0 < slots < config['batch_size'] or slots % groups:
            raise ValueError(f'auxiliary_slots must be positive, below batch_size and divisible by {groups}')
    data_dir = Path(config['data_dir'])
    if not data_dir.is_absolute():
        data_dir = (Path(config_path).resolve().parent / data_dir).resolve()
    return config, data_dir


def configure_numerics(config, device):
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(bool(config['deterministic']))
    torch.backends.cudnn.deterministic = bool(config['deterministic'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'deterministic_algorithms': bool(config['deterministic']), 'tf32': False, 'cudnn_benchmark': False,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if device.type == 'cuda' else None}


def pending_validation(curve, last, total, frequency):
    """True when only the validation at ``last`` is missing; raise on any other gap or repetition."""
    if not 0 <= last <= total:
        raise ValueError('Resume step outside training budget')
    expected = sorted({0, *range(frequency, last + 1, frequency), *([total] if last == total else [])})
    actual = [entry['step'] for entry in curve]
    if actual == expected:
        return False
    if expected[-1] == last and actual == expected[:-1]:
        return True
    raise ValueError('Resume state has missing or repeated past validation steps')


def segment_record(event, step, **extra):
    record = {'event': event, 'step': step, 'unix_time': time.time(), 'host': platform.node(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'), 'pid': os.getpid()}
    if torch.cuda.is_available():
        record['gpu'] = torch.cuda.get_device_name(torch.cuda.current_device())
    record.update(extra)
    return record


class Run:
    """One run directory: training, validation, selection, commits and completion."""

    def __init__(self, config, data_dir, run_dir, device='cpu', deadline_seconds=None, stop_after=None,
                 stop_before_validation=None, deadline_safety=1.5, started=None):
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.deadline_seconds = deadline_seconds
        self.stop_after = stop_after
        self.stop_before_validation = stop_before_validation
        self.deadline_safety = deadline_safety
        self.started = PROCESS_START if started is None else started
        self.segment_started = time.monotonic()
        self.numerics = configure_numerics(config, self.device)
        self.data = PreparedData(data_dir)
        self.diseases, self.rows = self.data.validation_panel(config['validation_diseases'])
        self.groups, self.training_queries = self.training_groups()
        self.identity = self.build_identity()
        self.identity_sha256 = canonical_sha(self.identity)
        self.total = config['steps']
        self.frequency = config['eval_every']
        self.timing = {'step': 0.0, 'validation': 0.0, 'commit': 0.0}
        self.elapsed_before = 0.0
        self.curve = []
        self.best = {}
        self.baseline = {}
        self.artifacts = {}
        self.last_commit = (None, time.monotonic())

    def training_groups(self):
        """Query groups and their identity record for an auxiliary run; ``(None, None)`` for the default run.

        Index construction draws no random numbers, so building the groups here
        leaves the model initialisation in ``build`` unchanged.
        """
        mode = self.config['auxiliary']
        if mode is None:
            return None, None
        self.index = QueryIndex(self.data.triples, self.data.num_entities)
        spec = AUXILIARY[mode]
        share = self.config['auxiliary_slots'] // len(spec['groups'])
        groups = [(BASE_GROUP, self.index, self.config['batch_size'] - self.config['auxiliary_slots'], None, ())]
        for number, (name, relations) in enumerate(spec['groups'], start=1):
            ids = tuple(self.data.relations[relation] for relation in relations)
            index = PathQueryIndex(self.index, ids) if len(ids) == 2 else RelationQueryView(self.index, ids[0])
            groups.append((name, index, share, (spec['stream'], number), relations))
        record = {'mode': mode, 'groups': []}
        for name, index, slots, stream, relations in groups:
            entry = {'name': name, 'relations': list(relations), 'slots': slots,
                     'stream': None if stream is None else list(stream), 'queries': len(index),
                     'answer_pairs': int(np.asarray(index.counts).sum()), 'digest_sha256': query_digest(index),
                     'mean_subsampling_weight': math.fsum(np.asarray(index.weights, dtype=np.float64).tolist())
                     / len(index)}
            if isinstance(index, PathQueryIndex):
                entry.update({'paths_before_union': index.paths, 'intermediates': len(index.intermediates)})
            record['groups'].append(entry)
        if mode == 'two_hop':
            record['two_hop_checks'] = verify_two_hop(self.data, self.index, groups[1][1], groups[2][1])
        return [group[:4] for group in groups], record

    def build_identity(self):
        identity = {
            'format': 'compositional-gda-run/1',
            'config': {key: value for key, value in self.config.items() if key != 'data_dir'},
            'method_label': METHOD_LABELS[self.config['method']],
            'protocol': protocol(self.config),
            'data': {'identity_sha256': self.data.identity_sha256, 'files': self.data.file_digests},
            'validation_panel': {'diseases': self.diseases, 'rows': len(self.rows)},
            'source': {'trainer': source_hashes(), 'references': SOURCES},
            'versions': installed_versions(),
            'device_type': self.device.type,
            'numerics': self.numerics,
        }
        if self.training_queries is not None:
            identity['training_queries'] = self.training_queries
        return identity

    def path(self, name):
        return self.run_dir / name

    def remaining(self):
        if self.deadline_seconds is None:
            return math.inf
        return self.deadline_seconds - (time.monotonic() - self.started)

    def can_train(self):
        return self.remaining() >= self.deadline_safety * (self.timing['step'] + self.timing['commit'])

    def can_validate(self):
        return self.remaining() >= self.deadline_safety * (self.timing['validation'] + 2 * self.timing['commit'])

    @staticmethod
    def pause_requested():
        return PAUSE_REQUEST['signal'] is not None

    def build(self):
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = build_model(self.config, self.data.num_entities, self.data.num_relations).to(self.device)
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config['learning_rate'])
        if self.groups is None:
            self.index = QueryIndex(self.data.triples, self.data.num_entities)
            self.sampler = Sampler(self.index, self.config['batch_size'], self.config['negative_sample_size'], seed)
        else:
            self.sampler = MixedSampler(self.groups, self.config['negative_sample_size'], seed)
            if self.sampler.batch_size != self.config['batch_size']:
                raise ValueError('Mixed sampler groups do not fill batch_size')

    def rng_state(self):
        cuda = torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': cuda}

    def restore_rng(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None and self.device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda'][:torch.cuda.device_count()])

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.segment_started

    def state(self, step):
        return {'format': 1, 'identity_sha256': self.identity_sha256, 'step': step,
                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': None,
                'sampler': self.sampler.state_dict(), 'rng': self.rng_state(), 'curve': self.curve,
                'best': self.best, 'baseline': self.baseline, 'artifacts': self.artifacts,
                'timing': self.timing, 'elapsed_seconds': self.elapsed()}

    def progress(self, step):
        return {'step': step, 'steps': self.total, 'eval_every': self.frequency,
                'pending_validation': pending_validation(self.curve, step, self.total, self.frequency),
                'curve': self.curve, 'best': self.best, 'baseline': self.baseline,
                'elapsed_seconds': self.elapsed(), 'identity_sha256': self.identity_sha256}

    def commit(self, step):
        tic = time.monotonic()
        atomic_torch_save(self.path('resume.pt'), self.state(step))
        write_json(self.path('progress.json'), self.progress(step))
        self.timing['commit'] = max(self.timing['commit'], time.monotonic() - tic)
        self.last_commit = (step, time.monotonic())

    def log(self, record):
        with self.path('training.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')

    def segment(self, record):
        with self.path('segments.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')

    def publish(self):
        for readout, record in self.best.items():
            artifact = self.artifacts[str(record['step'])]
            atomic_copy(self.path(artifact['checkpoint']['file']), self.path(f'selected-{readout}.pt'),
                        artifact['checkpoint']['sha256'])
            atomic_copy(self.path(artifact['validation']['file']), self.path(f'selected-{readout}-validation.npz'),
                        artifact['validation']['sha256'])

    def verify_artifacts(self):
        for entry in self.curve:
            for kind in ('checkpoint', 'validation'):
                record = self.artifacts[str(entry['step'])][kind]
                if sha(self.path(record['file'])) != record['sha256']:
                    raise ValueError(f'Committed artifact changed: {record["file"]}')

    def validate_step(self, step):
        tic = time.monotonic()
        arrays, readouts = validate(self.model, self.data, self.diseases, self.rows, self.config, self.device,
                                    interrupt=self.pause_requested)
        checkpoint_file = f'checkpoints/step-{step:06d}.pt'
        validation_file = f'validation/step-{step:06d}.npz'
        model_state = {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()}
        atomic_torch_save(self.path(checkpoint_file), {'step': step, 'identity_sha256': self.identity_sha256,
                                                      'config': self.identity['config'], 'model': model_state})
        atomic_savez(self.path(validation_file), **arrays)
        self.artifacts[str(step)] = {'checkpoint': {'file': checkpoint_file, 'sha256': sha(self.path(checkpoint_file))},
                                     'validation': {'file': validation_file, 'sha256': sha(self.path(validation_file))}}
        entry = {'step': step, 'readouts': {}, 'diagnostics': self.model.diagnostics(),
                 'checkpoint': checkpoint_file, 'validation': validation_file}
        for readout in READOUTS:
            current = readouts[readout]['calibrated']
            self.baseline.setdefault(readout, current)
            passes = eligible(current, self.baseline[readout])
            chosen = readout not in self.best or (passes and selection_key(current) > selection_key(
                self.best[readout]['calibrated']))
            if chosen:
                self.best[readout] = {'step': step, 'calibrated': current, 'raw': readouts[readout]['raw']}
            entry['readouts'][readout] = {'raw': readouts[readout]['raw'], 'calibrated': current,
                                          'eligible': passes, 'selected': chosen}
        self.publish()
        self.curve.append(entry)
        seconds = time.monotonic() - tic
        self.timing['validation'] = max(self.timing['validation'], seconds)
        self.log({'event': 'validation', 'step': step, 'readouts': entry['readouts'], 'diagnostics': entry['diagnostics'],
                  'seconds': seconds})
        self.commit(step)

    def group_logits(self):
        """Concatenated positive ``[B]``, negative ``[B, K]`` logits and weights ``[B]`` of a mixed batch.

        Each group is encoded separately because path lengths differ. Also
        returns each group's own loss halves, weight sum and loss-weight share.
        """
        device = self.device
        parts = []
        for name, anchors, relations, positives, negatives, weights in self.sampler.next_batch():
            anchors, relations, positives, negatives = (torch.as_tensor(array).to(device)
                                                        for array in (anchors, relations, positives, negatives))
            weights = torch.as_tensor(weights).to(device)
            query = self.model.encode(anchors, relations)
            parts.append((name, self.model.score(query, positives.unsqueeze(1)).squeeze(1),
                          self.model.score(query, negatives, negative=True), weights))
        positive_logit = torch.cat([part[1] for part in parts])
        negative_logit = torch.cat([part[2] for part in parts])
        weights = torch.cat([part[3] for part in parts])
        total_weight = float(weights.sum())
        positions = self.sampler.positions()
        records = {}
        with torch.no_grad():
            for name, positive, negative, weight in parts:
                loss, positive_loss, negative_loss = query_loss(positive, negative, weight)
                records[name] = {'queries': int(weight.numel()), 'query_loss': float(loss),
                                 'positive_loss': float(positive_loss), 'negative_loss': float(negative_loss),
                                 'weight_sum': float(weight.sum()), 'weight_share': float(weight.sum()) / total_weight,
                                 **positions[name]}
        return positive_logit, negative_logit, weights, records

    def train_step(self, step):
        tic = time.monotonic()
        groups = None
        if self.groups is None:
            anchors, relations, positives, negatives, weights = self.sampler.next_batch()
            device = self.device
            anchors, relations, positives, negatives = (torch.as_tensor(array).to(device)
                                                        for array in (anchors, relations, positives, negatives))
            weights = torch.as_tensor(weights).to(device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            query = self.model.encode(anchors, relations)
            positive_logit = self.model.score(query, positives.unsqueeze(1)).squeeze(1)
            negative_logit = self.model.score(query, negatives, negative=True)
        else:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            positive_logit, negative_logit, weights, groups = self.group_logits()
        loss, positive_loss, negative_loss = query_loss(positive_logit, negative_logit, weights)
        auxiliary = self.model.auxiliary_losses()
        total = loss + sum(auxiliary.values()) if auxiliary else loss
        record = {'event': 'train', 'step': step, 'query_loss': float(loss), 'positive_loss': float(positive_loss),
                  'negative_loss': float(negative_loss)}
        if groups is not None:
            record['groups'] = groups
        for name, value in auxiliary.items():
            gradients = torch.autograd.grad(value, self.parameters, retain_graph=True, allow_unused=True)
            record[f'{name}_loss'] = float(value)
            record[f'{name}_gradient_norm'] = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gradients
                                                                   if g is not None)))
        if not torch.isfinite(total):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        total.backward()
        gradient_norm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in self.parameters if p.grad is not None))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f'Nonfinite gradient at step {step}')
        self.optimizer.step()
        seconds = time.monotonic() - tic
        self.timing['step'] = max(self.timing['step'], seconds)
        record.update({'loss': float(total), 'gradient_norm': float(gradient_norm),
                       'learning_rate': self.optimizer.param_groups[0]['lr'], 'epoch': self.sampler.epoch,
                       'offset': self.sampler.offset, 'seconds': seconds})
        self.log(record)

    def resume(self):
        state = torch.load(self.path('resume.pt'), map_location='cpu', weights_only=False)
        if state['identity_sha256'] != self.identity_sha256:
            raise IdentityMismatch('resume.pt was written by another identity')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if state['scheduler'] is not None:
            raise ValueError('Unexpected scheduler state')
        self.sampler.load_state_dict(state['sampler'])
        self.restore_rng(state['rng'])
        self.curve = state['curve']
        self.best = state['best']
        self.baseline = state['baseline']
        self.artifacts = state['artifacts']
        self.timing = state['timing']
        self.elapsed_before = state['elapsed_seconds']
        last = state['step']
        self.verify_artifacts()
        self.publish()
        write_json(self.path('progress.json'), self.progress(last))
        self.trim_log(last)
        return last

    def trim_log(self, last):
        log = self.path('training.jsonl')
        if not log.exists():
            return
        validated = {entry['step'] for entry in self.curve}
        kept = []
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record['event'] == 'train' and record['step'] <= last:
                kept.append(line)
            elif record['event'] == 'validation' and record['step'] in validated:
                kept.append(line)
        data = ''.join(line + '\n' for line in kept).encode()
        atomic_write_bytes(log, lambda handle: handle.write(data))

    def pause(self, step, reason):
        self.commit(step)
        self.segment(segment_record('pause', step, reason=reason, signal=PAUSE_REQUEST['signal'],
                                    remaining_seconds=None if self.deadline_seconds is None else self.remaining(),
                                    timing=self.timing))
        return EXIT_PAUSE

    def check_completed(self):
        completion = read_json(self.path('COMPLETE.json'))
        if read_json(self.path('identity.json')) != self.identity or completion['identity_sha256'] != self.identity_sha256:
            raise IdentityMismatch('Completed run belongs to another identity')
        for name, digest in completion['files'].items():
            if sha(self.path(name)) != digest:
                raise ValueError(f'Completed artifact changed: {name}')
        return EXIT_COMPLETE

    def complete(self):
        if pending_validation(self.curve, self.total, self.total, self.frequency):
            raise ValueError('Training lacks its final scheduled validation')
        self.verify_artifacts()
        self.publish()
        files = {}
        for readout in READOUTS:
            files[f'selected-{readout}.pt'] = sha(self.path(f'selected-{readout}.pt'))
            files[f'selected-{readout}-validation.npz'] = sha(self.path(f'selected-{readout}-validation.npz'))
        for step, record in sorted(self.artifacts.items(), key=lambda item: int(item[0])):
            for kind in ('checkpoint', 'validation'):
                files[record[kind]['file']] = record[kind]['sha256']
        files['identity.json'] = sha(self.path('identity.json'))
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if self.device.type == 'cuda' else None
        write_json(self.path('COMPLETE.json'), {
            'identity_sha256': self.identity_sha256, 'method': self.config['method'],
            'membership_mode': self.config.get('membership_mode'), 'auxiliary': self.config['auxiliary'],
            'auxiliary_slots': self.config['auxiliary_slots'], 'steps': self.total,
            'validation_steps': [entry['step'] for entry in self.curve], 'best': self.best, 'baseline': self.baseline,
            'files': files, 'elapsed_seconds': self.elapsed(), 'peak_gpu_gib_this_segment': peak})
        self.segment(segment_record('complete', self.total, timing=self.timing))
        return EXIT_COMPLETE

    def history(self):
        """Paths that show committed or partial progress; identity, segments, lock and top-level temporaries excluded."""
        found = [name for name in ('progress.json',) if self.path(name).exists()]
        log = self.path('training.jsonl')
        if log.exists() and log.stat().st_size > 0:
            found.append('training.jsonl')
        found += sorted(path.name for path in self.run_dir.glob('selected-*'))
        for directory in ('checkpoints', 'validation'):
            if self.path(directory).is_dir():
                found += sorted(str(path.relative_to(self.run_dir)) for path in self.path(directory).rglob('*')
                                if not path.is_dir())
        return found

    def execute(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path('COMPLETE.json').exists():
            return self.check_completed()
        if self.path('identity.json').exists():
            if read_json(self.path('identity.json')) != self.identity:
                raise IdentityMismatch('Run directory identity differs; use a new run directory')
        if not self.path('resume.pt').exists():
            found = self.history()
            if found:
                raise OrphanedHistory(f'Run has history but no resume.pt; use a new run directory: {found}')
        if not self.path('identity.json').exists():
            write_json(self.path('identity.json'), self.identity)
        self.build()
        resumed = self.path('resume.pt').exists()
        step = self.resume() if resumed else 0
        self.segment(segment_record('start', step, resumed=resumed, device=str(self.device),
                                    deadline_seconds=self.deadline_seconds))
        if not resumed or pending_validation(self.curve, step, self.total, self.frequency):
            if self.stop_before_validation == step and resumed:
                return self.pause(step, 'stop_before_validation')
            if not self.can_validate():
                return self.pause(step, 'deadline')
            try:
                self.validate_step(step)
            except ValidationInterrupted:
                return self.pause(step, 'signal')
            if resumed and self.stop_after == step and step < self.total:
                return self.pause(step, 'stop_after')
        while step < self.total:
            if self.pause_requested():
                return self.pause(step, 'signal')
            if not self.can_train():
                return self.pause(step, 'deadline')
            step += 1
            self.train_step(step)
            if step % self.frequency == 0 or step == self.total:
                if self.stop_before_validation == step:
                    return self.pause(step, 'stop_before_validation')
                if self.pause_requested():
                    return self.pause(step, 'signal')
                if not self.can_validate():
                    return self.pause(step, 'deadline')
                try:
                    self.validate_step(step)
                except ValidationInterrupted:
                    return self.pause(step, 'signal')
            elif (step - (self.last_commit[0] or 0) >= self.config['checkpoint_every_steps']
                  or time.monotonic() - self.last_commit[1] >= self.config['checkpoint_every_seconds']):
                self.commit(step)
            if self.stop_after == step and step < self.total:
                return self.pause(step, 'stop_after')
        return self.complete()


def acquire_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / '.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='JSON run configuration')
    parser.add_argument('--run-dir', type=Path, required=True, help='Output directory for this run only')
    parser.add_argument('--device', default='cuda', help='torch device, e.g. cuda or cpu')
    parser.add_argument('--deadline-seconds', type=float, default=None,
                        help='Pause with exit 75 before this many seconds since process start would be exceeded')
    parser.add_argument('--deadline-safety', type=float, default=1.5,
                        help='Multiplier on the measured step, validation and commit costs used by the deadline check')
    parser.add_argument('--stop-after', type=int, default=None, help='Technical check: pause after this update')
    parser.add_argument('--stop-before-validation', type=int, default=None,
                        help='Technical check: pause after this update with its scheduled validation pending')
    args = parser.parse_args(argv)
    raw = read_json(args.config)
    config, data_dir = resolve_config(raw, args.config)
    lock = acquire_lock(args.run_dir)
    if lock is None:
        print(json.dumps({'status': 'locked', 'run_dir': str(args.run_dir)}), flush=True)
        return EXIT_LOCKED
    previous = install_signal_handlers()
    try:
        run = Run(config, data_dir, args.run_dir, args.device, args.deadline_seconds, args.stop_after,
                  args.stop_before_validation, args.deadline_safety)
        code = run.execute()
        print(json.dumps({'status': {EXIT_COMPLETE: 'complete', EXIT_PAUSE: 'paused'}[code],
                          'run_dir': str(args.run_dir)}), flush=True)
        return code
    finally:
        restore_signal_handlers(previous)
        PAUSE_REQUEST['signal'] = None
        lock.close()


if __name__ == '__main__':
    sys.exit(main())
