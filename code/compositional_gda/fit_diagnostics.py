"""Read-only training-fit diagnostic for a completed compositional GDA run.

Scores a bounded, deterministic sample of TRAINING-graph queries with a saved
checkpoint and reports filtered ranks of training answers. It measures how well
the model fits edges and two-hop paths it was trained on. It is not a
validation or generalization measurement: no validation or test row, panel or
profile is scored.

Six query structures are sampled, each against an explicit typed candidate set:

- ``1p_associated_with``: gene -> associated_with, over training diseases;
- ``1p_has_symptom``: training disease -> has_symptom, over training phenotypes
  (the HP tails of has_symptom in graph.npz);
- ``1p_inverse_has_symptom``: phenotype -> inverse:has_symptom, over training diseases;
- ``1p_inverse_associated_with``: training disease -> inverse:associated_with,
  over the 13,625 frozen candidate genes;
- ``2p_forward``: gene -> associated_with -> has_symptom, over training phenotypes;
- ``2p_inverse``: phenotype -> inverse:has_symptom -> inverse:associated_with,
  over the candidate genes.

Answer sets come only from graph.npz triples of the four readout relations; the
two-hop answer set is the union over every intermediate training disease. For
each structure, up to ``--queries-per-relation`` anchors are drawn without
replacement from all anchors with a non-empty answer set, and one positive is
drawn uniformly from the anchor's full answer set, from a generator seeded by
``(seed, structure index)``. The sample depends only on the prepared data, the
seed and the count, so its SHA-256 is identical across models.

Every candidate is scored with ``score(..., negative=True)``, the form used to
rank candidates at validation. The positive's rank is filtered (all other known
answers of the query are removed), raw (no cross-query calibration) and uses
average-rank ties.

The run directory is never written. The checkpoint is loaded with
``torch.load(weights_only=True)`` after its SHA-256 matches ``COMPLETE.json``;
its step, identity hash and configuration must match the run's ``identity.json``,
the prepared data must match the run's recorded digests, and ``models.py`` must
match the hash the run recorded (``--allow-model-source-drift`` downgrades that
to a recorded warning). Other trainer modules may differ from the run's; the
differences are recorded, because this is scoring, not a resumed run.

Results are written to ``<output>.partial`` and renamed to ``<output>`` on
success. Neither may exist beforehand.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from compositional_gda.common import (atomic_savez, canonical_sha, installed_versions, read_json, sha, source_hashes,
                                      write_json)
from compositional_gda.data import PreparedData
from compositional_gda.models import build_model

FORMAT = 'compositional-gda-fit-diagnostic/1'
LABEL = ('training-fit diagnostic on a sampled subset of training-graph queries; not validation and not '
         'generalization; raw scores, filtered against every training answer of the query; no calibration')

STRUCTURES = (
    ('1p_associated_with', ('associated_with',), 'training_diseases'),
    ('1p_has_symptom', ('has_symptom',), 'training_phenotypes'),
    ('1p_inverse_has_symptom', ('inverse:has_symptom',), 'training_diseases'),
    ('1p_inverse_associated_with', ('inverse:associated_with',), 'candidate_genes'),
    ('2p_forward', ('associated_with', 'has_symptom'), 'training_phenotypes'),
    ('2p_inverse', ('inverse:has_symptom', 'inverse:associated_with'), 'candidate_genes'),
)

CANDIDATE_LABELS = {
    'training_diseases': 'diseases of training.tsv (tails of associated_with in graph.npz)',
    'training_phenotypes': 'tails of has_symptom in graph.npz (HP phenotypes of training diseases)',
    'candidate_genes': 'candidates.txt, the frozen gene candidates used at validation, in file order',
}

CHECKPOINT_KEYS = {'step', 'identity_sha256', 'config', 'model'}


class ProvenanceError(RuntimeError):
    """The run, checkpoint, data or source does not match what the run recorded."""


class Log:
    """Timestamped lines to a file and stdout."""

    def __init__(self, path):
        self.handle = Path(path).open('a')

    def __call__(self, message, **fields):
        line = json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'message': message, **fields},
                          default=str)
        self.handle.write(line + '\n')
        self.handle.flush()
        print(line, flush=True)

    def close(self):
        self.handle.close()


def answer_table(triples, relation):
    """``{head: sorted unique tails}`` for one relation id."""
    rows = triples[triples[:, 1] == relation][:, [0, 2]]
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    heads, starts = np.unique(rows[:, 0], return_index=True)
    ends = np.append(starts[1:], len(rows))
    return {int(head): np.unique(rows[start:end, 1]) for head, start, end in zip(heads, starts, ends)}


def compose(first, second):
    """Existential two-hop answers: the union of ``second`` over every intermediate of ``first``."""
    result = {}
    for anchor, middles in first.items():
        parts = [second[int(middle)] for middle in middles if int(middle) in second]
        if parts:
            result[anchor] = np.unique(np.concatenate(parts))
    return result


def candidate_sets(data, tables):
    entities = data.entities
    training_diseases = sorted({disease for _, disease in data.rows['training']})
    held_out = {disease for split in ('validation', 'test') for _, disease in data.rows[split]}
    if held_out & set(training_diseases):
        raise ProvenanceError('Training diseases overlap validation or test diseases')
    disease_ids = np.array(sorted(entities[disease] for disease in training_diseases), dtype=np.int64)
    phenotype_ids = np.unique(np.concatenate(list(tables['has_symptom'].values()))).astype(np.int64)
    gene_ids = np.array([entities[gene] for gene in data.candidates], dtype=np.int64)
    names = {index: name for name, index in entities.items()}
    associated_tails = np.unique(np.concatenate(list(tables['associated_with'].values())))
    checks = {
        'training_diseases_equal_associated_with_tails': bool(np.array_equal(disease_ids, associated_tails)),
        'training_diseases_equal_has_symptom_heads': bool(np.array_equal(
            disease_ids, np.array(sorted(tables['has_symptom']), dtype=np.int64))),
        'training_phenotypes_with_hp_iri': int(sum('/HP_' in names[int(index)] for index in phenotype_ids)),
        'training_diseases_disjoint_from_held_out': True,
    }
    return {'training_diseases': disease_ids, 'training_phenotypes': phenotype_ids, 'candidate_genes': gene_ids}, checks


def structure_answers(data):
    tables = {name: answer_table(data.triples, data.relations[name])
              for name in ('associated_with', 'has_symptom', 'inverse:has_symptom', 'inverse:associated_with')}
    answers = {}
    for name, path, _ in STRUCTURES:
        answers[name] = tables[path[0]] if len(path) == 1 else compose(tables[path[0]], tables[path[1]])
    forward = {(anchor, int(answer)) for anchor, values in answers['2p_forward'].items() for answer in values}
    inverse = {(int(answer), anchor) for anchor, values in answers['2p_inverse'].items() for answer in values}
    if forward != inverse:
        raise ProvenanceError('Forward two-hop pairs differ from the transposed inverse pairs')
    return tables, answers, {'two_hop_forward_equals_transposed_inverse': True, 'two_hop_pairs': len(forward)}


def build_sample(data, count, seed):
    """Deterministic sample of training queries; returns arrays, candidate sets, per-structure metadata and checks."""
    tables, answers, checks = structure_answers(data)
    candidates, candidate_checks = candidate_sets(data, tables)
    checks.update(candidate_checks)
    arrays = {}
    metadata = {}
    for index, (name, path, candidate_name) in enumerate(STRUCTURES):
        table = answers[name]
        pool = np.array(sorted(table), dtype=np.int64)
        candidate_ids = candidates[candidate_name]
        position = {int(entity): column for column, entity in enumerate(candidate_ids)}
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(index,))))
        anchors = np.sort(generator.choice(pool, size=min(count, len(pool)), replace=False)).astype(np.int64)
        positives = np.empty(len(anchors), dtype=np.int64)
        offsets = [0]
        columns = []
        for row, anchor in enumerate(anchors):
            known = table[int(anchor)]
            positives[row] = known[generator.integers(len(known))]
            missing = [int(entity) for entity in known if int(entity) not in position]
            if missing:
                raise ProvenanceError(f'{name}: answers outside candidate set {candidate_name}: {missing[:5]}')
            columns.extend(position[int(entity)] for entity in known)
            offsets.append(len(columns))
        arrays[f'{name}_anchors'] = anchors
        arrays[f'{name}_path'] = np.array([data.relations[relation] for relation in path], dtype=np.int64)
        arrays[f'{name}_positives'] = positives
        arrays[f'{name}_positive_columns'] = np.array([position[int(entity)] for entity in positives], dtype=np.int64)
        arrays[f'{name}_answer_offsets'] = np.array(offsets, dtype=np.int64)
        arrays[f'{name}_answer_columns'] = np.array(columns, dtype=np.int64)
        arrays[f'{name}_candidates'] = candidate_ids
        sizes = np.array([len(table[int(anchor)]) for anchor in pool])
        metadata[name] = {'path': list(path), 'candidate_set': candidate_name,
                          'candidate_set_label': CANDIDATE_LABELS[candidate_name],
                          'candidate_count': int(len(candidate_ids)),
                          'candidate_sha256': hashlib.sha256(candidate_ids.tobytes()).hexdigest(),
                          'anchor_pool': int(len(pool)), 'answer_pairs_in_pool': int(sizes.sum()),
                          'sampled_queries': int(len(anchors)), 'spawn_key': [index]}
    return arrays, metadata, checks


def sample_sha256(arrays, data, count, seed):
    """SHA-256 over the prepared-data identity, seed, count and every sample array (name, dtype, shape, bytes)."""
    digest = hashlib.sha256(json.dumps({'format': FORMAT, 'data_identity_sha256': data.identity_sha256,
                                        'seed': seed, 'queries_per_relation': count,
                                        'structures': STRUCTURES}, sort_keys=True).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(f'{key}|{value.dtype.str}|{value.shape}'.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def expected_step(name, completion):
    match = re.fullmatch(r'checkpoints/step-(\d{6})\.pt', name)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r'selected-(forward|inverse)\.pt', name)
    if match:
        return int(completion['best'][match.group(1)]['step'])
    raise ProvenanceError(f'Unrecognised checkpoint name {name}')


def verify_run(run_dir, checkpoint_name, data, allow_model_source_drift, log):
    """Check identity, completion, checkpoint, data and source; return the identity, payload and provenance."""
    run_dir = Path(run_dir)
    identity_path = run_dir / 'identity.json'
    completion_path = run_dir / 'COMPLETE.json'
    if not completion_path.exists():
        raise ProvenanceError(f'{run_dir} has no COMPLETE.json; only completed runs are scored')
    identity = read_json(identity_path)
    completion = read_json(completion_path)
    identity_sha256 = canonical_sha(identity)
    if identity_sha256 != completion['identity_sha256']:
        raise ProvenanceError('identity.json does not hash to COMPLETE.json identity_sha256')
    if sha(identity_path) != completion['files'].get('identity.json'):
        raise ProvenanceError('identity.json file digest differs from COMPLETE.json')
    config = identity['config']
    if checkpoint_name is None:
        checkpoint_name = f'checkpoints/step-{config["steps"]:06d}.pt'
    if checkpoint_name not in completion['files']:
        raise ProvenanceError(f'{checkpoint_name} is not recorded in COMPLETE.json')
    checkpoint_path = run_dir / checkpoint_name
    checkpoint_sha256 = sha(checkpoint_path)
    if checkpoint_sha256 != completion['files'][checkpoint_name]:
        raise ProvenanceError(f'{checkpoint_name} differs from its COMPLETE.json digest')
    log('checkpoint digest verified', checkpoint=checkpoint_name, sha256=checkpoint_sha256)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_KEYS:
        raise ProvenanceError(f'Unexpected checkpoint structure: {sorted(payload) if isinstance(payload, dict) else type(payload)}')
    step = expected_step(checkpoint_name, completion)
    if payload['step'] != step:
        raise ProvenanceError(f'Checkpoint step {payload["step"]} differs from expected {step}')
    if payload['identity_sha256'] != identity_sha256:
        raise ProvenanceError('Checkpoint identity_sha256 differs from the run identity')
    if payload['config'] != config:
        raise ProvenanceError('Checkpoint config differs from identity.json config')
    if data.identity_sha256 != identity['data']['identity_sha256']:
        raise ProvenanceError('Prepared identity.json differs from the one the run trained on')
    recorded = identity['data']['files']
    if {name: data.file_digests.get(name) for name in recorded} != recorded:
        raise ProvenanceError('Prepared file digests differ from those the run recorded')
    current = source_hashes()
    archived = identity['source']['trainer']
    drift = sorted(name for name in set(current) | set(archived) if current.get(name) != archived.get(name))
    if 'models.py' in drift and not allow_model_source_drift:
        raise ProvenanceError('models.py differs from the source the run recorded')
    if drift:
        log('trainer source differs from the run; scoring is read-only, not a resume', modules=drift)
    provenance = {
        'run_dir': str(run_dir.resolve()), 'run_name': run_dir.name, 'mode': 'read-only scoring; not a resume',
        'identity_sha256': identity_sha256, 'identity_file_sha256': sha(identity_path),
        'complete_file_sha256': sha(completion_path), 'checkpoint': checkpoint_name,
        'checkpoint_sha256': checkpoint_sha256, 'checkpoint_bytes': checkpoint_path.stat().st_size,
        'checkpoint_step': step, 'run_steps': config['steps'], 'config': config,
        'method_label': identity.get('method_label'),
        'source': {'recorded_by_run': archived, 'current': current, 'differing_modules': drift,
                   'models_py_matches_run': 'models.py' not in drift,
                   'fit_diagnostics_sha256': sha(Path(__file__).resolve())},
        'run_versions': identity.get('versions'), 'run_numerics': identity.get('numerics'),
    }
    return identity, payload, provenance


def load_model(config, payload, data):
    model = build_model(config, data.num_entities, data.num_relations)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    saved = {name: tuple(tensor.shape) for name, tensor in payload['model'].items()}
    if expected != saved:
        raise ProvenanceError(f'Checkpoint tensors differ from the rebuilt model: '
                              f'{sorted(set(expected) ^ set(saved)) or "shapes"}')
    for name, tensor in payload['model'].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ProvenanceError(f'Nonfinite checkpoint tensor {name}')
    model.load_state_dict(payload['model'], strict=True)
    return model


def configure_numerics(device):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'deterministic_algorithms': True, 'tf32': False, 'cudnn_benchmark': False,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if device.type == 'cuda' else None}


def score_matrix(model, anchors, path, candidates, device, query_chunk, chunk_elements, dim):
    """float32 ``[len(anchors), len(candidates)]`` of ``score(negative=True)``; at most ``chunk_elements`` pair elements per block."""
    result = np.empty((len(anchors), len(candidates)), dtype=np.float32)
    targets = torch.as_tensor(candidates, device=device)
    relations = torch.as_tensor(path, device=device)
    with torch.no_grad():
        for first in range(0, len(anchors), query_chunk):
            chunk = torch.as_tensor(anchors[first:first + query_chunk], device=device)
            query = model.encode(chunk, relations.unsqueeze(0).expand(len(chunk), -1))
            width = max(1, chunk_elements // (len(chunk) * dim))
            for start in range(0, len(targets), width):
                block = targets[start:start + width].unsqueeze(0)
                result[first:first + len(chunk), start:start + block.shape[1]] = model.score(
                    query, block, negative=True).float().cpu().numpy()
    return result


def filtered_ranks(scores, positive_columns, offsets, columns):
    """Average-tie rank of each positive among itself and every candidate that is not a known answer."""
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite scores')
    ranks = np.empty(len(scores), dtype=np.float64)
    pool = np.empty(len(scores), dtype=np.int64)
    for row, values in enumerate(scores):
        keep = np.ones(values.shape[0], dtype=bool)
        keep[columns[offsets[row]:offsets[row + 1]]] = False
        competitors = values[keep]
        target = values[positive_columns[row]]
        ranks[row] = 1 + (competitors > target).sum() + 0.5 * (competitors == target).sum()
        pool[row] = keep.sum() + 1
    return ranks, pool


def summarize(ranks, pool):
    pool = pool.astype(np.float64)
    return {'count': int(len(ranks)), 'mrr': float(np.mean(1 / ranks)), 'mr': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)), 'hits@1': float(np.mean(ranks <= 1)),
            'hits@3': float(np.mean(ranks <= 3)), 'hits@10': float(np.mean(ranks <= 10)),
            'hits@100': float(np.mean(ranks <= 100)), 'mean_filtered_pool': float(pool.mean()),
            'random_expected_mr': float(np.mean((pool + 1) / 2)),
            'random_expected_hits@10': float(np.mean(np.minimum(10, pool) / pool))}


def environment(device):
    record = {'host': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'), 'device': str(device),
              'versions': installed_versions()}
    if device is not None and device.type == 'cuda':
        record['device_name'] = torch.cuda.get_device_name(device)
    return record


def run(args, output, log):
    started = time.monotonic()
    device = None if args.sample_only else torch.device(args.device)
    if device is not None and device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
    numerics = configure_numerics(device) if device is not None else None
    data = PreparedData(args.data_dir)
    log('prepared data verified', identity_sha256=data.identity_sha256)
    arrays, structures, checks = build_sample(data, args.queries_per_relation, args.seed)
    digest = sample_sha256(arrays, data, args.queries_per_relation, args.seed)
    log('sample built', sample_sha256=digest, queries={name: meta['sampled_queries'] for name, meta in structures.items()})
    if args.expect_sample_sha256 and args.expect_sample_sha256 != digest:
        raise ProvenanceError(f'Sample SHA-256 {digest} differs from expected {args.expect_sample_sha256}')
    common = {'format': FORMAT, 'label': LABEL, 'queries_per_relation': args.queries_per_relation, 'seed': args.seed,
              'sample_sha256': digest, 'data_dir': str(Path(args.data_dir).resolve()),
              'data_identity_sha256': data.identity_sha256, 'data_files': data.file_digests,
              'sample_checks': checks, 'structures': structures,
              'ranking': {'score': 'model.score(query, candidates, negative=True)', 'ties': 'average rank',
                          'filter': 'every other training answer of the query removed from the candidate set',
                          'calibration': 'none (raw scores)'},
              'scope': {'scored': 'training-graph queries only',
                        'not_scored': 'validation and test rows, panels and disease profiles'}}
    if args.sample_only:
        atomic_savez(output / 'sample.npz', **arrays)
        write_json(output / 'provenance.json', {**common, 'environment': environment(device),
                                                'fit_diagnostics_sha256': sha(Path(__file__).resolve())})
        write_json(output / 'summary.json', {'format': FORMAT, 'label': LABEL, 'sample_sha256': digest,
                                             'structures': structures, 'seconds': time.monotonic() - started})
        return digest
    identity, payload, run_provenance = verify_run(args.run_dir, args.checkpoint, data,
                                                   args.allow_model_source_drift, log)
    config = identity['config']
    model = load_model(config, payload, data).to(device)
    model.eval()
    del payload
    log('model loaded', method=config['method'], membership_mode=config.get('membership_mode'),
        step=run_provenance['checkpoint_step'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    results = {}
    scores = {}
    timings = {}
    for name, _, _ in STRUCTURES:
        tic = time.monotonic()
        matrix = score_matrix(model, arrays[f'{name}_anchors'], arrays[f'{name}_path'], arrays[f'{name}_candidates'],
                              device, config['query_chunk'], config['score_chunk_elements'], config['dim'])
        ranks, pool = filtered_ranks(matrix, arrays[f'{name}_positive_columns'], arrays[f'{name}_answer_offsets'],
                                     arrays[f'{name}_answer_columns'])
        arrays[f'{name}_ranks'] = ranks
        arrays[f'{name}_filtered_pool'] = pool
        arrays[f'{name}_positive_scores'] = matrix[np.arange(len(matrix)), arrays[f'{name}_positive_columns']]
        scores[f'{name}_scores'] = matrix
        timings[name] = time.monotonic() - tic
        results[name] = summarize(ranks, pool)
        log('structure scored', structure=name, seconds=timings[name], **{key: results[name][key]
                                                                           for key in ('count', 'mrr', 'hits@10')})
    atomic_savez(output / 'fit.npz', **arrays)
    if not args.no_score_matrices:
        atomic_savez(output / 'scores.npz', **scores)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else None
    write_json(output / 'provenance.json', {**common, 'run': run_provenance, 'numerics': numerics,
                                            'environment': environment(device), 'peak_gpu_gib': peak,
                                            'score_matrices_saved': not args.no_score_matrices,
                                            'scoring_chunks': {'query_chunk': config['query_chunk'],
                                                               'score_chunk_elements': config['score_chunk_elements']}})
    write_json(output / 'summary.json', {
        'format': FORMAT, 'label': LABEL, 'run_name': run_provenance['run_name'], 'method': config['method'],
        'membership_mode': config.get('membership_mode'), 'checkpoint': run_provenance['checkpoint'],
        'checkpoint_step': run_provenance['checkpoint_step'], 'checkpoint_sha256': run_provenance['checkpoint_sha256'],
        'sample_sha256': digest, 'queries_per_relation': args.queries_per_relation, 'seed': args.seed,
        'structures': {name: {**structures[name], 'metrics': results[name], 'seconds': timings[name]}
                       for name, _, _ in STRUCTURES},
        'source_differing_modules': run_provenance['source']['differing_modules'],
        'seconds': time.monotonic() - started})
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, help='Completed run directory (read only)')
    parser.add_argument('--data-dir', type=Path, required=True, help='Prepared fold the run trained on')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
    parser.add_argument('--device', help='torch device for scoring, e.g. cuda; required unless --sample-only')
    parser.add_argument('--queries-per-relation', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', help='Checkpoint path relative to the run directory, as listed in '
                                             'COMPLETE.json (default: the final step checkpoint)')
    parser.add_argument('--expect-sample-sha256', help='Fail unless the sample hashes to this value')
    parser.add_argument('--no-score-matrices', action='store_true', help='Do not save the full candidate score matrices')
    parser.add_argument('--sample-only', action='store_true', help='Build and save the sample on CPU; no model')
    parser.add_argument('--allow-model-source-drift', action='store_true',
                        help='Record, instead of refusing, a models.py that differs from the run')
    args = parser.parse_args(argv)
    if args.queries_per_relation <= 0:
        parser.error('--queries-per-relation must be positive')
    if not args.sample_only and (args.run_dir is None or args.device is None):
        parser.error('--run-dir and --device are required unless --sample-only')
    if args.device is not None and torch.device(args.device).type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    output = args.output
    partial = output.with_name(output.name + '.partial')
    for path in (output, partial):
        if path.exists():
            parser.error(f'{path} exists; choose a new output directory')
    partial.mkdir(parents=True)
    log = Log(partial / 'fit.log')
    try:
        log('start', argv=sys.argv if argv is None else argv, label=LABEL)
        digest = run(args, partial, log)
        log('done', sample_sha256=digest)
    except BaseException as error:
        log('failed', error=repr(error))
        raise
    finally:
        log.close()
    partial.rename(output)
    print(json.dumps({'status': 'complete', 'output': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
