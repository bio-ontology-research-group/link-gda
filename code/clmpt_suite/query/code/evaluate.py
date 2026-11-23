"""Saved VALIDATION evaluation isolated from the continuing training state.

Adapted from code/clmpt_boundary_queries/query/code/evaluate.py. Differences: case, association and
candidate counts come from the protocol's admitted-bank record and are asserted against the bank (no
hardcoded 545/586); scores are produced through the encoder's own representation so the independent-branch
and plain-mean encoders are evaluated by their training semantics; every archive records the score
semantics. Raw and own-LOO calibrated metrics are computed from the same saved matrix. No TEST access.
"""
import copy
import hashlib
import json
import os
import pathlib
import random
import time

import numpy as np
import torch

from metrics import calibrate, measure
from model import SCORE_SEMANTICS
from state_compare import compare


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def profile_query(hp_ids, relations):
    return {'type': 'profile', 'hp_ids': list(hp_ids), 'constants': {'s' + str(i): h for i, h in enumerate(hp_ids)},
            'atoms': [['s' + str(i), relations['inverse_has_symptom'], 'e1'] for i in range(len(hp_ids))] + [['e1', relations['inverse_associated_with'], 'f']]}


def verify_split(rows, spec, prefix):
    assert len(rows) == spec[prefix + '_cases'], (len(rows), spec[prefix + '_cases'])
    assert sum(len(row['gene_ids']) for row in rows) == spec[prefix + '_associations']


def score_cases(model, rows, bank, should_stop=None, chunk_size=32):
    """Representations and raw scores for clinical cases, or None if asked to stop."""
    candidate = torch.tensor(bank['candidate_ids'], device=model.device)
    queries = [profile_query(row['hp_ids'], bank['relations']) for row in rows]
    representations, scores = [], []
    with torch.no_grad():
        for start in range(0, len(queries), chunk_size):
            if should_stop is not None and should_stop():
                return None
            representation = model.profile_representation(queries[start:start + chunk_size])
            representations.append(representation.cpu().numpy())
            scores.append(model.scores_from_representation(representation, candidate).cpu().numpy())
    return np.concatenate(representations), np.concatenate(scores)


def empty_control(model, bank):
    """Phenotype-free query encoded by the shared path for every encoder; cosine score."""
    empty = {'type': 'empty', 'hp_ids': [], 'constants': {}, 'atoms': [['e1', bank['relations']['inverse_associated_with'], 'f']]}
    candidate = torch.tensor(bank['candidate_ids'], device=model.device)
    with torch.no_grad():
        representation = model.encode([empty])
        genes = torch.nn.functional.normalize(model.nbp._entity_embedding(candidate), dim=-1)
        score = torch.nn.functional.normalize(representation, dim=-1) @ genes.T
    return representation.cpu().numpy(), score.cpu().numpy()


def metrics_for(arrays, answers, bank, seen):
    result = {}
    for control, key in (('real', 'validation_real'), ('shuffled', 'validation_shuffled'), ('empty', 'validation_empty')):
        result[control] = {'raw': measure(arrays[key], answers, bank['candidate_ids'], seen_genes=seen)}
        if control != 'empty':
            result[control]['calibrated'] = measure(calibrate(arrays[key]), answers, bank['candidate_ids'], seen_genes=seen)
    return result


def evaluate(model, sampler, bank, spec, output, step, identity_sha256, checkpoint_sha256, should_stop=None):
    output = pathlib.Path(output)
    output.mkdir(exist_ok=True)
    prefix = output / ('fit-step-%06d' % step)
    archive, metadata = prefix.with_suffix('.npz'), prefix.with_suffix('.json')
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        assert previous['identity_sha256'] == identity_sha256 and previous['step'] == step and previous['checkpoint_sha256'] == checkpoint_sha256
        assert sha(archive) == previous['scores_sha256']
        return previous
    for orphan in (archive, archive.with_suffix('.npz.tmp'), metadata.with_suffix('.json.tmp')):
        if orphan.exists():
            recovery = output / 'orphaned'
            recovery.mkdir(exist_ok=True)
            os.replace(orphan, recovery / (str(time.time_ns()) + '-' + orphan.name))
    before = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state().clone(),
              'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()], 'sampler': copy.deepcopy(sampler.state())}
    was_training = model.training
    versions = [p._version for p in model.parameters()]
    begun = time.time()
    model.eval()
    try:
        rows = bank['validation']
        verify_split(rows, spec, 'validation')
        assert len(bank['candidate_ids']) == spec['candidate_genes'] == len(set(bank['candidate_ids']))
        assert len(bank['validation_shuffle']) == len(rows) and all(0 <= int(i) < len(rows) for i in bank['validation_shuffle'])
        scored = score_cases(model, rows, bank, should_stop)
        if scored is None:
            return None
        representation, real = scored
        empty_representation, empty_score = empty_control(model, bank)
        assert np.isfinite(real).all() and np.isfinite(empty_score).all()
        arrays = {'validation_real': real, 'validation_shuffled': real[bank['validation_shuffle']], 'validation_empty': np.repeat(empty_score, len(rows), axis=0),
                  'validation_representation': representation, 'empty_representation': empty_representation, 'candidate_ids': np.array(bank['candidate_ids']),
                  'score_semantics': np.array(SCORE_SEMANTICS[model.encoder])}
        if should_stop is not None and should_stop():
            return None
        answers = [row['gene_ids'] for row in rows]
        seen = {g for row in bank['training_profiles'] for g in row['answers']}
        metrics = metrics_for(arrays, answers, bank, seen)
        after = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'sampler': sampler.state()}
        invariant = compare(before, after)
        assert invariant['exact'] and versions == [p._version for p in model.parameters()]
        if should_stop is not None and should_stop():
            return None
        temporary = archive.with_suffix('.npz.tmp')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, archive)
        result = {'step': step, 'identity_sha256': identity_sha256, 'checkpoint_sha256': checkpoint_sha256, 'validation': metrics,
                  'encoder': model.encoder, 'score_semantics': SCORE_SEMANTICS[model.encoder], 'counts': {key: spec[key] for key in ('validation_cases', 'validation_associations', 'candidate_genes')},
                  'raw_mrr': metrics['real']['raw']['answer_micro']['mrr'], 'scores_sha256': sha(archive), 'RNG_sampler_unchanged': invariant,
                  'parameter_versions_unchanged': True, 'elapsed_seconds': time.time() - begun,
                  'scope': 'Unfiltered deterministic average-rank metrics on VALIDATION; raw and own-LOO calibrated views from one saved matrix, selected separately; '
                           'empty control uses the shared joint-query path for every encoder; no TEST.'}
        temporary = metadata.with_suffix('.json.tmp')
        with temporary.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata)
        return result
    finally:
        random.setstate(before['python'])
        np.random.set_state(before['numpy'])
        torch.set_rng_state(before['torch'])
        torch.cuda.set_rng_state_all(before['cuda'])
        model.train(was_training)
