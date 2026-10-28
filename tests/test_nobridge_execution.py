"""Exercise the real no-bridge CLI, including its own callback and score exporter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from pykeen.models import TransD
from pykeen.triples import TriplesFactory
from training.retrain_nobridge_kge import FastIndigenaScorer

ROOT = Path(__file__).resolve().parents[1]


def test_indigena_empty_gene_annotations_score_zero(tmp_path):
    factory = TriplesFactory.from_labeled_triples(np.array([
        ['g0', 'causes_phenotype', 'p0'], ['g1', 'causes_phenotype', 'p1']
    ]), create_inverse_triples=True)
    model = TransD(triples_factory=factory, embedding_dim=4, relation_dim=4, random_seed=4)
    annotations = tmp_path/'annotations.csv'
    pd.DataFrame({'Gene': ['g0'], 'Phenotype': ['p0']}).to_csv(annotations, index=False)
    scorer = FastIndigenaScorer(factory, ['g0', 'g1'], {'d0': ['p0']}, annotations, None)
    scores = scorer.score_diseases(model, ['d0'])['d0']
    assert np.isfinite(scores).all() and scores[1] == 0
    emb = model.entity_representations[0](indices=None).detach()
    p = emb[factory.entity_to_id['p0']]
    assert scores[0] == pytest.approx(float(torch.sigmoid(p @ p)))


def test_t1_cli_interruption_scores_and_completion_integrity(tmp_path):
    data = tmp_path/'data'/'fold0'
    (data/'raw').mkdir(parents=True)
    genes = [f'g{i}' for i in range(8)]
    (data/'candidates.txt').write_text('\n'.join(genes)+'\n')
    triples = [[g, 'causes_phenotype', f'p{i%2}'] for i,g in enumerate(genes)]
    triples += [['g0', 'associated_with', 'd0'], ['d0', 'has_symptom', 'p0']]
    (data/'raw/train.txt').write_text('\n'.join('\t'.join(t) for t in triples)+'\n')
    (data/'profiles.json').write_text(json.dumps({f'd{i}': [f'p{i%2}'] for i in range(6)}))
    for split,indices in [('training', [0,1]), ('validation', [2,3]), ('test', [4,5])]:
        pd.DataFrame({'Gene': [f'g{i}' for i in indices], 'Disease': [f'd{i}' for i in indices]}).to_csv(
            data/f'{split}.tsv', sep='\t', index=False)
    command = [sys.executable, str(ROOT/'code/training/retrain_nobridge_kge.py'),
               '--fold', '0', '--condition', 'T1', '--data-root', str(tmp_path/'data'),
               '--code-root', str(ROOT), '--epochs', '25', '--sub-batch-size', '8',
               '--device', 'cpu', '--expected-candidates', '8', '--checkpoint-frequency', '5']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def invoke(root, extra=(), success=True):
        result = subprocess.run([*command, '--out-root', str(root), *extra],
                                text=True, capture_output=True, env=env, timeout=90)
        if success: assert result.returncode == 0, result.stdout+'\n'+result.stderr
        else: assert result.returncode != 0
        return result
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    invoke(full)
    invoke(resumed, ['--stop-after-epoch', '21'])
    assert not (resumed/'fold0/t1_completion.json').exists()
    assert not list((resumed/'fold0').glob('*.npz'))
    invoke(resumed)
    assert 'verified complete' in invoke(resumed).stdout
    for arm in ['raw_head', 'calibrated_head', 'raw_mr', 'calibrated_mr']:
        a = torch.load(full/f'fold0/ckpt_t1_nobridge_{arm}.pt', weights_only=True)
        b = torch.load(resumed/f'fold0/ckpt_t1_nobridge_{arm}.pt', weights_only=True)
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for filename in ['linkgda_t1_nobridge_cal_head_scores.npz', 'linkgda_t1_nobridge_raw_head_scores.npz']:
        with np.load(full/'fold0'/filename) as a, np.load(resumed/'fold0'/filename) as b:
            for key in a: np.testing.assert_array_equal(a[key], b[key])
            assert a['scores'].shape == (2, 8) and a['train_scores'].shape == (2, 8)
    (resumed/'fold0/linkgda_t1_nobridge_cal_head_scores.npz').write_bytes(b'corruption')
    failure = invoke(resumed, success=False)
    assert 'Completed artifact checksum mismatch' in failure.stderr
