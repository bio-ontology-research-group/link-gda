"""Replay fold-zero validation metrics from completed U1 and T1 checkpoints."""
import gc
from pathlib import Path
import time

import numpy as np
import torch

from verify import BASE, RELEASE, WORK, SOURCE, context, read, sha, write
from link_gda.head_experiment import profile_metrics
from training.retrain_nobridge_kge import FastGraphScorer


def main():
    output = WORK/'validation-inference-v1'
    if output.exists():
        raise ValueError('Choose a new output directory; inference artifacts exist')
    proof = read(WORK/'verification.json')
    assert proof['status'] == 'passed' and proof['checkpoint_files'] == 80
    assert proof['verifier_sha256'] == sha(WORK/'verify.py')
    assert proof['plan_sha256'] == sha(WORK/'plan.json')
    assert proof['training_source_files'] == {str(p.relative_to(SOURCE/'code')):sha(p)
        for p in sorted((SOURCE/'code').rglob('*.py'))}
    assert read(BASE/'inference-environments-v1/matched-gpu-54758758.json')['status'] == 'passed'
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    plan = read(WORK/'plan.json')
    groups = {g['relative']:g for g in plan['groups']}
    # Fixed fold and currently available model/graph combinations, not metric-based sampling.
    runs = ('fold0/original/U1', 'fold0/nobridge/U1', 'fold0/original/T1')
    assert all(name in groups for name in runs)
    output.mkdir()
    started, records = time.time(), []
    for name in runs:
        folder = Path(plan['destination'])/name
        for filename, digest in groups[name]['files'].items():
            assert sha(folder/filename) == digest
        identity, factory, genes, profiles, val, model = context(folder, device='cuda')
        complete = read(folder/'TRAINED.json')
        curve = {row['epoch']:row['metrics'] for row in read(folder/'validation.json')['curve']}
        scorer = FastGraphScorer(factory, genes, profiles, eval_chunk_size=200000)
        diseases = sorted(set(val.Disease))
        for choice in ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head'):
            tic = time.time()
            selected = complete['selected'][choice]
            checkpoint = folder/selected['checkpoint']
            assert sha(checkpoint) == complete['files'][checkpoint.name]
            state = torch.load(checkpoint, map_location='cuda', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            scores = scorer.score_diseases(model, diseases)
            measured = profile_metrics(scores, val, genes)
            actual = {mode:measured[mode][0] for mode in ('raw', 'calibrated')}
            expected = curve[selected['epoch']]
            assert set(actual) == set(expected)
            assert all(set(actual[mode]) == set(expected[mode]) for mode in actual)
            differences = {mode:{metric:actual[mode][metric]-expected[mode][metric]
                                for metric in actual[mode]} for mode in actual}
            destination = output/name/(choice+'.npz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, scores=np.stack([scores[d] for d in diseases]),
                                diseases=np.array(diseases), genes=np.array(genes))
            record = dict(run=name, choice=choice, epoch=selected['epoch'],
                checkpoint_sha256=sha(checkpoint), scores_sha256=sha(destination),
                validation_pairs=len(val), validation_diseases=len(diseases), candidates=len(genes),
                metrics_exact=actual == expected, expected=expected, actual=actual,
                metric_differences=differences, seconds=time.time()-tic)
            records.append(record)
            write(output/'progress.json', dict(completed=len(records), expected=12, records=records))
            print(record, flush=True)
            del state, scores
        del scorer, model, factory, genes, profiles, val
        gc.collect()
        torch.cuda.empty_cache()
    assert len(records) == 12
    exact = all(r['metrics_exact'] for r in records)
    result = dict(status='passed' if exact else 'requires_review',
        scope='Fold-zero checkpoint-to-validation-metric replay for U1 on both graphs and T1 on the original graph; all four checkpoint choices. No production test inference or training-trajectory parity claim.',
        limitation='Original matched validation score matrices were not retained; this verifies recorded aggregate metrics, not elementwise identity to those original matrices.',
        selection_rule='Fixed fold zero, all four choices for each currently complete model/graph combination; no sampling by performance.',
        checkpoint_files=12, all_validation_metrics_exact=exact, records=records,
        artifact_proof_sha256=sha(WORK/'verification.json'), plan_sha256=sha(WORK/'plan.json'),
        replay_sha256=sha(__file__), context_sha256=sha(WORK/'verify.py'),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32),
        seconds=time.time()-started)
    write(output/'verification.json', result)
    print({k:v for k,v in result.items() if k != 'records'}, flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
