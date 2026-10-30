"""Replay fold-zero validation metrics from completed U1 and T1 checkpoints."""
import gc
from pathlib import Path
import time

import numpy as np
import torch

from context import WORK, SOURCE, context, read, sha, write
from preflight import verify
from link_gda.head_experiment import profile_metrics
from training.retrain_nobridge_kge import FastGraphScorer


def main():
    output = WORK/'validation-inference-v1'
    if output.exists():
        raise ValueError('Choose a new output directory; inference artifacts exist')
    plan, groups = verify(WORK)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert 'H200' in torch.cuda.get_device_name(0)
    assert str(torch.__version__) == '2.10.0+cu128' and np.__version__ == '2.4.2'
    import os
    from importlib.metadata import version
    assert version('pykeen') == '1.11.1'
    assert os.environ['SLURM_CPUS_PER_TASK'] == '8'
    assert all(os.environ[k] == '8' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    torch.set_num_threads(8)
    runs = ('fold0/original/U1', 'fold0/nobridge/U1', 'fold0/original/T1')
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
        scope='Fold-zero checkpoint-to-validation-metric replay for U1 on both graphs and T1 on the original graph; all four checkpoint choices. Validation compatibility replay from audited native weights using fresh public source. Public-checkpoint test inference remains separate.',
        limitation='Original matched validation score matrices were not retained; this verifies recorded aggregate metrics, not elementwise identity to those original matrices.',
        selection_rule='Fixed fold zero: U1 on both graphs and T1 on the original graph, all four checkpoint choices; selected before this replay.',
        checkpoint_files=12, all_validation_metrics_exact=exact, records=records,
        artifact_proof_sha256=sha(WORK/'CHECKPOINTS_VERIFIED.json'), plan_sha256=sha(WORK/'plan.json'),
        replay_sha256=sha(__file__), context_sha256=sha(WORK/'context.py'), source_proof_sha256=plan['source_proof_sha256'],
        freeze_sha256=plan['freeze_sha256'], preflight_sha256=sha(WORK/'preflight.py'),
        job_id=int(os.environ['SLURM_JOB_ID']), cpu_threads=torch.get_num_threads(),
        allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']), cpu_affinity=sorted(os.sched_getaffinity(0)),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, pykeen=version('pykeen'), cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32),
        seconds=time.time()-started)
    assert sha(Path(plan['destination'])/'FROZEN.json') == plan['freeze_sha256']
    verify(WORK)
    write(output/'verification.json', result)
    print({k:v for k,v in result.items() if k != 'records'}, flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
