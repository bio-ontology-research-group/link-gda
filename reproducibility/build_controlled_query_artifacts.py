"""Export frozen controlled-query checkpoints and their selection evidence."""
import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.summarize_controlled_query import verify_plan, verify_validation_choice
from phenoquery.campaign import verify_gates, verify_source
from phenoquery.common import read, sha, write
from checkpoint_archives import CheckpointInventory, MAX_PART_BYTES, verify_packed
import checkpoint_archives


def build(root, training_source, max_part_bytes=MAX_PART_BYTES):
    root = Path(root).resolve()
    training_source = Path(training_source).resolve()
    training_code = training_source/'code'
    plan, global_sha = verify_plan(root)
    if sha(training_source/'reproducibility/protocols/controlled-query-20261028.md') != plan['protocol_sha256']:
        raise ValueError('Frozen training checkout has a different protocol')
    inventory = CheckpointInventory(root)
    include = inventory.include

    for name in ['CONTROLLED.json', 'GLOBAL-SELECTION.json']:
        include(name)
    for initialization in plan['initializations']:
        gate = read(include(f'GATE-{initialization}.json'))
        for name, digest in gate['artifacts'].items():
            include(name, digest)
    verify_gates(root)
    upstream = {}
    for campaign in plan['campaigns']:
        child = root / campaign
        verify_source(child, training_code)
        initialization, seed_text = campaign.rsplit('-seed', 1)
        seed = int(seed_text)
        for name in ['source.sha256', 'PROTOCOL.md', 'SELECTION.json', 'campaign.json']:
            include(f'{campaign}/{name}')
        configuration = read(child / 'campaign.json')
        expected_campaign = dict(seed=seed, initialization=initialization, kind='eq',
            steps=500, cohort='both', arms=plan['arms'],
            protocol_sha256=plan['protocol_sha256'], source_sha256=plan['source_sha256'])
        if configuration != expected_campaign:
            raise ValueError('Campaign metadata differs from the controlled design')
        selection = read(child / 'SELECTION.json')
        for run, frozen in selection['runs'].items():
            prefix = f'{campaign}/training/{run}'
            trained = read(include(f'{prefix}/COMPLETE.json', frozen['completion_sha256']))
            identity = read(include(f'{prefix}/identity.json', trained['identity_sha256']))
            arm, fold_text = run.rsplit('-fold', 1)
            expected_config = dict(phase='train', arm=arm, fold=int(fold_text), pilot=False,
                steps=500, seed=seed, initialization=initialization,
                protocol_sha256=plan['protocol_sha256'], global_selection_required=True)
            config = read(include(f'{campaign}/configs/{run}.json'))
            if config != expected_config or identity['config'] != config:
                raise ValueError('Training configuration differs from the controlled design')
            if identity['seed'] != seed or identity['initialization'] != initialization:
                raise ValueError('Training seed/initialization changed')
            for record in [identity, trained]:
                if (record['protocol_sha256'] != plan['protocol_sha256']
                        or record['source_manifest_sha256'] != plan['source_sha256']):
                    raise ValueError('Training source/protocol changed')
            if (trained['arm'] != arm or trained['fold'] != int(fold_text)
                    or trained['pilot'] or trained['steps'] != 500
                    or trained['best'] != frozen['best']
                    or trained['checkpoint_sha256'] != frozen['checkpoint_sha256']):
                raise ValueError('Training completion differs from the global selection')
            initial_sha = identity['initial_checkpoint_sha256']
            if initialization in upstream and upstream[initialization] != initial_sha:
                raise ValueError('Upstream initialization differs across runs')
            upstream[initialization] = initial_sha
            progress = read(include(f'{prefix}/progress.json'))
            verify_validation_choice(progress, trained)
            log = include(f'{prefix}/training.jsonl')
            if [json.loads(line)['step'] for line in log.read_text().splitlines()] != list(range(1, 501)):
                raise ValueError('Training log does not contain all 500 updates')
            for filename, key in [('best.pt', 'checkpoint_sha256'),
                    ('initial.pt', 'initial_checkpoint_sha256'),
                    ('validation.npz', 'validation_sha256'),
                    ('initial-validation.npz', 'initial_validation_sha256')]:
                include(f'{prefix}/{filename}', trained[key])
    if sha(root / 'GLOBAL-SELECTION.json') != global_sha:
        raise ValueError('Global selection changed during inventory')

    return dict(description='All 240 controlled-query selections, initial/selected checkpoints and validation evidence',
        provenance=dict(builder_sha256=sha(__file__), global_selection_sha256=global_sha,
            inventory_builder_sha256=sha(checkpoint_archives.__file__),
            protocol_sha256=plan['protocol_sha256'], source_manifest_sha256=plan['source_sha256'],
            upstream_initialization_sha256=upstream, training_runs=240, model_checkpoints=480,
            scope='Checkpoint-to-prediction inputs; prepared graphs, upstream source/environment, test scores and resumable optimizer generations are separate artifacts',
            expected_files=inventory.expected_files('controlled-query')),
        components=inventory.components('controlled-checkpoints', 'controlled-query', max_part_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-source', type=Path, required=True,
                   help='Frozen training checkout whose code matches every archived source.sha256')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify-packed')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        result = build(args.root, args.training_source)
        if args.output.exists() and read(args.output) != result:
            raise ValueError('Existing plan differs; choose a new output file')
        write(args.output, result)
        print('Verified 240 training selections and 480 initial/selected model checkpoints')
    else:
        print(json.dumps(verify_packed(args.plan, args.manifest), indent=2))


if __name__ == '__main__':
    main()
