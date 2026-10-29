"""Export all four frozen checkpoint choices for the complete matched KGE campaign."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))
from phenoquery.common import read, sha, write
from build_matched_kge_manifest import build as build_manifest
from checkpoint_archives import CheckpointInventory, MAX_PART_BYTES, verify_packed
import checkpoint_archives


def build(root, data_root, score_manifest, gene_phenotypes, max_part_bytes=MAX_PART_BYTES):
    root = Path(root).resolve()
    frozen = read(root / 'FROZEN.json')
    if frozen.get('schema') != 1 or set(frozen) != {'schema', 'folds', 'models', 'graphs', 'training'}:
        raise ValueError('Global freeze must use the original trainer schema')
    # Reuse the complete scientific checks, including schedules, all four choices,
    # source/environment consistency, inputs and prescribed U1 warm starts.
    verified = build_manifest(root, data_root, score_manifest, gene_phenotypes)
    if read(score_manifest) != verified:
        raise ValueError('Score manifest differs from the verified complete campaign')
    inventory = CheckpointInventory(root)
    inventory.include('FROZEN.json', verified['freeze_sha256'])
    for run, evidence in sorted(verified['evidence'].items()):
        trained = read(inventory.include(f'{run}/TRAINED.json', evidence['training_sha256']))
        inventory.include(f'{run}/identity.json', evidence['identity_sha256'])
        for name, digest in trained['files'].items():
            inventory.include(f'{run}/{name}', digest)
    if sha(root / 'FROZEN.json') != verified['freeze_sha256']:
        raise ValueError('Global freeze changed during inventory')
    return dict(description='All 100 matched KGE trajectories and 400 validation-selected model checkpoints',
        provenance=dict(builder_sha256=sha(__file__), inventory_builder_sha256=sha(checkpoint_archives.__file__),
            score_manifest_sha256=sha(score_manifest), freeze_sha256=verified['freeze_sha256'],
            protocol_sha256=verified['protocol_sha256'], training_runs=100, model_checkpoints=400,
            source=verified['source'], settings=verified['settings'],
            gene_phenotypes_sha256=verified['gene_phenotypes_sha256'],
            scope='Frozen matched model inference inputs; prepared graphs, gene phenotype annotations, execution environment, test scores and optimizer generations are separate artifacts',
            expected_files=inventory.expected_files('matched-kge')),
        components=inventory.components('matched-checkpoints', 'matched-kge', max_part_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--score-manifest', type=Path, required=True)
    p.add_argument('--gene-phenotypes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify-packed')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        result = build(args.root, args.data_root, args.score_manifest, args.gene_phenotypes)
        if args.output.exists() and read(args.output) != result:
            raise ValueError('Existing plan differs; choose a new output file')
        write(args.output, result)
        print('Verified 100 training trajectories and 400 selected model checkpoints')
    else:
        print(json.dumps(verify_packed(args.plan, args.manifest), indent=2))


if __name__ == '__main__':
    main()
