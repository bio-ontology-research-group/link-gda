"""Write one ComplEx pretraining config that train.py's check_config admits, with identity hashes filled in."""
import argparse
import json
from pathlib import Path

from train import SOURCE_FILES, check_config, sha

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--purpose', required=True, choices=['qualification', 'production'])
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--recipe-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--maximum-steps', type=int, help='qualification only; production uses the recipe total')
    parser.add_argument('--checkpoint-every-updates', type=int, default=50000)
    parser.add_argument('--checkpoint-every-seconds', type=int, default=1800)
    parser.add_argument('--maximum-seconds', type=float, default=86400)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    data = Path(args.data_dir).resolve()
    data_sha = sha(data / 'COMPLETE.json')
    graph = protocol['admitted_graphs'][data_sha]
    recipe = protocol['recipes'][args.recipe_id]
    total = recipe['total_updates'] if args.purpose == 'production' else args.maximum_steps
    config = {'experiment_id': args.experiment_id, 'purpose': args.purpose, 'protocol_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
              'source_hashes': {name: sha(HERE / name) for name in SOURCE_FILES}, 'data_dir': str(data), 'data_complete_sha256': data_sha,
              'graph': {key: graph[key] for key in ('taxonomy_mode', 'modality', 'fold', 'cohort')}, 'recipe_id': args.recipe_id, 'seed': args.seed,
              'learning_rate': recipe['learning_rate'], 'regularization': recipe['regularization'], 'maximum_steps': total, 'milestone_updates': [total],
              'training_triples': graph['training_triples'], 'entity_count': graph['entity_count'], 'relation_count': graph['relation_count'],
              'steps_per_epoch': (graph['training_triples'] + 999) // 1000, 'strict_deterministic_algorithms': True, 'rolling_full_checkpoints': 2,
              'checkpoint_every_updates': args.checkpoint_every_updates, 'checkpoint_every_seconds': args.checkpoint_every_seconds, 'maximum_seconds': args.maximum_seconds,
              **protocol['fixed']}
    check_config(config, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'config': str(output), 'experiment_id': args.experiment_id}))


if __name__ == '__main__':
    main()
