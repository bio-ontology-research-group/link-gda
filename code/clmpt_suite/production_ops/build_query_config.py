"""Build and validate one production query config with the campaign's copy of the worker's make_config (query interpreter).

Reads a JSON request {output, protocol, experiment_id, data_dir, pretrained, pretrained_sha256, initializer, seed, sampler_seed,
encoder, curriculum, learning_rate, temperature, maximum_steps, evaluation_every_updates, maximum_seconds} and writes the
config once, then a receipt with the worker's validate_config checks (the pretrained export is rehashed).
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-code', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--skip-pretrained-hash', action='store_true', help='template configs for the reuse audit only; the export is not on this host')
    args = parser.parse_args()
    worker = Path(args.worker_code).resolve()
    sys.path.insert(0, str(worker))
    import make_config
    assert Path(make_config.__file__).resolve().parent == worker
    item = json.loads(Path(args.request).read_text())
    receipt = Path(args.receipt)
    assert not receipt.exists()
    config = make_config.build_config(item['protocol'], 'production', item['experiment_id'], item['data_dir'], item['pretrained'], item['initializer'], item['seed'],
                                      item['encoder'], item['curriculum'], item['learning_rate'], item['temperature'], sampler_seed=item['sampler_seed'],
                                      maximum_steps=item['maximum_steps'], evaluation_every_updates=item['evaluation_every_updates'],
                                      maximum_seconds=item['maximum_seconds'], pretrained_sha256=item['pretrained_sha256'])
    make_config.write_config(config, item['output'])
    validation = make_config.validate_config(item['output'], verify_pretrained=not args.skip_pretrained_hash)
    assert validation['passed'] is True
    receipt.write_text(json.dumps({'output': item['output'], 'checks': validation['checks'], 'clinical_training_gene_domain': validation.get('clinical_training_gene_domain'),
                                   'passed': True}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'config': item['output'], 'passed': True}))


if __name__ == '__main__':
    main()
