"""Build query-worker configs through the worker's own make_config.build_config and validate each with validate_config.

Runs with the query interpreter (torch is imported by the worker module). The worker directory given here is the one
the configs pin: its make_config hashes exactly train.SOURCE_FILES from that directory and train.REFERENCE_FILES from
its ../references. Input is a plan JSON {"configs": [{"output", "protocol", "purpose", "experiment_id", "data_dir",
"pretrained", "pretrained_sha256", "initializer", "seed", "encoder", "curriculum", "learning_rate", "temperature",
"maximum_steps", "evaluation_every_updates", "maximum_seconds", "continuation_parent"}]}. Writes each config once and
a receipt listing every config with its validation result. Pretrained exports are not rehashed here; the stager
hashes them once and the trainer verifies them at every start.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker-code', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    worker = Path(args.worker_code).resolve()
    sys.path.insert(0, str(worker))
    import make_config
    import train
    receipt_path = Path(args.receipt)
    assert not receipt_path.exists(), 'receipt exists: %s' % receipt_path
    assert Path(make_config.__file__).resolve().parent == worker and Path(train.__file__).resolve().parent == worker
    plan = json.loads(Path(args.plan).read_text())
    rows = []
    for item in plan['configs']:
        config = make_config.build_config(item['protocol'], item['purpose'], item['experiment_id'], item['data_dir'], item['pretrained'], item['initializer'],
                                          item['seed'], item['encoder'], item['curriculum'], item['learning_rate'], item['temperature'],
                                          maximum_steps=item['maximum_steps'], evaluation_every_updates=item['evaluation_every_updates'],
                                          maximum_seconds=item['maximum_seconds'], continuation_parent=item.get('continuation_parent'),
                                          pretrained_sha256=item['pretrained_sha256'])
        output = Path(item['output'])
        make_config.write_config(config, output)
        validation = make_config.validate_config(output, verify_pretrained=False)
        assert validation['passed'] is True
        rows.append({'experiment_id': item['experiment_id'], 'output': str(output), 'sha256': sha(output), 'checks': validation['checks'],
                     'clinical_training_gene_domain': validation.get('clinical_training_gene_domain')})
    value = {'worker_code': str(worker), 'source_files': list(train.SOURCE_FILES), 'reference_files': list(train.REFERENCE_FILES),
             'make_config_sha256': sha(worker / 'make_config.py'), 'configs': rows, 'passed': len(rows) == len(plan['configs'])}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'built_and_validated': len(rows), 'receipt': str(receipt_path)}))


if __name__ == '__main__':
    main()
