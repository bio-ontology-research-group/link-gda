"""Released TEST scoring of retained checkpoints from one completed suite training run.

Refuses unless the config has phase 'test' and pins a TEST-RELEASE.json whose status is RELEASED and which
names this training run's COMPLETE and this TEST directory's COMPLETE. Views: 'initial', 'raw' and
'calibrated' (model-only exports retained by the trainer) and 'terminal' (terminal full state, only if not
pruned). Each view stores the raw score matrix, representations, score semantics and candidate IDs, and
raw plus own-LOO calibrated metrics with per-association ranks, computed from that one matrix. Nothing is
selected here; selections are frozen before release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from evaluate import score_cases, verify_split
from metrics import calibrate, measure
from model import SCORE_SEMANTICS
from train import REFERENCE_FILES, SOURCE_FILES, build_model, canonical, dump, numerics_record, pin_numerics, sha

TEST_SOURCES = SOURCE_FILES + ('test_eval.py',)
VIEWS = ('initial', 'raw', 'calibrated', 'terminal')


def load_view(model, directory, complete, view):
    terminal = complete['terminal_checkpoint']
    if view == 'terminal':
        path = directory / terminal['checkpoint']
        assert path.exists(), 'terminal full state was pruned'
        assert sha(path) == terminal['checkpoint_sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        model.load_state_dict(state, strict=True)
        return {'path': terminal['checkpoint'], 'sha256': terminal['checkpoint_sha256'], 'step': terminal['step']}
    pointer = terminal['initial_checkpoint'] if view == 'initial' else terminal['retained_best'][view]
    path = Path(pointer['root']) / pointer['path']
    if not path.exists():
        path = directory / pointer['path']
    assert sha(path) == pointer['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = dict(saved['model'])
    for key, target in saved['aliases'].items():
        state[key] = state[target]
    model.load_state_dict(state, strict=True)
    return {'path': pointer['path'], 'sha256': pointer['sha256'], 'step': pointer['step']}


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    assert config['phase'] == 'test'
    assert set(config['views']) <= set(VIEWS) and config['views']
    release = json.loads(Path(config['release_path']).read_text())
    assert sha(config['release_path']) == config['release_sha256'] and release['status'] == 'RELEASED'
    directory = Path(config['training_dir'])
    assert sha(directory / 'COMPLETE.json') == config['training_complete_sha256']
    assert config['training_complete_sha256'] in release['training_complete_sha256']
    test_dir = Path(config['test_dir'])
    assert sha(test_dir / 'COMPLETE.json') == config['test_complete_sha256'] and config['test_complete_sha256'] in release['test_complete_sha256']
    test_manifest = json.loads((test_dir / 'COMPLETE.json').read_text())
    for name, digest in test_manifest['files'].items():
        assert sha(test_dir / name) == (digest['sha256'] if isinstance(digest, dict) else digest)
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    identity = json.loads((directory / 'identity.json').read_text())
    assert canonical(identity) == complete['identity_sha256']
    training = identity['config']
    here = Path(__file__).resolve().parent
    sources = {name: sha(here / name) for name in TEST_SOURCES}
    assert sources == config['source_hashes'] and {name: sources[name] for name in SOURCE_FILES} == training['source_hashes']
    assert {name: sha(here.parent / 'references' / name) for name in REFERENCE_FILES} == training['reference_hashes']
    assert test_manifest['bank_complete_sha256'] == training['data_complete_sha256']
    bank = json.loads((Path(config['bank_dir']) / 'bank.json').read_text())
    assert sha(Path(config['bank_dir']) / 'COMPLETE.json') == training['data_complete_sha256']
    cases = json.loads((test_dir / 'test-cases.json').read_text())
    assert cases['bank_complete_sha256'] == training['data_complete_sha256'] and cases['candidate_ids'] == bank['candidate_ids']
    rows = cases['cases']
    verify_split(rows, {'test_cases': test_manifest['test_cases'], 'test_associations': test_manifest['test_associations']}, 'test')
    record = {'config': config, 'config_sha256': sha(config_path), 'sources': sources, 'training_identity_sha256': complete['identity_sha256'],
              'release_sha256': config['release_sha256'], 'test_complete_sha256': config['test_complete_sha256']}
    if output.exists():
        assert canonical(json.loads((output / 'identity.json').read_text())) == canonical(record)
        if (output / 'COMPLETE.json').exists():
            return 0
    else:
        output.mkdir(parents=True)
        dump(output / 'identity.json', record)
    pin_numerics()
    training = dict(training, pretrained_checkpoint=config.get('pretrained_checkpoint', training['pretrained_checkpoint']))
    model = build_model(training, bank)
    model.eval()
    seen = {g for row in bank['training_profiles'] for g in row['answers']}
    answers = [row['gene_ids'] for row in rows]
    files, summary = {}, {}
    for view in config['views']:
        loaded = load_view(model, directory, complete, view)
        representation, real = score_cases(model, rows, bank)
        assert np.isfinite(real).all()
        archive = output / ('test-%s.npz' % view)
        temporary = archive.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, test_real=real, test_representation=representation, candidate_ids=np.array(bank['candidate_ids']),
                                score_semantics=np.array(SCORE_SEMANTICS[model.encoder]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, archive)
        metrics = {'raw': measure(real, answers, bank['candidate_ids'], seen_genes=seen),
                   'calibrated': measure(calibrate(real), answers, bank['candidate_ids'], seen_genes=seen)}
        result = {'view': view, 'checkpoint': loaded, 'metrics': metrics, 'scores_sha256': sha(archive), 'encoder': model.encoder,
                  'score_semantics': SCORE_SEMANTICS[model.encoder], 'case_source_rows': [row.get('source_rows') for row in rows]}
        dump(output / ('test-%s.json' % view), result)
        files['test-%s.npz' % view] = result['scores_sha256']
        files['test-%s.json' % view] = sha(output / ('test-%s.json' % view))
        summary[view] = {score: metrics[score]['answer_micro'] for score in ('raw', 'calibrated')}
    files['identity.json'] = sha(output / 'identity.json')
    dump(output / 'COMPLETE.json', {'files': files, 'summary': summary, 'runtime': {'gpu': torch.cuda.get_device_name(0), 'flags': numerics_record()}, 'time': time.time(),
                                    'scope': 'TEST released after frozen selection; raw and own-LOO calibrated from the same matrix; unfiltered average ranks.'})
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output))
