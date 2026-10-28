import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('baseline_manifest',
    Path(__file__).resolve().parents[1] / 'reproducibility/build_baseline_manifest.py')
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    data_root = root / 'data'
    roots = {c: root / c for c in manifest.COMPONENTS}
    for fold in range(10):
        data = data_root / f'fold{fold}'
        (data / 'raw').mkdir(parents=True)
        for file in manifest.INPUTS:
            (data / file).write_text(f'fixture {fold} {file}\n')
        hashes = {name: manifest.sha(data / name) for name in manifest.INPUTS}
        identity = {'fold': fold, **{key + '_sha256': hashes[name] for name, key in manifest.INPUTS.items()}}
        write(data / 'preparation.json', identity)
        for component, base in roots.items():
            folder = base / f'fold{fold}'
            folder.mkdir(parents=True)
            if component.startswith('upheno'):
                pairs = manifest.PHRANK if component == 'upheno-phrank' else manifest.PHENORANK
                outputs = {}
                for file, _ in pairs:
                    np.savez_compressed(folder / file, scores=np.ones((2, 3)), genes=['a', 'b', 'c'], diseases=['d0', 'd1'])
                    outputs[file] = manifest.sha(folder / file)
                complete = {'fold': fold, 'inputs': hashes, 'outputs': outputs,
                            'phrank_root_policy': 'native', 'sampler': 'published-tier',
                            'simulations': 1000, 'query_length_cap': None,
                            'rwr_iterations': 20, 'restart_probability': .1}
                write(folder / 'COMPLETE.json', complete)
                continue
            np.savez_compressed(folder / 'scores.npz', scores=np.ones((2, 3)), genes=['a', 'b', 'c'], diseases=['d0', 'd1'])
            shutil.copyfile(data / 'test.tsv', folder / 'pairs.tsv')
            shutil.copyfile(data / 'candidates.txt', folder / 'candidates.txt')
            complete = {'status': 'complete', 'fold': fold,
                        'preparation_sha256': manifest.sha(data / 'preparation.json')}
            if component == 'native-phenorank':
                write(folder / 'config.json', {'fold': fold, 'seed': 0, 'nperm': 1000,
                    'restart': .1, 'iterations': 20, 'technical_pilot': False})
                write(folder / 'resource-audit.json', {'fixture': True})
                complete['resource_audit_sha256'] = manifest.sha(folder / 'resource-audit.json')
                raw = folder / 'raw-propagation'
                raw.mkdir()
                shutil.copyfile(folder / 'scores.npz', raw / 'scores.npz')
                write(folder / 'raw-propagation-checksums.json', {'scores.npz': manifest.sha(raw / 'scores.npz')})
            else:
                mode = 'direct' if component == 'native-phrank' else 'disease-max'
                write(folder / 'provenance.json', {'scoring_mode': mode,
                    'held_out_annotation_sources': 0, 'prepared_identity': identity})
                complete['provenance_sha256'] = manifest.sha(folder / 'provenance.json')
            write(folder / 'checksums.json', {p.name: manifest.sha(p) for p in folder.iterdir() if p.is_file()})
            complete['checksums_sha256'] = manifest.sha(folder / 'checksums.json')
            write(folder / 'completion.json', complete)
    return data_root, roots


def test_complete_manifest_relocates_and_retains_every_variant(tmp_path):
    data, roots = fixture(tmp_path / 'original')
    output = tmp_path / 'original/scores.json'
    result = manifest.build(data, roots, output)
    assert result['all_baseline_components'] and len(result['models']) == 26
    assert len(result['completions']) == 50
    assert all(set(m['folds']) == {str(i) for i in range(10)} for m in result['models'])
    write(output, result)
    relocated = tmp_path / 'relocated'
    shutil.copytree(output.parent, relocated)
    for model in manifest.read(relocated / 'scores.json')['models']:
        for entry in model['folds'].values():
            assert not Path(entry['path']).is_absolute()
            manifest.check(relocated / entry['path'], entry['sha256'])
    rebuilt = manifest.build(relocated / 'data', {c: relocated / c for c in roots}, relocated / 'scores.json')
    assert rebuilt == result


@pytest.mark.parametrize('change', ['old-root', 'old-sampler', 'capped', 'input', 'missing-fold', 'corrupt-score', 'wrong-native-mode', 'native-pilot', 'corrupt-raw'])
def test_manifest_rejects_incomplete_or_superseded_outputs(tmp_path, change):
    data, roots = fixture(tmp_path)
    if change in ['old-root', 'old-sampler', 'capped', 'input']:
        component = 'upheno-phrank' if change == 'old-root' else 'upheno-phenorank'
        p = roots[component] / 'fold0/COMPLETE.json'
        value = manifest.read(p)
        if change == 'old-root': value['phrank_root_policy'] = 'zero'
        if change == 'old-sampler': value['sampler'] = 'legacy-weighted'
        if change == 'capped': value['query_length_cap'] = 20
        if change == 'input': value['inputs']['training.tsv'] = 'changed'
        write(p, value)
    elif change == 'missing-fold':
        (roots['upheno-phrank'] / 'fold9/COMPLETE.json').unlink()
    elif change == 'corrupt-score':
        (roots['upheno-phrank'] / 'fold0' / manifest.PHRANK[0][0]).write_bytes(b'changed')
    elif change == 'corrupt-raw':
        (roots['native-phenorank'] / 'fold0/raw-propagation/scores.npz').write_bytes(b'changed')
    else:
        component = 'native-phrank-disease-max' if change == 'wrong-native-mode' else 'native-phenorank'
        folder = roots[component] / 'fold0'
        file = 'provenance.json' if change == 'wrong-native-mode' else 'config.json'
        value = manifest.read(folder / file)
        value['scoring_mode' if change == 'wrong-native-mode' else 'technical_pilot'] = 'direct' if change == 'wrong-native-mode' else True
        write(folder / file, value)
        checksums = manifest.read(folder / 'checksums.json')
        checksums[file] = manifest.sha(folder / file)
        write(folder / 'checksums.json', checksums)
        complete = manifest.read(folder / 'completion.json')
        complete['checksums_sha256'] = manifest.sha(folder / 'checksums.json')
        if change == 'wrong-native-mode': complete['provenance_sha256'] = manifest.sha(folder / file)
        write(folder / 'completion.json', complete)
    with pytest.raises((ValueError, FileNotFoundError)):
        manifest.build(data, roots, tmp_path / 'scores.json')


def test_requested_components_are_explicit_and_missing_ones_are_not_implied(tmp_path):
    data, roots = fixture(tmp_path)
    result = manifest.build(data, {'upheno-phrank': roots['upheno-phrank']}, tmp_path / 'scores.json')
    assert len(result['models']) == 10
    assert result['components'] == ['upheno-phrank']
    assert not result['all_baseline_components']
