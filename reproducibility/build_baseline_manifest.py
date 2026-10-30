"""Assemble explicitly requested, completed baseline components for bias evaluation.

All ten folds are required for every requested component. This validates provenance
and file hashes; evaluate_stepwise_bias.py separately validates score axes, finite
values and evaluation cohorts. Paths are relative to the output manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


COMPONENTS = ['native-phrank', 'native-phrank-disease-max', 'native-phenorank',
              'upheno-phrank', 'upheno-phenorank']
INPUTS = {'training.tsv': 'training', 'test.tsv': 'test', 'profiles.json': 'profiles',
          'candidates.txt': 'candidates', 'raw/train.txt': 'graph'}
PHRANK = [
    ('phrank_human_upheno_disease_raw_scores.npz', 'Phrank-HPO (disease_raw)'),
    ('phrank_human_upheno_disease_norm_scores.npz', 'Phrank-HPO (disease_normalized)'),
    ('phrank_human_upheno_direct_raw_scores.npz', 'Phrank-HPO (direct_raw)'),
    ('phrank_human_upheno_direct_norm_scores.npz', 'Phrank-HPO (direct_normalized)'),
    ('phrank_upheno_hm_disease_raw_scores.npz', 'Phrank-UPheno (Human+MGI Disease Raw)'),
    ('phrank_upheno_hm_disease_norm_scores.npz', 'Phrank-UPheno (Human+MGI Disease Norm)'),
    ('phrank_upheno_hm_direct_raw_scores.npz', 'Phrank-UPheno (Human+MGI Direct Raw)'),
    ('phrank_upheno_hm_direct_norm_scores.npz', 'Phrank-UPheno (Human+MGI Direct Norm)'),
    ('phrank_upheno_mgi_raw_scores.npz', 'Phrank-UPheno (MGI-Only Raw)'),
    ('phrank_upheno_mgi_norm_scores.npz', 'Phrank-UPheno (MGI-Only Norm)'),
]
PHENORANK = [(f'phenorank_upheno_{source}_{suffix}_scores.npz',
              f'PhenoRank-UPheno ({label} {kind})')
             for source, label in [('hm_add', 'HM-Add+PPI'), ('hm_bal', 'HM-Bal+PPI'),
                                   ('mgi_ppi', 'Mouse+PPI'), ('mgi_noppi', 'Mouse-NoPPI')]
             for suffix, kind in [('nosim', 'NoSim'), ('sim', '1000-Sim'),
                                  ('sim_tb', '1000-Sim TB')]]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def native_records(folder, component, fold, preparation):
    complete = read(folder / 'completion.json')
    if complete.get('status') != 'complete' or complete['fold'] != fold:
        raise ValueError(f'Incomplete or wrong fold: {folder}')
    check(preparation, complete['preparation_sha256'])
    check(folder / 'checksums.json', complete['checksums_sha256'])
    checksums = read(folder / 'checksums.json')
    for name, digest in checksums.items():
        if Path(name).name != name:
            raise ValueError('Non-local checksum entry')
        check(folder / name, digest)
    data = preparation.parent
    if checksums['pairs.tsv'] != sha(data / 'test.tsv') or checksums['candidates.txt'] != sha(data / 'candidates.txt'):
        raise ValueError('Baseline associations or candidates differ from the benchmark')
    if component == 'native-phenorank':
        config = read(folder / 'config.json')
        required = {'fold': fold, 'seed': 0, 'nperm': 1000, 'restart': .1,
                    'iterations': 20, 'technical_pilot': False}
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError('Native PhenoRank configuration is not the full published protocol')
        check(folder / 'resource-audit.json', complete['resource_audit_sha256'])
        raw = read(folder / 'raw-propagation-checksums.json')
        for name, digest in raw.items():
            if Path(name).name != name:
                raise ValueError('Non-local raw-propagation checksum entry')
            check(folder / 'raw-propagation' / name, digest)
        return [('scores.npz', 'PhenoRank (Native 2016 1000-Sim)', checksums['scores.npz']),
                ('raw-propagation/scores.npz', 'PhenoRank (Native 2016 NoSim)', raw['scores.npz'])]
    check(folder / 'provenance.json', complete['provenance_sha256'])
    evidence = read(folder / 'provenance.json')
    mode = evidence.get('scoring_mode')
    if mode is None and evidence['method'] == 'Phrank 1.6 native direct gene scoring':
        mode = 'direct'  # Original verified direct-only runner predates this field.
    expected = 'direct' if component == 'native-phrank' else 'disease-max'
    if mode != expected or evidence['held_out_annotation_sources'] != 0:
        raise ValueError('Wrong Phrank scoring rule or held-out annotation sources')
    if evidence['prepared_identity'] != read(preparation):
        raise ValueError('Native Phrank preparation differs from benchmark')
    label = 'Direct' if expected == 'direct' else 'Disease-Max'
    return [('scores.npz', f'Phrank (Native HPO {label} Raw)', checksums['scores.npz'])]


def build(data_root, roots, output):
    if not roots or set(roots) - set(COMPONENTS):
        raise ValueError('Supply at least one recognized baseline component')
    output = Path(output).resolve()
    models, evidence, identities = {}, {}, {}
    relative = lambda path: os.path.relpath(Path(path).resolve(), output.parent)
    for fold in range(10):
        data = Path(data_root) / f'fold{fold}'
        preparation = data / 'preparation.json'
        identity = read(preparation)
        if identity['fold'] != fold:
            raise ValueError('Wrong benchmark fold identity')
        hashes = {name: sha(data / name) for name in INPUTS}
        for name, key in INPUTS.items():
            if hashes[name] != identity[key + '_sha256']:
                raise ValueError(f'Prepared benchmark file changed: fold {fold}, {name}')
        identities[str(fold)] = {'path': relative(preparation), 'sha256': sha(preparation),
                                 'inputs': hashes}
        for component in COMPONENTS:
            if component not in roots:
                continue
            folder = Path(roots[component]) / f'fold{fold}'
            if component.startswith('native-'):
                records = native_records(folder, component, fold, preparation)
                completion = folder / 'completion.json'
            else:
                completion = folder / 'COMPLETE.json'
                complete = read(completion)
                if complete['fold'] != fold or complete['inputs'] != hashes:
                    raise ValueError('UPheno fold or scientific inputs differ from benchmark')
                if component == 'upheno-phrank':
                    if complete.get('phrank_root_policy') != 'native':
                        raise ValueError('UPheno Phrank requires the author-compatible root policy')
                    pairs = PHRANK
                else:
                    required = {'sampler': 'published-tier', 'simulations': 1000,
                                'query_length_cap': None, 'rwr_iterations': 20,
                                'restart_probability': .1}
                    if any(k not in complete or complete[k] != v for k, v in required.items()):
                        raise ValueError('UPheno PhenoRank requires uncapped published-tier simulations')
                    pairs = PHENORANK
                records = [(file, label, complete['outputs'][file]) for file, label in pairs]
            evidence[f'{component}/fold{fold}'] = {'path': relative(completion), 'sha256': sha(completion)}
            for file, label, digest in records:
                path = folder / file
                check(path, digest)
                record = models.setdefault(label, {'name': label, 'component': component, 'folds': {}})
                record['folds'][str(fold)] = {'path': relative(path), 'sha256': digest}
    return {'schema': 1, 'components': [c for c in COMPONENTS if c in roots],
            'all_baseline_components': set(roots) == set(COMPONENTS),
            'folds': list(range(10)), 'benchmark': identities, 'completions': evidence,
            'legacy_label_note': 'Phrank-HPO keys denote human-only UPheno adapters; native HPO arms have separate Native HPO labels.',
            'builder_sha256': sha(__file__), 'models': list(models.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Prepared benchmark data/ directory')
    parser.add_argument('--output', type=Path, required=True)
    for component in COMPONENTS:
        parser.add_argument('--' + component, type=Path, help='Completed component directory containing fold0..fold9')
    args = parser.parse_args()
    roots = {name: getattr(args, name.replace('-', '_')) for name in COMPONENTS
             if getattr(args, name.replace('-', '_')) is not None}
    result = build(args.data_root, roots, args.output)
    if args.output.exists() and read(args.output) != result:
        raise ValueError('Existing manifest differs; use a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.pending')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(f'Verified {len(result["models"])} baseline variants across ten folds; all components: {result["all_baseline_components"]}')


if __name__ == '__main__':
    main()
