"""Recover 25 historical comparator variants from saved scores and an ULTRA cache.

These are descriptive retained experiments, not the new matched training study.
The direct-overlap control is recomputed as score aggregation, not graph removal.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from analysis.summarize_query_tables import metadata
from link_gda.score_io import load_scores
from phenoquery.common import pairs, read, sha, write

INPUTS = {'raw/train.txt':'graph', 'training.tsv':'training', 'validation.tsv':'validation',
          'test.tsv':'test', 'profiles.json':'profiles', 'candidates.txt':'candidates'}

LOCAL = [
    ('Exomiser-Phive', 'phive'), ('LinkGDA-U1 (PFS)', 'main-linkgda-pfs-calsel'),
    ('LinkGDA-U1 (PF, No Expr)', 'main-linkgda-pf-calsel'),
    ('LinkGDA-U1 (F, No Mouse Pheno)', 'main-linkgda-f-calsel'),
    ('ConvKB-D (PFS)', 'main-convkb-pfs-calsel'), ('INDIGENA (PFS)', 'main-indigena-pfs-calsel'),
    ('ULTRA (Zero-Shot BMA)', 'main-ultra-pfs'), ('Resnik-BMA', 'resnik_resnik_bma'),
    ('Lin-BMA', 'resnik_lin_bma'), ('SimGIC', 'resnik_simgic')]
OTHER = [
    ('LinkGDA-T1 (Typed Neg)', 'transd-diagnostic/runs/score-fold{fold}/T1_calibrated_head/test/scores.npz'),
    ('LinkGDA-Ppool (Profile Hard-Neg)', 'transd-diagnostic/runs/score-fold{fold}/P-pool_raw_head/test/scores.npz'),
    ('ULTRA (Control Mean-Pool)', 'transd-ultra-mechanism/gpu/fold{fold}/control/scores.npz'),
    ('ULTRA (No Disease Bridges)', 'transd-ultra-mechanism/gpu/fold{fold}/no_disease_bridges/scores.npz'),
    ('ULTRA (No Expression Hubs)', 'transd-ultra-mechanism/gpu/fold{fold}/no_expression/scores.npz'),
    *[(label, 'nobridge_runs/fold{fold}/' + file + '.npz') for label, file in [
        ('LinkGDA-U1 (Retrained No-Bridge, Cal-MR)', 'linkgda_u1_nobridge_cal_mr_scores'),
        ('LinkGDA-U1 (Retrained No-Bridge, Raw-Head)', 'linkgda_u1_nobridge_raw_head_scores'),
        ('LinkGDA-T1 (Retrained No-Bridge, Cal-Head)', 'linkgda_t1_nobridge_cal_head_scores'),
        ('LinkGDA-T1 (Retrained No-Bridge, Raw-Head)', 'linkgda_t1_nobridge_raw_head_scores'),
        ('LinkGDA-Ppool (Retrained No-Bridge)', 'linkgda_ppool_nobridge_scores'),
        ('INDIGENA (Retrained No-Bridge, Raw-MR)', 'indigena_nobridge_raw_mr_scores'),
        ('INDIGENA (Retrained No-Bridge, Raw-Head)', 'indigena_nobridge_raw_head_scores'),
        ('ConvKB-D (Retrained No-Bridge, Cal-MR)', 'convkbd_nobridge_cal_mr_scores'),
        ('ConvKB-D (Retrained No-Bridge, Raw-MR)', 'convkbd_nobridge_raw_mr_scores')]]]


def source_files(base, comparators, fold):
    return [(label, Path(comparators) / folder / f'fold_{fold}.npz') for label, folder in LOCAL] + [
        (label, Path(base) / relative.format(fold=fold)) for label, relative in OTHER]


def masked_scores(cache, genes, diseases, training, profiles):
    if (cache / 'candidates.txt').read_text().splitlines() != genes:
        raise ValueError('Phenotype cache candidate axis differs')
    with np.load(cache / 'test-phenotypes.npz', allow_pickle=False) as z:
        values, phenotypes = z['scores'].astype(np.float64), z['phenotypes'].astype(str).tolist()
    if (len(set(phenotypes)) != len(phenotypes) or values.shape != (len(phenotypes), len(genes))
            or not np.isfinite(values).all()):
        raise ValueError('Invalid phenotype score cache')
    pi, gi = {p: i for i, p in enumerate(phenotypes)}, {g: i for i, g in enumerate(genes)}
    by_gene = defaultdict(set)
    for gene, disease in training: by_gene[gene].update(profiles[disease])
    mask = np.zeros_like(values, dtype=bool)
    for gene, ps in by_gene.items():
        if gene in gi:
            for p in ps:
                if p in pi: mask[pi[p], gi[gene]] = True
    background = np.where(mask, 0., values).sum(0) / np.maximum((~mask).sum(0), 1)
    result = []
    for disease in diseases:
        indices = [pi[p] for p in profiles[disease]]
        omitted = mask[indices]
        counts = (~omitted).sum(0)
        sums = np.where(omitted, 0., values[indices]).sum(0)
        result.append(np.where(counts > 0, sums / np.maximum(counts, 1), background))
    return np.stack(result)


def build(base, comparators, output, expected_candidates=13625):
    base, comparators, output = Path(base).resolve(), Path(comparators).resolve(), Path(output).resolve()
    relative = lambda p: os.path.relpath(p, output.parent)
    models, inputs, caches = {}, {}, {}
    for fold in range(10):
        data = base / 'data' / f'fold{fold}'
        genes, rows, _, _, _ = metadata(data)
        if len(genes) != expected_candidates: raise ValueError('Wrong candidate count')
        diseases = sorted({d for _, d in rows})
        profiles = read(data / 'profiles.json')
        training = pairs(data / 'training.tsv')
        inputs[str(fold)] = {name: sha(data / name) for name in [*INPUTS, 'preparation.json']}
        prepared = read(data / 'preparation.json')
        if prepared['fold'] != fold or any(inputs[str(fold)][name] != prepared[key + '_sha256'] for name, key in INPUTS.items()):
            raise ValueError('Prepared benchmark inputs changed')
        cache = base / 'phrank-inspired/ultra-cache' / f'fold{fold}'
        complete = read(cache / 'completion.json')
        if (complete['status'] != 'complete' or complete['fold'] != fold
                or complete['preparation_sha256'] != inputs[str(fold)]['preparation.json']
                or sha(cache / 'checksums.json') != complete['checksums_sha256']
                or read(cache / 'checksums.json') != complete['files']):
            raise ValueError('Phenotype cache provenance differs')
        for name, digest in complete['files'].items():
            if Path(name).name != name or sha(cache / name) != digest:
                raise ValueError('Phenotype cache file changed')
        caches[str(fold)] = dict(completion_sha256=sha(cache / 'completion.json'), files=complete['files'])
        for label, path in source_files(base, comparators, fold):
            digest = sha(path)
            load_scores(path, genes, diseases, expected_sha256=digest)
            models.setdefault(label, dict(name=label, folds={}))['folds'][str(fold)] = dict(path=relative(path), sha256=digest)
        derived = output.parent / 'derived-overlap' / f'fold{fold}.npz'
        derived.parent.mkdir(parents=True, exist_ok=True)
        scores = masked_scores(cache, genes, diseases, training, profiles)
        if derived.exists():
            if not np.array_equal(load_scores(derived, genes, diseases), scores):
                raise ValueError('Existing overlap scores differ')
        else:
            temporary = derived.with_suffix('.npz.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, scores=scores, genes=genes, diseases=diseases)
            temporary.replace(derived)
        models.setdefault('ULTRA (Direct Overlap Masked)', dict(name='ULTRA (Direct Overlap Masked)', folds={}))['folds'][str(fold)] = dict(path=relative(derived), sha256=sha(derived))
        print(f'Verified 25 historical comparator variants in fold {fold}', flush=True)
    return dict(schema=1, scope='Historical descriptive comparisons; not the new matched KGE experiment',
                folds=list(range(10)), models=list(models.values()), inputs=inputs, caches=caches,
                builder_sha256=sha(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir', type=Path, required=True)
    p.add_argument('--comparator-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = build(a.base_dir, a.comparator_dir or a.base_dir / 'comparators/local-npz', a.output)
    if a.output.exists() and read(a.output) != result:
        raise ValueError('Existing manifest differs; choose a new output path')
    write(a.output, result)


if __name__ == '__main__': main()
