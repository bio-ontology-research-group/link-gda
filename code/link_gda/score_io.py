"""Validated, lossless interchange of disease-by-candidate score matrices."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_scores(path, genes, diseases, *, expected_sha256=None):
    """Align explicit identities; reject missing/duplicate candidates and queries."""
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f'Score checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as bundle:
        x = bundle['scores'].astype(np.float64)
        source_genes = bundle['genes'].astype(str).tolist()
        source_diseases = bundle['diseases'].astype(str).tolist()
    if len(set(source_genes)) != len(source_genes) or len(set(source_diseases)) != len(source_diseases):
        raise ValueError(f'Duplicate identities in {path}')
    if x.shape != (len(source_diseases), len(source_genes)) or not np.isfinite(x).all():
        raise ValueError(f'Invalid score matrix in {path}')
    if set(source_genes) != set(genes):
        raise ValueError(f'Candidate universe differs in {path}')
    if not set(diseases) <= set(source_diseases):
        raise ValueError(f'Missing test diseases in {path}')
    gi = {g: i for i, g in enumerate(source_genes)}
    di = {d: i for i, d in enumerate(source_diseases)}
    return x[np.ix_([di[d] for d in diseases], [gi[g] for g in genes])]


def convert_prediction_tsv(source, output, candidates_path=None):
    """Convert legacy gene/disease/target-index/score rows without reducing precision.

    Repeated diseases must have exactly identical candidate scores. Every target
    index is checked against the candidate sidecar; ranks are never used as scores.
    """
    source, output = Path(source), Path(output)
    candidates_path = Path(candidates_path or str(source) + '.candidates.txt')
    genes = candidates_path.read_text().splitlines()
    if not genes or len(set(genes)) != len(genes):
        raise ValueError('Candidate sidecar must contain distinct genes')
    by_disease, pairs = {}, set()
    with source.open() as stream:
        for lineno, line in enumerate(stream, 1):
            gene, disease, index, rest = line.rstrip('\n').split('\t', 3)
            index = int(index)
            if not 0 <= index < len(genes) or genes[index] != gene:
                raise ValueError(f'Target identity mismatch at {source}:{lineno}')
            if (gene, disease) in pairs:
                raise ValueError(f'Duplicate association at {source}:{lineno}')
            pairs.add((gene, disease))
            scores = np.asarray(rest.split('\t'), dtype=np.float64)
            if scores.shape != (len(genes),) or not np.isfinite(scores).all():
                raise ValueError(f'Invalid score row at {source}:{lineno}')
            if disease in by_disease and not np.array_equal(scores, by_disease[disease]):
                raise ValueError(f'Inconsistent repeated query: {disease}')
            by_disease[disease] = scores
    if not pairs:
        raise ValueError('Empty prediction file')
    diseases = sorted(by_disease)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, scores=np.stack([by_disease[d] for d in diseases]),
                            diseases=np.asarray(diseases), genes=np.asarray(genes))
    temporary.replace(output)
    metadata = {'source_sha256': sha256(source), 'candidates_sha256': sha256(candidates_path),
                'scores_sha256': sha256(output), 'pairs': len(pairs),
                'diseases': len(diseases), 'candidates': len(genes)}
    output.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
