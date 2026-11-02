"""Composed two-hop validation scoring, aggregation, calibration, ranking and selection.

Forward readout: one query per candidate gene, gene then ``associated_with``
then ``has_symptom``, scored against every profile phenotype. Inverse readout:
one query per profile phenotype, phenotype then ``inverse:has_symptom`` then
``inverse:associated_with``, scored against every candidate gene. Neither
readout looks up a disease entity. Both produce a phenotype-by-gene matrix that
is aggregated over each disease's profile multiset with the fixed mix
``weight * max + (1 - weight) * mean``.

Calibration, average-rank ties, metrics, the selection key and the step-0
eligibility gate reproduce ``code/phenoquery/common.py`` (``cal``, ``ranks``,
``metrics``, ``key``, ``eligible``). Aggregation does not: PhenoQuery's
``aggregate`` pools over ``sorted(set(profile))``, whereas this module keeps
every repeated profile entry, so the mean term (never the max) differs on any
profile with a repeated phenotype. Aggregated scores, ranks and metrics are
therefore not comparable across the two studies on such profiles.

Each validation archive records the scoring settings and the device that
produced it (``scoring_*`` arrays). BetaE scores depend on ``query_chunk`` and
on the device model through batch-shape-dependent matrix kernels, so
re-scoring a saved checkpoint with other chunking or on another GPU model is
not bit-identical; recompute metrics from the saved phenotype-level scores.
"""
import numpy as np
import torch

from .data import READOUTS, profile_phenotypes


class ValidationInterrupted(Exception):
    """Raised between scoring chunks when a pause was requested."""


def path_scores(model, anchors, relations, targets, device, query_chunk, chunk_elements, dim, interrupt=None):
    """Score every anchor's path query against every target entity.

    Returns float32 ``[len(anchors), len(targets)]``. Memory per chunk is bounded
    by ``chunk_elements`` pairwise vector elements.
    """
    anchors = torch.as_tensor(np.asarray(anchors, dtype=np.int64))
    targets = torch.as_tensor(np.asarray(targets, dtype=np.int64))
    result = np.empty((len(anchors), len(targets)), dtype=np.float32)
    path = torch.as_tensor(np.asarray(relations, dtype=np.int64))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for first in range(0, len(anchors), query_chunk):
                chunk = anchors[first:first + query_chunk].to(device)
                query = model.encode(chunk, path.to(device).unsqueeze(0).expand(len(chunk), -1))
                width = max(1, chunk_elements // (len(chunk) * dim))
                for start in range(0, len(targets), width):
                    if interrupt is not None and interrupt():
                        raise ValidationInterrupted
                    block = targets[start:start + width].to(device).unsqueeze(0)
                    result[first:first + len(chunk), start:start + block.shape[1]] = model.score(
                        query, block, negative=True).float().cpu().numpy()
    finally:
        model.train(was_training)
    return result


def aggregate(matrix, phenotypes, profiles, diseases, weight):
    """Disease-by-gene float64 scores from a phenotype-by-gene matrix; profile duplicates are kept."""
    position = {phenotype: index for index, phenotype in enumerate(phenotypes)}
    rows = []
    for disease in diseases:
        block = matrix[[position[phenotype] for phenotype in profiles[disease]]].astype(np.float64)
        rows.append(weight * block.max(0) + (1 - weight) * block.mean(0))
    return np.stack(rows)


def calibrate(scores):
    """Leave-one-disease-out per-gene standardisation over the query panel (one row per disease)."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores[:1]
    count = len(scores) - 1
    if count < 1:
        raise ValueError('Need multiple queries for calibration')
    mean = (scores.sum(0) - scores) / count
    variance = (scores * scores).sum(0) / count - scores * scores / count - mean * mean
    return (scores - mean) / (np.sqrt(np.maximum(variance, 0)) + 1e-12)


def average_ranks(scores, diseases, genes, rows):
    """Rank of each (gene, disease) row among all candidates, ties at their average position."""
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    disease_index = {disease: index for index, disease in enumerate(diseases)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    result = []
    for gene, disease in rows:
        values = scores[disease_index[disease]]
        target = values[gene_index[gene]]
        result.append(1 + (values > target).sum() + 0.5 * ((values == target).sum() - 1))
    return np.asarray(result, dtype=float)


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=float)
    return {'hits@1': float(np.mean(ranks <= 1)), 'hits@3': float(np.mean(ranks <= 3)),
            'hits@10': float(np.mean(ranks <= 10)), 'hits@100': float(np.mean(ranks <= 100)),
            'mrr': float(np.mean(1 / ranks)), 'mr': float(np.mean(ranks)), 'count': int(len(ranks))}


def selection_key(values):
    return values['hits@10'], values['hits@1'], values['mrr'], -values['mr']


def eligible(values, baseline):
    return values['hits@1'] >= baseline['hits@1'] and values['hits@10'] >= baseline['hits@10']


def readout_tables(phenotype_scores, phenotypes, profiles, diseases, genes, rows, weight):
    """Aggregated scores, calibrated scores, ranks and metrics in raw and calibrated modes."""
    aggregated = aggregate(phenotype_scores, phenotypes, profiles, diseases, weight)
    calibrated = calibrate(aggregated)
    raw_ranks = average_ranks(aggregated, diseases, genes, rows)
    calibrated_ranks = average_ranks(calibrated, diseases, genes, rows)
    return {'aggregated': aggregated, 'calibrated': calibrated, 'raw_ranks': raw_ranks,
            'calibrated_ranks': calibrated_ranks,
            'metrics': {'raw': metrics(raw_ranks), 'calibrated': metrics(calibrated_ranks)}}


def validate(model, data, diseases, rows, config, device, interrupt=None):
    """Score both readouts for the validation panel. Pure: touches no RNG and no parameters."""
    phenotypes = profile_phenotypes(data.profiles, diseases)
    phenotype_ids = [data.entities[phenotype] for phenotype in phenotypes]
    gene_ids = [data.entities[gene] for gene in data.candidates]
    options = dict(device=device, query_chunk=config['query_chunk'], chunk_elements=config['score_chunk_elements'],
                   dim=config['dim'], interrupt=interrupt)
    forward = path_scores(model, gene_ids, data.relation_ids('forward'), phenotype_ids, **options).T.copy()
    inverse = path_scores(model, phenotype_ids, data.relation_ids('inverse'), gene_ids, **options)
    matrices = {'forward': forward, 'inverse': inverse}
    tables = {readout: readout_tables(matrices[readout], phenotypes, data.profiles, diseases, data.candidates, rows,
                                      config['aggregation_weight']) for readout in READOUTS}
    arrays = {'phenotypes': np.array(phenotypes), 'genes': np.array(data.candidates), 'diseases': np.array(diseases),
              'row_genes': np.array([gene for gene, _ in rows]), 'row_diseases': np.array([disease for _, disease in rows])}
    for readout in READOUTS:
        arrays[f'{readout}_phenotype_scores'] = matrices[readout]
        arrays[f'{readout}_aggregated'] = tables[readout]['aggregated']
        arrays[f'{readout}_raw_ranks'] = tables[readout]['raw_ranks']
        arrays[f'{readout}_calibrated_ranks'] = tables[readout]['calibrated_ranks']
    arrays.update(scoring_metadata(config, device))
    return arrays, {readout: tables[readout]['metrics'] for readout in READOUTS}


def scoring_metadata(config, device):
    """Arrays naming the chunking, device and device model used to score one validation."""
    device = torch.device(device)
    name = torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type
    return {'scoring_query_chunk': np.array(config['query_chunk'], dtype=np.int64),
            'scoring_score_chunk_elements': np.array(config['score_chunk_elements'], dtype=np.int64),
            'scoring_device': np.array(str(device)), 'scoring_device_name': np.array(name)}


def recompute_from_arrays(arrays, profiles, weight):
    """Recompute both readouts' metrics from saved phenotype-level scores alone."""
    phenotypes = arrays['phenotypes'].tolist()
    diseases = arrays['diseases'].tolist()
    genes = arrays['genes'].tolist()
    rows = list(zip(arrays['row_genes'].tolist(), arrays['row_diseases'].tolist()))
    result = {}
    for readout in READOUTS:
        tables = readout_tables(arrays[f'{readout}_phenotype_scores'], phenotypes, profiles, diseases, genes, rows, weight)
        for name in ('aggregated', 'raw_ranks', 'calibrated_ranks'):
            if not np.array_equal(tables[name], arrays[f'{readout}_{name}']):
                raise ValueError(f'Saved {readout} {name} differ from the recomputation')
        result[readout] = tables['metrics']
    return result


def replay_selection(curve_metrics):
    """Apply the fixed rule to ``[(step, {readout: {'calibrated': metrics}})]`` in step order.

    Raises ``ValueError`` unless the curve is non-empty, starts at step 0 and has
    strictly increasing integer steps, because the first entry is the baseline
    and the fallback.
    """
    steps = [step for step, _ in curve_metrics]
    if not steps:
        raise ValueError('Selection curve is empty')
    if any(isinstance(step, bool) or not isinstance(step, (int, np.integer)) for step in steps):
        raise ValueError(f'Selection curve steps must be integers: {steps}')
    if steps[0] != 0:
        raise ValueError(f'Selection curve must start at step 0, got {steps[0]}')
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError(f'Selection curve steps must strictly increase: {steps}')
    best = {}
    baseline = {}
    for step, readouts in curve_metrics:
        for readout in READOUTS:
            current = readouts[readout]['calibrated']
            if readout not in baseline:
                baseline[readout] = current
                best[readout] = step, current
            elif eligible(current, baseline[readout]) and selection_key(current) > selection_key(best[readout][1]):
                best[readout] = step, current
    return {readout: best[readout][0] for readout in READOUTS}
