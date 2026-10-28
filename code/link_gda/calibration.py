"""Per-gene standardization with explicit reference-panel units.

The historical paper used association-row leave-one-out. Revised runs use
one score vector per disease; no association multiplicity enters the panel.
"""
import numpy as np


def unique_scores(scores, diseases):
    scores = np.asarray(scores, dtype=np.float64)
    diseases = np.asarray(diseases, dtype=str)
    if scores.ndim != 2 or len(scores) != len(diseases) or not len(scores):
        raise ValueError('Expected nonempty score matrix and one disease per row')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite')
    _, first, inverse = np.unique(diseases, return_index=True, return_inverse=True)
    unique = scores[first]
    if not np.array_equal(scores, unique[inverse]):
        raise ValueError('Repeated disease has inconsistent candidate scores')
    return unique, inverse


def standardize(scores, diseases, unit='disease', epsilon=1e-12):
    scores = np.asarray(scores, dtype=np.float64)
    if unit == 'disease':
        reference, inverse = unique_scores(scores, diseases)
    elif unit == 'row':
        if scores.ndim != 2 or len(scores) != len(diseases) or not np.isfinite(scores).all():
            raise ValueError('Invalid scores or disease identifiers')
        reference, inverse = scores, np.arange(len(scores))
    else:
        raise ValueError('Calibration unit must be disease or row')
    n = len(reference)
    if n < 2:
        raise ValueError('Leave-one-out requires at least two reference units')
    # Centre before computing moments to avoid catastrophic cancellation for
    # nearly constant scores with a large offset. Keep explicit historical row
    # mode numerically unchanged for reproducing the archived study.
    if unit == 'disease':
        reference = reference-reference[0:1]
    mean = (reference.sum(axis=0, keepdims=True)-reference)/(n-1)
    variance = ((reference**2).sum(axis=0, keepdims=True)-reference**2)/(n-1)-mean**2
    z = (reference-mean)/(np.sqrt(np.clip(variance,0,None))+epsilon)
    return z[inverse]


def reference_statistics(scores, diseases, unit='disease'):
    scores = np.asarray(scores, dtype=np.float64)
    if unit == 'disease':
        scores, _ = unique_scores(scores, diseases)
    elif unit != 'row':
        raise ValueError('Calibration unit must be disease or row')
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid reference scores')
    return scores.mean(axis=0), scores.std(axis=0)


def standardize_fixed(scores, reference_scores, reference_diseases, epsilon=1e-12):
    mean, sd = reference_statistics(reference_scores, reference_diseases)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(mean) or not np.isfinite(scores).all():
        raise ValueError('Query and reference candidate columns must match')
    return (scores-mean)/(sd+epsilon)
