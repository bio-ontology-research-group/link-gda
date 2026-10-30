"""PhenoRank's decreasing co-occurrence-tier phenotype simulation.

Algorithm: Cornish et al. (2018), doi:10.1093/bioinformatics/bty028;
reference implementation: alexjcornish/PhenoRank, phenorank/scoring.py.
Copyright (c) 2017, Alex J. Cornish. BSD terms: third_party/PhenoRank-LICENSE.txt.
The final partially used tier is sampled WITH replacement, as in the author
implementation. Ancestor closure later treats the sampled list as a set.
"""
from collections import Counter, defaultdict
import numpy as np


def cooccurrence_tiers(profiles):
    counts = defaultdict(Counter)
    for profile in profiles:
        terms = sorted(set(profile))
        for anchor in terms:
            counts[anchor].update(terms)  # Includes the anchor itself (probability 1).
    result = {}
    for anchor in sorted(counts):
        groups = defaultdict(list)
        for term, count in sorted(counts[anchor].items()):
            groups[count / counts[anchor][anchor]].append(term)
        result[anchor] = dict(groups)
    if not result:
        raise ValueError('No training phenotypes for simulation')
    return result


def simulate_disease(length, tiers, rng):
    """Follow the author's tier order, retry rule and partial-tier sampling."""
    if length < 1:
        raise ValueError('Query length must be positive')
    anchors = list(tiers)
    if not anchors or max(sum(map(len, groups.values())) for groups in tiers.values()) < length:
        raise ValueError(f'Training co-occurrence vocabulary cannot simulate query length {length}')
    for _ in range(999):
        groups = tiers[rng.choice(anchors, 1)[0]]
        selected = []
        for probability in sorted(groups, reverse=True):
            terms = groups[probability]
            needed = length - len(selected)
            if len(terms) <= needed:
                selected.extend(terms)
            else:
                selected.extend(rng.choice(terms, needed, replace=True).tolist())
            if len(selected) == length:
                return selected
    raise RuntimeError('No sufficiently large co-occurrence profile found after 999 attempts')
