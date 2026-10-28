import numpy as np
import pytest
from link_gda.phenorank_simulation import cooccurrence_tiers, simulate_disease


def reference(length, tiers, rng):
    # Direct transcription of the author algorithm for a fixed RNG stream.
    for _ in range(999):
        profile = []
        probabilities = tiers[rng.choice(list(tiers.keys()), 1)[0]]
        for probability in sorted(probabilities, reverse=True):
            group = probabilities[probability]
            if len(group) <= length - len(profile):
                profile += group
            else:
                profile += list(rng.choice(group, length - len(profile)))
            if len(profile) == length:
                return profile
    raise RuntimeError


def test_sampler_matches_author_tiers_and_preserves_long_queries():
    tiers = cooccurrence_tiers([list(range(25)), [0, 1, 2], [0, 1], [1, 3]])
    assert 0 in tiers[0][1.0]
    assert tiers[0][1.0] == [0, 1]
    for length in [1, 3, 21, 25]:
        a, b = np.random.RandomState(7), np.random.RandomState(7)
        for _ in range(20):
            assert simulate_disease(length, tiers, a) == reference(length, tiers, b)
    with pytest.raises(ValueError, match='cannot simulate'):
        simulate_disease(26, tiers, np.random.RandomState(0))
