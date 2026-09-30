"""Characterization of the two AUC definitions currently in the codebase.

They are DIFFERENT quantities and this test pins both, on purpose, without
unifying them:

- rq1_table.metrics: algebraic, AUC = (n - MR) / (n - 1), where MR is the mean
  average rank over the n candidates.
- evaluate_sem_sim.compute_rank_roc: rank-ROC, the trapezoidal integral of the
  cumulative-rank distribution over candidate ranks, divided by n.

On the fixture below (ranks 1, 1, 2.5 over n=4) the algebraic value is 5/6 and
the trapezoid value is 11/16; they are not interchangeable.
"""
import unittest

import numpy as np

import analysis.rq1_table as rq1_table

FIXTURE_SCORES = np.array([
    [0.9, 0.5, 0.2, 0.1],
    [0.1, 0.2, 0.5, 0.9],
    [0.5, 0.5, 0.5, 0.5],
])
FIXTURE_TRUE_INDEX = np.array([0, 3, 0])
N_CANDIDATES = 4
RANK_HISTOGRAM = {1.0: 2, 2.5: 1}


def reference_trapezoid_auc(rank_histogram, n_candidates):
    """Independent pure-python trapezoid integral of the cumulative rank CDF."""
    total = sum(rank_histogram.values())
    xs, ys = [], []
    cumulative = 0
    for rank in sorted(rank_histogram):
        cumulative += rank_histogram[rank]
        xs.append(rank)
        ys.append(cumulative / total)
    xs.append(n_candidates)
    ys.append(1.0)
    area = sum(0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    return area / n_candidates


class AlgebraicAucTests(unittest.TestCase):
    def test_hand_computed_values(self):
        result = rq1_table.metrics(FIXTURE_SCORES, FIXTURE_TRUE_INDEX)
        self.assertAlmostEqual(result["mr"], 1.5)
        self.assertAlmostEqual(result["auc"], 5.0 / 6.0)


class TrapezoidAucTests(unittest.TestCase):
    def test_reference_is_hand_computable(self):
        self.assertAlmostEqual(reference_trapezoid_auc(RANK_HISTOGRAM, N_CANDIDATES), 11.0 / 16.0)
        uniform = {1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1}
        self.assertAlmostEqual(reference_trapezoid_auc(uniform, 4), 0.46875)

    def test_production_compute_rank_roc_matches_reference(self):
        trapezoid_available = hasattr(np, "trapezoid")
        if not trapezoid_available:
            self.skipTest(
                f"GAP-auc-trapezoid: numpy {np.__version__} lacks np.trapezoid (needs >=2.0), "
                "so evaluate_sem_sim.compute_rank_roc is not executable in this environment. "
                "Explicit gap, recorded in the baseline; the definition is characterized by "
                "reference_trapezoid_auc instead."
            )
        try:
            import link_gda.evaluate_sem_sim as evaluate_sem_sim
        except ModuleNotFoundError as exc:
            self.skipTest(f"GAP-auc-trapezoid: evaluate_sem_sim dependency unavailable: {exc}")
        self.assertAlmostEqual(
            evaluate_sem_sim.compute_rank_roc(RANK_HISTOGRAM, N_CANDIDATES), 11.0 / 16.0
        )


class DefinitionSplitTests(unittest.TestCase):
    def test_the_two_definitions_give_different_values(self):
        algebraic = rq1_table.metrics(FIXTURE_SCORES, FIXTURE_TRUE_INDEX)["auc"]
        trapezoid = reference_trapezoid_auc(RANK_HISTOGRAM, N_CANDIDATES)
        self.assertNotAlmostEqual(algebraic, trapezoid)


if __name__ == "__main__":
    unittest.main()
