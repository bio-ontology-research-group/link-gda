"""Hand-computable checks for the leave-one-out per-gene normalization.

The LOO reference values below are derived by hand from the fixture matrix:
for each query row the baseline of a gene is computed over the OTHER queries
only (the scored query never contributes to its own baseline).
"""
import unittest

import numpy as np

import calibrate_scores
import rq1_table

FIXED = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [5.0, 2.0, 3.0]])


class LeaveOneOutBaselineTests(unittest.TestCase):
    def test_loo_baseline_excludes_query_and_matches_hand_reference(self):
        mean, spread = calibrate_scores.baselines(FIXED)
        expected_mean = np.array([[4.0, 2.0, 3.0], [3.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        expected_spread = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(mean, expected_mean, atol=1e-9)
        np.testing.assert_allclose(spread, expected_spread, atol=1e-9)

    def test_zero_variance_gene_is_clipped_to_epsilon_not_nan(self):
        constant = np.full((4, 1), 7.0)
        mean, spread = calibrate_scores.baselines(constant)
        np.testing.assert_allclose(mean, 7.0, atol=1e-9)
        np.testing.assert_allclose(spread, 1e-12, atol=1e-15)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(spread).all())

    def test_single_query_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_scores.baselines(np.array([[1.0, 2.0]]))


class Rq1CalibrateTests(unittest.TestCase):
    def test_zscore_matches_hand_reference(self):
        result = rq1_table.calibrate(FIXED)
        expected = np.array([
            [1.0 - 4.0, 0.0, 0.0],
            [3.0 - 3.0, 0.0, 0.0],
            [5.0 - 2.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_constant_gene_calibrates_to_exactly_zero(self):
        result = rq1_table.calibrate(FIXED)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[:, 2], 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
