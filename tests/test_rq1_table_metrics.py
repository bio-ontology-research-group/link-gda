import unittest

import numpy as np

from analysis.rq1_table import metrics


class MetricsRankTests(unittest.TestCase):
    def test_all_equal_scores_use_average_rank(self):
        scores = np.ones((2, 4))
        result = metrics(scores, np.array([0, 3]))
        self.assertEqual(result["mr"], 2.5)
        self.assertEqual(result["mrr"], 0.4)
        self.assertEqual(result["h1"], 0.0)

    def test_tie_at_truth_gets_midrank(self):
        scores = np.array([[0.8, 0.8, 0.1]])
        result = metrics(scores, np.array([0]))
        self.assertEqual(result["mr"], 1.5)
        self.assertEqual(result["h1"], 0.0)
        self.assertEqual(result["h3"], 1.0)

    def test_untied_truth_uses_one_based_rank(self):
        scores = np.array([[0.9, 0.5, 0.2]])
        result = metrics(scores, np.array([1]))
        self.assertEqual(result["mr"], 2.0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["h1"], 0.0)


if __name__ == "__main__":
    unittest.main()
