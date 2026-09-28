import math
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from metrics import (  # noqa: E402
    adaptive_calibration_error,
    auroc,
    expected_calibration_error,
    fit_temperature,
    log_loss,
    probabilities,
    selective_accuracy,
    summarize,
)


class MetricsTest(unittest.TestCase):
    def test_perfect_binary_metrics(self):
        labels = [0, 0, 1, 1]
        probs = [0.0, 0.1, 0.9, 1.0]
        result = summarize(labels, probs)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertLess(result["brier"], 0.01)

    def test_auroc_ties(self):
        self.assertAlmostEqual(auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(auroc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0)

    def test_equal_width_and_adaptive_ece(self):
        labels = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(expected_calibration_error(labels, probs, bins=2), 0.15)
        ace, rows = adaptive_calibration_error(labels, probs, bins=2)
        self.assertAlmostEqual(ace, 0.15)
        self.assertEqual(sum(row["count"] for row in rows), 4)

    def test_adaptive_ece_does_not_split_constant_score_ties(self):
        labels = [0, 1, 1, 0, 1]
        probs = [0.6] * len(labels)
        ace, rows = adaptive_calibration_error(labels, probs, bins=3)
        self.assertAlmostEqual(ace, 0.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 5)

    def test_temperature_reduces_overconfidence_loss(self):
        labels = [1, 0, 1, 0, 1, 0]
        deltas = [8.0, 7.0, 6.0, -8.0, -7.0, -6.0]
        raw = log_loss(labels, probabilities(deltas))
        temperature = fit_temperature(labels, deltas)
        calibrated = log_loss(labels, probabilities(deltas, temperature))
        self.assertGreater(temperature, 1.0)
        self.assertLess(calibrated, raw)

    def test_selective_accuracy_uses_confidence(self):
        labels = [1, 0, 1, 0]
        probs = [0.99, 0.01, 0.51, 0.51]
        rows = selective_accuracy(labels, probs, coverages=(0.5, 1.0))
        self.assertEqual(rows[0]["accuracy"], 1.0)
        self.assertEqual(rows[1]["accuracy"], 0.75)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            probabilities([0.0], 0.0)
        with self.assertRaises(ValueError):
            probabilities([0.0], math.nan)


if __name__ == "__main__":
    unittest.main()
