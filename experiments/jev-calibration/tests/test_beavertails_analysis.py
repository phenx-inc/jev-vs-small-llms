import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_beavertails_matrix import (  # noqa: E402
    clustered_paired_bootstrap,
    metric_deltas,
    parse_extra_models,
)
from analyze_beavertails_distilled import parse_models  # noqa: E402
from analyze_c_safeqa_matrix import comparison_pairs  # noqa: E402


class BeaverTailsAnalysisTests(unittest.TestCase):
    def test_extra_model_specs_are_validated(self):
        result = parse_extra_models(["nexa_ai=/tmp/nexa", "spark_x25=/tmp/spark"])
        self.assertEqual(set(result), {"nexa_ai", "spark_x25"})
        with self.assertRaises(ValueError):
            parse_extra_models(["jev_live=/tmp/jev"])
        with self.assertRaises(ValueError):
            parse_extra_models(["missing-separator"])

    def test_metric_delta_direction_favors_left(self):
        labels = [0, 0, 1, 1]
        better = [0.01, 0.1, 0.9, 0.99]
        worse = [0.4, 0.6, 0.4, 0.6]
        deltas = metric_deltas(labels, better, worse)
        self.assertGreater(deltas["auroc_left_minus_right"], 0)
        self.assertGreater(deltas["log_loss_gain_right_minus_left"], 0)
        self.assertGreater(deltas["brier_gain_right_minus_left"], 0)

    def test_cluster_bootstrap_is_reproducible(self):
        kwargs = {
            "labels": [0, 0, 1, 1, 0, 1],
            "left": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "right": [0.3, 0.4, 0.6, 0.7, 0.45, 0.55],
            "groups": ["a", "a", "b", "b", "c", "c"],
            "seed": 17,
            "draws": 100,
        }
        self.assertEqual(
            clustered_paired_bootstrap(**kwargs),
            clustered_paired_bootstrap(**kwargs),
        )

    def test_distilled_matrix_pairs_include_required_controls(self):
        names = {
            "qwen35_4b_base",
            "qwen35_4b_jev",
            "qwen35_4b_hard",
            "qwen35_4b_hybrid",
            "qwen35_9b_base",
            "qwen35_9b_jev",
            "qwen35_9b_hard",
            "qwen35_9b_hybrid",
            "nemotron3_nano_4b_base",
        }
        pairs = set(comparison_pairs(names))
        self.assertIn(("qwen35_4b_hard", "qwen35_4b_base"), pairs)
        self.assertIn(("qwen35_4b_hard", "nemotron3_nano_4b_base"), pairs)
        self.assertIn(("qwen35_9b_hard", "qwen35_4b_hard"), pairs)
        self.assertIn(("jev_live", "qwen35_4b_hard"), pairs)

    def test_distilled_model_specs_are_validated(self):
        parsed = parse_models(["qwen35_4b_hard=/tmp/adapter"])
        self.assertEqual(set(parsed), {"qwen35_4b_hard"})
        with self.assertRaises(ValueError):
            parse_models(["jev_live=/tmp/jev"])


if __name__ == "__main__":
    unittest.main()
