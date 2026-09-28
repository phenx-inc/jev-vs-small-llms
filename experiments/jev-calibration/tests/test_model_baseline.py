import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_model_baseline import select_rows, selection_sha256, sigmoid  # noqa: E402
from analyze_model_matrix import indexed_success_rows, parse_baseline_spec  # noqa: E402
from analyze_split_calibration import logit  # noqa: E402


class ModelBaselineTest(unittest.TestCase):
    def test_selection_matches_lossless_state_limit(self):
        rows = [
            {"benchmark": "b", "id": "1", "state": {"message": "short"}},
            {"benchmark": "b", "id": "2", "state": {"message": "x" * 4100}},
        ]
        selected = select_rows(rows, 4000)
        self.assertEqual([row["id"] for row in selected], ["1"])

    def test_selection_hash_is_order_sensitive_and_deterministic(self):
        rows = [
            {"benchmark": "a", "id": "1", "state": "x"},
            {"benchmark": "b", "id": "2", "state": "y"},
        ]
        self.assertEqual(selection_sha256(rows), selection_sha256(list(rows)))
        self.assertNotEqual(selection_sha256(rows), selection_sha256(list(reversed(rows))))

    def test_sigmoid_is_stable(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(1000), 0.999)
        self.assertLess(sigmoid(-1000), 0.001)

    def test_baseline_spec(self):
        alias, path = parse_baseline_spec("spark=/tmp/predictions.jsonl")
        self.assertEqual(alias, "spark")
        self.assertEqual(path, Path("/tmp/predictions.jsonl").resolve())
        with self.assertRaises(ValueError):
            parse_baseline_spec("missing-path")

    def test_matrix_indexer_can_retain_failed_attempt_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {"benchmark": "b", "id": "1", "error": "http_502"},
                {"benchmark": "b", "id": "1", "error": None, "label": 1},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            indexed = indexed_success_rows(path, allow_errors=True)
            self.assertEqual(set(indexed), {"b:1"})
            self.assertEqual(indexed["b:1"]["label"], 1)

    def test_probability_logit_round_trip(self):
        for probability in (0.01, 0.2, 0.5, 0.8, 0.99):
            self.assertAlmostEqual(sigmoid(logit(probability)), probability)


if __name__ == "__main__":
    unittest.main()
