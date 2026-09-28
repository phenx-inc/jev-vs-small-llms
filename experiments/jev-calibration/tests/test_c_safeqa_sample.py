import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sample_c_safeqa import allocate_hamilton, build_sample, canonical_json  # noqa: E402


class CSafeQASampleTests(unittest.TestCase):
    def test_hamilton_allocation_is_exact_and_deterministic(self):
        counts = {
            ("base", "Safe", "__base__"): 8,
            ("base", "Unsafe", "__base__"): 2,
            ("transformed", "Safe", "translation"): 5,
        }
        first = allocate_hamilton(counts, 7)
        second = allocate_hamilton(counts, 7)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 7)
        self.assertEqual(first[("base", "Safe", "__base__")], 4)

    def test_sample_is_reproducible_and_excludes_disputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            rows = []
            for index in range(24):
                rows.append(
                    {
                        "prompt": f"prompt-{index}",
                        "response": f"response-{index}",
                        "model": "fixture-model",
                        "judge_label": "Disputed" if index in {4, 19} else ("Safe" if index % 3 else "Unsafe"),
                        "Label": "fixture-risk",
                    }
                )
            source = data / "base.jsonl"
            source.write_text(
                "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
            )
            config = {
                "protocol_id": "fixture",
                "dataset": {
                    "repo_id": "fixture/repo",
                    "revision": "abc",
                    "license": "fixture",
                    "expected_total_rows": 24,
                    "files": [{"split": "base", "path": "base.jsonl", "expected_rows": 24}],
                },
                "sampling": {
                    "seed": 20260926,
                    "target_rows": 10,
                    "reference_label_field": "judge_label",
                    "eligible_reference_labels": ["Safe", "Unsafe"],
                    "excluded_reference_labels": ["Disputed"],
                    "base_method_value": "__base__",
                    "ranking": "fixture ranking description",
                    "deduplication": "fixture exact normalized prompt-response deduplication",
                },
            }
            sample_a, manifest_a = build_sample(config, data)
            sample_b, manifest_b = build_sample(config, data)
            self.assertEqual(sample_a, sample_b)
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(len(sample_a), 10)
            self.assertTrue(all(row["judge_label"] != "Disputed" for row in sample_a))
            self.assertEqual(manifest_a["dataset"]["excluded_rows"], 2)
            self.assertEqual(
                len({row["_c_safeqa_sample"]["sample_id"] for row in sample_a}), 10
            )
            self.assertEqual(manifest_a["sample_unique_exact_prompt_groups"], 10)
            self.assertEqual(
                manifest_a["sample_unique_normalized_prompt_response_pairs"], 10
            )

            config["sampling"]["seed"] = 7
            sample_c, _ = build_sample(config, data)
            ids_a = {row["_c_safeqa_sample"]["sample_id"] for row in sample_a}
            ids_c = {row["_c_safeqa_sample"]["sample_id"] for row in sample_c}
            self.assertNotEqual(ids_a, ids_c)

    def test_exact_prompt_response_duplicates_are_backfilled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "prompt": "Same prompt",
                    "response": "Same response",
                    "model": "a",
                    "judge_label": "Safe",
                    "method": "m",
                },
                {
                    "prompt": "  SAME   PROMPT ",
                    "response": "same response",
                    "model": "b",
                    "judge_label": "Safe",
                    "method": "m",
                },
                {
                    "prompt": "unique 1",
                    "response": "answer 1",
                    "model": "a",
                    "judge_label": "Safe",
                    "method": "m",
                },
                {
                    "prompt": "unique 2",
                    "response": "answer 2",
                    "model": "b",
                    "judge_label": "Safe",
                    "method": "m",
                },
            ]
            source = root / "rows.jsonl"
            source.write_text(
                "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
            )
            config = {
                "protocol_id": "dedup-fixture",
                "dataset": {
                    "repo_id": "fixture/repo",
                    "revision": "abc",
                    "license": "fixture",
                    "expected_total_rows": 4,
                    "files": [{"split": "x", "path": "rows.jsonl", "expected_rows": 4}],
                },
                "sampling": {
                    "seed": 1,
                    "target_rows": 3,
                    "reference_label_field": "judge_label",
                    "eligible_reference_labels": ["Safe", "Unsafe"],
                    "excluded_reference_labels": ["Disputed"],
                    "base_method_value": "__base__",
                    "ranking": "fixture",
                    "deduplication": "fixture",
                },
            }
            sample, manifest = build_sample(config, root)
            self.assertEqual(len(sample), 3)
            self.assertEqual(manifest["sample_unique_normalized_prompt_response_pairs"], 3)
            self.assertGreaterEqual(
                manifest["selection_algorithm"]["duplicate_candidates_skipped_during_selection"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
