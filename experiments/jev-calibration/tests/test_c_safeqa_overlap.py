import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_c_safeqa_overlap import (  # noqa: E402
    build_c_documents,
    build_rlcd_documents,
    exact_matches,
    internal_audit,
    near_matches,
    normalize_text,
)


class CSafeQAOverlapTest(unittest.TestCase):
    def sample_rows(self):
        return [
            {
                "prompt": "How do I bake bread?",
                "response": "Mix flour and water, then knead the dough and bake it slowly.",
                "judge_label": "Safe",
                "_c_safeqa_sample": {"sample_id": "c-1"},
            },
            {
                "prompt": "How do I bake bread?",
                "response": "Use flour, water, yeast, and salt. Knead well before baking.",
                "judge_label": "Safe",
                "_c_safeqa_sample": {"sample_id": "c-2"},
            },
        ]

    def test_normalization(self):
        self.assertEqual(normalize_text("  CAFÉ\nTest "), "café test")

    def test_internal_prompt_grouping(self):
        result = internal_audit(self.sample_rows())
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["unique_normalized_prompts"], 1)
        self.assertEqual(result["duplicate_prompt_groups"], 1)
        self.assertEqual(result["duplicate_prompt_response_groups"], 0)

    def test_exact_and_near_matches_are_state_free(self):
        sample = build_c_documents(self.sample_rows())
        rlcd = build_rlcd_documents(
            [
                {
                    "benchmark": "fixture",
                    "id": "r-1",
                    "state": {
                        "query": "HOW DO I BAKE BREAD?",
                        "answer": "Mix flour and water, then knead the dough and bake it very slowly.",
                    },
                }
            ]
        )
        exact = exact_matches(sample, rlcd)
        self.assertEqual(len(exact), 2)
        self.assertTrue(all(item["c_safeqa_field"] == "prompt" for item in exact))
        near, _ = near_matches(
            sample,
            rlcd,
            min_tokens=5,
            max_postings=200,
            jaccard_threshold=0.50,
            containment_threshold=0.70,
        )
        self.assertTrue(any(item["c_safeqa_field"] == "response" for item in near))
        serialized = json.dumps({"exact": exact, "near": near})
        self.assertNotIn("Mix flour", serialized)
        self.assertNotIn("bake bread", serialized)


if __name__ == "__main__":
    unittest.main()
