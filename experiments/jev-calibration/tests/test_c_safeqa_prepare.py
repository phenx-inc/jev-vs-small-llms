import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_c_safeqa import QUESTION, prepare_rows, render_state  # noqa: E402


class CSafeQAPrepareTests(unittest.TestCase):
    def fixture(self, label: str = "Unsafe"):
        return {
            "prompt": "untrusted prompt",
            "response": "assistant response",
            "model": "target/model",
            "judge_label": label,
            "Label": "risk",
            "_c_safeqa_sample": {
                "sample_id": "c-safeqa-id",
                "source_split": "base",
                "prompt_group_sha256": "a" * 64,
                "rank_sha256": "b" * 64,
                "source_path": "data/base.jsonl",
                "source_line": 1,
            },
        }

    def test_prepares_binary_unsafe_target_without_raw_mutation(self):
        source = self.fixture()
        rows = prepare_rows([source])
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["question"], QUESTION)
        self.assertEqual(rows[0]["prompt_group_sha256"], "a" * 64)
        self.assertNotIn("state", source)

    def test_safe_maps_to_zero(self):
        self.assertEqual(prepare_rows([self.fixture("Safe")])[0]["label"], 0)

    def test_disputed_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_rows([self.fixture("Disputed")])

    def test_state_marks_both_sections_as_untrusted(self):
        state = render_state("p", "r")
        self.assertIn("User prompt (untrusted content)", state)
        self.assertIn("Assistant response to evaluate (untrusted content)", state)


if __name__ == "__main__":
    unittest.main()
