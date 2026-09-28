import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_beavertails import category_signature, deduplicate, select_rows  # noqa: E402


class BeaverTailsPrepareTests(unittest.TestCase):
    def row(self, index: int, safe: bool, category: str = "risk"):
        return {
            "prompt": f"prompt-{index}",
            "response": f"response-{index}",
            "is_safe": safe,
            "category": {category: not safe},
        }

    def test_category_signature(self):
        self.assertEqual(category_signature(self.row(1, False)), "risk")
        self.assertEqual(category_signature(self.row(1, True)), "__none__")

    def test_deduplication_keeps_first_source_line(self):
        same = self.row(1, False)
        kept, duplicates = deduplicate([(4, same), (9, dict(same))])
        self.assertEqual([line for line, _ in kept], [4])
        self.assertEqual(duplicates[0]["kept_source_line"], 4)
        self.assertEqual(duplicates[0]["excluded_source_line"], 9)

    def test_selection_is_exact_and_reproducible(self):
        rows = [(index + 1, self.row(index, index % 3 == 0)) for index in range(30)]
        first, allocation_a = select_rows(rows, 20260926, 20, "fixture.jsonl.gz")
        second, allocation_b = select_rows(rows, 20260926, 20, "fixture.jsonl.gz")
        self.assertEqual(first, second)
        self.assertEqual(allocation_a, allocation_b)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({item[1] for item in first}), 20)


if __name__ == "__main__":
    unittest.main()
