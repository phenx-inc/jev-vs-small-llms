import math
import unittest

from rlcd_distillation import (
    assign_benchmark_folds,
    soft_cross_entropy,
    target_for_arm,
    teacher_metrics,
    validate_and_partition,
)


class DistillationProtocolTests(unittest.TestCase):
    def test_targets_are_registered(self):
        row = {"jev_probability_yes": 0.2, "label": 1}
        self.assertEqual(target_for_arm(row, "jev"), 0.2)
        self.assertEqual(target_for_arm(row, "hard"), 1.0)
        self.assertEqual(target_for_arm(row, "hybrid"), 0.6)

    def test_soft_cross_entropy(self):
        actual = soft_cross_entropy([0.25, 0.75], [0.0, 1.0])
        self.assertAlmostEqual(actual, -math.log(0.75))
        metrics = teacher_metrics([0.2, 0.8], [0.2, 0.8])
        self.assertAlmostEqual(metrics["kl_teacher_to_model"], 0.0)
        self.assertAlmostEqual(metrics["mean_squared_error"], 0.0)

    def test_grouped_folds_never_split_benchmark(self):
        rows = [
            {"benchmark": benchmark, "id": f"{benchmark}:{index}"}
            for benchmark, count in (("a", 10), ("b", 7), ("c", 5), ("d", 3))
            for index in range(count)
        ]
        assignment, totals = assign_benchmark_folds(rows, 3, 123)
        self.assertEqual(set(assignment), {"a", "b", "c", "d"})
        self.assertEqual(sum(totals), len(rows))
        self.assertEqual(assignment, assign_benchmark_folds(rows, 3, 123)[0])

    def test_partition_rejects_wrong_registry(self):
        with self.assertRaises(RuntimeError):
            validate_and_partition([], expected_source_rows=1)


if __name__ == "__main__":
    unittest.main()
