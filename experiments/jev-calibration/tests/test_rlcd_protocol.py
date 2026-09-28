import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rlcd_protocol import (  # noqa: E402
    bootstrap_median_ci,
    choose_response_file,
    extract_jev_probability,
    generic_question,
    module_name_from_battery,
    promise_verdict,
    render_state,
    selected_release_paths,
)


class RlcdProtocolTest(unittest.TestCase):
    def test_render_state_matches_upstream_layout(self):
        self.assertEqual(
            render_state({"question": "Q", "turns": [{"role": "user", "content": "x"}]}),
            '### question\nQ\n\n### turns\n[\n {\n  "role": "user",\n  "content": "x"\n }\n]',
        )

    def test_battery_name_parsing(self):
        self.assertEqual(module_name_from_battery("battery_a_judge_v2"), "battery_a_judge")
        self.assertEqual(module_name_from_battery("battery_refusal"), "battery_refusal")
        with self.assertRaises(ValueError):
            module_name_from_battery("battery_unknown_v1")

    def test_extract_probability_validation(self):
        self.assertEqual(
            extract_jev_probability({"answers": {"misbehaves_noul": {"noul": 0.73}}}),
            0.73,
        )
        with self.assertRaises(ValueError):
            extract_jev_probability({"answers": {}})

    def test_exact_generic_question_reconstruction_across_batteries(self):
        vendor_code = Path(__file__).resolve().parents[1] / "vendor" / "RLCDAlignBench" / "code"
        sys.path.insert(0, str(vendor_code))
        checked = 0
        for module_name in (
            "battery_a_judge",
            "battery_bc_paired",
            "battery_d_rule",
            "battery_e",
            "battery_f_trajectory",
            "battery_refusal",
        ):
            module = importlib.import_module(module_name)
            for sample in module.synthetic_samples():
                questions = module.questions(sample)
                if "misbehaves_noul" not in questions:
                    continue
                released_shape = dict(sample)
                released_shape["benchmark"] = released_shape.pop("leg")
                question = generic_question(
                    released_shape, f"{module_name}_v2", vendor_code
                )
                self.assertEqual(question, questions["misbehaves_noul"])
                checked += 1
        self.assertGreater(checked, 50)

    def test_choose_one_complete_object_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "battery_a_judge_v2.jsonl"
            complete.write_text(
                "\n".join(
                    json.dumps({"id": value, "answers": {"misbehaves_noul": {"noul": 0.5}}})
                    for value in ("a", "b", "officially_excluded_extra")
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "battery_a_judge_v2__text.jsonl").write_text(
                complete.read_text(encoding="utf-8"), encoding="utf-8"
            )
            path, rows = choose_response_file(root, {"a", "b"})
            self.assertEqual(path, complete)
            self.assertEqual(len(rows), 3)

    def test_release_path_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.csv"
            variants = root / "variants.csv"
            with main.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["leg", "naive_noul_auroc"])
                writer.writeheader()
                for index in range(31):
                    writer.writerow({"leg": f"b{index}", "naive_noul_auroc": "0.7"})
            with variants.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["benchmark", "variant", "is_canonical", "path"]
                )
                writer.writeheader()
                for index in range(31):
                    writer.writerow(
                        {
                            "benchmark": f"b{index}",
                            "variant": "main",
                            "is_canonical": "True",
                            "path": f"data/variants/b{index}/main.jsonl",
                        }
                    )
            files = ["README.md", "benchmarks.csv", "file_map.csv", "variants.csv"]
            for index in range(31):
                files += [
                    f"data/variants/b{index}/main.jsonl",
                    f"jev/responses/b{index}/main/battery_a_judge_v2.jsonl",
                    f"jev/responses/b{index}/main/battery_a_judge_v2__text.jsonl",
                    f"jev/metrics/b{index}/main/battery_a_judge_v2.json",
                ]
            selected = selected_release_paths(files, main, variants)
            self.assertEqual(sum(path.startswith("data/variants/") for path in selected), 31)
            self.assertEqual(sum(path.startswith("jev/responses/") for path in selected), 31)
            self.assertEqual(sum(path.startswith("jev/metrics/") for path in selected), 31)
            self.assertFalse(any("__text" in path for path in selected))

    def test_promise_gate(self):
        self.assertEqual(
            promise_verdict(
                median_auroc_delta_jev_minus_qwen=0.03,
                median_log_loss_gain_qwen_minus_jev=-0.1,
                median_brier_gain_qwen_minus_jev=-0.1,
            ),
            "clear_promise",
        )
        self.assertEqual(
            promise_verdict(
                median_auroc_delta_jev_minus_qwen=-0.03,
                median_log_loss_gain_qwen_minus_jev=-0.01,
                median_brier_gain_qwen_minus_jev=-0.01,
            ),
            "clear_no_promise",
        )
        self.assertEqual(
            promise_verdict(
                median_auroc_delta_jev_minus_qwen=0.0,
                median_log_loss_gain_qwen_minus_jev=0.03,
                median_brier_gain_qwen_minus_jev=0.0,
            ),
            "clear_promise",
        )

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_median_ci([0.1, 0.2, 0.3], seed=7, draws=100)
        second = bootstrap_median_ci([0.1, 0.2, 0.3], seed=7, draws=100)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
