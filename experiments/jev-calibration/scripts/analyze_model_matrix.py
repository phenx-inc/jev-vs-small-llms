#!/usr/bin/env python3
"""Compare live Jev with multiple state-free local-model prediction files."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import summarize
from rlcd_protocol import bootstrap_median_ci, read_jsonl, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev-dir", required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Alias=predictions.jsonl; may be repeated",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def parse_baseline_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("baseline must be alias=path")
    alias, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", alias):
        raise ValueError(f"invalid baseline alias: {alias!r}")
    if not raw_path:
        raise ValueError("baseline path is empty")
    return alias, Path(raw_path).resolve()


def key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {name: finite_or_none(item) for name, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def binary_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    labels = [int(row["label"]) for row in rows]
    if not rows or len(set(labels)) < 2:
        return None
    return finite_or_none(summarize(labels, [float(row[field]) for row in rows]))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / math.sqrt(left_scale * right_scale)


def probability_field(rows: list[dict[str, Any]]) -> str:
    candidates = ("model_probability_yes", "qwen_probability_yes")
    present = [name for name in candidates if rows and name in rows[0]]
    if len(present) != 1:
        raise RuntimeError(f"could not identify one probability field: {present}")
    return present[0]


def indexed_success_rows(path: Path, allow_errors: bool = False) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if not allow_errors and any(row.get("error") is not None for row in rows):
        raise RuntimeError(f"prediction errors present in {path}")
    successful = [row for row in rows if row.get("error") is None]
    result = {key(row): row for row in successful}
    if len(result) != len(successful):
        raise RuntimeError(f"duplicate prediction keys in {path}")
    return result


def manifest_for_predictions(path: Path) -> dict[str, Any] | None:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    jev_dir = Path(args.jev_dir).resolve()
    jev_predictions = jev_dir / "predictions.jsonl"
    jev_manifest = json.loads((jev_dir / "manifest.json").read_text(encoding="utf-8"))
    jev_status = json.loads((jev_dir / "status.json").read_text(encoding="utf-8"))
    if jev_status.get("state") != "complete":
        raise RuntimeError("Jev run is not complete")
    jev_attempt_rows = read_jsonl(jev_predictions)
    jev_failed_attempt_records = sum(
        row.get("error") is not None for row in jev_attempt_rows
    )
    # Resume-safe Jev files retain failed attempts for auditability. A later
    # successful retry for the same key is authoritative; failed attempt rows
    # are counted below and never enter the comparison matrix.
    jev_by_key = indexed_success_rows(jev_predictions, allow_errors=True)
    if len(jev_by_key) != int(jev_manifest["selected_examples"]):
        raise RuntimeError("Jev successful row count does not match its manifest")
    if jev_failed_attempt_records != int(jev_status.get("failed_attempt_records", 0)):
        raise RuntimeError("Jev failed-attempt count does not match its status")
    if any({"state", "question", "questions"} & set(row) for row in jev_by_key.values()):
        raise RuntimeError("gated text leaked into Jev prediction output")

    specs = [parse_baseline_spec(value) for value in args.baseline]
    if len({alias for alias, _ in specs}) != len(specs):
        raise RuntimeError("duplicate baseline alias")
    baseline_data: dict[str, dict[str, Any]] = {}
    expected_keys = set(jev_by_key)
    for alias, path in specs:
        rows_by_key = indexed_success_rows(path)
        missing = expected_keys - set(rows_by_key)
        if missing:
            raise RuntimeError(f"{alias} is missing {len(missing)} Jev-selected rows")
        selected = [rows_by_key[item] for item in jev_by_key]
        field = probability_field(selected)
        for item_key, jev_row in jev_by_key.items():
            row = rows_by_key[item_key]
            if int(row["label"]) != int(jev_row["label"]):
                raise RuntimeError(f"label mismatch for {alias}/{item_key}")
            if row.get("question_sha256") != jev_row.get("question_sha256"):
                raise RuntimeError(f"question hash mismatch for {alias}/{item_key}")
        baseline_data[alias] = {
            "path": path,
            "rows": rows_by_key,
            "field": field,
            "manifest": manifest_for_predictions(path),
        }

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "comparison_predictions.jsonl"
    matrix_rows: list[dict[str, Any]] = []
    with matrix_path.open("w", encoding="utf-8", buffering=1) as output:
        for item_key, jev_row in jev_by_key.items():
            row = {
                "id": jev_row["id"],
                "benchmark": jev_row["benchmark"],
                "failure_type": jev_row.get("failure_type"),
                "benchmark_split": jev_row.get("benchmark_split"),
                "label": int(jev_row["label"]),
                "question_sha256": jev_row.get("question_sha256"),
                "live_jev": float(jev_row["live_jev_probability_yes"]),
            }
            for alias, data in baseline_data.items():
                source = data["rows"][item_key]
                row[alias] = float(source[data["field"]])
            output.write(json.dumps(row, sort_keys=True) + "\n")
            matrix_rows.append(row)

    model_names = ["live_jev", *baseline_data]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        grouped[row["benchmark"]].append(row)
    per_benchmark = []
    for benchmark, rows in sorted(grouped.items()):
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "n": len(rows),
                "failure_type": rows[0]["failure_type"],
                "benchmark_split": rows[0]["benchmark_split"],
                "models": {name: binary_metrics(rows, name) for name in model_names},
            }
        )

    aggregate: dict[str, Any] = {}
    for name in model_names:
        valid = [
            row["models"][name]
            for row in per_benchmark
            if row["models"][name] is not None
        ]
        aggregate[name] = {
            "comparable_benchmarks": len(valid),
            "median": {
                metric: statistics.median(
                    float(metrics[metric])
                    for metrics in valid
                    if metrics[metric] is not None
                )
                for metric in ("accuracy", "balanced_accuracy", "auroc", "log_loss", "brier", "ece_15", "adaptive_ece_15")
            },
            "pooled": binary_metrics(matrix_rows, name),
        }

    comparisons: dict[str, Any] = {}
    for alias in baseline_data:
        comparable = [
            row
            for row in per_benchmark
            if row["models"]["live_jev"] is not None
            and row["models"][alias] is not None
        ]
        auroc_deltas = [
            float(row["models"]["live_jev"]["auroc"])
            - float(row["models"][alias]["auroc"])
            for row in comparable
        ]
        log_loss_gains = [
            float(row["models"][alias]["log_loss"])
            - float(row["models"]["live_jev"]["log_loss"])
            for row in comparable
        ]
        brier_gains = [
            float(row["models"][alias]["brier"])
            - float(row["models"]["live_jev"]["brier"])
            for row in comparable
        ]
        jev_values = [float(row["live_jev"]) for row in matrix_rows]
        baseline_values = [float(row[alias]) for row in matrix_rows]
        comparisons[alias] = {
            "comparable_benchmarks": len(comparable),
            "median_auroc_delta_jev_minus_baseline": statistics.median(auroc_deltas),
            "auroc_delta_bootstrap_ci95": bootstrap_median_ci(auroc_deltas, args.seed),
            "median_log_loss_gain_baseline_minus_jev": statistics.median(log_loss_gains),
            "log_loss_gain_bootstrap_ci95": bootstrap_median_ci(log_loss_gains, args.seed + 1),
            "median_brier_gain_baseline_minus_jev": statistics.median(brier_gains),
            "brier_gain_bootstrap_ci95": bootstrap_median_ci(brier_gains, args.seed + 2),
            "benchmark_auroc_wins_ties_losses_for_jev": {
                "wins": sum(value > 0 for value in auroc_deltas),
                "ties": sum(value == 0 for value in auroc_deltas),
                "losses": sum(value < 0 for value in auroc_deltas),
            },
            "probability_pearson": pearson(jev_values, baseline_values),
            "threshold_disagreement_rate": sum(
                (jev >= 0.5) != (candidate >= 0.5)
                for jev, candidate in zip(jev_values, baseline_values)
            )
            / len(matrix_rows),
        }

    report = finite_or_none(
        {
            "examples": len(matrix_rows),
            "benchmarks": len(grouped),
            "primary_estimand": "Median per-benchmark metric; pooled metrics are secondary because domains and base rates differ.",
            "models": model_names,
            "aggregate": aggregate,
            "jev_vs_baseline": comparisons,
            "per_benchmark": per_benchmark,
            "run_metadata": {
                "live_jev": {
                    "manifest": jev_manifest,
                    "status": jev_status,
                    "predictions_path": str(jev_predictions),
                },
                **{
                    alias: {
                        "manifest": data["manifest"],
                        "predictions_path": str(data["path"]),
                        "probability_field": data["field"],
                    }
                    for alias, data in baseline_data.items()
                },
            },
            "integrity": {
                "jev_selection_sha256": jev_manifest["selection_sha256"],
                "matrix_sha256": sha256_file(matrix_path),
                "jev_predictions_sha256": sha256_file(jev_predictions),
                "jev_failed_attempt_records_retained": jev_failed_attempt_records,
                "baseline_predictions_sha256": {
                    alias: sha256_file(data["path"])
                    for alias, data in baseline_data.items()
                },
                "gated_text_absent": True,
            },
        }
    )
    write_json(out_dir / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
