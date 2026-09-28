#!/usr/bin/env python3
"""Fit scalar temperatures on hill-climb rows and score held-out benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import fit_temperature, probabilities, summarize
from rlcd_protocol import read_jsonl, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return math.log(clipped) - math.log1p(-clipped)


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def compact(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        key: metrics.get(key)
        for key in (
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "log_loss",
            "brier",
            "ece_15",
            "adaptive_ece_15",
        )
    }


def median_benchmark_metrics(
    rows: list[dict[str, Any]], probabilities_by_key: dict[str, float]
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    summaries = []
    for items in grouped.values():
        labels = [int(row["label"]) for row in items]
        if len(set(labels)) < 2:
            continue
        summaries.append(
            summarize(
                labels,
                [probabilities_by_key[f"{row['benchmark']}:{row['id']}"] for row in items],
            )
        )
    return {
        metric: statistics.median(float(item[metric]) for item in summaries)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "log_loss",
            "brier",
            "ece_15",
            "adaptive_ece_15",
        )
    }


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    rows = read_jsonl(matrix_path)
    if not rows:
        raise RuntimeError("matrix is empty")
    required = {"id", "benchmark", "benchmark_split", "label", "question_sha256"}
    if not required <= set(rows[0]):
        raise RuntimeError("matrix is missing required fields")
    model_names = sorted(set(rows[0]) - required - {"failure_type"})
    train = [row for row in rows if row["benchmark_split"] == "hill-climb"]
    held_out = [row for row in rows if row["benchmark_split"] == "held-out"]
    if not train or not held_out:
        raise RuntimeError("both hill-climb and held-out rows are required")

    report: dict[str, Any] = {
        "design": {
            "fit_split": "hill-climb",
            "evaluation_split": "held-out",
            "method": "One positive scalar temperature per model, fitted by held-in log-loss minimization. No intercept and no prompt tuning.",
            "ranking_note": "Positive temperature scaling is monotone, so AUROC is unchanged by construction.",
            "fit_rows": len(train),
            "held_out_rows": len(held_out),
            "held_out_benchmarks": len({row["benchmark"] for row in held_out}),
        },
        "models": {},
        "integrity": {
            "matrix_sha256": sha256_file(matrix_path),
            "gated_text_absent": all(
                not ({"state", "question", "questions"} & set(row)) for row in rows
            ),
        },
    }
    for model in model_names:
        train_labels = [int(row["label"]) for row in train]
        train_deltas = [logit(float(row[model])) for row in train]
        temperature = fit_temperature(train_labels, train_deltas)
        held_labels = [int(row["label"]) for row in held_out]
        held_deltas = [logit(float(row[model])) for row in held_out]
        raw_probabilities = [float(row[model]) for row in held_out]
        scaled_probabilities = probabilities(held_deltas, temperature)
        raw_metrics = summarize(held_labels, raw_probabilities)
        scaled_metrics = summarize(held_labels, scaled_probabilities)
        raw_by_key = {
            f"{row['benchmark']}:{row['id']}": value
            for row, value in zip(held_out, raw_probabilities)
        }
        scaled_by_key = {
            f"{row['benchmark']}:{row['id']}": value
            for row, value in zip(held_out, scaled_probabilities)
        }
        report["models"][model] = {
            "temperature": temperature,
            "held_out_pooled_raw": compact(raw_metrics),
            "held_out_pooled_temperature_scaled": compact(scaled_metrics),
            "held_out_median_per_benchmark_raw": median_benchmark_metrics(
                held_out, raw_by_key
            ),
            "held_out_median_per_benchmark_temperature_scaled": median_benchmark_metrics(
                held_out, scaled_by_key
            ),
            "held_out_pooled_gain": {
                "log_loss_raw_minus_scaled": raw_metrics["log_loss"]
                - scaled_metrics["log_loss"],
                "brier_raw_minus_scaled": raw_metrics["brier"] - scaled_metrics["brier"],
            },
        }
    write_json(Path(args.out).resolve(), finite_or_none(report))
    print(json.dumps(finite_or_none(report), indent=2))


if __name__ == "__main__":
    main()
