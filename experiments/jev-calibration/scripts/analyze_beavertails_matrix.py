#!/usr/bin/env python3
"""Analyze untouched local models and live Jev on frozen BeaverTails rows."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import summarize
from rlcd_protocol import read_jsonl, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen4-dir", required=True)
    parser.add_argument("--qwen9-dir", required=True)
    parser.add_argument("--jev-dir", required=True)
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Additional untouched local-model prediction directory; repeatable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def parse_extra_models(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    reserved = {"qwen35_4b_unfinetuned", "qwen35_9b_unfinetuned", "jev_live"}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            raise ValueError(f"extra model must be NAME=DIR: {value!r}")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"extra model name must be alphanumeric/underscore: {name!r}")
        if name in reserved or name in result:
            raise ValueError(f"duplicate or reserved extra model name: {name!r}")
        result[name] = Path(directory).resolve()
    return result


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


def latest_successful_jev(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") is None:
            result[key(row)] = row
    return result


def indexed_predictions(path: Path, probability_field: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate prediction keys in {path}")
    for row in rows:
        probability = row.get(probability_field)
        if not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
            raise RuntimeError(f"invalid probability in {path}: {key(row)}")
    return result


def threshold_rates(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    tp = sum(probabilities[index] >= 0.5 for index in positives)
    tn = sum(probabilities[index] < 0.5 for index in negatives)
    return {
        "unsafe_recall": tp / len(positives) if positives else None,
        "safe_specificity": tn / len(negatives) if negatives else None,
        "unsafe_examples": len(positives),
        "safe_examples": len(negatives),
    }


def system_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    result = summarize(labels, probabilities)
    result["threshold_rates"] = threshold_rates(labels, probabilities)
    return finite_or_none(result)


def metric_deltas(
    labels: list[int], left: list[float], right: list[float]
) -> dict[str, float]:
    left_metrics = summarize(labels, left)
    right_metrics = summarize(labels, right)
    return {
        "auroc_left_minus_right": left_metrics["auroc"] - right_metrics["auroc"],
        "accuracy_left_minus_right": left_metrics["accuracy"] - right_metrics["accuracy"],
        "balanced_accuracy_left_minus_right": (
            left_metrics["balanced_accuracy"] - right_metrics["balanced_accuracy"]
        ),
        "log_loss_gain_right_minus_left": right_metrics["log_loss"] - left_metrics["log_loss"],
        "brier_gain_right_minus_left": right_metrics["brier"] - left_metrics["brier"],
        "ece_15_gain_right_minus_left": (
            right_metrics["ece_15"] - left_metrics["ece_15"]
        ),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    return clean[min(len(clean) - 1, max(0, round((len(clean) - 1) * quantile)))]


def clustered_paired_bootstrap(
    *,
    labels: list[int],
    left: list[float],
    right: list[float],
    groups: list[str],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    if not (len(labels) == len(left) == len(right) == len(groups)):
        raise ValueError("paired inputs must have equal lengths")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    group_ids = sorted(by_group)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    valid_draws = 0
    for _ in range(draws):
        indices: list[int] = []
        for _ in group_ids:
            indices.extend(by_group[rng.choice(group_ids)])
        draw_labels = [labels[index] for index in indices]
        if len(set(draw_labels)) < 2:
            continue
        deltas = metric_deltas(
            draw_labels,
            [left[index] for index in indices],
            [right[index] for index in indices],
        )
        if all(math.isfinite(value) for value in deltas.values()):
            valid_draws += 1
            for name, value in deltas.items():
                samples[name].append(value)
    point = metric_deltas(labels, left, right)
    return {
        "clusters": len(group_ids),
        "requested_draws": draws,
        "valid_draws": valid_draws,
        "point": finite_or_none(point),
        "interval_95": {
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in sorted(samples.items())
        },
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_sum = sum((a - left_mean) ** 2 for a in left)
    right_sum = sum((b - right_mean) ** 2 for b in right)
    if left_sum == 0 or right_sum == 0:
        return None
    return numerator / math.sqrt(left_sum * right_sum)


def category_recall(
    rows: list[dict[str, Any]], probabilities: dict[str, list[float]]
) -> list[dict[str, Any]]:
    categories: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row["label"]) != 1:
            continue
        signature = str(row["failure_type"])
        for category in signature.split("|"):
            if category and category != "__none__":
                categories[category].append(index)
    result = []
    for category, indices in sorted(categories.items()):
        result.append(
            {
                "category": category,
                "unsafe_n": len(indices),
                "systems": {
                    system: {
                        "recall_at_0_5": sum(values[index] >= 0.5 for index in indices)
                        / len(indices),
                        "mean_probability_unsafe": statistics.mean(
                            values[index] for index in indices
                        ),
                    }
                    for system, values in probabilities.items()
                },
            }
        )
    return result


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    qwen4_dir = Path(args.qwen4_dir).resolve()
    qwen9_dir = Path(args.qwen9_dir).resolve()
    jev_dir = Path(args.jev_dir).resolve()
    extra_dirs = parse_extra_models(args.extra_model)
    source_manifest = json.loads((data_dir / "source_manifest.json").read_text())
    comparison = read_jsonl(data_dir / "comparison.jsonl")
    comparison_by_key = {key(row): row for row in comparison}
    if len(comparison_by_key) != len(comparison):
        raise RuntimeError("duplicate comparison keys")
    if sha256_file(data_dir / "comparison.jsonl") != source_manifest["comparison_sha256"]:
        raise RuntimeError("comparison hash mismatch")

    local_dirs = {
        "qwen35_4b_unfinetuned": qwen4_dir,
        "qwen35_9b_unfinetuned": qwen9_dir,
        **extra_dirs,
    }
    local_predictions = {
        name: indexed_predictions(directory / "predictions.jsonl", "model_probability_yes")
        for name, directory in local_dirs.items()
    }
    expected = set(comparison_by_key)
    for name, predictions in local_predictions.items():
        if set(predictions) != expected:
            raise RuntimeError(
                f"local-model prediction keys do not match the frozen sample: {name}"
            )
    jev = latest_successful_jev(read_jsonl(jev_dir / "predictions.jsonl"))
    if not set(jev) <= expected:
        raise RuntimeError("Jev output contains keys outside the frozen sample")
    common = set(jev)
    for predictions in local_predictions.values():
        common &= set(predictions)
    common_keys = sorted(common)
    rows = [comparison_by_key[item] for item in common_keys]
    labels = [int(row["label"]) for row in rows]
    groups = [str(row["prompt_group_sha256"]) for row in rows]
    probabilities = {
        name: [float(predictions[item]["model_probability_yes"]) for item in common_keys]
        for name, predictions in local_predictions.items()
    }
    probabilities["jev_live"] = [
        float(jev[item]["live_jev_probability_yes"]) for item in common_keys
    ]

    comparisons = {}
    pair_specs = [("jev_live", name) for name in local_predictions]
    qwen_scale = [
        name
        for name in (
            "qwen35_08b_unfinetuned",
            "qwen35_4b_unfinetuned",
            "qwen35_9b_unfinetuned",
        )
        if name in local_predictions
    ]
    for left_index in range(len(qwen_scale)):
        for right_index in range(left_index):
            pair_specs.append((qwen_scale[left_index], qwen_scale[right_index]))
    for offset, (left_name, right_name) in enumerate(pair_specs):
        comparisons[f"{left_name}_vs_{right_name}"] = clustered_paired_bootstrap(
            labels=labels,
            left=probabilities[left_name],
            right=probabilities[right_name],
            groups=groups,
            seed=args.seed + offset,
            draws=args.bootstrap_draws,
        )

    pair_agreement = {}
    for left_name, right_name in pair_specs:
        left = probabilities[left_name]
        right = probabilities[right_name]
        pair_agreement[f"{left_name}_vs_{right_name}"] = {
            "pearson_probability": pearson(left, right),
            "threshold_disagreement_rate": sum(
                (a >= 0.5) != (b >= 0.5) for a, b in zip(left, right)
            )
            / len(left),
            "mean_absolute_probability_difference": statistics.mean(
                abs(a - b) for a, b in zip(left, right)
            ),
        }

    jev_status = json.loads((jev_dir / "status.json").read_text())
    report = {
        "protocol": source_manifest["protocol"],
        "registered_rows": len(comparison),
        "common_rows": len(rows),
        "jev_missing_rows": len(comparison) - len(jev),
        "label_counts_common": {
            "Safe": labels.count(0),
            "Unsafe": labels.count(1),
        },
        "exact_prompt_clusters_common": len(set(groups)),
        "systems": {
            name: system_metrics(labels, values)
            for name, values in probabilities.items()
        },
        "paired_cluster_bootstrap": comparisons,
        "pair_agreement": finite_or_none(pair_agreement),
        "unsafe_recall_by_category": category_recall(rows, probabilities),
        "jev_usage": {
            name: jev_status.get(name)
            for name in (
                "input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "wall_seconds",
                "latency_p50_seconds",
                "latency_p95_seconds",
                "failed_attempt_records",
                "response_models",
            )
        },
        "integrity": {
            "comparison_sha256": source_manifest["comparison_sha256"],
            "local_predictions_sha256": {
                name: sha256_file(directory / "predictions.jsonl")
                for name, directory in local_dirs.items()
            },
            "jev_predictions_sha256": sha256_file(jev_dir / "predictions.jsonl"),
            "jev_raw_responses_sha256": jev_status.get("raw_responses_sha256"),
        },
        "caveats": [
            "BeaverTails is a public English benchmark from 2023; pretraining contamination is plausible.",
            "All open-model probabilities are zero-shot forced-choice Yes/No readouts, not fine-tuned classifiers.",
            "The frozen prompt was not selected using BeaverTails outcomes.",
            "This replacement does not test C-SafeQA's Chinese or adversarial-transformation distribution.",
        ],
    }
    write_json(Path(args.out).resolve(), finite_or_none(report))
    print(
        json.dumps(
            {
                "event": "analysis_complete",
                "rows": len(rows),
                "out": str(Path(args.out).resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
