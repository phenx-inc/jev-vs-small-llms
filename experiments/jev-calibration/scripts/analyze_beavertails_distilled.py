#!/usr/bin/env python3
"""Analyze Jev, untouched models, and frozen RLCD adapters on BeaverTails."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from analyze_beavertails_matrix import (
    category_recall,
    clustered_paired_bootstrap,
    finite_or_none,
    indexed_predictions,
    key,
    latest_successful_jev,
    pearson,
    system_metrics,
)
from analyze_c_safeqa_matrix import comparison_pairs
from rlcd_protocol import read_jsonl, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--jev-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            raise ValueError(f"model must be NAME=DIR: {value!r}")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid model name: {name!r}")
        if name == "jev_live" or name in result:
            raise ValueError(f"duplicate or reserved model name: {name!r}")
        result[name] = Path(directory).resolve()
    return result


def bootstrap_one(task: tuple) -> tuple[str, dict]:
    name, labels, left, right, groups, seed, draws = task
    return name, clustered_paired_bootstrap(
        labels=labels,
        left=left,
        right=right,
        groups=groups,
        seed=seed,
        draws=draws,
    )


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    data_dir = Path(args.data_dir).resolve()
    model_dirs = parse_models(args.model)
    jev_dir = Path(args.jev_dir).resolve()
    comparison_path = data_dir / "comparison.jsonl"
    source_manifest_path = data_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(comparison_path) != source_manifest["comparison_sha256"]:
        raise RuntimeError("comparison hash mismatch")

    comparison = read_jsonl(comparison_path)
    comparison_by_key = {key(row): row for row in comparison}
    if len(comparison_by_key) != len(comparison):
        raise RuntimeError("duplicate comparison keys")
    expected = set(comparison_by_key)
    local = {
        name: indexed_predictions(directory / "predictions.jsonl", "model_probability_yes")
        for name, directory in model_dirs.items()
    }
    for name, predictions in local.items():
        if set(predictions) != expected:
            raise RuntimeError(f"prediction keys differ from frozen sample: {name}")
    jev = latest_successful_jev(read_jsonl(jev_dir / "predictions.jsonl"))
    if set(jev) != expected:
        raise RuntimeError("Jev predictions do not cover the exact frozen sample")

    keys = sorted(expected)
    rows = [comparison_by_key[item] for item in keys]
    labels = [int(row["label"]) for row in rows]
    groups = [str(row["prompt_group_sha256"]) for row in rows]
    probabilities = {
        name: [float(predictions[item]["model_probability_yes"]) for item in keys]
        for name, predictions in local.items()
    }
    probabilities["jev_live"] = [
        float(jev[item]["live_jev_probability_yes"]) for item in keys
    ]

    pairs = comparison_pairs(set(local))
    tasks = [
        (
            f"{left}_vs_{right}",
            labels,
            probabilities[left],
            probabilities[right],
            groups,
            args.seed + offset,
            args.bootstrap_draws,
        )
        for offset, (left, right) in enumerate(pairs)
    ]
    if args.jobs == 1:
        bootstraps = dict(bootstrap_one(task) for task in tasks)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.jobs, len(tasks))
        ) as executor:
            bootstraps = dict(executor.map(bootstrap_one, tasks))

    agreements = {}
    for left, right in pairs:
        left_values = probabilities[left]
        right_values = probabilities[right]
        agreements[f"{left}_vs_{right}"] = {
            "pearson_probability": pearson(left_values, right_values),
            "threshold_disagreement_rate": sum(
                (a >= 0.5) != (b >= 0.5)
                for a, b in zip(left_values, right_values)
            )
            / len(left_values),
            "mean_absolute_probability_difference": sum(
                abs(a - b) for a, b in zip(left_values, right_values)
            )
            / len(left_values),
        }

    report = {
        "protocol": source_manifest["protocol"],
        "registered_rows": len(rows),
        "exact_prompt_clusters": len(set(groups)),
        "label_counts": {"Safe": labels.count(0), "Unsafe": labels.count(1)},
        "systems": {
            name: system_metrics(labels, values)
            for name, values in probabilities.items()
        },
        "paired_prompt_cluster_bootstrap": bootstraps,
        "pair_agreement": finite_or_none(agreements),
        "unsafe_recall_by_category": category_recall(rows, probabilities),
        "integrity": {
            "comparison_sha256": source_manifest["comparison_sha256"],
            "model_predictions_sha256": {
                name: sha256_file(directory / "predictions.jsonl")
                for name, directory in model_dirs.items()
            },
            "jev_predictions_sha256": sha256_file(jev_dir / "predictions.jsonl"),
        },
        "decisions": [
            "The 2,000-row sample, semantic prompt, model revisions, adapters, and 0.5 threshold were frozen before this evaluation.",
            "No BeaverTails result was used for training, prompt selection, calibration, threshold selection, or model selection.",
            "Existing Jev and untouched-model predictions were reused; no new Jev API calls were made.",
            "Confidence intervals resample exact-prompt clusters rather than individual prompt-response rows.",
        ],
        "caveats": [
            "BeaverTails is public English data from 2023, so pretraining or post-training contamination is plausible.",
            "The adapters were trained on RLCDAlignBench and use one deterministic optimization seed.",
            "Model-native chat templates are semantically matched but not byte-identical.",
            "This evaluates generic binary response-safety detection, not Jev's full typed decision interface.",
        ],
    }
    output = Path(args.out).resolve()
    write_json(output, finite_or_none(report))
    print(
        json.dumps(
            {
                "event": "beavertails_distilled_complete",
                "systems": len(probabilities),
                "rows": len(rows),
                "pairs": len(pairs),
                "out": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
