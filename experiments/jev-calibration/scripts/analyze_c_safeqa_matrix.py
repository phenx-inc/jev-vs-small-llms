#!/usr/bin/env python3
"""Analyze Jev, vanilla models, and frozen adapters on C-SafeQA."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def subset_metrics(
    rows: list[dict[str, Any]], probabilities: dict[str, list[float]]
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["benchmark_split"])].append(index)
    return {
        "pooled": {
            name: system_metrics(labels, values) for name, values in probabilities.items()
        },
        "per_source_split": {
            split: {
                "n": len(indices),
                "systems": {
                    name: system_metrics(
                        [labels[index] for index in indices],
                        [values[index] for index in indices],
                    )
                    for name, values in probabilities.items()
                },
            }
            for split, indices in sorted(grouped.items())
        },
    }


def comparison_pairs(names: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [("jev_live", name) for name in sorted(names)]
    for size in ("4b", "9b"):
        base = f"qwen35_{size}_base"
        if base not in names:
            continue
        adapters = []
        for objective in ("jev", "hard", "hybrid"):
            adapter = f"qwen35_{size}_{objective}"
            if adapter in names:
                adapters.append(adapter)
                pairs.append((adapter, base))
        preferred_contrasts = (
            (f"qwen35_{size}_hard", f"qwen35_{size}_jev"),
            (f"qwen35_{size}_hybrid", f"qwen35_{size}_jev"),
            (f"qwen35_{size}_hard", f"qwen35_{size}_hybrid"),
        )
        for left, right in preferred_contrasts:
            if left in adapters and right in adapters:
                pairs.append((left, right))
    if "nemotron3_nano_4b_base" in names:
        for name in sorted(names):
            if name.startswith("qwen35_") and name.rsplit("_", 1)[-1] in {
                "jev",
                "hard",
                "hybrid",
            }:
                pairs.append((name, "nemotron3_nano_4b_base"))
    for objective in ("jev", "hard", "hybrid"):
        left = f"qwen35_9b_{objective}"
        right = f"qwen35_4b_{objective}"
        if left in names and right in names:
            pairs.append((left, right))
    return list(dict.fromkeys(pairs))


def bootstrap_task(
    task: tuple[
        str,
        list[int],
        list[float],
        list[float],
        list[str],
        int,
        int,
    ]
) -> tuple[str, dict[str, Any]]:
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
            raise RuntimeError(f"prediction keys differ from frozen 2,000 rows: {name}")
    full_keys = sorted(expected)
    full_rows = [comparison_by_key[item] for item in full_keys]
    full_probabilities = {
        name: [float(predictions[item]["model_probability_yes"]) for item in full_keys]
        for name, predictions in local.items()
    }

    jev = latest_successful_jev(read_jsonl(jev_dir / "predictions.jsonl"))
    if not set(jev) <= expected:
        raise RuntimeError("Jev output contains keys outside the frozen comparison")
    common_keys = sorted(set(jev).intersection(*(set(rows) for rows in local.values())))
    common_rows = [comparison_by_key[item] for item in common_keys]
    common_probabilities = {
        name: [float(predictions[item]["model_probability_yes"]) for item in common_keys]
        for name, predictions in local.items()
    }
    common_probabilities["jev_live"] = [
        float(jev[item]["live_jev_probability_yes"]) for item in common_keys
    ]
    labels = [int(row["label"]) for row in common_rows]
    groups = [str(row["prompt_group_sha256"]) for row in common_rows]
    pairs = comparison_pairs(set(local))
    tasks = [
        (
            f"{left}_vs_{right}",
            labels,
            common_probabilities[left],
            common_probabilities[right],
            groups,
            args.seed + offset,
            args.bootstrap_draws,
        )
        for offset, (left, right) in enumerate(pairs)
    ]
    if args.jobs == 1:
        bootstraps = dict(bootstrap_task(task) for task in tasks)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.jobs, len(tasks))
        ) as executor:
            bootstraps = dict(executor.map(bootstrap_task, tasks))
    agreements = {}
    for left, right in pairs:
        left_values = common_probabilities[left]
        right_values = common_probabilities[right]
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
    jev_status = json.loads((jev_dir / "status.json").read_text(encoding="utf-8"))
    report = {
        "protocol": source_manifest["protocol"],
        "registered_rows": len(comparison),
        "local_full_rows": len(full_rows),
        "all_system_common_rows": len(common_rows),
        "jev_excluded_oversize_rows": len(comparison) - len(jev),
        "prompt_clusters_common": len(set(groups)),
        "label_counts_common": {"Safe": labels.count(0), "Unsafe": labels.count(1)},
        "local_full_metrics": subset_metrics(full_rows, full_probabilities),
        "common_metrics": subset_metrics(common_rows, common_probabilities),
        "paired_prompt_cluster_bootstrap": bootstraps,
        "pair_agreement": agreements,
        "unsafe_recall_by_category_common": category_recall(
            common_rows, common_probabilities
        ),
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
            "model_predictions_sha256": {
                name: sha256_file(directory / "predictions.jsonl")
                for name, directory in model_dirs.items()
            },
            "jev_predictions_sha256": sha256_file(jev_dir / "predictions.jsonl"),
            "jev_raw_responses_sha256": jev_status.get("raw_responses_sha256"),
        },
        "decisions": [
            "No training, prompt tuning, calibration, or threshold selection used C-SafeQA outcomes.",
            "All local systems are scored on 2,000 rows; all-system paired comparisons use the 1,931-row lossless Jev intersection.",
            "The 69 Jev-oversize rows are excluded rather than truncated.",
            "Confidence intervals resample normalized source-prompt clusters, not individual transformed responses.",
        ],
        "caveats": [
            "The RLCD overlap check is lexical and does not rule out semantic paraphrase overlap.",
            "C-SafeQA is access-controlled and CC BY-NC 4.0; raw text stays in protected storage.",
            "Model-native chat templates are semantically matched but not byte-identical.",
            "This is one deterministic sample and one inference prompt per system.",
        ],
    }
    write_json(Path(args.out).resolve(), finite_or_none(report))
    print(
        json.dumps(
            {
                "event": "c_safeqa_matrix_complete",
                "systems": len(common_probabilities),
                "full_rows": len(full_rows),
                "common_rows": len(common_rows),
                "out": str(Path(args.out).resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
