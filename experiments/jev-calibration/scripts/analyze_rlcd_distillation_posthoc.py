#!/usr/bin/env python3
"""Post-hoc, test-only uncertainty and teacher-correlation analysis for distillation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from metrics import summarize
from rlcd_distillation import teacher_metrics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((a - left_mean) ** 2 for a in left)
    right_scale = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_benchmark_bootstrap(
    deltas: dict[str, float], *, draws: int, seed: int
) -> dict[str, Any]:
    names = sorted(deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [deltas[rng.choice(names)] for _ in names]
        samples.append(statistics.median(selected))
    samples.sort()
    observed = statistics.median(deltas.values())
    return {
        "benchmarks": len(names),
        "draws": draws,
        "observed_median_delta": observed,
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "bootstrap_probability_positive": sum(value > 0 for value in samples) / draws,
        "wins_ties_losses": {
            "wins": sum(value > 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
        },
        "per_benchmark_delta": dict(sorted(deltas.items())),
    }


def by_benchmark(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["benchmark"])].append(row)
    return grouped


def official_metrics(rows: list[dict[str, Any]], probability: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    return summarize([int(row["label"]) for row in rows], [probability(row) for row in rows])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    system_rows: dict[str, list[dict[str, Any]]] = {}
    for model in ("qwen35-4b", "qwen35-9b"):
        for system in ("base", "jev", "hard", "hybrid"):
            name = f"{model}/{system}"
            path = run_dir / "heldout_eval" / model / system / "predictions.jsonl"
            system_rows[name] = read_jsonl(path)

    canonical_name = "qwen35-4b/base"
    canonical = system_rows[canonical_name]
    canonical_keys = [key(row) for row in canonical]
    if len(canonical_keys) != len(set(canonical_keys)):
        raise RuntimeError("duplicate canonical prediction keys")
    canonical_by_key = {key(row): row for row in canonical}
    for name, rows in system_rows.items():
        indexed = {key(row): row for row in rows}
        if set(indexed) != set(canonical_by_key):
            raise RuntimeError(f"prediction key mismatch for {name}")
        for item_key, row in indexed.items():
            reference = canonical_by_key[item_key]
            if row["label"] != reference["label"] or row.get("jev_probability_yes") != reference.get(
                "jev_probability_yes"
            ):
                raise RuntimeError(f"label/teacher mismatch for {name}:{item_key}")
        system_rows[name] = [indexed[item_key] for item_key in canonical_keys]

    paired_keys = {
        item_key for item_key, row in canonical_by_key.items() if row.get("jev_probability_yes") is not None
    }
    paired_reference = [canonical_by_key[item_key] for item_key in canonical_keys if item_key in paired_keys]
    jev_grouped = by_benchmark(paired_reference)
    jev_per_benchmark = {
        benchmark: official_metrics(items, lambda row: float(row["jev_probability_yes"]))
        for benchmark, items in jev_grouped.items()
    }
    jev_macro = {
        metric: statistics.median(values[metric] for values in jev_per_benchmark.values())
        for metric in ("auroc", "log_loss", "brier", "ece_15")
    }

    systems: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for system_index, (name, rows) in enumerate(system_rows.items()):
        paired = [row for row in rows if key(row) in paired_keys]
        model_probabilities = [float(row["model_probability_yes"]) for row in paired]
        teacher_probabilities = [float(row["jev_probability_yes"]) for row in paired]
        fidelity = teacher_metrics(model_probabilities, teacher_probabilities)
        fidelity.update(
            {
                "pearson": pearson(model_probabilities, teacher_probabilities),
                "spearman": spearman(model_probabilities, teacher_probabilities),
                "threshold_agreement": mean(
                    [
                        float((model >= 0.5) == (teacher >= 0.5))
                        for model, teacher in zip(model_probabilities, teacher_probabilities)
                    ]
                ),
            }
        )
        grouped = by_benchmark(paired)
        per_benchmark = {
            benchmark: official_metrics(items, lambda row: float(row["model_probability_yes"]))
            for benchmark, items in grouped.items()
        }
        macro = {
            metric: statistics.median(values[metric] for values in per_benchmark.values())
            for metric in ("auroc", "log_loss", "brier", "ece_15")
        }
        systems.append(
            {
                "name": name,
                "n": len(paired),
                "official_pooled": official_metrics(
                    paired, lambda row: float(row["model_probability_yes"])
                ),
                "official_macro_median": macro,
                "teacher_fidelity": fidelity,
            }
        )
        metric_specs = {
            "auroc": lambda student, teacher: student - teacher,
            "log_loss_improvement": lambda student, teacher: teacher - student,
            "brier_improvement": lambda student, teacher: teacher - student,
        }
        metric_keys = {"auroc": "auroc", "log_loss_improvement": "log_loss", "brier_improvement": "brier"}
        for metric_name, delta_function in metric_specs.items():
            metric_key = metric_keys[metric_name]
            deltas = {
                benchmark: delta_function(values[metric_key], jev_per_benchmark[benchmark][metric_key])
                for benchmark, values in per_benchmark.items()
            }
            comparisons.append(
                {
                    "system": name,
                    "comparator": "jev_reference",
                    "metric": metric_name,
                    **paired_benchmark_bootstrap(
                        deltas,
                        draws=args.draws,
                        seed=args.seed + system_index * 101 + len(comparisons),
                    ),
                }
            )

    result = {
        "analysis_type": "post-hoc analysis of frozen one-shot predictions; no model selection or retraining",
        "seed": args.seed,
        "draws": args.draws,
        "n_teacher_paired": len(paired_reference),
        "benchmarks": sorted(jev_grouped),
        "jev_reference": {
            "official_pooled": official_metrics(
                paired_reference, lambda row: float(row["jev_probability_yes"])
            ),
            "official_macro_median": jev_macro,
        },
        "systems": systems,
        "paired_benchmark_bootstrap_vs_jev": comparisons,
        "caveats": [
            "Only six held-out benchmark groups are available, so cluster-bootstrap intervals are coarse and descriptive.",
            "The bootstrap resamples benchmark-level metric deltas; it does not treat repeated rows within a public benchmark as independent domains.",
            "This analysis was run after the one-shot held-out predictions were frozen and did not affect training or model selection.",
        ],
    }
    write_json(run_dir / "posthoc.json", result)
    print(json.dumps({"event": "posthoc_complete", "path": str(run_dir / "posthoc.json")}))


if __name__ == "__main__":
    main()
