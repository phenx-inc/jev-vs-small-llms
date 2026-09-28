#!/usr/bin/env python3
"""Aggregate cross-validation and one-shot held-out distillation results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import summarize
from rlcd_distillation import load_config, read_and_verify, teacher_metrics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.median(finite) if finite else None


def official_on_teacher_paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row.get("jev_probability_yes") is not None]
    return summarize(
        [int(row["label"]) for row in paired],
        [float(row["model_probability_yes"]) for row in paired],
    )


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    heldout = read_and_verify(data_dir / "heldout.jsonl", data_manifest["heldout"]["sha256"])
    jev_paired = [row for row in heldout if row.get("jev_probability_yes") is not None]
    jev_reference = summarize(
        [int(row["label"]) for row in jev_paired],
        [float(row["jev_probability_yes"]) for row in jev_paired],
    )

    final_systems: list[dict[str, Any]] = []
    for model_key in config["models"]:
        for system in ("base", *config["arms"]):
            directory = run_dir / "heldout_eval" / model_key / system
            metrics = json.loads((directory / "metrics.json").read_text())
            rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
            per_benchmark = metrics["per_benchmark"]
            final_systems.append(
                {
                    "model_key": model_key,
                    "system": system,
                    "n_official": metrics["pooled"]["n_official"],
                    "n_teacher": metrics["pooled"]["n_teacher"],
                    "pooled": metrics["pooled"],
                    "official_on_teacher_paired": official_on_teacher_paired(rows),
                    "macro_median": {
                        "official_auroc": median(
                            [item["official_labels"].get("auroc") for item in per_benchmark]
                        ),
                        "official_log_loss": median(
                            [item["official_labels"].get("log_loss") for item in per_benchmark]
                        ),
                        "official_brier": median(
                            [item["official_labels"].get("brier") for item in per_benchmark]
                        ),
                        "teacher_kl": median(
                            [item["teacher_fidelity"].get("kl_teacher_to_model") for item in per_benchmark]
                        ),
                        "teacher_mse": median(
                            [item["teacher_fidelity"].get("mean_squared_error") for item in per_benchmark]
                        ),
                    },
                    "metrics_path": str(directory / "metrics.json"),
                }
            )

    cv_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arm in config["arms"]:
        for fold in range(int(config["cv"]["folds"])):
            path = run_dir / "cv" / "qwen35-4b" / arm / f"fold-{fold}" / "cv_metrics.json"
            metrics = json.loads(path.read_text())
            cv_by_arm[arm].append(metrics)
    cv_summary = []
    for arm, folds in cv_by_arm.items():
        cv_summary.append(
            {
                "arm": arm,
                "folds": len(folds),
                "mean_pooled_teacher_kl": statistics.mean(
                    fold["pooled"]["teacher_fidelity"]["kl_teacher_to_model"] for fold in folds
                ),
                "mean_pooled_teacher_mse": statistics.mean(
                    fold["pooled"]["teacher_fidelity"]["mean_squared_error"] for fold in folds
                ),
                "mean_pooled_official_log_loss": statistics.mean(
                    fold["pooled"]["official_labels"]["log_loss"] for fold in folds
                ),
                "mean_pooled_official_brier": statistics.mean(
                    fold["pooled"]["official_labels"]["brier"] for fold in folds
                ),
            }
        )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "counts": {
            "train": data_manifest["train"]["rows"],
            "heldout_official": len(heldout),
            "heldout_teacher_paired": len(jev_paired),
        },
        "jev_reference_on_teacher_paired": jev_reference,
        "cross_validation": cv_summary,
        "final_systems": final_systems,
        "interpretation_guardrails": [
            "Teacher-fidelity metrics measure imitation of Jev, not ground-truth correctness.",
            "Official-label metrics measure benchmark performance; Jev comparisons use the 1,174-row teacher-paired subset.",
            "No final held-out result was used to select hyperparameters or adapters.",
            "BeaverTails is not included in this run and remains an external test.",
        ],
    }
    write_json(run_dir / "results.json", result)
    print(json.dumps({"event": "analysis_complete", "path": str(run_dir / "results.json")}))


if __name__ == "__main__":
    main()
