#!/usr/bin/env python3
"""Validate a completed RLCDAlignBench comparison without exposing gated text."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rlcd_protocol import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    predictions_path = out_dir / "predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    source = manifest["prepared_dataset_manifest"]
    keys = [f"{row['benchmark']}:{row['id']}" for row in predictions]
    qwen = [float(row["qwen_probability_yes"]) for row in predictions]
    errors = []
    expected = int(manifest["selected_examples"])
    if len(predictions) != expected:
        errors.append(f"prediction count {len(predictions)} != expected {expected}")
    if len(set(keys)) != len(keys):
        errors.append("duplicate benchmark:id prediction key")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in qwen):
        errors.append("invalid Qwen probability")
    if any(row.get("label") not in (0, 1) for row in predictions):
        errors.append("non-binary prediction label")
    if any("state" in row or "question" in row for row in predictions):
        errors.append("gated state/question leaked into predictions")
    if status.get("state") != "complete":
        errors.append(f"status is not complete: {status.get('state')}")

    incomplete = [
        {
            "benchmark": item["benchmark"],
            "canonical_samples": item["canonical_samples"],
            "paired_samples": item["paired_samples"],
            "missing_cached_responses": item["missing_cached_responses"],
        }
        for item in source["sources"]
        if item["missing_cached_responses"]
    ]
    inconsistent = [
        {
            "benchmark": item["benchmark"],
            "raw_cached_auroc": item["jev_cached_subset"]["auroc"],
            "released_auroc": item["published_jev_auroc"],
        }
        for item in metrics["per_benchmark"]
        if item["cache_complete"]
        and item["cached_auroc_reproduces_published"] is False
    ]
    result = {
        "ok": not errors,
        "errors": errors,
        "predictions": len(predictions),
        "unique_keys": len(set(keys)),
        "benchmarks": len({row["benchmark"] for row in predictions}),
        "null_jev_probabilities": sum(
            row.get("jev_probability_yes") is None for row in predictions
        ),
        "qwen_probability_min": min(qwen),
        "qwen_probability_max": max(qwen),
        "max_prompt_tokens": max(int(row["prompt_tokens"]) for row in predictions),
        "registered_max_input_tokens": manifest["config"]["max_input_tokens"],
        "prediction_sha256": sha256_file(predictions_path),
        "status": status,
        "source_counts": {
            "canonical_examples": source["canonical_examples"],
            "paired_examples": source["paired_examples"],
            "missing_cached_responses": source["missing_cached_responses"],
        },
        "incomplete_caches": incomplete,
        "complete_cache_metric_disagreements": inconsistent,
        "primary": metrics["aggregates"]["all"]["primary_complete_raw_cache"],
        "secondary": metrics["aggregates"]["all"][
            "secondary_released_aggregate_auroc"
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
