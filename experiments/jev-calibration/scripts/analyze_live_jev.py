#!/usr/bin/env python3
"""Analyze state-free live Jev replay outputs against cached Jev and Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import summarize
from rlcd_protocol import bootstrap_median_ci, read_jsonl, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
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
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    ld = sum((a - lm) ** 2 for a in left)
    rd = sum((b - rm) ** 2 for b in right)
    if ld == 0 or rd == 0:
        return None
    return numerator / math.sqrt(ld * rd)


def drift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row.get("cached_jev_probability_yes") is not None]
    live = [float(row["live_jev_probability_yes"]) for row in paired]
    cached = [float(row["cached_jev_probability_yes"]) for row in paired]
    absolute = [abs(a - b) for a, b in zip(live, cached)]
    signed = [a - b for a, b in zip(live, cached)]
    return {
        "paired_examples": len(paired),
        "missing_cached_examples": len(rows) - len(paired),
        "mean_absolute_probability_change": statistics.mean(absolute) if absolute else None,
        "median_absolute_probability_change": statistics.median(absolute) if absolute else None,
        "max_absolute_probability_change": max(absolute) if absolute else None,
        "mean_signed_probability_change_live_minus_cached": statistics.mean(signed) if signed else None,
        "pearson_probability_correlation": pearson(live, cached),
        "threshold_decision_disagreement_rate": (
            sum((a >= 0.5) != (b >= 0.5) for a, b in zip(live, cached)) / len(paired)
            if paired
            else None
        ),
        "live_metrics_on_paired": binary_metrics(paired, "live_jev_probability_yes"),
        "cached_metrics_on_paired": binary_metrics(paired, "cached_jev_probability_yes"),
    }


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for item in value.values():
            result.update(nested_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(nested_keys(item))
        return result
    return set()


def canonical_response_sha256(response: dict[str, Any]) -> str:
    payload = json.dumps(
        response, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_raw_responses(
    raw_path: Path, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_rows = read_jsonl(raw_path)
    prediction_by_key = {
        f"{row['benchmark']}:{row['id']}": row for row in predictions
    }
    raw_by_key = {f"{row['benchmark']}:{row['id']}": row for row in raw_rows}
    if len(raw_by_key) != len(raw_rows):
        raise RuntimeError("duplicate saved raw response key")
    if set(raw_by_key) != set(prediction_by_key):
        raise RuntimeError("saved raw response keys do not match successful predictions")
    forbidden = {"state", "question", "questions"}
    allowed_response_keys = {"model", "answers", "usage", "meta"}
    for item_key, envelope in raw_by_key.items():
        response = envelope.get("response")
        if not isinstance(response, dict):
            raise RuntimeError(f"raw response is not an object: {item_key}")
        if set(response) - allowed_response_keys:
            raise RuntimeError(f"unexpected raw response keys: {item_key}")
        if forbidden & nested_keys(envelope):
            raise RuntimeError(f"gated text key leaked into raw response: {item_key}")
        digest = canonical_response_sha256(response)
        if digest != envelope.get("response_sha256"):
            raise RuntimeError(f"raw response hash mismatch: {item_key}")
        prediction = prediction_by_key[item_key]
        if digest != prediction.get("raw_response_sha256"):
            raise RuntimeError(f"prediction/raw response hash mismatch: {item_key}")
        answer = response["answers"]["misbehaves_noul"]["noul"]
        if float(answer) != float(prediction["live_jev_probability_yes"]):
            raise RuntimeError(f"raw response probability mismatch: {item_key}")
        usage = response["usage"]
        if int(usage["input_tokens"]) != int(prediction["input_tokens"]):
            raise RuntimeError(f"raw response input-token mismatch: {item_key}")
        if int(usage.get("output_tokens", 0)) != int(prediction["output_tokens"]):
            raise RuntimeError(f"raw response output-token mismatch: {item_key}")
    return {
        "rows": len(raw_rows),
        "sha256": sha256_file(raw_path),
        "response_hashes_verified": True,
        "prediction_links_verified": True,
        "gated_text_keys_absent": True,
        "response_schema_keys": sorted(allowed_response_keys),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    predictions_path = out_dir / "predictions.jsonl"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    raw = read_jsonl(predictions_path)
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in raw:
        if row.get("error") is None:
            rows_by_key[f"{row['benchmark']}:{row['id']}"] = row
    rows = sorted(rows_by_key.values(), key=lambda row: (row["benchmark"], row["id"]))
    if len(rows) != int(manifest["selected_examples"]):
        raise RuntimeError("successful unique prediction count does not match manifest selection")
    forbidden = {"state", "question"}
    if any(forbidden & set(row) for row in rows):
        raise RuntimeError("gated text field leaked into live prediction output")
    if any(not 0 <= float(row["live_jev_probability_yes"]) <= 1 for row in rows):
        raise RuntimeError("invalid live Jev probability")
    raw_response_audit = audit_raw_responses(out_dir / "raw_responses.jsonl", rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    per_benchmark = []
    for benchmark, items in sorted(grouped.items()):
        live = binary_metrics(items, "live_jev_probability_yes")
        qwen = binary_metrics(items, "qwen_probability_yes")
        cached_items = [item for item in items if item.get("cached_jev_probability_yes") is not None]
        cached = binary_metrics(cached_items, "cached_jev_probability_yes")
        live_on_cached = binary_metrics(cached_items, "live_jev_probability_yes")
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "n": len(items),
                "benchmark_split": items[0].get("benchmark_split"),
                "failure_type": items[0].get("failure_type"),
                "live_jev": live,
                "qwen": qwen,
                "cached_jev_on_available": cached,
                "live_jev_on_cached_available": live_on_cached,
                "live_minus_qwen_auroc": (
                    live["auroc"] - qwen["auroc"] if live and qwen else None
                ),
            }
        )

    comparable = [item for item in per_benchmark if item["live_minus_qwen_auroc"] is not None]
    deltas = [float(item["live_minus_qwen_auroc"]) for item in comparable]
    report = {
        "mode": manifest["mode"],
        "examples": len(rows),
        "benchmarks": len(grouped),
        "model_requested": manifest["selected_model"],
        "models_reported": status.get("response_models"),
        "usage": {
            key: status.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "wall_seconds",
                "latency_p50_seconds",
                "latency_p95_seconds",
            )
        },
        "pooled_warning": "Pooled metrics mix domains and base rates; full-run per-benchmark medians are primary.",
        "pooled": {
            "live_jev": binary_metrics(rows, "live_jev_probability_yes"),
            "qwen": binary_metrics(rows, "qwen_probability_yes"),
        },
        "live_vs_cached_drift": drift(rows),
        "benchmark_aggregate": {
            "comparable_benchmarks": len(comparable),
            "median_live_minus_qwen_auroc": statistics.median(deltas) if deltas else None,
            "bootstrap_ci95_over_benchmarks": (
                bootstrap_median_ci(deltas, args.seed) if len(deltas) >= 2 else None
            ),
            "live_auroc_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "qwen_auroc_wins": sum(delta < 0 for delta in deltas),
        },
        "per_benchmark": per_benchmark,
        "integrity": {
            "predictions_sha256": sha256_file(predictions_path),
            "selection_sha256": manifest["selection_sha256"],
            "forbidden_text_fields_absent": True,
            "failed_attempt_records": status.get("failed_attempt_records"),
            "raw_responses": raw_response_audit,
        },
    }
    write_json(out_dir / "metrics.json", finite_or_none(report))
    print(json.dumps(finite_or_none(report), indent=2))


if __name__ == "__main__":
    main()
