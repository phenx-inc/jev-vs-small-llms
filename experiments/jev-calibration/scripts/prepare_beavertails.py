#!/usr/bin/env python3
"""Deduplicate, sample, and prepare BeaverTails for identical Jev/Qwen scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_c_safeqa import QUESTION, render_state
from rlcd_protocol import sha256_file, write_json
from sample_c_safeqa import allocate_hamilton, canonical_json, rank_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def category_signature(row: dict[str, Any]) -> str:
    category = row.get("category")
    if not isinstance(category, dict):
        raise ValueError("category must be an object")
    active = sorted(str(key) for key, value in category.items() if value is True)
    return "|".join(active) if active else "__none__"


def read_source(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                raise ValueError(f"{path}:{line_number}: prompt/response must be strings")
            if not isinstance(row.get("is_safe"), bool):
                raise ValueError(f"{path}:{line_number}: is_safe must be boolean")
            category_signature(row)
            rows.append((line_number, row))
    return rows


def deduplicate(
    rows: list[tuple[int, dict[str, Any]]]
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    kept: list[tuple[int, dict[str, Any]]] = []
    seen: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []
    for line_number, row in rows:
        identity = hashlib.sha256(
            canonical_json([row["prompt"], row["response"]]).encode("utf-8")
        ).hexdigest()
        if identity in seen:
            duplicates.append(
                {
                    "identity_sha256": identity,
                    "kept_source_line": seen[identity],
                    "excluded_source_line": line_number,
                }
            )
            continue
        seen[identity] = line_number
        kept.append((line_number, row))
    return kept, duplicates


def select_rows(
    rows: list[tuple[int, dict[str, Any]]], seed: int, target: int, source_path: str
) -> tuple[list[tuple[int, str, str, dict[str, Any]]], list[dict[str, Any]]]:
    strata: Counter[tuple[str, str, str]] = Counter()
    for _, row in rows:
        stratum = (
            "unsafe" if not row["is_safe"] else "safe",
            category_signature(row),
            "__all__",
        )
        strata[stratum] += 1
    quotas = allocate_hamilton(strata, target)
    heaps: dict[
        tuple[str, str, str], list[tuple[int, int, str, str, dict[str, Any]]]
    ] = defaultdict(list)
    for line_number, row in rows:
        stratum = (
            "unsafe" if not row["is_safe"] else "safe",
            category_signature(row),
            "__all__",
        )
        quota = quotas[stratum]
        if quota == 0:
            continue
        rank_int, rank_sha256, record_sha256 = rank_record(
            seed, source_path, line_number, row
        )
        entry = (-rank_int, line_number, rank_sha256, record_sha256, row)
        heap = heaps[stratum]
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    selected: list[tuple[int, str, str, dict[str, Any]]] = []
    for heap in heaps.values():
        for _, line_number, rank_sha256, record_sha256, row in heap:
            selected.append((line_number, rank_sha256, record_sha256, row))
    selected.sort(key=lambda item: item[1])
    if len(selected) != target:
        raise AssertionError(f"expected {target} selected rows, found {len(selected)}")
    allocation = [
        {
            "unsafe_label": stratum[0],
            "harm_category_signature": stratum[1],
            "population": strata[stratum],
            "selected": quotas[stratum],
        }
        for stratum in sorted(strata)
    ]
    return selected, allocation


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    sampling = config["sampling"]
    download_dir = Path(args.download_dir).resolve()
    source_path = download_dir / "test.jsonl.gz"
    if source_path.stat().st_size != dataset["expected_bytes"]:
        raise RuntimeError("source byte count mismatch")
    if sha256_file(source_path) != dataset["expected_sha256"]:
        raise RuntimeError("source SHA-256 mismatch")
    source_rows = read_source(source_path)
    if len(source_rows) != dataset["expected_rows"]:
        raise RuntimeError("source row count mismatch")
    unique_rows, duplicates = deduplicate(source_rows)
    if len(unique_rows) != dataset["expected_unique_prompt_response_pairs"]:
        raise RuntimeError("deduplicated row count mismatch")
    selected, allocation = select_rows(
        unique_rows,
        int(sampling["seed"]),
        int(sampling["target_rows"]),
        dataset["path"],
    )

    comparison: list[dict[str, Any]] = []
    for line_number, rank_sha256, record_sha256, row in selected:
        signature = category_signature(row)
        prompt_group = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        comparison.append(
            {
                "id": f"beavertails-{rank_sha256}",
                "benchmark": "beavertails_30k_test",
                "failure_type": signature,
                "benchmark_split": "30k_test",
                "target_model": "__not_released__",
                "variant": signature,
                "label": int(not row["is_safe"]),
                "label_text": "Unsafe" if not row["is_safe"] else "Safe",
                "battery": "beavertails_response_safety_v1",
                "state": render_state(row["prompt"], row["response"]),
                "question": dict(QUESTION),
                "jev_probability_yes": None,
                "prompt_group_sha256": prompt_group,
                "sample_rank_sha256": rank_sha256,
                "record_sha256": record_sha256,
                "source_line": line_number,
            }
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "comparison.jsonl"
    with comparison_path.open("w", encoding="utf-8") as handle:
        for row in comparison:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    state_lengths = [len(row["state"]) for row in comparison]
    label_counts = Counter(row["label_text"] for row in comparison)
    manifest = {
        "protocol": "beavertails_response_safety_v1",
        "dataset": dataset,
        "sampling": sampling,
        "config_sha256": sha256_file(config_path),
        "source_sha256": sha256_file(source_path),
        "source_rows": len(source_rows),
        "unique_prompt_response_pairs": len(unique_rows),
        "duplicates_excluded": duplicates,
        "comparison_sha256": sha256_file(comparison_path),
        "rows": len(comparison),
        "label_counts": dict(sorted(label_counts.items())),
        "stratum_allocation": allocation,
        "question": QUESTION,
        "question_frozen_before_scoring": True,
        "jev_max_state_chars": 4000,
        "jev_eligible_rows": sum(length <= 4000 for length in state_lengths),
        "jev_oversize_rows": sum(length > 4000 for length in state_lengths),
        "state_chars": {
            "min": min(state_lengths),
            "median": statistics.median(state_lengths),
            "p95": percentile(state_lengths, 0.95),
            "max": max(state_lengths),
            "total": sum(state_lengths),
        },
        "unique_exact_prompt_groups": len({row["prompt_group_sha256"] for row in comparison}),
        "data_handling": (
            "Raw prompt/response text remains under protected /data; result artifacts "
            "contain labels, identifiers, hashes, and probabilities only."
        ),
    }
    write_json(out_dir / "source_manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "beavertails_prepared",
                "rows": len(comparison),
                "label_counts": manifest["label_counts"],
                "jev_eligible_rows": manifest["jev_eligible_rows"],
                "comparison_sha256": manifest["comparison_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
