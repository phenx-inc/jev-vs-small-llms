#!/usr/bin/env python3
"""Freeze train/test artifacts for RLCDAlignBench distillation without leaking test outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlcd_distillation import (
    assign_benchmark_folds,
    atomic_write_jsonl,
    keys_sha256,
    load_config,
    read_and_verify,
    sha256_file,
    validate_and_partition,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def counts_by_benchmark(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["benchmark"]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    comparison_path = Path(args.comparison).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    dataset = config["dataset"]
    rows = read_and_verify(comparison_path, dataset["comparison_sha256"])
    train, heldout, audit = validate_and_partition(
        rows, expected_source_rows=int(dataset["expected_source_rows"])
    )
    expected = {
        "train_rows": int(dataset["expected_train_rows"]),
        "heldout_rows": int(dataset["expected_heldout_rows"]),
        "heldout_teacher_rows": int(dataset["expected_heldout_teacher_rows"]),
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"{key}: expected {value}, found {audit[key]}")
    if audit["overlap_keys"]:
        raise RuntimeError("training and held-out keys overlap")

    assignments, fold_totals = assign_benchmark_folds(
        train, int(config["cv"]["folds"]), int(config["seed"])
    )
    for row in train:
        row["cv_fold"] = assignments[str(row["benchmark"])]
    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"
    atomic_write_jsonl(train_path, train)
    atomic_write_jsonl(heldout_path, heldout)

    folds: dict[int, list[str]] = defaultdict(list)
    for benchmark, fold in assignments.items():
        folds[fold].append(benchmark)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_comparison_path": str(comparison_path),
        "source_comparison_sha256": sha256_file(comparison_path),
        "audit": audit,
        "train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "keys_sha256": keys_sha256(train),
            "rows": len(train),
            "by_benchmark": counts_by_benchmark(train),
        },
        "heldout": {
            "path": str(heldout_path),
            "sha256": sha256_file(heldout_path),
            "keys_sha256": keys_sha256(heldout),
            "rows": len(heldout),
            "by_benchmark": counts_by_benchmark(heldout),
        },
        "cv": {
            "folds": int(config["cv"]["folds"]),
            "fold_row_totals": fold_totals,
            "benchmarks_by_fold": {
                str(fold): sorted(benchmarks) for fold, benchmarks in sorted(folds.items())
            },
            "assignment": dict(sorted(assignments.items())),
        },
        "data_handling": (
            "The gated state text remains on the GPU server. Trainer processes read only train.jsonl; "
            "heldout.jsonl is opened only after all final adapters finish."
        ),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "prepared", "out_dir": str(out_dir), "audit": audit}))


if __name__ == "__main__":
    main()
