#!/usr/bin/env python3
"""Audit released generic-Noul metrics against the prepared raw cache rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from metrics import summarize
from rlcd_protocol import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    rows = read_jsonl(data_dir / "comparison.jsonl")
    manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    sources = {source["benchmark"]: source for source in manifest["sources"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)

    results = []
    for benchmark, items in sorted(grouped.items()):
        cached = [row for row in items if row.get("jev_probability_yes") is not None]
        metrics = summarize(
            [row["label"] for row in cached],
            [row["jev_probability_yes"] for row in cached],
        )
        source = sources[benchmark]
        published = float(source["published_jev_auroc"])
        results.append(
            {
                "benchmark": benchmark,
                "canonical_n": len(items),
                "cached_n": len(cached),
                "missing": len(items) - len(cached),
                "response_only_extra": source["response_rows_outside_canonical_denominator"],
                "published_auroc": published,
                "raw_cached_auroc": metrics["auroc"],
                "raw_minus_published": metrics["auroc"] - published,
                "raw_accuracy_at_0_5": metrics["accuracy"],
                "released_accuracy_at_0_5": source["released_metrics_generic"].get("accuracy"),
            }
        )

    mismatches = [
        row for row in results if abs(row["raw_minus_published"]) > 0.000051
    ]
    print(
        json.dumps(
            {
                "benchmarks": len(results),
                "raw_cache_mismatches": len(mismatches),
                "mismatches": mismatches,
                "all": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
