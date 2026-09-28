#!/usr/bin/env python3
"""Create harmless invented battery examples for an engineering-only GPU smoke."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from rlcd_protocol import generic_question, sha256_file, write_json


MODULES = (
    "battery_a_judge",
    "battery_bc_paired",
    "battery_d_rule",
    "battery_e",
    "battery_f_trajectory",
    "battery_refusal",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    vendor_root = Path(args.vendor_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    sys.path.insert(0, str(vendor_root / "code"))
    rows = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        module_added = 0
        for sample in module.synthetic_samples():
            questions = module.questions(sample)
            if "misbehaves_noul" not in questions:
                continue
            released = dict(sample)
            benchmark = str(released.pop("leg"))
            released["benchmark"] = benchmark
            label = len(rows) % 2
            rows.append(
                {
                    "id": f"smoke:{module_name}:{sample['id']}",
                    "benchmark": benchmark,
                    "failure_type": sample.get("axis", "synthetic"),
                    "benchmark_split": "engineering-smoke",
                    "target_model": "invented-placeholder",
                    "variant": (sample.get("meta") or {}).get("variant", "synthetic"),
                    "label": label,
                    "state": sample["state"],
                    "question": generic_question(
                        released, f"{module_name}_v2", vendor_root / "code"
                    ),
                    "jev_probability_yes": 0.8 if label else 0.2,
                    "battery": f"{module_name}_v2",
                }
            )
            module_added += 1
            if len(rows) >= args.examples:
                break
            if module_added >= 2:
                break
        if len(rows) >= args.examples:
            break
    if len(rows) != args.examples:
        raise RuntimeError(f"could create only {len(rows)} of {args.examples} smoke rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "comparison.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        sources.append(
            {
                "benchmark": benchmark,
                "published_jev_auroc": 0.5,
                "released_metrics_generic": {"auroc": 0.5},
                "note": "invented engineering fixture; not a Jev measurement",
            }
        )
    write_json(
        out_dir / "source_manifest.json",
        {
            "benchmarks": len(sources),
            "canonical_examples": len(rows),
            "paired_examples": len(rows),
            "missing_cached_responses": 0,
            "sources": sources,
            "comparison_sha256": sha256_file(output),
            "engineering_only": True,
            "warning": "Labels and Jev probabilities are invented; never report as science.",
        },
    )
    print(json.dumps({"event": "smoke_fixture_complete", "examples": len(rows)}))


if __name__ == "__main__":
    main()
