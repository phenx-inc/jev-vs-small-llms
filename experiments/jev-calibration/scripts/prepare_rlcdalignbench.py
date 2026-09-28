#!/usr/bin/env python3
"""Join canonical RLCDAlignBench samples to cached generic-Noul Jev probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlcd_protocol import build_comparison_rows, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    release_dir = Path(args.release_dir).resolve()
    vendor_root = Path(args.vendor_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_comparison_rows(
        release_dir=release_dir,
        vendor_root=vendor_root,
    )
    output_path = out_dir / "comparison.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.update(
        {
            "config_sha256": sha256_file(config_path),
            "comparison_path": str(output_path),
            "comparison_sha256": sha256_file(output_path),
            "note": "Prepared data contain gated harmful model outputs and remain on the GPU server.",
        }
    )
    write_json(out_dir / "source_manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "prepare_complete",
                "benchmarks": manifest["benchmarks"],
                "canonical_examples": manifest["canonical_examples"],
                "paired_examples": manifest["paired_examples"],
                "missing_cached_responses": manifest["missing_cached_responses"],
                "out_dir": str(out_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
