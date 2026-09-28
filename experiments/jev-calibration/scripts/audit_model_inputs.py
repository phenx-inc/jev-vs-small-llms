#!/usr/bin/env python3
"""Audit frozen comparison inputs against pinned model tokenizers without scoring."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from rlcd_protocol import read_jsonl, sha256_file, write_json
from run_model_baseline import prompt_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_path = data_dir / "comparison.jsonl"
    source_manifest_path = data_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(data_path) != source_manifest["comparison_sha256"]:
        raise RuntimeError("prepared comparison hash mismatch")
    rows = read_jsonl(data_path)
    results: list[dict[str, Any]] = []
    for config_arg in args.config:
        config_path = Path(config_arg).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            cache_dir=config.get("cache_dir"),
            local_files_only=True,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        lengths = [len(prompt_ids(tokenizer, config, row)) for row in rows]
        cap = int(config["max_input_tokens"])
        results.append(
            {
                "run_family": config["run_family"],
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "rows": len(rows),
                "max_input_tokens": cap,
                "overflow_rows": sum(length > cap for length in lengths),
                "prompt_tokens": {
                    "min": min(lengths),
                    "median": statistics.median(lengths),
                    "p95": percentile(lengths, 0.95),
                    "max": max(lengths),
                },
            }
        )
    report = {
        "protocol": "frozen_model_input_token_audit_v1",
        "comparison_sha256": source_manifest["comparison_sha256"],
        "rows": len(rows),
        "models": results,
        "pass": all(item["overflow_rows"] == 0 for item in results),
    }
    write_json(Path(args.out).resolve(), report)
    print(json.dumps({"event": "token_audit_complete", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
