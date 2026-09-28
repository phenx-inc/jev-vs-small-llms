#!/usr/bin/env python3
"""Download a pinned BoolQ mirror, verify it, and materialize stable JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists() and sha256_file(target) == expected_sha256:
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "jev-calibration-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    temporary.replace(target)


def write_jsonl_from_parquet(source: Path, target: Path) -> int:
    table = pq.read_table(source)
    rows: list[dict[str, Any]] = table.to_pylist()
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            if not isinstance(row.get("answer"), bool):
                raise ValueError(f"non-boolean answer in {source}")
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    if dataset.get("source_format") != "parquet":
        raise ValueError("prepare_boolq.py expects source_format=parquet")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = out_dir / "train.parquet"
    dev_parquet = out_dir / "dev.parquet"
    download_verified(dataset["train_url"], train_parquet, dataset["train_sha256"])
    download_verified(dataset["dev_url"], dev_parquet, dataset["dev_sha256"])
    train_jsonl = out_dir / "train.jsonl"
    dev_jsonl = out_dir / "dev.jsonl"
    train_count = write_jsonl_from_parquet(train_parquet, train_jsonl)
    dev_count = write_jsonl_from_parquet(dev_parquet, dev_jsonl)
    if train_count != dataset["train_examples"] or dev_count != dataset["dev_examples"]:
        raise RuntimeError(
            f"row-count mismatch: train={train_count}, dev={dev_count}, "
            f"expected={dataset['train_examples']}/{dataset['dev_examples']}"
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": dataset["source_revision"],
        "sources": {
            "train": {
                "url": dataset["train_url"],
                "sha256": sha256_file(train_parquet),
                "bytes": train_parquet.stat().st_size,
            },
            "dev": {
                "url": dataset["dev_url"],
                "sha256": sha256_file(dev_parquet),
                "bytes": dev_parquet.stat().st_size,
            },
        },
        "materialized": {
            "train": {"rows": train_count, "sha256": sha256_file(train_jsonl)},
            "dev": {"rows": dev_count, "sha256": sha256_file(dev_jsonl)},
        },
    }
    temporary = out_dir / "source_manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out_dir / "source_manifest.json")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
