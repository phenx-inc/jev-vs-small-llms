#!/usr/bin/env python3
"""Fetch and verify the small, ungated BeaverTails 30k test artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "test.jsonl.gz"
    partial = output.with_suffix(output.suffix + ".partial")

    request = urllib.request.Request(dataset["url"], headers={"User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, output)
    actual_bytes = output.stat().st_size
    actual_sha256 = sha256_file(output)
    if actual_bytes != dataset["expected_bytes"]:
        raise RuntimeError(
            f"download byte count mismatch: expected {dataset['expected_bytes']}, got {actual_bytes}"
        )
    if actual_sha256 != dataset["expected_sha256"]:
        raise RuntimeError("download SHA-256 mismatch")

    manifest = {
        "repo_id": dataset["repo_id"],
        "revision": dataset["revision"],
        "split": dataset["split"],
        "source_path": dataset["path"],
        "source_url": dataset["url"],
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "config_sha256": sha256_file(config_path),
    }
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "download_complete", **manifest}))


if __name__ == "__main__":
    main()
