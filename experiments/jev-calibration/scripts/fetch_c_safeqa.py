#!/usr/bin/env python3
"""Download the exact pinned C-SafeQA release after gated access is approved."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


METADATA_FILES = ["README.md", "manifest.json", "schema.json", "verify_jsonl_artifact.py"]


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
    selected = [item["path"] for item in dataset["files"]] + METADATA_FILES

    snapshot_download(
        repo_id=dataset["repo_id"],
        repo_type="dataset",
        revision=dataset["revision"],
        local_dir=out_dir,
        allow_patterns=selected,
        token=True,
    )

    missing = [relative for relative in selected if not (out_dir / relative).is_file()]
    if missing:
        raise RuntimeError(f"download incomplete; missing files: {missing}")

    manifest = {
        "repo_id": dataset["repo_id"],
        "revision": dataset["revision"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "files": [
            {
                "path": relative,
                "bytes": (out_dir / relative).stat().st_size,
                "sha256": sha256_file(out_dir / relative),
            }
            for relative in selected
        ],
    }
    manifest_path = out_dir / "comparison_download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "download_complete", "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
