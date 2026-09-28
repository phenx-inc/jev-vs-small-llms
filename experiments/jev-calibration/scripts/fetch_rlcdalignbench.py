#!/usr/bin/env python3
"""Download only the gated RLCDAlignBench files required by this comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from rlcd_protocol import selected_release_paths, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vendor_root = Path(args.vendor_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    dataset = config["dataset"]
    api = HfApi(token=True)
    repo_files = api.list_repo_files(
        repo_id=dataset["repo_id"],
        repo_type="dataset",
        revision=dataset["revision"],
    )
    selected = selected_release_paths(
        repo_files,
        vendor_root / "results" / "main_results.csv",
        vendor_root / "data" / "variants.csv",
    )
    snapshot_download(
        repo_id=dataset["repo_id"],
        repo_type="dataset",
        revision=dataset["revision"],
        local_dir=out_dir,
        allow_patterns=selected,
        token=True,
    )
    missing = [path for path in selected if not (out_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"download incomplete; missing {missing[:10]}")
    write_json(
        out_dir / "comparison_download_manifest.json",
        {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "selected_files": [
                {
                    "path": path,
                    "bytes": (out_dir / path).stat().st_size,
                    "sha256": sha256_file(out_dir / path),
                }
                for path in selected
            ],
        },
    )
    print(json.dumps({"event": "download_complete", "files": len(selected), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
