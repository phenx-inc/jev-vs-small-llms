#!/usr/bin/env python3
"""Two-GPU manager for the registered RLCDAlignBench distillation experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from rlcd_distillation import load_config, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def launch(command: list[str], gpu: str, log_path: Path) -> tuple[subprocess.Popen[Any], IO[str]]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, text=True)
    return process, handle


def wait_lanes(lanes: list[tuple[str, subprocess.Popen[Any], IO[str]]]) -> None:
    failures = []
    for name, process, handle in lanes:
        code = process.wait()
        handle.close()
        if code:
            failures.append((name, code))
    if failures:
        raise RuntimeError(f"lane failures: {failures}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory must be absent or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    data_manifest_path = data_dir / "manifest.json"
    manifest = {
        "created_utc": utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "gpu_lanes": {"0": "qwen35-4b CV then final", "1": "qwen35-9b final"},
        "final_test_gate": "Held-out evaluation launches only after both training lanes exit successfully.",
        "arms": config["arms"],
    }
    write_json(run_dir / "manifest.json", manifest)
    status_path = run_dir / "status.json"
    write_json(status_path, {"state": "training", "utc": utc_now()})
    scripts = Path(__file__).resolve().parent
    lane_script = scripts / "run_rlcd_distillation_lane.py"
    common = [
        "--config",
        str(config_path),
        "--data-dir",
        str(data_dir),
        "--run-dir",
        str(run_dir),
    ]
    lane4_command = [
        sys.executable,
        str(lane_script),
        "--phase",
        "train",
        *common,
        "--model-key",
        "qwen35-4b",
        "--include-cv",
    ]
    lane9_command = [
        sys.executable,
        str(lane_script),
        "--phase",
        "train",
        *common,
        "--model-key",
        "qwen35-9b",
    ]
    p4, h4 = launch(lane4_command, "0", run_dir / "logs" / "train-qwen35-4b.log")
    p9, h9 = launch(lane9_command, "1", run_dir / "logs" / "train-qwen35-9b.log")
    write_json(
        status_path,
        {"state": "training", "utc": utc_now(), "pids": {"qwen35-4b": p4.pid, "qwen35-9b": p9.pid}},
    )
    wait_lanes([("qwen35-4b", p4, h4), ("qwen35-9b", p9, h9)])

    write_json(status_path, {"state": "heldout_evaluation", "utc": utc_now()})
    eval4 = [
        sys.executable,
        str(lane_script),
        "--phase",
        "evaluate",
        *common,
        "--model-key",
        "qwen35-4b",
    ]
    eval9 = [
        sys.executable,
        str(lane_script),
        "--phase",
        "evaluate",
        *common,
        "--model-key",
        "qwen35-9b",
    ]
    e4, eh4 = launch(eval4, "0", run_dir / "logs" / "eval-qwen35-4b.log")
    e9, eh9 = launch(eval9, "1", run_dir / "logs" / "eval-qwen35-9b.log")
    wait_lanes([("eval-qwen35-4b", e4, eh4), ("eval-qwen35-9b", e9, eh9)])

    write_json(status_path, {"state": "analyzing", "utc": utc_now()})
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_rlcd_distillation.py"),
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )
    write_json(status_path, {"state": "complete", "utc": utc_now(), "results": str(run_dir / "results.json")})
    print(json.dumps({"event": "complete", "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "fatal", "type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise
