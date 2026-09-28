#!/usr/bin/env python3
"""Run one model's sequential train or final-evaluation jobs on a single GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rlcd_distillation import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--include-cv", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(json.dumps({"event": "launch_child", "command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    config = load_config(config_path)
    arms = list(config["arms"])
    scripts = Path(__file__).resolve().parent
    if args.phase == "train":
        if args.include_cv:
            for arm in arms:
                for fold in range(int(config["cv"]["folds"])):
                    output = run_dir / "cv" / args.model_key / arm / f"fold-{fold}"
                    run(
                        [
                            sys.executable,
                            str(scripts / "train_rlcd_distillation.py"),
                            "--config",
                            str(config_path),
                            "--data-dir",
                            str(data_dir),
                            "--output-dir",
                            str(output),
                            "--model-key",
                            args.model_key,
                            "--arm",
                            arm,
                            "--validation-fold",
                            str(fold),
                        ]
                    )
        for arm in arms:
            output = run_dir / "final" / args.model_key / arm
            run(
                [
                    sys.executable,
                    str(scripts / "train_rlcd_distillation.py"),
                    "--config",
                    str(config_path),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output),
                    "--model-key",
                    args.model_key,
                    "--arm",
                    arm,
                ]
            )
        return

    systems: list[tuple[str, Path | None]] = [("base", None)]
    systems.extend(
        (arm, run_dir / "final" / args.model_key / arm / "adapter") for arm in arms
    )
    for name, adapter in systems:
        output = run_dir / "heldout_eval" / args.model_key / name
        command = [
            sys.executable,
            str(scripts / "evaluate_rlcd_distillation.py"),
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output),
            "--model-key",
            args.model_key,
            "--system-name",
            name,
        ]
        if adapter is not None:
            command.extend(("--adapter-dir", str(adapter)))
        run(command)


if __name__ == "__main__":
    main()
