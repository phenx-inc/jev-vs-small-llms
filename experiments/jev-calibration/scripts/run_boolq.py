#!/usr/bin/env python3
"""Run the registered no-Jev-key BoolQ calibration baseline on one CUDA GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import fit_temperature, probabilities, summarize
from protocol import extract_input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-calibration", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) == expected_sha256:
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    temporary.replace(target)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("answer"), bool):
                    raise ValueError(f"missing boolean answer at {path}:{line_number}")
                rows.append(row)
    return rows


def select_calibration(rows: list[dict[str, Any]], size: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    if size > len(rows):
        raise ValueError(f"calibration_size {size} exceeds {len(rows)} training rows")
    indices = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [(index, rows[index]) for index in indices]


def build_prompt_ids(tokenizer: Any, config: dict[str, Any], row: dict[str, Any]) -> list[int]:
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": config["user_template"].format(
                passage=row["passage"], question=row["question"]
            ),
        },
    ]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return extract_input_ids(ids)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_inventory() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception as exc:  # recorded rather than hidden
        return [f"inventory_failed: {type(exc).__name__}: {exc}"]


def read_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = []
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['split']}:{row['source_index']}"
            if key in keys:
                raise RuntimeError(f"duplicate prediction key in resume file: {key}")
            keys.add(key)
            rows.append(row)
    return keys, rows


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_rows(
    *,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    examples: list[tuple[str, int, dict[str, Any]]],
    output_path: Path,
    completed: set[str],
) -> list[dict[str, Any]]:
    batch_size = int(config["batch_size"])
    max_tokens = int(config["max_input_tokens"])
    yes_ids = tokenizer.encode(config["candidate_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(config["candidate_no"], add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            "registered scorer requires single-token candidates; "
            f"got Yes={yes_ids}, No={no_ids}"
        )
    if yes_ids[0] == no_ids[0]:
        raise RuntimeError("Yes and No resolve to the same token")

    pending = [item for item in examples if f"{item[0]}:{item[1]}" not in completed]
    emitted: list[dict[str, Any]] = []
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", buffering=1) as output:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            prompt_ids = [build_prompt_ids(tokenizer, config, row) for _, _, row in batch]
            oversize = [len(ids) for ids in prompt_ids if len(ids) > max_tokens]
            if oversize:
                raise RuntimeError(
                    f"input exceeds registered {max_tokens}-token limit; longest={max(oversize)}"
                )
            encoded = tokenizer.pad(
                {"input_ids": prompt_ids}, padding=True, return_tensors="pt"
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, logits_to_keep=1)
                last_logits = outputs.logits[:, -1, :].float()
                pair_logits = last_logits[:, [no_ids[0], yes_ids[0]]]
                pair_log_probs = torch.log_softmax(pair_logits, dim=-1).cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            for row_offset, ((split, source_index, row), ids) in enumerate(zip(batch, prompt_ids)):
                log_p_no = float(pair_log_probs[row_offset, 0])
                log_p_yes = float(pair_log_probs[row_offset, 1])
                delta = log_p_yes - log_p_no
                p_yes = 1.0 / (1.0 + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1.0 + math.exp(delta))
                record = {
                    "split": split,
                    "source_index": source_index,
                    "label": int(row["answer"]),
                    "title": row.get("title"),
                    "question": row["question"],
                    "prompt_tokens": len(ids),
                    "log_probability_no_renormalized": log_p_no,
                    "log_probability_yes_renormalized": log_p_yes,
                    "logit_delta_yes_minus_no": delta,
                    "probability_yes_raw": p_yes,
                    "prediction_raw": int(p_yes >= 0.5),
                    "batch_elapsed_seconds": elapsed,
                    "batch_size_actual": len(batch),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                emitted.append(record)
            output.flush()
            os.fsync(output.fileno())
            done = batch_start + len(batch)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed_new": done,
                        "pending_total": len(pending),
                        "last_batch_seconds": elapsed,
                        "utc": utc_now(),
                    }
                ),
                flush=True,
            )
    return emitted


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory is non-empty; use --resume: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else out_dir / "data"
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"

    status_path = out_dir / "status.json"
    write_json(status_path, {"state": "starting", "utc": utc_now()})
    dataset_config = config["dataset"]
    prepared_manifest = None
    if args.data_dir:
        prepared_manifest_path = data_dir / "source_manifest.json"
        if not prepared_manifest_path.exists() or not train_path.exists() or not dev_path.exists():
            raise RuntimeError(f"prepared dataset is incomplete: {data_dir}")
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        for split, path in (("train", train_path), ("dev", dev_path)):
            expected = prepared_manifest["materialized"][split]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"prepared {split} hash mismatch: expected {expected}, got {actual}"
                )
    else:
        if dataset_config.get("source_format", "jsonl") != "jsonl":
            raise RuntimeError(
                "this config uses a Parquet source; run prepare_boolq.py and pass --data-dir"
            )
        download_verified(dataset_config["train_url"], train_path, dataset_config["train_sha256"])
        download_verified(dataset_config["dev_url"], dev_path, dataset_config["dev_sha256"])
    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    if len(train_rows) != dataset_config["train_examples"]:
        raise RuntimeError(f"unexpected train row count: {len(train_rows)}")
    if len(dev_rows) != dataset_config["dev_examples"]:
        raise RuntimeError(f"unexpected dev row count: {len(dev_rows)}")

    calibration_size = min(
        int(config["calibration_size"]),
        args.max_calibration if args.max_calibration is not None else int(config["calibration_size"]),
    )
    evaluation_limit = args.max_eval
    if evaluation_limit is None:
        evaluation_limit = config.get("evaluation_size")
    selected_calibration = select_calibration(train_rows, calibration_size, int(config["seed"]))
    selected_eval = list(enumerate(dev_rows))
    if evaluation_limit is not None:
        selected_eval = selected_eval[: int(evaluation_limit)]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode(config["candidate_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(config["candidate_no"], add_special_tokens=False)

    manifest = {
        "created_utc": utc_now(),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_hashes": {"train": sha256_file(train_path), "dev": sha256_file(dev_path)},
        "prepared_dataset_manifest": prepared_manifest,
        "selected_counts": {"calibration": len(selected_calibration), "evaluation": len(selected_eval)},
        "candidate_token_ids": {"yes": yes_ids, "no": no_ids},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "source_hashes": {
            "run_boolq.py": sha256_file(Path(__file__).resolve()),
            "metrics.py": sha256_file(Path(__file__).resolve().with_name("metrics.py")),
            "protocol.py": sha256_file(Path(__file__).resolve().with_name("protocol.py")),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": gpu_inventory(),
        "torch_cuda": torch.version.cuda,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(status_path, {"state": "loading_model", "utc": utc_now()})

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered GPU run")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    raw_path = out_dir / "predictions.jsonl"
    completed, existing = read_completed(raw_path) if args.resume else (set(), [])
    examples = [
        ("calibration", index, row) for index, row in selected_calibration
    ] + [("evaluation", index, row) for index, row in selected_eval]
    write_json(status_path, {"state": "scoring", "utc": utc_now(), "remaining": len(examples) - len(completed)})
    emitted = score_rows(
        model=model,
        tokenizer=tokenizer,
        config=config,
        examples=examples,
        output_path=raw_path,
        completed=completed,
    )
    all_rows = existing + emitted
    by_key = {(row["split"], row["source_index"]): row for row in all_rows}
    expected_keys = {(split, index) for split, index, _ in examples}
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))[:10]
        extra = sorted(set(by_key) - expected_keys)[:10]
        raise RuntimeError(f"prediction key mismatch; missing={missing}, extra={extra}")

    calibration = [by_key[("calibration", index)] for index, _ in selected_calibration]
    evaluation = [by_key[("evaluation", index)] for index, _ in selected_eval]
    calibration_labels = [row["label"] for row in calibration]
    calibration_deltas = [row["logit_delta_yes_minus_no"] for row in calibration]
    temperature = fit_temperature(calibration_labels, calibration_deltas)
    eval_labels = [row["label"] for row in evaluation]
    eval_deltas = [row["logit_delta_yes_minus_no"] for row in evaluation]
    raw_probs = probabilities(eval_deltas)
    calibrated_probs = probabilities(eval_deltas, temperature)
    base_rate = sum(calibration_labels) / len(calibration_labels)
    constant_probs = [base_rate] * len(eval_labels)
    majority_value = int(base_rate >= 0.5)
    majority_probs = [float(majority_value)] * len(eval_labels)

    report = {
        "completed_utc": utc_now(),
        "temperature": temperature,
        "calibration_subset": {
            "n": len(calibration_labels),
            "positive_rate": base_rate,
            "raw": summarize(calibration_labels, probabilities(calibration_deltas)),
            "temperature_scaled_in_sample": summarize(
                calibration_labels, probabilities(calibration_deltas, temperature)
            ),
        },
        "evaluation": {
            "raw_qwen_choice_probability": summarize(eval_labels, raw_probs),
            "temperature_scaled_qwen": summarize(eval_labels, calibrated_probs),
            "constant_calibration_base_rate": summarize(eval_labels, constant_probs),
            "majority_class": summarize(eval_labels, majority_probs),
        },
        "timing": {
            "scored_examples": len(all_rows),
            "gpu_scoring_seconds": sum(
                row["batch_elapsed_seconds"] / row["batch_size_actual"] for row in all_rows
            ),
        },
    }
    write_json(out_dir / "metrics.json", report)
    write_json(status_path, {"state": "complete", "utc": utc_now(), "metrics": "metrics.json"})
    print(json.dumps({"event": "complete", "out_dir": str(out_dir), "temperature": temperature}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "fatal", "type": type(exc).__name__, "message": str(exc)}), file=sys.stderr, flush=True)
        raise
