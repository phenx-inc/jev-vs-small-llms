#!/usr/bin/env python3
"""Score the Jev-eligible RLCDAlignBench subset with a pinned local model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import summarize
from protocol import extract_input_ids
from rlcd_protocol import (
    build_prompt_text,
    read_jsonl,
    render_state,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-state-chars", type=int, default=4000)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--system-name")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selection_key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def selection_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(selection_key(row) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_rows(
    rows: list[dict[str, Any]], max_state_chars: int, max_examples: int | None = None
) -> list[dict[str, Any]]:
    if max_state_chars <= 0:
        raise ValueError("max_state_chars must be positive")
    selected = [
        row
        for row in rows
        if len(render_state(row["state"])) <= max_state_chars
    ]
    return selected[:max_examples] if max_examples is not None else selected


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "accelerate", "safetensors", "peft"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


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
    except Exception as exc:
        return [f"inventory_failed: {type(exc).__name__}: {exc}"]


def prompt_ids(tokenizer: Any, config: dict[str, Any], row: dict[str, Any]) -> list[int]:
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": build_prompt_text(config, row)},
    ]
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking")
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
    return extract_input_ids(encoded)


def sigmoid(delta: float) -> float:
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    exp_delta = math.exp(delta)
    return exp_delta / (1.0 + exp_delta)


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def forward_kwargs(model: Any) -> dict[str, int | bool]:
    parameters = inspect.signature(model.forward).parameters
    result: dict[str, int | bool] = {"use_cache": False}
    if "logits_to_keep" in parameters:
        result["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        result["num_logits_to_keep"] = 1
    return result


def read_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = read_jsonl(path)
    keys = [selection_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate prediction key in resume file")
    return set(keys), rows


def score_rows(
    *,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
    completed: set[str],
) -> list[dict[str, Any]]:
    import torch

    yes_ids = tokenizer.encode(config["candidate_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(config["candidate_no"], add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids[0] == no_ids[0]:
        raise RuntimeError(
            f"single distinct Yes/No tokens required; Yes={yes_ids}, No={no_ids}"
        )
    batch_size = int(config["batch_size"])
    max_tokens = int(config["max_input_tokens"])
    pending = [row for row in rows if selection_key(row) not in completed]
    emitted: list[dict[str, Any]] = []
    mode = "a" if output_path.exists() else "w"
    model_kwargs = forward_kwargs(model)
    with output_path.open(mode, encoding="utf-8", buffering=1) as output:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            ids_by_row = [prompt_ids(tokenizer, config, row) for row in batch]
            longest = max(map(len, ids_by_row), default=0)
            if longest > max_tokens:
                offenders = [
                    (row["benchmark"], row["id"], len(ids))
                    for row, ids in zip(batch, ids_by_row)
                    if len(ids) > max_tokens
                ]
                raise RuntimeError(
                    f"input exceeds frozen {max_tokens}-token limit: {offenders[:5]}"
                )
            encoded = tokenizer.pad(
                {"input_ids": ids_by_row}, padding=True, return_tensors="pt"
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model(**encoded, **model_kwargs)
                logits = result.logits[:, -1, :].float()
                pair_log_probs = torch.log_softmax(
                    logits[:, [no_ids[0], yes_ids[0]]], dim=-1
                ).cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for offset, (row, ids) in enumerate(zip(batch, ids_by_row)):
                log_p_no = float(pair_log_probs[offset, 0])
                log_p_yes = float(pair_log_probs[offset, 1])
                delta = log_p_yes - log_p_no
                record = {
                    "id": row["id"],
                    "benchmark": row["benchmark"],
                    "failure_type": row["failure_type"],
                    "benchmark_split": row["benchmark_split"],
                    "target_model": row["target_model"],
                    "variant": row["variant"],
                    "label": int(row["label"]),
                    "battery": row["battery"],
                    "model_id": config["model_id"],
                    "system_name": config.get("_system_name", config["model_id"]),
                    "model_log_probability_no_renormalized": log_p_no,
                    "model_log_probability_yes_renormalized": log_p_yes,
                    "model_logit_delta_yes_minus_no": delta,
                    "model_probability_yes": sigmoid(delta),
                    "prompt_group_sha256": row.get("prompt_group_sha256"),
                    "sample_rank_sha256": row.get("sample_rank_sha256"),
                    "prompt_tokens": len(ids),
                    "question_sha256": hashlib.sha256(
                        json.dumps(row["question"], sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "rendered_state_sha256": hashlib.sha256(
                        render_state(row["state"]).encode("utf-8")
                    ).hexdigest(),
                    "batch_elapsed_seconds": elapsed,
                    "batch_size_actual": len(batch),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                emitted.append(record)
            output.flush()
            os.fsync(output.fileno())
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed_new": batch_start + len(batch),
                        "pending_total": len(pending),
                        "last_batch_seconds": elapsed,
                        "last_batch_longest_tokens": longest,
                        "utc": utc_now(),
                    }
                ),
                flush=True,
            )
    return emitted


def build_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    per_benchmark = []
    for benchmark, items in sorted(grouped.items()):
        metrics = summarize(
            [int(row["label"]) for row in items],
            [float(row["model_probability_yes"]) for row in items],
        )
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "n": len(items),
                "failure_type": items[0]["failure_type"],
                "benchmark_split": items[0]["benchmark_split"],
                "metrics": finite_or_none(metrics),
            }
        )
    pooled = summarize(
        [int(row["label"]) for row in rows],
        [float(row["model_probability_yes"]) for row in rows],
    )
    return finite_or_none(
        {
            "system_name": config.get("_system_name", config["model_id"]),
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "examples": len(rows),
            "benchmarks": len(grouped),
            "pooled_warning": "Pooled metrics mix domains and base rates; per-benchmark comparisons are primary.",
            "pooled": pooled,
            "per_benchmark": per_benchmark,
        }
    )


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_system_name"] = args.system_name or config["run_family"]
    data_dir = Path(args.data_dir).resolve()
    data_path = data_dir / "comparison.jsonl"
    source_manifest_path = data_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(data_path) != source_manifest["comparison_sha256"]:
        raise RuntimeError("prepared comparison hash mismatch")
    all_rows = read_jsonl(data_path)
    rows = select_rows(all_rows, args.max_state_chars, args.max_examples)
    if not rows:
        raise RuntimeError("no comparison rows selected")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and not args.resume:
        unexpected = [
            path.name
            for path in out_dir.iterdir()
            if path.name != "pid" and not path.name.endswith(".console.log")
        ]
        if unexpected:
            raise RuntimeError(
                f"output directory contains prior run artifacts; use --resume: {out_dir}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    write_json(status_path, {"state": "starting", "utc": utc_now()})

    trust_remote_code = bool(config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        cache_dir=config.get("cache_dir"),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode(config["candidate_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(config["candidate_no"], add_special_tokens=False)

    manifest = {
        "created_utc": utc_now(),
        "config": config,
        "system_name": config["_system_name"],
        "adapter_dir": str(Path(args.adapter_dir).resolve()) if args.adapter_dir else None,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prepared_dataset_manifest": source_manifest,
        "registered_examples": len(all_rows),
        "selected_examples": len(rows),
        "max_state_chars": args.max_state_chars,
        "selection_sha256": selection_sha256(rows),
        "candidate_token_ids": {"yes": yes_ids, "no": no_ids},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "source_hashes": {
            "run_model_baseline.py": sha256_file(Path(__file__).resolve()),
            "rlcd_protocol.py": sha256_file(Path(__file__).resolve().with_name("rlcd_protocol.py")),
            "metrics.py": sha256_file(Path(__file__).resolve().with_name("metrics.py")),
            "protocol.py": sha256_file(Path(__file__).resolve().with_name("protocol.py")),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": gpu_inventory(),
        "data_handling": "Raw gated states stay in data_dir and are represented only by hashes in predictions.",
    }
    write_json(out_dir / "manifest.json", manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    write_json(status_path, {"state": "loading_model", "utc": utc_now()})
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    load_kwargs: dict[str, Any] = {
        "revision": config["model_revision"],
        "cache_dir": config.get("cache_dir"),
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    if config.get("attn_implementation"):
        load_kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], **load_kwargs
    ).to("cuda")
    if args.adapter_dir:
        from peft import PeftModel

        adapter_dir = Path(args.adapter_dir).resolve()
        if not (adapter_dir / "adapter_config.json").is_file():
            raise RuntimeError(f"adapter not found: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    manifest["loaded_model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    manifest["forward_parameters"] = list(inspect.signature(model.forward).parameters)
    write_json(out_dir / "manifest.json", manifest)

    predictions_path = out_dir / "predictions.jsonl"
    completed, existing = read_completed(predictions_path) if args.resume else (set(), [])
    write_json(
        status_path,
        {"state": "scoring", "utc": utc_now(), "remaining": len(rows) - len(completed)},
    )
    started = time.perf_counter()
    emitted = score_rows(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rows=rows,
        output_path=predictions_path,
        completed=completed,
    )
    wall_seconds = time.perf_counter() - started
    scored = existing + emitted
    by_key = {selection_key(row): row for row in scored}
    expected = {selection_key(row) for row in rows}
    if set(by_key) != expected:
        raise RuntimeError(
            f"prediction key mismatch; missing={sorted(expected - set(by_key))[:10]}, "
            f"extra={sorted(set(by_key) - expected)[:10]}"
        )
    ordered = [by_key[selection_key(row)] for row in rows]
    metrics = build_metrics(ordered, config)
    metrics["timing"] = {
        "scoring_wall_seconds": wall_seconds,
        "gpu_forward_seconds_allocated": sum(
            row["batch_elapsed_seconds"] / row["batch_size_actual"] for row in ordered
        ),
    }
    metrics["integrity"] = {
        "predictions_sha256": sha256_file(predictions_path),
        "selection_sha256": manifest["selection_sha256"],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        status_path,
        {
            "state": "complete",
            "utc": utc_now(),
            "examples": len(ordered),
            "wall_seconds": wall_seconds,
        },
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "model_id": config["model_id"],
                "out_dir": str(out_dir),
                "examples": len(ordered),
                "benchmarks": metrics["benchmarks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {"event": "fatal", "type": type(exc).__name__, "message": str(exc)}
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
