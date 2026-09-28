#!/usr/bin/env python3
"""Score RLCDAlignBench's exact generic-Noul questions with Qwen3.5-4B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import summarize
from protocol import extract_input_ids
from rlcd_protocol import (
    bootstrap_median_ci,
    build_prompt_text,
    promise_verdict,
    read_jsonl,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def prompt_ids(tokenizer: Any, config: dict[str, Any], row: dict[str, Any]) -> list[int]:
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": build_prompt_text(config, row)},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return extract_input_ids(encoded)


def sigmoid(delta: float) -> float:
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    exp_delta = math.exp(delta)
    return exp_delta / (1.0 + exp_delta)


def prediction_key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def read_completed(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = read_jsonl(path)
    keys = [prediction_key(row) for row in rows]
    if len(set(keys)) != len(keys):
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
    batch_size = int(config["batch_size"])
    max_tokens = int(config["max_input_tokens"])
    yes_ids = tokenizer.encode(config["candidate_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(config["candidate_no"], add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids[0] == no_ids[0]:
        raise RuntimeError(f"single distinct Yes/No tokens required; Yes={yes_ids}, No={no_ids}")

    pending = [row for row in rows if prediction_key(row) not in completed]
    emitted: list[dict[str, Any]] = []
    mode = "a" if output_path.exists() else "w"
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
                model_output = model(**encoded, use_cache=False, logits_to_keep=1)
                logits = model_output.logits[:, -1, :].float()
                pair_log_probs = torch.log_softmax(
                    logits[:, [no_ids[0], yes_ids[0]]], dim=-1
                ).cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for offset, (row, ids) in enumerate(zip(batch, ids_by_row)):
                log_p_no = float(pair_log_probs[offset, 0])
                log_p_yes = float(pair_log_probs[offset, 1])
                delta = log_p_yes - log_p_no
                probability = sigmoid(delta)
                record = {
                    "id": row["id"],
                    "benchmark": row["benchmark"],
                    "failure_type": row["failure_type"],
                    "benchmark_split": row["benchmark_split"],
                    "target_model": row["target_model"],
                    "variant": row["variant"],
                    "label": int(row["label"]),
                    "battery": row["battery"],
                    "jev_probability_yes": (
                        None
                        if row.get("jev_probability_yes") is None
                        else float(row["jev_probability_yes"])
                    ),
                    "qwen_log_probability_no_renormalized": log_p_no,
                    "qwen_log_probability_yes_renormalized": log_p_yes,
                    "qwen_logit_delta_yes_minus_no": delta,
                    "qwen_probability_yes": probability,
                    "prompt_tokens": len(ids),
                    "question_sha256": hashlib.sha256(
                        json.dumps(row["question"], sort_keys=True).encode("utf-8")
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


def median_finite(values: list[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        raise ValueError("cannot take median of no finite values")
    return statistics.median(values)


def build_report(
    rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    config: dict[str, Any],
    complete_scope: bool,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark"], []).append(row)
    sources = {source["benchmark"]: source for source in source_manifest["sources"]}
    per_benchmark = []
    for benchmark, items in sorted(grouped.items()):
        labels = [item["label"] for item in items]
        qwen = [item["qwen_probability_yes"] for item in items]
        source = sources[benchmark]
        published_auroc = float(source["published_jev_auroc"])
        qwen_metrics = summarize(labels, qwen)
        cached_items = [item for item in items if item.get("jev_probability_yes") is not None]
        cached_labels = [item["label"] for item in cached_items]
        cached_jev = [item["jev_probability_yes"] for item in cached_items]
        cached_qwen = [item["qwen_probability_yes"] for item in cached_items]
        jev_cached_metrics = summarize(cached_labels, cached_jev)
        qwen_cached_metrics = summarize(cached_labels, cached_qwen)
        cache_complete = len(cached_items) == len(items)
        cached_reproduces_published = (
            abs(jev_cached_metrics["auroc"] - published_auroc) <= 0.000051
            if cache_complete
            else None
        )
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "failure_type": items[0]["failure_type"],
                "benchmark_split": items[0]["benchmark_split"],
                "canonical_n": len(items),
                "cached_jev_n": len(cached_items),
                "cached_jev_coverage": len(cached_items) / len(items),
                "cache_complete": cache_complete,
                "published_jev_auroc": published_auroc,
                "released_metrics_generic": source["released_metrics_generic"],
                "cached_auroc_reproduces_published": cached_reproduces_published,
                "jev_cached_subset": jev_cached_metrics,
                "qwen_same_cached_subset": qwen_cached_metrics,
                "qwen_canonical": qwen_metrics,
                "delta_jev_minus_qwen": {
                    "auroc_raw_complete_cache_minus_qwen": (
                        jev_cached_metrics["auroc"] - qwen_metrics["auroc"]
                        if cache_complete
                        else None
                    ),
                    "auroc_released_minus_qwen_canonical": published_auroc
                    - qwen_metrics["auroc"],
                    "log_loss_gain_qwen_minus_jev": (
                        qwen_metrics["log_loss"] - jev_cached_metrics["log_loss"]
                        if cache_complete
                        else None
                    ),
                    "brier_gain_qwen_minus_jev": (
                        qwen_metrics["brier"] - jev_cached_metrics["brier"]
                        if cache_complete
                        else None
                    ),
                },
            }
        )

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"benchmarks": 0}
        probability_items = [row for row in items if row["cache_complete"]]
        raw_auroc_deltas = [
            row["delta_jev_minus_qwen"]["auroc_raw_complete_cache_minus_qwen"]
            for row in probability_items
        ]
        released_auroc_deltas = [
            row["delta_jev_minus_qwen"]["auroc_released_minus_qwen_canonical"]
            for row in items
        ]
        log_loss_gains = [
            row["delta_jev_minus_qwen"]["log_loss_gain_qwen_minus_jev"]
            for row in probability_items
        ]
        brier_gains = [
            row["delta_jev_minus_qwen"]["brier_gain_qwen_minus_jev"]
            for row in probability_items
        ]
        return {
            "benchmarks": len(items),
            "canonical_examples": sum(row["canonical_n"] for row in items),
            "complete_probability_benchmarks": len(probability_items),
            "complete_probability_examples": sum(
                row["canonical_n"] for row in probability_items
            ),
            "complete_caches_reproducing_released_auroc": sum(
                row["cached_auroc_reproduces_published"] is True
                for row in probability_items
            ),
            "primary_complete_raw_cache": {
                "jev_median": {
                    metric: median_finite(
                        [row["jev_cached_subset"][metric] for row in probability_items]
                    )
                    for metric in (
                        "auroc",
                        "log_loss",
                        "brier",
                        "ece_15",
                        "adaptive_ece_15",
                    )
                },
                "qwen_median": {
                    metric: median_finite(
                        [row["qwen_canonical"][metric] for row in probability_items]
                    )
                    for metric in (
                        "auroc",
                        "log_loss",
                        "brier",
                        "ece_15",
                        "adaptive_ece_15",
                    )
                },
                "paired_median": {
                    "auroc_delta_jev_minus_qwen": statistics.median(raw_auroc_deltas),
                    "auroc_delta_bootstrap_ci95_over_benchmarks": bootstrap_median_ci(
                        raw_auroc_deltas, int(config["seed"])
                    ),
                    "log_loss_gain_qwen_minus_jev": statistics.median(log_loss_gains),
                    "brier_gain_qwen_minus_jev": statistics.median(brier_gains),
                },
            },
            "secondary_released_aggregate_auroc": {
                "jev_median": median_finite(
                    [row["published_jev_auroc"] for row in items]
                ),
                "qwen_median": median_finite(
                    [row["qwen_canonical"]["auroc"] for row in items]
                ),
                "paired_median_delta_jev_minus_qwen": statistics.median(
                    released_auroc_deltas
                ),
                "paired_median_delta_bootstrap_ci95": bootstrap_median_ci(
                    released_auroc_deltas, int(config["seed"])
                ),
            },
        }

    aggregates = {"all": aggregate(per_benchmark)}
    for split in ("hill-climb", "held-out"):
        aggregates[split] = aggregate(
            [row for row in per_benchmark if row["benchmark_split"] == split]
        )
    labels = [row["label"] for row in rows]
    complete_probability_benchmarks = {
        row["benchmark"] for row in per_benchmark if row["cache_complete"]
    }
    probability_rows = [
        row for row in rows if row["benchmark"] in complete_probability_benchmarks
    ]
    pooled = {
        "warning": "Pooled metrics mix domains and base rates; per-benchmark medians are primary.",
        "qwen_all_canonical": summarize(
            labels, [row["qwen_probability_yes"] for row in rows]
        ),
        "complete_cache_benchmark_subset": {
            "benchmarks": len(complete_probability_benchmarks),
            "examples": len(probability_rows),
            "jev": summarize(
                [row["label"] for row in probability_rows],
                [row["jev_probability_yes"] for row in probability_rows],
            ),
            "qwen": summarize(
                [row["label"] for row in probability_rows],
                [row["qwen_probability_yes"] for row in probability_rows],
            ),
        },
    }
    verdict = None
    if complete_scope:
        primary = aggregates["all"]["primary_complete_raw_cache"]["paired_median"]
        gate = config["promise_gate"]
        verdict = promise_verdict(
            median_auroc_delta_jev_minus_qwen=primary["auroc_delta_jev_minus_qwen"],
            median_log_loss_gain_qwen_minus_jev=primary["log_loss_gain_qwen_minus_jev"],
            median_brier_gain_qwen_minus_jev=primary["brier_gain_qwen_minus_jev"],
            auroc_margin=float(gate["auroc_clear_margin"]),
            calibration_margin=float(gate["calibration_clear_margin"]),
        )
    return {
        "completed_utc": utc_now(),
        "complete_registered_scope": complete_scope,
        "canonical_examples": len(rows),
        "cached_jev_examples": sum(
            row["cached_jev_n"] for row in per_benchmark
        ),
        "benchmarks": len(per_benchmark),
        "primary_estimand": (
            "median same-row AUROC difference on benchmarks whose raw Jev cache covers "
            "every canonical labeled row; all probability metrics use those complete caches"
        ),
        "release_caveat": (
            "Three response caches are incomplete even though their released metrics declare full "
            "coverage, and three fully covered grounded caches do not reproduce their bundled "
            "AUROC/accuracy. The 28 complete raw caches define the primary comparison; all 31 "
            "release-reported AUROCs are secondary and explicitly non-auditable from the bundled "
            "raw files for six benchmarks."
        ),
        "promise_verdict": verdict,
        "promise_gate": config["promise_gate"],
        "aggregates": aggregates,
        "pooled": pooled,
        "per_benchmark": per_benchmark,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    data_path = data_dir / "comparison.jsonl"
    source_manifest_path = data_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(data_path) != source_manifest["comparison_sha256"]:
        raise RuntimeError("prepared comparison hash mismatch")
    all_source_rows = read_jsonl(data_path)
    rows = all_source_rows[: args.max_examples] if args.max_examples is not None else all_source_rows
    if not rows:
        raise RuntimeError("no comparison rows selected")
    complete_scope = len(rows) == len(all_source_rows)

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and not args.resume:
        # The guarded launcher creates its log (and may create its PID file)
        # before this process starts. Those orchestration files are not scored
        # output and must not make a brand-new run look like a resume.
        unexpected = [
            path.name
            for path in out_dir.iterdir()
            if path.name != "pid" and not path.name.endswith(".console.log")
        ]
        if unexpected:
            raise RuntimeError(
                f"output directory contains prior run artifacts; use --resume: "
                f"{out_dir} ({sorted(unexpected)[:10]})"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    write_json(status_path, {"state": "starting", "utc": utc_now()})

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
        "prepared_dataset_manifest": source_manifest,
        "selected_examples": len(rows),
        "registered_examples": len(all_source_rows),
        "candidate_token_ids": {"yes": yes_ids, "no": no_ids},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "source_hashes": {
            "run_rlcdalignbench.py": sha256_file(Path(__file__).resolve()),
            "rlcd_protocol.py": sha256_file(Path(__file__).resolve().with_name("rlcd_protocol.py")),
            "metrics.py": sha256_file(Path(__file__).resolve().with_name("metrics.py")),
            "protocol.py": sha256_file(Path(__file__).resolve().with_name("protocol.py")),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": gpu_inventory(),
        "torch_cuda": torch.version.cuda,
        "data_handling": "Raw gated states stay in data_dir and are not copied into predictions.",
    }
    write_json(out_dir / "manifest.json", manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    write_json(status_path, {"state": "loading_model", "utc": utc_now()})
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

    predictions_path = out_dir / "predictions.jsonl"
    completed, existing = read_completed(predictions_path) if args.resume else (set(), [])
    write_json(
        status_path,
        {"state": "scoring", "utc": utc_now(), "remaining": len(rows) - len(completed)},
    )
    emitted = score_rows(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rows=rows,
        output_path=predictions_path,
        completed=completed,
    )
    scored = existing + emitted
    by_key = {prediction_key(row): row for row in scored}
    expected = {prediction_key(row) for row in rows}
    if set(by_key) != expected:
        raise RuntimeError(
            f"prediction key mismatch; missing={sorted(expected - set(by_key))[:10]}, "
            f"extra={sorted(set(by_key) - expected)[:10]}"
        )
    ordered = [by_key[prediction_key(row)] for row in rows]
    report = build_report(ordered, source_manifest, config, complete_scope)
    report["timing"] = {
        "gpu_forward_seconds": sum(
            row["batch_elapsed_seconds"] / row["batch_size_actual"] for row in ordered
        ),
        "note": "Forward-pass allocation only; excludes loading, tokenization, I/O and network.",
    }
    write_json(out_dir / "metrics.json", report)
    write_json(
        status_path,
        {
            "state": "complete",
            "utc": utc_now(),
            "promise_verdict": report["promise_verdict"],
        },
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "out_dir": str(out_dir),
                "examples": len(ordered),
                "benchmarks": report["benchmarks"],
                "promise_verdict": report["promise_verdict"],
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
