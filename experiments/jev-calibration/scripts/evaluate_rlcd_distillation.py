#!/usr/bin/env python3
"""Evaluate a base or distilled model once on the frozen official held-out split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import summarize
from rlcd_distillation import (
    candidate_token_ids,
    forward_kwargs,
    load_config,
    package_versions,
    prompt_ids,
    read_and_verify,
    row_key,
    sha256_file,
    teacher_metrics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--system-name", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_or_none(value: Any) -> Any:
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    official = summarize(
        [int(row["label"]) for row in rows],
        [float(row["model_probability_yes"]) for row in rows],
    )
    paired = [row for row in rows if row.get("jev_probability_yes") is not None]
    teacher = teacher_metrics(
        [float(row["model_probability_yes"]) for row in paired],
        [float(row["jev_probability_yes"]) for row in paired],
    )
    return finite_or_none(
        {
            "n_official": len(rows),
            "n_teacher": len(paired),
            "official_labels": official,
            "teacher_fidelity": teacher,
        }
    )


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = parse_args()
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "starting", "utc": utc_now()})
    config = load_config(config_path)
    model_config = config["models"][args.model_key]
    scoring = config["scoring"]
    data_manifest_path = data_dir / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    heldout_path = data_dir / "heldout.jsonl"
    rows = read_and_verify(heldout_path, data_manifest["heldout"]["sha256"])
    expected = int(config["dataset"]["expected_heldout_rows"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} held-out rows, found {len(rows)}")
    if any(row["benchmark_split"] != "held-out" for row in rows):
        raise RuntimeError("non-held-out row in final test file")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config["model_revision"],
        cache_dir=model_config["cache_dir"],
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = candidate_token_ids(tokenizer, scoring)
    max_tokens = int(config["training"]["max_input_tokens"])
    prepared: list[dict[str, Any]] = []
    for row in rows:
        ids = prompt_ids(tokenizer, scoring, row)
        if len(ids) > max_tokens:
            raise RuntimeError(
                f"no-truncation gate failed: {row_key(row)} has {len(ids)} tokens, cap={max_tokens}"
            )
        prepared.append({"row": row, "input_ids": ids})

    manifest = {
        "created_utc": utc_now(),
        "system_name": args.system_name,
        "model_key": args.model_key,
        "model": model_config,
        "adapter_dir": str(Path(args.adapter_dir).resolve()) if args.adapter_dir else None,
        "config_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "heldout_sha256": sha256_file(heldout_path),
        "heldout_rows": len(rows),
        "candidate_token_ids": {"yes": yes_id, "no": no_id},
        "prompt_tokens": {
            "min": min(len(item["input_ids"]) for item in prepared),
            "max": max(len(item["input_ids"]) for item in prepared),
        },
        "packages": package_versions(("torch", "transformers", "peft", "bitsandbytes")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(output_dir / "manifest.json", manifest)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    write_json(status_path, {"state": "loading_model", "utc": utc_now()})
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        revision=model_config["model_revision"],
        cache_dir=model_config["cache_dir"],
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir).resolve()
        if not (adapter_dir / "adapter_config.json").exists():
            raise RuntimeError(f"adapter not found: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    kwargs = forward_kwargs(model)
    batch_size = int(model_config["micro_batch_size"])
    predictions_path = output_dir / "predictions.jsonl"
    result_rows: list[dict[str, Any]] = []
    write_json(status_path, {"state": "scoring", "utc": utc_now(), "rows": len(prepared)})
    with predictions_path.open("w", encoding="utf-8", buffering=1) as handle:
        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                items = prepared[start : start + batch_size]
                encoded = tokenizer.pad(
                    {"input_ids": [item["input_ids"] for item in items]},
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                output = model(**encoded, **kwargs)
                logits = output.logits[:, -1, :].float()
                deltas = logits[:, yes_id] - logits[:, no_id]
                probabilities = torch.sigmoid(deltas).cpu().tolist()
                for item, delta, probability in zip(items, deltas.cpu().tolist(), probabilities):
                    row = item["row"]
                    record = {
                        "id": row["id"],
                        "benchmark": row["benchmark"],
                        "failure_type": row["failure_type"],
                        "benchmark_split": row["benchmark_split"],
                        "label": int(row["label"]),
                        "jev_probability_yes": row.get("jev_probability_yes"),
                        "model_probability_yes": float(probability),
                        "model_logit_delta_yes_minus_no": float(delta),
                        "prompt_tokens": len(item["input_ids"]),
                    }
                    handle.write(json.dumps(record) + "\n")
                    result_rows.append(record)
                if start == 0 or (start + len(items)) % 100 == 0:
                    progress = start + len(items)
                    print(json.dumps({"event": "eval_progress", "rows": progress}), flush=True)
                    write_json(
                        status_path,
                        {"state": "scoring", "utc": utc_now(), "completed": progress, "rows": len(prepared)},
                    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[str(row["benchmark"])].append(row)
    metrics = {
        "system_name": args.system_name,
        "model_key": args.model_key,
        "pooled_warning": "Per-benchmark results are primary because pooled metrics mix domains and base rates.",
        "pooled": metric_bundle(result_rows),
        "per_benchmark": [
            {"benchmark": benchmark, **metric_bundle(items)}
            for benchmark, items in sorted(grouped.items())
        ],
        "integrity": {"predictions_sha256": sha256_file(predictions_path)},
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        status_path,
        {"state": "complete", "utc": utc_now(), "rows": len(result_rows), "metrics": str(output_dir / "metrics.json")},
    )
    print(json.dumps({"event": "complete", "rows": len(result_rows)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps({"event": "fatal", "type": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
            flush=True,
        )
        raise
