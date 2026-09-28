#!/usr/bin/env python3
"""QLoRA-train Qwen on cached Jev, hard, or hybrid binary targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
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
    stable_shuffle,
    target_for_arm,
    teacher_metrics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--arm", choices=("jev", "hard", "hybrid"), required=True)
    parser.add_argument("--validation-fold", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def optimizer_schedule(optimizer: Any, total_steps: int, warmup_ratio: float) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def tokenize_rows(
    rows: list[dict[str, Any]], tokenizer: Any, scoring: dict[str, Any], max_tokens: int
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    prepared: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        ids = prompt_ids(tokenizer, scoring, row)
        if len(ids) > max_tokens:
            raise RuntimeError(
                f"no-truncation gate failed: {row_key(row)} has {len(ids)} tokens, cap={max_tokens}"
            )
        prepared.append({"row": row, "input_ids": ids})
        lengths.append(len(ids))
    ordered = sorted(lengths)
    return prepared, {
        "n": len(lengths),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def batch_tensors(tokenizer: Any, items: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    encoded = tokenizer.pad(
        {"input_ids": [item["input_ids"] for item in items]},
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def predict(
    *,
    model: Any,
    tokenizer: Any,
    prepared: list[dict[str, Any]],
    yes_id: int,
    no_id: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    result_rows: list[dict[str, Any]] = []
    kwargs = forward_kwargs(model)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            items = prepared[start : start + batch_size]
            tensors = batch_tensors(tokenizer, items, model.device)
            output = model(**tensors, **kwargs)
            logits = output.logits[:, -1, :].float()
            deltas = logits[:, yes_id] - logits[:, no_id]
            probabilities = torch.sigmoid(deltas).cpu().tolist()
            for item, delta, probability in zip(items, deltas.cpu().tolist(), probabilities):
                row = item["row"]
                result_rows.append(
                    {
                        "id": row["id"],
                        "benchmark": row["benchmark"],
                        "label": int(row["label"]),
                        "jev_probability_yes": float(row["jev_probability_yes"]),
                        "model_probability_yes": float(probability),
                        "model_logit_delta_yes_minus_no": float(delta),
                        "prompt_tokens": len(item["input_ids"]),
                    }
                )
    model.train()
    return result_rows


def validation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["benchmark"])].append(row)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        probabilities = [float(row["model_probability_yes"]) for row in items]
        teacher = [float(row["jev_probability_yes"]) for row in items]
        labels = [int(row["label"]) for row in items]
        return {
            "teacher_fidelity": teacher_metrics(probabilities, teacher),
            "official_labels": summarize(labels, probabilities),
        }

    return {
        "pooled": metrics(rows),
        "per_benchmark": [
            {"benchmark": benchmark, "n": len(items), **metrics(items)}
            for benchmark, items in sorted(grouped.items())
        ],
    }


def main() -> None:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    if args.model_key not in config["models"]:
        raise ValueError(f"unknown model key: {args.model_key}")
    model_config = config["models"][args.model_key]
    training = config["training"]
    scoring = config["scoring"]
    data_manifest_path = data_dir / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    train_path = data_dir / "train.jsonl"
    rows = read_and_verify(train_path, data_manifest["train"]["sha256"])
    registered_rows = len(rows)
    if args.validation_fold is not None:
        folds = int(config["cv"]["folds"])
        if not 0 <= args.validation_fold < folds:
            raise ValueError(f"validation fold must be in [0,{folds})")
        train_rows = [row for row in rows if int(row["cv_fold"]) != args.validation_fold]
        validation_rows = [row for row in rows if int(row["cv_fold"]) == args.validation_fold]
    else:
        train_rows = rows
        validation_rows = []
    train_rows = stable_shuffle(train_rows, int(config["seed"]))
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max-examples must be positive")
        train_rows = train_rows[: args.max_examples]
        if validation_rows:
            validation_rows = validation_rows[: args.max_examples]

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
    max_tokens = int(training["max_input_tokens"])
    write_json(status_path, {"state": "tokenizing", "utc": utc_now()})
    prepared_train, train_lengths = tokenize_rows(train_rows, tokenizer, scoring, max_tokens)
    if validation_rows:
        prepared_validation, validation_lengths = tokenize_rows(
            validation_rows, tokenizer, scoring, max_tokens
        )
    else:
        prepared_validation, validation_lengths = [], None

    micro_batch = int(model_config["micro_batch_size"])
    accumulation = int(model_config["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    micro_steps = math.ceil(len(prepared_train) / micro_batch) * epochs
    total_optimizer_steps = math.ceil(micro_steps / accumulation)
    if args.max_optimizer_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.max_optimizer_steps)
    manifest = {
        "created_utc": utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "model_key": args.model_key,
        "model": model_config,
        "arm": args.arm,
        "validation_fold": args.validation_fold,
        "registered_train_rows": registered_rows,
        "selected_train_rows": len(train_rows),
        "selected_validation_rows": len(validation_rows),
        "train_key_sha256": hashlib.sha256(
            "\n".join(row_key(row) for row in train_rows).encode()
        ).hexdigest(),
        "training": training,
        "hybrid_jev_weight": config["hybrid_jev_weight"],
        "candidate_token_ids": {"yes": yes_id, "no": no_id},
        "prompt_lengths_train": train_lengths,
        "prompt_lengths_validation": validation_lengths,
        "optimizer_steps_planned": total_optimizer_steps,
        "packages": package_versions(
            ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_handling": "This process opens train.jsonl only; it never opens heldout.jsonl or BeaverTails.",
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(training["lora_target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    manifest["loaded_model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    manifest["trainable_parameters"] = trainable
    manifest["total_parameters"] = total
    manifest["trainable_fraction"] = trainable / total
    write_json(output_dir / "manifest.json", manifest)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = optimizer_schedule(optimizer, total_optimizer_steps, float(training["warmup_ratio"]))
    kwargs = forward_kwargs(model)
    target_weight = float(config["hybrid_jev_weight"])
    log_path = output_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    started = time.perf_counter()
    write_json(
        status_path,
        {
            "state": "training",
            "utc": utc_now(),
            "optimizer_steps_planned": total_optimizer_steps,
        },
    )
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        stop = False
        for epoch in range(epochs):
            epoch_items = list(prepared_train)
            random.Random(int(config["seed"]) + epoch).shuffle(epoch_items)
            for start in range(0, len(epoch_items), micro_batch):
                items = epoch_items[start : start + micro_batch]
                tensors = batch_tensors(tokenizer, items, model.device)
                targets = torch.tensor(
                    [target_for_arm(item["row"], args.arm, target_weight) for item in items],
                    device=model.device,
                    dtype=torch.float32,
                )
                output = model(**tensors, **kwargs)
                logits = output.logits[:, -1, :].float()
                deltas = logits[:, yes_id] - logits[:, no_id]
                loss = F.binary_cross_entropy_with_logits(deltas, targets)
                (loss / accumulation).backward()
                micro_step += 1
                boundary = micro_step % accumulation == 0 or start + micro_batch >= len(epoch_items)
                if not boundary:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(training["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                record = {
                    "utc": utc_now(),
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "optimizer_steps_planned": total_optimizer_steps,
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.perf_counter() - started,
                    "max_prompt_tokens_in_micro_batch": max(len(item["input_ids"]) for item in items),
                }
                log.write(json.dumps(record) + "\n")
                if optimizer_step == 1 or optimizer_step % 10 == 0:
                    print(json.dumps({"event": "train_progress", **record}), flush=True)
                    write_json(status_path, {"state": "training", **record})
                if optimizer_step >= total_optimizer_steps:
                    stop = True
                    break
            if stop:
                break

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    result: dict[str, Any] = {
        "optimizer_steps": optimizer_step,
        "wall_seconds": time.perf_counter() - started,
        "adapter_path": str(adapter_dir),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
    }
    if prepared_validation:
        write_json(status_path, {"state": "validating", "utc": utc_now()})
        predictions = predict(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared_validation,
            yes_id=yes_id,
            no_id=no_id,
            batch_size=micro_batch,
        )
        cv_metrics = validation_metrics(predictions)
        cv_metrics["validation_fold"] = args.validation_fold
        cv_metrics["arm"] = args.arm
        cv_metrics["model_key"] = args.model_key
        write_json(output_dir / "cv_metrics.json", cv_metrics)
        result["cv_metrics_sha256"] = sha256_file(output_dir / "cv_metrics.json")
    write_json(output_dir / "result.json", result)
    write_json(status_path, {"state": "complete", "utc": utc_now(), **result})
    print(json.dumps({"event": "complete", **result}), flush=True)


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
