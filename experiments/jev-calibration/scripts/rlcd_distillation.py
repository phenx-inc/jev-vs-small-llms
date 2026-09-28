"""Shared, pure helpers for the split-safe RLCDAlignBench distillation study."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from protocol import extract_input_ids
from rlcd_protocol import build_prompt_text, read_jsonl, render_state, sha256_file, write_json


TRAIN_BENCHMARKS = frozenset(
    {
        "bias_refletter",
        "confaide",
        "elephant_aita",
        "faith_mt_claimcheck",
        "faith_mt_grounded",
        "harmbench",
        "instrumental_eval",
        "jbb_artifacts",
        "llm_aggrefact_A",
        "llm_aggrefact_B",
        "mask_continuation",
        "mask_disinformation",
        "mask_factual",
        "open_prompt_injection",
        "privaci_bench",
        "privacylens",
        "ragtruth",
        "reward_hacking_freeform",
        "reward_harm_freeform",
        "rh_mt_reward",
        "sycophancy_eval",
        "sycophancy_feedback",
        "tensor_trust_extract",
        "tensor_trust_hijack",
        "verbalized_ece",
    }
)

HELDOUT_BENCHMARKS = frozenset(
    {
        "bias_scene_heldout",
        "deceptionbench_reward",
        "injecagent",
        "privaci_gdpr_heldout",
        "strongreject",
        "summedits",
    }
)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def row_key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def keys_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(row_key(row) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_for_arm(row: dict[str, Any], arm: str, hybrid_jev_weight: float = 0.5) -> float:
    teacher = float(row["jev_probability_yes"])
    label = float(row["label"])
    if arm == "jev":
        return teacher
    if arm == "hard":
        return label
    if arm == "hybrid":
        if not 0.0 <= hybrid_jev_weight <= 1.0:
            raise ValueError("hybrid_jev_weight must lie in [0,1]")
        return hybrid_jev_weight * teacher + (1.0 - hybrid_jev_weight) * label
    raise ValueError(f"unknown training arm: {arm}")


def assign_benchmark_folds(
    rows: list[dict[str, Any]], folds: int, seed: int
) -> tuple[dict[str, int], list[int]]:
    """Greedily balance whole benchmarks across deterministic folds."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["benchmark"])] += 1
    if folds > len(counts):
        raise ValueError("more folds than benchmark groups")
    tie_break = {
        benchmark: hashlib.sha256(f"{seed}:{benchmark}".encode()).hexdigest()
        for benchmark in counts
    }
    ordered = sorted(counts, key=lambda name: (-counts[name], tie_break[name]))
    totals = [0] * folds
    assignments: dict[str, int] = {}
    for benchmark in ordered:
        fold = min(range(folds), key=lambda index: (totals[index], index))
        assignments[benchmark] = fold
        totals[fold] += counts[benchmark]
    return assignments, totals


def validate_and_partition(
    rows: list[dict[str, Any]], *, expected_source_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != expected_source_rows:
        raise RuntimeError(f"expected {expected_source_rows} source rows, found {len(rows)}")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate comparison row key")
    observed_train = {r["benchmark"] for r in rows if r["benchmark_split"] == "hill-climb"}
    observed_heldout = {r["benchmark"] for r in rows if r["benchmark_split"] == "held-out"}
    if observed_train != TRAIN_BENCHMARKS:
        raise RuntimeError(f"hill-climb benchmark mismatch: {sorted(observed_train ^ TRAIN_BENCHMARKS)}")
    if observed_heldout != HELDOUT_BENCHMARKS:
        raise RuntimeError(f"held-out benchmark mismatch: {sorted(observed_heldout ^ HELDOUT_BENCHMARKS)}")
    unknown_splits = {r["benchmark_split"] for r in rows} - {"hill-climb", "held-out"}
    if unknown_splits:
        raise RuntimeError(f"unknown benchmark splits: {sorted(unknown_splits)}")
    train = [
        dict(row)
        for row in rows
        if row["benchmark_split"] == "hill-climb" and row.get("jev_probability_yes") is not None
    ]
    heldout = [dict(row) for row in rows if row["benchmark_split"] == "held-out"]
    train_missing_teacher = sum(
        row["benchmark_split"] == "hill-climb" and row.get("jev_probability_yes") is None
        for row in rows
    )
    heldout_missing_teacher = sum(row.get("jev_probability_yes") is None for row in heldout)
    audit = {
        "source_rows": len(rows),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "heldout_teacher_rows": len(heldout) - heldout_missing_teacher,
        "excluded_hill_climb_missing_teacher": train_missing_teacher,
        "heldout_missing_teacher": heldout_missing_teacher,
        "train_benchmarks": sorted(TRAIN_BENCHMARKS),
        "heldout_benchmarks": sorted(HELDOUT_BENCHMARKS),
        "overlap_keys": len({row_key(r) for r in train} & {row_key(r) for r in heldout}),
    }
    return train, heldout, audit


def prompt_ids(tokenizer: Any, scoring: dict[str, Any], row: dict[str, Any]) -> list[int]:
    prompt_config = {
        "user_template": scoring["user_template"],
    }
    messages = [
        {"role": "system", "content": scoring["system_prompt"]},
        {"role": "user", "content": build_prompt_text(prompt_config, row)},
    ]
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    try:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    return extract_input_ids(encoded)


def candidate_token_ids(tokenizer: Any, scoring: dict[str, Any]) -> tuple[int, int]:
    yes = tokenizer.encode(scoring["candidate_yes"], add_special_tokens=False)
    no = tokenizer.encode(scoring["candidate_no"], add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1 or yes[0] == no[0]:
        raise RuntimeError(f"single distinct Yes/No tokens required; Yes={yes}, No={no}")
    return yes[0], no[0]


def forward_kwargs(model: Any) -> dict[str, int | bool]:
    parameters = inspect.signature(model.forward).parameters
    result: dict[str, int | bool] = {"use_cache": False}
    if "logits_to_keep" in parameters:
        result["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        result["num_logits_to_keep"] = 1
    return result


def soft_cross_entropy(probabilities: list[float], targets: list[float]) -> float:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("non-empty aligned probability and target vectors required")
    eps = 1e-12
    return sum(
        -(t * math.log(min(max(p, eps), 1 - eps)) + (1 - t) * math.log(min(max(1 - p, eps), 1 - eps)))
        for p, t in zip(probabilities, targets)
    ) / len(targets)


def teacher_metrics(probabilities: list[float], targets: list[float]) -> dict[str, float]:
    cross_entropy = soft_cross_entropy(probabilities, targets)
    entropy = soft_cross_entropy(targets, targets)
    return {
        "n": len(targets),
        "soft_cross_entropy": cross_entropy,
        "teacher_entropy": entropy,
        "kl_teacher_to_model": cross_entropy - entropy,
        "mean_squared_error": sum((p - t) ** 2 for p, t in zip(probabilities, targets)) / len(targets),
        "mean_absolute_error": sum(abs(p - t) for p, t in zip(probabilities, targets)) / len(targets),
    }


def sigmoid_tensor_delta(pair_logits: Any, yes_id: int, no_id: int) -> Any:
    return pair_logits[:, yes_id] - pair_logits[:, no_id]


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    import importlib.metadata

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_and_verify(path: Path, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"hash mismatch: {path}")
    return read_jsonl(path)


def stable_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


__all__ = [
    "HELDOUT_BENCHMARKS",
    "TRAIN_BENCHMARKS",
    "assign_benchmark_folds",
    "atomic_write_jsonl",
    "candidate_token_ids",
    "forward_kwargs",
    "keys_sha256",
    "load_config",
    "package_versions",
    "prompt_ids",
    "read_and_verify",
    "row_key",
    "sha256_file",
    "soft_cross_entropy",
    "stable_shuffle",
    "target_for_arm",
    "teacher_metrics",
    "validate_and_partition",
    "write_json",
]
