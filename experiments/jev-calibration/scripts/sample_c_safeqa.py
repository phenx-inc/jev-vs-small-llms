#!/usr/bin/env python3
"""Create a pinned, deterministic, stratified C-SafeQA evaluation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


Stratum = tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalized_pair_sha256(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    response = row.get("response")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("missing or invalid prompt")
    if not isinstance(response, str):
        raise ValueError("missing or invalid response")
    value = normalize_text(prompt) + "\0" + normalize_text(response)
    return sha256_bytes(value.encode("utf-8"))


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def normalized_label(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {field!r}")
    return value.strip().casefold()


def display_label(row: dict[str, Any], field: str) -> str:
    return str(row[field]).strip()


def stratum_for(
    split: str,
    row: dict[str, Any],
    label_field: str,
    base_method_value: str,
) -> Stratum:
    method = row.get("method")
    if method is None or str(method).strip() == "":
        method = base_method_value
    return split, display_label(row, label_field), str(method)


def stratum_key(stratum: Stratum) -> str:
    return canonical_json(list(stratum))


def allocate_hamilton(counts: Counter[Stratum], target: int) -> dict[Stratum, int]:
    population = sum(counts.values())
    if target <= 0:
        raise ValueError("target_rows must be positive")
    if target > population:
        raise ValueError(f"target_rows={target} exceeds eligible population={population}")

    quotas: dict[Stratum, int] = {}
    remainders: list[tuple[float, str, Stratum]] = []
    for stratum, count in counts.items():
        exact = target * count / population
        floor = int(exact)
        quotas[stratum] = floor
        remainders.append((exact - floor, stratum_key(stratum), stratum))

    seats = target - sum(quotas.values())
    for _, _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1]))[:seats]:
        quotas[stratum] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("Hamilton allocation did not reach the requested target")
    return quotas


def rank_record(
    seed: int,
    source_path: str,
    source_line: int,
    row: dict[str, Any],
) -> tuple[int, str, str]:
    record_json = canonical_json(row)
    record_sha256 = sha256_bytes(record_json.encode("utf-8"))
    rank_material = "\0".join(
        [str(seed), source_path, str(source_line), record_json]
    ).encode("utf-8")
    rank_sha256 = sha256_bytes(rank_material)
    return int(rank_sha256, 16), rank_sha256, record_sha256


def summarize(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, int]]:
    counters = {field: Counter() for field in fields}
    for row in rows:
        for field in fields:
            value = row.get(field, "__missing__")
            if isinstance(value, (dict, list)):
                value = canonical_json(value)
            counters[field][str(value)] += 1
    return {
        field: dict(sorted(counter.items()))
        for field, counter in counters.items()
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def build_sample(config: dict[str, Any], data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = config["dataset"]
    sampling = config["sampling"]
    label_field = sampling["reference_label_field"]
    eligible = {str(label).casefold() for label in sampling["eligible_reference_labels"]}
    excluded = {str(label).casefold() for label in sampling["excluded_reference_labels"]}
    if eligible & excluded:
        raise ValueError("eligible and excluded label sets overlap")

    total_rows = 0
    eligible_rows = 0
    source_files: list[dict[str, Any]] = []
    strata: Counter[Stratum] = Counter()
    source_distributions: dict[str, Counter[str]] = {
        field: Counter() for field in ["split", label_field, "model", "method", "Label", "label"]
    }

    for source in dataset["files"]:
        relative = source["path"]
        split = source["split"]
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows = 0
        for line_number, row in iter_jsonl(path):
            del line_number
            file_rows += 1
            total_rows += 1
            label = normalized_label(row, label_field)
            if label not in eligible | excluded:
                raise ValueError(f"{relative}: unexpected {label_field}={row.get(label_field)!r}")
            for field in source_distributions:
                value: Any = split if field == "split" else row.get(field, "__missing__")
                if isinstance(value, (dict, list)):
                    value = canonical_json(value)
                source_distributions[field][str(value)] += 1
            if label in eligible:
                eligible_rows += 1
                strata[
                    stratum_for(split, row, label_field, sampling["base_method_value"])
                ] += 1
        if file_rows != source["expected_rows"]:
            raise ValueError(
                f"{relative}: expected {source['expected_rows']} rows, found {file_rows}"
            )
        source_files.append(
            {
                "path": relative,
                "split": split,
                "rows": file_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    if total_rows != dataset["expected_total_rows"]:
        raise ValueError(
            f"expected {dataset['expected_total_rows']} total rows, found {total_rows}"
        )

    target = int(sampling["target_rows"])
    quotas = allocate_hamilton(strata, target)
    candidates: list[
        tuple[int, str, str, int, str, dict[str, Any], Stratum, str]
    ] = []

    for source in dataset["files"]:
        relative = source["path"]
        split = source["split"]
        path = data_dir / relative
        for line_number, row in iter_jsonl(path):
            if normalized_label(row, label_field) not in eligible:
                continue
            stratum = stratum_for(split, row, label_field, sampling["base_method_value"])
            if quotas[stratum] == 0:
                continue
            rank_int, rank_sha256, record_sha256 = rank_record(
                int(sampling["seed"]), relative, line_number, row
            )
            candidates.append(
                (
                    rank_int,
                    rank_sha256,
                    relative,
                    line_number,
                    record_sha256,
                    row,
                    stratum,
                    normalized_pair_sha256(row),
                )
            )

    selected: list[dict[str, Any]] = []
    selected_strata: Counter[Stratum] = Counter()
    selected_pair_hashes: set[str] = set()
    duplicate_candidates_skipped = 0
    for (
        rank_int,
        rank_sha256,
        relative,
        line_number,
        record_sha256,
        row,
        stratum,
        pair_sha256,
    ) in sorted(candidates):
        del rank_int
        if selected_strata[stratum] >= quotas[stratum]:
            continue
        if pair_sha256 in selected_pair_hashes:
            duplicate_candidates_skipped += 1
            continue
        prompt = row["prompt"]
        normalized_prompt_sha256 = sha256_bytes(
            normalize_text(prompt).encode("utf-8")
        )
        output = dict(row)
        output["_c_safeqa_sample"] = {
            "sample_id": f"c-safeqa-{rank_sha256}",
            "seed": int(sampling["seed"]),
            "source_path": relative,
            "source_line": line_number,
            "source_split": stratum[0],
            "stratum": {
                "split": stratum[0],
                "judge_label": stratum[1],
                "method": stratum[2],
            },
            "rank_sha256": rank_sha256,
            "record_sha256": record_sha256,
            "prompt_group_sha256": normalized_prompt_sha256,
            "prompt_raw_sha256": sha256_bytes(prompt.encode("utf-8")),
            "normalized_prompt_response_sha256": pair_sha256,
        }
        selected.append(output)
        selected_strata[stratum] += 1
        selected_pair_hashes.add(pair_sha256)

    unfilled = {
        stratum_key(stratum): quotas[stratum] - selected_strata[stratum]
        for stratum in quotas
        if selected_strata[stratum] != quotas[stratum]
    }
    if unfilled:
        raise RuntimeError(f"could not fill deduplicated stratum quotas: {unfilled}")

    selected.sort(key=lambda row: row["_c_safeqa_sample"]["rank_sha256"])
    if len(selected) != target:
        raise AssertionError(f"expected {target} selected rows, found {len(selected)}")
    sample_ids = [row["_c_safeqa_sample"]["sample_id"] for row in selected]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate source identities in selected sample")
    if selected_strata != Counter({key: value for key, value in quotas.items() if value}):
        raise AssertionError("selected stratum counts do not match allocation")

    original_selected = [
        {key: value for key, value in row.items() if key != "_c_safeqa_sample"}
        for row in selected
    ]
    manifest = {
        "protocol_id": config["protocol_id"],
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "license": dataset["license"],
            "source_files": source_files,
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "excluded_rows": total_rows - eligible_rows,
        },
        "sampling": sampling,
        "selection_algorithm": {
            "allocation": "Hamilton largest remainder within split x judge_label x method",
            "tie_break": "lexicographic canonical JSON of the stratum",
            "within_stratum_rank": sampling["ranking"],
            "cross_stratum_duplicate_policy": sampling["deduplication"],
            "normalized_pair": "Unicode NFKC, casefold, collapse whitespace, prompt + NUL + response",
            "duplicate_candidates_skipped_during_selection": duplicate_candidates_skipped,
            "without_replacement": True,
        },
        "strata": [
            {
                "split": stratum[0],
                "judge_label": stratum[1],
                "method": stratum[2],
                "population": strata[stratum],
                "selected": quotas[stratum],
            }
            for stratum in sorted(strata, key=stratum_key)
        ],
        "population_distributions": {
            field: dict(sorted(counter.items()))
            for field, counter in source_distributions.items()
        },
        "sample_distributions": summarize(
            original_selected, [label_field, "model", "method", "Label", "label"]
        ),
        "sample_split_counts": dict(
            sorted(Counter(row["_c_safeqa_sample"]["source_split"] for row in selected).items())
        ),
        "sample_unique_record_hashes": len(
            {row["_c_safeqa_sample"]["record_sha256"] for row in selected}
        ),
        "sample_unique_exact_prompt_groups": len(
            {row["_c_safeqa_sample"]["prompt_group_sha256"] for row in selected}
        ),
        "sample_unique_normalized_prompt_response_pairs": len(selected_pair_hashes),
        "sample_rows": len(selected),
    }
    return selected, manifest


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected, manifest = build_sample(config, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.jsonl"
    write_jsonl_atomic(sample_path, selected)
    manifest["config_path"] = str(config_path)
    manifest["config_sha256"] = sha256_file(config_path)
    manifest["sample_path"] = str(sample_path)
    manifest["sample_sha256"] = sha256_file(sample_path)
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "sample_complete",
                "rows": len(selected),
                "sample_sha256": manifest["sample_sha256"],
                "out_dir": str(out_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
