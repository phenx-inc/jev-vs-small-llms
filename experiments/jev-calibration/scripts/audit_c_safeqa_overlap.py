#!/usr/bin/env python3
"""Audit a frozen C-SafeQA sample for duplication and RLCD train leakage.

Raw gated text is read locally but never written to the report.  Findings contain
only source identifiers, text hashes, lengths, and lexical similarity scores.
The near-duplicate check is deliberately lexical (word 3-gram Jaccard and
containment); it is not a claim that semantically equivalent paraphrases are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--rlcd-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--containment-threshold", type=float, default=0.92)
    parser.add_argument("--min-near-tokens", type=int, default=12)
    parser.add_argument("--max-postings", type=int, default=200)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def string_leaves(value: Any) -> Iterable[tuple[str, str]]:
    """Yield stable JSON-pointer-like paths and string values."""
    if isinstance(value, str):
        yield "$", value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            for suffix, text in string_leaves(value[key]):
                yield f"$.{key}{suffix[1:]}", text
    elif isinstance(value, list):
        for index, item in enumerate(value):
            for suffix, text in string_leaves(item):
                yield f"$[{index}]{suffix[1:]}", text


def row_key(row: dict[str, Any], index: int) -> str:
    benchmark = row.get("benchmark")
    identifier = row.get("id")
    if benchmark is not None and identifier is not None:
        return f"{benchmark}:{identifier}"
    if identifier is not None:
        return str(identifier)
    return f"row-{index}"


def word_shingles(value: str) -> frozenset[str]:
    words = WORD_RE.findall(value)
    if len(words) < 3:
        return frozenset(words)
    return frozenset(" ".join(words[index : index + 3]) for index in range(len(words) - 2))


def safe_c_id(row: dict[str, Any], index: int) -> str:
    metadata = row.get("_c_safeqa_sample")
    if isinstance(metadata, dict) and metadata.get("sample_id"):
        return str(metadata["sample_id"])
    return f"c-safeqa-row-{index}"


def build_rlcd_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        state = row.get("state")
        leaves = [(path, normalize_text(text)) for path, text in string_leaves(state)]
        leaves = [(path, text) for path, text in leaves if text]
        candidates = list(leaves)
        if len(leaves) > 1:
            candidates.append(("$.__flattened__", " ".join(text for _, text in leaves)))
        for field, text in candidates:
            digest = text_sha256(text)
            unique_key = (row_key(row, index), digest)
            if unique_key in seen:
                continue
            seen.add(unique_key)
            documents.append(
                {
                    "row_key": row_key(row, index),
                    "field": field,
                    "text": text,
                    "sha256": digest,
                    "tokens": WORD_RE.findall(text),
                }
            )
    return documents


def build_c_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"C-SafeQA row {index} lacks string prompt/response")
        candidates = {
            "prompt": normalize_text(prompt),
            "response": normalize_text(response),
            "prompt_response": normalize_text(prompt + "\n" + response),
        }
        for field, text in candidates.items():
            documents.append(
                {
                    "row_key": safe_c_id(row, index),
                    "field": field,
                    "text": text,
                    "sha256": text_sha256(text),
                    "tokens": WORD_RE.findall(text),
                }
            )
    return documents


def exact_matches(
    c_documents: list[dict[str, Any]], rlcd_documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in rlcd_documents:
        by_hash[document["sha256"]].append(document)
    findings: list[dict[str, Any]] = []
    for source in c_documents:
        if not source["text"]:
            continue
        for target in by_hash.get(source["sha256"], []):
            findings.append(
                {
                    "c_safeqa_id": source["row_key"],
                    "c_safeqa_field": source["field"],
                    "c_safeqa_sha256": source["sha256"],
                    "c_safeqa_chars": len(source["text"]),
                    "rlcd_key": target["row_key"],
                    "rlcd_field": target["field"],
                    "rlcd_sha256": target["sha256"],
                    "rlcd_chars": len(target["text"]),
                }
            )
    return findings


def near_matches(
    c_documents: list[dict[str, Any]],
    rlcd_documents: list[dict[str, Any]],
    *,
    min_tokens: int,
    max_postings: int,
    jaccard_threshold: float,
    containment_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible_targets: list[dict[str, Any]] = []
    postings: dict[str, list[int]] = defaultdict(list)
    for target in rlcd_documents:
        if len(target["tokens"]) < min_tokens:
            continue
        shingles = word_shingles(target["text"])
        target["shingles"] = shingles
        target_index = len(eligible_targets)
        eligible_targets.append(target)
        for shingle in shingles:
            postings[shingle].append(target_index)

    usable_postings = {
        shingle: indices
        for shingle, indices in postings.items()
        if len(indices) <= max_postings
    }
    findings: list[dict[str, Any]] = []
    candidate_pairs = 0
    exact_hash_pairs = {
        (item["c_safeqa_sha256"], item["rlcd_sha256"])
        for item in exact_matches(c_documents, rlcd_documents)
    }
    for source in c_documents:
        if len(source["tokens"]) < min_tokens:
            continue
        source_shingles = word_shingles(source["text"])
        candidate_indices: set[int] = set()
        for shingle in source_shingles:
            candidate_indices.update(usable_postings.get(shingle, ()))
        candidate_pairs += len(candidate_indices)
        for target_index in sorted(candidate_indices):
            target = eligible_targets[target_index]
            if (source["sha256"], target["sha256"]) in exact_hash_pairs:
                continue
            intersection = len(source_shingles & target["shingles"])
            if not intersection:
                continue
            union = len(source_shingles | target["shingles"])
            jaccard = intersection / union
            containment = intersection / min(len(source_shingles), len(target["shingles"]))
            substring = (
                min(len(source["text"]), len(target["text"])) >= 80
                and (source["text"] in target["text"] or target["text"] in source["text"])
            )
            if (
                jaccard < jaccard_threshold
                and containment < containment_threshold
                and not substring
            ):
                continue
            findings.append(
                {
                    "c_safeqa_id": source["row_key"],
                    "c_safeqa_field": source["field"],
                    "c_safeqa_sha256": source["sha256"],
                    "c_safeqa_chars": len(source["text"]),
                    "rlcd_key": target["row_key"],
                    "rlcd_field": target["field"],
                    "rlcd_sha256": target["sha256"],
                    "rlcd_chars": len(target["text"]),
                    "word_3gram_jaccard": round(jaccard, 6),
                    "word_3gram_containment": round(containment, 6),
                    "normalized_substring": substring,
                }
            )
    findings.sort(
        key=lambda item: (
            -item["word_3gram_jaccard"],
            -item["word_3gram_containment"],
            item["c_safeqa_id"],
            item["rlcd_key"],
        )
    )
    diagnostics = {
        "rlcd_eligible_documents": len(eligible_targets),
        "total_unique_shingles": len(postings),
        "usable_unique_shingles": len(usable_postings),
        "candidate_pairs_compared": candidate_pairs,
    }
    return findings, diagnostics


def internal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_labels: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    prompt_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for row in rows:
        prompt = normalize_text(str(row["prompt"]))
        response = normalize_text(str(row["response"]))
        label = str(row.get("judge_label", "__missing__"))
        prompt_hash = text_sha256(prompt)
        pair_hash = text_sha256(prompt + "\0" + response)
        prompt_counts[prompt_hash] += 1
        pair_counts[pair_hash] += 1
        prompt_labels[prompt_hash].add(label)
        pair_labels[pair_hash].add(label)
    return {
        "rows": len(rows),
        "unique_normalized_prompts": len(prompt_counts),
        "unique_normalized_prompt_response_pairs": len(pair_counts),
        "duplicate_prompt_groups": sum(count > 1 for count in prompt_counts.values()),
        "duplicate_prompt_response_groups": sum(count > 1 for count in pair_counts.values()),
        "duplicate_prompt_response_rows_beyond_first": sum(
            count - 1 for count in pair_counts.values() if count > 1
        ),
        "prompt_groups_with_label_conflicts": sum(
            len(labels) > 1 for labels in prompt_labels.values()
        ),
        "prompt_response_groups_with_label_conflicts": sum(
            len(labels) > 1 for labels in pair_labels.values()
        ),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample).resolve()
    train_path = Path(args.rlcd_train).resolve()
    out_path = Path(args.out).resolve()
    sample_rows = read_jsonl(sample_path)
    train_rows = read_jsonl(train_path)
    c_documents = build_c_documents(sample_rows)
    rlcd_documents = build_rlcd_documents(train_rows)
    exact = exact_matches(c_documents, rlcd_documents)
    near, diagnostics = near_matches(
        c_documents,
        rlcd_documents,
        min_tokens=args.min_near_tokens,
        max_postings=args.max_postings,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
    )
    report = {
        "protocol": "c_safeqa_rlcd_overlap_audit_v1",
        "sample": {
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "rows": len(sample_rows),
        },
        "rlcd_train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "rows": len(train_rows),
        },
        "normalization": "Unicode NFKC, casefold, collapse whitespace",
        "internal": internal_audit(sample_rows),
        "cross_dataset": {
            "c_safeqa_documents": len(c_documents),
            "rlcd_documents": len(rlcd_documents),
            "exact_matches": exact,
            "exact_match_count": len(exact),
            "near_matches": near,
            "near_match_count": len(near),
            "near_method": {
                "shingles": "normalized word 3-grams",
                "minimum_tokens": args.min_near_tokens,
                "jaccard_threshold": args.jaccard_threshold,
                "containment_threshold": args.containment_threshold,
                "normalized_substring_min_chars": 80,
                "max_postings_per_shingle": args.max_postings,
                "scope_caveat": "Lexical audit only; semantic paraphrase leakage is not ruled out.",
            },
            "diagnostics": diagnostics,
        },
        "raw_text_written": False,
        "pass": len(exact) == 0 and len(near) == 0,
    }
    write_json_atomic(out_path, report)
    print(
        json.dumps(
            {
                "event": "overlap_audit_complete",
                "pass": report["pass"],
                "exact_matches": len(exact),
                "near_matches": len(near),
                "internal": report["internal"],
                "out": str(out_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
