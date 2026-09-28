"""Protocol helpers for the cached-Jev RLCDAlignBench comparison."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


BATTERY_MODULES = (
    "battery_a_judge",
    "battery_bc_paired",
    "battery_d_rule",
    "battery_e",
    "battery_f_trajectory",
    "battery_refusal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_state(state: Any) -> str:
    """Match the upstream canonical object-to-text rendering."""
    if isinstance(state, str):
        return state
    if not isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False, indent=1)
    parts = []
    for key, value in state.items():
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
        parts.append(f"### {key}\n{body}")
    return "\n\n".join(parts)


def is_float(value: str | None) -> bool:
    try:
        float(value or "")
        return True
    except ValueError:
        return False


def eligible_main_rows(main_results_path: Path) -> list[dict[str, str]]:
    """The paper's 31-benchmark scope: generic AUROC exists and AUROC is nondegenerate."""
    with main_results_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if is_float(row.get("naive_noul_auroc"))
            and not str(row.get("status", "")).startswith("degenerate:")
        ]
    if len(rows) != 31:
        raise RuntimeError(f"expected 31 generic-Noul benchmarks, found {len(rows)}")
    return rows


def canonical_variants(variants_path: Path) -> dict[str, dict[str, str]]:
    with variants_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("is_canonical", "")).lower() != "true":
            continue
        benchmark = row["benchmark"]
        if benchmark in result:
            raise RuntimeError(f"multiple canonical variants for {benchmark}")
        result[benchmark] = row
    return result


def selected_release_paths(
    repo_files: Iterable[str], main_results_path: Path, variants_path: Path
) -> list[str]:
    """Select only canonical samples and their cached Jev response files."""
    files = set(repo_files)
    variants = canonical_variants(variants_path)
    selected = {"README.md", "benchmarks.csv", "file_map.csv", "variants.csv"} & files
    for row in eligible_main_rows(main_results_path):
        benchmark = row["leg"]
        variant = variants[benchmark]["variant"]
        sample_path = variants[benchmark].get("path")
        if not sample_path or sample_path not in files:
            raise RuntimeError(
                f"canonical variant file not found for {benchmark}/{variant}: {sample_path}"
            )
        selected.add(sample_path)
        prefix = f"jev/responses/{benchmark}/{variant}/"
        response_paths = sorted(
            path
            for path in files
            if path.startswith(prefix)
            and path.endswith(".jsonl")
            and "__text" not in Path(path).stem
        )
        if not response_paths:
            raise RuntimeError(f"cached object response not found for {benchmark}/{variant}")
        selected.update(response_paths)
        for response_path in response_paths:
            metric_path = response_path.replace("jev/responses/", "jev/metrics/").removesuffix(
                ".jsonl"
            ) + ".json"
            if metric_path not in files:
                raise RuntimeError(f"released metrics file not found for {response_path}")
            selected.add(metric_path)
    return sorted(selected)


def module_name_from_battery(battery: str) -> str:
    for module_name in BATTERY_MODULES:
        if battery == module_name or battery.startswith(module_name + "_"):
            return module_name
    raise ValueError(f"unknown upstream battery: {battery!r}")


def load_battery_module(battery: str, vendor_code_dir: Path) -> Any:
    module_name = module_name_from_battery(battery)
    code = str(vendor_code_dir.resolve())
    if code not in sys.path:
        sys.path.insert(0, code)
    return importlib.import_module(module_name)


def generic_question(sample: dict[str, Any], battery: str, vendor_code_dir: Path) -> dict[str, Any]:
    module = load_battery_module(battery, vendor_code_dir)
    battery_sample = dict(sample)
    battery_sample.setdefault("leg", sample.get("benchmark"))
    questions = module.questions(battery_sample)
    question = questions.get("misbehaves_noul")
    if not isinstance(question, dict) or question.get("type") != "noul":
        raise RuntimeError(
            f"{sample.get('benchmark')} / {battery} has no generic Noul question"
        )
    if not isinstance(question.get("instructions"), str):
        raise RuntimeError("generic Noul question has no instructions")
    return question


def extract_jev_probability(response: dict[str, Any]) -> float:
    answer = (response.get("answers") or {}).get("misbehaves_noul") or {}
    value = answer.get("noul")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"cached response has no finite misbehaves_noul probability: {response.get('id')}")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Jev probability outside [0,1]: {probability}")
    return probability


def choose_response_file(
    response_dir: Path, sample_ids: set[str]
) -> tuple[Path, list[dict[str, Any]]]:
    candidates: list[tuple[int, Path, list[dict[str, Any]]]] = []
    for path in sorted(response_dir.glob("*.jsonl")):
        if "__text" in path.stem:
            continue
        rows = read_jsonl(path)
        usable = {
            row.get("id")
            for row in rows
            if row.get("answers")
            and "misbehaves_noul" in (row.get("answers") or {})
        }
        overlap = len(sample_ids & usable)
        if overlap:
            candidates.append((overlap, path, rows))
    if not candidates:
        raise RuntimeError(f"no generic response file overlaps {response_dir}")
    best_overlap = max(overlap for overlap, _, _ in candidates)
    best = [(path, rows) for overlap, path, rows in candidates if overlap == best_overlap]
    if len(best) != 1:
        names = [path.name for path, _ in best]
        raise RuntimeError(
            f"multiple generic response files tie at {best_overlap} canonical IDs in "
            f"{response_dir}: {names}"
        )
    return best[0]


def build_comparison_rows(
    *,
    release_dir: Path,
    vendor_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    main_results = vendor_root / "results" / "main_results.csv"
    variants_path = vendor_root / "data" / "variants.csv"
    variants = canonical_variants(variants_path)
    comparison: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for result_row in eligible_main_rows(main_results):
        benchmark = result_row["leg"]
        variant = variants[benchmark]["variant"]
        sample_path = release_dir / variants[benchmark]["path"]
        if not sample_path.is_file():
            raise RuntimeError(f"canonical variant file is missing for {benchmark}: {sample_path}")
        all_samples = read_jsonl(sample_path)
        samples = [sample for sample in all_samples if sample.get("label") in (0, 1)]
        sample_ids = {str(sample["id"]) for sample in samples}
        if len(sample_ids) != len(samples):
            raise RuntimeError(f"duplicate sample IDs in {sample_path}")
        response_dir = release_dir / "jev" / "responses" / benchmark / variant
        response_path, responses = choose_response_file(response_dir, sample_ids)
        by_id = {str(row["id"]): row for row in responses}
        if len(by_id) != len(responses):
            raise RuntimeError(f"duplicate response IDs in {response_path}")
        battery_values = {str(row.get("battery")) for row in responses}
        if len(battery_values) != 1:
            raise RuntimeError(f"mixed battery IDs in {response_path}: {battery_values}")
        battery = next(iter(battery_values))
        metric_path = (
            release_dir
            / "jev"
            / "metrics"
            / benchmark
            / variant
            / f"{response_path.stem}.json"
        )
        if not metric_path.is_file():
            raise RuntimeError(f"released metrics file is missing: {metric_path}")
        released_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        generic_metrics = (released_metrics.get("strategies") or {}).get("naive:noul") or {}
        released_auroc = generic_metrics.get("auroc")
        published_auroc = float(result_row["naive_noul_auroc"])
        if not isinstance(released_auroc, (int, float)) or abs(
            float(released_auroc) - published_auroc
        ) > 0.000051:
            raise RuntimeError(
                f"released/published generic AUROC mismatch for {benchmark}: "
                f"metrics={released_auroc}, table={published_auroc}"
            )
        paired = 0
        missing_cached_ids: list[str] = []
        for sample in samples:
            label = sample.get("label")
            if label not in (0, 1):
                raise RuntimeError(f"non-binary label in canonical release: {sample.get('id')}")
            question = generic_question(sample, battery, vendor_root / "code")
            response = by_id.get(str(sample["id"]))
            jev_probability = None
            if not response or "misbehaves_noul" not in (response.get("answers") or {}):
                missing_cached_ids.append(str(sample["id"]))
            else:
                if response.get("label") not in (None, label):
                    raise RuntimeError(f"sample/response label mismatch: {sample.get('id')}")
                jev_probability = extract_jev_probability(response)
                paired += 1
            comparison.append(
                {
                    "id": str(sample["id"]),
                    "benchmark": benchmark,
                    "failure_type": sample.get("failure_type") or result_row["axis"],
                    "benchmark_split": result_row["split"],
                    "target_model": sample.get("target_model") or result_row["target_model"],
                    "variant": variant,
                    "label": int(label),
                    "state": sample["state"],
                    "question": question,
                    "jev_probability_yes": jev_probability,
                    "battery": battery,
                }
            )
        expected = int(result_row["n"])
        if len(samples) != expected:
            raise RuntimeError(
                f"{benchmark} row count changed: paper={expected}, release={len(samples)}"
            )
        sources.append(
            {
                "benchmark": benchmark,
                "variant": variant,
                "canonical_samples": len(samples),
                "variant_rows_total": len(all_samples),
                "variant_rows_unlabeled": len(all_samples) - len(samples),
                "paired_samples": paired,
                "missing_cached_responses": len(missing_cached_ids),
                "missing_cached_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(missing_cached_ids)).encode("utf-8")
                ).hexdigest(),
                "sample_path": str(sample_path.relative_to(release_dir)),
                "sample_sha256": sha256_file(sample_path),
                "response_path": str(response_path.relative_to(release_dir)),
                "response_sha256": sha256_file(response_path),
                "response_rows_total": len(responses),
                "response_rows_outside_canonical_denominator": len(set(by_id) - sample_ids),
                "battery": battery,
                "metrics_path": str(metric_path.relative_to(release_dir)),
                "metrics_sha256": sha256_file(metric_path),
                "released_metrics_status": released_metrics.get("status"),
                "released_metrics_generic": generic_metrics,
                "published_jev_auroc": published_auroc,
            }
        )
    if len({row["id"] for row in comparison}) != len(comparison):
        raise RuntimeError("sample IDs are not globally unique across comparison benchmarks")
    manifest = {
        "benchmarks": len(sources),
        "canonical_examples": sum(source["canonical_samples"] for source in sources),
        "paired_examples": sum(source["paired_samples"] for source in sources),
        "missing_cached_responses": sum(
            source["missing_cached_responses"] for source in sources
        ),
        "sources": sources,
        "vendor_commit_expected": "473b72f290234a49484c4d89c03aababcd74f758",
    }
    return comparison, manifest


def question_text(question: dict[str, Any]) -> str:
    text = question["instructions"]
    criteria = question.get("criteria")
    if criteria:
        text += "\n\nCriteria:\n" + json.dumps(criteria, ensure_ascii=False, indent=1)
    return text


def build_prompt_text(config: dict[str, Any], row: dict[str, Any]) -> str:
    return config["user_template"].format(
        state=render_state(row["state"]),
        question=question_text(row["question"]),
    )


def bootstrap_median_ci(values: list[float], seed: int, draws: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choice(values) for _ in values) for _ in range(draws)
    )
    return [medians[int(0.025 * draws)], medians[int(0.975 * draws)]]


def promise_verdict(
    *, median_auroc_delta_jev_minus_qwen: float,
    median_log_loss_gain_qwen_minus_jev: float,
    median_brier_gain_qwen_minus_jev: float,
    auroc_margin: float = 0.02,
    calibration_margin: float = 0.02,
) -> str:
    if median_auroc_delta_jev_minus_qwen >= auroc_margin:
        return "clear_promise"
    calibration_win = max(
        median_log_loss_gain_qwen_minus_jev, median_brier_gain_qwen_minus_jev
    ) >= calibration_margin
    if median_auroc_delta_jev_minus_qwen >= -0.01 and calibration_win:
        return "clear_promise"
    if median_auroc_delta_jev_minus_qwen < -auroc_margin and not calibration_win:
        return "clear_no_promise"
    return "mixed"
