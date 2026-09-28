#!/usr/bin/env python3
"""Run a bounded, resumable live Jev replay without exporting gated states.

The input comparison file contains gated benchmark text. It is read only long
enough to construct API requests. Output records contain identifiers, labels,
probabilities, hashes, usage, and latency, but never the raw state or question.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlcd_protocol import read_jsonl, render_state, sha256_file, write_json


DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_PRICE_PER_MTOK = 0.042


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Gated comparison.jsonl")
    parser.add_argument("--qwen-predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--api-style",
        choices=("systemone", "decide"),
        default="systemone",
        help="systemone discovers a model and sends it; decide uses /v1/decide and records the response model",
    )
    parser.add_argument("--model", default="auto")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--max-state-chars",
        type=int,
        default=4000,
        help="Documented decide-endpoint string-state limit; oversize rows are excluded, never truncated",
    )
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--price-per-mtok", type=float)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_secret(path: Path) -> str:
    values: dict[str, str] = {}
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value[:1] == value[-1:] and value[:1] in ("'", '"'):
                value = value[1:-1]
            values.setdefault(key, value)
    key = (
        os.environ.get("TYPESAFE_API_KEY")
        or os.environ.get("jev_api")
        or values.get("TYPESAFE_API_KEY")
        or values.get("jev_api")
        or ""
    )
    if not key:
        raise RuntimeError("Jev credential missing (expected TYPESAFE_API_KEY or jev_api)")
    return key


def prediction_key(row: dict[str, Any]) -> str:
    return f"{row['benchmark']}:{row['id']}"


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}:{row['benchmark']}:{row['id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_smoke_rows(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Select all benchmarks, both labels when available, then hash-rank fill."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["benchmark"])].append(row)
    minimum = 2 * len(grouped)
    if size < minimum:
        raise ValueError(f"smoke size {size} is too small for two rows x {len(grouped)} benchmarks")
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for benchmark in sorted(grouped):
        items = sorted(grouped[benchmark], key=lambda row: stable_rank(seed, row))
        for label in (0, 1):
            candidates = [row for row in items if int(row["label"]) == label]
            if candidates:
                row = candidates[0]
                selected.append(row)
                selected_keys.add(prediction_key(row))
        target = min(3, len(items))
        for row in items:
            if sum(item["benchmark"] == benchmark for item in selected) >= target:
                break
            if prediction_key(row) not in selected_keys:
                selected.append(row)
                selected_keys.add(prediction_key(row))
    remaining = sorted(
        (row for row in rows if prediction_key(row) not in selected_keys),
        key=lambda row: stable_rank(seed + 1, row),
    )
    selected.extend(remaining[: max(0, size - len(selected))])
    if len(selected) != size:
        raise RuntimeError(f"could select only {len(selected)} of {size} smoke rows")
    return sorted(selected, key=lambda row: (str(row["benchmark"]), str(row["id"])))


def selection_sha256(rows: list[dict[str, Any]]) -> str:
    keys = "\n".join(prediction_key(row) for row in rows)
    return hashlib.sha256(keys.encode("utf-8")).hexdigest()


def request_sha256(body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def http_json(
    *,
    url: str,
    key: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        # The early-access decide endpoint currently rejects urllib's default
        # Python-urllib user agent at its edge while accepting curl clients.
        "User-Agent": "curl/8.7.1",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("API response is not a JSON object")
        return int(response.status), result


def discover_models(base_url: str, key: str, timeout: float) -> list[dict[str, str]]:
    _, response = http_json(
        url=base_url.rstrip("/") + "/v1/models", key=key, timeout=timeout
    )
    models = response.get("models")
    if not isinstance(models, list) or not models:
        raise RuntimeError("GET /v1/models returned no models")
    clean: list[dict[str, str]] = []
    for item in models:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError("GET /v1/models returned invalid metadata")
        clean.append(
            {
                "name": item["name"],
                "description": str(item.get("description", "")),
                "release_date": str(item.get("release_date", "")),
            }
        )
    return clean


def select_model(models: list[dict[str, str]], requested: str) -> str:
    names = {item["name"] for item in models}
    if requested != "auto":
        if requested not in names:
            raise RuntimeError(f"requested model is unavailable: {requested}")
        return requested
    pinned = [item for item in models if "latest" not in item["name"].lower()]
    if pinned:
        return max(pinned, key=lambda item: (item["release_date"], item["name"]))["name"]
    if "jev-latest" in names:
        return "jev-latest"
    return max(models, key=lambda item: (item["release_date"], item["name"]))["name"]


class RateLimiter:
    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_at)
            self.next_at = scheduled + self.interval
        if scheduled > now:
            time.sleep(scheduled - now)


def validate_systemone_response(response: dict[str, Any]) -> tuple[float, int, int, str]:
    answer = (response.get("answers") or {}).get("misbehaves_noul")
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError("missing_noul_answer")
    probability = answer.get("noul")
    if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
        raise ValueError("invalid_noul_probability")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("noul_probability_out_of_range")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("missing_usage")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or input_tokens < 0:
        raise ValueError("invalid_input_tokens")
    if not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError("invalid_output_tokens")
    model = response.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("missing_response_model")
    return probability, input_tokens, output_tokens, model


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    price_per_mtok = (
        args.price_per_mtok
        if args.price_per_mtok is not None
        else (0.42 if args.api_style == "decide" else DEFAULT_PRICE_PER_MTOK)
    )
    if args.workers <= 0 or args.smoke_size <= 0 or args.max_input_tokens <= 0:
        raise ValueError("workers, smoke-size, and max-input-tokens must be positive")
    data_path = Path(args.data).resolve()
    qwen_path = Path(args.qwen_predictions).resolve()
    env_path = Path(args.env_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    data_sha = sha256_file(data_path)
    if args.expected_data_sha256 and data_sha != args.expected_data_sha256:
        raise RuntimeError("comparison data SHA-256 does not match the frozen registration")
    all_rows = read_jsonl(data_path)
    if args.api_style == "decide":
        eligible_rows = [
            row for row in all_rows if len(render_state(row["state"])) <= args.max_state_chars
        ]
        excluded_rows = [
            row for row in all_rows if len(render_state(row["state"])) > args.max_state_chars
        ]
    else:
        eligible_rows = list(all_rows)
        excluded_rows = []
    selected = (
        select_smoke_rows(eligible_rows, args.smoke_size, args.seed)
        if args.mode == "smoke"
        else list(eligible_rows)
    )
    qwen_rows = {prediction_key(row): row for row in read_jsonl(qwen_path)}
    expected_keys = {prediction_key(row) for row in selected}
    if not expected_keys <= set(qwen_rows):
        missing = sorted(expected_keys - set(qwen_rows))
        raise RuntimeError(f"Qwen predictions missing selected keys: {missing[:5]}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory is non-empty; use --resume: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    manifest_path = out_dir / "manifest.json"
    prior_status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.exists()
        else {}
    )
    prior_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.exists()
        else {}
    )
    prior_active_wall_seconds = float(
        prior_status.get("wall_seconds_total_active", prior_status.get("wall_seconds", 0.0))
    )
    authentication_utc = utc_now()
    write_json(
        status_path,
        {
            "state": "authenticating",
            "utc": authentication_utc,
            "prior_active_wall_seconds": prior_active_wall_seconds,
        },
    )
    key = load_secret(env_path)
    if args.api_style == "systemone":
        try:
            models = discover_models(args.base_url, key, args.timeout)
        except urllib.error.HTTPError as exc:
            write_json(
                status_path,
                {
                    "state": "authentication_failed",
                    "utc": utc_now(),
                    "http_status": int(exc.code),
                    "benchmark_requests_sent": 0,
                    "input_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            raise RuntimeError(
                f"GET /v1/models authentication failed with HTTP {exc.code}; "
                "no benchmark requests were sent"
            ) from None
        selected_model: str | None = select_model(models, args.model)
        endpoint = args.base_url.rstrip("/") + "/v1/systemone"
    else:
        if args.model != "auto":
            raise ValueError("--model is not supported by the decide endpoint")
        models = []
        selected_model = None
        endpoint = args.base_url.rstrip("/") + "/v1/decide"
    predictions_path = out_dir / "predictions.jsonl"
    raw_responses_path = out_dir / "raw_responses.jsonl"
    completed_rows = read_jsonl(predictions_path) if args.resume and predictions_path.exists() else []
    completed = {prediction_key(row): row for row in completed_rows if row.get("error") is None}
    if len(completed) != len([row for row in completed_rows if row.get("error") is None]):
        raise RuntimeError("duplicate successful prediction key in resume file")
    if any(row.get("requested_model") != selected_model for row in completed.values()):
        raise RuntimeError("resume model differs from the newly selected model")

    manifest = {
        "created_utc": prior_manifest.get("created_utc", authentication_utc),
        "resumed_utc": authentication_utc if args.resume else None,
        "resume_count": int(prior_manifest.get("resume_count", 0)) + int(args.resume),
        "mode": args.mode,
        "api_style": args.api_style,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "available_models": models,
        "requested_model_argument": args.model,
        "selected_model": selected_model,
        "comparison_path": str(data_path),
        "comparison_sha256": data_sha,
        "qwen_predictions_path": str(qwen_path),
        "qwen_predictions_sha256": sha256_file(qwen_path),
        "registered_examples": len(all_rows),
        "eligible_examples": len(eligible_rows),
        "excluded_oversize_examples": len(excluded_rows),
        "excluded_oversize_benchmarks": len(
            {row["benchmark"] for row in excluded_rows}
        ),
        "max_state_chars": args.max_state_chars if args.api_style == "decide" else None,
        "eligible_rendered_state_chars_total": sum(
            len(render_state(row["state"])) for row in eligible_rows
        ),
        "oversize_policy": (
            "Exclude rows whose losslessly rendered state exceeds the documented limit; never truncate."
            if args.api_style == "decide"
            else "Not applicable."
        ),
        "selected_examples": len(selected),
        "selected_benchmarks": len({row["benchmark"] for row in selected}),
        "selection_sha256": selection_sha256(selected),
        "seed": args.seed,
        "workers": args.workers,
        "rpm": args.rpm,
        "max_input_tokens": args.max_input_tokens,
        "price_per_million_input_tokens_usd": price_per_mtok,
        "credential_variable_names_supported": ["TYPESAFE_API_KEY", "jev_api"],
        "data_handling": "Gated state and question text remain in the source file and API request only; outputs contain hashes and scores, never text.",
    }
    write_json(manifest_path, manifest)
    pending = [row for row in selected if prediction_key(row) not in completed]
    prior_tokens = sum(int(row.get("input_tokens", 0)) for row in completed.values())
    if prior_tokens > args.max_input_tokens:
        raise RuntimeError("resumed token usage already exceeds max-input-tokens")
    write_json(
        status_path,
        {
            "state": "running",
            "utc": utc_now(),
            "selected": len(selected),
            "complete": len(completed),
            "remaining": len(pending),
        },
    )

    limiter = RateLimiter(args.rpm)
    lock = threading.Lock()
    total_tokens = [prior_tokens]
    success_count = [len(completed)]
    failure_count = [0]
    latencies = [float(row["latency_seconds"]) for row in completed.values()]
    stop = threading.Event()
    mode = "a" if predictions_path.exists() else "w"
    output = predictions_path.open(mode, encoding="utf-8", buffering=1)
    raw_mode = "a" if raw_responses_path.exists() else "w"
    raw_output = raw_responses_path.open(raw_mode, encoding="utf-8", buffering=1)

    def score(row: dict[str, Any]) -> dict[str, Any]:
        if stop.is_set():
            return {"skipped": True}
        state = render_state(row["state"]) if args.api_style == "decide" else row["state"]
        if args.api_style == "decide" and len(state) > args.max_state_chars:
            raise RuntimeError("oversize state escaped eligibility filter")
        body = {
            "state": state,
            "questions": {"misbehaves_noul": row["question"]},
        }
        if selected_model is not None:
            body["model"] = selected_model
        body_sha = request_sha256(body)
        error_code = None
        response: dict[str, Any] | None = None
        elapsed = 0.0
        attempts = 0
        for attempt in range(args.retries + 1):
            attempts = attempt + 1
            limiter.wait()
            started = time.monotonic()
            try:
                _, response = http_json(
                    url=endpoint,
                    key=key,
                    method="POST",
                    body=body,
                    timeout=args.timeout,
                )
                elapsed += time.monotonic() - started
                break
            except urllib.error.HTTPError as exc:
                elapsed += time.monotonic() - started
                error_code = f"http_{exc.code}"
                retryable = exc.code in (408, 409, 425, 429, 500, 502, 503, 504, 529)
            except (TimeoutError, urllib.error.URLError):
                elapsed += time.monotonic() - started
                error_code = "network_or_timeout"
                retryable = True
            except Exception as exc:  # response parsing/schema; never persist its text
                elapsed += time.monotonic() - started
                error_code = f"client_{type(exc).__name__}"
                retryable = False
            if not retryable or attempt >= args.retries:
                break
            time.sleep(min(2**attempt, 20) + random.random() * 0.25)

        qwen = qwen_rows[prediction_key(row)]
        qwen_probability = qwen.get("qwen_probability_yes")
        if qwen_probability is None:
            qwen_probability = qwen.get("model_probability_yes")
        if qwen_probability is None:
            raise RuntimeError("baseline prediction has no probability field")
        record: dict[str, Any] = {
            "id": str(row["id"]),
            "benchmark": str(row["benchmark"]),
            "failure_type": row.get("failure_type"),
            "benchmark_split": row.get("benchmark_split"),
            "target_model": row.get("target_model"),
            "variant": row.get("variant"),
            "label": int(row["label"]),
            "battery": row.get("battery"),
            "cached_jev_probability_yes": row.get("jev_probability_yes"),
            "qwen_probability_yes": float(qwen_probability),
            "prompt_group_sha256": row.get("prompt_group_sha256"),
            "sample_rank_sha256": row.get("sample_rank_sha256"),
            "requested_model": selected_model,
            "request_sha256": body_sha,
            "question_sha256": hashlib.sha256(
                json.dumps(row["question"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "state_chars": len(state) if isinstance(state, str) else None,
            "attempts": attempts,
            "latency_seconds": round(elapsed, 6),
            "error": error_code,
            "completed_utc": utc_now(),
        }
        if response is not None:
            response_json = json.dumps(
                response, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            record["raw_response_sha256"] = hashlib.sha256(
                response_json.encode("utf-8")
            ).hexdigest()
            record["_raw_response"] = response
            try:
                probability, input_tokens, output_tokens, response_model = validate_systemone_response(response)
            except ValueError as exc:
                record["error"] = str(exc)
            else:
                record.update(
                    {
                        "live_jev_probability_yes": probability,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "response_model": response_model,
                        "remaining_tokens_after": (response.get("usage") or {}).get(
                            "remaining_tokens"
                        ),
                        "server_latency_ms": (response.get("meta") or {}).get(
                            "latency_ms"
                        ),
                        "provider": (response.get("meta") or {}).get("provider"),
                        "demo": (response.get("meta") or {}).get("demo"),
                        "error": None,
                    }
                )
        return record

    started_wall = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(score, row) for row in pending]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                if record.get("skipped"):
                    continue
                with lock:
                    raw_response = record.pop("_raw_response", None)
                    if raw_response is not None:
                        raw_output.write(
                            json.dumps(
                                {
                                    "id": record["id"],
                                    "benchmark": record["benchmark"],
                                    "request_sha256": record["request_sha256"],
                                    "response_sha256": record["raw_response_sha256"],
                                    "received_utc": record["completed_utc"],
                                    "response": raw_response,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        raw_output.flush()
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    if record.get("error") is None:
                        total_tokens[0] += int(record["input_tokens"])
                        success_count[0] += 1
                        latencies.append(float(record["latency_seconds"]))
                        if total_tokens[0] > args.max_input_tokens:
                            stop.set()
                    else:
                        failure_count[0] += 1
                    processed = success_count[0] + failure_count[0]
                    if processed % 10 == 0 or processed == len(selected):
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "processed": processed,
                                    "selected": len(selected),
                                    "success": success_count[0],
                                    "failures": failure_count[0],
                                    "input_tokens": total_tokens[0],
                                    "estimated_cost_usd": round(
                                        total_tokens[0] / 1_000_000 * price_per_mtok, 6
                                    ),
                                }
                            ),
                            flush=True,
                        )
    finally:
        output.close()
        raw_output.close()

    all_output = read_jsonl(predictions_path)
    latest_by_key: dict[str, dict[str, Any]] = {}
    for row in all_output:
        if row.get("error") is None:
            latest_by_key[prediction_key(row)] = row
    missing_keys = expected_keys - set(latest_by_key)
    final_tokens = sum(int(row["input_tokens"]) for row in latest_by_key.values())
    failures = [row for row in all_output if row.get("error") is not None]
    state = "complete" if not missing_keys and final_tokens <= args.max_input_tokens else "stopped"
    wall_seconds_this_attempt = time.monotonic() - started_wall
    wall_seconds_total_active = prior_active_wall_seconds + wall_seconds_this_attempt
    status = {
        "state": state,
        "utc": utc_now(),
        "selected": len(selected),
        "successful_unique": len(latest_by_key),
        "missing": len(missing_keys),
        "failed_attempt_records": len(failures),
        "input_tokens": final_tokens,
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in latest_by_key.values()),
        "estimated_cost_usd": final_tokens / 1_000_000 * price_per_mtok,
        "wall_seconds": wall_seconds_total_active,
        "wall_seconds_this_attempt": wall_seconds_this_attempt,
        "wall_seconds_total_active": wall_seconds_total_active,
        "latency_p50_seconds": statistics.median(latencies) if latencies else None,
        "latency_p95_seconds": percentile(latencies, 0.95),
        "response_models": sorted({row["response_model"] for row in latest_by_key.values()}),
        "remaining_tokens_after": min(
            (
                int(row["remaining_tokens_after"])
                for row in latest_by_key.values()
                if isinstance(row.get("remaining_tokens_after"), int)
            ),
            default=None,
        ),
        "token_ceiling": args.max_input_tokens,
        "token_ceiling_exceeded": final_tokens > args.max_input_tokens,
        "raw_responses_saved": sum(
            1 for row in read_jsonl(raw_responses_path) if isinstance(row.get("response"), dict)
        ),
        "raw_responses_sha256": sha256_file(raw_responses_path),
    }
    write_json(status_path, status)
    print(json.dumps({"event": "finished", **status}), flush=True)
    if state != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
