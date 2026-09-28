"""Dependency-free binary calibration metrics used by the Jev comparison."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


EPS = 1e-12


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def probabilities(deltas: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return [sigmoid(float(delta) / temperature) for delta in deltas]


def log_loss(labels: Sequence[int], probs: Sequence[float]) -> float:
    _validate(labels, probs)
    total = 0.0
    for label, prob in zip(labels, probs):
        p = min(1.0 - EPS, max(EPS, float(prob)))
        total -= label * math.log(p) + (1 - label) * math.log(1.0 - p)
    return total / len(labels)


def brier_score(labels: Sequence[int], probs: Sequence[float]) -> float:
    _validate(labels, probs)
    return sum((float(prob) - label) ** 2 for label, prob in zip(labels, probs)) / len(labels)


def accuracy(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> float:
    _validate(labels, probs)
    return sum((float(prob) >= threshold) == bool(label) for label, prob in zip(labels, probs)) / len(labels)


def balanced_accuracy(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> float:
    _validate(labels, probs)
    tp = tn = pos = neg = 0
    for label, prob in zip(labels, probs):
        pred = float(prob) >= threshold
        if label:
            pos += 1
            tp += int(pred)
        else:
            neg += 1
            tn += int(not pred)
    if not pos or not neg:
        return float("nan")
    return 0.5 * (tp / pos + tn / neg)


def f1_score(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> float:
    _validate(labels, probs)
    tp = fp = fn = 0
    for label, prob in zip(labels, probs):
        pred = float(prob) >= threshold
        tp += int(pred and label)
        fp += int(pred and not label)
        fn += int(not pred and label)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def auroc(labels: Sequence[int], probs: Sequence[float]) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    _validate(labels, probs)
    pairs = sorted((float(prob), int(label)) for label, prob in zip(labels, probs))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[k] = average_rank
        i = j
    n_pos = sum(label for _, label in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    pos_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def equal_width_bins(labels: Sequence[int], probs: Sequence[float], bins: int = 15) -> list[dict]:
    _validate(labels, probs)
    if bins < 1:
        raise ValueError("bins must be positive")
    rows: list[dict] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (label, float(prob))
            for label, prob in zip(labels, probs)
            if lower <= float(prob) < upper or (index == bins - 1 and float(prob) == 1.0)
        ]
        if not members:
            rows.append({"bin": index, "lower": lower, "upper": upper, "count": 0})
            continue
        mean_p = sum(prob for _, prob in members) / len(members)
        empirical = sum(label for label, _ in members) / len(members)
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_probability": mean_p,
                "empirical_positive_rate": empirical,
                "absolute_gap": abs(mean_p - empirical),
            }
        )
    return rows


def expected_calibration_error(labels: Sequence[int], probs: Sequence[float], bins: int = 15) -> float:
    rows = equal_width_bins(labels, probs, bins=bins)
    n = len(labels)
    return sum(row["count"] * row.get("absolute_gap", 0.0) for row in rows) / n


def adaptive_calibration_error(labels: Sequence[int], probs: Sequence[float], bins: int = 15) -> tuple[float, list[dict]]:
    _validate(labels, probs)
    if bins < 1:
        raise ValueError("bins must be positive")
    # Sort only by probability. More importantly, never split an exact score tie
    # between quantile bins: doing so lets label ordering create fictitious
    # calibration error for a constant predictor or quantized BF16 logits.
    ordered = sorted(
        ((float(prob), int(label)) for label, prob in zip(labels, probs)),
        key=lambda item: item[0],
    )
    rows: list[dict] = []
    n = len(ordered)
    start = 0
    index = 0
    while start < n:
        target_end = round((index + 1) * n / bins)
        end = max(start + 1, min(n, target_end))
        while end < n and ordered[end][0] == ordered[end - 1][0]:
            end += 1
        members = ordered[start:end]
        mean_p = sum(prob for prob, _ in members) / len(members)
        empirical = sum(label for _, label in members) / len(members)
        rows.append(
            {
                "bin": index,
                "lower_probability": members[0][0],
                "upper_probability": members[-1][0],
                "count": len(members),
                "mean_probability": mean_p,
                "empirical_positive_rate": empirical,
                "absolute_gap": abs(mean_p - empirical),
            }
        )
        start = end
        index += 1
    ace = sum(row["count"] * row["absolute_gap"] for row in rows) / n
    return ace, rows


def selective_accuracy(labels: Sequence[int], probs: Sequence[float], coverages: Iterable[float] = (0.5, 0.8, 0.9, 1.0)) -> list[dict]:
    _validate(labels, probs)
    ordered = sorted(
        zip(labels, probs), key=lambda item: abs(float(item[1]) - 0.5), reverse=True
    )
    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        count = max(1, math.ceil(len(ordered) * coverage))
        kept = ordered[:count]
        correct = sum((float(prob) >= 0.5) == bool(label) for label, prob in kept)
        rows.append(
            {
                "requested_coverage": coverage,
                "count": count,
                "actual_coverage": count / len(ordered),
                "accuracy": correct / count,
                "risk": 1.0 - correct / count,
                "minimum_confidence": min(abs(float(prob) - 0.5) * 2 for _, prob in kept),
            }
        )
    return rows


def high_confidence_errors(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.9) -> dict:
    _validate(labels, probs)
    selected = [
        (label, float(prob))
        for label, prob in zip(labels, probs)
        if float(prob) >= threshold or float(prob) <= 1.0 - threshold
    ]
    errors = sum((prob >= 0.5) != bool(label) for label, prob in selected)
    return {
        "threshold": threshold,
        "count": len(selected),
        "coverage": len(selected) / len(labels),
        "errors": errors,
        "error_rate": errors / len(selected) if selected else None,
    }


def fit_temperature(labels: Sequence[int], deltas: Sequence[float]) -> float:
    """Fit one positive temperature by golden-section search over log(T)."""
    if len(labels) != len(deltas) or not labels:
        raise ValueError("labels and deltas must be non-empty and have equal length")

    def objective(log_temperature: float) -> float:
        return log_loss(labels, probabilities(deltas, math.exp(log_temperature)))

    left, right = -6.0, 6.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(120):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    return math.exp((left + right) / 2.0)


def summarize(labels: Sequence[int], probs: Sequence[float], bins: int = 15) -> dict:
    _validate(labels, probs)
    ace, adaptive_bins = adaptive_calibration_error(labels, probs, bins=bins)
    return {
        "n": len(labels),
        "positive_rate": sum(labels) / len(labels),
        "mean_probability": sum(probs) / len(probs),
        "accuracy": accuracy(labels, probs),
        "balanced_accuracy": balanced_accuracy(labels, probs),
        "f1": f1_score(labels, probs),
        "auroc": auroc(labels, probs),
        "log_loss": log_loss(labels, probs),
        "brier": brier_score(labels, probs),
        "ece_15": expected_calibration_error(labels, probs, bins=bins),
        "adaptive_ece_15": ace,
        "high_confidence": high_confidence_errors(labels, probs),
        "selective": selective_accuracy(labels, probs),
        "equal_width_bins": equal_width_bins(labels, probs, bins=bins),
        "adaptive_bins": adaptive_bins,
    }


def _validate(labels: Sequence[int], probs: Sequence[float]) -> None:
    if len(labels) != len(probs) or not labels:
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    if any(label not in (0, 1, False, True) for label in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(float(prob)) or not 0.0 <= float(prob) <= 1.0 for prob in probs):
        raise ValueError("probabilities must be finite and in [0, 1]")
