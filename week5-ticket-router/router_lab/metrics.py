"""Scheduled-case classification scores with explicit invalid-output accounting."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .config import LABELS
from .data import Ticket


INVALID_LABEL = "__invalid__"
_TELEMETRY = ("latency_seconds", "input_tokens", "output_tokens", "total_tokens")


def _field(record: Ticket | Mapping, key: str) -> Any:
    return getattr(record, key) if isinstance(record, Ticket) else record.get(key)


def _telemetry_value(record: Mapping, field: str) -> int | float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a nonnegative finite number or None.")
    if field != "latency_seconds" and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or None.")
    return value


def _summarize_telemetry(rows: list[dict]) -> dict:
    summary = {}
    for field in _TELEMETRY:
        values = [row[field] for row in rows if row[field] is not None]
        complete = bool(rows) and len(values) == len(rows)
        item = {"reported_cases": len(values), "missing_cases": len(rows) - len(values),
                "reported_total": sum(values) if values else None,
                "total": sum(values) if complete else None}
        if field == "latency_seconds":
            ordered = sorted(values)
            mean = sum(values) / len(values) if values else None
            p95 = ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None
            item.update({"mean": mean if complete else None, "p95": p95 if complete else None,
                         "reported_mean": mean, "reported_p95": p95, "p95_method": "nearest rank"})
        summary[field] = item
    return summary


def evaluate_predictions(
    tickets: Iterable[Ticket | Mapping], predictions: Iterable[Mapping], *, labels=LABELS,
) -> dict:
    """Score all scheduled tickets, including absent/invalid/error predictions.

    A prediction record has ``id``, ``prediction`` (an exact label), optional
    ``error``, and optional numeric telemetry fields. Duplicate/unknown IDs fail
    rather than silently masking a stale or replayed run. Missing usage remains
    unknown and is never inferred from other token fields.
    """
    labels = tuple(labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Labels must be nonempty strings.")
    if len(labels) != len(set(labels)) or INVALID_LABEL in labels:
        raise ValueError("Labels must be unique and cannot use the invalid-output bucket.")
    scheduled = list(tickets)
    truths: dict[str, str] = {}
    for ticket in scheduled:
        if not isinstance(ticket, (Ticket, Mapping)):
            raise ValueError("Scheduled tickets must be Ticket records or mappings.")
        identity, label = _field(ticket, "id"), _field(ticket, "label")
        if not isinstance(identity, str) or not identity or identity in truths:
            raise ValueError("Scheduled ticket IDs must be nonempty and unique.")
        if label not in labels:
            raise ValueError(f"Unknown true label for {identity}: {label!r}")
        truths[identity] = label
    received = {}
    for record in predictions:
        if not isinstance(record, Mapping):
            raise ValueError("Predictions must be mappings.")
        identity = record.get("id")
        if not isinstance(identity, str) or identity not in truths:
            raise ValueError(f"Prediction has unknown scheduled ID: {identity!r}")
        if identity in received:
            raise ValueError(f"Duplicate prediction ID: {identity}")
        if record.get("error") is not None and not isinstance(record["error"], str):
            raise ValueError("Prediction error must be text or None.")
        received[identity] = {**record, **{field: _telemetry_value(record, field) for field in _TELEMETRY}}
    columns = (*labels, INVALID_LABEL)
    matrix = [[0 for _ in columns] for _ in labels]
    case_results = []
    counts = {"valid_predictions": 0, "invalid_outputs": 0, "error_cases": 0,
              "missing_predictions": 0, "correct_cases": 0}
    for identity, truth in truths.items():
        record = received.get(identity)
        prediction = record.get("prediction") if record else None
        if record is None:
            status = "missing"
            counts["missing_predictions"] += 1
        elif record.get("error"):
            status = "error"
            counts["error_cases"] += 1
        elif isinstance(prediction, str) and prediction in labels:
            status = "valid"
            counts["valid_predictions"] += 1
        else:
            status = "invalid"
            counts["invalid_outputs"] += 1
        effective = prediction if status == "valid" else INVALID_LABEL
        correct = effective == truth
        counts["correct_cases"] += int(correct)
        matrix[labels.index(truth)][columns.index(effective)] += 1
        case_results.append({"id": identity, "label": truth,
                             "prediction": prediction if isinstance(prediction, str) else None,
                             "scored_prediction": effective, "status": status, "correct": correct,
                             "error": record.get("error") if record else None,
                             **{field: record.get(field) if record else None for field in _TELEMETRY}})
    per_class = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "precision": precision, "recall": recall,
                          "f1": f1, "support": support, "predicted_count": predicted_count})
    return {
        "labels": list(labels), "scheduled_cases": len(scheduled), "received_predictions": len(received),
        **counts,
        "accuracy": counts["correct_cases"] / len(scheduled) if scheduled else None,
        "macro_f1": sum(row["f1"] for row in per_class) / len(labels) if scheduled else None,
        "macro_f1_policy": "unweighted mean across every configured label; zero division scores zero",
        "invalid_policy": "invalid outputs, errors, and missing predictions count as incorrect scheduled cases",
        "per_class": per_class,
        "confusion_matrix": {"true_labels": list(labels), "predicted_labels": list(columns), "counts": matrix},
        "telemetry": _summarize_telemetry(case_results),
        "case_results": case_results,
    }
