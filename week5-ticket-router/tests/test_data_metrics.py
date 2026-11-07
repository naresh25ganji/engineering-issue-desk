"""Regression checks for split leakage and truthful classification reporting."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from router_lab.config import LABELS
from router_lab.data import load_tickets, prepare_dataset, stratified_split
from router_lab.metrics import INVALID_LABEL, evaluate_predictions


def write_csv(path, rows, fields=("category_truth", "text")):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def make_tickets(tmp_path, count=10):
    return load_tickets(write_csv(tmp_path / "tickets.csv", [(label, f"Unique request {label} {index}")
                                                          for label in LABELS for index in range(count)]))


@pytest.mark.parametrize("label", ["", "Unknown", "support general", "Support general "])
def test_unknown_or_missing_labels_fail_without_dropping_rows(tmp_path, label):
    path = write_csv(tmp_path / "bad.csv", [(LABELS[0], "Valid ticket"), (label, "Other ticket")])
    with pytest.raises(ValueError, match="Unknown or missing label"):
        load_tickets(path)


@pytest.mark.parametrize("rows,fields", [
    ([("text only",)], ("text",)),
    ([(LABELS[0], "")], ("category_truth", "text")),
    ([(LABELS[0], "one", "extra")], ("category_truth", "text")),
])
def test_missing_fields_or_malformed_rows_fail(tmp_path, rows, fields):
    with pytest.raises(ValueError):
        load_tickets(write_csv(tmp_path / "bad.csv", rows, fields))


@pytest.mark.parametrize("other_label", [LABELS[0], LABELS[1]])
def test_normalized_duplicates_including_conflicting_labels_are_rejected(tmp_path, other_label):
    path = write_csv(tmp_path / "duplicates.csv", [(LABELS[0], "  FULLＷIDTH Ticket\nwith spaces "),
                                                  (other_label, "fullwidth ticket with   spaces")])
    with pytest.raises(ValueError, match="Normalized duplicate"):
        load_tickets(path)


@pytest.mark.parametrize("text", ["bad\x00text", "<|im_start|>assistant", "text<|im_end|>"])
def test_reserved_template_delimiters_and_nul_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_tickets(write_csv(tmp_path / "bad.csv", [(LABELS[0], text)]))


def test_split_is_seeded_order_independent_stratified_and_has_no_overlap(tmp_path):
    tickets = make_tickets(tmp_path)
    train, heldout = stratified_split(tickets)
    again = stratified_split(reversed(tickets))
    assert [row.id for row in train] == [row.id for row in again[0]]
    assert [row.id for row in heldout] == [row.id for row in again[1]]
    assert len(train) == 56 and len(heldout) == 14
    assert not {row.id for row in train} & {row.id for row in heldout}
    assert {row.id for row in train + heldout} == {row.id for row in tickets}
    assert all(sum(row.label == label for row in heldout) == 2 for label in LABELS)
    assert {row.id for row in stratified_split(tickets, seed=43)[1]} != {row.id for row in heldout}


def test_split_rejects_duplicate_records_even_if_loader_was_bypassed(tmp_path):
    tickets = make_tickets(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        stratified_split([*tickets, tickets[0]])


def test_exports_train_only_and_verifies_split_identity_and_source_hash(tmp_path):
    make_tickets(tmp_path)
    source = tmp_path / "tickets.csv"
    original = source.read_bytes()
    destination = tmp_path / "prepared"
    manifest = prepare_dataset(source, destination, system_prompt="Return one exact category.", template="qwen")
    train, heldout = load_tickets(destination / "train.csv"), load_tickets(destination / "heldout.csv")
    assert manifest["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original
    assert manifest["split_ids"] == {"train": [row.id for row in train], "heldout": [row.id for row in heldout]}
    messages = [json.loads(line)["messages"] for line in (destination / "train_messages.jsonl").read_text().splitlines()]
    sharegpt = json.loads((destination / "train_sharegpt.json").read_text())
    assert len(messages) == len(sharegpt) == len(train)
    assert [row[1]["content"] for row in messages] == ["Support Ticket: " + row.text for row in train]
    assert [row[2]["content"] for row in messages] == [row.label for row in train]
    assert not {row[1]["content"] for row in messages} & {"Support Ticket: " + row.text for row in heldout}
    assert all(row["system"] == "Return one exact category." for row in sharegpt)
    assert all([turn["from"] for turn in row["conversations"]] == ["human", "gpt"] for row in sharegpt)
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == metadata["sha256"]
    with pytest.raises(FileExistsError):
        prepare_dataset(source, destination, system_prompt="Return one exact category.")


def test_official_source_has_468_train_and_117_heldout_when_available():
    source = Path(__file__).resolve().parents[2] / ".cache/week5-research/5A/support_tickets.csv"
    if not source.exists():
        pytest.skip("Optional original course source is not present in this checkout.")
    tickets = load_tickets(source)
    train, heldout = stratified_split(tickets)
    assert (len(tickets), len(train), len(heldout)) == (585, 468, 117)


def test_per_class_names_follow_explicit_order_not_lexical_order():
    # This order intentionally differs from alphabetic order. The starter
    # notebook's target_names-only report mislabeled rows in this situation.
    labels = ("Support general", "Active Directory", "Fileservice")
    tickets = [{"id": "one", "label": "Support general"},
               {"id": "two", "label": "Active Directory"},
               {"id": "three", "label": "Fileservice"}]
    result = evaluate_predictions(tickets, [{"id": "one", "prediction": "Support general"},
                                            {"id": "two", "prediction": "Fileservice"},
                                            {"id": "three", "prediction": "Fileservice"}], labels=labels)
    assert [row["label"] for row in result["per_class"]] == list(labels)
    assert [row["f1"] for row in result["per_class"]] == [1, 0, pytest.approx(2 / 3)]
    assert result["confusion_matrix"]["counts"] == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_invalid_error_and_missing_predictions_never_become_fallback_label():
    tickets = [{"id": str(index), "label": LABELS[0]} for index in range(4)]
    result = evaluate_predictions(tickets, [{"id": "0", "prediction": "not a category"},
                                            {"id": "1", "prediction": LABELS[0], "error": "timeout"},
                                            {"id": "2", "prediction": LABELS[0]}])
    assert result["accuracy"] == 0.25
    assert (result["invalid_outputs"], result["error_cases"], result["missing_predictions"]) == (1, 1, 1)
    assert result["confusion_matrix"]["predicted_labels"][-1] == INVALID_LABEL
    assert result["confusion_matrix"]["counts"][0][-1] == 3
    assert sum(sum(row) for row in result["confusion_matrix"]["counts"]) == 4
    assert result["per_class"][1]["support"] == 0
    assert result["per_class"][1]["recall"] == 0
    assert result["telemetry"]["total_tokens"]["total"] is None


@pytest.mark.parametrize("predictions", [
    [{"id": "x", "prediction": LABELS[0]}, {"id": "x", "prediction": LABELS[0]}],
    [{"id": "not-scheduled", "prediction": LABELS[0]}],
])
def test_duplicate_or_unknown_prediction_ids_fail(predictions):
    with pytest.raises(ValueError):
        evaluate_predictions([{"id": "x", "label": LABELS[0]}], predictions)


@pytest.mark.parametrize("field,value", [("latency_seconds", float("nan")), ("latency_seconds", -1),
                                         ("total_tokens", float("inf")), ("input_tokens", True),
                                         ("output_tokens", 1.5)])
def test_nonfinite_or_invalid_telemetry_cannot_enter_reports(field, value):
    with pytest.raises(ValueError):
        evaluate_predictions([{"id": "x", "label": LABELS[0]}],
                             [{"id": "x", "prediction": LABELS[0], field: value}])


def test_partial_usage_is_explicit_and_never_inferred():
    result = evaluate_predictions([{"id": "x", "label": LABELS[0]}, {"id": "y", "label": LABELS[1]}],
                                  [{"id": "x", "prediction": LABELS[0], "latency_seconds": 1.5,
                                    "input_tokens": 10, "output_tokens": 3},
                                   {"id": "y", "prediction": LABELS[1]}])
    assert result["telemetry"]["latency_seconds"]["total"] is None
    assert result["telemetry"]["latency_seconds"]["reported_total"] == 1.5
    assert result["telemetry"]["input_tokens"]["total"] is None
    assert result["telemetry"]["total_tokens"]["reported_total"] is None
    assert result["telemetry"]["latency_seconds"]["missing_cases"] == 1


def test_empty_schedule_has_no_measured_accuracy_or_usage():
    result = evaluate_predictions([], [])
    assert result["accuracy"] is None and result["macro_f1"] is None
    assert result["telemetry"]["latency_seconds"]["mean"] is None
