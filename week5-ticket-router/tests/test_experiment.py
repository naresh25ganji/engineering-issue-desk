"""Independent saved-result checks; no model weights or provider calls."""
from __future__ import annotations

from copy import deepcopy
import csv
import json

import pytest

from router_lab import experiment
from router_lab.config import LABELS, MODEL_ID, SYSTEM_PROMPT, TEMPLATE
from router_lab.data import prepare_dataset
from router_lab.metrics import evaluate_predictions
from router_lab.modeling import LocalRouter


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category_truth", "text"])
        writer.writerows((label, f"Independent request {label} example {i}")
                         for label in LABELS for i in range(10))
    destination = tmp_path / "prepared"
    prepare_dataset(source, destination, system_prompt=SYSTEM_PROMPT, template=TEMPLATE)
    return destination


def saved_run(prepared, variant="base", decoding="generate"):
    manifest, _, heldout = experiment.load_prepared(prepared)
    predictions = [{"id": row.id, "text": row.text, "label": row.label,
                    "prediction": row.label, "valid": True, "raw_output": row.label,
                    "error": None, "latency_seconds": .25,
                    "input_tokens": 100, "output_tokens": 2, "total_tokens": 102}
                   for row in heldout]
    run_manifest = experiment._manifest(prepared, manifest, variant, decoding, heldout)
    run_manifest["protocol"] = LocalRouter(MODEL_ID, decoding=decoding).prompt_metadata
    return {"kind": "model_evaluation", "variant": variant, "manifest": run_manifest,
            "predictions": predictions, "metrics": evaluate_predictions(heldout, predictions),
            "measurement_note": "Offline synthetic predictions for provenance regression tests."}


def persist(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_prepared_loader_rejects_changed_split_even_when_csv_is_still_valid(prepared):
    path = prepared / "heldout.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["label"] = next(label for label in LABELS if label != rows[0]["label"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="changed"):
        experiment.load_prepared(prepared)


def test_prepared_loader_requires_hash_entry_for_every_split(prepared):
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["heldout.csv"]
    persist(manifest_path, manifest)
    with pytest.raises(ValueError):
        experiment.load_prepared(prepared)


def test_valid_saved_run_and_missing_scheduled_cases_recompute(prepared):
    report = saved_run(prepared)
    experiment._validate_saved_run(report)
    report["predictions"] = report["predictions"][:-1]
    truths = report["manifest"]["heldout_tickets"]
    report["metrics"] = evaluate_predictions(truths, report["predictions"])
    experiment._validate_saved_run(report)
    assert report["metrics"]["scheduled_cases"] == 14
    assert report["metrics"]["missing_predictions"] == 1
    assert report["metrics"]["accuracy"] == pytest.approx(13 / 14)


@pytest.mark.parametrize("tamper", ["accuracy", "macro_f1", "case_correctness", "cropped_cases"])
def test_saved_scores_cannot_disagree_with_actual_predictions(prepared, tamper):
    report = saved_run(prepared)
    if tamper in {"accuracy", "macro_f1"}:
        report["metrics"][tamper] = 0.123
    elif tamper == "case_correctness":
        report["metrics"]["case_results"][0]["correct"] = False
    else:
        report["metrics"]["case_results"] = report["metrics"]["case_results"][:-1]
    with pytest.raises(ValueError):
        experiment._validate_saved_run(report)


@pytest.mark.parametrize("tamper", ["duplicate_id", "unknown_id", "wrong_truth", "changed_text", "wrong_hash"])
def test_saved_raw_rows_must_match_the_frozen_schedule(prepared, tamper):
    report = saved_run(prepared)
    row = report["predictions"][0]
    if tamper == "duplicate_id":
        report["predictions"].append(deepcopy(row))
    elif tamper == "unknown_id":
        row["id"] = "stale-run-ticket"
    elif tamper == "wrong_truth":
        row["label"] = next(label for label in LABELS if label != row["label"])
    elif tamper == "changed_text":
        row["text"] += " modified after the run"
    else:
        report["manifest"]["heldout_tickets"][0]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        experiment._validate_saved_run(report)


def test_saved_schedule_cannot_omit_ids_while_claiming_the_original_manifest(prepared):
    report = saved_run(prepared)
    report["manifest"]["heldout_tickets"] = report["manifest"]["heldout_tickets"][:-1]
    with pytest.raises(ValueError):
        experiment._validate_saved_run(report)


@pytest.mark.parametrize("key", ["dataset_manifest_sha256", "base_revision", "heldout_ids"])
def test_comparison_rejects_different_dataset_revision_or_schedule_before_output(prepared, tmp_path, key):
    base, tuned = saved_run(prepared), saved_run(prepared, variant="merged")
    if key == "heldout_ids":
        tuned["manifest"][key] = list(reversed(tuned["manifest"][key]))
    else:
        tuned["manifest"][key] = "different"
    destination = tmp_path / "comparison"
    with pytest.raises(ValueError):
        experiment.compare_reports(persist(tmp_path / "base.json", base),
                                   persist(tmp_path / "tuned.json", tuned), destination)
    assert not destination.exists()


def test_comparison_rejects_changed_truth_even_if_each_run_is_internally_consistent(prepared, tmp_path):
    base, tuned = saved_run(prepared), saved_run(prepared, variant="merged")
    changed = tuned["manifest"]["heldout_tickets"][0]
    changed["label"] = next(label for label in LABELS if label != changed["label"])
    tuned["predictions"][0]["label"] = changed["label"]
    tuned["metrics"] = evaluate_predictions(tuned["manifest"]["heldout_tickets"], tuned["predictions"])
    with pytest.raises(ValueError):
        experiment.compare_reports(persist(tmp_path / "base.json", base),
                                   persist(tmp_path / "tuned.json", tuned), tmp_path / "comparison")


def test_comparison_rejects_actual_protocol_metadata_drift(prepared, tmp_path):
    base, tuned = saved_run(prepared), saved_run(prepared, variant="merged")
    tuned["manifest"]["protocol"]["system_prompt"] += " extra instructions"
    with pytest.raises(ValueError):
        experiment.compare_reports(persist(tmp_path / "base.json", base),
                                   persist(tmp_path / "tuned.json", tuned), tmp_path / "comparison")


def test_mismatched_decoders_require_explicit_confounded_comparison(prepared, tmp_path):
    base = saved_run(prepared, decoding="letter_choice")
    tuned = saved_run(prepared, variant="merged", decoding="generate")
    base_path, tuned_path = persist(tmp_path / "base.json", base), persist(tmp_path / "tuned.json", tuned)
    with pytest.raises(ValueError):
        experiment.compare_reports(base_path, tuned_path, tmp_path / "comparison")
    report = experiment.compare_reports(base_path, tuned_path, tmp_path / "explicit-comparison",
                                        allow_protocol_difference=True)
    assert report["protocol_matched"] is False
    assert "protocol" in report["interpretation"]


def test_missing_cases_are_present_in_prediction_and_failure_exports(prepared, tmp_path, monkeypatch):
    report = saved_run(prepared)
    missing_id = report["predictions"][-1]["id"]
    report["predictions"].pop()
    report["metrics"] = evaluate_predictions(report["manifest"]["heldout_tickets"], report["predictions"])
    monkeypatch.setattr(experiment, "plot_confusion", lambda *_: None)
    destination = tmp_path / "report"
    experiment.save_report(report, destination)
    with (destination / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    with (destination / "failures.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert len(predictions) == report["metrics"]["scheduled_cases"]
    assert missing_id in {row["id"] for row in predictions}
    assert missing_id in {row["id"] for row in failures}
