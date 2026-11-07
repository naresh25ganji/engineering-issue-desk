"""Run frozen routing comparisons and save reports without hiding failed cases."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import time

from .config import LABELS, MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, TEMPLATE, prompt_metadata
from .data import load_tickets
from .metrics import evaluate_predictions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new output directory; results already exist: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_prepared(path):
    """Validate saved exports against the preparation manifest before use."""
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    required = {"train.csv", "heldout.csv", "train_sharegpt.json", "train_messages.jsonl", "dataset_info.json"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("Prepared manifest is missing required export fingerprints or contains unexpected files.")
    for name, info in manifest["files"].items():
        if Path(name).name != name or sha_file(path / name) != info["sha256"]:
            raise ValueError(f"Prepared file changed: {name}. Create a new dataset version.")
    if manifest["labels"] != list(LABELS) or manifest["system_prompt"] != SYSTEM_PROMPT or manifest["template"] != TEMPLATE:
        raise ValueError("Prepared labels, prompt, or template differs from this experiment.")
    train, heldout = load_tickets(path / "train.csv"), load_tickets(path / "heldout.csv")
    for label, rows in (("train", train), ("heldout", heldout)):
        if [row.id for row in rows] != manifest["split_ids"][label]:
            raise ValueError(f"Prepared {label} membership differs from the manifest.")
        count_key = "train_count" if label == "train" else "heldout_count"
        if len(rows) != manifest[count_key] or dict(Counter(row.label for row in rows)) != manifest["label_counts"][label]:
            raise ValueError(f"Prepared {label} counts differ from the manifest.")
    if {row.normalized_sha256 for row in train} & {row.normalized_sha256 for row in heldout}:
        raise ValueError("Training and held-out ticket texts overlap.")
    return manifest, train, heldout


def _pct(value):
    return "unknown" if value is None else f"{value:.1%}"


def save_report(report, directory):
    """Write machine-readable results, actual predictions, failure rows, and HTML."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "report.json", report)
    raw_rows = {row["id"]: row for row in report.get("predictions", [])}
    # Include every scheduled case, including predictions missing after interruption.
    rows = [{**raw_rows.get(row["id"], {}), "id": row["id"], "label": row["label"],
             "prediction": row["prediction"], "valid": row["status"] == "valid",
             "error": row["error"] or ("missing_prediction" if row["status"] == "missing" else None)}
            for row in report["metrics"]["case_results"]]
    fields = ["id", "text", "label", "prediction", "valid", "error", "raw_output", "latency_seconds", "input_tokens", "output_tokens"]
    for name, selected in (("predictions.csv", rows), ("failures.csv", [row for row in rows if row.get("error") or row.get("prediction") != row["label"]])):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    scores = report["metrics"]
    escaped = lambda value: html.escape(str(value))
    body = "".join("<tr>" + "".join(f"<td>{escaped(row.get(k, ''))}</td>" for k in ("id", "label", "prediction", "error")) + "</tr>" for row in rows if row.get("error") or row.get("prediction") != row["label"])
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Week 5 routing results</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #ddd;text-align:left}}th{{background:#edf1f4}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>{escaped(report['variant'])} routing evaluation</h1><p>Accuracy {_pct(scores['accuracy'])}; macro F1 {scores['macro_f1']:.3f}; invalid outputs {scores['invalid_outputs']}; errors {scores['error_cases']}; missing {scores['missing_predictions']}.</p>
<p>{escaped(report['measurement_note'])}</p><p>These scores use the supplied course labels; their semantic correctness has not been independently reviewed.</p>
<img src="confusion_matrix.png" alt="Confusion matrix including invalid outputs" style="max-width:100%"><h2>Failed predictions</h2><table><tr><th>Ticket ID</th><th>Expected</th><th>Predicted</th><th>Error</th></tr>{body}</table></body></html>'''
    (directory / "report.html").write_text(document, encoding="utf-8")
    plot_confusion(scores, directory / "confusion_matrix.png")


def plot_confusion(scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = scores["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(matrix["counts"], cmap="Blues")
    ax.set(xticks=range(len(matrix["predicted_labels"])), yticks=range(len(matrix["true_labels"])), xlabel="Predicted queue", ylabel="Supplied true queue", title="Support-ticket routing confusion matrix")
    ax.set_xticklabels([s.replace(" ", "\n") for s in matrix["predicted_labels"]], rotation=35, ha="right")
    ax.set_yticklabels(matrix["true_labels"])
    maximum = max(max(row) for row in matrix["counts"])
    for i, row in enumerate(matrix["counts"]):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > maximum * .55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _manifest(data_dir, prepared, variant, decoding, heldout):
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_sha256": sha_file(Path(data_dir) / "manifest.json"),
        "source_sha256": prepared["source_sha256"],
        "heldout_ids": [row.id for row in heldout], "labels": list(LABELS),
        "heldout_tickets": [{"id": row.id, "label": row.label, "text_sha256": row.text_sha256} for row in heldout],
        "heldout_cases": len(heldout), "full_heldout_cases": prepared["heldout_count"],
        "scope": "full held-out evaluation" if len(heldout) == prepared["heldout_count"] else "held-out smoke subset",
        "variant": variant, "decoding": decoding, "prompt": prompt_metadata(),
        "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
        "platform": platform.platform(),
    }


def run_baselines(data_dir, output_dir):
    """Fit majority and TF-IDF/logistic baselines on training tickets only."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import sklearn
    prepared, train, heldout = load_prepared(data_dir)
    destination = new_output(output_dir)
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    }
    summary = {}
    for variant, model in models.items():
        started = time.perf_counter()
        model.fit([row.text for row in train], [row.label for row in train])
        fit_seconds = time.perf_counter() - started
        predictions = []
        for row in heldout:
            started = time.perf_counter()
            predicted = str(model.predict([row.text])[0])
            predictions.append({"id": row.id, "text": row.text, "label": row.label, "prediction": predicted,
                                "valid": True, "error": None, "raw_output": predicted,
                                "latency_seconds": time.perf_counter() - started,
                                "input_tokens": None, "output_tokens": None, "total_tokens": None})
        report = {
            "kind": "cpu_baseline", "variant": variant,
            "manifest": {**_manifest(data_dir, prepared, variant, "sklearn", heldout), "sklearn_version": sklearn.__version__},
            "fit_seconds": fit_seconds, "predictions": predictions,
            "metrics": evaluate_predictions(heldout, predictions),
            "measurement_note": "CPU baseline; timing includes per-ticket text transformation and prediction, excludes fitting. Tokens and dollar cost are not measured.",
        }
        save_report(report, destination / variant)
        summary[variant] = {key: report["metrics"][key] for key in ("accuracy", "macro_f1", "scheduled_cases", "invalid_outputs")}
    write_json(destination / "summary.json", summary)
    return summary


def run_model(data_dir, output_dir, *, variant, model_path, adapter_path=None, decoding="generate", limit=None, device="auto", dtype="auto"):
    """Run one checkpoint against frozen inputs; save each completed prediction."""
    from .modeling import LocalRouter
    prepared, _, heldout = load_prepared(data_dir)
    if limit is not None:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be positive.")
        heldout = heldout[:limit]
    if variant not in ("base", "adapter", "merged"):
        raise ValueError("Model variant must be base, adapter, or merged.")
    if variant == "base" and (model_path != MODEL_ID or adapter_path):
        raise ValueError("The base run must use the pinned, unchanged Qwen checkpoint.")
    if variant == "adapter" and (model_path != MODEL_ID or not adapter_path):
        raise ValueError("The adapter run requires the pinned base plus an adapter.")
    if variant == "adapter":
        from .training import validate_adapter
        validate_adapter(adapter_path, sha_file(Path(data_dir) / "manifest.json"))
    if variant == "merged":
        proof = read_json(Path(model_path) / "router_model.json")
        if proof["base_model"] != MODEL_ID or proof["base_revision"] != MODEL_REVISION:
            raise ValueError("Merged model provenance does not match the baseline checkpoint.")
        if not proof.get("fresh_reload_verified"):
            raise ValueError("Run the fresh merge/reload verification first.")
        if proof.get("dataset_manifest_sha256") != sha_file(Path(data_dir) / "manifest.json"):
            raise ValueError("Merged model was trained using a different prepared dataset.")
        files = proof.get("model_files", {})
        actual_files = {path.name for path in Path(model_path).iterdir() if path.is_file()
                        and path.suffix in (".json", ".safetensors") and path.name != "router_model.json"}
        if set(files) != actual_files or not any(name.endswith(".safetensors") for name in files):
            raise ValueError("Merged model has missing or unexpected checkpoint files.")
        for name, expected_hash in files.items():
            if Path(name).name != name or sha_file(Path(model_path) / name) != expected_hash:
                raise ValueError("Merged model files changed after verification.")
    destination = new_output(output_dir)
    manifest = _manifest(data_dir, prepared, variant, decoding, heldout)
    manifest.update({"model_path": str(model_path), "adapter_path": str(adapter_path) if adapter_path else None, "device_requested": device, "dtype_requested": dtype})
    write_json(destination / "manifest.json", manifest)
    load_started = time.perf_counter()
    router = LocalRouter(model_path, adapter_path=adapter_path,
                         revision=MODEL_REVISION if model_path == MODEL_ID else None,
                         device=device, dtype=dtype, decoding=decoding)
    router.load()
    load_seconds = time.perf_counter() - load_started
    manifest["protocol"] = router.prompt_metadata
    manifest["runtime_device"] = str(router._model.device)
    manifest["runtime_dtype"] = str(router._model.dtype)
    write_json(destination / "manifest.json", manifest)
    predictions = []
    with (destination / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in heldout:
            try:
                record = router.predict(row.text)
                record["total_tokens"] = sum(record[key] for key in ("input_tokens", "output_tokens")) if all(record.get(key) is not None for key in ("input_tokens", "output_tokens")) else None
                record["error"] = None
            except Exception as exc:
                record = {"prediction": None, "valid": False, "raw_output": None, "error": type(exc).__name__, "latency_seconds": None}
            record.update({"id": row.id, "text": row.text, "label": row.label})
            predictions.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"{len(predictions)}/{len(heldout)} {variant}: {row.id}", flush=True)
            if record.get("error") in ("OutOfMemoryError",):
                break
    report = {"kind": "model_evaluation", "variant": variant, "manifest": manifest,
              "model_load_seconds": load_seconds, "predictions": predictions,
              "metrics": evaluate_predictions(heldout, predictions),
              "measurement_note": "Local checkpoint inference on the saved held-out tickets. Per-ticket latency excludes model loading; no hidden warmup. Invalid, error, and missing outputs count as incorrect. Hardware/dtype must be considered in latency comparisons. Token totals are model-computation counts, not provider billing. Dollar cost is unknown."}
    save_report(report, destination)
    return report


def _validate_saved_run(report):
    """Recompute saved scores and bind every prediction to the scheduled inputs."""
    manifest = report["manifest"]
    tickets = manifest["heldout_tickets"]
    if [row["id"] for row in tickets] != manifest["heldout_ids"]:
        raise ValueError("Saved run ticket membership/order differs from its manifest.")
    expected = {row["id"]: row for row in tickets}
    if not tickets or len(expected) != len(tickets):
        raise ValueError("Saved run needs nonempty unique scheduled tickets.")
    for row in report["predictions"]:
        ticket = expected.get(row["id"])
        if ticket is None or row.get("label") != ticket["label"]:
            raise ValueError("Saved prediction ID/truth differs from the scheduled ticket.")
        if hashlib.sha256(row.get("text", "").encode()).hexdigest() != ticket["text_sha256"]:
            raise ValueError("Saved prediction text differs from the scheduled input.")
    recomputed = evaluate_predictions(tickets, report["predictions"])
    if recomputed != report["metrics"]:
        raise ValueError("Saved run metrics are stale or differ from its actual predictions.")
    return recomputed


def compare_reports(base_path, tuned_path, output_dir, *, allow_protocol_difference=False):
    """Compare paired results only when dataset, case order, and checkpoint agree."""
    base, tuned = read_json(base_path), read_json(tuned_path)
    if base.get("kind") != "model_evaluation" or tuned.get("kind") != "model_evaluation":
        raise ValueError("Comparison needs two model-evaluation reports.")
    _validate_saved_run(base)
    _validate_saved_run(tuned)
    for key in ("dataset_manifest_sha256", "heldout_ids", "heldout_tickets", "labels", "base_model", "base_revision", "prompt"):
        if base["manifest"][key] != tuned["manifest"][key]:
            raise ValueError(f"Comparison mismatch: {key}")
    if base["variant"] != "base" or tuned["variant"] not in ("adapter", "merged"):
        raise ValueError("Compare an unchanged base run with an adapter or merged run.")
    protocol_equal = base["manifest"]["decoding"] == tuned["manifest"]["decoding"]
    if protocol_equal and base["manifest"].get("protocol") != tuned["manifest"].get("protocol"):
        raise ValueError("Actual inference prompt/decoder metadata differs.")
    if not protocol_equal and not allow_protocol_difference:
        raise ValueError("Decoding protocols differ. Use matched runs or explicitly label the handout comparison.")
    destination = new_output(output_dir)
    before, after = base["metrics"], tuned["metrics"]
    changes = []
    b_cases = {row["id"]: row for row in before["case_results"]}
    for row in after["case_results"]:
        prior = b_cases[row["id"]]
        status = "fixed" if row["correct"] and not prior["correct"] else "regression" if prior["correct"] and not row["correct"] else "unchanged"
        changes.append({"id": row["id"], "label": row["label"], "baseline": prior["prediction"], "tuned": row["prediction"], "change": status})
    report = {
        "kind": "paired_comparison", "protocol_matched": protocol_equal,
        "interpretation": "Same checkpoint, prompt and decoder; observed routing comparison." if protocol_equal else "Handout-style comparison: decoding differs, so the delta combines training and inference protocol effects.",
        "baseline_report": str(base_path), "tuned_report": str(tuned_path),
        "metrics": {"baseline": before, "tuned": after},
        "delta": {key: after[key] - before[key] for key in ("accuracy", "macro_f1")},
        "case_fixes": sum(row["change"] == "fixed" for row in changes),
        "case_regressions": sum(row["change"] == "regression" for row in changes), "cases": changes,
    }
    write_json(destination / "comparison.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([-.18, .82], [before["accuracy"], before["macro_f1"]], width=.36, label="Base")
    ax.bar([.18, 1.18], [after["accuracy"], after["macro_f1"]], width=.36, label="Tuned")
    ax.set(xticks=[0, 1], xticklabels=["Accuracy", "Macro F1"], ylim=(0, 1.05), title="Base versus fine-tuned support router")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=150)
    plt.close(fig)
    return report
