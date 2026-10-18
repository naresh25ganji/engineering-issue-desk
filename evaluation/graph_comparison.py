"""Compare hybrid and exact-entity graph source retrieval, without generation.

Both methods select at most six distinct sources. The hybrid method gets one
18-chunk candidate search; graph lookup uses exact API/reported-version seeds
and bounded two-hop expansion. This is not an answer-quality evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time

from evaluation.run import average, matches_family, normalized_url, percentile, retrieval_scores
from triage_desk import TriageDesk
from triage_desk.core import KINDS, _load_sources, _quote
from triage_desk.embeddings import MODEL_NAME
from triage_desk.knowledge_graph import EngineeringKnowledgeGraph


SOURCE_BUDGET = 6
BASELINE_CANDIDATE_BUDGET = 18
GRAPH_FILTER_CANDIDATE_BUDGET = 100
METHODS = ("hybrid_sources", "exact_entity_graph")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _load_datasets(paths: list[Path], sources: list[dict]) -> list[dict]:
    datasets = []
    for path in paths:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
            raise ValueError("Question datasets require a questions list")
        questions, seen = [], set()
        for question in document["questions"]:
            if not isinstance(question, dict) or not isinstance(question.get("id"), str) or not question["id"].strip():
                raise ValueError("Every question needs a nonempty text ID")
            if question["id"] in seen:
                raise ValueError("Question IDs must be unique within each dataset")
            seen.add(question["id"])
            if question.get("expected_behavior") != "evidence":
                continue
            if not isinstance(question.get("question"), str) or not question["question"].strip():
                raise ValueError("Every answerable question needs text")
            families = question.get("expected_families")
            if not isinstance(families, list) or not families or any(not isinstance(f, dict) for f in families):
                raise ValueError("Answerable questions need nonempty expected source families")
            if any(not any(matches_family(source, family) for source in sources) for family in families):
                raise ValueError(f"Expected source family for {question['id']} is absent from the corpus")
            kinds = question.get("source_kinds")
            if kinds is not None and (not isinstance(kinds, list) or any(not isinstance(k, str) or k not in KINDS for k in kinds)):
                raise ValueError("Question source_kinds must contain recognized source kinds")
            questions.append(question)
        datasets.append({"dataset_id": f"dataset-{len(datasets) + 1}", "path": str(path),
                         "name": document.get("name", str(path)), "sha256": hashlib.sha256(raw).hexdigest(),
                         "total_cases": len(document["questions"]), "answerable_cases": len(questions),
                         "excluded_boundary_cases": len(document["questions"]) - len(questions),
                         "questions": questions})
    return datasets


def _inspect_source(source: dict, question: str, *, candidate: dict | None = None) -> dict:
    candidate = candidate or {}
    return {key: source.get(key) for key in
            ("source_id", "title", "url", "kind", "snapshot_at", "source_commit", "version_scope")} | {
                "quote": candidate.get("quote") or _quote(source["text"], question),
                "representative_chunk_id": candidate.get("chunk_id"),
                "quote_role": "Representative literal passage for inspection; not proof of a diagnosis or a complete answer",
                "first_stage_score": candidate.get("score"),
            }


def _reference_checks(evidence: list[dict], source_map: dict) -> dict:
    failures = []
    for item in evidence:
        source = source_map.get(item["source_id"])
        if source is None:
            failures.append("Unknown source: " + item["source_id"])
        elif normalized_url(str(item["url"])) != normalized_url(source["url"]):
            failures.append("Source URL mismatch: " + item["source_id"])
        elif not isinstance(item["quote"], str) or not item["quote"] or item["quote"] not in source["text"]:
            failures.append("Quote is not a literal source span: " + item["source_id"])
    return {"passed": not failures, "checked_references": len(evidence), "failures": failures,
            "scope": "Source IDs, URLs and literal quote membership only; not entailment"}


def _retrieve(method: str, question: dict, desk, graph, source_map: dict) -> tuple[list[dict], dict]:
    text, kinds = question["question"], question.get("source_kinds")
    if method == "hybrid_sources":
        candidates = desk.search(text, mode="hybrid", source_kinds=kinds, top_k=BASELINE_CANDIDATE_BUDGET)
        selected, seen = [], set()
        for candidate in candidates:
            source_id = candidate.get("source_id")
            if source_id not in source_map:
                raise ValueError("Hybrid search returned an unknown source ID")
            if normalized_url(str(candidate.get("url", ""))) != normalized_url(source_map[source_id]["url"]):
                raise ValueError("Hybrid search returned a URL that does not match its source")
            if source_id in seen:
                continue
            if kinds is not None and source_map[source_id]["kind"] not in kinds:
                raise ValueError("Hybrid search violated the selected source-kind filter")
            selected.append(_inspect_source(source_map[source_id], text, candidate=candidate))
            seen.add(source_id)
            if len(selected) == SOURCE_BUDGET:
                break
        return selected, {"candidate_budget": BASELINE_CANDIDATE_BUDGET, "candidate_unit": "chunks",
                          "raw_candidate_count": len(candidates),
                          "raw_candidate_distinct_sources": len({item.get("source_id") for item in candidates}), "source_budget": SOURCE_BUDGET,
                          "no_entity_seed": None, "seed_node_ids": None}
    # Inspect known exact seeds through the graph's public inspection API. Only
    # API/version entities seed retrieval; explicit metadata browsing is excluded.
    seed_ids = [node for node in graph.explore(text, max_nodes=1)["seed_node_ids"]
                if node.startswith(("api:", "version:"))]
    candidate_budget = GRAPH_FILTER_CANDIDATE_BUDGET if kinds is not None else SOURCE_BUDGET
    ids = graph.search_sources(text, limit=candidate_budget)
    if len(ids) != len(set(ids)) or any(source_id not in source_map for source_id in ids):
        raise ValueError("Graph returned an unknown or repeated source ID")
    selected_ids = [source_id for source_id in ids if kinds is None or source_map[source_id]["kind"] in kinds][:SOURCE_BUDGET]
    selected = [_inspect_source(source_map[source_id], text) for source_id in selected_ids]
    return selected, {"candidate_budget": candidate_budget, "candidate_unit": "sources",
                      "raw_candidate_count": len(ids), "raw_candidate_distinct_sources": len(ids), "source_budget": SOURCE_BUDGET,
                      "filter_before_source_cap": True, "max_traversal_nodes": 600, "max_hops": 2,
                      "no_entity_seed": not seed_ids, "seed_node_ids": seed_ids}


def _summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["error"] is None]
    referenced = [row for row in completed if row["references"]["checked_references"]]
    misses = [row for row in completed if row["retrieval"]["family_recall_at_k"] < 1]
    no_seed = [row for row in completed if row["counts"]["no_entity_seed"] is True]
    return {
        "answerable_scheduled": len(rows), "answerable_completed": len(completed), "query_errors": len(rows) - len(completed),
        "hit_at_6": average([row["retrieval"]["hit_at_k"] for row in completed]),
        "source_family_recall_at_6": average([row["retrieval"]["family_recall_at_k"] for row in completed]),
        "gold_source_precision_at_6": average([row["retrieval"]["gold_source_precision_at_k"] for row in completed]),
        "mean_reciprocal_rank": average([row["retrieval"]["reciprocal_rank"] for row in completed]),
        "hit_at_6_counting_errors_as_misses": sum(row.get("retrieval", {}).get("hit_at_k") or 0 for row in rows) / len(rows) if rows else None,
        "queries_with_sources": sum(bool(row["source_ids"]) for row in completed),
        "source_return_coverage": average([bool(row["source_ids"]) for row in completed]),
        "mean_distinct_sources": average([len(row["source_ids"]) for row in completed]),
        "no_entity_seed_cases": len(no_seed),
        "no_entity_seed_fraction": len(no_seed) / len(completed) if completed and rows[0]["method"] == "exact_entity_graph" else None,
        "no_entity_seed_question_ids": [row["question_id"] for row in no_seed],
        "reference_integrity_queries": len(referenced),
        "reference_integrity_pass_rate": average([row["references"]["passed"] for row in referenced]),
        "retrieval_latency_p50_ms": percentile([row["retrieval_ms"] for row in completed], .5),
        "retrieval_latency_p95_ms": percentile([row["retrieval_ms"] for row in completed], .95),
        "miss_cases": [{"question_id": row["question_id"], "source_ids": row["source_ids"],
                        "no_entity_seed": row["counts"]["no_entity_seed"], "missed_families": row["missed_families"]} for row in misses],
        "error_cases": [{"question_id": row["question_id"], "error": row["error"]} for row in rows if row["error"]],
        "quality_denominator": "Completed expected_behavior=evidence cases, including empty exact-seed results as misses; errors shown separately and in failure-as-miss hit rate",
        "timing_denominator": "Completed answerable retrieval queries only, after one shared-model/cache warmup; excludes construction, warmup, generation and browser rendering",
        "answer_quality": None, "faithfulness": None, "boundary_behavior": None,
    }


def run_comparison(args: argparse.Namespace, *, desk_factory=TriageDesk,
                   graph_factory=EngineeringKnowledgeGraph) -> dict:
    corpus_raw = (args.data_dir / "sources.json").read_bytes()
    sources = _load_sources(args.data_dir)
    source_map = {source["source_id"]: source for source in sources}
    datasets = _load_datasets(args.questions or [Path("evaluation/questions.json")], sources)
    started = time.perf_counter()
    desk = desk_factory(args.data_dir, args.cache_dir)
    desk_construction_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    graph = graph_factory(sources)
    graph_construction_ms = (time.perf_counter() - started) * 1000
    warm_question = next((question for dataset in datasets for question in dataset["questions"]), None)
    warmups, results = {}, []
    for method in METHODS:
        started = time.perf_counter()
        warm_error = None
        try:
            if warm_question:
                _retrieve(method, warm_question, desk, graph, source_map)
        except Exception as exc:
            warm_error = f"{type(exc).__name__}: {exc}"
        warmups[method] = {"performed": warm_question is not None,
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "error": warm_error}
        for dataset in datasets:
            for question in dataset["questions"]:
                row = {"dataset_id": dataset["dataset_id"], "question_id": question["id"], "question": question["question"],
                       "method": method, "expected_behavior": "evidence", "expected_families": question["expected_families"],
                       "source_kinds": question.get("source_kinds"), "source_ids": [], "retrieved_sources": [], "error": None,
                       "retrieval_ms": None}
                if warm_error:
                    row["error"] = "Method initialization failed: " + warm_error
                else:
                    started = time.perf_counter()
                    try:
                        evidence, counts = _retrieve(method, question, desk, graph, source_map)
                        row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 3)
                        row.update(counts={**counts, "distinct_sources": len(evidence)}, source_ids=[item["source_id"] for item in evidence],
                                   retrieved_sources=evidence, retrieval=retrieval_scores(evidence, question["expected_families"]),
                                   references=_reference_checks(evidence, source_map),
                                   missed_families=[family for family in question["expected_families"] if not any(matches_family(item, family) for item in evidence)])
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["failed_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                results.append(row)
    dense = getattr(desk, "_dense", None)
    model_identity = {"configured_local_model": MODEL_NAME, "mode": "hybrid", "dimension_default": 384,
                      "desk_backend_class": f"{type(desk).__module__}.{type(desk).__qualname__}",
                      "dense_backend_class": f"{type(dense).__module__}.{type(dense).__qualname__}" if dense is not None else None,
                      "index_fingerprint": getattr(dense, "fingerprint", None), "index_origin": getattr(dense, "index_origin", None),
                      "fastembed_version": _package_version("fastembed"), "weights_sha256": None}
    identity_sha = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "report_kind": "graph_source_retrieval_comparison",
            "generated_at": datetime.now(timezone.utc).isoformat(), "corpus_sources": len(sources),
            "corpus_sha256": hashlib.sha256(corpus_raw).hexdigest(), "source_budget": SOURCE_BUDGET,
            "datasets": [{key: value for key, value in dataset.items() if key != "questions"} for dataset in datasets],
            "baseline_model": {**model_identity, "identity_metadata_sha256": identity_sha,
                               "hash_scope": "Declared model/backend/index metadata only; not a downloaded model-weight checksum"},
            "graph": graph.summary(), "initialization": {"desk_construction_ms": round(desk_construction_ms, 3),
                                                         "graph_construction_ms": round(graph_construction_ms, 3)},
            "warmup": warmups, "summary": {dataset["dataset_id"]: {method: _summary([
                row for row in results if row["dataset_id"] == dataset["dataset_id"] and row["method"] == method]) for method in METHODS} for dataset in datasets},
            "measurement_limits": [
                "Only expected_behavior=evidence cases are queried/scored. Boundary and vague-request behavior, answer quality, faithfulness and citation entailment are not evaluated.",
                "Both methods select at most six distinct sources. Baseline makes one hybrid top-18-chunk search then keeps the first representative chunk per source; the remaining candidates are not queried again.",
                "Graph uses exact known st.<identifier>/explicit reported-version seeds and bounded two-hop traversal (600-node cap), without a semantic, fuzzy or LLM fallback. No-entity queries legitimately return no graph sources and count as misses.",
                "Graph candidate budget is six without a source-kind filter, or 100 before applying a selected source-kind filter and the six-source cap. Candidate units/algorithms differ; this is a method comparison, not an isolated ablation.",
                "Graph edges are literal/provenance facts, not causality, duplicate, affected-version, compatibility or fix claims. Representative quotations are for source inspection, not evidence that a diagnosis or full answer was established.",
                "Source-family labels are incomplete relevance judgments. Precision is gold-family source precision, not passage precision; these metrics are not RAGAS or organizational-entity recall scores from the reference guide.",
                "Construction and one method warmup are separate from warm per-query timing. The baseline uses real local embeddings by default; unavailable embeddings produce explicit errors with no lexical substitution. Shared caches and a small local sample do not establish a production SLO.",
                "No generation, provider calls or environment-file loading occur in this runner. Local model initialization may download missing embedding files. Injected offline test adapters are test fixtures, not evidence that real-model quality was measured.",
            ], "results": results}


def render_report(report: dict) -> str:
    esc = lambda value: html.escape(str(value if value is not None else "Not scored"))
    fields = ("answerable_scheduled", "answerable_completed", "query_errors", "hit_at_6", "source_family_recall_at_6",
              "gold_source_precision_at_6", "mean_reciprocal_rank", "hit_at_6_counting_errors_as_misses", "queries_with_sources",
              "source_return_coverage", "mean_distinct_sources", "no_entity_seed_cases", "no_entity_seed_fraction",
              "reference_integrity_pass_rate", "retrieval_latency_p50_ms", "retrieval_latency_p95_ms")
    sections = []
    for dataset in report["datasets"]:
        summaries = report["summary"][dataset["dataset_id"]]
        header = "<tr><th>Measure</th>" + "".join(f"<th>{esc(method)}</th>" for method in METHODS) + "</tr>"
        body = "".join(f"<tr><th>{esc(field)}</th>" + "".join(f"<td>{esc(summaries[method][field])}</td>" for method in METHODS) + "</tr>" for field in fields)
        misses = "".join(f"<h3>{esc(method)} missed families/errors</h3><pre>{esc(json.dumps({'miss_cases': summaries[method]['miss_cases'], 'error_cases': summaries[method]['error_cases']}, indent=2))}</pre>" for method in METHODS)
        sections.append(f"<h2>{esc(dataset['name'])}</h2><p>{esc(dataset['answerable_cases'])} answerable cases scored; {esc(dataset['excluded_boundary_cases'])} boundary cases excluded. Dataset SHA-256: <code>{esc(dataset['sha256'])}</code></p><table>{header}{body}</table>{misses}")
    details = []
    for row in report["results"]:
        sources = "".join(f"<li><code>{esc(item['source_id'])}</code> · {esc(item['title'])} · {esc(item['kind'])}<br>{esc(item['url'])}<blockquote>{esc(item['quote'])}</blockquote><small>{esc(item['quote_role'])}</small></li>" for item in row["retrieved_sources"])
        detail = {key: row.get(key) for key in ("counts", "source_ids", "retrieval", "missed_families", "references", "error", "retrieval_ms")}
        details.append(f"<details><summary>{esc(row['dataset_id'])}/{esc(row['question_id'])} · {esc(row['method'])}</summary><p>{esc(row['question'])}</p><pre>{esc(json.dumps(detail,indent=2))}</pre><ol>{sources}</ol></details>")
    limits = "".join(f"<li>{esc(note)}</li>" for note in report["measurement_limits"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph and hybrid source-retrieval comparison</title><style>body{{font:15px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 18px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7df;padding:8px;text-align:left}}th{{background:#eef4f5}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{border:1px solid #ccd7df;padding:12px;margin:10px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>Graph and hybrid source-retrieval comparison</h1><p>At most six distinct sources per method, drawn from the same corpus. No answer generation or answer-quality measurement was performed.</p>
<p>Corpus SHA-256: <code>{esc(report['corpus_sha256'])}</code></p><h2>Model and graph identity</h2><pre>{esc(json.dumps({'baseline_model':report['baseline_model'],'graph':report['graph']},indent=2))}</pre>
{''.join(sections)}<h2>Construction and warmup</h2><pre>{esc(json.dumps({'initialization':report['initialization'],'warmup':report['warmup']},indent=2))}</pre>
<h2>Interpretation and denominators</h2><ul>{limits}</ul><p>Empty exact-seed results remain in the source-recall denominator. Errors are shown explicitly; their failure-as-miss hit rate is separate from completed-case means. Human and AI answer-quality scores are not assigned.</p>
<h2>Question sources</h2>{''.join(details)}</html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/retrieval"))
    parser.add_argument("--questions", type=Path, action="append", help="Repeat for each frozen dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/graph-comparison-results"))
    args = parser.parse_args()
    report = run_comparison(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, html_path = args.output_dir / "report.json", args.output_dir / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "json": str(json_path), "html": str(html_path)}, indent=2))
    return int(any(row["error"] or not row["references"]["passed"] for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
