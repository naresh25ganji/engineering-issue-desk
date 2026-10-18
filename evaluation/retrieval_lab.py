"""Compare literal fixed/structured chunks and actual optional reranking.

No LLM calls or environment-file reads. The CLI's dense and reranking paths may
initialize local models; prepare their caches before running in an offline setup.
Gold metrics use evaluation.run's existing source-family definitions unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time

from evaluation.run import average, matches_family, percentile, reference_integrity, retrieval_scores
from triage_desk.core import _load_sources
from triage_desk.retrieval_lab import RetrievalLab

CONFIGURATIONS = (
    ("fixed_hybrid", "fixed", False),
    ("structured_hybrid", "structured", False),
    ("structured_hybrid_rerank", "structured", True),
)


def _summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("error") is None]
    answerable = [row for row in rows if row["expected_behavior"] == "evidence"]
    scored = [row for row in completed if row["expected_behavior"] == "evidence"]
    boundaries = [row for row in rows if row["expected_behavior"] != "evidence"]
    referenced = [row for row in completed if row["references"]["checked_references"]]
    return {
        "queries_scheduled": len(rows), "queries_completed": len(completed), "query_errors": len(rows) - len(completed),
        "answerable_scheduled": len(answerable), "answerable_scored": len(scored), "answerable_failed": len(answerable) - len(scored),
        "boundary_queries": len(boundaries), "boundary_behavior_scored": False,
        "hit_at_k": average([row["retrieval"]["hit_at_k"] for row in scored]),
        "family_recall_at_k": average([row["retrieval"]["family_recall_at_k"] for row in scored]),
        "gold_source_precision_at_k": average([row["retrieval"]["gold_source_precision_at_k"] for row in scored]),
        "mean_reciprocal_rank": average([row["retrieval"]["reciprocal_rank"] for row in scored]),
        "hit_at_k_counting_failures_as_misses": sum(row.get("retrieval", {}).get("hit_at_k") or 0 for row in answerable) / len(answerable) if answerable else None,
        "reference_integrity_queries": len(referenced),
        "reference_integrity_pass_rate": average([row["references"]["passed"] for row in referenced]),
        "retrieval_latency_p50_ms": percentile([row["retrieval_ms"] for row in completed], .5),
        "retrieval_latency_p95_ms": percentile([row["retrieval_ms"] for row in completed], .95),
        "quality_denominator": "Successfully completed cases with expected_behavior=evidence; failures are also shown separately and in failure-as-miss hit rate.",
        "timing_denominator": "All successfully completed retrieval queries, including boundary prompts; no generation or behavior pipeline is run.",
        "answer_faithfulness": None, "citation_entailment": None,
    }


def run_comparison(args: argparse.Namespace, *, lab_factory=RetrievalLab) -> dict:
    source_bytes = (args.data_dir / "sources.json").read_bytes()
    sources = _load_sources(args.data_dir)
    source_map = {source["source_id"]: source for source in sources}
    lab = lab_factory(sources, args.cache_dir)
    paths = args.questions or [Path("evaluation/questions.json")]
    datasets = []
    for path in paths:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        questions = document["questions"][:args.max_questions] if args.max_questions else document["questions"]
        ids = [question["id"] for question in questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique within each dataset")
        for question in questions:
            for family in question.get("expected_families", []):
                if not any(matches_family(source, family) for source in sources):
                    raise ValueError(f"Expected family for {question['id']} is absent from the corpus")
        datasets.append({"dataset_id": f"dataset-{len(datasets) + 1}", "path": str(path), "name": document.get("name", str(path)),
                         "sha256": hashlib.sha256(raw).hexdigest(), "question_count": len(questions),
                         "question_kinds": dict(Counter(q["kind"] for q in questions)), "questions": questions})
    results, warmups, chunk_ids = [], {}, {key: {c["chunk_id"] for c in chunks} for key, chunks in lab.chunks_by_strategy.items()}
    warm_query = next((q["question"] for ds in datasets for q in ds["questions"] if q["expected_behavior"] == "evidence"), "Streamlit documentation")
    for name, strategy, rerank in CONFIGURATIONS:
        started = time.perf_counter()
        try:
            lab.search(warm_query, strategy=strategy, mode="hybrid", rerank=rerank, top_k=args.top_k, candidate_k=args.candidate_k)
            warmups[name] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "error": None}
        except Exception as exc:
            warmups[name] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "error": f"{type(exc).__name__}: {exc}"}
        for dataset in datasets:
            for question in dataset["questions"]:
                row = {"dataset_id": dataset["dataset_id"], "question_id": question["id"], "question": question["question"],
                       "configuration": name, "chunk_strategy": strategy, "mode": "hybrid", "rerank_requested": rerank,
                       "kind": question["kind"], "expected_behavior": question["expected_behavior"],
                       "expected_families": question.get("expected_families", []), "error": None}
                if warmups[name]["error"]:
                    row["error"] = "Configuration initialization failed: " + warmups[name]["error"]
                else:
                    try:
                        started = time.perf_counter()
                        evidence = lab.search(question["question"], strategy=strategy, mode="hybrid", rerank=rerank,
                                              top_k=args.top_k, candidate_k=args.candidate_k, source_kinds=question.get("source_kinds"))
                        row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 3)
                        row["retrieval"] = retrieval_scores(evidence, row["expected_families"])
                        row["references"] = reference_integrity(evidence, source_map, chunk_ids[strategy])
                        row["retrieved_sources"] = evidence
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                results.append(row)
    return {
        "schema_version": 1, "report_kind": "retrieval_lab_comparison", "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sources": len(sources), "corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "datasets": [{key: value for key, value in ds.items() if key != "questions"} for ds in datasets],
        "top_k": args.top_k, "candidate_k": args.candidate_k, "lab": lab.description(), "warmup": warmups,
        "summary": {ds["dataset_id"]: {name: _summary([r for r in results if r["dataset_id"] == ds["dataset_id"] and r["configuration"] == name])
                                      for name, _, _ in CONFIGURATIONS} for ds in datasets},
        "measurement_limits": [
            "All three configurations use the same corpus, gold source-family labels and hybrid ranking; chunk structure and optional actual reranking differ.",
            "Fixed chunks have word overlap; structured chunks preserve bounded blocks without duplicated overlap. This is a configuration comparison, not an isolated causal estimate of one parameter.",
            "Structured means deterministic sentence/paragraph/Markdown boundaries, not embedding-derived semantic breakpoints. Oversized blocks are explicitly partial and no missing text is invented.",
            "Reranking can only reorder retrieved candidates. Its scores are not calibrated confidence or proof that an answer is supported.",
            "Source-family precision uses incomplete gold labels and is not passage relevance precision. These are not RAGAS scores.",
            "No model-generated answers or abstention pipeline are run; answer faithfulness, entailment and boundary behavior remain unscored.",
            "Warmup includes initialization/cache loading. Configuration order shares caches; warm query timings are not independent cold-start or production SLO measurements.",
            "Initialization failures stay explicit and block that configuration; another retrieval or reranking method is never silently substituted.",
        ],
        "results": results,
    }


def render_report(report: dict) -> str:
    esc = lambda value: html.escape(str(value if value is not None else "Not scored"))
    sections = []
    fields = ("answerable_scheduled", "answerable_scored", "answerable_failed", "hit_at_k", "family_recall_at_k",
              "gold_source_precision_at_k", "mean_reciprocal_rank", "hit_at_k_counting_failures_as_misses",
              "reference_integrity_pass_rate", "retrieval_latency_p50_ms", "retrieval_latency_p95_ms", "query_errors")
    for dataset in report["datasets"]:
        summaries = report["summary"][dataset["dataset_id"]]
        header = "<tr><th>Measure</th>" + "".join(f"<th>{esc(name)}</th>" for name in summaries) + "</tr>"
        body = "".join(f"<tr><th>{esc(field)}</th>" + "".join(f"<td>{esc(values.get(field))}</td>" for values in summaries.values()) + "</tr>" for field in fields)
        sections.append(f"<h2>{esc(dataset['name'])}</h2><p>{esc(dataset['question_count'])} cases · dataset SHA-256 <code>{esc(dataset['sha256'])}</code></p><table>{header}{body}</table>")
    details = []
    for row in report["results"]:
        sources = "".join(f"<li><code>{esc(c['source_id'])}</code> · {esc(c['section'])} · retrieval {esc(c['score'])} · rerank {esc(c.get('rerank_score'))}<pre>{esc(c['quote'])}</pre></li>" for c in row.get("retrieved_sources", []))
        details.append(f"<details><summary>{esc(row['dataset_id'])}/{esc(row['question_id'])} · {esc(row['configuration'])}</summary><p>{esc(row['question'])}</p><p>Error: {esc(row.get('error'))}</p><pre>{esc(json.dumps(row.get('retrieval'), indent=2))}</pre><ol>{sources}</ol></details>")
    limits = "".join(f"<li>{esc(line)}</li>" for line in report["measurement_limits"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Issue Desk retrieval experiment</title><style>body{{font:15px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 18px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7df;padding:9px;text-align:left}}th{{background:#eef4f5}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{border:1px solid #ccd7df;padding:12px;margin:10px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>Retrieval configuration comparison</h1><p>Same corpus, gold metrics and query sets. No answer generation or LLM judging was performed.</p><p>Top {esc(report['top_k'])} output chunks · {esc(report['candidate_k'])} first-stage candidates · {esc(report['corpus_sources'])} sources</p>
<h2>Chunk and model settings</h2><pre>{esc(json.dumps(report['lab'],indent=2))}</pre>{''.join(sections)}
<h2>Initialization timing</h2><pre>{esc(json.dumps(report['warmup'],indent=2))}</pre><h2>Interpretation and denominators</h2><ul>{limits}</ul>
<p>Quality averages cover successfully completed answerable cases; errors and failure-as-miss hit rates remain visible. Boundary queries contribute timing but have no behavior/abstention score.</p><h2>Question evidence</h2>{''.join(details)}
<p>Corpus SHA-256: <code>{esc(report['corpus_sha256'])}</code></p></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/retrieval"))
    parser.add_argument("--questions", type=Path, action="append", help="Repeat for original and frozen validation datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/retrieval-lab-results"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-k", type=int, default=18)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 30 or not args.top_k <= args.candidate_k <= 100 or (args.max_questions is not None and args.max_questions < 1):
        parser.error("Require top-k 1–30, candidate-k between top-k and 100, and positive max-questions")
    report = run_comparison(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, html_path = args.output_dir / "report.json", args.output_dir / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "json": str(json_path), "html": str(html_path)}, indent=2))
    return int(any(row.get("error") for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
