"""Evaluate one complete investigation graph per question.

The default uses local embeddings without loading an environment file. Only
explicit hosted embeddings or --include-generation load project configuration.
Local dense/reranker models may need a prepared cache for fully offline runs.
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

from evaluation.run import (MODES, behavior_score, generation_summary,
                            matches_family, percentile, reference_integrity,
                            render_report as render_diagnostic_report,
                            retrieval_scores, summarize)
from triage_desk.workflow import InvestigationWorkflow, provider_call_summary


EVIDENCE_SCOPE = "Final selected evidence after workflow grading and core relevance checks; not raw first-stage retrieval."


def workflow_summary(rows: list[dict]) -> dict:
    completed = [row for row in rows if not row.get("error")]
    calls = [call for row in rows for call in row.get("analysis", {}).get("workflow", {}).get("provider_calls", [])]
    rollup = provider_call_summary(calls)
    unknown_rows = sum(bool(row.get("error")) and "analysis" not in row for row in rows)
    answerable = [row for row in rows if row["expected_behavior"] == "evidence"]
    traces = [row.get("analysis", {}).get("workflow", {}) for row in completed]
    latencies = [row["analysis_ms"] for row in completed]
    return {
        "workflow_provider_calls": rollup["attempted_calls"],
        "workflow_provider_call_records": rollup["recorded_calls"],
        "workflow_provider_calls_by_stage": rollup["calls_by_stage"],
        "workflow_token_usage": rollup["reported_usage"],
        "workflow_usage_complete": rollup["usage_complete"] and not unknown_rows,
        "workflow_usage_missing_calls_by_field": rollup["usage_missing_calls_by_field"],
        "workflow_call_accounting_unknown_error_rows": unknown_rows,
        "workflow_usage_note": rollup["usage_note"],
        "workflow_total_latency_p50_ms": percentile(latencies, .50),
        "workflow_total_latency_p95_ms": percentile(latencies, .95),
        "workflow_handoffs_required": sum(item.get("handoff", {}).get("required") is True for item in traces),
        "workflow_drafts_available": sum(item.get("handoff", {}).get("draft_available") is True for item in traces),
        "workflow_routes": dict(Counter(item.get("route", "unknown") for item in traces)),
        "workflow_evidence_status_coverage": sum(row.get("analysis", {}).get("status") == "evidence_found" for row in rows) / len(rows) if rows else None,
        "workflow_answerable_evidence_status_coverage": sum(row.get("analysis", {}).get("status") == "evidence_found" for row in answerable) / len(answerable) if answerable else None,
        "workflow_coverage_denominator": "All scheduled cases (or all scheduled expected-evidence cases), including errors. evidence_found is an observed status, not a faithfulness or correctness score.",
        "workflow_scheduled_cases": len(rows), "workflow_completed_cases": len(completed),
    }


def run_evaluation(args: argparse.Namespace, *, desk_factory=None, lab_factory=None,
                   graph_factory=None, workflow_factory=InvestigationWorkflow, dense_factory=None) -> dict:
    """Run once per question; no warmup or separate pre-grade search is hidden."""
    if desk_factory is None:
        from triage_desk import TriageDesk
        desk_factory = TriageDesk
    if lab_factory is None:
        from triage_desk.retrieval_lab import RetrievalLab
        lab_factory = RetrievalLab
    if graph_factory is None:
        from triage_desk.knowledge_graph import EngineeringKnowledgeGraph
        graph_factory = EngineeringKnowledgeGraph
    if args.embedding_provider == "nebius" and dense_factory is None:
        from triage_desk.hosted_embeddings import HostedDenseIndex
        dense_factory = HostedDenseIndex
    source_bytes = (args.data_dir / "sources.json").read_bytes()
    desk = desk_factory(args.data_dir, args.cache_dir)
    sources = desk.sources
    manifest = {source["source_id"]: source for source in sources}
    lab_options = {"dense_factory": dense_factory} if dense_factory is not None else {}
    lab = lab_factory(sources, args.cache_dir, **lab_options)
    graph = graph_factory(sources) if args.graph else None
    workflow = workflow_factory(desk, retrieval_lab=lab, knowledge_graph=graph)
    dataset_bytes = args.questions.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = dataset["questions"][:args.max_questions] if args.max_questions else dataset["questions"]
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    for question in questions:
        if question["expected_behavior"] not in {"evidence", "needs_details", "abstain"}:
            raise ValueError(f"Unknown expected behavior for {question['id']}")
        for family in question.get("expected_families", []):
            if not any(matches_family(source, family) for source in sources):
                raise ValueError(f"Expected family for {question['id']} is absent from the corpus")
    valid_chunk_ids = {chunk["chunk_id"] for chunk in lab.chunks_by_strategy[args.strategy]}
    rows = []
    for question in questions:
        row = {"question_id": question["id"], "question": question["question"],
               "kind": question.get("kind", "unspecified"), "mode": args.mode,
               "expected_behavior": question["expected_behavior"],
               "expected_families": question.get("expected_families", []),
               "expected_answer_checks": question.get("expected_answer_checks", []),
               "manual_review": {"faithfulness_0_to_2": None, "answer_relevance_0_to_2": None,
                                 "citation_entailment_0_to_2": None, "notes": ""},
               "retrieval_scope": EVIDENCE_SCOPE, "error": None}
        started = time.perf_counter()
        try:
            analysis = workflow.run(question["question"], mode=args.mode, strategy=args.strategy,
                source_kinds=question.get("source_kinds"), version=question.get("version", ""),
                include_generation=args.include_generation, rerank=args.rerank, use_graph=args.graph,
                history=question.get("history"))
            row["analysis_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["analysis"] = analysis
            returned_evidence = analysis.get("evidence", [])
            # A failed/negative grade may retain candidates for manual inspection.
            # They are not a post-grade selected set and must not earn gold hits.
            analyzed = any(stage.get("stage") == "analyze" and stage.get("decision") != "failed_closed"
                           for stage in analysis.get("workflow", {}).get("trace", []))
            evidence = returned_evidence if analyzed else []
            row["retrieval_ms"] = sum(stage.get("latency_ms", 0.) for stage in analysis.get("workflow", {}).get("trace", []) if stage.get("stage") == "retrieve")
            row["retrieval"] = retrieval_scores(evidence, row["expected_families"])
            row["behavior_passed"] = behavior_score(question["expected_behavior"], analysis)
            row["references"] = reference_integrity(returned_evidence, manifest, valid_chunk_ids, analysis.get("answer", ""))
            row["retrieved_sources"] = [{key: item.get(key) for key in
                ("citation_id", "source_id", "chunk_id", "title", "url", "score", "quote")} for item in evidence]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["analysis_ms"] = round((time.perf_counter() - started) * 1000, 3)
        rows.append(row)
    normal_summary = summarize(rows)
    normal_summary.update(workflow_summary(rows))
    generation = generation_summary(rows)
    return {"schema_version": 1, "report_kind": "investigation_workflow",
        "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_name": dataset.get("name"),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus_sha256": hashlib.sha256(source_bytes).hexdigest(), "corpus_sources": len(sources),
        "question_count": len(questions), "question_kinds": dict(Counter(row["kind"] for row in rows)),
        "top_k": 6, "analysis_top_k": 8 if args.graph else 6, "generation_requested": args.include_generation,
        "configuration": {"mode": args.mode, "strategy": args.strategy, "embedding_provider": args.embedding_provider,
                          "rerank": args.rerank, "graph": args.graph},
        "lab": lab.description(), "retrieval_scope": EVIDENCE_SCOPE,
        "generation_model_identity": {"requested_models": generation["generation_requested_models"],
            "returned_models": generation["generation_returned_models"], "endpoints": generation["generation_endpoints"],
            "source": "Observed answer-pipeline call records. Embedding and grading identities remain in workflow.provider_calls."},
        "timing_notes": "One complete graph run per case, no warmup and no second search. First-use indexing/model initialization is included in that case's full pipeline time; subsequent cases share caches. Retrieval time is only the graph's retrieve stage, while post-grade quality uses final selected evidence.",
        "warmup": {}, "summary": {args.mode: normal_summary},
        "measurement_limits": [EVIDENCE_SCOPE,
            "Automatic reference integrity measures known source/chunk identities and literal quotes, not entailment or faithfulness. Human quality fields remain unscored.",
            "Online directness grading and claim verification are AI judgments, not independent human validation. Offline directness grading is explicitly unscored.",
            "Gold source families are incomplete; source precision is a conservative source-selection proxy, not passage relevance precision or RAGAS.",
            "Workflow usage sums actual per-call embedding, grading, synthesis and verification records once. Missing applicable token counts remain explicit; no cost or token usage is inferred.",
            "Small sequential samples with shared caches are diagnostic measurements, not production latency guarantees. Local dense and reranker models require cached artifacts for fully offline use.",
            "An available handoff draft has not been posted and does not establish a diagnosis, duplicate or fix."],
        "results": rows}


def render_report(report: dict) -> str:
    """Reuse the existing answer/evidence renderer and append workflow telemetry."""
    document = render_diagnostic_report(report)
    esc = lambda obj: html.escape(json.dumps(obj, ensure_ascii=False, indent=2))
    summaries = {mode: {key: value for key, value in summary.items() if key.startswith("workflow_")}
                 for mode, summary in report["summary"].items()}
    details = "".join('<details><summary>' + html.escape(str(row["question_id"]))
        + '</summary><pre>' + esc(row.get("analysis", {}).get("workflow", {})) + '</pre></details>' for row in report["results"])
    extra = ('<h2>Complete workflow accounting</h2><p>' + html.escape(EVIDENCE_SCOPE)
             + '</p><p>Provider totals below include embeddings and directness grading as well as answer synthesis and verification, without adding answer-pipeline token totals twice.</p><pre>'
             + esc(summaries) + '</pre><h2>Graph routes, handoffs and call traces</h2>' + details)
    return document.replace('<h2>Question evidence</h2>', '<h2>Post-grade selected evidence</h2><p>' + html.escape(EVIDENCE_SCOPE) + '</p>', 1).replace('</html>', extra + '</html>')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/retrieval"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/workflow-results"))
    parser.add_argument("--embedding-provider", choices=("local", "nebius"), default="local")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--strategy", choices=("structured", "fixed"), default="structured")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--include-generation", action="store_true")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args(argv)
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("max-questions must be positive")
    if args.include_generation or args.embedding_provider == "nebius":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    report = run_evaluation(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, html_path = args.output_dir / "report.json", args.output_dir / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "json": str(json_path), "html": str(html_path)}, indent=2))
    return int(any(row.get("error") for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
