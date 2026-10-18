"""Run the same source-family questions against each retrieval mode.

The automatic scores measure retrieval and reference integrity. They deliberately
do not claim that an optional language model's answers are faithful or relevant.
Those judgments have their own empty fields in the exported review worksheet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any
from urllib.parse import unquote, urlsplit


MODES = ("keyword", "semantic", "hybrid")
ABSTAIN_STATUSES = {"insufficient_evidence", "needs_details"}
AI_REVIEW_FIELDS = ("faithfulness_0_to_2", "answer_relevance_0_to_2", "citation_entailment_0_to_2")


def normalized_url(value: str) -> str:
    """Ignore tracking and trailing slashes, preserve version-bearing paths."""
    parsed = urlsplit(value)
    return (parsed.netloc.lower() + unquote(parsed.path)).rstrip("/")


def matches_family(evidence: dict[str, Any], family: dict[str, Any]) -> bool:
    if evidence.get("source_id") in family.get("source_ids", []):
        return True
    url = normalized_url(str(evidence.get("url", "")))
    return any(url == normalized_url(item) for item in family.get("urls", []))


def retrieval_scores(
    evidence: list[dict[str, Any]], families: list[dict[str, Any]]
) -> dict[str, float | None]:
    """Score distinct sources, preventing many chunks from inflating coverage.

    The source precision is conservative: the hand-labeled families are not an
    exhaustive judgment of every possible relevant passage in the corpus.
    """
    if not families:
        return {"hit_at_k": None, "family_recall_at_k": None,
                "gold_source_precision_at_k": None, "reciprocal_rank": None}
    unique = []
    seen = set()
    for item in evidence:
        key = item.get("source_id") or normalized_url(str(item.get("url", "")))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    flags = [any(matches_family(item, family) for family in families) for item in unique]
    covered = [any(matches_family(item, family) for item in unique) for family in families]
    return {
        "hit_at_k": float(any(flags)),
        "family_recall_at_k": sum(covered) / len(families),
        "gold_source_precision_at_k": sum(flags) / len(flags) if flags else 0.0,
        "reciprocal_rank": next((1.0 / (rank + 1) for rank, hit in enumerate(flags) if hit), 0.0),
    }


def behavior_score(expected: str, analysis: dict[str, Any]) -> bool:
    status = analysis.get("status")
    if expected == "evidence":
        return status == "evidence_found" and bool(analysis.get("evidence"))
    if expected == "needs_details":
        return status == "needs_details" and bool(analysis.get("missing_details"))
    if expected == "abstain":
        return status in ABSTAIN_STATUSES
    raise ValueError(f"Unknown expected_behavior: {expected}")


def reference_integrity(
    evidence: list[dict[str, Any]], manifest: dict[str, dict[str, Any]],
    valid_chunk_ids: set[str] | None = None, answer: str = "",
) -> dict[str, Any]:
    """Check that returned references and verbatim quotes exist, not entailment."""
    failures = []
    citation_ids = []
    for item in evidence:
        source = manifest.get(str(item.get("source_id", "")))
        if source is None:
            failures.append(f"Unknown source: {item.get('source_id')}")
            continue
        if normalized_url(str(item.get("url", ""))) != normalized_url(str(source.get("url", ""))):
            failures.append(f"URL does not match source: {item.get('source_id')}")
        if not item.get("chunk_id"):
            failures.append(f"Missing chunk_id: {item.get('source_id')}")
        elif valid_chunk_ids is not None and item["chunk_id"] not in valid_chunk_ids:
            failures.append(f"Unknown chunk_id: {item['chunk_id']}")
        citation_id = item.get("citation_id") or item.get("citation")
        if citation_id:
            citation_ids.append(str(citation_id))
        quote = " ".join(str(item.get("quote", "")).split())
        source_text = " ".join(str(source.get("text", "")).split())
        if not quote or quote not in source_text:
            failures.append(f"Quote not found verbatim in source: {item.get('source_id')}")
    if len(citation_ids) != len(set(citation_ids)):
        failures.append("Repeated citation IDs")
    unknown_references = set(re.findall(r"\[(S\d+)\]", answer)) - set(citation_ids)
    if unknown_references:
        failures.append("Unknown answer citations: " + ", ".join(sorted(unknown_references)))
    return {"passed": not failures, "checked_references": len(evidence), "failures": failures}


def average(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate observed calls; boundary cases are not generation failures.

    Usage includes accepted and rejected provider responses when reported. No
    missing token counts, costs, model names, or provider timing are inferred.
    """
    completed = [row for row in rows if not row.get("error")]
    records = [row.get("analysis", {}).get("metrics", {}) for row in completed]
    attempted = [item for item in records if item.get("generation_attempted") is True or (
        "generation_attempted" not in item and item.get("generation_mode") in {
            "cited_model_synthesis", "model_abstained"})]
    accepted = [item for item in attempted if item.get("generation_mode") == "cited_model_synthesis"]
    abstained = [item for item in attempted if item.get("generation_mode") == "model_abstained"]
    fallback = [item for item in records if str(item.get("generation_mode", "")).startswith("extracted_evidence_")]
    eligible = [row for row in completed if row.get("expected_behavior") == "evidence"
                or row.get("analysis", {}).get("metrics", {}).get("generation_attempted") is True
                or row.get("analysis", {}).get("metrics", {}).get("generation_mode") == "model_abstained"]
    usage_reports = [item["generation_usage"] for item in attempted if isinstance(item.get("generation_usage"), dict)
                     and item["generation_usage"]]
    token_usage = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [usage[field] for usage in usage_reports if isinstance(usage.get(field), int)
                  and not isinstance(usage.get(field), bool) and usage[field] >= 0]
        token_usage[field] = sum(values) if values else None
    latencies = [float(item["generation_latency_ms"]) for item in attempted
                 if isinstance(item.get("generation_latency_ms"), (int, float))
                 and not isinstance(item.get("generation_latency_ms"), bool)
                 and item["generation_latency_ms"] >= 0]
    calls = []
    for item in attempted:
        recorded = item.get('generation_calls')
        if isinstance(recorded, list):
            calls.extend(call for call in recorded if isinstance(call, dict) and call.get('attempted'))
        else:
            calls.append({'stage': 'synthesis', 'attempted': True, 'usage': item.get('generation_usage', {}),
                          'returned_model': item.get('generation_returned_model')})
    verifications = [item['generation_verification'] for item in attempted
                     if isinstance(item.get('generation_verification'), dict) and item['generation_verification'].get('attempted')]
    def identity(field):
        values = {str(item[field]) for item in attempted if item.get(field)}
        call_field = field.removeprefix('generation_')
        values.update(str(call[call_field]) for call in calls if call.get(call_field))
        return sorted(values)
    def claim_count(field):
        if any(not isinstance(item.get(field), int) or isinstance(item.get(field), bool) or item[field] < 0 for item in verifications):
            return None
        values = [item[field] for item in verifications if isinstance(item.get(field), int)
                  and not isinstance(item[field], bool) and item[field] >= 0]
        return sum(values) if values or not verifications else None
    return {
        "generation_modes": dict(Counter(item.get("generation_mode", "unknown") for item in records)),
        "generation_eligible_questions": len(eligible),
        "generation_attempts": len(attempted),
        "generation_attempts_unknown": sum("generation_attempted" not in item for item in fallback),
        "generation_accepted_answers": len(accepted),
        "generation_fallbacks": len(fallback),
        "generation_model_abstentions": len(abstained),
        "generation_not_attempted_boundary_cases": len(completed) - len(eligible),
        "generation_acceptance_rate_among_attempts": len(accepted) / len(attempted) if attempted else None,
        "generation_acceptance_denominator": "Answer-pipeline attempts; each may make synthesis and verification calls. Guarded cases without provider calls are excluded.",
        "generation_provider_calls": len(calls),
        "generation_provider_calls_by_stage": dict(Counter(call.get('stage', 'unknown') for call in calls)),
        "generation_provider_calls_with_usage": sum(bool(call.get('usage')) for call in calls),
        "generation_provider_calls_without_usage": sum(not bool(call.get('usage')) for call in calls),
        "generation_usage_incomplete_pipelines": sum(item.get('generation_usage_complete') is False for item in attempted),
        "generation_verification_attempts": len(verifications),
        "generation_verification_retained_claims": claim_count('accepted_claims'),
        "generation_verification_removed_claims": claim_count('rejected_claims'),
        "generation_verification_unknown_claim_counts": sum(item.get('accepted_claims') is None or item.get('rejected_claims') is None for item in verifications),
        "generation_partial_answers": sum(item.get('generation_status') == 'success_partial' for item in accepted),
        "generation_fallback_reasons": dict(Counter(item.get("generation_error", "unspecified") for item in fallback)),
        "generation_statuses": dict(Counter(item.get("generation_status", "not_recorded") for item in records)),
        "generation_requested_models": identity("generation_requested_model"),
        "generation_returned_models": identity("generation_returned_model"),
        "generation_endpoints": identity("generation_endpoint"),
        "generation_finish_reasons": dict(Counter(item.get("generation_finish_reason") for item in attempted
                                                  if item.get("generation_finish_reason"))),
        "generation_token_usage": token_usage,
        "generation_usage_reported_attempts": len(usage_reports),
        "generation_usage_missing_attempts": len(attempted) - len(usage_reports),
        "generation_latency_p50_ms": percentile(latencies, 0.50),
        "generation_latency_p95_ms": percentile(latencies, 0.95),
        "generation_latency_recorded_attempts": len(latencies),
    }


def summarize_ai_review(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize separate ordinal AI ratings; leave human/manual fields alone."""
    scored = [row for row in annotations if row.get("review_status") == "scored_generated_answer"]
    seen = set()
    for row in scored:
        identity = (row["question_id"], row["mode"])
        if identity in seen:
            raise ValueError("Duplicate AI review for question and mode")
        seen.add(identity)
        for field in AI_REVIEW_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
                raise ValueError("AI review ratings must be numeric values from 0 to 2")
    return {
        "review_kind": "AI review; not a human assessment",
        "reviewed_generated_answers": len(scored),
        **{field: average([row[field] for row in scored]) for field in AI_REVIEW_FIELDS},
        "note": "Ordinal AI judgments on a small public diagnostic set; independent human review remains separate.",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    answerable = [row for row in successful if row["expected_behavior"] == "evidence"]
    abstentions = [row for row in successful if row["expected_behavior"] != "evidence"]
    retrieval_times = [row["retrieval_ms"] for row in successful]
    referenced = [row for row in successful if row["references"]["checked_references"]]
    return {
        "questions": len(rows),
        "completed": len(successful),
        "errors": len(rows) - len(successful),
        "answerable_questions": len(answerable),
        "hit_at_k": average([row["retrieval"]["hit_at_k"] for row in answerable]),
        "family_recall_at_k": average([row["retrieval"]["family_recall_at_k"] for row in answerable]),
        "gold_source_precision_at_k": average([row["retrieval"]["gold_source_precision_at_k"] for row in answerable]),
        "mean_reciprocal_rank": average([row["retrieval"]["reciprocal_rank"] for row in answerable]),
        "behavior_pass_rate": average([row["behavior_passed"] for row in successful]),
        "abstention_or_clarification_pass_rate": average([row["behavior_passed"] for row in abstentions]),
        "reference_integrity_pass_rate": average([row["references"]["passed"] for row in referenced]),
        "analyses_with_references": len(referenced),
        "references_checked": sum(row["references"]["checked_references"] for row in referenced),
        **generation_summary(rows),
        "retrieval_latency_p50_ms": percentile(retrieval_times, 0.50),
        "retrieval_latency_p95_ms": percentile(retrieval_times, 0.95),
        "analysis_latency_p50_ms": percentile([row["analysis_ms"] for row in successful], 0.50),
        "answerable_analysis_latency_p50_ms": percentile([row["analysis_ms"] for row in answerable], 0.50),
        "answerable_analysis_latency_p95_ms": percentile([row["analysis_ms"] for row in answerable], 0.95),
        "faithfulness": None,
        "answer_relevance": None,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from triage_desk import TriageDesk

    source_path = args.data_dir / "sources.json"
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes)
    sources = raw if isinstance(raw, list) else raw["sources"]
    manifest = {str(source["source_id"]): source for source in sources}
    dataset_bytes = args.questions.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = dataset["questions"][:args.max_questions] if args.max_questions else dataset["questions"]
    for question in questions:
        for family in question.get("expected_families", []):
            if not any(matches_family(source, family) for source in sources):
                raise ValueError(f"Question {question['id']} has an expected source family absent from this corpus. Review the diagnostic labels after refreshing sources.")
    desk = TriageDesk(args.data_dir, args.cache_dir)
    valid_chunk_ids = {chunk["chunk_id"] for chunk in desk.chunks}
    results = []
    warmups = {}
    for mode in args.modes:
        # Dense model/index initialization is separately timed, so warm query
        # latency does not masquerade as first-use latency (or vice versa).
        started = time.perf_counter()
        try:
            desk.search("session state", mode=mode, top_k=args.top_k)
            warmups[mode] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "error": None}
        except Exception as exc:
            warmups[mode] = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                             "error": f"{type(exc).__name__}: {exc}"}
        for question in questions:
            row = {"question_id": question["id"], "question": question["question"],
                   "kind": question["kind"], "mode": mode,
                   "expected_behavior": question["expected_behavior"],
                   "expected_families": question.get("expected_families", []),
                   "expected_answer_checks": question.get("expected_answer_checks", []),
                   "manual_review": {"faithfulness_0_to_2": None, "answer_relevance_0_to_2": None,
                                     "citation_entailment_0_to_2": None, "notes": ""}}
            if warmups[mode]["error"]:
                row["error"] = "Mode initialization failed: " + warmups[mode]["error"]
                results.append(row)
                continue
            try:
                started = time.perf_counter()
                evidence = desk.search(question["question"], mode=mode, top_k=args.top_k,
                                       source_kinds=question.get("source_kinds"))
                row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 3)
                started = time.perf_counter()
                analysis = desk.analyze(question["question"], mode=mode,
                                        source_kinds=question.get("source_kinds"),
                                        version=question.get("version", ""),
                                        include_generation=args.include_generation)
                row["analysis_ms"] = round((time.perf_counter() - started) * 1000, 3)
                row["retrieval"] = retrieval_scores(evidence, row["expected_families"])
                row["behavior_passed"] = behavior_score(question["expected_behavior"], analysis)
                row["references"] = reference_integrity(analysis.get("evidence", []), manifest,
                                                         valid_chunk_ids, analysis.get("answer", ""))
                row["retrieved_sources"] = [{key: item.get(key) for key in
                    ("source_id", "chunk_id", "title", "url", "score", "quote")} for item in evidence]
                row["analysis"] = analysis
                row["error"] = None
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset.get("name"),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "corpus_sources": len(sources),
        "question_count": len(questions),
        "question_kinds": dict(Counter(question["kind"] for question in questions)),
        "top_k": args.top_k,
        "analysis_top_k": 6,
        "generation_requested": args.include_generation,
        "generation_model_identity": {
            "requested_models": generation_summary(results)["generation_requested_models"],
            "returned_models": generation_summary(results)["generation_returned_models"],
            "endpoints": generation_summary(results)["generation_endpoints"],
            "source": "Observed per-call metadata; no configured identity is substituted for an unreported provider response.",
        },
        "timing_notes": "One warmup per mode; warmed retrieval timed separately from analyze. Small local sample, not a production SLO. Modes run sequentially with shared cache; warmup is initialization/cache-load cost, not independent cold-model benchmarking.",
        "measurement_limits": [
            "Gold source families are manually selected and incomplete; source precision is a conservative proxy, not passage relevance precision.",
            "Reference integrity checks source IDs, URLs, chunk IDs, and verbatim quote existence. It does not measure citation entailment or answer faithfulness.",
            "Faithfulness and answer relevance require review of actual generated answers. Human/manual fields remain unscored; any AI review is separately identified and is not human validation.",
            "This small, public, hand-authored diagnostic set is not a hidden test set or evidence of general production reliability.",
            "The original diagnostic set was used during development. Independently authored validation questions reduce direct tuning, but remain a small AI-authored project evaluation rather than an external human benchmark.",
        ],
        "warmup": warmups,
        "summary": {mode: summarize([row for row in results if row["mode"] == mode]) for mode in args.modes},
        "results": results,
    }


def display(value: Any) -> str:
    if value is None:
        return "Not scored"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(report: dict[str, Any]) -> str:
    esc = lambda value: html.escape(display(value))
    fields = ["hit_at_k", "family_recall_at_k", "gold_source_precision_at_k", "mean_reciprocal_rank",
              "behavior_pass_rate", "abstention_or_clarification_pass_rate", "reference_integrity_pass_rate",
              "retrieval_latency_p50_ms", "retrieval_latency_p95_ms",
              "generation_attempts", "generation_accepted_answers", "generation_fallbacks",
              "generation_provider_calls", "generation_provider_calls_by_stage",
              "generation_verification_removed_claims", "generation_partial_answers",
              "generation_model_abstentions", "generation_not_attempted_boundary_cases",
              "generation_acceptance_rate_among_attempts", "generation_latency_p50_ms", "generation_latency_p95_ms",
              "answerable_analysis_latency_p50_ms", "answerable_analysis_latency_p95_ms",
              "generation_token_usage", "faithfulness", "answer_relevance"]
    header = "<tr><th>Metric</th>" + "".join(f"<th>{esc(mode)}</th>" for mode in report["summary"]) + "</tr>"
    summary = "".join("<tr><th>" + esc(field) + "</th>" + "".join(
        f"<td>{esc(values.get(field))}</td>" for values in report["summary"].values()) + "</tr>" for field in fields)
    details = []
    for row in report["results"]:
        analysis = row.get("analysis", {})
        metrics = analysis.get("metrics", {})
        generation = (f'<h3>Observed generation</h3><p>Mode: <code>{esc(metrics.get("generation_mode", "not recorded"))}</code>'
                      f'<br>Failure or fallback: {esc(metrics.get("generation_error") or "None recorded")}'
                      f'<br>Returned model: {esc(metrics.get("generation_returned_model") or "Not recorded")}</p>')
        claim_items = []
        for claim in analysis.get("generated_claims", []):
            if not isinstance(claim, dict):
                continue
            supports = "".join(
                f'<blockquote><code>{esc(support.get("citation_id"))} · {esc(support.get("quote_id") or "literal quote")}</code>'
                f'<pre>{esc(support.get("quote"))}</pre></blockquote>'
                for support in claim.get("support", []) if isinstance(support, dict))
            claim_items.append(f'<li><p>{esc(claim.get("statement"))}</p>{supports}</li>')
        claims = ('<h3>Model-generated claims and exact selected source quotes</h3><ol>'
                  + "".join(claim_items) + '</ol>') if claim_items else ""
        ai_assessment = ""
        if row.get("ai_review") is not None:
            ai_assessment = ('<section class="note"><h3>AI assessment · not human review</h3>'
                             '<p>This assessment judges answer quality and claim support. It is separate from automatic reference-integrity checks.</p>'
                             f'<pre>{esc(json.dumps(row["ai_review"], indent=2, ensure_ascii=False, sort_keys=True))}</pre></section>')
        sources = "".join(f'<li><a href="{html.escape(str(item.get("url", "")), quote=True)}">'
                          f'{esc(item.get("title"))}</a><blockquote>{esc(item.get("quote"))}</blockquote></li>'
                          for item in row.get("retrieved_sources", []))
        details.append(f'<details><summary>{esc(row["question_id"])} · {esc(row["mode"])} · '
                       f'{esc(analysis.get("status", row.get("error")))} · behavior pass: '
                       f'{esc(row.get("behavior_passed"))}</summary><p>{esc(row["question"])}</p>'
                       f'{generation}<pre>{esc(analysis.get("answer", row.get("error")))}</pre>{claims}{ai_assessment}<p>Retrieval: '
                       f'{esc(row.get("retrieval"))}</p><ol>{sources}</ol></details>')
    limits = "".join(f"<li>{esc(limit)}</li>" for limit in report["measurement_limits"])
    ai_summary = report.get("ai_review_summary")
    ai_section = ""
    if isinstance(ai_summary, dict):
        ai_rows = "".join(f"<tr><th>{esc(field)}</th><td>{esc(ai_summary.get(field))}</td></tr>"
                          for field in AI_REVIEW_FIELDS)
        standard_fields = set(AI_REVIEW_FIELDS) | {"reviewer", "review_kind", "reviewed_generated_answers", "note"}
        extra_rows = "".join(f"<tr><th>{esc(field)}</th><td>{esc(value)}</td></tr>"
                             for field, value in ai_summary.items() if field not in standard_fields)
        extra_measures = ("<h3>Additional AI claim-support assessments</h3>"
                          "<p>These counts and fractions are AI judgments about selected quotations, not human review or measured accuracy.</p>"
                          f"<table><tr><th>AI assessment measure</th><th>Value</th></tr>{extra_rows}</table>") if extra_rows else ""
        ai_section = ("<h2>AI-reviewed answer quality</h2><p class=\"note\">"
                      "These are AI judgments, not human review or automatic citation checks. "
                      "Human/manual quality fields remain unscored.</p>"
                      f"<p>Reviewer: {esc(ai_summary.get('reviewer', 'AI reviewer; identity not recorded'))}<br>"
                      f"Generated answers reviewed: {esc(ai_summary.get('reviewed_generated_answers'))}</p>"
                      f"<table><tr><th>Ordinal dimension (0–2)</th><th>Mean AI rating</th></tr>{ai_rows}</table>"
                      f"{extra_measures}"
                      f"<p>{esc(ai_summary.get('note', 'Small public diagnostic set; human validation remains separate.'))}</p>")
    title = "Retrieval and generation diagnostic report" if report.get("generation_requested") else "Retrieval diagnostic report"
    identity = report.get("generation_model_identity", {})
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Engineering Issue Desk evaluation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:36px auto;padding:0 20px;color:#172033;background:#f6f8fc}}
table{{border-collapse:collapse;background:white;width:100%}}th,td{{padding:10px;border:1px solid #d9dfeb;text-align:left}}
th{{background:#eaf0fa}}details{{background:white;border:1px solid #d9dfeb;border-radius:8px;padding:14px;margin:10px 0}}
summary{{cursor:pointer;font-weight:650}}blockquote{{font-size:14px;color:#44536b;border-left:3px solid #ccd7eb;padding-left:12px}}
code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}a{{color:#174ea6}}.note{{padding:16px;background:#eaf0fa;border-radius:8px}}
</style><h1>{esc(title)}</h1><p>{esc(report["question_count"])} questions · {esc(report["corpus_sources"])} sources · top {esc(report["top_k"])} chunks · generated {esc(report["generated_at"])}</p>
<p class="note">Automatic scores cover retrieval, reference integrity, and observed generation outcomes. Human/manual faithfulness and answer relevance are <strong>not scored</strong>. Any AI quality review appears separately below.</p>
<table>{header}{summary}</table><h2>Observed generation identity</h2><pre>{esc(json.dumps(identity, indent=2))}</pre>
<p>Generation acceptance uses answer-pipeline attempts as its denominator; guarded questions without provider calls are excluded. A pipeline may make separate synthesis and verification calls, counted by stage. Token totals include rejected outputs and verification when usage was reported. Unreported usage is not inferred. Answerable analysis latency includes warm retrieval and the complete answer pipeline.</p>
{ai_section}<h2>Initialization and timing</h2><p>{esc(report["timing_notes"])}</p><pre>{esc(json.dumps(report["warmup"], indent=2))}</pre>
<h2>Limits</h2><ul>{limits}</ul><h2>Question evidence</h2>{''.join(details)}
<h2>Reproducibility</h2><p>Corpus SHA-256: <code>{esc(report["corpus_sha256"])}</code><br>Question SHA-256: <code>{esc(report["dataset_sha256"])}</code></p></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/retrieval"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--include-generation", action="store_true", help="Requests configured model generation; quality still requires manual review.")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 30 or (args.max_questions is not None and args.max_questions < 1):
        parser.error("top-k must be 1–30 and max-questions must be positive")
    if args.include_generation:
        # Only the explicit CLI generation path reads the project environment.
        # Importing this module, offline evaluation and unit tests do not load it.
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    report = run_evaluation(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    html_path = args.output_dir / "report.html"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "json": str(json_path), "html": str(html_path)}, indent=2))
    return int(any(summary["errors"] for summary in report["summary"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
