"""Workflow reporting uses local fixtures and mocks, never environment/provider I/O."""
import argparse
import json
from pathlib import Path

import pytest

from evaluation import workflow as evaluation
from triage_desk.retrieval_lab import RetrievalLab
from triage_desk.workflow import InvestigationWorkflow


@pytest.fixture
def fixture_args(tmp_path):
    sources = [
        {"source_id": "doc-cache", "title": "st.cache_data", "text": "st.cache_data caches serializable values and reuses cached values across reruns.",
         "url": "https://docs.example.test/cache", "kind": "documentation"},
        {"source_id": "doc-state", "title": "st.session_state", "text": "st.session_state preserves session state values between reruns.",
         "url": "https://docs.example.test/state", "kind": "documentation"}]
    for source in sources:
        source.update(source_commit="fixture", snapshot_at="2026-10-01T00:00:00Z", version_scope="fixture snapshot")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sources.json").write_text(json.dumps(sources))
    questions = {"name": "Synthetic workflow fixture", "questions": [
        {"id": "cache", "kind": "documentation", "question": "Explain st.cache_data cached values.", "expected_behavior": "evidence",
         "expected_families": [{"source_ids": ["doc-cache"]}]},
        {"id": "local", "kind": "boundary", "question": "Which process is using port 8501?", "expected_behavior": "needs_details"}]}
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(questions))
    return argparse.Namespace(data_dir=data_dir, cache_dir=tmp_path / "cache", questions=question_path,
        output_dir=tmp_path / "report", embedding_provider="local", mode="keyword", strategy="structured",
        rerank=False, graph=False, include_generation=False, max_questions=None)


def test_offline_evaluation_runs_actual_graph_once_per_question_and_no_extra_search(fixture_args, monkeypatch):
    searches = []
    original = RetrievalLab.search
    def search(self, question, **kwargs):
        searches.append(question)
        return original(self, question, **kwargs)
    monkeypatch.setattr(RetrievalLab, "search", search)
    report = evaluation.run_evaluation(fixture_args)
    assert len(report["results"]) == 2
    assert len(searches) == 1  # The local-environment case branches before retrieval.
    assert report["warmup"] == {}
    assert report["results"][0]["references"]["passed"] is True
    assert report["results"][1]["analysis"]["status"] == "needs_details"
    assert report["results"][0]["manual_review"]["faithfulness_0_to_2"] is None
    summary = report["summary"]["keyword"]
    assert summary["workflow_provider_calls"] == 0
    assert summary["workflow_token_usage"]["total_tokens"] == 0
    assert summary["workflow_handoffs_required"] == 1
    assert summary["workflow_drafts_available"] == 2
    assert summary["behavior_pass_rate"] == 1.
    assert all(row["analysis"]["metrics"]["embedding_model"] == "not used" for row in report["results"])
    document = evaluation.render_report(report)
    assert "Post-grade selected evidence" in document
    assert "Complete workflow accounting" in document
    assert "not raw first-stage retrieval" in document


def test_metrics_score_post_grade_documents_not_the_raw_candidate_pool(fixture_args, monkeypatch):
    fixture_args.include_generation, fixture_args.max_questions = True, 1
    raw = json.loads(fixture_args.questions.read_text())
    raw["questions"][0]["question"] = "Compare cached values and session state."
    fixture_args.questions.write_text(json.dumps(raw))
    calls = []
    def grade(self, question, items, provider_calls):
        calls.append(items)
        selected = next(item for item in items if item["source_id"] == "doc-state")
        provider_calls.append({"stage": "evidence_grading", "attempted": True, "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
        return {"answerable": True, "relevant_citation_ids": [selected["citation_id"]], "reason": "Fixture selects state evidence only.", "missing_info": []}
    monkeypatch.setattr(InvestigationWorkflow, "_grade_evidence", grade)
    def synthesize(*args, **kwargs):
        return {"answer": "Source evidence [S1]", "claims": [], "metadata": {"status": "success", "attempted": True,
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}, "calls": [
                {"stage": "synthesis", "attempted": True, "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}},
                {"stage": "verification", "attempted": True, "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}]}}
    monkeypatch.setattr("triage_desk.core.synthesize", synthesize)
    report = evaluation.run_evaluation(fixture_args)
    row = report["results"][0]
    assert len(calls) == 1
    assert {item["source_id"] for item in calls[0]} == {"doc-cache", "doc-state"}
    assert row["error"] is None
    assert [item["source_id"] for item in row["retrieved_sources"]] == ["doc-state"]
    assert row["retrieval"]["hit_at_k"] == 0.
    assert row["references"]["passed"] is True
    summary = report["summary"]["keyword"]
    assert summary["workflow_provider_calls"] == 3
    assert summary["workflow_provider_calls_by_stage"] == {"evidence_grading": 1, "synthesis": 1, "verification": 1}
    assert summary["workflow_token_usage"]["total_tokens"] == 31  # 7+14+10, never +24 again.


def test_unavailable_grade_stays_a_reported_safe_handoff_with_missing_usage(fixture_args, monkeypatch):
    fixture_args.include_generation, fixture_args.max_questions = True, 1
    def unavailable(self, question, items, calls):
        calls.append({"stage": "evidence_grading", "attempted": True, "status": "timeout", "usage": {}})
        raise TimeoutError("Mock timeout")
    monkeypatch.setattr(InvestigationWorkflow, "_grade_evidence", unavailable)
    report = evaluation.run_evaluation(fixture_args)
    summary = report["summary"]["keyword"]
    assert report["results"][0]["analysis"]["status"] == "insufficient_evidence"
    assert summary["workflow_provider_calls"] == 1
    assert summary["workflow_usage_complete"] is False
    assert summary["workflow_token_usage"] == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    assert summary["workflow_answerable_evidence_status_coverage"] == 0.
    assert report["results"][0]["retrieval"]["hit_at_k"] == 0.
    assert report["results"][0]["retrieved_sources"] == []


def test_hosted_backend_is_selected_only_explicitly_without_eager_requests(fixture_args):
    seen = []
    class HostedFixture:
        def __init__(self, chunks, cache_dir):
            seen.append(len(chunks))
            self.model_name, self.fingerprint, self.calls = "mock-hosted", "fixture-hash", []
        def scores(self, question):
            pytest.fail("Keyword mode must not call hosted embeddings")
    fixture_args.embedding_provider = "nebius"
    report = evaluation.run_evaluation(fixture_args, dense_factory=HostedFixture)
    assert len(seen) == 2  # Lazy index objects for each chunk strategy, no requests.
    assert report["configuration"]["embedding_provider"] == "nebius"
    assert report["summary"]["keyword"]["workflow_provider_calls"] == 0


@pytest.mark.parametrize("flags,expected_loads", [([], 0), (["--include-generation"], 1),
    (["--embedding-provider", "nebius"], 1), (["--embedding-provider", "nebius", "--include-generation"], 1)])
def test_cli_environment_file_loading_requires_an_explicit_provider_choice(tmp_path, monkeypatch, flags, expected_loads):
    import dotenv
    loads = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda path, override: loads.append((path, override)))
    monkeypatch.setattr(evaluation, "run_evaluation", lambda args: {"summary": {}, "results": []})
    monkeypatch.setattr(evaluation, "render_report", lambda report: "<html>fixture</html>")
    assert evaluation.main(["--output-dir", str(tmp_path / "out"), *flags]) == 0
    assert len(loads) == expected_loads
    if loads:
        assert loads[0][0].name == ".env" and loads[0][1] is False
    assert json.loads((tmp_path / "out/report.json").read_text()) == {"summary": {}, "results": []}


def test_errors_remain_in_coverage_denominator_and_flag_unobserved_accounting():
    rows = [{"question_id": "failed", "expected_behavior": "evidence", "error": "Mock graph failure"}]
    summary = evaluation.workflow_summary(rows)
    assert summary["workflow_scheduled_cases"] == 1 and summary["workflow_completed_cases"] == 0
    assert summary["workflow_answerable_evidence_status_coverage"] == 0.
    assert summary["workflow_call_accounting_unknown_error_rows"] == 1
    assert summary["workflow_usage_complete"] is False


def test_absent_gold_family_is_rejected_before_any_workflow_run(fixture_args, monkeypatch):
    raw = json.loads(fixture_args.questions.read_text())
    raw["questions"][0]["expected_families"] = [{"source_ids": ["not-in-corpus"]}]
    fixture_args.questions.write_text(json.dumps(raw))
    monkeypatch.setattr(InvestigationWorkflow, "run", lambda *args, **kwargs: pytest.fail("Invalid labels must be caught before any graph/provider run"))
    with pytest.raises(ValueError, match="absent from the corpus"):
        evaluation.run_evaluation(fixture_args)
