"""Exercise the actual compiled graph using local fixtures and mocked grading."""
import json

import pytest

from triage_desk.generation import GenerationUnavailable
from triage_desk.workflow import InvestigationWorkflow, _validated_grade, provider_call_summary


def evidence(source_id="doc-cache", chunk_id="cache-1", text=None, kind="documentation"):
    text = text or "Use st.cache_data for serializable values. Caching avoids repeated work across reruns."
    return {"source_id": source_id, "chunk_id": chunk_id, "title": "Source " + source_id,
            "kind": kind, "section": "Overview", "text": text, "quote": text, "score": 1.,
            "url": "https://docs.example.test/" + source_id, "source_commit": "fixture",
            "snapshot_at": "2026-10-01T00:00:00Z", "version_scope": "test snapshot"}


class FakeDesk:
    def __init__(self, retrieved=None):
        self.chunks = [evidence(), evidence("doc-state", "state-1", "Session State stores values between reruns.")]
        self.sources = self.chunks
        self.retrieved = retrieved if retrieved is not None else [self.chunks[0]]
        self.search_calls, self.analysis_calls = [], []

    def search(self, question, **kwargs):
        self.search_calls.append((question, kwargs))
        return [dict(item) for item in self.retrieved]

    def analyze(self, question, **kwargs):
        self.analysis_calls.append((question, kwargs))
        return {"question": question, "status": "evidence_found", "answer": "Retrieved evidence for inspection.",
                "evidence": kwargs["_evidence"], "suggested_type": "question", "component": "Caching",
                "missing_details": [], "related_issues": [], "draft_markdown": "A local draft; not posted.",
                "generated_claims": [], "metrics": {"latency_ms": 1., "mode": kwargs["mode"]}}


def test_actual_langgraph_offline_pipeline_uses_documents_and_one_analysis():
    desk = FakeDesk()
    workflow = InvestigationWorkflow(desk)
    result = workflow.run("How does st.cache_data cache values?", mode="keyword")
    assert len(desk.search_calls) == len(desk.analysis_calls) == 1
    assert desk.analysis_calls[0][1]["_evidence"][0]["source_id"] == "doc-cache"
    assert [row["stage"] for row in result["workflow"]["trace"]] == ["prepare", "retrieve", "grade", "analyze", "handoff", "finish"]
    assert result["workflow"]["grade"]["answerable"] is None
    assert "unscored" in result["workflow"]["grade"]["reason"].lower()
    assert result["workflow"]["grade_calls"] == []
    assert result["workflow"]["confidence"]["is_probability"] is False
    assert result["workflow"]["handoff"]["available"] is True
    assert result["workflow"]["handoff"]["required"] is False
    assert "prepare" in workflow.graph.get_graph().nodes
    assert "retrieve" in result["workflow"]["mermaid"]
    assert result["metrics"]["core_analysis_latency_ms"] == 1.


@pytest.mark.parametrize("question", [
    "Why does my st.dataframe fail?", "Which process is using port 8501?",
    "What packages are installed on my machine?", "What Streamlit version is installed on my computer?",
])
def test_personal_diagnosis_or_local_runtime_unknowns_clarify_without_calls(question):
    desk = FakeDesk()
    result = InvestigationWorkflow(desk).run(question, include_generation=True)
    assert result["status"] == "needs_details"
    assert result["missing_details"]
    assert desk.search_calls == desk.analysis_calls == []
    assert result["workflow"]["grade_calls"] == []
    assert result["workflow"]["handoff"]["draft_available"] is True
    assert [row["stage"] for row in result["workflow"]["trace"]] == ["prepare", "handoff", "finish"]


def test_history_resolves_explicit_followups_from_last_three_user_turns_only():
    history = [{"role": "user", "content": "OLDEST_EXCLUDED"},
               {"role": "assistant", "content": "ASSISTANT_UNTRUSTED_FACT"},
               {"role": "user", "content": "Review st.cache_data behavior."},
               {"role": "user", "content": "The input is a pandas value."},
               {"role": "assistant", "content": "ANOTHER_ASSISTANT_FACT"},
               {"role": "user", "content": "The relevant API is st.cache_data."}]
    desk = FakeDesk()
    result = InvestigationWorkflow(desk).run("What about the same issue after a rerun?", mode="keyword", history=history)
    resolved = desk.search_calls[0][0]
    assert "OLDEST_EXCLUDED" not in resolved and "ASSISTANT" not in resolved
    assert "not source evidence" in resolved
    assert result["workflow"]["history_user_turns_used"] == 3
    assert result["workflow"]["history_is_evidence"] is False
    unrelated = FakeDesk()
    InvestigationWorkflow(unrelated).run("How does caching work?", mode="keyword", history=history)
    assert unrelated.search_calls[0][0] == "How does caching work?"


def test_single_batch_grade_selects_only_supported_documents_before_analysis(monkeypatch):
    desk = FakeDesk(retrieved=[evidence(), evidence("doc-state", "state-1")])
    workflow = InvestigationWorkflow(desk)
    calls = []
    def grade(question, items, provider_calls):
        calls.append(items)
        provider_calls.append({"stage": "evidence_grading", "attempted": True, "usage": {"total_tokens": 23}})
        return {"answerable": True, "relevant_citation_ids": ["S2"], "reason": "The second context directly supports the request.", "missing_info": []}
    monkeypatch.setattr(workflow, "_grade_evidence", grade)
    result = workflow.run("Explain caching values.", mode="keyword", include_generation=True)
    assert len(calls) == 1 and len(calls[0]) == 2
    assert len(desk.analysis_calls) == 1
    assert [item["source_id"] for item in desk.analysis_calls[0][1]["_evidence"]] == ["doc-state"]
    assert result["workflow"]["grade"]["relevant_sources"][0]["source_id"] == "doc-state"
    assert result["workflow"]["grade_calls"][0]["usage"]["total_tokens"] == 23


def test_negative_directness_grade_collects_details_without_synthesis(monkeypatch):
    desk = FakeDesk()
    workflow = InvestigationWorkflow(desk)
    monkeypatch.setattr(workflow, "_grade_evidence", lambda *args: {
        "answerable": False, "relevant_citation_ids": [], "reason": "The sources do not establish a personal root cause.",
        "missing_info": ["A small reproduction and the complete error"]})
    result = workflow.run("The cache output is unexpected; investigate it.", mode="keyword", include_generation=True)
    assert result["status"] == "needs_details" and desk.analysis_calls == []
    assert result["workflow"]["handoff"]["action"] == "collect_details"
    assert "analyze" not in [row["stage"] for row in result["workflow"]["trace"]]


def test_grader_failure_fails_closed_but_retains_the_paid_call(monkeypatch):
    desk = FakeDesk()
    workflow = InvestigationWorkflow(desk)
    def unavailable(question, items, calls):
        calls.append({"stage": "evidence_grading", "attempted": True, "status": "invalid_grade", "usage": {"total_tokens": 17}})
        raise GenerationUnavailable("Mock malformed grade")
    monkeypatch.setattr(workflow, "_grade_evidence", unavailable)
    result = workflow.run("Explain cache behavior.", mode="keyword", include_generation=True)
    assert result["status"] == "insufficient_evidence"
    assert desk.analysis_calls == []
    assert result["workflow"]["grade_calls"][0]["usage"]["total_tokens"] == 17
    assert result["workflow"]["trace"][2]["decision"] == "failed_closed"
    assert result["generated_claims"] == []


def test_graph_source_augmentation_resolves_only_known_local_chunks():
    class Graph:
        def search_sources(self, question, limit):
            return ["doc-state", "unknown-invented-source"]
    desk = FakeDesk()
    result = InvestigationWorkflow(desk, knowledge_graph=Graph()).run("Explain session state.", mode="keyword", use_graph=True)
    assert {item["source_id"] for item in result["evidence"]} == {"doc-cache", "doc-state"}
    retrieve = next(row for row in result["workflow"]["trace"] if row["stage"] == "retrieve")
    assert retrieve["graph_sources_added"] == 1


def test_lab_options_are_real_passthroughs_without_a_baseline_fallback():
    class Lab:
        chunks_by_strategy = {"fixed": [evidence(chunk_id="fixed-chunk")]}
        def __init__(self):
            self.calls = []
        def search(self, question, **options):
            self.calls.append(options)
            return [evidence(chunk_id="fixed-chunk")]
    desk, lab = FakeDesk(), Lab()
    result = InvestigationWorkflow(desk, retrieval_lab=lab).run("How does caching work?", mode="semantic", strategy="fixed", rerank=True)
    assert lab.calls == [{"strategy": "fixed", "mode": "semantic", "rerank": True, "top_k": 6, "source_kinds": None}]
    assert desk.search_calls == []
    assert result["evidence"][0]["chunk_id"] == "fixed-chunk"
    with pytest.raises(ValueError, match="RetrievalLab"):
        InvestigationWorkflow(desk).run("Caching question", rerank=True)


def test_retrieval_failure_never_switches_mode_or_calls_analysis(monkeypatch):
    desk = FakeDesk()
    requested = []
    def fail(question, **options):
        requested.append(options["mode"])
        raise RuntimeError("Mock dense index unavailable")
    monkeypatch.setattr(desk, "search", fail)
    result = InvestigationWorkflow(desk).run("How does caching work?", mode="hybrid", include_generation=True)
    assert requested == ["hybrid"] and desk.analysis_calls == []
    assert result["status"] == "insufficient_evidence"
    assert "alternative mode" in result["answer"]
    assert result["workflow"]["grade_calls"] == []


@pytest.mark.parametrize("grade", [
    {"answerable": "true", "relevant_citation_ids": ["S1"], "reason": "A reason", "missing_info": []},
    {"answerable": True, "relevant_citation_ids": ["S99"], "reason": "A reason", "missing_info": []},
    {"answerable": True, "relevant_citation_ids": [], "reason": "A reason", "missing_info": []},
    {"answerable": True, "relevant_citation_ids": ["S1", "S1"], "reason": "A reason", "missing_info": []},
])
def test_directness_grade_requires_boolean_known_unique_supported_references(grade):
    with pytest.raises(ValueError):
        _validated_grade(json.dumps(grade), [{**evidence(), "citation_id": "S1"}])


def test_workflow_accounts_only_new_embedding_calls_and_each_generation_call_once(monkeypatch):
    class Index:
        model_name, fingerprint, index_origin, cache_hit = "hosted-test-model", "hash-1", "computed", False
        def __init__(self):
            self.calls = [{"stage": "embedding", "attempted": True, "usage": {"prompt_tokens": 999, "total_tokens": 999}}]
    index = Index()
    class Lab:
        dense_by_strategy = {"structured": index}
        chunks_by_strategy = {"structured": [evidence()]}
        def search(self, question, **kwargs):
            index.calls.extend([
                {"stage": "embedding", "attempted": True, "usage": {"prompt_tokens": 10, "total_tokens": 10}},
                {"stage": "embedding", "attempted": True, "usage": {"prompt_tokens": 2, "total_tokens": 2}}])
            return [evidence()]
    desk = FakeDesk()
    original_analyze = desk.analyze
    def analyze(question, **kwargs):
        result = original_analyze(question, **kwargs)
        result["metrics"].update({"generation_usage": {"total_tokens": 25}, "generation_calls": [
            {"stage": "synthesis", "attempted": True, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            {"stage": "verification", "attempted": True, "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}]})
        return result
    monkeypatch.setattr(desk, "analyze", analyze)
    workflow = InvestigationWorkflow(desk, retrieval_lab=Lab())
    def grade(question, items, calls):
        calls.append({"stage": "evidence_grading", "attempted": True, "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})
        return {"answerable": True, "relevant_citation_ids": ["S1"], "reason": "Direct evidence", "missing_info": []}
    monkeypatch.setattr(workflow, "_grade_evidence", grade)
    result = workflow.run("Explain cache behavior.", include_generation=True)
    telemetry = result["workflow"]
    assert len(telemetry["embedding_calls"]) == 2
    assert telemetry["provider_usage"]["attempted_calls"] == 5
    assert telemetry["provider_usage"]["reported_usage"] == {"prompt_tokens": 33, "completion_tokens": 8, "total_tokens": 41}
    assert telemetry["provider_usage"]["usage_complete"] is True
    assert result["metrics"]["embedding_model"] == "hosted-test-model"
    assert result["metrics"]["index_fingerprint"] == "hash-1"


def test_failed_retrieval_retains_new_embedding_request_and_unknown_usage():
    class Index:
        model_name, fingerprint, index_origin = "hosted-test-model", "hash-2", "not_loaded"
        calls = []
    index = Index()
    class Lab:
        dense_by_strategy = {"structured": index}
        chunks_by_strategy = {"structured": [evidence()]}
        def search(self, question, **kwargs):
            index.calls.append({"stage": "embedding", "attempted": True, "status": "timeout", "usage": {}})
            raise RuntimeError("Mock timeout")
    desk = FakeDesk()
    result = InvestigationWorkflow(desk, retrieval_lab=Lab()).run("Explain caching.", include_generation=True)
    usage = result["workflow"]["provider_usage"]
    assert result["status"] == "insufficient_evidence" and desk.analysis_calls == []
    assert usage["calls_by_stage"] == {"embedding": 1}
    assert usage["reported_usage"] == {"prompt_tokens": None, "completion_tokens": 0, "total_tokens": None}
    assert usage["usage_complete"] is False
    assert result["workflow"]["grade_calls"] == []


def test_keyword_and_guarded_runs_report_embeddings_not_used():
    class Index:
        model_name, fingerprint, index_origin = "unused-hosted-model", "never-used", "computed"
        calls = []
    class Lab:
        dense_by_strategy = {"structured": Index()}
        chunks_by_strategy = {"structured": [evidence()]}
        def search(self, question, **kwargs):
            assert kwargs["mode"] == "keyword"
            return [evidence()]
    workflow = InvestigationWorkflow(FakeDesk(), retrieval_lab=Lab())
    for question in ("How does caching work?", "What version is installed on my computer?"):
        result = workflow.run(question, mode="keyword")
        assert result["metrics"]["embedding_model"] == "not used"
        assert result["metrics"]["index_origin"] == "not_used"
        assert result["workflow"]["provider_calls"] == []


def test_call_rollup_never_turns_partial_unknown_or_boolean_usage_into_known_zero():
    rollup = provider_call_summary([
        {"stage": "synthesis", "attempted": True, "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}},
        {"stage": "verification", "attempted": True, "usage": {"prompt_tokens": True, "completion_tokens": -1}},
        {"stage": "embedding", "attempted": False, "usage": {"total_tokens": 500}}])
    assert rollup["reported_usage"] == {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert rollup["usage_complete"] is False
    assert rollup["usage_missing_calls_by_field"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 1}
    assert rollup["attempted_calls"] == 2


def test_graph_filters_before_adding_bounded_known_source_candidates():
    docs = [evidence(f"doc-{index}", f"chunk-{index}") for index in range(5)]
    issue = evidence("issue-1", "issue-1:chunk", kind="issue")
    class Graph:
        def search_sources(self, question, limit):
            assert limit == 100
            return [item["source_id"] for item in [*docs, issue]]
    desk = FakeDesk(retrieved=[])
    desk.chunks = [*docs, issue]
    result = InvestigationWorkflow(desk, knowledge_graph=Graph()).run(
        "Investigate caching reports.", mode="keyword", source_kinds=["issue"], use_graph=True)
    assert [item["source_id"] for item in result["evidence"]] == ["issue-1"]
    trace = next(item for item in result["workflow"]["trace"] if item["stage"] == "retrieve")
    assert trace["graph_candidate_source_count"] == 6
    assert trace["graph_sources_added"] == 1
    assert trace["graph_added_source_limit"] == 4 and trace["output_chunk_limit"] == 8


@pytest.mark.parametrize("question", [
    "Our environment uses Python 3.10. What do the pinned release notes say about supported Python versions?",
    "Our app is running locally. Which Python versions are supported in the official documentation?",
    "Our environment uses Python 3.10. Which command lists active processes?",
])
def test_reported_runtime_context_does_not_block_public_lookup_or_instructions(question):
    desk = FakeDesk()
    result = InvestigationWorkflow(desk).run(question, mode="keyword")
    assert len(desk.search_calls) == len(desk.analysis_calls) == 1
    assert result["workflow"]["route"] == "documentation_question"
    assert "retrieve" in [item["stage"] for item in result["workflow"]["trace"]]


def test_public_lookup_clause_does_not_hide_a_separate_local_inspection_request():
    desk = FakeDesk()
    result = InvestigationWorkflow(desk).run(
        "Check the public documentation; which process is listening on port 8501?", mode="keyword")
    assert result["status"] == "needs_details"
    assert desk.search_calls == desk.analysis_calls == []


@pytest.mark.parametrize("generation_mode,generation_status,required,reason_fragment", [
    ("extracted_evidence_provider_unavailable", "provider_unavailable", True, "accepted cited synthesis"),
    ("extracted_evidence_rejected_answer", "answer_rejected", True, "accepted cited synthesis"),
    ("cited_model_synthesis", "success_partial", True, "partial model synthesis"),
    ("cited_model_synthesis", "success", False, "confirm applicability"),
])
def test_requested_generation_fallback_and_partial_results_require_human_handoff(
        monkeypatch, generation_mode, generation_status, required, reason_fragment):
    desk = FakeDesk()
    original = desk.analyze
    def analyze(question, **kwargs):
        result = original(question, **kwargs)
        result["metrics"].update(generation_mode=generation_mode, generation_status=generation_status)
        return result
    monkeypatch.setattr(desk, "analyze", analyze)
    workflow = InvestigationWorkflow(desk)
    monkeypatch.setattr(workflow, "_grade_evidence", lambda *args: {
        "answerable": True, "relevant_citation_ids": ["S1"], "reason": "Direct fixture support", "missing_info": []})
    result = workflow.run("Explain caching behavior.", mode="keyword", include_generation=True)
    assert result["status"] == "evidence_found"
    handoff = result["workflow"]["handoff"]
    assert handoff["required"] is required
    assert handoff["posted"] is False and handoff["draft_available"] is True
    assert reason_fragment in handoff["reason"]
    assert next(row for row in result["workflow"]["trace"] if row["stage"] == "handoff")["reason"] == handoff["reason"]
