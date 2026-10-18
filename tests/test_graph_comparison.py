"""Offline source-budget, scoring, provenance and failure tests; no API keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.graph_comparison import render_report, run_comparison
from triage_desk.core import _load_sources, _quote
from triage_desk.knowledge_graph import EngineeringKnowledgeGraph


class OfflineDesk:
    """Explicit candidate fixture matching the core search contract."""
    instances = []

    def __init__(self, data_dir, cache_dir):
        self.sources = _load_sources(Path(data_dir))
        self.calls = []
        self._dense = SimpleNamespace(fingerprint="offline-fixture", index_origin="offline-test")
        self.instances.append(self)

    def search(self, question, *, mode, source_kinds, top_k):
        self.calls.append({"question": question, "mode": mode, "source_kinds": source_kinds, "top_k": top_k})
        assert mode == "hybrid" and top_k == 18
        sources = [source for source in self.sources if source_kinds is None or source["kind"] in source_kinds]
        candidates = []
        for source in sources:
            # Deliberately return two chunks per source. A chunk count must not
            # silently become the source-level quality/context denominator.
            for index in range(2):
                candidates.append({**source, "chunk_id": source["source_id"] + f":{index}",
                                   "quote": _quote(source["text"], question), "score": .5})
        return candidates[:top_k]


def write_inputs(tmp_path, sources, questions):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"name": "Offline source comparison", "questions": questions}), encoding="utf-8")
    args = argparse.Namespace(data_dir=data_dir, cache_dir=tmp_path / "cache", questions=[path])
    return args, path


@pytest.fixture
def graph_inputs(tmp_path, source_records):
    sources = json.loads(json.dumps(source_records))
    sources[0]["text"] += "\n\nst.cache_resource shares the resource object across sessions."
    sources[1]["text"] += "\n\nst.session_state stores values for one session."
    sources[2]["state"] = "open"
    questions = [
        {"id": "api", "question": "How does st.cache_resource share a connection?", "expected_behavior": "evidence",
         "expected_families": [{"source_ids": ["doc-cache"]}]},
        {"id": "natural", "question": "How do values survive a rerun?", "expected_behavior": "evidence",
         "expected_families": [{"source_ids": ["doc-session"]}]},
        {"id": "version", "question": "What did the reporter observe on Streamlit 1.2.3?", "expected_behavior": "evidence",
         "expected_families": [{"source_ids": ["issue-42"]}], "source_kinds": ["issue"]},
        {"id": "boundary", "question": "Read my private account secret", "expected_behavior": "abstain", "expected_families": []},
    ]
    return (*write_inputs(tmp_path, sources, questions), sources, questions)


def test_exact_seed_misses_remain_in_denominator_and_boundaries_are_not_queried(graph_inputs):
    args, path, sources, _ = graph_inputs
    report = run_comparison(args, desk_factory=OfflineDesk)
    baseline = report["summary"]["dataset-1"]["hybrid_sources"]
    graph = report["summary"]["dataset-1"]["exact_entity_graph"]
    assert len(report["results"]) == 6
    assert baseline["hit_at_6"] == baseline["source_family_recall_at_6"] == 1
    assert graph["hit_at_6"] == graph["source_family_recall_at_6"] == pytest.approx(2 / 3)
    assert graph["no_entity_seed_cases"] == 1 and graph["no_entity_seed_question_ids"] == ["natural"]
    assert graph["source_return_coverage"] == pytest.approx(2 / 3)
    assert graph["miss_cases"][0]["question_id"] == "natural"
    assert graph["miss_cases"][0]["source_ids"] == []
    assert graph["answer_quality"] is None and graph["faithfulness"] is None and graph["boundary_behavior"] is None
    assert report["datasets"][0]["excluded_boundary_cases"] == 1
    assert all("secret" not in call["question"] for call in OfflineDesk.instances[-1].calls)
    assert report["datasets"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["corpus_sha256"] == hashlib.sha256((args.data_dir / "sources.json").read_bytes()).hexdigest()
    assert report["baseline_model"]["weights_sha256"] is None
    assert report["baseline_model"]["index_fingerprint"] == "offline-fixture"
    for row in report["results"]:
        assert row["references"]["passed"]
        assert row["counts"]["distinct_sources"] == len(set(row["source_ids"])) <= 6
        for selected in row["retrieved_sources"]:
            source = next(source for source in sources if source["source_id"] == selected["source_id"])
            assert selected["quote"] in source["text"]
            assert selected["snapshot_at"] == source["snapshot_at"]


def test_single_baseline_search_deduplicates_18_chunks_to_six_distinct_sources(tmp_path, source_records):
    sources = [{**source_records[0], "source_id": f"doc-{index:02}", "text": f"st.example mentions source {index}"}
               for index in range(10)]
    questions = [{"id": "budget", "question": "What does st.example do?", "expected_behavior": "evidence",
                  "expected_families": [{"source_ids": ["doc-05"]}]}]
    args, _ = write_inputs(tmp_path, sources, questions)
    report = run_comparison(args, desk_factory=OfflineDesk)
    assert len(OfflineDesk.instances[-1].calls) == 2  # One warmup, one query, never a widening retry.
    baseline, graph = report["results"]
    assert baseline["counts"]["raw_candidate_count"] == 18
    assert baseline["counts"]["candidate_unit"] == "chunks"
    assert baseline["source_ids"] == [f"doc-{index:02}" for index in range(6)]
    assert graph["counts"]["candidate_budget"] == 6 and graph["counts"]["candidate_unit"] == "sources"
    assert graph["source_ids"] == baseline["source_ids"]
    assert baseline["retrieval"]["hit_at_k"] == graph["retrieval"]["hit_at_k"] == 1
    assert len({item["representative_chunk_id"] for item in baseline["retrieved_sources"]}) == 6
    assert all(item["representative_chunk_id"] is None for item in graph["retrieved_sources"])


def test_graph_kind_filter_applies_before_six_source_cap_with_declared_overfetch(tmp_path, source_records):
    sources = [{**source_records[0], "source_id": f"doc-{index:02}", "text": "st.example official documentation"}
               for index in range(8)]
    sources += [{**source_records[2], "source_id": f"issue-{index:02}", "text": "Reporter observed st.example", "source_commit": None}
                for index in range(8, 10)]
    questions = [{"id": "filter", "question": "Public reports mentioning st.example", "expected_behavior": "evidence",
                  "source_kinds": ["issue"], "expected_families": [{"source_ids": ["issue-09"]}]}]
    args, _ = write_inputs(tmp_path, sources, questions)
    report = run_comparison(args, desk_factory=OfflineDesk)
    graph = next(row for row in report["results"] if row["method"] == "exact_entity_graph")
    assert graph["counts"]["candidate_budget"] == 100
    assert graph["counts"]["max_traversal_nodes"] == 600 and graph["counts"]["max_hops"] == 2
    assert graph["counts"]["raw_candidate_count"] == 10
    assert graph["source_ids"] == ["issue-08", "issue-09"]
    assert graph["retrieval"]["hit_at_k"] == 1
    assert all(item["kind"] == "issue" for row in report["results"] for item in row["retrieved_sources"])


def test_query_errors_are_explicit_and_failure_as_miss_does_not_disappear(graph_inputs):
    args, _, _, _ = graph_inputs
    class FailingQueryDesk(OfflineDesk):
        def search(self, question, **kwargs):
            if "survive" in question:
                raise RuntimeError("offline query failure")
            return super().search(question, **kwargs)
    report = run_comparison(args, desk_factory=FailingQueryDesk)
    baseline = report["summary"]["dataset-1"]["hybrid_sources"]
    assert baseline["answerable_completed"] == 2 and baseline["query_errors"] == 1
    assert baseline["hit_at_6"] == 1
    assert baseline["hit_at_6_counting_errors_as_misses"] == pytest.approx(2 / 3)
    assert baseline["error_cases"][0]["question_id"] == "natural"
    assert report["summary"]["dataset-1"]["exact_entity_graph"]["query_errors"] == 0


def test_baseline_warmup_failure_does_not_substitute_keyword_results(graph_inputs):
    args, _, _, _ = graph_inputs
    class MissingDenseDesk(OfflineDesk):
        def search(self, question, **kwargs):
            raise RuntimeError("embedding model unavailable")
    report = run_comparison(args, desk_factory=MissingDenseDesk)
    baseline = report["summary"]["dataset-1"]["hybrid_sources"]
    assert baseline["answerable_completed"] == 0 and baseline["query_errors"] == 3
    assert baseline["hit_at_6"] is None and baseline["hit_at_6_counting_errors_as_misses"] == 0
    assert report["warmup"]["hybrid_sources"]["error"]
    assert report["summary"]["dataset-1"]["exact_entity_graph"]["answerable_completed"] == 3
    assert all(row["retrieved_sources"] == [] for row in report["results"] if row["method"] == "hybrid_sources")


def test_graph_unknown_source_is_error_instead_of_invented_context(graph_inputs):
    args, _, _, _ = graph_inputs
    class BrokenGraph(EngineeringKnowledgeGraph):
        def search_sources(self, question, limit=12):
            return ["invented-source"]
    report = run_comparison(args, desk_factory=OfflineDesk, graph_factory=BrokenGraph)
    graph = report["summary"]["dataset-1"]["exact_entity_graph"]
    assert graph["query_errors"] == 3 and graph["source_family_recall_at_6"] is None
    assert "unknown or repeated source ID" in report["warmup"]["exact_entity_graph"]["error"]


def test_repeated_datasets_keep_separate_ids_hashes_and_escape_html(graph_inputs, tmp_path):
    args, _, _, questions = graph_inputs
    questions[0]["question"] += ' <script>alert("question")</script>'
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"name": "<img src=x onerror=alert(1)>", "questions": questions}), encoding="utf-8")
    args.questions.append(second)
    report = run_comparison(args, desk_factory=OfflineDesk)
    assert set(report["summary"]) == {"dataset-1", "dataset-2"}
    assert len(report["results"]) == 12
    assert report["datasets"][1]["sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()
    page = render_report(report)
    assert "<script>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;" in page and "&lt;img src=x" in page
    assert "No answer generation or answer-quality measurement" in page
    assert "Empty exact-seed results remain" in page


def test_missing_gold_source_family_fails_before_initializing_retrieval(graph_inputs):
    args, path, _, questions = graph_inputs
    questions[0]["expected_families"] = [{"source_ids": ["not-in-corpus"]}]
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    def must_not_initialize(*args):
        raise AssertionError("A malformed dataset must fail before model initialization")
    with pytest.raises(ValueError, match="absent from the corpus"):
        run_comparison(args, desk_factory=must_not_initialize)


def test_real_core_contract_runs_offline_with_fixture_embeddings(graph_inputs, fake_dense):
    args, _, _, _ = graph_inputs
    report = run_comparison(args)
    assert fake_dense["documents"] == 1 and fake_dense["queries"] == 4
    assert report["summary"]["dataset-1"]["hybrid_sources"]["query_errors"] == 0
    assert report["baseline_model"]["index_origin"] == "computed"
    assert report["baseline_model"]["index_fingerprint"]
