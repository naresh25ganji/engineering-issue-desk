"""Offline tests for source boundaries, real-reranker contract and experiment scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from triage_desk.embeddings import EmbeddingUnavailable
from triage_desk.retrieval_lab import (
    RERANK_MODEL, RerankerUnavailable, RetrievalLab, fixed_chunks, structured_chunks,
)


def record(text: str) -> dict:
    return {"source_id": "doc-structure", "title": "Structured example", "kind": "documentation",
            "url": "https://docs.example.test/structured", "text": text,
            "source_commit": "fixture", "snapshot_at": "2026-10-18T00:00:00Z", "version_scope": "Test only"}


class OfflineDense:
    def __init__(self, chunks, cache_dir):
        self.chunks, self.cache_dir = chunks, cache_dir
        self.fingerprint = hashlib.sha256("\n".join(c["chunk_id"] for c in chunks).encode()).hexdigest()
        self.calls = 0

    def scores(self, question):
        self.calls += 1
        return np.arange(len(self.chunks), 0, -1, dtype=float) / max(len(self.chunks), 1)


class OfflineReranker:
    def __init__(self, cache_dir):
        self.cache_dir, self.calls = cache_dir, []

    def rerank(self, query, passages):
        self.calls.append((query, passages))
        return [{**p, "score": 0.95 if p["id"].startswith("issue-42:") else .1} for p in passages]


def lab_for(sources, cache_dir):
    return RetrievalLab(sources, cache_dir, dense_factory=OfflineDense, reranker_factory=OfflineReranker)


def test_structure_preserves_literal_code_table_and_heading_metadata():
    code = "```python\ndef do_work():\n    return 42\n```\n"
    table = "| Version | Result |\n| --- | --- |\n| 1.2 | supported |\n"
    source = record("# API\n\nA short description.\n\n" + code + "\n## Compatibility\n\n" + table)
    chunks = structured_chunks([source], target_words=40, max_chars=500)
    assert any(code in chunk["text"] for chunk in chunks)
    assert any(table in chunk["text"] for chunk in chunks)
    assert any(chunk["section"] == "API > Compatibility" for chunk in chunks)
    for chunk in chunks:
        assert chunk["text"] == source["text"][chunk["source_start"]:chunk["source_end"]]
        assert chunk["source_commit"] == source["source_commit"]
        assert chunk["structure_complete"] and not chunk["split_blocks"]


def test_oversized_code_and_table_are_bounded_and_explicitly_partial():
    code = "```python\n" + "".join(f"    value_{i} = result_{i}\n" for i in range(25)) + "```\n"
    table = "| Key | Value |\n| --- | --- |\n" + "".join(f"| key{i} | value{i} |\n" for i in range(25))
    source = record("# Large blocks\n\n" + code + "\n" + table)
    chunks = structured_chunks([source], target_words=20, max_chars=200)
    assert all(chunk["word_count"] <= 20 and chunk["character_count"] <= 200 for chunk in chunks)
    partials = [block for chunk in chunks for block in chunk["blocks"] if block["parts"] > 1]
    assert {b["kind"] for b in partials} >= {"code", "table"}
    assert all(b["split_reason"].startswith("oversized_") for b in partials)
    assert any(chunk["split_blocks"] and not chunk["structure_complete"] for chunk in chunks)
    code_spans = sorted((b["source_start"], b["source_end"]) for b in partials if b["kind"] == "code")
    assert "    value_" in "".join(source["text"][start:end] for start, end in code_spans)
    for chunk in chunks:
        assert chunk["text"] == source["text"][chunk["source_start"]:chunk["source_end"]]


def test_long_sentence_and_single_long_token_have_explicit_hard_boundaries():
    source = record("# Long token\n\n" + "x" * 901)
    chunks = structured_chunks([source], target_words=20, max_chars=200)
    assert all(len(chunk["text"]) <= 200 for chunk in chunks)
    assert any("hard_boundary" in b["split_reason"] for c in chunks for b in c["blocks"])
    assert sum(c["text"].count("x") for c in chunks) == 901


def test_long_prose_prefers_complete_sentence_boundaries():
    sentence = "This sentence has several distinct words to keep its thought together. "
    chunks = structured_chunks([record(sentence * 5)], target_words=20, max_chars=400)
    assert all(chunk["text"].rstrip().endswith(".") for chunk in chunks)
    assert all(chunk["word_count"] <= 20 for chunk in chunks)


def test_unterminated_original_fence_is_not_silently_completed():
    source = record("# Broken example\n\n```python\nx = 1\n")
    chunks = structured_chunks([source])
    assert any(block["unterminated_fence"] for chunk in chunks for block in chunk["blocks"])
    assert "".join(chunk["text"] for chunk in chunks).count("```") == 1
    assert any(not chunk["structure_complete"] for chunk in chunks)


def test_fixed_word_baseline_overlaps_and_can_cut_structure():
    text = " ".join(f"word{i}" for i in range(60))
    source = record(text)
    chunks = fixed_chunks([source], target_words=20, overlap_words=4)
    assert chunks[0]["text"].split()[-4:] == chunks[1]["text"].split()[:4]
    assert all(chunk["word_count"] <= 20 and not chunk["structure_complete"] for chunk in chunks)
    assert chunks == fixed_chunks([source], target_words=20, overlap_words=4)
    assert chunks[0]["chunk_id"] != fixed_chunks([source], target_words=25, overlap_words=4)[0]["chunk_id"]


def test_keyword_mode_never_invokes_dense_or_reranker(source_records, tmp_path):
    lab = lab_for(source_records, tmp_path)
    result = lab.search("cache database connection", mode="keyword")
    assert result and result[0]["source_id"] == "doc-cache"
    assert all(index.calls == 0 for index in lab.dense_by_strategy.values())
    assert lab._reranker is None
    assert all("rerank_score" not in item for item in result)
    assert lab.dense_by_strategy["fixed"].cache_dir == lab.dense_by_strategy["structured"].cache_dir


def test_reranking_uses_real_candidate_identity_and_keeps_first_stage_score(source_records, tmp_path):
    lab = lab_for(source_records, tmp_path)
    baseline = lab.search("widget reset", mode="semantic", top_k=4)
    by_id = {chunk["chunk_id"]: chunk for chunk in baseline}
    result = lab.search("widget reset", mode="semantic", rerank=True, top_k=4)
    assert result[0]["source_id"] == "issue-42"
    assert result[0]["rerank_score"] == .95 and result[0]["rerank_model"] == RERANK_MODEL
    assert result[0]["score"] == by_id[result[0]["chunk_id"]]["score"]
    assert [c["citation_id"] for c in result] == [f"S{i + 1}" for i in range(len(result))]
    assert all(item["quote"] in item["text"] for item in result)
    query, passages = lab._reranker.calls[0]
    assert query == "widget reset" and {p["id"] for p in passages} == set(by_id)


@pytest.mark.parametrize("response", [[], [{"id": "invented", "score": .5}], "not a list"])
def test_reranker_incomplete_or_unknown_response_fails_explicitly(source_records, tmp_path, response):
    class Broken:
        def __init__(self, cache_dir): pass
        def rerank(self, query, passages): return response
    lab = RetrievalLab(source_records, tmp_path, dense_factory=OfflineDense, reranker_factory=Broken)
    with pytest.raises(RerankerUnavailable):
        lab.search("cache database", mode="keyword", rerank=True)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), True, "0.9"])
def test_reranker_nonfinite_or_non_numeric_scores_are_not_accepted(source_records, tmp_path, bad_score):
    class Broken:
        def __init__(self, cache_dir): pass
        def rerank(self, query, passages): return [{**p, "score": bad_score} for p in passages]
    lab = RetrievalLab(source_records, tmp_path, dense_factory=OfflineDense, reranker_factory=Broken)
    with pytest.raises(RerankerUnavailable):
        lab.search("cache database", mode="keyword", rerank=True)


def test_default_flashrank_model_load_failure_does_not_fake_results(source_records, tmp_path, monkeypatch):
    import sys
    class BrokenRanker:
        def __init__(self, **kwargs): raise OSError("offline model unavailable")
    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(Ranker=BrokenRanker, RerankRequest=object))
    lab = RetrievalLab(source_records, tmp_path, dense_factory=OfflineDense)
    with pytest.raises(RerankerUnavailable, match="no|not produced"):
        lab.search("cache database", mode="keyword", rerank=True)


def test_filtering_and_empty_results_do_not_initialize_models(source_records, tmp_path):
    lab = lab_for(source_records, tmp_path)
    assert lab.search("anything", source_kinds=[], rerank=True) == []
    assert lab.search("unfindablequux", mode="keyword", rerank=True) == []
    assert lab._reranker is None
    issues = lab.search("widget", mode="keyword", source_kinds=["issue"])
    assert issues and all(chunk["kind"] == "issue" for chunk in issues)


def test_invalid_options_and_dense_failure_remain_explicit(source_records, tmp_path):
    lab = lab_for(source_records, tmp_path)
    for kwargs in [{"strategy": "semantic"}, {"mode": "mystery"}, {"rerank": "yes"},
                   {"top_k": True}, {"top_k": 8, "candidate_k": 6}, {"source_kinds": ["secret"]}]:
        with pytest.raises(ValueError): lab.search("cache", **kwargs)
    lab.dense_by_strategy["structured"].scores = lambda question: np.array([float("nan")])
    with pytest.raises(EmbeddingUnavailable): lab.search("cache", mode="hybrid")


def test_langchain_document_adapter_preserves_source_metadata(source_records, tmp_path):
    pytest.importorskip("langchain_core.documents")
    lab = lab_for(source_records, tmp_path)
    evidence = lab.search("cache", mode="keyword")
    docs = lab.to_documents(evidence)
    assert docs[0].page_content == evidence[0]["text"]
    assert docs[0].metadata["chunk_id"] == evidence[0]["chunk_id"]
    assert docs[0].metadata["url"] == evidence[0]["url"]


def test_comparison_uses_shared_gold_metrics_and_honest_denominators(corpus_dir, tmp_path):
    from evaluation.retrieval_lab import render_report, run_comparison
    questions = {"name": "Offline test", "questions": [
        {"id": "one", "kind": "answerable", "expected_behavior": "evidence", "question": "cache database",
         "expected_families": [{"source_ids": ["doc-cache"]}]},
        {"id": "two", "kind": "unsupported", "expected_behavior": "abstain", "question": "unavailable private values", "expected_families": []},
    ]}
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions))
    args = argparse.Namespace(data_dir=corpus_dir, cache_dir=tmp_path / "cache", questions=[path],
                              max_questions=None, top_k=4, candidate_k=18)
    report = run_comparison(args, lab_factory=lab_for)
    assert len(report["results"]) == 6
    for values in report["summary"]["dataset-1"].values():
        assert values["answerable_scheduled"] == values["answerable_scored"] == 1
        assert values["boundary_queries"] == 1 and values["boundary_behavior_scored"] is False
        assert values["hit_at_k"] == values["family_recall_at_k"] == 1
        assert values["answer_faithfulness"] is None and values["citation_entailment"] is None
    assert "No answer generation" in render_report(report)
    assert report["datasets"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
