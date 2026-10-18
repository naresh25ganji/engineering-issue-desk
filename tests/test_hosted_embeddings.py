"""No-network hosted embedding tests; configuration and HTTP are always mocked."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

import triage_desk.hosted_embeddings as hosted
from triage_desk.embeddings import DenseIndex, EmbeddingUnavailable
from triage_desk.retrieval_lab import structured_chunks


@pytest.fixture(autouse=True)
def fake_runtime_configuration(monkeypatch):
    monkeypatch.setattr(hosted, "_configuration", lambda: ("fixture-secret-key", "chat-model-not-used", "https://provider.example.test/v1"))


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
    def json(self):
        if isinstance(self.data, Exception): raise self.data
        return self.data


def install_http(monkeypatch, responder):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def post(self, endpoint, *, headers, json):
            calls.append({"endpoint": endpoint, "headers": headers, "json": json})
            return responder(json, len(calls))
    monkeypatch.setattr(hosted.httpx, "Client", Client)
    return calls


def successful(payload, number):
    dimension = payload["dimensions"]
    # Reverse response order to verify index-based reconstruction.
    return Response({"model": "returned-embedding-model", "data": [
        {"index": i, "embedding": [float(i + 1)] + [1.0] * (dimension - 1)}
        for i in reversed(range(len(payload["input"])))],
        "usage": {"prompt_tokens": 18, "total_tokens": 18}})


def make_index(source_records, tmp_path, dimensions=4, model=None):
    return hosted.HostedDenseIndex(structured_chunks(source_records), tmp_path, model=model, dimensions=dimensions)


def test_hosted_identity_is_separate_from_local_endpoint_model_dimensions_and_corpus(source_records, tmp_path, monkeypatch):
    chunks = structured_chunks(source_records)
    local = DenseIndex(chunks, tmp_path, prebuilt_dir=tmp_path / "bundled")
    index = make_index(source_records, tmp_path)
    assert index.fingerprint != local.fingerprint and index.index_path != local.index_path
    assert index.prebuilt_path is None and index.calls == []
    assert make_index(source_records, tmp_path, dimensions=5).fingerprint != index.fingerprint
    assert make_index(source_records, tmp_path, model="another-model").fingerprint != index.fingerprint
    modified = [{**source_records[0], "text": source_records[0]["text"] + " New fact."}] + source_records[1:]
    assert make_index(modified, tmp_path).fingerprint != index.fingerprint
    monkeypatch.setattr(hosted, "_configuration", lambda: ("other-key", "chat", "https://another.example.test/v1"))
    assert make_index(source_records, tmp_path).fingerprint != index.fingerprint


def test_request_fields_actual_usage_out_of_order_indices_and_normalization(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    index = make_index(source_records, tmp_path)
    vectors = index.embed_documents(["alpha", "beta"])
    assert vectors.shape == (2, 4) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert np.allclose(vectors[0], [.5, .5, .5, .5])
    assert vectors[1, 0] > vectors[0, 0]
    assert requests[0]["endpoint"] == "https://provider.example.test/v1/embeddings"
    assert requests[0]["json"] == {"model": hosted.HOSTED_MODEL_NAME, "input": ["alpha", "beta"], "dimensions": 4}
    metadata = index.calls[0]
    assert metadata["requested_model"] == hosted.HOSTED_MODEL_NAME
    assert metadata["returned_model"] == "returned-embedding-model"
    assert metadata["usage"] == {"prompt_tokens": 18, "total_tokens": 18}
    assert metadata["input_count"] == 2 and metadata["status"] == "success" and metadata["latency_ms"] >= 0
    assert "fixture-secret-key" not in json.dumps(index.calls)
    assert "alpha" not in json.dumps(index.calls)


def test_serial_batches_are_bounded_and_coverage_is_complete(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    index = make_index(source_records, tmp_path)
    vectors = index.embed_documents([f"text-{i}" for i in range(65)])
    assert vectors.shape == (65, 4)
    assert [len(r["json"]["input"]) for r in requests] == [32, 32, 1]
    assert [r["input_count"] for r in index.calls] == [32, 32, 1]
    assert sum(r["usage"]["total_tokens"] for r in index.calls) == 54


@pytest.mark.parametrize("data", [
    {}, {"data": []}, {"data": [{"embedding": [1, 1, 1, 1]}]},
    {"data": [{"index": 1, "embedding": [1, 1, 1, 1]}]},
    {"data": [{"index": 0, "embedding": [1, 1, 1]}]},
    {"data": [{"index": 0, "embedding": [1, 1, 1, float("nan")]}]},
    {"data": [{"index": 0, "embedding": [1, 1, 1, float("inf")]}]},
    {"data": [{"index": 0, "embedding": [0, 0, 0, 0]}]},
    {"data": [{"index": 0, "embedding": [True, 1, 1, 1]}]},
    {"data": [{"index": 0, "embedding": ["1", 1, 1, 1]}]},
])
def test_invalid_provider_response_never_becomes_a_vector(source_records, tmp_path, monkeypatch, data):
    install_http(monkeypatch, lambda payload, number: Response(data))
    index = make_index(source_records, tmp_path)
    with pytest.raises(EmbeddingUnavailable): index.embed_query("query")
    assert index.calls[-1]["status"] == "invalid_response"
    assert not index._query_cache


def test_duplicate_indices_and_partial_batch_do_not_save_an_index(source_records, tmp_path, monkeypatch):
    def broken(payload, number):
        return Response({"data": [{"index": 0, "embedding": [1, 1, 1, 1]} for _ in payload["input"]]})
    install_http(monkeypatch, broken)
    index = make_index(source_records, tmp_path)
    with pytest.raises(EmbeddingUnavailable): index.vectors()
    assert not index.index_path.exists() and index._vectors is None


def test_http_errors_never_expose_provider_body_or_key(source_records, tmp_path, monkeypatch):
    secret_body = {"error": "fixture-secret-key body must never be shown"}
    requests = install_http(monkeypatch, lambda payload, number: Response(secret_body, status=401))
    index = make_index(source_records, tmp_path)
    with pytest.raises(EmbeddingUnavailable) as error: index.embed_query("private input")
    assert "HTTP 401" in str(error.value)
    assert "fixture-secret-key" not in str(error.value) + json.dumps(index.calls)
    assert "body must never" not in str(error.value) + json.dumps(index.calls)
    assert requests and index.calls[-1]["http_status"] == 401


def test_network_error_records_only_exception_type(source_records, tmp_path, monkeypatch):
    def broken(payload, number): raise httpx.ConnectError("secret fixture-secret-key in exception")
    install_http(monkeypatch, broken)
    index = make_index(source_records, tmp_path)
    with pytest.raises(EmbeddingUnavailable) as error: index.embed_query("query")
    assert "ConnectError" in str(error.value)
    assert "fixture-secret-key" not in str(error.value) + json.dumps(index.calls)


def test_cached_index_and_query_cache_do_not_make_hidden_document_calls(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    first = make_index(source_records, tmp_path)
    first.vectors()
    assert len(requests) == 1
    second = make_index(source_records, tmp_path)
    second.vectors()
    assert second.cache_hit and len(requests) == 1 and second.calls == []
    score = second.scores("cache resource")
    again = second.scores("cache resource")
    assert score.shape == (len(second.chunks),) and np.allclose(score, again)
    assert len(requests) == 2 and len(second.calls) == 1 and second.calls[0]["kind"] == "query"
    query = second.embed_query("cache resource")
    query[:] = 0
    assert np.linalg.norm(second.embed_query("cache resource")) > .9


def test_wrong_dimension_derived_cache_is_rebuilt_by_hosted_provider(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    index = make_index(source_records, tmp_path)
    tmp_path.mkdir(exist_ok=True)
    np.savez_compressed(index.index_path, vectors=np.ones((len(index.chunks), 3)), fingerprint=np.asarray(index.fingerprint))
    assert index.vectors().shape == (len(index.chunks), 4)
    assert len(requests) == 1 and not index.cache_hit and index.index_origin == "computed"


def test_changed_endpoint_invalidates_even_the_query_memory_cache(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    index = make_index(source_records, tmp_path)
    index.embed_query("query")
    monkeypatch.setattr(hosted, "_configuration", lambda: ("fixture-secret-key", "chat", "https://different.example.test/v1"))
    with pytest.raises(EmbeddingUnavailable, match="endpoint changed"): index.embed_query("query")
    assert len(requests) == 1


def test_query_cache_is_bounded_and_does_not_persist_query_text(source_records, tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    monkeypatch.setattr(hosted, "QUERY_CACHE_SIZE", 3)
    index = make_index(source_records, tmp_path)
    for i in range(4): index.embed_query(f"private query {i}")
    assert len(index._query_cache) == 3
    assert all(len(key) == 64 and "private" not in key for key in index._query_cache)
    index.embed_query("private query 0")
    assert len(requests) == 5


def test_empty_index_and_empty_documents_have_correct_shape_without_requests(tmp_path, monkeypatch):
    requests = install_http(monkeypatch, successful)
    index = hosted.HostedDenseIndex([], tmp_path, dimensions=7)
    assert index.vectors().shape == (0, 7)
    assert index.embed_documents([]).shape == (0, 7)
    assert index.scores("question").shape == (0,)
    assert requests == [] and index.calls == []


@pytest.mark.parametrize("dimensions", [0, -1, True, 1.5])
def test_invalid_dimensions_are_rejected_before_any_request(source_records, tmp_path, dimensions):
    with pytest.raises(ValueError): make_index(source_records, tmp_path, dimensions=dimensions)
