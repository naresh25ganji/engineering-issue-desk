# Frozen source excerpt: tests/test_hosted_embeddings.py:89-114
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
