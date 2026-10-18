import json

import httpx
import pytest

from triage_desk.generation import GenerationUnavailable, _validated_claims, _validated_verdicts, provider_status, quote_spans, synthesize


@pytest.fixture
def cited_evidence():
    return [{"citation_id": "S1", "title": "Resource caching", "kind": "documentation", "version_scope": "test snapshot",
             "text": "Cached resources are shared across users and sessions. Objects must be thread safe."}]


def test_generated_claim_requires_a_real_passage_and_citation(cited_evidence):
    payload = json.dumps({"claims": [{"statement": "Resources can be shared across users.",
        "support": [{"citation_id": "S1", "quote": "Cached resources are shared across users and sessions."}]}]})
    accepted = _validated_claims(payload, cited_evidence)
    assert accepted[0]["citation_ids"] == ["S1"]


@pytest.mark.parametrize("support", [
    [{"citation_id": "S99", "quote": "Cached resources are shared across users and sessions."}],
    [{"citation_id": "S1", "quote": "The documentation guarantees that your application is fixed."}],
    [],
])
def test_unknown_citations_and_fabricated_quotes_are_rejected(cited_evidence, support):
    payload = json.dumps({"claims": [{"statement": "A claim.", "support": support}]})
    with pytest.raises(GenerationUnavailable):
        _validated_claims(payload, cited_evidence)


def test_source_quote_does_not_authorize_confirmed_fix_claim(cited_evidence):
    payload = json.dumps({"claims": [{"statement": "This is a duplicate and your issue is fixed.",
        "support": [{"citation_id": "S1", "quote": "Cached resources are shared across users and sessions."}]}]})
    with pytest.raises(GenerationUnavailable, match="overstated"):
        _validated_claims(payload, cited_evidence)


def test_provider_status_never_exposes_credential(monkeypatch):
    monkeypatch.setenv("TRIAGE_API_KEY", "private-unit-test-token")
    monkeypatch.setenv("TRIAGE_MODEL", "example-model")
    monkeypatch.setenv("TRIAGE_BASE_URL", "https://api.example.test/v1")
    state = provider_status()
    assert state["configured"]
    assert "private-unit-test-token" not in json.dumps(state)


def test_unconfigured_provider_is_explicit(monkeypatch):
    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    state = provider_status()
    assert not state["configured"]
    assert "TRIAGE_API_KEY" in state["reason"]


@pytest.fixture
def mock_provider(monkeypatch):
    """Exercise the real HTTP/response path entirely in memory, with fake credentials."""
    monkeypatch.setenv("TRIAGE_API_KEY", "private-unit-test-token")
    monkeypatch.setenv("TRIAGE_MODEL", "test/requested-model")
    monkeypatch.setenv("TRIAGE_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    original_client = httpx.Client
    requests = []

    def install(body, status=200):
        responses = iter(body) if isinstance(body, list) else None
        def handler(request):
            requests.append({"url": str(request.url), "body": json.loads(request.content)})
            response_body = next(responses) if responses is not None else body
            if isinstance(response_body, Exception):
                raise response_body
            return httpx.Response(status, json=response_body)
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr("triage_desk.generation.httpx.Client", lambda **kwargs: original_client(transport=transport, **kwargs))
        return requests
    return install


def _completion(claims, finish_reason="stop"):
    return {
        "model": "test/returned-model",
        "choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps({"claims": claims})}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
    }


def _supported_claim():
    return {"statement": "Resources can be shared across users.",
            "support": [{"citation_id": "S1", "quote": "Cached resources are shared across users and sessions."}]}


def _supported_id_claim():
    return {"statement": "Resources can be shared across users.",
            "support": [{"citation_id": "S1", "quote_id": "S1.Q1"}]}


def _audit_completion(verdicts=None, finish_reason="stop"):
    if verdicts is None:
        verdicts = [{"claim_index": 0, "supported": True, "reason": "The selected quote states this."}]
    return {"model": "test/verifier-model",
            "choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps({"verdicts": verdicts})}}],
            "usage": {"prompt_tokens": 70, "completion_tokens": 20, "total_tokens": 90}}


def test_http_success_retains_actual_support_model_usage_and_legacy_string(mock_provider, cited_evidence):
    requests = mock_provider([_completion([_supported_id_claim()]), _audit_completion(),
                              _completion([_supported_id_claim()]), _audit_completion()])
    result = synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    assert result["answer"].endswith("[S1]")
    assert result["claims"][0]["support"][0]["quote"] in cited_evidence[0]["text"]
    assert result["claims"][0]["support"][0]["quote_id"] == "S1.Q1"
    meta = result["metadata"]
    assert meta["requested_model"] == "test/requested-model"
    assert meta["returned_model"] == "test/returned-model"
    assert meta["usage"]["total_tokens"] == 290
    assert meta["finish_reason"] == "stop"
    assert meta["attempted"] is True and meta["status"] == "success"
    assert meta["latency_ms"] >= 0
    assert meta["endpoint"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert "private-unit-test-token" not in json.dumps(result)
    assert requests[0]["body"]["response_format"] == {"type": "json_object"}
    sent_evidence = json.loads(requests[0]["body"]["messages"][1]["content"])["evidence"][0]
    assert "text" not in sent_evidence
    assert sent_evidence["quotes"] == quote_spans(cited_evidence[0]["text"], "S1")
    assert meta["protocol"] == "quote_ids_batch_verifier_v3"
    assert len(meta["system_prompt_sha256"]) == 64
    assert meta["request_options"] == {"temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"}}
    assert meta["usage_complete"] is True
    assert [call["stage"] for call in meta["calls"]] == ["synthesis", "verification"]
    assert meta["calls"][1]["returned_model"] == "test/verifier-model"
    assert meta["verification"]["accepted_claims"] == 1
    assert len(meta["verification_prompt_sha256"]) == 64
    audit_input = json.loads(requests[1]["body"]["messages"][1]["content"])
    assert set(audit_input) == {"claims"}
    assert audit_input["claims"][0]["support"][0]["quote"] == cited_evidence[0]["text"]
    assert "source_kind" in audit_input["claims"][0]["support"][0]
    assert isinstance(synthesize("When are resources shared?", cited_evidence), str)


def test_generic_provider_does_not_assume_nebius_json_option(mock_provider, cited_evidence, monkeypatch):
    requests = mock_provider([_completion([_supported_id_claim()]), _audit_completion()])
    monkeypatch.setenv("TRIAGE_BASE_URL", "https://api.example.test/v1")
    result = synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    assert "response_format" not in requests[0]["body"]
    assert result["metadata"]["response_format"] == "prompted_json"


def test_truncated_completion_rejected_even_if_json_parseable_and_usage_preserved(mock_provider, cited_evidence):
    mock_provider(_completion([_supported_id_claim()], finish_reason="length"))
    with pytest.raises(GenerationUnavailable, match="output-token limit") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    assert error.value.reason == "output_truncated"
    meta = error.value.metadata
    assert meta["usage"]["total_tokens"] == 200
    assert meta["finish_reason"] == "length"
    assert meta["returned_model"] == "test/returned-model"
    assert meta["attempted"] is True and meta["latency_ms"] >= 0


def test_valid_empty_claims_is_abstention_with_usage_not_provider_failure(mock_provider, cited_evidence):
    requests = mock_provider(_completion([]))
    result = synthesize("A question not supported by these passages", cited_evidence, return_metadata=True)
    assert result["answer"] == "" and result["claims"] == []
    assert result["metadata"]["status"] == "abstained"
    assert result["metadata"]["usage"]["total_tokens"] == 200
    assert len(requests) == 1
    assert result["metadata"]["verification"]["attempted"] is False


def test_rejected_paid_answer_keeps_usage_and_model_metadata(mock_provider, cited_evidence):
    claim = _supported_id_claim()
    claim["support"][0]["citation_id"] = "S99"
    mock_provider(_completion([claim]))
    with pytest.raises(GenerationUnavailable, match="unknown") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    assert error.value.reason == "answer_rejected"
    assert error.value.metadata["usage"]["total_tokens"] == 200
    assert error.value.metadata["requested_model"] == "test/requested-model"
    assert error.value.metadata["returned_model"] == "test/returned-model"
    assert error.value.metadata["finish_reason"] == "stop"


@pytest.mark.parametrize("statement", [
    "Read [this page](/develop/api-reference).", "See https://invented.example.test/fix.",
    "<img src='https://invented.example.test/tracker'>", "Use <script>alert('x')</script>.",
    "![figure](relative-image.png)", "```python\nprint('a code block')\n```",
])
def test_model_links_html_and_fenced_blocks_are_rejected(cited_evidence, statement):
    claim = _supported_claim()
    claim["statement"] = statement
    with pytest.raises(GenerationUnavailable, match="links, HTML"):
        _validated_claims(json.dumps({"claims": [claim]}), cited_evidence)


def test_quote_spans_are_stable_literal_overlapping_and_cover_the_full_passage():
    words = [f"word{i:03d}" for i in range(180)]
    passage = "\n  " + "\n\t".join(words) + "  \n"
    spans = quote_spans(passage, "S7")
    assert spans == quote_spans(passage, "S7")
    assert [span["quote_id"] for span in spans] == ["S7.Q1", "S7.Q2", "S7.Q3", "S7.Q4"]
    assert all(span["text"] in passage for span in spans)
    assert {word for span in spans for word in span["text"].split()} == set(words)
    assert spans[0]["text"].split()[-10:] == spans[1]["text"].split()[:10]
    assert spans[0]["text"].startswith("\n  ") and spans[-1]["text"].endswith("  \n")


def test_known_quote_id_resolves_to_exact_original_text(cited_evidence):
    payload = json.dumps({"claims": [_supported_id_claim()]})
    claims = _validated_claims(payload, cited_evidence, require_quote_ids=True)
    assert claims[0]["support"] == [{"citation_id": "S1", "quote_id": "S1.Q1", "quote": cited_evidence[0]["text"]}]


@pytest.mark.parametrize("support", [
    {"citation_id": "S1", "quote_id": "S1.Q99"},
    {"citation_id": "S1", "quote_id": "S2.Q1"},
    {"citation_id": "S1", "quote": "Cached resources are shared across users and sessions."},
    {"citation_id": "S1", "quote_id": "S1.Q1", "quote": "Cached resources shared across users."},
])
def test_live_protocol_rejects_unknown_mismatched_missing_or_altered_quote_ids(cited_evidence, support):
    evidence = cited_evidence + [{"citation_id": "S2", "text": "A different source explains a separate topic."}]
    claim = _supported_id_claim()
    claim["support"] = [support]
    with pytest.raises(GenerationUnavailable):
        _validated_claims(json.dumps({"claims": [claim]}), evidence, require_quote_ids=True)


def test_legacy_quote_parser_does_not_accept_fuzzy_transcriptions(cited_evidence):
    claim = _supported_claim()
    claim["support"][0]["quote"] = "Cached resources shared across users and sessions."
    with pytest.raises(GenerationUnavailable, match="not present"):
        _validated_claims(json.dumps({"claims": [claim]}), cited_evidence)


def test_question_version_cannot_become_a_claim_without_selected_source_support(mock_provider, cited_evidence):
    claim = _supported_id_claim()
    claim["statement"] = "Shared caching was introduced in Streamlit 1.50.0."
    mock_provider(_completion([claim]))
    with pytest.raises(GenerationUnavailable, match="numerical or version") as error:
        synthesize("I use version 1.50.0; when was shared caching introduced?", cited_evidence, return_metadata=True)
    assert error.value.metadata["usage"]["total_tokens"] == 200


def test_version_literal_can_be_claimed_when_selected_source_span_contains_it(cited_evidence):
    evidence = [{**cited_evidence[0], "text": "Since version 1.50.0, this example describes a shared cached resource."}]
    claim = _supported_id_claim()
    claim["statement"] = "This example describes a shared cached resource since version 1.50.0."
    assert _validated_claims(json.dumps({"claims": [claim]}), evidence, require_quote_ids=True)


def test_verifier_rejection_fails_closed_and_keeps_both_paid_calls(mock_provider, cited_evidence):
    rejected = [{"claim_index": 0, "supported": False, "reason": "The source does not support the entire assertion."}]
    requests = mock_provider([_completion([_supported_id_claim()]), _audit_completion(rejected)])
    with pytest.raises(GenerationUnavailable, match="rejected every") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    meta = error.value.metadata
    assert error.value.reason == "verification_rejected"
    assert len(requests) == 2
    assert meta["usage"]["total_tokens"] == 290 and meta["usage_complete"] is True
    assert meta["verification"]["accepted_claims"] == 0
    assert meta["verification"]["rejected_claims"] == 1
    assert meta["verification"]["verdicts"][0]["statement"] == _supported_id_claim()["statement"]
    assert [call["status"] for call in meta["calls"]] == ["success", "success"]


def test_partial_verification_omits_unsupported_claim_without_repair(mock_provider, cited_evidence):
    supported = _supported_id_claim()
    unsupported = {"statement": "Resources automatically encrypt data.", "support": supported["support"]}
    verdicts = [{"claim_index": 0, "supported": True, "reason": "The quote establishes shared resources."},
                {"claim_index": 1, "supported": False, "reason": "The quote does not mention encryption."}]
    requests = mock_provider([_completion([supported, unsupported]), _audit_completion(verdicts)])
    result = synthesize("How are resources handled?", cited_evidence, return_metadata=True)
    assert len(requests) == 2
    assert len(result["claims"]) == 1 and result["claims"][0]["statement"] == supported["statement"]
    assert result["claims"][0]["draft_claim_index"] == 0
    assert "retained 1 of 2 draft claims" in result["answer"]
    assert "encrypt" not in result["answer"]
    meta = result["metadata"]
    assert meta["status"] == "success_partial"
    assert meta["verification"]["accepted_claims"] == 1 and meta["verification"]["rejected_claims"] == 1
    assert meta["verification"]["verdicts"][1]["statement"] == unsupported["statement"]


@pytest.mark.parametrize("verdicts", [
    [],
    [{"claim_index": 0, "supported": True, "reason": "Supported."}],
    [{"claim_index": 0, "supported": True, "reason": "Supported."},
     {"claim_index": 0, "supported": True, "reason": "Repeated index."}],
    [{"claim_index": 0, "supported": True, "reason": "Supported."},
     {"claim_index": 2, "supported": True, "reason": "Unknown index."}],
    [{"claim_index": 0, "supported": "true", "reason": "Not a JSON boolean."},
     {"claim_index": 1, "supported": True, "reason": "Supported."}],
    [{"claim_index": False, "supported": True, "reason": "Not an integer index."},
     {"claim_index": 1, "supported": True, "reason": "Supported."}],
    [{"claim_index": 0, "supported": True, "reason": "x" * 401},
     {"claim_index": 1, "supported": True, "reason": "Supported."}],
])
def test_incomplete_duplicate_unknown_or_malformed_verdicts_never_approve_claims(mock_provider, cited_evidence, verdicts):
    mock_provider([_completion([_supported_id_claim(), _supported_id_claim()]), _audit_completion(verdicts)])
    with pytest.raises(GenerationUnavailable, match="incomplete or malformed") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    assert error.value.reason == "verification_unavailable"
    meta = error.value.metadata
    assert meta["calls"][1]["status"] == "invalid_verdicts"
    assert meta["usage"]["total_tokens"] == 290
    assert meta["verification"]["accepted_claims"] is None
    assert meta["verification"]["rejected_claims"] is None


def test_verifier_timeout_fails_closed_without_inventing_missing_usage(mock_provider, cited_evidence):
    requests = mock_provider([_completion([_supported_id_claim()]), httpx.ReadTimeout("Mock verifier timeout")])
    with pytest.raises(GenerationUnavailable, match="verification call failed") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    meta = error.value.metadata
    assert error.value.reason == "verification_unavailable" and len(requests) == 2
    assert meta["calls"][1]["attempted"] is True
    assert meta["calls"][1]["status"] == "timeout"
    assert meta["calls"][1]["usage"] == {}
    assert meta["usage"]["total_tokens"] == 200 and meta["usage_complete"] is False
    assert meta["latency_ms"] >= max(call["latency_ms"] for call in meta["calls"])


def test_verifier_truncation_fails_closed_even_for_parseable_verdicts(mock_provider, cited_evidence):
    mock_provider([_completion([_supported_id_claim()]), _audit_completion(finish_reason="length")])
    with pytest.raises(GenerationUnavailable, match="verification model reached") as error:
        synthesize("When are resources shared?", cited_evidence, return_metadata=True)
    meta = error.value.metadata
    assert error.value.reason == "verification_unavailable"
    assert meta["calls"][1]["finish_reason"] == "length"
    assert meta["calls"][1]["status"] == "output_truncated"
    assert meta["usage"]["total_tokens"] == 290 and meta["usage_complete"] is True


def test_verdict_parser_does_not_use_model_rewritten_statements():
    payload = json.dumps({"verdicts": [{"claim_index": 0, "supported": True, "reason": "Supported.",
                                       "statement": "A model rewrite must not replace the original."}]})
    result = _validated_verdicts(payload, 1)
    assert result == [{"claim_index": 0, "supported": True, "reason": "Supported."}]
