"""Offline checks for provider contracts, immutable evidence, and tool boundaries."""

from contextlib import contextmanager
import json

import httpx
import pytest

from triage_desk.agent_budget import BudgetExceeded
from triage_desk.agent_tools import MAX_RESPONSE_BYTES, build_investigation_tools
from triage_desk.core import TriageDesk


class Registry:
    def __init__(self):
        self.records = []

    def register_source(self, **record):
        evidence = {"evidence_id": f"E{len(self.records) + 1}", **record}
        self.records.append(evidence)
        return evidence


class Budget:
    def __init__(self, maximum=30):
        self.maximum = maximum
        self.events = []

    @contextmanager
    def tool_call(self, name, arguments):
        if len(self.events) >= self.maximum:
            raise BudgetExceeded("Tool budget exhausted")
        event = {"tool": name, "arguments": arguments}
        self.events.append(event)
        yield event


@pytest.fixture
def make_tools():
    def build(handler=None, *, desk=None, registry=None, budget=None, **kwargs):
        requests = []
        registry = registry or Registry()
        budget = budget or Budget()

        def wrapped(request):
            requests.append(request)
            if handler is None:
                pytest.fail("Unexpected network request")
            return handler(request)

        tools = build_investigation_tools(
            desk, registry, budget,
            client_factory=lambda **options: httpx.Client(transport=httpx.MockTransport(wrapped), **options),
            **kwargs,
        )
        return {tool.name: tool for tool in tools}, registry, budget, requests
    return build


def test_knowledge_base_uses_keyword_search_and_preserves_snapshot(make_tools, corpus_dir, tmp_path, fake_dense):
    desk = TriageDesk(corpus_dir, tmp_path / "cache")
    tools, registry, budget, requests = make_tools(desk=desk)
    result = tools["search_knowledge_base"].invoke({"query": "cache shared database resource"})
    evidence = result["evidence"][0]
    source = next(s for s in desk.sources if s["source_id"] == evidence["source_id"])
    assert evidence["text"] in source["text"]
    assert evidence["snapshot_at"] == source["snapshot_at"]
    assert evidence["version_scope"] == source["version_scope"]
    assert evidence["origin"] == "snapshot"
    assert evidence["commit_sha"] == source["source_commit"]
    assert evidence["fetched_at"] == ""
    assert registry.records
    assert fake_dense == {"documents": 0, "queries": 0}
    assert budget.events[0]["status"] == "ok"
    assert not requests


def test_tool_schemas_do_not_expose_credentials_or_provider_urls(make_tools):
    tools, *_ = make_tools()
    assert set(tools) == {"search_knowledge_base", "search_web", "fetch_web_page", "read_github_issue", "read_github_pr"}
    assert tools["search_web"].args_schema.model_json_schema()["properties"].keys() == {"query"}
    assert tools["read_github_pr"].args_schema.model_json_schema()["properties"].keys() == {"pr_number"}
    with pytest.raises(ValueError):
        tools["read_github_issue"].invoke({"issue_number": True})


def test_missing_keenable_key_returns_recoverable_error_without_http(make_tools):
    tools, registry, budget, requests = make_tools()
    for name, arguments in [("search_web", {"query": "widget reset"}),
                            ("fetch_web_page", {"url": "https://docs.streamlit.io/"})]:
        result = tools[name].invoke(arguments)
        assert result["status"] == "error"
        assert "KEENABLE_API_KEY" in result["error"]
    assert not registry.records and not requests
    assert all(event["status"] == "error" for event in budget.events)


def test_keenable_search_matches_api_and_filters_unapproved_results(make_tools):
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "https://api.keenable.ai/v1/search"
        assert request.headers["X-API-Key"] == "test-keenable-key"
        payload = json.loads(request.content)
        assert payload["query"].startswith("widget reset ")
        assert "site:docs.streamlit.io" in payload["query"]
        assert payload["max_results"] == 10
        assert payload["snippet_max_length"] == 1200
        assert "prompt" not in payload
        return httpx.Response(200, json={"results": [
            {"title": "Official docs", "url": "https://docs.streamlit.io/develop/api-reference", "snippet": "Literal snippet.", "acquired_at": "2026-10-20T00:00:00Z"},
            {"title": "Other repo", "url": "https://github.com/other/repo/issues/1", "snippet": "Wrong repository."},
            {"title": "Lookalike", "url": "https://docs.streamlit.io.evil.example/", "snippet": "Wrong host."},
            {"title": "No text", "url": "https://docs.python.org/3/"},
        ]})
    tools, registry, _, requests = make_tools(handler, keenable_api_key="test-keenable-key")
    result = tools["search_web"].invoke({"query": "widget reset"})
    assert result["status"] == "ok"
    assert len(result["evidence"]) == len(registry.records) == len(requests) == 1
    record = result["evidence"][0]
    assert record["text"] == "Literal snippet."
    assert record["evidence_type"] == "search_snippet"
    assert record["acquired_at"] == "2026-10-20T00:00:00Z"
    assert record["fetched_at"].endswith("Z")
    assert record["origin"] == "live"


@pytest.mark.parametrize("url", [
    "http://docs.streamlit.io/", "https://localhost/", "https://127.0.0.1/",
    "https://169.254.169.254/latest/meta-data/", "https://[::1]/", "file:///etc/passwd",
    "https://docs.streamlit.io.evil.test/", "https://user:pass@docs.streamlit.io/",
    "https://@docs.streamlit.io/", "https://docs.streamlit.io/\\private", "https://docs.python.org/\x7f",
    "https://docs.streamlit.io:8443/", "https://github.com/streamlit/streamlit-extra/",
    "https://github.com/streamlit/streamlit/../../private", "https://github.com/streamlit/streamlit/%2e%2e/private",
])
def test_fetch_rejects_unapproved_targets_before_request(make_tools, url):
    tools, registry, _, requests = make_tools(keenable_api_key="test-key")
    result = tools["fetch_web_page"].invoke({"url": url})
    assert result["status"] == "error"
    assert not registry.records and not requests


def test_fetch_uses_fixed_provider_and_literal_content_without_extraction_prompt(make_tools):
    content = "# Documentation\n\nA literal paragraph.\n"
    def handler(request):
        assert request.url.host == "api.keenable.ai"
        assert request.url.path == "/v1/fetch"
        assert request.url.params["url"] == "https://docs.python.org/3/library/asyncio.html"
        assert request.url.params["live"] == "true"
        assert request.url.params["max_chars"] == "6000"
        assert "prompt" not in request.url.params
        return httpx.Response(200, json={"url": "https://docs.python.org/3/library/asyncio.html",
                                        "title": "Asyncio", "content": content})
    tools, registry, *_ = make_tools(handler, keenable_api_key="test-key")
    result = tools["fetch_web_page"].invoke({"url": "https://docs.python.org/3/library/asyncio.html"})
    assert result["status"] == "ok"
    assert registry.records[0]["text"] == content
    assert result["evidence"][0]["evidence_type"] == "page_content"


@pytest.mark.parametrize(('path', 'kind', 'source_type'), [
    ('issues/123#issuecomment-456', 'issue', 'github_issue'),
    ('pull/123/files', 'issue', 'github_pull_request'),
    ('discussions/123', 'issue', 'github_discussion'),
    ('blob/main/lib/streamlit/example.py', 'code', 'github_repository'),
    ('', 'code', 'github_repository'),
    ('releases/tag/1.50.0', 'release', 'github_release'),
])
@pytest.mark.parametrize('tool_name', ['search_web', 'fetch_web_page'])
def test_keenable_github_authority_and_format_survive_saved_evidence(
    make_tools, tmp_path, path, kind, source_type, tool_name,
):
    from triage_desk.agent_evidence import EvidenceRegistry
    from triage_desk.cases import CaseStore

    database = tmp_path / 'cases.sqlite'
    store = CaseStore(database)
    case_id = store.create_case('Inspect public repository evidence')['case_id']
    registry = EvidenceRegistry(store, case_id)
    url = 'https://github.com/streamlit/streamlit/' + path
    text = 'Literal repository context; this does not establish a verified fix.'

    def handler(request):
        if request.url.path.endswith('/search'):
            return httpx.Response(200, json={'results': [{'url': url, 'title': 'Repository page', 'snippet': text}]})
        return httpx.Response(200, json={'url': url, 'title': 'Repository page', 'content': text})

    tools, *_ = make_tools(handler, registry=registry, keenable_api_key='test-key')
    arguments = {'query': 'widget report'} if tool_name == 'search_web' else {'url': url}
    result = tools[tool_name].invoke(arguments)
    assert result['status'] == 'ok'
    saved = CaseStore(database).evidence(case_id)[0]
    suffix = 'search_snippet' if tool_name == 'search_web' else 'page_content'
    assert result['evidence'][0]['kind'] == saved['kind'] == kind
    assert saved['kind'] != 'documentation'
    assert saved['evidence_type'] == f'{source_type}_{suffix}'
    assert saved['text'] == text and saved['url'] == url
    assert 'attributed report' in result['notes'] or 'attributed reporting' in result['notes']


def test_fetch_rejects_unapproved_returned_url(make_tools):
    tools, registry, *_ = make_tools(lambda _: httpx.Response(200, json={
        "url": "https://evil.example/redirect", "content": "Should not become evidence"}), keenable_api_key="test-key")
    result = tools["fetch_web_page"].invoke({"url": "https://docs.streamlit.io/"})
    assert result["status"] == "error"
    assert not registry.records


@pytest.mark.parametrize("status", [301, 401, 402, 403, 404, 429, 500])
def test_provider_errors_are_sanitized_and_redirects_are_not_followed(make_tools, status):
    def handler(_):
        return httpx.Response(status, text="Secret key credential-provider-secret", headers={"location": "https://evil.example/"})
    tools, registry, _, requests = make_tools(handler, keenable_api_key="credential-provider-secret")
    result = tools["search_web"].invoke({"query": "widget"})
    assert result["status"] == "error"
    assert "credential-provider-secret" not in json.dumps(result)
    assert len(requests) == 1 and not registry.records


def test_network_timeout_is_recoverable_without_leaking_exception(make_tools):
    def handler(request):
        raise httpx.ReadTimeout("credential-in-exception", request=request)
    tools, registry, *_ = make_tools(handler, keenable_api_key="test-key")
    result = tools["search_web"].invoke({"query": "widget"})
    assert result["status"] == "error" and "timed out" in result["error"]
    assert "credential-in-exception" not in json.dumps(result)
    assert not registry.records


def test_response_size_limit_prevents_registering_large_provider_response(make_tools):
    tools, registry, *_ = make_tools(
        lambda _: httpx.Response(200, content=b" " * (MAX_RESPONSE_BYTES + 1)), keenable_api_key="test-key")
    result = tools["search_web"].invoke({"query": "widget"})
    assert result["status"] == "error" and "size limit" in result["error"]
    assert not registry.records


def test_github_issue_is_fixed_repo_and_comments_are_bounded(make_tools):
    def handler(request):
        assert request.url.host == "api.github.com"
        assert request.headers["Authorization"] == "Bearer private-test-token"
        if request.url.path.endswith("/comments"):
            assert request.url.params["per_page"] == "5"
            return httpx.Response(200, json=[{"id": i, "body": f"Comment {i}", "updated_at": "2026-10-21T00:00:00Z"} for i in range(10)])
        assert request.url.path == "/repos/streamlit/streamlit/issues/42"
        return httpx.Response(200, json={"number": 42, "title": "A report", "body": "Literal issue body", "state": "open", "comments": 10})
    tools, registry, _, requests = make_tools(handler, github_token="private-test-token")
    result = tools["read_github_issue"].invoke({"issue_number": 42})
    assert result["status"] == "ok"
    assert len(registry.records) == 6 and len(requests) == 2
    assert registry.records[0]["text"] == "Literal issue body"
    assert all(record["origin"] == "live" and record["fetched_at"] for record in registry.records)
    assert "private-test-token" not in json.dumps(result)
    assert "unverified claims" in result["notes"]


def test_github_comment_failure_retains_issue_evidence(make_tools):
    def handler(request):
        if request.url.path.endswith("/comments"):
            return httpx.Response(503, text="secret-provider-message")
        return httpx.Response(200, json={"number": 42, "title": "Report", "body": "Issue body", "comments": 2})
    tools, registry, *_ = make_tools(handler)
    result = tools["read_github_issue"].invoke({"issue_number": 42})
    assert result["status"] == "ok" and len(registry.records) == 1
    assert "could not be read" in result["notes"]
    assert "secret-provider-message" not in json.dumps(result)


def test_github_issue_points_pull_requests_to_pinned_tool(make_tools):
    tools, registry, *_ = make_tools(lambda _: httpx.Response(200, json={"number": 42, "pull_request": {"url": "irrelevant"}}))
    result = tools["read_github_issue"].invoke({"issue_number": 42})
    assert result["status"] == "error" and "read_github_pr" in result["error"]
    assert not registry.records


def test_github_pr_compares_immutable_shas_and_limits_patch_evidence(make_tools):
    base, head = "a" * 40, "b" * 40
    def handler(request):
        if "/compare/" in request.url.path:
            assert request.url.path == f"/repos/streamlit/streamlit/compare/{base}...{head}"
            return httpx.Response(200, json={"files": [{"filename": f"lib/file{i}.py", "patch": "+literal patch\n" * 800} for i in range(10)]})
        assert request.url.path == "/repos/streamlit/streamlit/pulls/88"
        return httpx.Response(200, json={"number": 88, "title": "Fix widget", "body": "PR description", "state": "open", "base": {"sha": base}, "head": {"sha": head}})
    tools, registry, _, requests = make_tools(handler)
    result = tools["read_github_pr"].invoke({"pr_number": 88})
    assert result["status"] == "ok" and len(requests) == 2
    assert len(registry.records) == 6
    assert result["head_sha"] == head and result["base_sha"] == base
    assert result["patches_read"] == 5
    patches = result["evidence"][1:]
    assert all(e["commit_sha"] == head and e["base_sha"] == base and e["truncated"] for e in patches)
    assert all(len(e["text"]) == 6000 for e in patches)
    assert all(f"/compare/{base}...{head}" in e["url"] for e in patches)
    assert all("/files" not in request.url.path for request in requests)
    assert "No tests or code were executed" in result["notes"]


def test_github_pr_rejects_missing_or_injected_sha(make_tools):
    tools, registry, _, requests = make_tools(lambda _: httpx.Response(200, json={
        "number": 88, "base": {"sha": "a" * 40}, "head": {"sha": "../../secrets"}}))
    result = tools["read_github_pr"].invoke({"pr_number": 88})
    assert result["status"] == "error" and "immutable" in result["error"]
    assert len(requests) == 1 and not registry.records


def test_budget_exhaustion_propagates_instead_of_looking_like_tool_recovery(make_tools):
    tools, _, _, requests = make_tools(budget=Budget(maximum=0))
    with pytest.raises(BudgetExceeded):
        tools["search_web"].invoke({"query": "widget"})
    assert not requests


def test_web_provenance_is_persisted_by_real_registry(make_tools, tmp_path):
    from triage_desk.agent_evidence import EvidenceRegistry
    from triage_desk.cases import CaseStore

    store = CaseStore(tmp_path / "cases.sqlite")
    case = store.create_case("Investigate widget behavior")
    registry = EvidenceRegistry(store, case["case_id"])
    # Long approved URLs are valid citations and cannot exceed source-key limits.
    url = "https://docs.streamlit.io/develop/" + "a" * 1100
    tools, _, *_ = make_tools(lambda _: httpx.Response(200, json={"results": [{
        "title": "Docs", "url": url,
        "snippet": "A literal, bounded source passage from official documentation.",
        "published_at": "2026-10-20T00:00:00Z", "acquired_at": "2026-10-21T00:00:00Z",
    }]}), registry=registry, keenable_api_key="test-key")
    result = tools["search_web"].invoke({"query": "widget"})
    assert result["status"] == "ok"
    source = EvidenceRegistry(CaseStore(tmp_path / "cases.sqlite"), case["case_id"]).list_sources()[0]
    assert source["evidence_type"] == "search_snippet"
    assert source["published_at"] == "2026-10-20T00:00:00Z"
    assert source["acquired_at"] == "2026-10-21T00:00:00Z"
    assert source["url"] == url
    assert len(source["source_key"]) < 1000
    assert source["evidence_id"] == result["evidence"][0]["evidence_id"]


def test_commit_and_patch_limits_survive_reopening_saved_case(make_tools, tmp_path):
    from triage_desk.agent_evidence import EvidenceRegistry
    from triage_desk.cases import CaseStore

    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    case = store.create_case('Review an immutable PR patch')
    base, head = 'a' * 40, 'b' * 40

    def handler(request):
        if '/compare/' in request.url.path:
            return httpx.Response(200, json={'files': [{'filename': 'lib/widget.py', 'patch': '+changed line\n' * 700}]})
        return httpx.Response(200, json={'number': 88, 'title': 'Update', 'body': '',
                                       'base': {'sha': base}, 'head': {'sha': head}})

    tools, *_ = make_tools(handler, registry=EvidenceRegistry(store, case['case_id']))
    result = tools['read_github_pr'].invoke({'pr_number': 88})
    assert result['status'] == 'ok'
    saved = CaseStore(path).evidence(case['case_id'])[0]
    assert saved['commit_sha'] == head and saved['base_sha'] == base
    assert saved['filename'] == 'lib/widget.py'
    assert saved['truncated'] is True
    assert len(saved['text']) == 6000
