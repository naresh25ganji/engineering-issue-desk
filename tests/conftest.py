"""Small offline corpus; no network or provider credentials in unit tests."""

import json

import numpy as np
import pytest


@pytest.fixture
def source_records():
    common = {"source_commit": "fixture-commit", "snapshot_at": "2026-10-17T00:00:00Z",
              "version_scope": "Fixture documentation snapshot; not a version compatibility guarantee."}
    return [
        {**common, "source_id": "doc-cache", "title": "Cache shared resources", "kind": "documentation",
         "url": "https://docs.example.test/caching", "text":
         "# Resource caching\n\nUse cache_resource to reuse a shared database connection. "
         "Cached resources are shared across users and sessions. Objects must be thread safe. "
         "Use cache_data for serializable data when each caller needs a separate copy."},
        {**common, "source_id": "doc-session", "title": "Session State and widgets", "kind": "documentation",
         "url": "https://docs.example.test/session-state", "text":
         "# Session State\n\nSession State persists values between reruns for each user session. "
         "A widget callback runs before the rest of the script reruns. "
         "A widget key provides a way to access its value through Session State."},
        {**common, "source_id": "issue-42", "title": "Widget value resets after click", "kind": "issue",
         "url": "https://github.com/example/library/issues/42", "text":
         "# User report\n\nThe reporter says a widget resets after a click and rerun on version 1.2.3. "
         "This is a report, not a verified diagnosis. The issue remains open. "
         "The report does not establish that a fix works in other versions.", "reported_versions": ["1.2.3"]},
        {**common, "source_id": "release-123", "title": "Release 1.2.3 notes", "kind": "release",
         "url": "https://docs.example.test/release/1.2.3", "text":
         "# Release notes\n\nVersion 1.2.3 was published with documentation updates. "
         "This release note does not describe a widget reset fix."},
    ]


@pytest.fixture
def corpus_dir(tmp_path, source_records):
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "sources.json").write_text(json.dumps(source_records), encoding="utf-8")
    return directory


@pytest.fixture
def fake_dense(monkeypatch):
    """Deterministic vectors exercise production cache/search paths offline."""
    from triage_desk.embeddings import DenseIndex

    calls = {"documents": 0, "queries": 0}

    def vector(text):
        lower = text.lower()
        cache = sum(lower.count(word) for word in ("cache", "database", "memoize", "handle", "resource"))
        state = sum(lower.count(word) for word in ("session", "widget", "rerun", "callback", "click"))
        release = sum(lower.count(word) for word in ("release", "published", "version"))
        return np.asarray([cache, state, release, 1.0], dtype=np.float32)

    class FakeEmbedding:
        def embed(self, texts, batch_size=32):
            calls["documents"] += 1
            return iter(vector(text) for text in texts)

        def query_embed(self, question):
            calls["queries"] += 1
            return iter([vector(question)])

    monkeypatch.setattr(DenseIndex, "_make_model", lambda self: FakeEmbedding())
    return calls
