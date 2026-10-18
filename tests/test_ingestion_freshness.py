"""Fetch/cache timestamp checks with isolated bytes, clocks and mocked networking."""
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from types import SimpleNamespace

import pytest

from scripts import fetch_corpus as ingestion


URL = "https://raw.githubusercontent.com/streamlit/docs/pinned-commit/example.md"
ORIGINAL_TIME = "2026-10-01T12:34:56Z"


@pytest.fixture
def isolated_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "CACHE", tmp_path / "cache")
    # No actual process credentials or other environment settings are read.
    monkeypatch.setattr(ingestion, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(ingestion, "utc_now", lambda: ORIGINAL_TIME)
    calls = []
    def network(request, timeout):
        calls.append(request.full_url)
        return BytesIO(b"# Pinned source\nOriginal immutable bytes.")
    monkeypatch.setattr(ingestion, "urlopen", network)
    return calls


def _cache_path():
    return ingestion.CACHE / ingestion.sha256(URL.encode())


def _no_call(*args, **kwargs):
    raise AssertionError("A cache hit must not fetch upstream or read the current clock")


def test_immutable_cache_preserves_upstream_fetch_time_and_hash(isolated_fetch, monkeypatch):
    raw, first_time = ingestion.fetch(URL, immutable=True)
    cache = _cache_path()
    metadata = json.loads(ingestion._cache_metadata_path(cache).read_text())
    assert first_time == metadata["fetched_at"] == ORIGINAL_TIME
    assert metadata["sha256"] == ingestion.sha256(raw)
    assert metadata["timestamp_basis"] == "upstream_response"
    assert len(isolated_fetch) == 1
    # Even touching the cached file must not override its known fetch time.
    os.utime(cache, (1800000000, 1800000000))
    monkeypatch.setattr(ingestion, "utc_now", _no_call)
    monkeypatch.setattr(ingestion, "urlopen", _no_call)
    assert ingestion.fetch(URL, immutable=True) == (raw, ORIGINAL_TIME)


def test_legacy_cache_mtime_is_a_persisted_documented_approximation(isolated_fetch, monkeypatch):
    cache = _cache_path()
    cache.parent.mkdir(parents=True)
    raw = b"Legacy immutable source"
    cache.write_bytes(raw)
    old = datetime(2026, 9, 20, 10, 30, 0, tzinfo=timezone.utc)
    os.utime(cache, (old.timestamp(), old.timestamp()))
    monkeypatch.setattr(ingestion, "utc_now", _no_call)
    monkeypatch.setattr(ingestion, "urlopen", _no_call)
    expected = "2026-09-20T10:30:00Z"
    assert ingestion.fetch(URL, immutable=True) == (raw, expected)
    metadata = json.loads(ingestion._cache_metadata_path(cache).read_text())
    assert metadata["timestamp_basis"] == "legacy_cache_file_mtime_approximation"
    assert metadata["fetched_at"] == expected
    os.utime(cache, (1800000000, 1800000000))
    assert ingestion.fetch(URL, immutable=True) == (raw, expected)
    assert isolated_fetch == []


def test_tampered_cached_bytes_fail_without_an_upstream_request(isolated_fetch, monkeypatch):
    ingestion.fetch(URL, immutable=True)
    _cache_path().write_bytes(b"Changed after the original fetch")
    monkeypatch.setattr(ingestion, "urlopen", _no_call)
    with pytest.raises(ValueError, match="cache metadata is invalid"):
        ingestion.fetch(URL, immutable=True)


@pytest.mark.parametrize("content", ["{broken-json", json.dumps({
    "fetched_at": "2026-10-01", "sha256": "unused", "url_sha256": "unused"})])
def test_invalid_sidecars_do_not_silently_reset_fetch_time(isolated_fetch, monkeypatch, content):
    ingestion.fetch(URL, immutable=True)
    ingestion._cache_metadata_path(_cache_path()).write_text(content)
    monkeypatch.setattr(ingestion, "urlopen", _no_call)
    monkeypatch.setattr(ingestion, "utc_now", _no_call)
    with pytest.raises(ValueError, match="cache metadata is invalid"):
        ingestion.fetch(URL, immutable=True)


def test_mutable_fetch_does_not_reuse_immutable_cache_age(isolated_fetch, monkeypatch):
    ingestion.fetch(URL, immutable=True)
    monkeypatch.setattr(ingestion, "utc_now", lambda: "2026-10-18T08:00:00Z")
    _, fetched_at = ingestion.fetch(URL, immutable=False)
    assert fetched_at == "2026-10-18T08:00:00Z"
    assert len(isolated_fetch) == 2
    assert json.loads(ingestion._cache_metadata_path(_cache_path()).read_text())["fetched_at"] == ORIGINAL_TIME
