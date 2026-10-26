"""Fetch/cache timestamp checks with isolated bytes, clocks and mocked networking."""
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from types import SimpleNamespace
from urllib.error import URLError

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


def test_force_refetch_revalidates_identical_bytes_upstream_before_advancing_timestamp(isolated_fetch, monkeypatch):
    original, _ = ingestion.fetch(URL, immutable=True)
    refreshed_at = '2026-10-25T18:20:00Z'
    monkeypatch.setattr(ingestion, 'utc_now', lambda: refreshed_at)
    assert ingestion.fetch(URL, immutable=True, force_refetch=True) == (original, refreshed_at)
    assert isolated_fetch == [URL, URL], 'Identical immutable bytes still need a real upstream response'
    metadata = json.loads(ingestion._cache_metadata_path(_cache_path()).read_text())
    assert metadata['fetched_at'] == refreshed_at
    assert metadata['sha256'] == ingestion.sha256(original)
    assert metadata['url_sha256'] == ingestion.sha256(URL.encode())
    assert metadata['timestamp_basis'] == 'upstream_response'
    monkeypatch.setattr(ingestion, 'urlopen', _no_call)
    monkeypatch.setattr(ingestion, 'utc_now', _no_call)
    assert ingestion.fetch(URL, immutable=True) == (original, refreshed_at)


def test_force_refetch_repairs_corrupt_sidecar_from_a_successful_upstream_response(isolated_fetch, monkeypatch):
    original, _ = ingestion.fetch(URL, immutable=True)
    metadata_path = ingestion._cache_metadata_path(_cache_path())
    metadata_path.write_text('{invalid metadata')
    refreshed_at = '2026-10-25T18:21:00Z'
    monkeypatch.setattr(ingestion, 'utc_now', lambda: refreshed_at)
    assert ingestion.fetch(URL, immutable=True, force_refetch=True) == (original, refreshed_at)
    assert isolated_fetch == [URL, URL]
    repaired = json.loads(metadata_path.read_text())
    assert repaired['sha256'] == ingestion.sha256(original)
    assert repaired['fetched_at'] == refreshed_at and repaired['timestamp_basis'] == 'upstream_response'


@pytest.mark.parametrize('corrupt_sidecar', [False, True])
def test_failed_forced_network_request_preserves_existing_cache_and_metadata(isolated_fetch, monkeypatch, corrupt_sidecar):
    original, _ = ingestion.fetch(URL, immutable=True)
    metadata_path = ingestion._cache_metadata_path(_cache_path())
    if corrupt_sidecar:
        metadata_path.write_text('{invalid metadata')
    prior_metadata = metadata_path.read_bytes()
    attempts = []
    def unavailable(request, timeout):
        attempts.append(request.full_url)
        raise URLError('Offline test transport is unavailable')
    monkeypatch.setattr(ingestion, 'urlopen', unavailable)
    monkeypatch.setattr(ingestion, 'utc_now', _no_call)
    monkeypatch.setattr(ingestion.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='Network failure'):
        ingestion.fetch(URL, immutable=True, force_refetch=True)
    assert attempts and set(attempts) == {URL}
    assert _cache_path().read_bytes() == original
    assert metadata_path.read_bytes() == prior_metadata
    monkeypatch.setattr(ingestion, 'urlopen', _no_call)
    if corrupt_sidecar:
        with pytest.raises(ValueError, match='cache metadata is invalid'):
            ingestion.fetch(URL, immutable=True)
    else:
        assert ingestion.fetch(URL, immutable=True) == (original, ORIGINAL_TIME)


def test_oversized_forced_response_does_not_replace_previous_cache(isolated_fetch, monkeypatch):
    original, _ = ingestion.fetch(URL, immutable=True)
    metadata_path = ingestion._cache_metadata_path(_cache_path())
    prior_metadata = metadata_path.read_bytes()
    monkeypatch.setattr(ingestion, 'MAX_RESPONSE_BYTES', 10)
    monkeypatch.setattr(ingestion, 'urlopen', lambda request, timeout: BytesIO(b'x' * 11))
    monkeypatch.setattr(ingestion, 'utc_now', _no_call)
    with pytest.raises(ValueError, match='response-size limit'):
        ingestion.fetch(URL, immutable=True, force_refetch=True)
    assert _cache_path().read_bytes() == original
    assert metadata_path.read_bytes() == prior_metadata


@pytest.mark.parametrize('forced', [False, True])
def test_all_pinned_source_helpers_forward_explicit_refetch_policy(monkeypatch, forced):
    calls = []
    api_items = {symbol: {'name': symbol, 'signature': f'{symbol}()', 'description': '<p>A public API description.</p>'}
                 for _, symbol, _ in ingestion.API_DOCS}
    api_items['UploadedFile'] = {'name': 'UploadedFile', 'description': '<p>A public uploaded-file description.</p>'}
    markdown = ('---\ntitle: Example source\nslug: /example-source\n---\n# Example source\n\n'
                + 'Substantive pinned documentation with literal code and provenance. ' * 5).encode()
    def fetch(url, *, immutable=False, force_refetch=False):
        calls.append((url, immutable, force_refetch))
        if url.endswith('/python/streamlit.json'):
            raw = json.dumps({ingestion.API_VERSION: api_items}).encode()
        elif url.endswith(('/LICENSE', '/NOTICE', '/NOTICES')):
            raw = b'Public license and attribution bytes.'
        elif url.endswith('.yml'):
            raw = b'name: Public bug report template\ndescription: A bounded test template\n'
        else:
            raw = markdown
        return raw, ORIGINAL_TIME
    monkeypatch.setattr(ingestion, 'fetch', fetch)
    ingestion.get_document(ingestion.DOCS[0], force_refetch=forced)
    assert len(ingestion.get_api_documents(force_refetch=forced)) == len(ingestion.API_DOCS)
    ingestion.get_contributing('CONTRIBUTING.md', 'contributing-guide', 'Contributor guide', force_refetch=forced)
    ingestion.get_contributing('.github/ISSUE_TEMPLATE/bug_report.yml', 'contributing-template', 'Bug template', force_refetch=forced)
    ingestion.get_release(force_refetch=forced)
    licenses = ingestion.get_licenses(force_refetch=forced)
    assert 'ATTRIBUTION.md' in licenses
    assert len(calls) == 9  # One shared API JSON plus docs, contribution, release and four license URLs.
    assert all(immutable is True and force is forced for _, immutable, force in calls)
    assert all(ingestion.DOCS_COMMIT in url or ingestion.STREAMLIT_COMMIT in url for url, _, _ in calls)


def _mock_cli_sources(monkeypatch):
    calls = []
    def source(source_id, kind='documentation'):
        return {'source_id': source_id, 'title': source_id, 'url': 'https://docs.streamlit.io/example',
                'kind': kind, 'text': 'A mocked public snapshot source with sufficient provenance.',
                'raw_sha256': 'a' * 64, 'snapshot_at': ORIGINAL_TIME, 'state': 'open'}
    def document(spec, *, force_refetch=False):
        calls.append(('document', force_refetch))
        return source(spec[0])
    def api(*, force_refetch=False):
        calls.append(('api', force_refetch))
        return [source(item[0]) for item in ingestion.API_DOCS]
    def issue(number):
        calls.append(('issue', None))
        return source(f'issue-{number}', 'issue')
    def contributing(path, source_id, title, *, force_refetch=False):
        calls.append(('contributing', force_refetch))
        return source(source_id, 'contributing')
    def release(*, force_refetch=False):
        calls.append(('release', force_refetch))
        return source('release-notes-2026', 'release')
    def licenses(*, force_refetch=False):
        calls.append(('licenses', force_refetch))
        return {'docs-LICENSE.txt': b'Mocked license bytes.'}
    for name, function in (('get_document', document), ('get_api_documents', api), ('get_issue', issue),
                           ('get_contributing', contributing), ('get_release', release), ('get_licenses', licenses)):
        monkeypatch.setattr(ingestion, name, function)
    monkeypatch.setattr(ingestion, 'utc_now', lambda: ORIGINAL_TIME)
    monkeypatch.setattr(ingestion, 'check_snapshot', lambda path: {'source_count': 65})
    monkeypatch.setattr(ingestion, 'urlopen', _no_call)
    return calls


@pytest.mark.parametrize('forced', [False, True])
def test_cli_refresh_policy_reaches_every_pinned_source_family(monkeypatch, tmp_path, forced):
    calls = _mock_cli_sources(monkeypatch)
    arguments = ['fetch_corpus.py', '--data-dir', str(tmp_path / 'data')]
    if forced:
        arguments.append('--refresh-all')
    monkeypatch.setattr(ingestion.sys, 'argv', arguments)
    assert ingestion.main() == 0
    pinned = [(family, policy) for family, policy in calls if family != 'issue']
    assert len(pinned) == len(ingestion.DOCS) + 5  # API, two contribution files, release and licenses.
    assert {family for family, _ in pinned} == {'document', 'api', 'contributing', 'release', 'licenses'}
    assert all(policy is forced for _, policy in pinned)
    assert sum(family == 'issue' for family, _ in calls) == len(ingestion.ISSUE_IDS)


def test_cli_check_remains_offline_without_relabeling_snapshot_times(monkeypatch, tmp_path):
    checked = []
    monkeypatch.setattr(ingestion.sys, 'argv', ['fetch_corpus.py', '--check', '--data-dir', str(tmp_path)])
    monkeypatch.setattr(ingestion, 'check_snapshot', lambda path: checked.append(path) or {'source_count': 65})
    monkeypatch.setattr(ingestion, 'urlopen', _no_call)
    monkeypatch.setattr(ingestion, 'utc_now', _no_call)
    for name in ('get_document', 'get_api_documents', 'get_issue', 'get_contributing', 'get_release', 'get_licenses'):
        monkeypatch.setattr(ingestion, name, _no_call)
    assert ingestion.main() == 0 and checked == [tmp_path]


@pytest.mark.parametrize('extra_arguments', [['--check'], ['--issue-snapshot-dir', 'historical-snapshots']])
def test_refresh_all_rejects_offline_check_and_historical_issue_snapshots(monkeypatch, extra_arguments):
    monkeypatch.setattr(ingestion.sys, 'argv', ['fetch_corpus.py', '--refresh-all', *extra_arguments])
    monkeypatch.setattr(ingestion, 'urlopen', _no_call)
    monkeypatch.setattr(ingestion, 'utc_now', _no_call)
    for name in ('check_snapshot', 'get_document', 'get_api_documents', 'get_issue', 'get_contributing', 'get_release', 'get_licenses'):
        monkeypatch.setattr(ingestion, name, _no_call)
    with pytest.raises(SystemExit) as caught:
        ingestion.main()
    assert caught.value.code == 2
