#!/usr/bin/env python3
"""Refresh a bounded, attributed Streamlit corpus without executing source code.

Uses Python's standard library. Documentation/API text and contribution files are
pinned to commits. Public issue bodies and state are a dated snapshot, not proof
of a defect, a fix, or compatibility with another version. No comments, profiles,
attachments, private repositories, or academy solution code are downloaded.

Run from the project root: python scripts/fetch_corpus.py
Refetch every pinned and mutable source: python scripts/fetch_corpus.py --refresh-all
Validate the checked-in snapshot offline: python scripts/fetch_corpus.py --check
An optional GITHUB_TOKEN increases API rate limits; it is never persisted.

Immutable cache entries retain their original fetch timestamp in a hash-checked
metadata sidecar. Legacy entries without sidecars use the cache file's modification
time as an approximation, which may reflect a prior copy or touch operation.
Reading cached bytes never resets their snapshot age.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DOCS_COMMIT = "7a0a34742a9fe4a879fe3641f83b8136bc5448d7"
STREAMLIT_COMMIT = "4278206f51bde8026a20c7d91c4ef473fece6c69"
API_VERSION = "1.66.0"
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(tempfile.gettempdir()) / "engineering-issue-desk-corpus-fetch"
MAX_RESPONSE_BYTES = 60_000_000  # Pinned generated docstrings are about 47 MB.
MAX_ISSUE_CHARS = 30_000
MAX_RELEASE_CHARS = 38_000
WORKERS = 4
ISSUE_SNAPSHOTS: dict[int, tuple[dict, bytes, str, str]] = {}

# Explicit allowlist: no recursive crawl, no user-controlled URLs.
# IDs remain stable when a title or URL slug changes upstream.
DOCS = [
    ("doc-session-state", "develop/concepts/architecture/session-state.md"),
    ("doc-forms", "develop/concepts/architecture/forms.md"),
    ("doc-caching", "develop/concepts/architecture/caching.md"),
    ("doc-widget-behavior", "develop/concepts/architecture/widget-behavior.md"),
    ("doc-architecture", "develop/concepts/architecture/architecture.md"),
    ("doc-fragments", "develop/concepts/architecture/fragments.md"),
    ("doc-run-app", "develop/concepts/architecture/run-your-app.md"),
    ("doc-button-behavior", "develop/concepts/app-design/button-behavior-and-examples.md"),
    ("doc-multipage-widgets", "develop/concepts/multipage-apps/widgets.md"),
    ("doc-config-options", "develop/concepts/configuration/options.md"),
    ("doc-static-files", "develop/concepts/configuration/static-file-serving.md"),
    ("doc-multithreading", "develop/concepts/app-design/multithreading.md"),
    ("doc-security-reminders", "develop/concepts/connections/security-reminders.md"),
    ("doc-secrets-management", "develop/concepts/connections/secrets-management.md"),
    ("doc-apptest-start", "develop/concepts/app-testing/get-started.md"),
    ("doc-apptest-examples", "develop/concepts/app-testing/examples.md"),
    ("doc-install-command-line", "get-started/installation/command-line.md"),
    ("doc-install-requirements", "get-started/installation/_index.md"),
    ("doc-deployment-dependencies", "deploy/concepts/dependencies.md"),
    ("doc-community-app-dependencies", "deploy/community-cloud/deploy-your-app/app-dependencies.md"),
    ("doc-community-file-organization", "deploy/community-cloud/deploy-your-app/file-organization.md"),
    ("doc-community-secrets", "deploy/community-cloud/deploy-your-app/secrets-management.md"),
    ("doc-community-deploy", "deploy/community-cloud/deploy-your-app/deploy.md"),
    ("doc-docker", "deploy/tutorials/docker.md"),
    ("doc-community-limitations", "deploy/community-cloud/status-and-limitations.md"),
    ("doc-upload-storage", "kb/FAQ/where-file-uploader-store-when-deleted.md"),
    ("doc-upload-filename", "kb/FAQ/retrieve-filename-uploaded.md"),
    ("doc-upload-limit", "kb/deployments/increase-upload-limit-cloud.md"),
    ("doc-widget-state-faq", "kb/FAQ/widget-updating-session-state.md"),
    ("doc-upgrade-streamlit", "kb/FAQ/upgrade-version-streamlit.md"),
    ("doc-serializable-session-state", "kb/FAQ/serializable-session-state.md"),
    ("doc-resource-limits", "kb/deployments/resource-limits.md"),
    ("doc-api-session-state", "develop/api-reference/caching-and-state/session_state.md"),
]

API_DOCS = [
    ("doc-api-file-uploader", "streamlit.file_uploader", "/develop/api-reference/widgets/st.file_uploader"),
    ("doc-api-cache-data", "streamlit.cache_data", "/develop/api-reference/caching-and-state/st.cache_data"),
    ("doc-api-cache-resource", "streamlit.cache_resource", "/develop/api-reference/caching-and-state/st.cache_resource"),
    ("doc-api-rerun", "streamlit.rerun", "/develop/api-reference/execution-flow/st.rerun"),
    ("doc-api-form", "streamlit.form", "/develop/api-reference/execution-flow/st.form"),
]

# Curated real reports across uploads, state/reruns, widgets, caches, deployment.
# Stable allowlist makes scope reproducible; state/body are refreshed and dated.
ISSUE_IDS = [
    8534, 13627, 12911, 12607, 12146, 11396,
    11115, 7216, 16388, 17119,
    17215, 17175, 17210, 17011,
    7079, 4164, 4664, 5336,
    11528, 10615, 6920,
    17031, 16341, 12800,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".metadata.json")


def _write_cache_metadata(cache_path: Path, metadata: dict) -> None:
    destination = _cache_metadata_path(cache_path)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent,
                                         prefix=cache_path.name + ".", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(metadata, handle, sort_keys=True)
            handle.flush()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_immutable(cache_path: Path, url: str) -> tuple[bytes, str]:
    if cache_path.stat().st_size > MAX_RESPONSE_BYTES:
        raise ValueError("Immutable cache entry exceeds the response-size limit")
    raw = cache_path.read_bytes()
    metadata_path = _cache_metadata_path(cache_path)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fetched_at = metadata["fetched_at"]
            parsed = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None or metadata["sha256"] != sha256(raw) or metadata["url_sha256"] != sha256(url.encode()):
                raise ValueError("Cache provenance does not match its bytes or URL")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError("Immutable cache metadata is invalid; remove the affected cache entry and its sidecar before refetching") from exc
        return raw, fetched_at
    # Preserve this approximation once rather than replacing it with access time.
    # A sidecar also prevents later touches/copies from continually renewing age.
    fetched_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    _write_cache_metadata(cache_path, {
        "schema_version": 1, "fetched_at": fetched_at, "sha256": sha256(raw),
        "url_sha256": sha256(url.encode()), "timestamp_basis": "legacy_cache_file_mtime_approximation",
    })
    return raw, fetched_at


def fetch(url: str, *, immutable: bool = False, force_refetch: bool = False) -> tuple[bytes, str]:
    """Keep cache age by default; explicit revalidation requires a successful GET."""
    allowed = ("https://raw.githubusercontent.com/streamlit/", "https://api.github.com/repos/streamlit/streamlit/")
    if not url.startswith(allowed):
        raise ValueError("URL is outside the official Streamlit source allowlist")
    cache_path = CACHE / sha256(url.encode())
    if immutable and cache_path.exists() and not force_refetch:
        return _cached_immutable(cache_path, url)
    headers = {"User-Agent": "engineering-issue-desk-course-corpus/1.0", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Source exceeds the response-size limit")
            at = utc_now()
            if immutable:
                CACHE.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(raw)
                _write_cache_metadata(cache_path, {
                    "schema_version": 1, "fetched_at": at, "sha256": sha256(raw),
                    "url_sha256": sha256(url.encode()), "timestamp_basis": "upstream_response",
                })
            return raw, at
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code} fetching {url}; corpus was not replaced") from None
        except (URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(f"Network failure fetching {url}; corpus was not replaced") from None
        time.sleep(attempt + 1)
    raise RuntimeError("Unreachable fetch state")


def strip_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", markdown, re.DOTALL)
    if not match:
        return {}, markdown
    values = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip()] = value.strip().strip('"\'')
    return values, markdown[match.end():]


def sanitize(text: str) -> str:
    """Remove obvious contact/credential strings, preserving technical versions."""
    text = re.sub(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted email]", text)
    text = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b", "[redacted credential]", text)
    text = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~-]{20,}", r"\1[redacted credential]", text)
    text = re.sub(r"(?i)([?&](?:access_token|api_key|token)=)[^\s&#)\"]+", r"\1[redacted credential]", text)
    return text


def clean_markdown(text: str) -> str:
    """Retain headings, prose, tables and fenced code; remove MDX display chrome.

    This is a text conversion only: no MDX, HTML, JavaScript, or Python is run.
    Angle-bracket comparisons inside fenced examples are preserved.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    output, fence, component = [], False, []
    for line in text.splitlines():
        if re.match(r"\s*(```|~~~)", line):
            fence = not fence
            output.append(line)
            continue
        if fence:
            output.append(line)
            continue
        # Multiline self-closing MDX elements contain attributes, not prose.
        if component:
            component.append(line)
            if "/>" in line:
                component = []
            continue
        if re.match(r"\s*<(?:Cloud|Image|YouTube|Autofunction|GitHubIssuesHint|Tile|TileContainer)\b", line) and ">" not in line:
            component = [line]
            continue
        line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", line)
        # Keep the content of semantic callout blocks, lose presentation tags.
        line = re.sub(r"</?[A-Za-z][^>]*>", "", line)
        if re.match(r"\s*(?:import\s+.*\s+from\s+|export\s+default\b)", line):
            continue
        output.append(line.rstrip())
    result = "\n".join(output)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return sanitize(result)


def remove_prose_mentions(text: str) -> str:
    """Omit social handles while preserving Python decorators in repro code."""
    lines, fenced = [], False
    for line in text.splitlines():
        if re.match(r"\s*(```|~~~)", line):
            fenced = not fenced
        elif not fenced:
            line = re.sub(r"(?<![\w`])@[A-Za-z0-9_-]+\b", "[mentioned user]", line)
        lines.append(line)
    return "\n".join(lines)


class TextHTML(HTMLParser):
    """Small non-executing converter for pinned generated API docstring HTML."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        if self.skip:
            return
        if tag in ("p", "div", "ul", "ol", "table", "tr", "blockquote", "br"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "pre":
            self.parts.append("\n```python\n")
        elif tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if self.skip:
            return
        if tag == "pre":
            self.parts.append("\n```\n")
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def html_text(value: str) -> str:
    parser = TextHTML()
    parser.feed(value or "")
    return clean_markdown(unescape("".join(parser.parts)))


def base_source(source_id: str, title: str, url: str, kind: str, text: str,
                raw: bytes, at: str, commit: str | None, scope: str, **extra: object) -> dict:
    if not text.strip():
        raise ValueError(f"Empty source: {source_id}")
    return dict(source_id=source_id, title=title, url=url, kind=kind, text=text,
                source_commit=commit, snapshot_at=at, version_scope=scope,
                raw_sha256=sha256(raw), **extra)


def get_document(spec: tuple[str, str], *, force_refetch: bool = False) -> dict:
    source_id, path = spec
    raw_url = f"https://raw.githubusercontent.com/streamlit/docs/{DOCS_COMMIT}/content/{path}"
    raw, at = fetch(raw_url, immutable=True, force_refetch=force_refetch)
    meta, body = strip_frontmatter(raw.decode("utf-8"))
    title = meta.get("title") or next((x.lstrip("# ") for x in body.splitlines() if x.startswith("# ")), path)
    slug = meta.get("slug")
    if not slug or not slug.startswith("/"):
        raise ValueError(f"Missing canonical documentation slug for {path}")
    text = clean_markdown(body)
    if not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    if len(text) < 180:
        raise ValueError(f"Documentation has too little substantive text: {source_id}")
    return base_source(source_id, title, "https://docs.streamlit.io" + slug,
                       "documentation", text, raw, at, DOCS_COMMIT,
                       "Official documentation snapshot at the pinned docs commit; check the user's installed version before applying guidance.",
                       source_path="content/" + path, raw_url=raw_url)


def api_text(item: dict) -> str:
    chunks = [f"# {item['name']}", item.get("signature", ""), html_text(item.get("description", ""))]
    for field, heading in (("args", "Parameters"), ("returns", "Returns")):
        values = item.get(field, [])
        if values:
            chunks.append(f"## {heading}")
        for value in values:
            chunks.append(f"### {value.get('name', heading)} ({value.get('type_name', '')})\n\n" + html_text(value.get("description", "")))
    examples = item.get("examples", item.get("example", ""))
    if examples:
        chunks += ["## Examples", html_text(examples)]
    return "\n\n".join(x for x in chunks if x)


def get_api_documents(*, force_refetch: bool = False) -> list[dict]:
    raw_url = f"https://raw.githubusercontent.com/streamlit/docs/{DOCS_COMMIT}/python/streamlit.json"
    raw, at = fetch(raw_url, immutable=True, force_refetch=force_refetch)
    by_version = json.loads(raw)[API_VERSION]
    results = []
    for source_id, symbol, slug in API_DOCS:
        item = by_version[symbol]
        text = api_text(item)
        if symbol == "streamlit.file_uploader":
            text += "\n\n" + api_text(by_version["UploadedFile"])
        results.append(base_source(source_id, symbol.replace("streamlit.", "st."),
                                   f"https://docs.streamlit.io/{API_VERSION}" + slug,
                                   "documentation", text, raw, at, DOCS_COMMIT,
                                   f"Official generated API reference for Streamlit {API_VERSION}, from the pinned docs commit; not a promise for other versions.",
                                   source_path=f"python/streamlit.json:{API_VERSION}:{symbol}",
                                   api_version=API_VERSION, upstream_api_source=item.get("source"), raw_url=raw_url))
    return results


def get_issue(number: int) -> dict:
    url = f"https://api.github.com/repos/streamlit/streamlit/issues/{number}"
    retrieval_method = "GitHub issue API"
    if number in ISSUE_SNAPSHOTS:
        issue, raw, at, snapshot_name = ISSUE_SNAPSHOTS[number]
        retrieval_method = "Previously fetched GitHub search API result: " + snapshot_name
    else:
        raw, at = fetch(url)
        issue = json.loads(raw)
    if "pull_request" in issue:
        raise ValueError(f"Allowlist entry {number} is a pull request, not an issue")
    body = issue.get("body") or ""
    # Remove the repetitive template checklist while preserving diagnostic fields.
    body = re.sub(r"(?ims)^### Checklist\s*\n.*?(?=^### |\Z)", "", body)
    # Handles are unnecessary for technical retrieval; author/login are never serialized.
    body = remove_prose_mentions(body)
    text = clean_markdown(body)
    truncated = len(text) > MAX_ISSUE_CHARS
    if truncated:
        text = text[:MAX_ISSUE_CHARS].rsplit("\n", 1)[0] + "\n\n[Issue body truncated at the documented corpus limit.]"
    reported = sorted(set(re.findall(r"(?i)streamlit(?:\s+version)?\s*[:=v]*\s*(1\.\d+(?:\.\d+)?)", body)))
    text = f"# Issue #{number}: {sanitize(issue['title'])}\n\nPublic issue report. State at snapshot: {issue['state']}. This is a report, not official diagnosis or proof of a fix.\n\n{text}"
    return base_source(f"issue-{number}", sanitize(issue["title"]), issue["html_url"], "issue", text,
                       raw, at, None,
                       "Issue claims apply only to the reporter's described environment. State is a dated snapshot; closed does not establish a fix or affected-version range.",
                       issue_number=number, state=issue["state"],
                       labels=sorted(x["name"] for x in issue.get("labels", [])),
                       created_at=issue.get("created_at"), updated_at=issue.get("updated_at"),
                       closed_at=issue.get("closed_at"), reported_versions=reported,
                       body_truncated=truncated, comments_included=False,
                       retrieval_method=retrieval_method)


def get_contributing(path: str, source_id: str, title: str, *, force_refetch: bool = False) -> dict:
    raw_url = f"https://raw.githubusercontent.com/streamlit/streamlit/{STREAMLIT_COMMIT}/{path}"
    raw, at = fetch(raw_url, immutable=True, force_refetch=force_refetch)
    text = clean_markdown(raw.decode()) if path.endswith(".md") else f"# {title}\n\n```yaml\n{sanitize(raw.decode())}\n```"
    return base_source(source_id, title,
                       f"https://github.com/streamlit/streamlit/blob/{STREAMLIT_COMMIT}/{path}",
                       "contributing", text, raw, at, STREAMLIT_COMMIT,
                       "Contributor guidance/template snapshot at the pinned Streamlit repository commit.",
                       source_path=path, raw_url=raw_url)


def get_release(*, force_refetch: bool = False) -> dict:
    path = "content/develop/quick-references/release-notes/2026.md"
    raw_url = f"https://raw.githubusercontent.com/streamlit/docs/{DOCS_COMMIT}/{path}"
    raw, at = fetch(raw_url, immutable=True, force_refetch=force_refetch)
    meta, body = strip_frontmatter(raw.decode())
    text = clean_markdown(body)
    truncated = len(text) > MAX_RELEASE_CHARS
    if truncated:
        # Retain complete sections from the beginning rather than partial claims.
        boundaries = [m.start() for m in re.finditer(r"(?m)^## ", text) if m.start() < MAX_RELEASE_CHARS]
        cut = boundaries[-1] if len(boundaries) > 1 else MAX_RELEASE_CHARS
        text = text[:cut].rstrip() + "\n\n[Later release-note sections omitted at the documented corpus limit.]"
    slug = meta.get("slug", "/develop/quick-references/release-notes/2026")
    return base_source("release-notes-2026", "Streamlit 2026 release notes (bounded excerpt)",
                       "https://docs.streamlit.io" + slug, "release", text, raw, at, DOCS_COMMIT,
                       "Only versions and fixes explicitly present in this pinned, bounded release-note excerpt are covered. Absence is not evidence that a fix does not exist.",
                       source_path=path, raw_url=raw_url, body_truncated=truncated)


def get_licenses(*, force_refetch: bool = False) -> dict[str, bytes]:
    files = {}
    for repo, commit, names in (("docs", DOCS_COMMIT, ("LICENSE", "NOTICE")),
                               ("streamlit", STREAMLIT_COMMIT, ("LICENSE", "NOTICES"))):
        for name in names:
            url = f"https://raw.githubusercontent.com/streamlit/{repo}/{commit}/{name}"
            try:
                raw, _ = fetch(url, immutable=True, force_refetch=force_refetch)
            except RuntimeError as exc:
                # A repository can lack NOTICE; do not fabricate an upstream notice.
                if name in ("NOTICE", "NOTICES") and "HTTP 404" in str(exc):
                    continue
                raise
            files[f"{repo}-{name}.txt"] = raw
    files["ATTRIBUTION.md"] = (
        "# Corpus attribution and source distinctions\n\n"
        "Documentation and generated API docstrings: Streamlit documentation contributors, "
        f"https://github.com/streamlit/docs/tree/{DOCS_COMMIT}. "
        "Retained text is extracted from Markdown or generated docstring HTML; presentation "
        "components/frontmatter are removed, and API sections are reformatted. See docs-LICENSE.txt.\n\n"
        "Contributor guide and bug template: Streamlit contributors, "
        f"https://github.com/streamlit/streamlit/tree/{STREAMLIT_COMMIT}. "
        "See streamlit-LICENSE.txt and streamlit-NOTICES.txt. No upstream Python/JavaScript is executed.\n\n"
        "Public GitHub issues: body excerpts supplied by the respective issue contributors, linked "
        "individually in sources.json. Issue text is attributed through its canonical GitHub URL "
        "and remains the content of its contributors; the repository Apache license is not asserted "
        "to relicense every issue submission. Profile/author fields, comments and attachments are "
        "excluded. Template checklists, obvious emails, credentials and @mentions are removed. "
        "Reports and issue state are evidence of what was reported at snapshot time, not an "
        "official diagnosis, exhaustive issue search, or proof that closed means fixed.\n\n"
        "The corpus is a bounded educational snapshot, not an endorsement or an official Streamlit "
        "support service. Source records preserve URLs, commits where applicable, timestamps, "
        "version scope, and raw hashes. Source code snippets are retrieved as text only.\n"
    ).encode()
    return files


def check_snapshot(data_dir: Path) -> dict:
    sources_path = data_dir / "sources.json"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    sources = json.loads(sources_path.read_text())
    if sha256(sources_path.read_bytes()) != manifest["sources_sha256"]:
        raise ValueError("sources.json file hash differs from manifest")
    seen = set()
    hashes = {item["source_id"]: item["text_sha256"] for item in manifest["sources"]}
    for source in sources:
        required = ("source_id", "title", "url", "kind", "text", "snapshot_at", "version_scope")
        if any(not isinstance(source.get(key), str) or not source[key].strip() for key in required):
            raise ValueError("A source is missing a required string")
        if source["source_id"] in seen:
            raise ValueError("Duplicate source ID")
        seen.add(source["source_id"])
        if source["kind"] not in {"documentation", "issue", "release", "contributing"}:
            raise ValueError("Unknown source kind")
        if hashes.get(source["source_id"]) != sha256(source["text"].encode()):
            raise ValueError("Source text hash mismatch")
        if source["kind"] == "issue" and any(x in source for x in ("user", "author", "login", "assignee")):
            raise ValueError("Issue profile fields must not be serialized")
    counts = dict(Counter(source["kind"] for source in sources))
    if counts != manifest["counts"] or len(sources) != manifest["source_count"]:
        raise ValueError("Source counts differ from manifest")
    for entry in manifest["licenses"]:
        if sha256((data_dir / "licenses" / entry["file"]).read_bytes()) != entry["sha256"]:
            raise ValueError("License/attribution file hash mismatch")
    return {"source_count": len(sources), "counts": counts, "sources_sha256": manifest["sources_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate existing counts and hashes without network calls")
    mode.add_argument("--refresh-all", action="store_true",
                      help="Refetch every source from upstream, including commit-pinned cached documents; preserve the selected commits and API version")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--issue-snapshot-dir", type=Path,
                        help="Use previously fetched GitHub API search JSON files for the fixed issue allowlist (useful under API rate limits)")
    args = parser.parse_args()
    if args.refresh_all and args.issue_snapshot_dir:
        parser.error("--refresh-all cannot use historical --issue-snapshot-dir files")
    if args.check:
        print(json.dumps(check_snapshot(args.data_dir), indent=2))
        return 0
    if args.issue_snapshot_dir:
        for snapshot in sorted(args.issue_snapshot_dir.glob("issues-*.json")):
            raw = snapshot.read_bytes()
            data = json.loads(raw)
            at = datetime.fromtimestamp(snapshot.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            for issue in data.get("items", []):
                if issue.get("number") in ISSUE_IDS and issue.get("html_url", "").startswith("https://github.com/streamlit/streamlit/issues/"):
                    ISSUE_SNAPSHOTS[issue["number"]] = (issue, raw, at, snapshot.name)
        missing = set(ISSUE_IDS) - set(ISSUE_SNAPSHOTS)
        if missing:
            raise ValueError("Issue snapshots do not contain the full allowlist: " + ", ".join(str(x) for x in sorted(missing)))
    started = utc_now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        docs = list(pool.map(partial(get_document, force_refetch=args.refresh_all), DOCS))
    api = get_api_documents(force_refetch=args.refresh_all)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        issues = list(pool.map(get_issue, ISSUE_IDS))
    contribution = [get_contributing("CONTRIBUTING.md", "contributing-guide", "Streamlit contributing guide", force_refetch=args.refresh_all),
                    get_contributing(".github/ISSUE_TEMPLATE/bug_report.yml", "contributing-bug-report", "Streamlit bug report template", force_refetch=args.refresh_all)]
    sources = sorted(docs + api + issues + contribution + [get_release(force_refetch=args.refresh_all)], key=lambda x: x["source_id"])
    licenses = get_licenses(force_refetch=args.refresh_all)
    sources_bytes = (json.dumps(sources, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = {
        "schema_version": 1, "corpus": "Official Streamlit documentation and selected public issue reports",
        "started_at": started, "completed_at": utc_now(),
        "docs_repository": "https://github.com/streamlit/docs", "docs_commit": DOCS_COMMIT,
        "streamlit_repository": "https://github.com/streamlit/streamlit", "streamlit_commit": STREAMLIT_COMMIT,
        "api_reference_version": API_VERSION, "source_count": len(sources),
        "fetch_policy": "all_sources_refetched" if args.refresh_all else "immutable_cache_allowed",
        "counts": dict(Counter(x["kind"] for x in sources)),
        "issue_allowlist": ISSUE_IDS, "issue_states": dict(Counter(x["state"] for x in issues)),
        "selection": "Explicit curated allowlists in scripts/fetch_corpus.py; no live issue discovery in the app.",
        "limits": {"documentation_pages": len(DOCS) + len(API_DOCS), "issue_reports": len(ISSUE_IDS),
                   "issue_body_characters": MAX_ISSUE_CHARS, "release_excerpt_characters": MAX_RELEASE_CHARS,
                   "max_response_bytes": MAX_RESPONSE_BYTES, "workers": WORKERS,
                   "issue_comments": False, "attachments": False, "private_sources": False},
        "cleaning": ["Preserve headings, tables and fenced code as text; remove MDX display wrappers.",
                     "Convert generated API docstring HTML to text without executing it.",
                     "Remove issue template checklist, author/profile metadata and @mentions.",
                     "Redact obvious email addresses, GitHub/OpenAI/AWS credentials and URL tokens."],
        "limitations": ["Issue bodies are unverified reports; no issue comments or fix commits are included.",
                        "Closed issue state does not establish a fix or affected-version range.",
                        "Documentation is commit-pinned; generated API references specifically cover the recorded version.",
                        "Corpus is a selected snapshot, not exhaustive coverage or a live bug database.",
                        "Redaction is conservative pattern matching and is not a guarantee of PII detection."],
        "sources_sha256": sha256(sources_bytes),
        "sources": [{"source_id": x["source_id"], "kind": x["kind"], "url": x["url"],
                     "characters": len(x["text"]), "text_sha256": sha256(x["text"].encode()),
                     "raw_sha256": x["raw_sha256"]} for x in sources],
        "licenses": [{"file": name, "sha256": sha256(raw)} for name, raw in sorted(licenses.items())],
    }
    # Prepare all network results before replacing deliverables. Failures leave
    # the previous snapshot intact; these are the only project files written.
    args.data_dir.mkdir(parents=True, exist_ok=True)
    license_dir = args.data_dir / "licenses"
    license_dir.mkdir(exist_ok=True)
    for name, raw in licenses.items():
        (license_dir / name).write_bytes(raw)
    (args.data_dir / "sources.json").write_bytes(sources_bytes)
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(check_snapshot(args.data_dir), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Corpus refresh failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
