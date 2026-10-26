"""Explicit, replay-protected publication of an approved public-evidence PR summary.

Preparing a preview is offline. Only ``publish`` can make requests, after exact
plan confirmation and a fresh public-PR revision check. An uncertain POST is
never retried automatically. This module exposes no inline-review or PR-approval
action, and never publishes a case question, code context, history or next steps.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
import time
from urllib.parse import unquote, urlsplit
from uuid import UUID, uuid4

import httpx

from .agent_evidence import EvidenceRegistry, _escaped, canonical_http_url
from .cases import CaseExecutionBusy


REPOSITORY = 'streamlit/streamlit'
API = 'https://api.github.com/repos/' + REPOSITORY
WEB = 'https://github.com/' + REPOSITORY
MAX_BODY_CHARS = 12_000
MAX_RESPONSE_BYTES = 1_000_000
_SHA = re.compile(r'[0-9a-f]{40}\Z')
_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SECRET = re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|'
                     r'github_pat_[A-Za-z0-9_]{16,}|fw_[A-Za-z0-9_-]{20,}|keen_[A-Za-z0-9_-]{16,})')
_PUBLIC_HOSTS = frozenset({'docs.streamlit.io', 'docs.python.org', 'docs.langchain.com', 'github.com'})


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _identifier(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError('Use a valid publication plan ID.') from None
    return value


def _public_source(source):
    if source.get('origin') not in {'snapshot', 'live'} or source.get('kind') not in {
        'documentation', 'issue', 'release', 'contributing', 'code',
    }:
        raise ValueError('Publication supports must contain public source evidence only.')
    url = canonical_http_url(source.get('url', ''))
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname not in _PUBLIC_HOSTS
            or parts.port not in (None, 443) or parts.query):
        raise ValueError('Publication supports need approved public HTTPS source links.')
    path = unquote(parts.path)
    if (parts.hostname == 'github.com' and
            (not (path == '/streamlit/streamlit' or path.startswith('/streamlit/streamlit/'))
             or '\\' in path or any(part in {'.', '..'} for part in path.split('/')))):
        raise ValueError('GitHub publication evidence is limited to streamlit/streamlit.')
    if hashlib.sha256(source['text'].encode()).hexdigest() != source.get('sha256'):
        raise ValueError('The registered public evidence failed its integrity check.')
    return url


def _revision_support(source, number, head, base, url):
    """Never label an old/different PR's code as the preview's reviewed commit."""
    path = unquote(urlsplit(url).path).strip('/').split('/')
    section = path[2] if len(path) > 2 and urlsplit(url).hostname == 'github.com' else ''
    key = source.get('source_key', '')
    # Saved records may predate source classification fixes. A legacy
    # "documentation" label must not turn a mutable/old repository page into
    # evidence for the newly selected PR revision.
    code_page = section in {'blob', 'raw', 'tree', 'commit', 'commits', 'compare'}
    if (code_page or source.get('kind') == 'code' or source.get('evidence_type') == 'code_patch'
            or key.startswith('github:patch:')):
        filename = source.get('filename', '')
        expected = f'{WEB}/compare/{base}...{head}#diff-' + hashlib.sha256(filename.encode()).hexdigest()
        if (source.get('evidence_type') != 'code_patch' or source.get('commit_sha') != head
                or source.get('base_sha') != base or not filename
                or key != f'github:patch:{base}:{head}:{filename}' or url != expected):
            raise ValueError('Published code support must match the selected PR base and head commits.')
    elif section in {'pull', 'pulls'} or key.startswith('github:pr:'):
        if (key != f'github:pr:{number}:{head}' or source.get('evidence_type') != 'pr_description'
                or source.get('commit_sha') != head or source.get('base_sha') != base
                or url != f'{WEB}/pull/{number}'):
            raise ValueError('Published PR context must match the selected target and saved revision.')
    elif section == 'issues' and len(path) > 3 and path[3] == str(number):
        raise ValueError('PR-specific issue comments need pinned review provenance before publication.')


class _RemoteFailure(Exception):
    def __init__(self, code, *, uncertain=False):
        self.code, self.uncertain = code, uncertain


class ReviewPublisher:
    """Persist immutable previews beside CaseStore, with one publish attempt each.

    ``session`` may be an injected ``httpx.Client`` (for example MockTransport).
    No HTTP client is constructed and no credentials are needed during prepare.
    """

    def __init__(self, store, token='', session=None, *, forbidden_values=()):
        self.store, self.token, self.session = store, token, session
        self._forbidden = tuple(value for value in (*forbidden_values, token)
                                if isinstance(value, str) and 8 <= len(value) <= 4096)
        with self._connection() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS review_publication_plans (
                    plan_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    fingerprint TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('prepared','publishing','published','uncertain','failed')),
                    updated_at TEXT NOT NULL,
                    error_code TEXT NOT NULL DEFAULT '',
                    review_url TEXT NOT NULL DEFAULT '',
                    github_review_id INTEGER
                );
                CREATE UNIQUE INDEX IF NOT EXISTS review_publication_active_fingerprint
                    ON review_publication_plans(fingerprint) WHERE status != 'failed';
                CREATE TRIGGER IF NOT EXISTS review_publication_immutable_plan
                    BEFORE UPDATE OF plan_id,case_id,fingerprint,plan_json ON review_publication_plans
                    BEGIN SELECT RAISE(ABORT, 'Publication preview is immutable'); END;
            ''')
        self._recover_interrupted()

    def _recover_interrupted(self):
        with self._connection() as connection:
            pending = connection.execute("SELECT plan_id,case_id FROM review_publication_plans WHERE status='publishing'").fetchall()
        for row in pending:
            try:
                # Active publishers retain this same OS lock. Only a lock-free,
                # abandoned dispatch becomes uncertain after process restart.
                with self.store.execution_lock(row['case_id']):
                    with self._connection() as connection:
                        connection.execute("UPDATE review_publication_plans SET status='uncertain',error_code='interrupted_publication',updated_at=? WHERE plan_id=? AND status='publishing'",
                                           (_now(), row['plan_id']))
            except CaseExecutionBusy:
                continue

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.store.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, plan_id):
        _identifier(plan_id)
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM review_publication_plans WHERE plan_id=?', (plan_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown publication plan.')
        return {**json.loads(row['plan_json']), **{field: row[field] for field in
                ('status', 'updated_at', 'error_code', 'review_url', 'github_review_id')}}

    def _case_snapshot(self, case_id):
        case = self.store.get_case(case_id)
        report = case.get('report') or {}
        quality = report.get('quality_review') or {}
        if case['status'] != 'approved' or quality.get('status') not in {'completed', 'no_content'}:
            raise ValueError('Approve an evidence-reviewed case before preparing publication.')
        if quality['status'] == 'no_content' and (report.get('findings') or report.get('next_steps')):
            raise ValueError('The case does not have a completed evidence review.')
        checked = EvidenceRegistry(self.store, case_id).validate_report({field: report.get(field) for field in
                    ('findings', 'next_steps', 'missing_information')})
        sources = self.store.evidence(case_id)
        return case, checked, sources, _digest(report), _digest(sources)

    def _summary(self, report, sources, number, head, base):
        by_id = {source['evidence_id']: source for source in sources}
        lines = ['## Engineering Issue Desk review', '',
                 'Human-approved local report, published as a summary comment. Public-source excerpts only.',
                 'This publication step did not execute code or tests.', '', f'Reviewed commit: `{head}`.', '']
        raw_content = []
        for finding in report['findings']:
            raw_content.append(finding['statement'])
            lines.extend([f'### {finding["severity"].title()}', '', _escaped(finding['statement']), ''])
            for support in finding['support']:
                source = by_id[support['evidence_id']]
                url = _public_source(source)
                _revision_support(source, number, head, base, url)
                raw_content.extend((source['title'], url, support['quote']))
                lines.extend([f'Source: [{_escaped(source["title"])}]({url})', ''])
                lines.extend('> ' + _escaped(line) for line in support['quote'].splitlines())
                lines.append('')
        if not report['findings']:
            lines.extend(['No supported findings were retained in the approved report.', ''])
        body = '\n'.join(lines).rstrip() + '\n'
        if len(body) > MAX_BODY_CHARS:
            raise ValueError('The public summary exceeds 12000 characters; narrow the approved report first.')
        if any(_CONTROL.search(value) or _SECRET.search(value) or any(secret in value for secret in self._forbidden)
               for value in (*raw_content, body)):
            raise ValueError('The public summary contains unsafe or credential-like material.')
        return body

    def prepare(self, case_id, pr_number):
        if isinstance(pr_number, bool) or not isinstance(pr_number, int) or not 1 <= pr_number <= 10_000_000:
            raise ValueError('Supply a positive Streamlit pull request number.')
        _, report, sources, report_hash, evidence_hash = self._case_snapshot(case_id)
        descriptions = [source for source in sources if source.get('source_key', '').startswith(f'github:pr:{pr_number}:')]
        if not descriptions:
            raise ValueError('Read this public PR description and its commit-pinned patches before publication.')
        description = descriptions[-1]
        head, base = description.get('commit_sha', ''), description.get('base_sha', '')
        if (not isinstance(head, str) or not _SHA.fullmatch(head) or not isinstance(base, str) or not _SHA.fullmatch(base)
                or description.get('source_key') != f'github:pr:{pr_number}:{head}'
                or description.get('origin') != 'live' or description.get('evidence_type') != 'pr_description'
                or _public_source(description) != f'{WEB}/pull/{pr_number}'):
            raise ValueError('The saved PR description has invalid public revision provenance.')
        patches = [source for source in sources if source.get('evidence_type') == 'code_patch'
                   and source.get('commit_sha') == head and source.get('base_sha') == base]
        valid_patch = False
        for patch in patches:
            filename = patch.get('filename', '')
            expected_url = f'{WEB}/compare/{base}...{head}#diff-' + hashlib.sha256(filename.encode()).hexdigest()
            if (patch.get('origin') == 'live' and patch.get('kind') == 'code' and filename
                    and patch.get('source_key') == f'github:patch:{base}:{head}:{filename}'
                    and _public_source(patch) == expected_url):
                valid_patch = True
        if not valid_patch:
            raise ValueError('A registered public code patch must match the PR base and head commits.')
        body = self._summary(report, sources, pr_number, head, base)
        plan = {'plan_id': str(uuid4()), 'case_id': case_id, 'repository': REPOSITORY,
                'pr_number': pr_number, 'head_sha': head, 'base_sha': base, 'commit_id': head,
                'event': 'COMMENT', 'body': body, 'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                'report_sha256': report_hash, 'evidence_sha256': evidence_hash, 'created_at': _now()}
        fingerprint = _digest({key: value for key, value in plan.items() if key not in {'plan_id', 'created_at'}})
        with self._connection() as connection:
            connection.execute('INSERT OR IGNORE INTO review_publication_plans(plan_id,case_id,fingerprint,plan_json,status,updated_at) VALUES(?,?,?,?,?,?)',
                               (plan['plan_id'], case_id, fingerprint, json.dumps(plan, ensure_ascii=False), 'prepared', _now()))
            row = connection.execute("SELECT plan_id FROM review_publication_plans WHERE fingerprint=? AND status != 'failed'", (fingerprint,)).fetchone()
        return self.get(row['plan_id'])

    def _finish(self, plan_id, status, code='', *, review_id=None):
        plan = self.get(plan_id)
        review_url = f'{WEB}/pull/{plan["pr_number"]}#pullrequestreview-{review_id}' if review_id is not None else ''
        with self._connection() as connection:
            connection.execute("UPDATE review_publication_plans SET status=?,error_code=?,github_review_id=?,review_url=?,updated_at=? WHERE plan_id=? AND status='publishing'",
                               (status, code, review_id, review_url, _now(), plan_id))
        return self.get(plan_id)

    def _request(self, client, method, path, *, payload=None):
        started = time.monotonic()
        headers = {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {self.token}',
                   'X-GitHub-Api-Version': '2026-03-10'}
        try:
            with client.stream(method, f'{API}/{path}', headers=headers,
                               timeout=httpx.Timeout(30, connect=10), follow_redirects=False,
                               **({'json': payload} if payload is not None else {})) as response:
                if not 200 <= response.status_code < 300:
                    uncertain = method == 'POST' and response.status_code not in {400, 401, 403, 404, 410, 422, 429}
                    raise _RemoteFailure('github_rejected' if not uncertain else 'post_outcome_unknown', uncertain=uncertain)
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES or time.monotonic() - started > 30:
                        raise _RemoteFailure('response_unconfirmed', uncertain=method == 'POST')
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError
                return value
        except _RemoteFailure:
            raise
        except Exception:
            raise _RemoteFailure('post_outcome_unknown' if method == 'POST' else 'preflight_unavailable', uncertain=method == 'POST') from None

    def publish(self, plan_id, confirmation=None):
        plan = self.get(plan_id)
        if not isinstance(confirmation, str) or confirmation != plan_id:
            raise ValueError('Confirm publication using the exact preview plan ID.')
        if (not isinstance(self.token, str) or not self.token.strip() or len(self.token) > 4096
                or any(ord(character) < 33 or ord(character) == 127 for character in self.token)):
            raise ValueError('Configure a GitHub token only when explicitly publishing the preview.')
        with self.store.execution_lock(plan['case_id']):
            with self._connection() as connection:
                changed = connection.execute("UPDATE review_publication_plans SET status='publishing',updated_at=? WHERE plan_id=? AND status='prepared'",
                                             (_now(), plan_id)).rowcount
            if not changed:
                raise ValueError('This publication plan was already attempted; do not replay it.')
            post_started = False
            try:
                def unchanged():
                    _, report, sources, report_hash, evidence_hash = self._case_snapshot(plan['case_id'])
                    if report_hash != plan['report_sha256'] or evidence_hash != plan['evidence_sha256']:
                        raise ValueError
                    if self._summary(report, sources, plan['pr_number'], plan['head_sha'], plan['base_sha']) != plan['body']:
                        raise ValueError
                try:
                    unchanged()
                except (ValueError, KeyError, TypeError):
                    return self._finish(plan_id, 'failed', 'case_changed')
                if len(self.token) >= 8 and self.token in plan['body']:
                    return self._finish(plan_id, 'failed', 'credential_material')
                with (nullcontext(self.session) if self.session is not None else httpx.Client(trust_env=False)) as client:
                    pr = self._request(client, 'GET', f'pulls/{plan["pr_number"]}')
                    base, head = pr.get('base', {}), pr.get('head', {})
                    repo = base.get('repo', {}) if isinstance(base, dict) else {}
                    if (pr.get('number') != plan['pr_number'] or pr.get('state') != 'open'
                            or not isinstance(repo, dict) or repo.get('full_name') != REPOSITORY or repo.get('private') is not False):
                        return self._finish(plan_id, 'failed', 'public_pr_unconfirmed')
                    if not isinstance(head, dict) or head.get('sha') != plan['head_sha'] or base.get('sha') != plan['base_sha']:
                        return self._finish(plan_id, 'failed', 'stale_revision')
                    try:
                        unchanged()
                    except (ValueError, KeyError, TypeError):
                        return self._finish(plan_id, 'failed', 'case_changed')
                    post_started = True
                    result = self._request(client, 'POST', f'pulls/{plan["pr_number"]}/reviews',
                                           payload={key: plan[key] for key in ('commit_id', 'event', 'body')})
                review_id = result.get('id')
                expected_url = f'{WEB}/pull/{plan["pr_number"]}#pullrequestreview-{review_id}'
                if (isinstance(review_id, bool) or not isinstance(review_id, int) or review_id <= 0
                        or result.get('state') != 'COMMENTED' or result.get('commit_id') != plan['head_sha']
                        or result.get('html_url', expected_url) != expected_url):
                    return self._finish(plan_id, 'uncertain', 'response_unconfirmed')
                return self._finish(plan_id, 'published', review_id=review_id)
            except _RemoteFailure as exc:
                return self._finish(plan_id, 'uncertain' if exc.uncertain else 'failed', exc.code)
            except Exception:
                return self._finish(plan_id, 'uncertain' if post_started else 'failed',
                                    'post_outcome_unknown' if post_started else 'preflight_unavailable')

