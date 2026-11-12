"""Case-scoped citation integrity for agent reports.

Matching an exact quote proves that the citation exists in the recorded source;
it does not prove that the source entails the agent's statement. Human review is
still required. Source text is untrusted data, never an instruction to execute.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote, urlsplit, urlunsplit

from triage_desk.cases import CaseStore


MAX_REPORT_CHARS = 80_000
MAX_FINDINGS = 12
MAX_SUPPORTS = 4
MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 1400
MAX_STATEMENT_CHARS = 1200
MAX_ACTIONS = 8
MAX_ACTION_CHARS = 1000
_QUOTE_SPAN_CHARS = 1000
_QUOTE_SPAN_OVERLAP = 160
_CATEGORIES = frozenset({'docs', 'correctness', 'security', 'tests'})
_SEVERITIES = frozenset({'low', 'medium', 'high', 'critical'})
_LINK_OR_HTML = re.compile(
    r'(?i)(?:\b(?:https?|ftp|file|data|javascript)\s*:|\bwww\.|'
    r'\[[^\]\n]*\]\s*(?:\(|\[)|<(?:/?[a-z][a-z0-9:-]*(?=[\s/>])[^>]*|![^>]*)>)'
)
_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_MARKDOWN = re.compile(r'([\\`*_{}\[\]()#+\-.!|>~])')


def canonical_http_url(value: str) -> str:
    """Allow only ordinary HTTP(S) links and encode Markdown delimiters."""
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError('Source URL must be text of at most 4000 characters')
    if not value:
        return ''
    if re.search(r'\s', value) or _CONTROL.search(value) or '\\' in value:
        raise ValueError('Source URL contains unsafe whitespace or delimiters')
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {'http', 'https'} or not parts.hostname:
            raise ValueError('Source URL must use HTTP or HTTPS with a host')
        if parts.username is not None or parts.password is not None:
            raise ValueError('Source URL must not contain credentials')
        host = parts.hostname.encode('idna').decode('ascii').lower()
        if not re.fullmatch(r'[a-z0-9.:-]+', host):
            raise ValueError('Source URL contains an invalid host')
        port = parts.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Source URL is not a safe HTTP(S) URL') from exc
    if ':' in host:
        host = f'[{host}]'
    if port is not None and not (parts.scheme.lower() == 'https' and port == 443) and not (
        parts.scheme.lower() == 'http' and port == 80
    ):
        host += f':{port}'
    # Parentheses, square brackets, angle brackets, quotes, and backticks cannot
    # terminate the Markdown link produced by the application renderer.
    safe = "/:%@!$&'*+,;=-._~"
    return urlunsplit((parts.scheme.lower(), host, quote(parts.path, safe=safe),
                       quote(parts.query, safe=safe + '?='), quote(parts.fragment, safe=safe + '?=')))


def _plain(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{name} must be nonempty text of at most {limit} characters')
    if _CONTROL.search(value) or _LINK_OR_HTML.search(value):
        raise ValueError(f'{name} must be plain text without links or HTML')
    return value


def _array(value: object, name: str, limit: int) -> list:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f'{name} must be a list of at most {limit} items')
    return value


def _escaped(value: str) -> str:
    # Escaping angle brackets before Markdown stops code and source titles from
    # becoming raw HTML; backslashes neutralize all Markdown control characters.
    return _MARKDOWN.sub(r'\\\1', value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _literal_quote_spans(text: str, evidence_id: str) -> list[dict]:
    """Derive stable, overlapping character spans without changing source text.

    Word-based windows can exceed quote limits for code or long URLs. Character
    windows bound every span; a whitespace boundary near the end keeps ordinary
    prose readable. The final overlap retains short trailing passages.
    """
    spans, start = [], 0
    while start < len(text):
        end = min(start + _QUOTE_SPAN_CHARS, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r'\s+', text[end - _QUOTE_SPAN_OVERLAP:end]))
            if boundaries:
                end = end - _QUOTE_SPAN_OVERLAP + boundaries[-1].start()
        literal = text[start:end]
        if len(literal.strip()) >= MIN_QUOTE_CHARS:
            spans.append({'quote_id': f'{evidence_id}:q{len(spans) + 1}', 'quote': literal})
        if end == len(text):
            break
        start = end - _QUOTE_SPAN_OVERLAP
    return spans


def _project_source(source: dict) -> dict:
    return {**source, 'quote_spans': _literal_quote_spans(source['text'], source['evidence_id'])}


def _resolve_support(support: object, registered: dict[str, dict]) -> tuple[dict, dict]:
    """Resolve an ID to saved text, or verify a legacy literal citation."""
    if not isinstance(support, dict) or set(support) not in (
        {'evidence_id', 'quote_id'}, {'evidence_id', 'quote'}, {'evidence_id', 'quote_id', 'quote'},
    ):
        raise ValueError('Citation must select an evidence ID and a quote ID or legacy literal quote')
    evidence_id = support['evidence_id']
    if not isinstance(evidence_id, str) or evidence_id not in registered:
        raise ValueError('Citation refers to unregistered evidence')
    source = registered[evidence_id]
    resolved = {'evidence_id': evidence_id}
    if 'quote_id' in support:
        quote_id = support['quote_id']
        # Derive from saved text, never trust caller-supplied quote_spans.
        spans = {span['quote_id']: span['quote']
                 for span in _literal_quote_spans(source['text'], evidence_id)}
        if not isinstance(quote_id, str) or quote_id not in spans:
            raise ValueError('Citation quote ID does not belong to its registered evidence')
        literal = spans[quote_id]
        if 'quote' in support and support['quote'] != literal:
            raise ValueError('Citation quote does not exactly match its selected quote ID')
        resolved['quote_id'] = quote_id
    else:
        literal = support['quote']
    if not isinstance(literal, str) or len(literal.strip()) < MIN_QUOTE_CHARS or len(literal) > MAX_QUOTE_CHARS:
        raise ValueError('Citation needs a bounded, nonempty literal quote')
    if literal not in source['text']:
        raise ValueError('Citation quote is not an exact part of its registered source')
    resolved['quote'] = literal
    return source, resolved


class EvidenceRegistry:
    """Register source snapshots and validate literal citations within one case."""

    def __init__(self, store: CaseStore, case_id: str):
        store.get_case(case_id)
        self.store, self.case_id = store, case_id

    def register_source(
        self, *, source_key: str, title: str, text: str, url: str = '',
        kind: str = 'documentation', origin: str = 'snapshot', fetched_at: str = '',
        commit_sha: str = '', snapshot_at: str = '', version_scope: str = '',
        evidence_type: str = '', published_at: str = '', acquired_at: str = '',
        base_sha: str = '', filename: str = '', truncated: bool = False,
    ) -> dict:
        url = canonical_http_url(url)
        if origin == 'live' and not url:
            raise ValueError('Live evidence needs its source URL')
        source = self.store.put_evidence(self.case_id, {
            'source_key': source_key, 'title': title, 'text': text, 'url': url,
            'kind': kind, 'origin': origin, 'fetched_at': fetched_at,
            'commit_sha': commit_sha, 'snapshot_at': snapshot_at, 'version_scope': version_scope,
            'evidence_type': evidence_type, 'published_at': published_at, 'acquired_at': acquired_at,
            'base_sha': base_sha, 'filename': filename, 'truncated': truncated,
        })
        return _project_source(source)

    def list_sources(self) -> list[dict]:
        return [_project_source(source) for source in self.store.evidence(self.case_id)]

    def validate_report(self, report: dict) -> dict:
        """Reject fabricated citations and preserve proposed actions as proposals.

        This validates citation integrity, not entailment or factual correctness.
        Findings select quote IDs; the application supplies saved literal text.
        Legacy explicit quotes must still match an exact registered substring.
        """
        if not isinstance(report, dict) or set(report) - {'findings', 'next_steps', 'missing_information'}:
            raise ValueError('Report must contain only findings, next_steps, and missing_information')
        try:
            payload = json.dumps(report, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('Report must contain only JSON values') from exc
        if len(payload) > MAX_REPORT_CHARS:
            raise ValueError('Report is too large')
        registered = {source['evidence_id']: source for source in self.list_sources()}
        findings, cited = [], {}
        for index, finding in enumerate(_array(report.get('findings'), 'findings', MAX_FINDINGS), 1):
            if not isinstance(finding, dict) or set(finding) != {'statement', 'category', 'severity', 'support'}:
                raise ValueError(f'Finding {index} has an invalid structure')
            category, severity = finding['category'], finding['severity']
            if not isinstance(category, str) or category not in _CATEGORIES:
                raise ValueError(f'Finding {index} has an unknown category')
            if not isinstance(severity, str) or severity not in _SEVERITIES:
                raise ValueError(f'Finding {index} has an unknown severity')
            statement = _plain(finding['statement'], f'Finding {index} statement', MAX_STATEMENT_CHARS)
            supports = _array(finding['support'], f'Finding {index} support', MAX_SUPPORTS)
            if not supports:
                raise ValueError(f'Finding {index} needs at least one exact supporting quote')
            validated_supports, seen = [], set()
            for support in supports:
                source, resolved = _resolve_support(support, registered)
                evidence_id, literal = resolved['evidence_id'], resolved['quote']
                if (evidence_id, literal) in seen:
                    raise ValueError(f'Finding {index} repeats a citation')
                seen.add((evidence_id, literal))
                validated_supports.append(resolved)
                cited[evidence_id] = {key: value for key, value in source.items() if key not in {'text', 'quote_spans'}}
            findings.append({'statement': statement, 'category': category, 'severity': severity,
                             'support': validated_supports})
        return {
            'findings': findings,
            'missing_information': [
                _plain(item, 'missing_information item', MAX_ACTION_CHARS)
                for item in _array(report.get('missing_information'), 'missing_information', MAX_ACTIONS)
            ],
            'next_steps': [
                _plain(item, 'next_steps item', MAX_ACTION_CHARS)
                for item in _array(report.get('next_steps'), 'next_steps', MAX_ACTIONS)
            ],
            'sources': list(cited.values()),
        }


def render_report_markdown(validated_report: dict, sources: list[dict]) -> str:
    """Render escaped report text with links supplied only by the application.

    Use a report returned by ``validate_report``. This function independently
    checks support against the supplied source snapshots before rendering.
    """
    registered = {source['evidence_id']: source for source in sources}
    lines = ['## Investigation report', '',
             'Citation integrity checked against saved evidence. Human review is required to assess the findings.', '']
    findings = validated_report.get('findings', [])
    if not findings:
        lines.extend(['No supported findings were produced.', ''])
    for index, finding in enumerate(findings, 1):
        statement = _plain(finding['statement'], 'statement', MAX_STATEMENT_CHARS)
        category = finding['category']
        severity = finding['severity']
        if category not in _CATEGORIES or severity not in _SEVERITIES:
            raise ValueError('Unknown finding category or severity')
        lines.extend([f'### {index}. {_escaped(category.title())} · {_escaped(severity.title())}', '',
                      _escaped(statement), ''])
        if not finding['support']:
            raise ValueError('Cannot render an unsupported finding')
        for support in finding['support']:
            source, resolved = _resolve_support(support, registered)
            evidence_id, literal = resolved['evidence_id'], resolved['quote']
            title = _escaped(source['title'])
            url = canonical_http_url(source.get('url', ''))
            label = f'[{title}]({url})' if url else title
            origin = _escaped(source['origin'])
            lines.append(f'Source: {label} · {origin} · {_escaped(evidence_id)}')
            lines.append('')
            lines.extend(f'> {_escaped(line)}' for line in literal.splitlines())
            lines.append('')
    for field, heading in [('missing_information', 'Missing information'), ('next_steps', 'Proposed next steps')]:
        items = validated_report.get(field, [])
        if items:
            lines.extend([f'### {heading}', ''])
            for item in items:
                lines.append('- ' + _escaped(_plain(item, field, MAX_ACTION_CHARS)).replace('\n', ' '))
            lines.append('')
    return '\n'.join(lines).rstrip() + '\n'
