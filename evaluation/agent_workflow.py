"""Evaluate saved Deep Investigation handoffs on three real code samples.

Without --live this validates fixtures and writes an unscored input inventory.
Live runs leave every report awaiting human review; no approval or code execution
is part of this evaluation. Use a new or empty output directory for every run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time

from triage_desk.agent_evidence import canonical_http_url


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FILES = frozenset({'triage_desk/reviews.py', 'triage_desk/freshness.py', 'triage_desk/hosted_embeddings.py'})
_KINDS = frozenset({'documentation', 'issue', 'release', 'contributing', 'code'})
_LABEL = re.compile(r'[a-zA-Z0-9_-]{1,70}\Z')
_TOKEN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,}|keen_[A-Za-z0-9_-]{12,}|fw_[A-Za-z0-9_-]{16,})')
_SOURCE_FIELDS = ('evidence_id', 'title', 'url', 'kind', 'origin', 'evidence_type', 'filename', 'fetched_at',
                  'snapshot_at', 'version_scope', 'commit_sha', 'sha256', 'published_at', 'acquired_at')
_USAGE = ('input_tokens', 'output_tokens', 'total_tokens')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_cases(root: Path, path: Path, case_ids=None) -> tuple[dict, list[dict], str]:
    """Validate source identities and exact, bounded repository file excerpts."""
    raw = path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset.get('cases') if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError('Evaluation configuration needs a nonempty cases list.')
    corpus = json.loads((root / 'data' / 'sources.json').read_text(encoding='utf-8'))
    source_ids = {source['source_id'] for source in corpus}
    selected = set(case_ids) if case_ids else None
    seen, prepared = set(), []
    for item in cases:
        if not isinstance(item, dict) or not re.fullmatch(r'W3-\d{2}', str(item.get('id', ''))):
            raise ValueError('Each evaluation case needs a W3-NN ID.')
        if item['id'] in seen:
            raise ValueError('Evaluation case IDs must be unique.')
        seen.add(item['id'])
        question = item.get('question')
        if not isinstance(question, str) or not question.strip() or len(question) > 4500:
            raise ValueError('Each evaluation question must contain 1 to 4500 characters.')
        if item.get('code_file') not in FIXTURE_FILES:
            raise ValueError('Evaluation code_file must name one of the three declared source fixtures.')
        if not set(item.get('expected_source_ids', [])) <= source_ids:
            raise ValueError('An expected source ID is absent from the bundled corpus.')
        kinds = item.get('relevant_source_kinds')
        if not isinstance(kinds, list) or not kinds or not set(kinds) <= _KINDS:
            raise ValueError('Each case needs known relevant_source_kinds.')
        if not isinstance(item.get('goals'), list) or not item['goals'] or not item.get('expected_boundary'):
            raise ValueError('Each case needs goals and an expected boundary.')
        fixture = (root / item['code_file']).resolve()
        if not fixture.is_relative_to(root.resolve()):
            raise ValueError('Evaluation fixture must remain inside the repository.')
        file_bytes = fixture.read_bytes()
        lines = file_bytes.decode('utf-8').splitlines(keepends=True)
        start, end = item.get('start_line', 1), item.get('end_line', len(lines))
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                or not isinstance(end, int) or not 1 <= start <= end <= len(lines)):
            raise ValueError('Fixture line bounds must select an existing nonempty inclusive range.')
        excerpt = ''.join(lines[start - 1:end])
        context = f'Supplied repository file: {item["code_file"]}:{start}-{end}\nFile SHA-256: {_sha(file_bytes)}\n\n{excerpt}'
        if len(context) > 16000:
            raise ValueError('Fixture plus path/hash metadata exceeds the 16000-character context limit.')
        if selected is None or item['id'] in selected:
            prepared.append({**item, '_context': context, 'sample': {
                'path': item['code_file'], 'start_line': start, 'end_line': end,
                'file_sha256': _sha(file_bytes), 'excerpt_sha256': _sha(excerpt.encode('utf-8')),
                'excerpt_characters': len(excerpt), 'context_characters': len(context),
            }})
    if selected is not None and (not selected <= seen or not prepared):
        raise ValueError('Selected case IDs must exist in the evaluation configuration.')
    return dataset, prepared, _sha(raw)


def _label(value) -> str:
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else 'other'


def event_accounting(events: list[dict]) -> dict:
    """Count dispatch records once; model/tool_result events supply usage/errors."""
    starts = [event for event in events if event.get('phase') in {'model', 'tool'}]
    terminals = [event for event in events if event.get('phase') in {'model_result', 'tool_result'}]
    results = {event['call_id']: event for event in terminals if event.get('call_id')}
    model = [event for event in starts if event['phase'] == 'model']
    tool = [event for event in starts if event['phase'] == 'tool']
    usage, missing, errors, unfinished = Counter(), Counter(), Counter(), 0
    for event in starts:
        terminal = results.get(event.get('call_id'))
        # Older completed records are self-contained. New started records need
        # their paired terminal record and cannot imply a successful request.
        if terminal is None and event.get('status') != 'started':
            terminal = event
        if terminal is None:
            unfinished += 1
        elif terminal.get('status') in {'error', 'limited', 'failed'}:
            errors[_label(event.get('name'))] += 1
        if event['phase'] == 'model':
            counts = terminal.get('usage', {}) if terminal else {}
            for field in _USAGE:
                count = counts.get(field)
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    usage[field] += count
                else:
                    missing[field] += 1
    return {
        'model_calls': len(model), 'tool_calls': len(tool),
        'model_calls_by_role': dict(Counter(_label(event.get('name')) for event in model)),
        'tool_calls_by_name': dict(Counter(_label(event.get('name')) for event in tool)),
        'errors_by_role_or_tool': dict(errors), 'unfinished_call_records': unfinished,
        'reported_token_usage': {field: usage[field] for field in _USAGE},
        'model_calls_missing_usage_by_field': {field: missing[field] for field in _USAGE},
        'usage_complete': not any(missing.values()),
        'note': 'Counts are observed dispatch records, not billing totals. Usage is summed only when reported; no cost or missing tokens are inferred.',
    }


def citation_integrity(report: dict | None, sources: list[dict]) -> dict:
    """Check exact saved source text; this cannot establish entailment."""
    registered = {source.get('evidence_id'): source for source in sources}
    checked, failed = 0, 0
    for finding in (report or {}).get('findings', []):
        support = finding.get('support', [])
        if not support:
            failed += 1
        for reference in support:
            checked += 1
            source, quote = registered.get(reference.get('evidence_id')), reference.get('quote')
            if source is None or not isinstance(quote, str) or not quote.strip() or quote not in source.get('text', ''):
                failed += 1
    return {'status': 'failed' if failed else 'passed' if checked else 'not_applicable',
            'checked_supports': checked, 'failed_supports': failed,
            'meaning': 'Exact quote/source integrity only; not entailment, accuracy, or human validation.'}


def _provenance(sources: list[dict]) -> list[dict]:
    records = []
    for source in sources:
        record = {key: source.get(key, '') for key in _SOURCE_FIELDS}
        try:
            record['url'] = canonical_http_url(record['url'])
        except ValueError:
            record['url'] = ''
            record['url_rejected'] = True
        records.append(record)
    return records


def _redact(value, secrets: set[str]):
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, '[REDACTED]')
        return _TOKEN.sub('[REDACTED]', value)
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(_redact(key, secrets)): _redact(item, secrets) for key, item in value.items()}
    return value


def _default_service(root: Path):
    from triage_desk.core import TriageDesk
    from triage_desk.deep_investigation import AgentSettings, InvestigationService
    settings = AgentSettings.from_env()
    if not settings.readiness()['ready']:
        raise ValueError('Live evaluation requires Fireworks model configuration.')
    return InvestigationService(root, TriageDesk(root / 'data', root / '.cache' / 'retrieval'), settings)


def summarize(rows: list[dict], *, live: bool) -> dict:
    scheduled = len(rows)
    handoffs = sum(row['handoff_observed'] for row in rows)
    summary = {'scheduled_cases': scheduled, 'attempted_cases': sum(row['attempted'] for row in rows),
               'completed_handoffs': handoffs, 'failed_cases': sum(bool(row['error']) for row in rows),
               'not_run_cases': sum(row['status'] == 'not_run' for row in rows),
               'handoff_rate': handoffs / scheduled if live and scheduled else None,
               'handoff_denominator': 'All scheduled selected cases, including errors. This measures saved review handoff completion, not answer accuracy.',
               'human_accuracy': None, 'human_citation_entailment': None,
               'citation_integrity_passed_cases': sum(row['automatic_citation_integrity']['status'] == 'passed' for row in rows),
               'ai_rejected_findings': sum((row['independent_ai_review'] or {}).get('removed_findings', 0) for row in rows),
               'ai_rejected_next_steps': sum((row['independent_ai_review'] or {}).get('removed_next_steps', 0) for row in rows)}
    for field in ('model_calls', 'tool_calls', 'unfinished_call_records'):
        summary[field] = sum(row['accounting'][field] for row in rows)
    for field in ('model_calls_by_role', 'tool_calls_by_name', 'errors_by_role_or_tool',
                  'reported_token_usage', 'model_calls_missing_usage_by_field'):
        combined = Counter()
        for row in rows:
            combined.update(row['accounting'][field])
        summary[field] = dict(combined)
    summary['reported_tokens'] = summary['reported_token_usage'].get('total_tokens', 0)
    summary['accounting_unknown_error_rows'] = sum(row.get('accounting_unknown', False) for row in rows)
    summary['usage_complete'] = (all(row['accounting']['usage_complete'] for row in rows)
                                 and not summary['accounting_unknown_error_rows']) if live else None
    return summary


def run_evaluation(root, cases_path, output_dir, *, service_factory=None, case_ids=None, live=False,
                   repository_context=False) -> dict:
    """Write a reproducible report; providers are unreachable unless live=True."""
    root = Path(root).resolve()
    if not isinstance(repository_context, bool):
        raise ValueError('repository_context must be a boolean')
    cases_path, output_dir = Path(cases_path), Path(output_dir)
    cases_path = cases_path if cases_path.is_absolute() else root / cases_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    dataset, cases, dataset_hash = load_cases(root, cases_path, case_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError('Output directory must be new or empty; previous evaluation artifacts are never overwritten.')
    marker = output_dir / '.evaluation-in-progress'
    with marker.open('x'):
        pass
    service, initialization_error = None, None
    if live:
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
        try:
            service = (service_factory or _default_service)(root)
        except Exception as exc:
            initialization_error = {'type': type(exc).__name__, 'message': 'Live service configuration could not initialize.'}
    secrets = {value for name, value in os.environ.items()
               if re.search(r'(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)$', name, re.I) and 8 <= len(value) <= 4096}
    settings = getattr(service, 'settings', None)
    for field in ('api_key', 'keenable_api_key', 'github_token'):
        value = getattr(settings, field, '')
        if isinstance(value, str) and len(value) >= 8:
            secrets.add(value)
    rows = []
    for fixture in cases:
        row = {'id': fixture['id'], 'eval_id': fixture['id'], 'question': fixture['question'], 'sample': fixture['sample'],
               'goals': fixture['goals'], 'relevant_source_kinds': fixture['relevant_source_kinds'],
               'expected_boundary': fixture['expected_boundary'],
               'attempted': False, 'status': 'not_run', 'case_id': None,
               'handoff_required': True, 'handoff_observed': False, 'persisted_case_observed': False,
               'report': None, 'independent_ai_review': None, 'sources': [], 'error': None,
               'accounting': event_accounting([]), 'elapsed_ms': None,
               'automatic_citation_integrity': citation_integrity(None, []),
               'human_review': {'accuracy': None, 'relevance': None, 'citation_entailment': None, 'notes': None}}
        if live and initialization_error:
            row.update(status='not_started', error=initialization_error, accounting_unknown=True)
        elif live:
            started = time.perf_counter()
            row['attempted'] = True
            try:
                options = {'repository_context': True} if repository_context else {}
                case = service.start(fixture['question'], fixture.get('version', ''), fixture['_context'], **options)
                row.update(case_id=case['case_id'], status=case['status'])
                row['repository_context'] = case.get('repository_context', False)
                saved = service.store.get_case(case['case_id'])
                row['persisted_case_observed'] = saved['case_id'] == case['case_id'] and saved['status'] == case['status']
                sources = service.store.evidence(case['case_id'])
                row['accounting'] = event_accounting(service.store.events(case['case_id']))
                row['report'] = {key: value for key, value in (case.get('report') or {}).items()
                                 if key in {'findings', 'next_steps', 'missing_information', 'quality_review'}} or None
                row['independent_ai_review'] = (case.get('report') or {}).get('quality_review')
                row['sources'] = _provenance(sources)
                row['automatic_citation_integrity'] = citation_integrity(case.get('report'), sources)
                row['handoff_observed'] = case['status'] == 'awaiting_review' and row['persisted_case_observed']
                if case['status'] == 'failed' or case.get('error'):
                    row['error'] = {'type': 'InvestigationFailed', 'message': 'The saved investigation did not finish; inspect its local audit events.'}
                    from triage_desk.agent_review import REVIEW_FAILURE_CODES
                    codes = [event.get('error_code') for event in service.store.events(case['case_id'])
                             if event.get('phase') == 'execution' and event.get('status') == 'error'
                             and event.get('error_code') in REVIEW_FAILURE_CODES]
                    if codes:
                        row['error']['code'] = codes[-1]
            except Exception as exc:
                row.update(status='error', accounting_unknown=True,
                           error={'type': type(exc).__name__, 'message': 'The evaluation case could not finish; raw provider errors are excluded.'})
            row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
        rows.append(row)
    report = _redact({
        'schema_version': 1, 'report_kind': 'deep_agent_workflow',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'execution_mode': 'live' if live else 'validation_only', 'dataset_name': dataset.get('name'),
        'repository_context_enabled': repository_context,
        'dataset_sha256': dataset_hash, 'corpus_sha256': _sha((root / 'data' / 'sources.json').read_bytes()),
        'summary': summarize(rows, live=live), 'results': rows,
        'measurement_limits': [
            'No report is approved, rejected, posted, or executed by this evaluation. Saved awaiting_review status establishes a handoff only.',
            'Automatic citation checks establish literal registered quotes, not claim entailment or factual accuracy.',
            'Independent AI review is model judgment with the reported reviewer identity; it is not independent human validation.',
            'Human quality fields remain null until an actual human evaluates the cases. No accuracy, faithfulness, or entailment rate is inferred.',
            'Accounting uses observed dispatch/result events. Missing token usage and unobserved failed-call accounting remain explicit.',
            'Only generated reports and bounded source provenance are exported; full source bodies, raw tool history, environment values, and provider error bodies are excluded.',
            'Configured credential values and common token patterns are redacted before export. Integrity is checked against original local evidence before redaction.',
        ],
    }, secrets)
    with (output_dir / 'report.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (output_dir / 'report.html').open('x', encoding='utf-8') as stream:
        stream.write(render_report(report))
    marker.unlink()
    return report


def render_report(report: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    pretty = lambda value: esc(json.dumps(value, ensure_ascii=False, indent=2))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Week 3 investigation evaluation</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.5;color:#172033}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:18px;border-radius:8px}article{border-top:1px solid #ccd3df;margin-top:32px}h1,h2{line-height:1.2}a{color:#1458ab}</style>',
             '<h1>Week 3 Deep Investigation evaluation</h1>',
             '<p>Mode: ' + esc(report['execution_mode']) + '. Human quality fields are unscored.</p>',
             '<h2>Observed handoffs and accounting</h2><pre>' + pretty(report['summary']) + '</pre>',
             '<h2>Measurement limits</h2><ul>']
    parts.extend('<li>' + esc(item) + '</li>' for item in report['measurement_limits'])
    parts.append('</ul>')
    for row in report['results']:
        parts.extend(['<article><h2>' + esc(row['id']) + ' · ' + esc(row['status']) + '</h2>',
                      '<p>' + esc(row['question']) + '</p>', '<h3>Code sample provenance</h3><pre>' + pretty(row['sample']) + '</pre>',
                      '<h3>Generated report</h3><pre>' + pretty(row['report']) + '</pre>',
                      '<h3>Source provenance</h3><ul>'])
        for source in row['sources']:
            try:
                url = canonical_http_url(source.get('url', ''))
            except ValueError:
                url = ''
            title = esc(source.get('title', 'Source'))
            label = '<a rel="noreferrer" href="' + esc(url) + '">' + title + '</a>' if url else title
            parts.append('<li>' + label + '<pre>' + pretty(source) + '</pre></li>')
        parts.extend(['</ul><h3>Observed accounting and checks</h3><pre>' + pretty({
            key: row[key] for key in ('elapsed_ms', 'handoff_observed', 'persisted_case_observed', 'automatic_citation_integrity',
                                     'accounting', 'independent_ai_review', 'human_review', 'error')}) + '</pre></article>'])
    parts.append('</html>')
    return '\n'.join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cases', type=Path, default=Path('evaluation/week3-cases.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation/week3-results'))
    parser.add_argument('--case', action='append', dest='case_ids', help='Select a case ID; repeat to select more.')
    parser.add_argument('--live', action='store_true', help='Explicitly permit provider calls; leave cases awaiting human review.')
    parser.add_argument('--repository-context', action='store_true',
                        help='Allow bounded related application-code reads for each case; selected code is sent to Fireworks.')
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(args.root, args.cases, args.output_dir, case_ids=args.case_ids, live=args.live,
                                repository_context=args.repository_context)
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({'status': 'error', 'error': 'Evaluation input or output is invalid. Use valid fixtures and a new or empty output directory.'}))
        return 1
    print(json.dumps({'execution_mode': report['execution_mode'], 'summary': report['summary'],
                      'output_directory': str(args.output_dir)}, indent=2))
    return int(bool(report['summary']['failed_cases']))


if __name__ == '__main__':
    raise SystemExit(main())
