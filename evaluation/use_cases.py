"""Evaluate complete engineering workflows, with explicit opt-in provider calls.

The default validates all pinned fixtures and exports an unscored inventory.
Live cases use separate stores and checkpoints and stop at human review. They
never approve reports, execute supplied code, or publish a review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from evaluation.agent_workflow import citation_integrity, event_accounting, _provenance, _redact
from triage_desk.use_cases import USE_CASES, get_use_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path('evaluation/workflow-cases.json')
PROFILE_IDS = frozenset(profile.id for profile in USE_CASES)
_EVENT_FIELDS = frozenset({
    'event_id', 'created_at', 'phase', 'name', 'status', 'call_id', 'started_at',
    'latency_ms', 'usage', 'usage_reported', 'finish_reason', 'returned_model',
    'tool_call_count', 'invalid_tool_call_count', 'specialist', 'error_type',
    'error_code', 'revision', 'repair_number', 'findings', 'retained_findings',
    'removed_findings', 'removed_next_steps', 'required',
    'converted_count', 'policy_version',
})
_CODE_FILES = ('evaluation/use_cases.py', 'evaluation/agent_workflow.py',
               'triage_desk/use_cases.py', 'triage_desk/deep_investigation.py',
               'triage_desk/agent_review.py', 'triage_desk/agent_evidence.py',
               'triage_desk/agent_budget.py', 'triage_desk/agent_tools.py',
               'triage_desk/cases.py', 'triage_desk/core.py',
               'triage_desk/information_requests.py',
               'triage_desk/static_analysis.py', 'triage_desk/_static_analysis_worker.py',
               'triage_desk/review_extensions.py')
TARGETS = {'handoff_rate': {'operator': '>=', 'target': .95},
           'contract_pass_rate': {'operator': '>=', 'target': .90},
           'quoted_anchor_recall': {'operator': '>=', 'target': .85},
           'latency_p95_seconds': {'operator': '<=', 'target': 90},
           'mean_reported_tokens': {'operator': '<=', 'target': 20000}}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _strings(value, *, nonempty=True):
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(item, str) and item.strip() for item in value))


def input_projection(case: dict) -> dict:
    """The entire caller input. Evaluation labels are deliberately absent."""
    return {'question': case['question'], 'version': case.get('version', ''),
            'context': case['context']['text'], 'use_case': case['use_case']}


def load_cases(root: Path, path: Path = DEFAULT_CASES, case_ids=None, use_cases=None):
    """Validate the whole suite before applying any selection or loading .env."""
    root, path = Path(root).resolve(), Path(path)
    path = path if path.is_absolute() else root / path
    raw = path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset.get('cases') if isinstance(dataset, dict) else None
    if (not isinstance(dataset, dict) or type(dataset.get('version')) is not int
            or dataset.get('version') != 1 or dataset.get('label_status') != 'ai_proposed_pending_human_review'
            or not isinstance(dataset.get('name'), str) or not dataset['name'].strip()
            or not isinstance(cases, list) or len(cases) != 40):
        raise ValueError('The workflow suite requires version 1 and 40 provisional cases.')
    counts, seen = Counter(), set()
    for case in cases:
        if not isinstance(case, dict) or not re.fullmatch(r'WF-\d{2}', str(case.get('id', ''))):
            raise ValueError('Each workflow case needs a WF-NN ID.')
        if case['id'] in seen:
            raise ValueError('Workflow case IDs must be unique.')
        seen.add(case['id'])
        profile = get_use_case(case.get('use_case'))
        counts[profile.id] += 1
        for key, limit in (('title', 300), ('scenario', 1000), ('question', 4500)):
            if not isinstance(case.get(key), str) or not case[key].strip() or len(case[key]) > limit:
                raise ValueError(f'Workflow case {key} is missing or too long.')
        if not isinstance(case.get('version', ''), str) or len(case.get('version', '')) > 120:
            raise ValueError('Workflow case version must be bounded text.')
        context = case.get('context')
        if not isinstance(context, dict) or not isinstance(context.get('path'), str):
            raise ValueError('Workflow context needs a pinned fixture path.')
        relative = Path(context['path'])
        fixture = (root / relative).resolve()
        allowed = (root / 'evaluation' / 'fixtures' / 'workflows').resolve()
        if (relative.is_absolute() or not allowed.is_relative_to(root)
                or not fixture.is_relative_to(allowed) or not fixture.is_file()):
            raise ValueError('Workflow context must be an existing bundled workflow fixture.')
        body = fixture.read_bytes()
        text = context.get('text')
        if (not isinstance(text, str) or not text.strip() or len(text) > 16000
                or text.encode('utf-8') != body or context.get('sha256') != _sha(body)):
            raise ValueError('Workflow context text and SHA-256 must match the exact fixture bytes.')
        expected = case.get('expected')
        if (not isinstance(expected, dict) or not _integer(expected.get('min_findings'), 0, 12)
                or not _integer(expected.get('min_next_steps'), 0, 8)
                or not _strings(expected.get('required_tools'))
                or 'read_case_evidence' not in expected['required_tools']
                or not _strings(expected.get('evidence_anchors'), nonempty=False)
                or len(set(expected['evidence_anchors'])) != len(expected['evidence_anchors'])
                or any(anchor not in text for anchor in expected['evidence_anchors'])
                or not isinstance(expected.get('behavior_notes'), str)
                or not _strings(expected.get('human_rubric'))):
            raise ValueError('Workflow expectations need valid counts, tools, literal anchors, and a human rubric.')
        review = case.get('label_review')
        if not isinstance(review, dict) or review.get('status') != 'pending' or review.get('reviewer') is not None:
            raise ValueError('Provisional workflow labels must explicitly await human review.')
    if counts != Counter({identity: 10 for identity in PROFILE_IDS}):
        raise ValueError('The workflow suite requires ten cases for each known use case.')
    selected_ids, selected_profiles = set(case_ids or ()), set(use_cases or ())
    if selected_ids and not selected_ids <= seen:
        raise ValueError('Selected case IDs must exist in the workflow suite.')
    if selected_profiles and not selected_profiles <= PROFILE_IDS:
        raise ValueError('Selected use cases must be known workflow IDs.')
    selected = [case for case in cases if (not selected_ids or case['id'] in selected_ids)
                and (not selected_profiles or case['use_case'] in selected_profiles)]
    if not selected:
        raise ValueError('The selected case and use-case filters have no cases in common.')
    return dataset, selected, _sha(raw)


def _secrets(settings=None):
    values = {value for name, value in os.environ.items()
              if re.search(r'(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)$', name, re.I)
              and 8 <= len(value) <= 4096}
    for field in ('api_key', 'keenable_api_key', 'github_token'):
        value = getattr(settings, field, '')
        if isinstance(value, str) and len(value) >= 8:
            values.add(value)
    return values


def _write_json(path: Path, value, secrets):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _redact(value, secrets)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _default_service(workspace: Path, *, source_root: Path, settings, reviewer_prompt_version: str):
    from triage_desk.core import TriageDesk
    from triage_desk.deep_investigation import InvestigationService
    if not settings.readiness()['ready']:
        raise ValueError('Live evaluation requires Fireworks model configuration.')
    return InvestigationService(workspace, TriageDesk(source_root / 'data', workspace / '.cache' / 'retrieval'),
                                settings, reviewer_prompt_version=reviewer_prompt_version)


def _base_row(case):
    inputs = input_projection(case)
    return {'id': case['id'], 'eval_id': case['id'], 'use_case': case['use_case'],
            'title': case['title'], 'scenario': case['scenario'], 'question': case['question'],
            'input': inputs, 'input_sha256': _sha(_canonical(inputs)),
            'case_definition_sha256': _sha(_canonical(case)),
            'fixture': {key: case['context'][key] for key in ('path', 'sha256')},
            'expected': case['expected'], 'label_review': case['label_review'],
            'attempted': False, 'status': 'not_run', 'case_id': None, 'error': None,
            'handoff_observed': False, 'persisted_case_observed': False,
            'report': None, 'independent_ai_review': None, 'sources': [], 'events': [],
            'accounting': event_accounting([]), 'elapsed_ms': None,
            'automatic_citation_integrity': citation_integrity(None, []),
            'scores': {key: None for key in ('identity_preserved', 'persisted_handoff',
                'independent_review_completed', 'min_findings_met', 'min_next_steps_met',
                'required_tools_met', 'quoted_anchor_recall', 'anchors_checked',
                'anchors_covered', 'contract_passed')},
            'human_review': {'status': 'pending', 'accuracy': None, 'relevance': None,
                             'citation_entailment': None, 'rubric': case['expected']['human_rubric'],
                             'notes': ''}}


def score_case(case, row, saved, original_sources, original_events):
    """Measure literal contracts, never semantic accuracy or human agreement."""
    report = row.get('report') or {}
    findings, next_steps = report.get('findings', []), report.get('next_steps', [])
    integrity = citation_integrity(report, original_sources)
    registered = {source.get('evidence_id'): source for source in original_sources}
    quotes = [support['quote'] for finding in findings for support in finding.get('support', [])
              if isinstance(support.get('quote'), str)
              and support.get('evidence_id') in registered
              and support['quote'] in registered[support['evidence_id']].get('text', '')]
    anchors = case['expected']['evidence_anchors']
    covered = sum(any(anchor in quote for quote in quotes) for anchor in anchors)
    successful_tools = {event.get('name') for event in original_events
                        if event.get('phase') == 'tool_result' and event.get('status') in {'completed', 'ok'}}
    actual_reviewer = any(event.get('phase') == 'model_result' and event.get('name') == 'evidence_reviewer'
                          and event.get('status') == 'completed' for event in original_events)
    profile = get_use_case(case['use_case'])
    identity = bool(saved and saved.get('use_case_id') == profile.id
                    and saved.get('use_case_version') == profile.version)
    review = row.get('independent_ai_review') or {}
    scores = {'identity_preserved': identity, 'persisted_handoff': row['handoff_observed'],
              'independent_review_completed': review.get('status') == 'completed' and actual_reviewer,
              'min_findings_met': len(findings) >= case['expected']['min_findings'],
              'min_next_steps_met': len(next_steps) >= case['expected']['min_next_steps'],
              'required_tools_met': set(case['expected']['required_tools']) <= successful_tools,
              'quoted_anchor_recall': covered / len(anchors) if anchors else None,
              'anchors_checked': len(anchors), 'anchors_covered': covered}
    scores['contract_passed'] = (not row['error'] and all(scores[key] for key in (
        'identity_preserved', 'persisted_handoff', 'independent_review_completed',
        'min_findings_met', 'min_next_steps_met', 'required_tools_met'))
        and integrity['status'] != 'failed'
        and (bool(findings) or case['expected']['min_findings'] == 0))
    row['automatic_citation_integrity'] = integrity
    row['scores'] = scores


def _execute(case, *, root, run_id, settings, service_factory, reviewer_prompt, secrets):
    row, saved, sources, events = _base_row(case), None, [], []
    started = time.perf_counter()
    workspace = root / '.cache' / 'use-case-evals' / run_id / case['id']
    row['workspace'] = str(workspace.relative_to(root))
    try:
        # Each thread owns a service, model factory, SQLite store and checkpoint.
        workspace.mkdir(parents=True, exist_ok=True)
        from langsmith import tracing_context
        with tracing_context(enabled=False):
            service = (service_factory or _default_service)(workspace, source_root=root, settings=settings,
                reviewer_prompt_version=reviewer_prompt)
            secrets = secrets | _secrets(getattr(service, 'settings', None))
            row['attempted'] = True
            inputs = input_projection(case)
            result = service.start(inputs['question'], inputs['version'], inputs['context'], use_case=inputs['use_case'])
            row.update(case_id=result['case_id'], status=result['status'])
            saved = service.store.get_case(result['case_id'])
            sources = service.store.evidence(result['case_id'])
            events = service.store.events(result['case_id'])
            row['persisted_case_observed'] = (saved['case_id'] == result['case_id']
                                              and saved['status'] == result['status'])
            row['handoff_observed'] = (result['status'] == 'awaiting_review' and row['persisted_case_observed'])
            row['report'] = {key: value for key, value in (saved.get('report') or {}).items()
                             if key in {'findings', 'next_steps', 'missing_information', 'quality_review'}} or None
            row['independent_ai_review'] = (row['report'] or {}).get('quality_review')
            row['saved_use_case_id'] = saved.get('use_case_id')
            row['saved_use_case_version'] = saved.get('use_case_version')
            row['sources'] = _provenance(sources)
            row['events'] = [{key: value for key, value in event.items() if key in _EVENT_FIELDS} for event in events]
            row['accounting'] = event_accounting(events)
            if saved['status'] != 'awaiting_review' or saved.get('error'):
                row['error'] = {'type': 'InvestigationFailed', 'message': 'The saved investigation did not reach human review.'}
    except Exception:
        # Provider and constructor exceptions can contain source bodies or keys.
        row.update(status='error' if row['attempted'] else 'not_started', accounting_unknown=True,
                   error={'type': 'EvaluationExecutionError',
                          'message': 'The case could not finish. Raw provider and configuration errors are excluded.'})
    row['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
    score_case(case, row, saved, sources, events)
    row['trace'] = {'name': 'engineering_workflow_evaluation', 'case_id': case['id'],
                    'metadata': {'use_case_id': case['use_case'], 'input_sha256': row['input_sha256'],
                                 'reviewer_prompt_version': reviewer_prompt, 'remote_tracing': False},
                    'children': row['events']}
    return _redact(row, secrets)


def _safe_execute(case, **options):
    """Unexpected per-case failures also occupy their scheduled denominator."""
    started = time.perf_counter()
    try:
        return _execute(case, **options)
    except Exception:
        row = _base_row(case)
        row.update(status='error', accounting_unknown=True,
                   elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                   error={'type': 'EvaluationExecutionError',
                          'message': 'The case audit could not finish. Raw errors are excluded.'})
        score_case(case, row, None, [], [])
        return _redact(row, options['secrets'])


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = int(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(rows, *, live):
    total = len(rows)
    summary = {'scheduled_cases': total, 'attempted_cases': sum(row['attempted'] for row in rows),
        'completed_handoffs': sum(row['handoff_observed'] for row in rows),
        'failed_cases': sum(bool(row['error']) for row in rows),
        'not_run_cases': sum(row['status'] == 'not_run' for row in rows),
        'handoff_rate': sum(row['handoff_observed'] for row in rows) / total if live and total else None,
        'contract_pass_rate': sum(bool(row['scores']['contract_passed']) for row in rows) / total if live and total else None,
        'citation_integrity_passed_cases': sum(row['automatic_citation_integrity']['status'] == 'passed' for row in rows),
        'citation_integrity_applicable_cases': sum(row['automatic_citation_integrity']['status'] != 'not_applicable' for row in rows),
        'human_accuracy': None, 'human_citation_entailment': None,
        'human_rubric_pending_cases': total,
        'accounting_unknown_error_rows': sum(row.get('accounting_unknown', False) for row in rows)}
    checked = sum(row['scores']['anchors_checked'] or 0 for row in rows)
    covered = sum(row['scores']['anchors_covered'] or 0 for row in rows)
    summary.update(anchors_checked=checked if live else None, anchors_covered=covered if live else None,
                   quoted_anchor_recall=covered / checked if live and checked else None)
    latencies = [row['elapsed_ms'] / 1000 for row in rows if row['elapsed_ms'] is not None]
    summary['latency_p95_seconds'] = _percentile(latencies, .95) if live and len(latencies) == total else None
    summary['latency_measured_cases'] = len(latencies)
    for field in ('model_calls', 'tool_calls', 'unfinished_call_records'):
        summary[field] = sum(row['accounting'][field] for row in rows)
    for field in ('model_calls_by_role', 'tool_calls_by_name', 'errors_by_role_or_tool',
                  'reported_token_usage', 'model_calls_missing_usage_by_field'):
        counts = Counter()
        for row in rows:
            counts.update(row['accounting'][field])
        summary[field] = dict(counts)
    summary['reported_tokens'] = summary['reported_token_usage'].get('total_tokens', 0)
    summary['usage_complete'] = (all(row['accounting']['usage_complete'] for row in rows)
        and not summary['accounting_unknown_error_rows']) if live else None
    summary['mean_reported_tokens'] = (summary['reported_tokens'] / total
        if live and total and summary['usage_complete'] else None)
    summary['targets'] = {}
    for field, specification in TARGETS.items():
        value = summary[field]
        measured = live and value is not None
        if field == 'mean_reported_tokens' and not summary['usage_complete']:
            measured = False
        if field == 'latency_p95_seconds' and len(latencies) != total:
            measured = False
        passed = ((value >= specification['target'] if specification['operator'] == '>='
                   else value <= specification['target']) if measured else None)
        summary['targets'][field] = {**specification, 'measured': value, 'passed': passed}
    return summary


def _manifest_identity(root, dataset_hash, cases, *, settings, live, workers, reviewer_prompt, injected):
    from triage_desk.agent_review import review_prompt
    from triage_desk.deep_investigation import ORCHESTRATOR_PROMPT
    profiles = {profile.id: {'version': profile.version, 'sha256': _sha(_canonical(asdict(profile)))}
                for profile in USE_CASES}
    return {'dataset_sha256': dataset_hash, 'case_ids': [case['id'] for case in cases],
        'input_projection_sha256': _sha(_canonical([input_projection(case) for case in cases])),
        'case_definition_sha256': {case['id']: _sha(_canonical(case)) for case in cases},
        'live': live, 'workers': workers,
        'model': (settings.model or ('injected-test-service' if injected else 'not_configured')) if live else 'not_run',
        'budgets': {key: getattr(settings, key) for key in ('max_model_calls', 'max_tool_calls', 'max_tokens', 'max_seconds')},
        'reviewer_prompt_version': reviewer_prompt, 'reviewer_prompt_sha256': _sha(review_prompt(reviewer_prompt).encode()),
        'orchestrator_prompt_sha256': _sha(ORCHESTRATOR_PROMPT.encode()), 'profiles': profiles,
        'code_sha256': {name: _sha((ROOT / name).read_bytes()) for name in _CODE_FILES},
        'corpus_sha256': _sha((root / 'data' / 'sources.json').read_bytes()),
        'repository_context': False, 'memory_context': False,
        'web_search_configured': bool(settings.keenable_api_key) if live else False,
        'github_authenticated': bool(settings.github_token) if live else False}


MEASUREMENT_LIMITS = [
    'All selected cases, including provider errors and unstarted cases, remain in rate denominators.',
    'Contract checks measure saved workflow identity, handoff, actual AI review, literal citations, count floors, and successful required tool calls; they are not semantic accuracy.',
    'Quoted anchor recall measures exact fixture substrings within valid registered support quotes. It is a provisional coverage proxy, not entailment or factual correctness.',
    'AI evidence review uses a separate model call and the stated prompt, not an independent human. Human rubrics and labels remain pending.',
    'No evaluation approves, publishes, merges, executes supplied code, or resumes a human decision. Reports remain awaiting human review.',
    'Token totals use observed provider usage. Incomplete usage cannot pass the token target; costs and missing tokens are never inferred.',
    'Latency includes service initialization and the complete case. Small samples and concurrent runs do not establish production latency guarantees.',
    'Remote tracing is disabled. Exports contain public bundled fixture text, generated reports, bounded source provenance and safe audit event fields; provider error bodies, raw tool arguments and unrelated full source bodies are excluded.',
]


def render_report(report):
    esc = lambda value: html.escape(str(value), quote=True)
    pretty = lambda value: esc(json.dumps(value, ensure_ascii=False, indent=2))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Engineering workflow evaluation</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.5;color:#172033}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px;border-radius:8px}article{border-top:1px solid #ccd3df;margin-top:28px}h1,h2{line-height:1.2}</style>',
        '<h1>Engineering workflow evaluation</h1>',
        '<p>Mode: ' + esc(report['execution_mode']) + '. Labels await human review. Automatic checks do not establish semantic accuracy.</p>',
        '<h2>Observed results</h2><pre>' + pretty(report['summary']) + '</pre>',
        '<h2>By use case</h2><pre>' + pretty(report['by_use_case']) + '</pre>',
        '<h2>Measurement limits</h2><ul>']
    parts.extend('<li>' + esc(note) + '</li>' for note in report['measurement_limits'])
    parts.append('</ul>')
    for row in report['results']:
        parts.extend(['<article><h2>' + esc(row['id']) + ' · ' + esc(row['title']) + '</h2>',
                      '<p>' + esc(row['status']) + ' · ' + esc(row['use_case']) + '</p>',
                      '<p>' + esc(row['question']) + '</p>',
                      '<h3>Checks</h3><pre>' + pretty(row['scores']) + '</pre>',
                      '<h3>Saved report</h3><pre>' + pretty(row['report']) + '</pre>',
                      '<details><summary>Fixture, expectations and local audit trace</summary><pre>'
                      + pretty({key: row.get(key) for key in ('input', 'fixture', 'expected', 'label_review', 'sources',
                          'independent_ai_review', 'accounting', 'events', 'human_review', 'error')}) + '</pre></details></article>'])
    return '\n'.join([*parts, '</html>'])


def _export(output, report, secrets):
    report = _redact(report, secrets)
    _write_json(output / 'report.json', report, secrets)
    (output / 'report.html').write_text(render_report(report), encoding='utf-8')
    fields = ('id', 'use_case', 'scenario', 'status', 'handoff_observed', 'elapsed_ms',
              'contract_passed', 'quoted_anchor_recall', 'reported_tokens', 'error')
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['results']:
            record = {key: row.get(key) for key in fields}
            record.update(contract_passed=row['scores']['contract_passed'],
                          quoted_anchor_recall=row['scores']['quoted_anchor_recall'],
                          reported_tokens=row['accounting']['reported_token_usage']['total_tokens'],
                          error=(row['error'] or {}).get('type', ''))
            # Keep spreadsheet programs from treating untrusted text as a formula.
            writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value
                             for key, value in record.items()})


def run_evaluation(root=ROOT, cases_path=DEFAULT_CASES, output_dir=None, *, live=False,
                   workers=1, case_ids=None, use_cases=None, resume=False,
                   service_factory=None, reviewer_prompt='baseline', progress=True):
    """Inventory or execute full workflows; the service factory is a test seam.

    A factory receives (workspace, source_root=..., settings=...,
    reviewer_prompt_version=...). It must return an independent service per case.
    """
    root = Path(root).resolve()
    if output_dir is None or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Choose an output directory and 1 to 4 workers.')
    if type(live) is not bool or type(resume) is not bool:
        raise ValueError('live and resume must be booleans.')
    dataset, cases, dataset_hash = load_cases(root, cases_path, case_ids, use_cases)
    output = Path(output_dir)
    output = output if output.is_absolute() else root / output
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new or empty; prior artifacts are never overwritten.')
    from triage_desk.agent_review import review_prompt
    from triage_desk.deep_investigation import AgentSettings
    review_prompt(reviewer_prompt)
    if live:
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
        settings = AgentSettings.from_env()
    else:
        settings = AgentSettings()
    secrets = _secrets(settings)
    identity = _manifest_identity(root, dataset_hash, cases, settings=settings, live=live,
        workers=workers, reviewer_prompt=reviewer_prompt, injected=service_factory is not None)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if resume:
        if not manifest_path.is_file():
            raise ValueError('Resume requires an existing manifest.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError('Resume refused: dataset, inputs, code, profiles, prompt, mode, model, budgets, corpus and workers must match.')
    else:
        manifest = {**identity, 'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8],
            'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_name': dataset['name'],
            'dataset_version': dataset['version'], 'label_status': dataset['label_status'],
            'source_policy': dataset.get('source_policy'),
            'scope': 'Complete Deep Investigation graph, independent evidence review and persisted human handoff.',
            'tracing': 'Local root and child audit events only; remote tracing disabled.'}
        _write_json(manifest_path, manifest, secrets)
        _write_json(output / 'dataset.json', dataset, secrets)
    rows = {}
    for case in cases:
        path = output / 'cases' / f'{case["id"]}.json'
        if resume and path.exists():
            row = json.loads(path.read_text(encoding='utf-8'))
            if (row.get('id') != case['id'] or row.get('use_case') != case['use_case']
                    or row.get('input_sha256') != _sha(_canonical(input_projection(case)))
                    or row.get('case_definition_sha256') != _sha(_canonical(case))):
                raise ValueError('Saved case identity does not match the immutable manifest.')
            rows[case['id']] = row
    pending = [case for case in cases if case['id'] not in rows]

    def persist(case, row):
        rows[case['id']] = row
        _write_json(output / 'cases' / f'{case["id"]}.json', row, secrets)
        if progress:
            result = f'contract={row["scores"]["contract_passed"]}' if live else 'unscored inventory'
            print(f'{case["id"]} {row["use_case"]} {row["status"]} {result}', flush=True)

    if live:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_safe_execute, case, root=root, run_id=manifest['run_id'], settings=settings,
                service_factory=service_factory, reviewer_prompt=reviewer_prompt, secrets=secrets): case for case in pending}
            for future in as_completed(futures):
                persist(futures[future], future.result())
    else:
        for case in pending:
            persist(case, _base_row(case))
    ordered = [rows[case['id']] for case in cases]
    report = _redact({'schema_version': 1, 'report_kind': 'use_case_workflow',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'execution_mode': 'live' if live else 'validation_only',
        'dataset_name': dataset['name'], 'dataset_sha256': dataset_hash, 'manifest': manifest,
        'summary': summarize(ordered, live=live),
        'by_use_case': {identity: summarize([row for row in ordered if row['use_case'] == identity], live=live)
                        for identity in sorted({case['use_case'] for case in cases})},
        'results': ordered, 'measurement_limits': MEASUREMENT_LIMITS}, secrets)
    _export(output, report, secrets)
    return report


def compare_runs(baseline_dir, candidate_dir, output_dir):
    def read(path):
        path = Path(path)
        return json.loads((path / 'report.json' if path.is_dir() else path).read_text(encoding='utf-8'))
    baseline, candidate = read(baseline_dir), read(candidate_dir)
    if any(run.get('report_kind') != 'use_case_workflow' or run.get('execution_mode') != 'live'
           for run in (baseline, candidate)):
        raise ValueError('Comparison requires two live engineering workflow reports.')
    left, right = {row['id']: row for row in baseline['results']}, {row['id']: row for row in candidate['results']}
    if (baseline['dataset_sha256'] != candidate['dataset_sha256'] or set(left) != set(right)
            or len(left) != len(baseline['results']) or len(right) != len(candidate['results'])):
        raise ValueError('Paired comparison requires identical dataset hashes and matched unique case IDs.')
    for identity in left:
        if left[identity]['input_sha256'] != right[identity]['input_sha256']:
            raise ValueError('Paired comparison requires identical caller inputs.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Comparison output directory must be new or empty.')
    pairs = [{'id': identity, 'use_case': left[identity]['use_case'],
              'baseline_contract_passed': left[identity]['scores']['contract_passed'],
              'candidate_contract_passed': right[identity]['scores']['contract_passed'],
              'baseline_quoted_anchor_recall': left[identity]['scores']['quoted_anchor_recall'],
              'candidate_quoted_anchor_recall': right[identity]['scores']['quoted_anchor_recall']}
             for identity in sorted(left)]
    deltas = {key: candidate['summary'][key] - baseline['summary'][key]
              if candidate['summary'][key] is not None and baseline['summary'][key] is not None else None
              for key in TARGETS}
    report = {'schema_version': 1, 'report_kind': 'use_case_comparison',
        'dataset_sha256': baseline['dataset_sha256'], 'baseline': baseline['manifest'],
        'candidate': candidate['manifest'], 'metric_deltas': deltas, 'pairs': pairs,
        'baseline_summary': baseline['summary'], 'candidate_summary': candidate['summary'],
        'baseline_by_use_case': baseline['by_use_case'], 'candidate_by_use_case': candidate['by_use_case'],
        'measurement_limits': ['Matched diagnostic cases with provisional labels; improvement does not establish human-validated accuracy.']}
    report = _redact(report, _secrets())
    _write_json(output / 'comparison.json', report, _secrets())
    # The app discovers standard report.json exports; retain the descriptive
    # comparison filename for CLI consumers and make both views identical.
    _write_json(output / 'report.json', report, _secrets())
    page = render_comparison(report)
    (output / 'comparison.html').write_text(page, encoding='utf-8')
    (output / 'report.html').write_text(page, encoding='utf-8')
    return report


def render_comparison(report):
    """Readable paired measurements with the same limits as the JSON export."""
    esc = lambda value: html.escape(str(value), quote=True)
    pretty = lambda value: esc(json.dumps(value, ensure_ascii=False, indent=2))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Engineering workflow comparison</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.5;color:#172033}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px;border-radius:8px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd3df;padding:8px;text-align:left}</style>',
        '<h1>Engineering workflow comparison</h1>',
        '<p>Matched diagnostic cases with provisional labels. Contract changes and quoted anchor coverage do not establish human-validated accuracy.</p>',
        '<h2>Metric changes</h2><table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>']
    for metric in TARGETS:
        parts.append('<tr>' + ''.join('<td>' + esc(value if value is not None else 'Not measured') + '</td>'
            for value in (metric, report['baseline_summary'].get(metric), report['candidate_summary'].get(metric),
                          report['metric_deltas'].get(metric))) + '</tr>')
    parts.extend(['</table>', '<h2>Results by use case</h2><pre>' + pretty({
        'baseline': report['baseline_by_use_case'], 'candidate': report['candidate_by_use_case']}) + '</pre>',
        '<h2>Paired cases</h2><pre>' + pretty(report['pairs']) + '</pre>',
        '<h2>Run identities and measurement limits</h2><pre>' + pretty({
        key: report[key] for key in ('baseline', 'candidate', 'dataset_sha256', 'measurement_limits')}) + '</pre>', '</html>'])
    return '\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Validate fixtures or explicitly execute full workflows.')
    run.add_argument('--root', type=Path, default=ROOT)
    run.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--live', action='store_true')
    run.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    run.add_argument('--case', action='append', dest='case_ids')
    run.add_argument('--use-case', action='append', choices=sorted(PROFILE_IDS), dest='use_cases')
    run.add_argument('--resume', action='store_true')
    run.add_argument('--review-prompt', choices=('baseline', 'calibrated'), default='baseline')
    compare = sub.add_parser('compare', help='Compare matched live cases.')
    compare.add_argument('--baseline-dir', type=Path, required=True)
    compare.add_argument('--candidate-dir', type=Path, required=True)
    compare.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            report = run_evaluation(args.root, args.cases, args.output_dir, live=args.live, workers=args.workers,
                case_ids=args.case_ids, use_cases=args.use_cases, resume=args.resume, reviewer_prompt=args.review_prompt)
            print(json.dumps({'execution_mode': report['execution_mode'], 'summary': report['summary'],
                              'output_directory': str(args.output_dir)}, indent=2))
            return int(bool(report['summary']['failed_cases']))
        report = compare_runs(args.baseline_dir, args.candidate_dir, args.output_dir)
        print(json.dumps({'metric_deltas': report['metric_deltas'], 'output_directory': str(args.output_dir)}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({'status': 'error', 'error': 'Invalid workflow evaluation input or output. Raw errors are excluded.'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
