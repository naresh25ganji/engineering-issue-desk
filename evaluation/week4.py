"""Frozen, label-separated evaluation of the production evidence reviewer.

The default command is an offline input inventory. Only --live invokes a model.
Labels are provisional unless a human has reviewed them; agreement with these
labels must never be presented as human-validated accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from uuid import uuid4

from evaluation.agent_workflow import _redact, event_accounting

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path('evaluation/week4-cases.json')
_ID = re.compile(r'W4-\d{2}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_PROVENANCE = ('kind', 'origin', 'evidence_type', 'title', 'version_scope', 'snapshot_at',
               'fetched_at', 'published_at', 'acquired_at', 'commit_sha', 'base_sha', 'filename', 'truncated')
_THRESHOLDS = {'case_exact_match': .9, 'unsupported_rejection': .95,
               'supported_retention': .9, 'p95_latency_seconds': 30, 'mean_total_tokens': 4000}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_write(path: Path, payload, secrets: set[str]) -> None:
    """Atomic replacements make every completed case recoverable after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_redact(payload, secrets), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _secrets() -> set[str]:
    return {value for name, value in os.environ.items()
            if re.search(r'(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)$', name, re.I)
            and 8 <= len(value) <= 4096}


def _text(value, label, maximum=16000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{label} must be nonempty text of at most {maximum} characters.')
    return value


def load_cases(root: Path, path: Path, case_ids=None, *, verify_sources=True):
    """Validate the complete dataset before selecting cases; never refresh quotes."""
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root / path
    raw = path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset.get('cases') if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not 30 <= len(cases) <= 50:
        raise ValueError('The frozen Week 4 dataset must contain 30 to 50 cases.')
    for field in ('name', 'version', 'label_status', 'source_policy'):
        if not dataset.get(field):
            raise ValueError(f'Dataset requires {field}.')
    selected = set(case_ids) if case_ids else None
    seen, prepared, source_cache = set(), [], {}
    for case in cases:
        if not isinstance(case, dict) or not _ID.fullmatch(str(case.get('id', ''))):
            raise ValueError('Each case needs a W4-NN ID.')
        case_id = case['id']
        if case_id in seen:
            raise ValueError('Case IDs must be unique.')
        seen.add(case_id)
        for field in ('scenario', 'title', 'failure_mode'):
            _text(case.get(field), f'{case_id} {field}', 2000)
        context = case.get('context')
        if not isinstance(context, dict):
            raise ValueError(f'{case_id} requires context.')
        _text(context.get('question'), f'{case_id} question', 4500)
        if not isinstance(context.get('reported_version', ''), str) or len(context.get('reported_version', '')) > 120:
            raise ValueError(f'{case_id} reported_version must be bounded text.')
        for kind, field, limit in (('findings', 'statement', 12), ('next_steps', 'proposal', 8)):
            items = case.get(kind)
            if not isinstance(items, list) or len(items) > limit or (kind == 'findings' and not items):
                raise ValueError(f'{case_id} requires 1-12 findings and 0-8 next steps.')
            for item in items:
                if not isinstance(item, dict) or type(item.get('expected_supported')) is not bool:
                    raise ValueError(f'{case_id} expected_supported must be a strict boolean.')
                _text(item.get(field), f'{case_id} {field}', 4500)
                _text(item.get('rationale'), f'{case_id} rationale', 4500)
        review = case.get('label_review')
        if not isinstance(review, dict) or not review.get('status'):
            raise ValueError(f'{case_id} requires explicit label_review status.')
        source = case.get('source')
        if not isinstance(source, dict):
            raise ValueError(f'{case_id} requires a frozen source.')
        relative = _text(source.get('path'), f'{case_id} source path', 1000)
        fixture = (root / relative).resolve()
        if Path(relative).is_absolute() or not fixture.is_relative_to(root):
            raise ValueError('Frozen sources must remain inside the repository.')
        start, end = source.get('start_line'), source.get('end_line')
        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
            raise ValueError(f'{case_id} needs valid inclusive source line bounds.')
        if not isinstance(source.get('sha256'), str) or not _SHA.fullmatch(source['sha256']):
            raise ValueError(f'{case_id} source sha256 must pin the entire file.')
        quote = _text(source.get('quote'), f'{case_id} frozen quote', 60000)
        if verify_sources:
            if relative not in source_cache:
                source_cache[relative] = fixture.read_bytes()
            content = source_cache[relative]
            if _sha(content) != source['sha256']:
                raise ValueError(f'{case_id} source pin changed: {relative}; frozen quotes were not refreshed.')
            lines = content.decode('utf-8').splitlines(keepends=True)
            if end > len(lines):
                raise ValueError(f'{case_id} source line range no longer exists.')
            excerpt = ''.join(lines[start - 1:end])
            if quote not in (excerpt, excerpt.rstrip('\r\n')):
                raise ValueError(f'{case_id} frozen quote does not match its pinned line range.')
        if selected is None or case_id in selected:
            prepared.append(case)
    if selected is not None and (not selected <= seen or not prepared):
        raise ValueError('Every --case ID must exist in the dataset.')
    return dataset, prepared, _sha(raw)


def hydrate_report(case: dict) -> dict:
    """Project only production inputs; annotations never enter the reviewer call."""
    source = case['source']
    evidence_id, quote_id = f"{case['id']}-source", f"{case['id']}-quote"
    provenance = {field: source[field] for field in _PROVENANCE if field in source}
    provenance.setdefault('title', source['path'])
    provenance.setdefault('kind', 'code' if source['path'].endswith('.py') else 'documentation')
    provenance.setdefault('origin', 'snapshot')
    provenance.setdefault('filename', source['path'])
    # A local file snapshot establishes what was saved, not official authority.
    provenance['evidence_id'] = evidence_id
    return {'findings': [{'statement': item['statement'], 'support': [
                {'evidence_id': evidence_id, 'quote_id': quote_id, 'quote': source['quote']}],
                } for item in case['findings']],
            'next_steps': [item['proposal'] for item in case['next_steps']],
            'missing_information': [], 'sources': [provenance]}


def _expected(case):
    return {kind: [item['expected_supported'] for item in case[kind]]
            for kind in ('findings', 'next_steps')}


def score_case(case, quality_review=None, *, completed=False):
    """Unknown/invalid verdicts stay null and fail their scheduled denominators."""
    expected, predicted = _expected(case), {}
    valid = bool(completed)
    for kind, verdict_name in (('findings', 'finding_verdicts'), ('next_steps', 'next_step_verdicts')):
        verdicts = (quality_review or {}).get(verdict_name, [])
        count = len(expected[kind])
        correct_shape = (isinstance(verdicts, list) and len(verdicts) == count
                         and all(isinstance(v, dict) and type(v.get('index')) is int
                                 and type(v.get('supported')) is bool for v in verdicts)
                         and {v['index'] for v in verdicts} == set(range(count)))
        valid = valid and correct_shape
        by_index = {v['index']: v['supported'] for v in verdicts} if correct_shape and completed else {}
        predicted[kind] = [by_index.get(index) for index in range(count)]
    items = []
    for kind in expected:
        for index, (reference, prediction) in enumerate(zip(expected[kind], predicted[kind])):
            items.append({'kind': kind, 'index': index, 'expected': reference, 'predicted': prediction,
                          'correct': prediction is not None and prediction == reference,
                          'error_type': 'unknown' if prediction is None else
                          'false_positive' if prediction and not reference else
                          'false_negative' if reference and not prediction else None})
    return {'valid': valid, 'exact_match': valid and all(item['correct'] for item in items),
            'expected': expected, 'predicted': predicted, 'items': items,
            'false_positives': sum(item['error_type'] == 'false_positive' for item in items),
            'false_negatives': sum(item['error_type'] == 'false_negative' for item in items),
            'unknown_verdicts': sum(item['error_type'] == 'unknown' for item in items)}


def _threshold(name, value, live):
    passed = None if value is None else (value >= _THRESHOLDS[name] if name in ('case_exact_match', 'unsupported_rejection', 'supported_retention') else value <= _THRESHOLDS[name])
    return {'value': value if live else None, 'threshold': _THRESHOLDS[name],
            'status': 'not_measured' if not live or value is None else 'passed' if passed else 'failed'}


def summarize(rows, *, live):
    count = len(rows)
    items = [item for row in rows for item in row['scores']['items']]
    supported = [item for item in items if item['expected']]
    unsupported = [item for item in items if not item['expected']]
    latencies = sorted(row['latency_seconds'] for row in rows if row.get('latency_seconds') is not None)
    p95 = latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if live and len(latencies) == count and count else None
    token_values = [row['accounting']['reported_token_usage']['total_tokens'] for row in rows
                    if row['accounting']['usage_complete'] and row['accounting']['model_calls']]
    mean_tokens = statistics.mean(token_values) if live and len(token_values) == count and count else None
    values = {'case_exact_match': sum(row['scores']['exact_match'] for row in rows) / count if count and live else None,
              'unsupported_rejection': sum(item['predicted'] is False for item in unsupported) / len(unsupported) if unsupported and live else None,
              'supported_retention': sum(item['predicted'] is True for item in supported) / len(supported) if supported and live else None,
              'p95_latency_seconds': p95, 'mean_total_tokens': mean_tokens}
    metrics = {name: _threshold(name, value, live) for name, value in values.items()}
    costs = [row.get('estimated_cost_usd') for row in rows]
    return {'scheduled_cases': count, 'completed_cases': sum(row['status'] == 'completed' for row in rows),
            'error_cases': sum(row['status'] == 'error' for row in rows),
            'not_run_cases': sum(row['status'] == 'not_run' for row in rows),
            'valid_cases': sum(row['scores']['valid'] for row in rows) if live else None,
            'completion_rate': sum(row['status'] == 'completed' for row in rows) / count if count and live else None,
            'false_positives': sum(row['scores']['false_positives'] for row in rows) if live else None,
            'false_negatives': sum(row['scores']['false_negatives'] for row in rows) if live else None,
            'unknown_verdicts': sum(row['scores']['unknown_verdicts'] for row in rows) if live else None,
            'supported_item_denominator': len(supported), 'unsupported_item_denominator': len(unsupported),
            'metrics': metrics, 'all_pass_bars_met': all(metric['status'] == 'passed' for metric in metrics.values()) if live else None,
            'estimated_cost_usd': sum(costs) if live and costs and all(value is not None for value in costs) else None,
            'denominator_policy': 'All scheduled cases/items, including errors and unknown verdicts. Unknown verdicts fail correctness; they are never false.',
            'measurement_note': 'Agreement with dataset labels, not human-validated accuracy. Latency includes one case setup/review. Tokens are observed provider usage; missing usage is unknown.'}


def _prompt(version):
    from triage_desk.agent_review import review_prompt
    return review_prompt(version)


def _base_row(case):
    return {'id': case['id'], 'scenario': case['scenario'], 'title': case['title'],
            'failure_mode': case['failure_mode'], 'source': case['source'],
            'context': case['context'], 'label_review': case['label_review'],
            'annotations': {kind: case[kind] for kind in ('findings', 'next_steps')},
            'status': 'not_run', 'error': None, 'latency_seconds': None,
            'quality_review': None, 'filtered_report': None, 'events': [],
            'accounting': event_accounting([]), 'estimated_cost_usd': None,
            'scores': score_case(case)}


def _execute(case, *, root, run_id, variant, prompt_hash, model_factory, reviewer, secrets, rates):
    from triage_desk.agent_budget import InvestigationBudget
    from triage_desk.cases import CaseStore
    row, started, store, case_id = _base_row(case), time.perf_counter(), None, None
    try:
        store = CaseStore(root / '.cache' / 'week4' / run_id / case['id'] / 'cases.sqlite')
        case_id = store.create_case(case['context']['question'], case['context'].get('reported_version', ''))['case_id']
        row['case_id'] = case_id
        budget = InvestigationBudget(store, case_id, max_model_calls=1, max_tool_calls=0,
                                     max_tokens=10000, max_seconds=90)
        model = model_factory()
        row['model'] = getattr(model, 'model_name', None) or getattr(model, 'model', None) or getattr(model, '_llm_type', 'unknown')
        budget.event(phase='evaluation', name='week4_case', status='started', eval_id=case['id'],
                     run_name=run_id, prompt_version=variant, prompt_hash=prompt_hash, model=row['model'])
        # LangSmith uses context variables, so every worker can suppress remote
        # tracing without mutating environment variables shared by other threads.
        try:
            from langsmith import tracing_context
            trace_context = tracing_context(enabled=False)
        except ImportError:
            trace_context = nullcontext()
        with trace_context:
            result = reviewer(hydrate_report(case), model, budget, context=dict(case['context']), prompt_version=variant)
        row.update(status='completed', quality_review=result['quality_review'], filtered_report=result['report'])
        row['scores'] = score_case(case, result['quality_review'], completed=True)
        if not row['scores']['valid']:
            row.update(status='error', error={'type': 'InvalidVerdictCoverage', 'code': 'invalid_verdicts',
                                            'message': 'Reviewer output does not cover every item with strict boolean verdicts.'})
    except Exception as exc:
        row.update(status='error', error={'type': type(exc).__name__, 'code': getattr(exc, 'code', 'execution_failed'),
                                        'message': 'Evidence reviewer execution failed; inspect local audit events and finite failure code.'})
    finally:
        row['latency_seconds'] = round(time.perf_counter() - started, 6)
        if store is not None and case_id is not None:
            store.add_event(case_id, {'phase': 'evaluation_result', 'name': 'week4_case', 'status': row['status'],
                                    'eval_id': case['id'], 'run_name': run_id, 'prompt_hash': prompt_hash,
                                    'prompt_version': variant, 'expected': row['scores']['expected'],
                                    'predicted': row['scores']['predicted'], 'scores': row['scores'],
                                    'latency_seconds': row['latency_seconds'], 'error': row['error']})
            row['events'] = store.events(case_id)
            row['accounting'] = event_accounting(row['events'])
        usage, missing = row['accounting']['reported_token_usage'], row['accounting']['model_calls_missing_usage_by_field']
        if (rates[0] is not None and rates[1] is not None and row['accounting']['model_calls']
                and not missing['input_tokens'] and not missing['output_tokens']):
            row['estimated_cost_usd'] = (usage['input_tokens'] * rates[0] + usage['output_tokens'] * rates[1]) / 1_000_000
    return _redact(row, secrets)


def _safe_cell(value):
    value = '' if value is None else str(value)
    # CSV quoting alone does not prevent spreadsheet formula execution.
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value


def _csv_write(path, headers, records, secrets):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in records:
            writer.writerow([_safe_cell(_redact(record.get(header), secrets)) for header in headers])


def _export(output, result, secrets):
    _json_write(output / 'report.json', result, secrets)
    records, labels = [], []
    for row in result['rows']:
        for item in row['scores']['items']:
            annotation = row['annotations'][item['kind']][item['index']]
            text = annotation.get('statement', annotation.get('proposal'))
            verdict_name = 'finding_verdicts' if item['kind'] == 'findings' else 'next_step_verdicts'
            verdicts = (row.get('quality_review') or {}).get(verdict_name, [])
            reason = next((v.get('reason') for v in verdicts if v.get('index') == item['index']), None)
            records.append({'case_id': row['id'], 'scenario': row['scenario'], 'failure_mode': row['failure_mode'],
                            'status': row['status'], **item, 'text': text, 'reason': reason,
                            'label_status': row['label_review']['status'], 'latency_seconds': row['latency_seconds']})
            labels.append({'case_id': row['id'], 'kind': item['kind'], 'index': item['index'], 'text': text,
                           'source_path': row['source']['path'], 'source_quote': row['source']['quote'],
                           'provisional_label': item['expected'], 'provisional_rationale': annotation['rationale'],
                           'human_label': '', 'human_reviewer': '', 'human_notes': '',
                           'review_status': row['label_review']['status']})
    headers = ['case_id', 'scenario', 'failure_mode', 'status', 'kind', 'index', 'text', 'expected', 'predicted',
               'correct', 'error_type', 'reason', 'label_status', 'latency_seconds']
    _csv_write(output / 'results.csv', headers, records, secrets)
    _csv_write(output / 'label-review.csv', ['case_id', 'kind', 'index', 'text', 'source_path', 'source_quote',
                'provisional_label', 'provisional_rationale', 'human_label', 'human_reviewer', 'human_notes', 'review_status'], labels, secrets)
    safe = _redact(result, secrets)
    metrics = ''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(metric["value"]))}</td>'
                      f'<td>{html.escape(str(metric["threshold"]))}</td><td>{metric["status"]}</td></tr>'
                      for name, metric in safe['summary']['metrics'].items())
    rows = ''.join(f'<tr><td>{html.escape(row["id"])}</td><td>{html.escape(row["scenario"])}</td>'
                   f'<td>{row["status"]}</td><td>{row["scores"]["exact_match"] if safe["manifest"]["live"] else "unscored"}</td>'
                   f'<td>{row["scores"]["false_positives"]}</td><td>{row["scores"]["false_negatives"]}</td>'
                   f'<td><details><summary>Review output and evidence</summary><pre>{html.escape(json.dumps(row, indent=2, ensure_ascii=False))}</pre></details></td></tr>'
                   for row in safe['rows'])
    page = '<!doctype html><meta charset="utf-8"><title>Week 4 reviewer evaluation</title><style>body{font:15px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:1000px}</style>'
    page += f'<h1>Week 4 evidence reviewer evaluation</h1><p>Mode: {"live" if safe["manifest"]["live"] else "offline inventory; no scores measured"}. Dataset labels: {html.escape(str(safe["manifest"]["label_status"]))}. Agreement with provisional labels is not human-validated accuracy.</p>'
    page += '<table><tr><th>Metric</th><th>Observed</th><th>Pass bar</th><th>Status</th></tr>' + metrics + '</table><p>All scheduled cases and items count, including errors. Missing verdicts remain unknown. Tokens and cost are reported only when observed/configured. Remote tracing is disabled for this spreadsheet workflow.</p>'
    page += '<table><tr><th>Case</th><th>Scenario</th><th>Status</th><th>Exact match</th><th>FP</th><th>FN</th><th>Evidence and trace</th></tr>' + rows + '</table>'
    (output / 'report.html').write_text(page, encoding='utf-8')


def run_evaluation(root=ROOT, cases_path=DEFAULT_CASES, output_dir=None, *, live=False, variant='baseline',
                   case_ids=None, workers=1, resume=False, model_factory=None, reviewer=None,
                   input_usd_per_million=None, output_usd_per_million=None, progress=True):
    """Run the real reviewer with isolated stores; injected models are for tests."""
    root = Path(root).resolve()
    if output_dir is None:
        raise ValueError('Choose an output directory explicitly.')
    output = Path(output_dir)
    output = output if output.is_absolute() else root / output
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be between 1 and 4.')
    rates = (input_usd_per_million, output_usd_per_million)
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0) for value in rates):
        raise ValueError('Token prices must be finite nonnegative numbers.')
    dataset, cases, dataset_hash = load_cases(root, cases_path, case_ids)
    prompt = _prompt(variant)
    prompt_hash = _sha(prompt.encode('utf-8'))
    settings, model_name = None, 'not_run'
    if live:
        from dotenv import load_dotenv
        from triage_desk.deep_investigation import AgentSettings, fireworks_model
        load_dotenv(root / '.env', override=False)
        settings = AgentSettings.from_env()
        model_name = settings.model or 'injected-test-model' if model_factory else settings.model
        if model_factory is None:
            if not settings.readiness()['ready']:
                raise ValueError('Live evaluation requires FIREWORKS_API_KEY and FIREWORKS_MODEL.')
            model_factory = lambda: fireworks_model(settings)
        if reviewer is None:
            from triage_desk.agent_review import review_report
            reviewer = review_report
    secrets = _secrets()
    identity = {'dataset_sha256': dataset_hash, 'prompt_sha256': prompt_hash,
                'prompt_version': variant, 'case_ids': [case['id'] for case in cases],
                'live': live, 'model': model_name, 'input_usd_per_million': rates[0], 'output_usd_per_million': rates[1]}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if resume:
        if not manifest_path.exists():
            raise ValueError('--resume requires an existing manifest.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError('Resume refused: dataset, prompt, cases, model, mode and rates must match the original run.')
    else:
        if any(output.iterdir()):
            raise ValueError('Output directory must be new or empty; use --resume only for the identical run.')
        manifest = {**identity, 'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8],
                    'created_at': datetime.now(timezone.utc).isoformat(), 'dataset_name': dataset['name'],
                    'dataset_version': dataset['version'], 'label_status': dataset['label_status'],
                    'source_policy': dataset['source_policy'], 'scope': 'Production evidence reviewer only; no retrieval or full investigation.',
                    'tracing': 'Local root + LLM child events; remote LangSmith tracing disabled.'}
        _json_write(manifest_path, manifest, secrets)
        (output / 'prompt.txt').write_text(_redact(prompt, secrets), encoding='utf-8')
        _json_write(output / 'dataset.json', dataset, secrets)
    rows = {}
    for case in cases:
        case_path = output / 'cases' / f'{case["id"]}.json'
        if resume and case_path.exists():
            row = json.loads(case_path.read_text(encoding='utf-8'))
            if row.get('id') != case['id']:
                raise ValueError('Saved case identity does not match the manifest.')
            rows[case['id']] = row
    pending = [case for case in cases if case['id'] not in rows]
    def persist(case, row):
        rows[case['id']] = row
        _json_write(output / 'cases' / f'{case["id"]}.json', row, secrets)
        if progress:
            score = f'exact={row["scores"]["exact_match"]} FP={row["scores"]["false_positives"]} FN={row["scores"]["false_negatives"]}' if live else 'unscored inventory'
            print(f'{case["id"]} {row["status"]} {score}', flush=True)
    if live:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute, case, root=root, run_id=manifest['run_id'], variant=variant,
                        prompt_hash=prompt_hash, model_factory=model_factory, reviewer=reviewer, secrets=secrets, rates=rates): case for case in pending}
            for future in as_completed(futures):
                persist(futures[future], future.result())
    else:
        for case in pending:
            row = _base_row(case)
            row['scores'].update(valid=None, exact_match=None, false_positives=None, false_negatives=None, unknown_verdicts=None)
            for item in row['scores']['items']:
                item['correct'] = None
            persist(case, row)
    ordered = [rows[case['id']] for case in cases]
    result = {'report_kind': 'week4_evidence_review', 'manifest': manifest, 'summary': summarize(ordered, live=live), 'rows': ordered}
    _export(output, result, secrets)
    return result


def _load_run(path):
    path = Path(path)
    return json.loads((path / 'report.json' if path.is_dir() else path).read_text(encoding='utf-8'))


def compare_runs(baseline_dir, improved_dir, output_dir, *, review_minutes_per_failure=5):
    """Compare identical frozen inputs and expose regressions, not only averages."""
    baseline, improved = _load_run(baseline_dir), _load_run(improved_dir)
    if not baseline['manifest']['live'] or not improved['manifest']['live']:
        raise ValueError('Comparison requires two live runs; inventories have no scores.')
    if baseline['manifest']['model'] != improved['manifest']['model']:
        raise ValueError('Prompt comparison requires the same model in both runs.')
    if baseline['manifest']['dataset_sha256'] != improved['manifest']['dataset_sha256']:
        raise ValueError('Comparison requires the same frozen dataset hash.')
    before, after = {row['id']: row for row in baseline['rows']}, {row['id']: row for row in improved['rows']}
    if set(before) != set(after):
        raise ValueError('Comparison requires exactly the same case IDs.')
    if (isinstance(review_minutes_per_failure, bool) or not isinstance(review_minutes_per_failure, (int, float))
            or not math.isfinite(review_minutes_per_failure) or review_minutes_per_failure < 0):
        raise ValueError('Human review time assumption must be finite and nonnegative.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Comparison output directory must be new or empty.')
    deltas, cluster_map = [], {}
    for case_id, old in before.items():
        new = after[case_id]
        old_pass, new_pass = old['scores']['exact_match'], new['scores']['exact_match']
        change = 'fixed' if not old_pass and new_pass else 'regression' if old_pass and not new_pass else 'unchanged_pass' if new_pass else 'unchanged_failure'
        delta = {'id': case_id, 'scenario': old['scenario'], 'failure_mode': old['failure_mode'],
                 'change': change, 'baseline_pass': old_pass, 'improved_pass': new_pass,
                 'baseline_error': old['error'], 'improved_error': new['error'],
                 'baseline_false_positives': old['scores']['false_positives'], 'improved_false_positives': new['scores']['false_positives'],
                 'baseline_false_negatives': old['scores']['false_negatives'], 'improved_false_negatives': new['scores']['false_negatives'],
                 'baseline_unknown': old['scores']['unknown_verdicts'], 'improved_unknown': new['scores']['unknown_verdicts']}
        deltas.append(delta)
        for run, row in (('baseline', old), ('improved', new)):
            key = old['failure_mode']
            cluster = cluster_map.setdefault(key, {'failure_mode': key, 'baseline_failed_cases': 0, 'improved_failed_cases': 0,
                          'baseline_false_positives': 0, 'improved_false_positives': 0,
                          'baseline_false_negatives': 0, 'improved_false_negatives': 0,
                          'baseline_error_cases': 0, 'improved_error_cases': 0})
            cluster[f'{run}_failed_cases'] += not row['scores']['exact_match']
            cluster[f'{run}_false_positives'] += row['scores']['false_positives']
            cluster[f'{run}_false_negatives'] += row['scores']['false_negatives']
            cluster[f'{run}_error_cases'] += row['status'] == 'error'
    clusters = sorted(cluster_map.values(), key=lambda c: (-c['baseline_failed_cases'], c['failure_mode']))
    for cluster in clusters:
        for run in ('baseline', 'improved'):
            cluster[f'{run}_estimated_human_minutes'] = cluster[f'{run}_failed_cases'] * review_minutes_per_failure
    result = {'report_kind': 'week4_paired_comparison', 'dataset_sha256': baseline['manifest']['dataset_sha256'], 'label_status': baseline['manifest']['label_status'],
              'baseline': baseline['manifest'], 'improved': improved['manifest'],
              'summary': {change: sum(delta['change'] == change for delta in deltas)
                          for change in ('fixed', 'regression', 'unchanged_pass', 'unchanged_failure')},
              'metrics': {'baseline': baseline['summary'], 'improved': improved['summary']},
              'human_time_assumption': {'minutes_per_failed_case': review_minutes_per_failure,
                  'note': 'Rough planning assumption, not observed labor or verified business savings. Each failed case counted once.'},
              'clusters': clusters, 'cases': deltas,
              'limitation': 'Paired development-set agreement with provisional labels; not held-out or human-validated accuracy.'}
    secrets = _secrets()
    result = _redact(result, secrets)
    _json_write(output / 'comparison.json', result, secrets)
    _json_write(output / 'report.json', result, secrets)
    _csv_write(output / 'comparison.csv', list(deltas[0]) if deltas else ['id'], deltas, secrets)
    metric_rows = []
    for name, previous in baseline['summary']['metrics'].items():
        current = improved['summary']['metrics'][name]
        metric_rows.append({'metric': name, 'baseline_value': previous['value'], 'improved_value': current['value'],
                            'delta': current['value'] - previous['value'] if current['value'] is not None and previous['value'] is not None else None,
                            'threshold': current['threshold'], 'baseline_status': previous['status'], 'improved_status': current['status']})
    _csv_write(output / 'metrics.csv', ['metric', 'baseline_value', 'improved_value', 'delta', 'threshold', 'baseline_status', 'improved_status'], metric_rows, secrets)
    _csv_write(output / 'clusters.csv', list(clusters[0]) if clusters else ['failure_mode'], clusters, secrets)
    content = '<!doctype html><meta charset="utf-8"><title>Week 4 paired comparison</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap}</style><h1>Week 4 paired comparison</h1>'
    content += f'<p>{html.escape(result["limitation"])}</p><pre>{html.escape(json.dumps(result["summary"], indent=2))}</pre>'
    content += '<table><tr><th>Case</th><th>Failure mode</th><th>Change</th><th>Baseline FP / FN</th><th>Improved FP / FN</th></tr>'
    for row in result['cases']:
        content += f'<tr><td>{html.escape(row["id"])}</td><td>{html.escape(row["failure_mode"])}</td><td>{row["change"]}</td><td>{row["baseline_false_positives"]} / {row["baseline_false_negatives"]}</td><td>{row["improved_false_positives"]} / {row["improved_false_negatives"]}</td></tr>'
    content += '</table><h2>Failure clusters and explicit human time assumption</h2><pre>' + html.escape(json.dumps({'assumption': result['human_time_assumption'], 'clusters': result['clusters'], 'metrics': result['metrics']}, indent=2)) + '</pre>'
    (output / 'comparison.html').write_text(content, encoding='utf-8')
    (output / 'report.html').write_text(content, encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Offline inventory unless --live is explicit.')
    run.add_argument('--cases', dest='cases_path', type=Path, default=DEFAULT_CASES)
    run.add_argument('--output-dir', '--output', dest='output_dir', type=Path, required=True)
    run.add_argument('--live', action='store_true')
    run.add_argument('--variant', default='baseline')
    run.add_argument('--case', action='append', dest='case_ids')
    run.add_argument('--workers', type=int, default=1)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--input-usd-per-million', type=float)
    run.add_argument('--output-usd-per-million', type=float)
    compare = sub.add_parser('compare', help='Paired comparison of identical frozen datasets.')
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument('--improved', type=Path, required=True)
    compare.add_argument('--output-dir', '--output', dest='output_dir', type=Path, required=True)
    compare.add_argument('--review-minutes-per-failure', type=float, default=5)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command == 'run':
            result = run_evaluation(**args)
            print(json.dumps(result['summary'], indent=2))
        else:
            result = compare_runs(args.pop('baseline'), args.pop('improved'), args.pop('output_dir'), **args)
            print(json.dumps(result['summary'], indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {str(_redact(str(exc), _secrets()))}\n')


if __name__ == '__main__':
    main()
