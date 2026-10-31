"""Execute the report/citation acceptance contract without a model or network."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.cases import CaseStore

ROOT = Path(__file__).resolve().parents[1]
_SOURCE = 'The worker retries a failed request at most twice.\nThe timeout is set to 30 seconds.\n'
_SECOND = 'The second worker keeps its own isolated queue. Unicode label: δ.\n'
_DEFINITIONS = (
    ('valid_id', True, 'Application quote ID', 'valid'),
    ('valid_legacy', True, 'Exact legacy quote', 'valid'),
    ('valid_hydrated', True, 'ID and matching hydrated quote', 'valid'),
    ('valid_unicode', True, 'Exact Unicode quote', 'valid'),
    ('unrelated_claim', True, 'A real quote does not prove the claim', 'scope_limit'),
    ('empty_report', True, 'Empty findings remain schema-valid', 'scope_limit'),
    ('two_sources', True, 'Two distinct source citations', 'valid'),
    ('unknown_source', False, 'Unknown evidence identity', 'reference'),
    ('unknown_quote', False, 'Invented quote identity', 'reference'),
    ('wrong_source', False, 'Quote ID belongs to another source', 'reference'),
    ('mismatched_hydration', False, 'ID with rewritten quote text', 'reference'),
    ('changed_punctuation', False, 'Punctuation is not literal', 'reference'),
    ('blank_quote', False, 'Blank legacy quote', 'reference'),
    ('short_quote', False, 'Quote below minimum length', 'reference'),
    ('long_quote', False, 'Quote above maximum length', 'reference'),
    ('extra_support_field', False, 'Unknown support field', 'schema'),
    ('duplicate_support', False, 'Repeated citation in a finding', 'schema'),
    ('missing_support', False, 'Finding without support', 'schema'),
    ('extra_report_field', False, 'Unknown report field', 'schema'),
    ('extra_finding_field', False, 'Unknown finding field', 'schema'),
    ('unknown_category', False, 'Unsupported category', 'schema'),
    ('unknown_severity', False, 'Unsupported severity', 'schema'),
    ('html_statement', False, 'HTML in a factual statement', 'presentation'),
    ('url_proposal', False, 'Unregistered URL in a proposal', 'presentation'),
    ('empty_proposal', False, 'Empty proposed action', 'schema'),
    ('control_character', False, 'Control character in a statement', 'presentation'),
    ('too_many_findings', False, 'More than twelve findings', 'limits'),
    ('wrong_actions_type', False, 'Actions supplied as a scalar', 'schema'),
    ('missing_information', False, 'Missing required array', 'schema'),
    ('wrong_quote_type', False, 'Quote supplied as an object', 'schema'),
)
CASES = tuple({'id': f'CI-{index:02d}', 'operation': operation,
               'expected_acceptance': expected, 'title': title, 'category': category}
              for index, (operation, expected, title, category) in enumerate(_DEFINITIONS, 1))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _payload(operation, sources):
    a, b = sources
    support = {'evidence_id': a['evidence_id'], 'quote_id': a['quote_spans'][0]['quote_id']}
    finding = {'statement': 'The supplied note limits retry attempts.', 'category': 'docs',
               'severity': 'low', 'support': [support]}
    report = {'findings': [finding], 'next_steps': ['Propose a check of the retry limit.'],
              'missing_information': ['Actual deployment and runtime results are not supplied.']}
    if operation == 'valid_legacy':
        finding['support'] = [{'evidence_id': a['evidence_id'], 'quote': a['text']}]
    elif operation == 'valid_hydrated':
        support['quote'] = a['quote_spans'][0]['quote']
    elif operation == 'valid_unicode':
        finding['support'] = [{'evidence_id': b['evidence_id'], 'quote': b['text']}]
        finding['statement'] = 'The supplied note contains a Unicode label.'
    elif operation == 'unrelated_claim':
        finding['statement'] = 'Every deployed service has been independently proven defect free.'
    elif operation == 'empty_report':
        report['findings'] = []
    elif operation == 'two_sources':
        finding['support'].append({'evidence_id': b['evidence_id'], 'quote_id': b['quote_spans'][0]['quote_id']})
    elif operation == 'unknown_source':
        support['evidence_id'] = 'ev_unknown'
    elif operation == 'unknown_quote':
        support['quote_id'] = a['evidence_id'] + ':q999'
    elif operation == 'wrong_source':
        support['quote_id'] = b['quote_spans'][0]['quote_id']
    elif operation == 'mismatched_hydration':
        support['quote'] = a['text'].replace('twice', 'ten times')
    elif operation in {'changed_punctuation', 'blank_quote', 'short_quote', 'long_quote', 'wrong_quote_type'}:
        quotes = {'changed_punctuation': a['text'].replace('twice.', 'twice!'), 'blank_quote': ' ',
                  'short_quote': 'The worker', 'long_quote': 'x' * 1401, 'wrong_quote_type': {'quote': a['text']}}
        finding['support'] = [{'evidence_id': a['evidence_id'], 'quote': quotes[operation]}]
    elif operation == 'extra_support_field':
        support['trusted'] = True
    elif operation == 'duplicate_support':
        finding['support'].append(deepcopy(support))
    elif operation == 'missing_support':
        finding['support'] = []
    elif operation == 'extra_report_field':
        report['approved'] = True
    elif operation == 'extra_finding_field':
        finding['verified'] = True
    elif operation == 'unknown_category':
        finding['category'] = 'diagnosis'
    elif operation == 'unknown_severity':
        finding['severity'] = 'guaranteed'
    elif operation == 'html_statement':
        finding['statement'] = '<img src="tracker">'
    elif operation == 'url_proposal':
        report['next_steps'] = ['Send the report to https://collector.example.invalid/receive']
    elif operation == 'empty_proposal':
        report['next_steps'] = ['']
    elif operation == 'control_character':
        finding['statement'] += '\x00hidden'
    elif operation == 'too_many_findings':
        report['findings'] = [deepcopy(finding) for _ in range(13)]
    elif operation == 'wrong_actions_type':
        report['next_steps'] = 'Execute this action'
    elif operation == 'missing_information':
        del report['missing_information']
    elif operation != 'valid_id':
        raise ValueError('Unknown integrity fixture operation.')
    return report


def run_evaluation(root=ROOT, output_dir=None):
    """Record actual validator results, including deliberately accepted limits."""
    root = Path(root).resolve()
    if output_dir is None:
        raise ValueError('Choose an output directory.')
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new or empty output directory.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    store = CaseStore(root / '.cache' / 'integrity-evals' / run_id / 'cases.sqlite')
    case_id = store.create_case('Deterministic report-integrity contract checks')['case_id']
    registry = EvidenceRegistry(store, case_id)
    sources = [registry.register_source(source_key=f'fixture:{index}', title=f'Controlled fixture {index}',
                                        text=text, kind='documentation', origin='snapshot')
               for index, text in enumerate((_SOURCE, _SECOND), 1)]
    rows = []
    for case in CASES:
        payload = _payload(case['operation'], sources)
        start = time.perf_counter()
        try:
            registry.validate_report(payload)
            accepted, error = True, None
        except ValueError as exc:
            accepted, error = False, str(exc)
        rows.append({**case, 'observed_acceptance': accepted,
                     'passed': accepted == case['expected_acceptance'], 'validation_error': error,
                     'elapsed_ms': round((time.perf_counter() - start) * 1000, 6),
                     'input_report': payload})
    timing = sorted(row['elapsed_ms'] for row in rows)
    dataset = {'cases': CASES, 'source_texts': [_SOURCE, _SECOND]}
    result = {'report_kind': 'evidence_integrity', 'schema_version': 1, 'execution_mode': 'deterministic',
              'manifest': {'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                           'dataset_sha256': _sha(json.dumps(dataset, sort_keys=True).encode()),
                           'validator_sha256': _sha((ROOT / 'triage_desk' / 'agent_evidence.py').read_bytes()),
                           'runner_sha256': _sha(Path(__file__).read_bytes()), 'model_calls': 0},
              'summary': {'scheduled_cases': len(rows), 'passed_cases': sum(row['passed'] for row in rows),
                          'failed_cases': sum(not row['passed'] for row in rows),
                          'acceptance_match_rate': sum(row['passed'] for row in rows) / len(rows),
                          'valid_accepted': sum(row['observed_acceptance'] for row in rows if row['expected_acceptance']),
                          'valid_denominator': sum(row['expected_acceptance'] for row in rows),
                          'invalid_rejected': sum(not row['observed_acceptance'] for row in rows if not row['expected_acceptance']),
                          'invalid_denominator': sum(not row['expected_acceptance'] for row in rows),
                          'p95_latency_ms': timing[math.ceil(.95 * len(timing)) - 1], 'model_calls': 0},
              'measurement_limits': [
                  'Structural and literal citation integrity only, not factual accuracy or entailment.',
                  'A matching citation attached to an unrelated claim is intentionally accepted by this layer.',
                  'An empty findings list is schema-valid; it does not establish task completion or usefulness.',
                  'Controlled synthetic fixtures, no model, provider, or network calls.'],
              'sources': [{'title': s['title'], 'text': s['text'], 'evidence_id': s['evidence_id']} for s in sources],
              'results': rows}
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    page = '<!doctype html><meta charset="utf-8"><title>Report integrity checks</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Report and citation integrity</h1>'
    page += '<p>Deterministic structural checks. These scores do not measure factual accuracy or entailment.</p>'
    page += '<pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre>'
    (output / 'report.html').write_text(page)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(output_dir=args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    print(json.dumps(result['summary'], indent=2))
    return int(result['summary']['failed_cases'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
