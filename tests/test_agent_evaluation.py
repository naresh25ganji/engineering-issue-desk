"""Actual evaluation dispatch, provenance, accounting, and safe artifact export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation import agent_workflow as evaluation
from triage_desk.cases import CaseStore


@pytest.fixture
def fixture_root(tmp_path):
    root = tmp_path / 'repository'
    (root / 'evaluation').mkdir(parents=True)
    (root / 'data').mkdir()
    dataset = json.loads((evaluation.ROOT / 'evaluation' / 'week3-cases.json').read_text())
    (root / 'evaluation' / 'week3-cases.json').write_text(json.dumps(dataset))
    source_ids = {identity for case in dataset['cases'] for identity in case['expected_source_ids']}
    (root / 'data' / 'sources.json').write_text(json.dumps([{'source_id': identity} for identity in sorted(source_ids)]))
    for name in evaluation.FIXTURE_FILES:
        destination = root / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((evaluation.ROOT / name).read_bytes())
    return root


class Service:
    def __init__(self, root, *, fail_index=None, secret='', unsafe_url=False):
        self.store = CaseStore(root / '.cache' / 'agent' / 'cases.sqlite')
        self.calls = []
        self.fail_index, self.secret, self.unsafe_url = fail_index, secret, unsafe_url

    def start(self, question, version, context, *, repository_context=False):
        self.calls.append((question, version, context))
        case = self.store.create_case(question, version, context, repository_context=repository_context)
        case_id, index = case['case_id'], len(self.calls)
        self.store.add_event(case_id, {'phase': 'model', 'name': 'investigator', 'status': 'started', 'call_id': 'm1'})
        if index == self.fail_index:
            self.store.add_event(case_id, {'phase': 'model_result', 'name': 'investigator', 'status': 'error',
                                           'call_id': 'm1', 'error_type': 'ProviderError'})
            return self.store.update_case(case_id, status='failed', error='Local safe failure')
        self.store.add_event(case_id, {'phase': 'model_result', 'name': 'investigator', 'status': 'completed',
                                       'call_id': 'm1', 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}})
        self.store.add_event(case_id, {'phase': 'tool', 'name': 'search_knowledge_base', 'status': 'started', 'call_id': 't1'})
        self.store.add_event(case_id, {'phase': 'tool_result', 'name': 'search_knowledge_base', 'status': 'ok',
                                       'call_id': 't1', 'arguments': 'private-tool-history-body'})
        self.store.add_event(case_id, {'phase': 'model', 'name': 'evidence_reviewer', 'status': 'started', 'call_id': 'm2'})
        self.store.add_event(case_id, {'phase': 'model_result', 'name': 'evidence_reviewer', 'status': 'completed',
                                       'call_id': 'm2', 'usage': {'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6}})
        quote = 'A literal code observation sufficiently long for review.'
        source = self.store.put_evidence(case_id, {
            'source_key': 'code:fixture', 'title': 'Fixture <script>title</script>',
            'text': quote + '\nprivate-full-source-body',
            'url': 'javascript:alert(1)' if self.unsafe_url else 'https://docs.streamlit.io/',
            'origin': 'user_code', 'kind': 'code',
        })
        report = {'findings': [{'statement': 'Static observation ' + self.secret, 'category': 'correctness', 'severity': 'low',
                               'support': [{'evidence_id': source['evidence_id'], 'quote_id': 'q1', 'quote': quote}]}],
                  'next_steps': ['Propose a focused test.'], 'missing_information': ['Deployment scope?'],
                  'quality_review': {'status': 'completed', 'reviewer_model': 'test-reviewer',
                                     'removed_findings': 1, 'removed_next_steps': 1,
                                     'notes': '</pre><script>bad()</script>'}}
        return self.store.update_case(case_id, status='awaiting_review', report=report)

    def resume(self, *args, **kwargs):
        pytest.fail('Evaluation must never approve, reject, or revise a report')


def run(root, directory='results', **kwargs):
    return evaluation.run_evaluation(root, 'evaluation/week3-cases.json', directory, **kwargs)


def test_default_only_validates_fixtures_and_writes_unscored_inventory(fixture_root, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: pytest.fail('Validation must not load .env'))
    report = run(fixture_root, service_factory=lambda root: pytest.fail('Providers require explicit live=True'))
    assert report['execution_mode'] == 'validation_only'
    assert report['report_kind'] == 'deep_agent_workflow'
    assert report['summary']['scheduled_cases'] == report['summary']['not_run_cases'] == 3
    assert report['summary']['handoff_rate'] is None and report['summary']['usage_complete'] is None
    assert all(row['report'] is None and not row['attempted'] for row in report['results'])
    assert json.loads((fixture_root / 'results' / 'report.json').read_text()) == report
    assert 'validation_only' in (fixture_root / 'results' / 'report.html').read_text()


def test_fixture_line_ranges_and_hashes_match_exact_repository_bytes(fixture_root):
    _, cases, dataset_hash = evaluation.load_cases(fixture_root, fixture_root / 'evaluation' / 'week3-cases.json')
    assert len(dataset_hash) == 64
    for case in cases:
        sample = case['sample']
        raw = (fixture_root / sample['path']).read_bytes()
        excerpt = ''.join(raw.decode().splitlines(keepends=True)[sample['start_line'] - 1:sample['end_line']])
        assert sample['file_sha256'] == hashlib.sha256(raw).hexdigest()
        assert sample['excerpt_sha256'] == hashlib.sha256(excerpt.encode()).hexdigest()
        assert case['_context'].endswith(excerpt)
        assert len(case['_context']) <= 16000


@pytest.mark.parametrize('changes', [{'start_line': 0}, {'end_line': 100000}, {'start_line': True},
                                     {'code_file': '../outside.py'}, {'expected_source_ids': ['not-a-real-source']}])
def test_invalid_bounds_paths_or_sources_fail_before_any_dispatch(fixture_root, changes):
    path = fixture_root / 'evaluation' / 'week3-cases.json'
    dataset = json.loads(path.read_text())
    dataset['cases'][0].update(changes)
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        run(fixture_root, live=True, service_factory=lambda root: pytest.fail('Invalid fixtures must not start providers'))


def test_live_dispatch_preserves_failed_case_denominator_and_actual_usage(fixture_root):
    service = Service(fixture_root, fail_index=2)
    report = run(fixture_root, live=True, service_factory=lambda root: service)
    summary = report['summary']
    assert len(service.calls) == summary['attempted_cases'] == summary['scheduled_cases'] == 3
    assert summary['completed_handoffs'] == 2 and summary['failed_cases'] == 1
    assert summary['handoff_rate'] == pytest.approx(2 / 3)
    assert summary['model_calls'] == 5 and summary['tool_calls'] == 2
    assert summary['model_calls_by_role'] == {'investigator': 3, 'evidence_reviewer': 2}
    assert summary['reported_tokens'] == 42
    assert summary['errors_by_role_or_tool'] == {'investigator': 1}
    assert summary['model_calls_missing_usage_by_field']['total_tokens'] == 1
    assert not summary['usage_complete']
    assert summary['citation_integrity_passed_cases'] == 2
    assert summary['ai_rejected_findings'] == summary['ai_rejected_next_steps'] == 2
    assert summary['human_accuracy'] is None
    assert all(all(value is None for value in row['human_review'].values()) for row in report['results'])
    exported = (fixture_root / 'results' / 'report.json').read_text()
    assert 'private-tool-history-body' not in exported and 'private-full-source-body' not in exported
    assert all('Supplied repository file:' in call[2] for call in service.calls)


def test_selected_case_and_nonempty_output_preserve_prior_artifacts(fixture_root):
    service = Service(fixture_root)
    report = run(fixture_root, case_ids=['W3-02'], live=True, service_factory=lambda root: service)
    assert len(service.calls) == 1 and report['results'][0]['id'] == 'W3-02'
    output = fixture_root / 'results' / 'report.json'
    original = output.read_bytes()
    with pytest.raises(ValueError, match='never overwritten'):
        run(fixture_root, live=True, service_factory=lambda root: pytest.fail('Output conflict must precede providers'))
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match='Selected case IDs'):
        run(fixture_root, directory='other', case_ids=['W3-99'])


def test_exports_escape_html_reject_unsafe_links_and_redact_known_keys(fixture_root, monkeypatch):
    secret = 'test-sensitive-credential-value'
    monkeypatch.setenv('FIREWORKS_API_KEY', secret)
    service = Service(fixture_root, secret=secret, unsafe_url=True)
    report = run(fixture_root, live=True, service_factory=lambda root: service, case_ids=['W3-01'])
    rendered = (fixture_root / 'results' / 'report.html').read_text()
    exported = (fixture_root / 'results' / 'report.json').read_text()
    assert secret not in rendered + exported and '[REDACTED]' in exported
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert 'href="javascript:' not in rendered
    assert report['results'][0]['sources'][0]['url'] == ''
    assert report['results'][0]['sources'][0]['url_rejected'] is True


def test_initialization_and_unobserved_runtime_failures_are_not_successes(fixture_root):
    def broken_factory(root):
        raise RuntimeError('private-provider-error-key')
    report = run(fixture_root, live=True, service_factory=broken_factory)
    assert report['summary']['failed_cases'] == report['summary']['scheduled_cases'] == 3
    assert report['summary']['completed_handoffs'] == report['summary']['attempted_cases'] == 0
    assert report['summary']['handoff_rate'] == 0
    assert not report['summary']['usage_complete']
    assert report['summary']['accounting_unknown_error_rows'] == 3
    assert 'private-provider-error-key' not in json.dumps(report)


def test_interrupted_and_legacy_call_records_do_not_invent_usage():
    accounting = evaluation.event_accounting([
        {'phase': 'model', 'name': 'investigator', 'status': 'started', 'call_id': 'a'},
        {'phase': 'model_result', 'name': 'investigator', 'status': 'completed', 'call_id': 'a',
         'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5}},
        {'phase': 'model', 'name': 'evidence_reviewer', 'status': 'started', 'call_id': 'b'},
        {'phase': 'tool', 'name': 'search_web', 'status': 'error'},
    ])
    assert accounting['model_calls'] == 2 and accounting['tool_calls'] == 1
    assert accounting['reported_token_usage']['total_tokens'] == 5
    assert accounting['unfinished_call_records'] == 1 and not accounting['usage_complete']
    assert accounting['errors_by_role_or_tool'] == {'search_web': 1}


def test_citation_integrity_does_not_equate_to_entailment_or_empty_success():
    assert evaluation.citation_integrity(None, [])['status'] == 'not_applicable'
    result = evaluation.citation_integrity({'findings': [{'support': [{'evidence_id': 'unknown', 'quote': 'invented'}]}]}, [])
    assert result['status'] == 'failed' and result['failed_supports'] == 1
    assert 'not entailment' in result['meaning']
