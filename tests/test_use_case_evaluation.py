"""Full-workflow evaluation contracts, isolation and safe resumable exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import pytest

from evaluation import use_cases as evaluation
from triage_desk.cases import CaseStore


@pytest.fixture
def fixture_root(tmp_path):
    root = tmp_path / 'repository'
    fixture = root / 'evaluation' / 'fixtures' / 'workflows' / 'sample.py'
    fixture.parent.mkdir(parents=True)
    text = 'def limit(value):\n    return value < 10\n'
    fixture.write_text(text, encoding='utf-8')
    (root / 'data').mkdir()
    (root / 'data' / 'sources.json').write_text('[]')
    cases = []
    for index, profile in enumerate(evaluation.USE_CASES):
        for number in range(10):
            identity = f'WF-{index * 10 + number + 1:02}'
            cases.append({'id': identity, 'use_case': profile.id, 'scenario': 'boundary',
                'title': identity + ' evaluation-only fixture title',
                'question': 'Inspect the supplied comparison and propose a focused check.', 'version': '',
                'context': {'path': str(fixture.relative_to(root)), 'text': text,
                            'sha256': hashlib.sha256(text.encode()).hexdigest()},
                'expected': {'min_findings': 1, 'min_next_steps': 1,
                    'required_tools': ['read_case_evidence'], 'evidence_anchors': ['return value < 10'],
                    'behavior_notes': 'EVALUATOR_ONLY_BEHAVIOR_MARKER',
                    'human_rubric': ['EVALUATOR_ONLY_RUBRIC_MARKER']},
                'label_review': {'status': 'pending', 'reviewer': None, 'notes': ''}})
    dataset = {'name': 'Workflow contract test', 'version': 1,
        'label_status': 'ai_proposed_pending_human_review', 'description': 'Forty public code fixtures.',
        'source_policy': 'Synthetic public fixture evidence.', 'cases': cases}
    (root / evaluation.DEFAULT_CASES).write_text(json.dumps(dataset), encoding='utf-8')
    return root


class FakeService:
    def __init__(self, workspace, *, fail=False, broken_quote=False, no_review=False,
                 no_findings=False, missing_tool=False, secret='', unsafe_url=False):
        self.store = CaseStore(workspace / '.cache' / 'agent' / 'cases.sqlite')
        self.calls, self.fail, self.secret = [], fail, secret
        self.broken_quote, self.no_review, self.no_findings = broken_quote, no_review, no_findings
        self.missing_tool, self.unsafe_url = missing_tool, unsafe_url

    def start(self, question, version, context, *, use_case):
        self.calls.append({'question': question, 'version': version, 'context': context, 'use_case': use_case})
        case = self.store.create_case(question, version, context, use_case_id=use_case)
        identity = case['case_id']
        self.store.add_event(identity, {'phase': 'model', 'name': 'investigator', 'status': 'started', 'call_id': 'm1'})
        if self.fail:
            self.store.add_event(identity, {'phase': 'model_result', 'name': 'investigator', 'status': 'error', 'call_id': 'm1'})
            return self.store.update_case(identity, status='failed', error='Safe failure')
        self.store.add_event(identity, {'phase': 'model_result', 'name': 'investigator', 'status': 'completed',
            'call_id': 'm1', 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}})
        if not self.missing_tool:
            self.store.add_event(identity, {'phase': 'tool', 'name': 'read_case_evidence', 'status': 'started', 'call_id': 't1'})
            self.store.add_event(identity, {'phase': 'tool_result', 'name': 'read_case_evidence', 'status': 'completed',
                'call_id': 't1', 'arguments': 'PRIVATE_TOOL_ARGUMENTS', 'error': 'PRIVATE_RAW_PROVIDER_ERROR'})
        source = self.store.put_evidence(identity, {'source_key': 'fixture', 'text': context + '\nPRIVATE_UNRELATED_SOURCE_BODY',
            'title': 'Source <script>unsafe()</script>', 'url': 'javascript:alert(1)' if self.unsafe_url else 'https://example.com/fixture',
            'origin': 'user_code', 'kind': 'code'})
        report = {'findings': [] if self.no_findings else [{'statement': 'The shown comparison uses a strict boundary. ' + self.secret,
            'category': 'correctness', 'severity': 'low',
            'support': [{'evidence_id': source['evidence_id'], 'quote_id': 'Q1',
                         'quote': 'invented return value < 10' if self.broken_quote else 'return value < 10'}]}],
            'next_steps': ['Propose checking equality at ten.'], 'missing_information': []}
        if not self.no_review:
            self.store.add_event(identity, {'phase': 'model', 'name': 'evidence_reviewer', 'status': 'started', 'call_id': 'm2'})
            self.store.add_event(identity, {'phase': 'model_result', 'name': 'evidence_reviewer', 'status': 'completed', 'call_id': 'm2',
                'usage': {'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6}})
            report['quality_review'] = {'status': 'completed', 'finding_verdicts': [], 'next_step_verdicts': [],
                'removed_findings': 0, 'removed_next_steps': 0, 'reviewer_model': 'test-reviewer'}
        return self.store.update_case(identity, status='awaiting_review', report=report)

    def resume(self, *args, **kwargs):
        pytest.fail('The evaluator may never resume a human decision')


def factory_collector(**options):
    services, lock = [], threading.Lock()
    def factory(workspace, *, source_root, settings, reviewer_prompt_version):
        selected = options(workspace) if callable(options) else options
        service = FakeService(workspace, **selected)
        service.workspace, service.source_root, service.reviewer_prompt = workspace, source_root, reviewer_prompt_version
        with lock:
            services.append(service)
        return service
    return factory, services


def run(root, directory='results', **kwargs):
    return evaluation.run_evaluation(root, output_dir=directory, progress=False, **kwargs)


def test_default_validates_all_forty_without_environment_or_provider_access(fixture_root, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: pytest.fail('Offline inventory must not load .env'))
    report = run(fixture_root, service_factory=lambda *args, **kwargs: pytest.fail('Providers require explicit live=True'))
    assert report['report_kind'] == 'use_case_workflow'
    assert report['execution_mode'] == 'validation_only'
    assert report['summary']['scheduled_cases'] == report['summary']['not_run_cases'] == 40
    assert all(value['scheduled_cases'] == 10 for value in report['by_use_case'].values())
    assert report['summary']['handoff_rate'] is report['summary']['contract_pass_rate'] is None
    assert report['summary']['usage_complete'] is report['summary']['quoted_anchor_recall'] is None
    assert all(not row['attempted'] and all(value is None for value in row['scores'].values()) for row in report['results'])
    assert json.loads((fixture_root / 'results' / 'report.json').read_text()) == report
    assert (fixture_root / 'results' / 'report.html').is_file()
    assert len((fixture_root / 'results' / 'results.csv').read_text().splitlines()) == 41


@pytest.mark.parametrize('mutation', ['hash', 'text', 'escape', 'anchor', 'count', 'id', 'pending'])
def test_entire_suite_is_validated_before_filtering_or_dispatch(fixture_root, mutation):
    path = fixture_root / evaluation.DEFAULT_CASES
    dataset = json.loads(path.read_text())
    case = dataset['cases'][-1]
    if mutation == 'hash':
        case['context']['sha256'] = '0' * 64
    elif mutation == 'text':
        case['context']['text'] += 'different'
    elif mutation == 'escape':
        case['context']['path'] = '../outside.py'
    elif mutation == 'anchor':
        case['expected']['evidence_anchors'] = ['not present']
    elif mutation == 'count':
        case['expected']['min_findings'] = True
    elif mutation == 'id':
        case['id'] = 'WF-01'
    else:
        case['label_review']['status'] = 'approved'
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        run(fixture_root, live=True, case_ids=['WF-01'], service_factory=lambda *args, **kwargs: pytest.fail('Invalid data cannot dispatch'))
    assert not (fixture_root / 'results').exists()


def test_live_runs_all_profiles_in_isolated_stores_without_label_leakage(fixture_root):
    factory, services = factory_collector()
    report = run(fixture_root, live=True, workers=4, case_ids=['WF-01', 'WF-11', 'WF-21', 'WF-31'],
                 service_factory=factory, reviewer_prompt='calibrated')
    summary = report['summary']
    assert len(services) == summary['scheduled_cases'] == summary['completed_handoffs'] == 4
    assert len({service.store.path for service in services}) == 4
    assert len({service.workspace for service in services}) == 4
    assert all(service.source_root == fixture_root and service.reviewer_prompt == 'calibrated' for service in services)
    assert all('.cache/use-case-evals/' in str(service.workspace) for service in services)
    assert {service.calls[0]['use_case'] for service in services} == evaluation.PROFILE_IDS
    serialized_calls = json.dumps([service.calls for service in services])
    assert 'EVALUATOR_ONLY_' not in serialized_calls and 'evaluation-only fixture title' not in serialized_calls
    assert summary['handoff_rate'] == summary['contract_pass_rate'] == summary['quoted_anchor_recall'] == 1
    assert summary['model_calls'] == 8 and summary['tool_calls'] == 4 and summary['reported_tokens'] == 84
    assert summary['mean_reported_tokens'] == 21 and summary['usage_complete'] is True
    assert summary['citation_integrity_applicable_cases'] == summary['citation_integrity_passed_cases'] == 4
    assert summary['human_accuracy'] is None
    assert all(row['human_review']['status'] == 'pending' for row in report['results'])
    assert all(target['passed'] is True for target in summary['targets'].values())
    for row in report['results']:
        assert all(row['scores'][key] is True for key in ('identity_preserved', 'independent_review_completed', 'required_tools_met'))
        assert row['trace']['metadata']['remote_tracing'] is False
        assert row['trace']['children'] == row['events']


def test_errors_and_missing_findings_stay_in_all_denominators(fixture_root):
    services = []
    def factory(workspace, **kwargs):
        number = workspace.name
        service = FakeService(workspace, fail=number == 'WF-02', no_findings=number == 'WF-03')
        services.append(service)
        return service
    report = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02', 'WF-03'], service_factory=factory)
    summary = report['summary']
    assert summary['attempted_cases'] == summary['scheduled_cases'] == 3
    assert summary['failed_cases'] == 1 and summary['completed_handoffs'] == 2
    assert summary['handoff_rate'] == pytest.approx(2 / 3)
    assert summary['contract_pass_rate'] == summary['quoted_anchor_recall'] == pytest.approx(1 / 3)
    assert summary['anchors_checked'] == 3 and summary['anchors_covered'] == 1
    assert summary['citation_integrity_applicable_cases'] == 1
    assert not summary['usage_complete']
    assert summary['mean_reported_tokens'] is None
    assert summary['targets']['mean_reported_tokens']['passed'] is None
    assert not report['results'][2]['scores']['contract_passed']
    assert not report['results'][2]['scores']['min_findings_met']


@pytest.mark.parametrize('options,failed_score', [({'no_review': True}, 'independent_review_completed'),
    ({'missing_tool': True}, 'required_tools_met'), ({'broken_quote': True}, None)])
def test_handoff_alone_does_not_establish_contract_success(fixture_root, options, failed_score):
    factory, _ = factory_collector(**options)
    report = run(fixture_root, live=True, case_ids=['WF-01'], service_factory=factory)
    row = report['results'][0]
    assert row['handoff_observed'] and row['scores']['contract_passed'] is False
    if failed_score:
        assert row['scores'][failed_score] is False
    else:
        assert row['automatic_citation_integrity']['status'] == 'failed'
        assert row['scores']['quoted_anchor_recall'] == 0


def test_constructor_failure_exports_all_cases_without_raw_error(fixture_root):
    def broken_factory(*args, **kwargs):
        raise RuntimeError('PRIVATE_RAW_PROVIDER_ERROR')
    report = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=broken_factory)
    assert report['summary']['failed_cases'] == report['summary']['scheduled_cases'] == 2
    assert report['summary']['attempted_cases'] == 0
    assert report['summary']['handoff_rate'] == report['summary']['contract_pass_rate'] == 0
    assert report['summary']['quoted_anchor_recall'] == 0
    assert report['summary']['accounting_unknown_error_rows'] == 2
    assert not report['summary']['usage_complete']
    assert report['summary']['mean_reported_tokens'] is None
    assert 'PRIVATE_RAW_PROVIDER_ERROR' not in json.dumps(report)
    assert all(row['status'] == 'not_started' for row in report['results'])


def test_safe_exports_redact_secrets_escape_html_and_exclude_raw_events(fixture_root, monkeypatch):
    secret = 'sensitive-test-provider-credential'
    monkeypatch.setenv('FIREWORKS_API_KEY', secret)
    factory, _ = factory_collector(secret=secret, unsafe_url=True)
    report = run(fixture_root, live=True, case_ids=['WF-01'], service_factory=factory)
    all_exports = '\n'.join(path.read_text() for path in (fixture_root / 'results').rglob('*') if path.is_file())
    assert secret not in all_exports and '[REDACTED]' in all_exports
    assert 'PRIVATE_TOOL_ARGUMENTS' not in all_exports
    assert 'PRIVATE_RAW_PROVIDER_ERROR' not in all_exports
    assert 'PRIVATE_UNRELATED_SOURCE_BODY' not in all_exports
    rendered = (fixture_root / 'results' / 'report.html').read_text()
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert 'href="javascript:' not in rendered
    assert report['results'][0]['sources'][0]['url'] == ''
    assert report['results'][0]['sources'][0]['url_rejected']
    assert report['results'][0]['automatic_citation_integrity']['status'] == 'passed'


def test_resume_is_strict_skips_completed_cases_and_retains_failures(fixture_root):
    factory, services = factory_collector()
    original = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory)
    resumed = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory, resume=True)
    assert len(services) == 2 and original['results'] == resumed['results']
    assert resumed['manifest'] == original['manifest']
    with pytest.raises(ValueError, match='Resume refused'):
        run(fixture_root, live=True, case_ids=['WF-01'], service_factory=factory, resume=True)
    with pytest.raises(ValueError, match='Resume refused'):
        run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory,
            resume=True, reviewer_prompt='calibrated')
    case_path = fixture_root / 'results' / 'cases' / 'WF-01.json'
    row = json.loads(case_path.read_text())
    row['input_sha256'] = 'tampered'
    case_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Saved case identity'):
        run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory, resume=True)


def test_incremental_resume_only_executes_missing_case_artifacts(fixture_root):
    factory, services = factory_collector()
    report = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory)
    (fixture_root / 'results' / 'cases' / 'WF-02.json').unlink()
    resumed = run(fixture_root, live=True, case_ids=['WF-01', 'WF-02'], service_factory=factory, resume=True)
    assert len(services) == 3
    assert resumed['results'][0] == report['results'][0]
    assert resumed['results'][1]['case_id'] != report['results'][1]['case_id']


def test_existing_output_is_never_overwritten_and_filters_intersect(fixture_root):
    report = run(fixture_root, case_ids=['WF-01', 'WF-11'], use_cases=['incident_triage'])
    assert [row['id'] for row in report['results']] == ['WF-11']
    original = (fixture_root / 'results' / 'report.json').read_bytes()
    with pytest.raises(ValueError, match='never overwritten'):
        run(fixture_root, live=True, service_factory=lambda *args, **kwargs: pytest.fail('Output conflict precedes dispatch'))
    assert (fixture_root / 'results' / 'report.json').read_bytes() == original
    with pytest.raises(ValueError, match='no cases in common'):
        run(fixture_root, 'other', case_ids=['WF-01'], use_cases=['incident_triage'])


def test_matched_comparison_rejects_inventories_and_mismatched_case_sets(fixture_root):
    factory, _ = factory_collector()
    run(fixture_root, 'baseline', live=True, case_ids=['WF-01'], service_factory=factory)
    run(fixture_root, 'candidate', live=True, case_ids=['WF-01'], service_factory=factory, reviewer_prompt='calibrated')
    comparison = evaluation.compare_runs(fixture_root / 'baseline', fixture_root / 'candidate', fixture_root / 'comparison')
    assert comparison['report_kind'] == 'use_case_comparison'
    assert comparison['metric_deltas']['contract_pass_rate'] == 0
    assert comparison['pairs'][0]['id'] == 'WF-01'
    output = fixture_root / 'comparison'
    assert json.loads((output / 'comparison.json').read_text()) == comparison
    assert (output / 'comparison.json').read_bytes() == (output / 'report.json').read_bytes()
    assert (output / 'comparison.html').read_bytes() == (output / 'report.html').read_bytes()
    assert 'Engineering workflow comparison' in (output / 'report.html').read_text()
    from triage_desk.evaluation_lab import list_reports
    # Reports are discoverable from the same standard location used by the app.
    comparison_dir = fixture_root / 'evaluation' / 'paired-workflows'
    evaluation.compare_runs(fixture_root / 'baseline', fixture_root / 'candidate', comparison_dir)
    assert comparison_dir / 'report.json' in list_reports(fixture_root, 'workflows')
    run(fixture_root, 'inventory', case_ids=['WF-01'])
    with pytest.raises(ValueError, match='two live'):
        evaluation.compare_runs(fixture_root / 'baseline', fixture_root / 'inventory', fixture_root / 'invalid-comparison')
    run(fixture_root, 'different', live=True, case_ids=['WF-02'], service_factory=factory)
    with pytest.raises(ValueError, match='matched unique'):
        evaluation.compare_runs(fixture_root / 'baseline', fixture_root / 'different', fixture_root / 'invalid-comparison')


def test_partial_usage_and_incomplete_latency_do_not_display_complete_aggregates(fixture_root):
    _, cases, _ = evaluation.load_cases(fixture_root, case_ids=['WF-01', 'WF-02'])
    first, second = [evaluation._base_row(case) for case in cases]
    first['elapsed_ms'] = 1000
    first['accounting'] = evaluation.event_accounting([
        {'phase': 'model', 'name': 'investigator', 'status': 'started', 'call_id': 'm1'},
        {'phase': 'model_result', 'name': 'investigator', 'status': 'completed', 'call_id': 'm1',
         'usage': {'input_tokens': 4, 'output_tokens': 1, 'total_tokens': 5}},
        {'phase': 'model', 'name': 'evidence_reviewer', 'status': 'started', 'call_id': 'm2'},
        {'phase': 'model_result', 'name': 'evidence_reviewer', 'status': 'completed', 'call_id': 'm2', 'usage': {}},
    ])
    summary = evaluation.summarize([first, second], live=True)
    assert summary['reported_tokens'] == 5  # The observed subtotal remains useful.
    assert summary['usage_complete'] is False
    assert summary['mean_reported_tokens'] is None
    assert summary['latency_measured_cases'] == 1
    assert summary['latency_p95_seconds'] is None
    for metric in ('mean_reported_tokens', 'latency_p95_seconds'):
        assert summary['targets'][metric]['measured'] is None
        assert summary['targets'][metric]['passed'] is None


def test_cli_inventory_and_invalid_configuration(fixture_root, capsys):
    assert evaluation.main(['run', '--root', str(fixture_root), '--output-dir', 'cli', '--case', 'WF-01']) == 0
    assert 'validation_only' in capsys.readouterr().out
    assert evaluation.main(['run', '--root', str(fixture_root), '--output-dir', 'cli']) == 1
    assert 'Raw errors are excluded' in capsys.readouterr().out


def test_bundled_suite_is_pinned_and_has_ten_cases_per_profile():
    dataset, cases, sha = evaluation.load_cases(evaluation.ROOT)
    assert len(sha) == 64 and len(cases) == 40
    assert dataset['label_status'] == 'ai_proposed_pending_human_review'
    assert {identity: sum(case['use_case'] == identity for case in cases) for identity in evaluation.PROFILE_IDS} == {
        identity: 10 for identity in evaluation.PROFILE_IDS}
