"""Independent checks for evaluation repository opt-in and safe diagnostics."""
from __future__ import annotations

import json

import pytest

from evaluation import agent_workflow
from triage_desk.cases import CaseStore


@pytest.fixture
def repository(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'triage_desk').mkdir()
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'data' / 'sources.json').write_text(json.dumps([{'source_id': 'doc-fixture'}]))
    (tmp_path / 'triage_desk' / 'reviews.py').write_text('def save():\n    pass\n')
    (tmp_path / 'evaluation' / 'cases.json').write_text(json.dumps({'name': 'Context boundary fixture', 'cases': [{
        'id': 'W3-01', 'question': 'Inspect the supplied code statically and leave a human handoff.',
        'code_file': 'triage_desk/reviews.py', 'start_line': 1, 'end_line': 2,
        'goals': ['Ground static observations in code.'], 'relevant_source_kinds': ['code'],
        'expected_source_ids': ['doc-fixture'], 'expected_boundary': 'No execution or automatic approval.',
    }]}))
    return tmp_path


class LocalService:
    """Exercise actual case persistence without models, tools, or network access."""
    def __init__(self, root, *, fail=False):
        self.store = CaseStore(root / '.cache' / 'agent' / 'cases.sqlite')
        self.calls = []
        self.fail = fail

    def start(self, question, version, context, **options):
        self.calls.append(options)
        case = self.store.create_case(question, version, context, **options)
        case_id = case['case_id']
        if self.fail:
            self.store.add_event(case_id, {'phase': 'execution', 'name': 'start', 'status': 'error',
                                           'error_type': 'ReviewUnavailable', 'error_code': 'invalid_reason'})
            # Unknown text from an old or malformed event must not become a
            # displayed diagnostic, even when it occurs after a valid code.
            self.store.add_event(case_id, {'phase': 'execution', 'name': 'start', 'status': 'error',
                                           'error_code': 'PRIVATE PROVIDER BODY <script>run()</script>',
                                           'arguments': 'PRIVATE CREDENTIAL ARGUMENT'})
            return self.store.update_case(case_id, status='failed', error='Safe local failure.')
        return self.store.update_case(case_id, status='awaiting_review', report={
            'findings': [], 'next_steps': [], 'missing_information': [],
            'quality_review': {'status': 'no_content', 'removed_findings': 0, 'removed_next_steps': 0},
        })


@pytest.mark.parametrize('enabled', [False, True])
def test_opt_in_is_forwarded_and_exported_as_persisted_bool(repository, enabled):
    service = LocalService(repository)
    report = agent_workflow.run_evaluation(
        repository, 'evaluation/cases.json', 'results', live=True,
        repository_context=enabled, service_factory=lambda root: service,
    )
    assert service.calls == ([{'repository_context': True}] if enabled else [{}])
    assert report['repository_context_enabled'] is enabled
    row = report['results'][0]
    assert row['repository_context'] is enabled
    assert service.store.get_case(row['case_id'])['repository_context'] is enabled
    assert row['handoff_observed'] is True
    saved = json.loads((repository / 'results' / 'report.json').read_text())
    assert saved['repository_context_enabled'] is enabled
    assert saved['results'][0]['repository_context'] is enabled


def test_context_flag_without_live_does_not_initialize_service_or_load_env(repository, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: pytest.fail('No .env without --live'))
    report = agent_workflow.run_evaluation(
        repository, 'evaluation/cases.json', 'results', repository_context=True,
        service_factory=lambda root: pytest.fail('Repository opt-in alone must not execute providers'),
    )
    assert report['execution_mode'] == 'validation_only'
    assert report['repository_context_enabled'] is True
    assert report['results'][0]['status'] == 'not_run'
    assert report['summary']['attempted_cases'] == 0
    assert report['summary']['handoff_rate'] is None


def test_failure_code_export_uses_only_known_static_diagnostic(repository):
    service = LocalService(repository, fail=True)
    report = agent_workflow.run_evaluation(
        repository, 'evaluation/cases.json', 'results', live=True,
        repository_context=True, service_factory=lambda root: service,
    )
    row = report['results'][0]
    assert row['status'] == 'failed'
    assert row['error']['code'] == 'invalid_reason'
    assert report['summary']['failed_cases'] == report['summary']['scheduled_cases'] == 1
    assert not row['handoff_observed']
    exported = (repository / 'results' / 'report.json').read_text()
    rendered = (repository / 'results' / 'report.html').read_text()
    assert 'PRIVATE' not in exported + rendered
    assert '<script>' not in rendered


@pytest.mark.parametrize('live', [False, True])
def test_cli_forwards_repository_flag_without_implying_live(monkeypatch, tmp_path, capsys, live):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return {'execution_mode': 'live' if kwargs['live'] else 'validation_only', 'summary': {'failed_cases': 0}}
    monkeypatch.setattr(agent_workflow, 'run_evaluation', run)
    arguments = ['--root', str(tmp_path), '--repository-context', '--case', 'W3-01']
    if live:
        arguments.append('--live')
    assert agent_workflow.main(arguments) == 0
    assert calls == [{'case_ids': ['W3-01'], 'live': live, 'repository_context': True}]
    assert json.loads(capsys.readouterr().out)['execution_mode'] == ('live' if live else 'validation_only')


@pytest.mark.parametrize('invalid', [1, 'true', None])
def test_programmatic_repository_flag_requires_explicit_boolean(repository, invalid):
    with pytest.raises(ValueError, match='must be a boolean'):
        agent_workflow.run_evaluation(repository, 'evaluation/cases.json', 'results', repository_context=invalid)
    assert not (repository / 'results').exists()
