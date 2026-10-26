"""CLI dispatch and saved-case inspection without providers or credentials."""

import json
from types import SimpleNamespace

import pytest

from scripts import investigate_deep as cli
from langchain_core.messages import AIMessage, ToolMessage
from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.cases import CaseStore


def test_parser_requires_one_question_source_and_explicit_review_decision():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['start'])
    with pytest.raises(SystemExit):
        parser.parse_args(['start', '--question', 'text', '--question-file', 'question.txt'])
    with pytest.raises(SystemExit):
        parser.parse_args(['review', 'case'])
    parsed = parser.parse_args(['start', '--question-file', 'question.txt', '--code-file', 'code.py', '--version', '1.66'])
    assert parsed.question is None and parsed.code_file.name == 'code.py'


def test_list_and_show_use_saved_data_without_loading_credentials_or_corpus(tmp_path, monkeypatch, capsys):
    store = CaseStore(tmp_path / '.cache' / 'agent' / 'cases.sqlite')
    case = store.create_case('Private question not printed by inspection', code_context='private code body')
    registry = EvidenceRegistry(store, case['case_id'])
    source = registry.register_source(source_key='test', title='Docs', text='Literal source body',
                                     url='https://docs.streamlit.io/', origin='snapshot', snapshot_at='2026-10-17T00:00:00Z')
    store.add_event(case['case_id'], {'phase': 'tool', 'name': 'search_knowledge_base', 'status': 'ok',
                                    'arguments': {'secret': 'hidden argument'}, 'error_body': 'hidden provider body'})

    def forbidden(*args, **kwargs):
        pytest.fail('Read-only inspection must not initialize providers, corpus, or environment files')
    monkeypatch.setattr(cli, 'load_dotenv', forbidden)
    monkeypatch.setattr(cli, 'TriageDesk', forbidden)
    monkeypatch.setattr(cli, 'InvestigationService', forbidden)
    assert cli.main(['--root', str(tmp_path), 'list']) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == [{'case_id': case['case_id'], 'status': 'new', 'error': '', 'findings_count': 0}]
    assert cli.main(['--root', str(tmp_path), 'show', case['case_id']]) == 0
    output = capsys.readouterr().out
    shown = json.loads(output)
    assert shown['provenance'][0]['evidence_id'] == source['evidence_id']
    assert shown['provenance'][0]['snapshot_at'] == '2026-10-17T00:00:00Z'
    assert shown['events'][0]['name'] == 'search_knowledge_base'
    for hidden in ['private code body', 'Private question', 'Literal source body', 'hidden argument', 'hidden provider body']:
        assert hidden not in output


@pytest.fixture
def fake_service(monkeypatch):
    calls = []
    case = {'case_id': 'case-id', 'status': 'awaiting_review', 'error': '', 'report': {'findings': []}}
    class Service:
        def __init__(self, root, desk, settings):
            calls.append(('construct', root, settings))

        def start(self, question, version, code, **options):
            calls.append(('start', question, version, code))
            if options:
                calls.append(('start_options', options))
            return case

        def resume(self, case_id, decision, notes):
            calls.append(('review', case_id, decision, notes))
            return {**case, 'status': 'approved' if decision == 'approve' else 'awaiting_review'}

        def retry(self, case_id):
            calls.append(('retry', case_id))
            return case

    monkeypatch.setattr(cli, 'TriageDesk', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, 'InvestigationService', Service)
    monkeypatch.setattr(cli, 'load_dotenv', lambda *args, **kwargs: calls.append(('dotenv',)))
    return calls


def test_start_reads_explicit_files_and_dispatches_real_service_contract(tmp_path, monkeypatch, capsys, fake_service):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'fake-test-key')
    monkeypatch.setenv('FIREWORKS_MODEL', 'fake-test-model')
    question = tmp_path / 'question.txt'
    question.write_text('Investigate cached resources', encoding='utf-8')
    code = tmp_path / 'example.py'
    code.write_text('print("example")\n', encoding='utf-8')
    assert cli.main(['--root', str(tmp_path), 'start', '--question-file', str(question),
                     '--version', '1.66.0', '--code-file', str(code)]) == 0
    call = next(call for call in fake_service if call[0] == 'start')
    assert call[1:3] == ('Investigate cached resources', '1.66.0')
    assert 'Supplied file: example.py' in call[3] and 'print("example")' in call[3]
    result = capsys.readouterr().out
    assert json.loads(result)['status'] == 'awaiting_review'
    assert 'fake-test-key' not in result


def test_approval_dispatches_without_model_credentials_or_dotenv(tmp_path, monkeypatch, capsys, fake_service):
    monkeypatch.delenv('FIREWORKS_API_KEY', raising=False)
    monkeypatch.delenv('FIREWORKS_MODEL', raising=False)
    assert cli.main(['--root', str(tmp_path), 'review', 'case-id', '--decision', 'approve', '--notes', 'Reviewed']) == 0
    assert ('review', 'case-id', 'approve', 'Reviewed') in fake_service
    assert not any(call[0] == 'dotenv' for call in fake_service)
    assert json.loads(capsys.readouterr().out)['status'] == 'approved'


def test_repository_context_is_enabled_only_by_explicit_cli_flag(tmp_path, monkeypatch, capsys, fake_service):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'fake-test-key')
    monkeypatch.setenv('FIREWORKS_MODEL', 'fake-test-model')
    assert cli.main(['--root', str(tmp_path), 'start', '--question', 'Inspect helpers', '--repository-context']) == 0
    assert ('start_options', {'repository_context': True}) in fake_service
    assert json.loads(capsys.readouterr().out)['status'] == 'awaiting_review'


@pytest.mark.parametrize('command', [['start', '--question', 'Question'], ['retry', 'case-id'],
                                     ['review', 'case-id', '--decision', 'revise', '--notes', 'Check version']])
def test_model_actions_require_configuration_before_dispatch(tmp_path, monkeypatch, capsys, fake_service, command):
    monkeypatch.delenv('FIREWORKS_API_KEY', raising=False)
    monkeypatch.delenv('FIREWORKS_MODEL', raising=False)
    assert cli.main(['--root', str(tmp_path), *command]) == 1
    assert 'FIREWORKS_API_KEY' in json.loads(capsys.readouterr().err)['error']
    assert not any(call[0] == 'construct' for call in fake_service)


def test_unknown_case_and_runtime_errors_do_not_print_stack_or_provider_message(tmp_path, monkeypatch, capsys):
    assert cli.main(['--root', str(tmp_path), 'show', 'unknown-case']) == 1
    assert 'No saved case' in capsys.readouterr().err
    monkeypatch.setattr(cli, 'dispatch', lambda args: (_ for _ in ()).throw(RuntimeError('provider-secret-value')))
    assert cli.main(['list']) == 1
    result = capsys.readouterr().err
    assert 'provider-secret-value' not in result and 'Traceback' not in result


def test_manual_evaluation_plan_references_real_files_and_sources():
    plan = json.loads((cli.ROOT / 'evaluation' / 'agent-cases.json').read_text())
    sources = {source['source_id'] for source in json.loads((cli.ROOT / 'data' / 'sources.json').read_text())}
    assert len(plan['cases']) == 3
    assert plan['execution_status'] == 'not_run'
    for case in plan['cases']:
        assert set(case['expected_source_ids']) <= sources
        assert case['execution_status'] == 'not_run' and case['results'] is None
        for path in case.get('code_files', []):
            assert (cli.ROOT / path).is_file()


def test_probe_reports_missing_names_without_provider_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Missing configuration must not dispatch a provider call')
    monkeypatch.setattr(cli, '_probe_model', forbidden)
    monkeypatch.setattr(cli, 'build_investigation_tools', forbidden)
    result = cli._probe(cli.AgentSettings())
    assert result == {
        'status': 'error',
        'fireworks': {'status': 'error', 'tool_call_roundtrip': False,
                      'missing': ['FIREWORKS_API_KEY', 'FIREWORKS_MODEL']},
        'keenable': {'status': 'error', 'evidence_count': 0, 'missing': ['KEENABLE_API_KEY']},
    }


def test_explicit_probe_verifies_tool_result_and_outputs_only_status_counts(monkeypatch):
    class Model:
        calls = 0

        def bind_tools(self, tools, tool_choice):
            assert len(tools) == 1 and tool_choice == tools[0].name == 'readiness_probe'
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='', tool_calls=[{'name': 'readiness_probe', 'args': {}, 'id': 'probe-call'}])
            assert isinstance(messages[-1], ToolMessage)
            assert messages[-1].content == 'engineering_issue_desk_probe_ok'
            assert messages[-1].tool_call_id == 'probe-call'
            return AIMessage(content='private-provider-output')

    model = Model()
    monkeypatch.setattr(cli, '_probe_model', lambda settings: model)
    monkeypatch.setattr(cli, 'build_investigation_tools', lambda *args, **kwargs: [SimpleNamespace(
        name='search_web', invoke=lambda arguments: {'status': 'ok', 'evidence': [{'text': 'private-source-content'}]})])
    result = cli._probe(cli.AgentSettings(api_key='fake-fireworks-key', model='fake-model', keenable_api_key='fake-search-key'))
    assert result == {'status': 'ok', 'fireworks': {'status': 'ok', 'tool_call_roundtrip': True},
                      'keenable': {'status': 'ok', 'evidence_count': 1}}
    assert model.calls == 2
    assert all(value not in json.dumps(result) for value in [
        'private-provider-output', 'private-source-content', 'fake-fireworks-key', 'fake-search-key'])


def test_review_memory_cli_requires_explicit_flag(tmp_path, monkeypatch, capsys, fake_service):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'fake-test-key')
    monkeypatch.setenv('FIREWORKS_MODEL', 'fake-test-model')
    assert cli.main(['--root', str(tmp_path), 'start', '--question', 'Recall reviews', '--review-memory']) == 0
    assert ('start_options', {'memory_context': True}) in fake_service
    assert json.loads(capsys.readouterr().out)['status'] == 'awaiting_review'


def test_publication_cli_preview_and_show_do_not_initialize_models_or_post(tmp_path, monkeypatch, capsys):
    import triage_desk.review_publication as publication
    actions = []
    class Publisher:
        def __init__(self, store, token='', **options):
            assert not token
        def prepare(self, case_id, pr_number):
            actions.append(('prepare', case_id, pr_number))
            return {'plan_id': 'plan', 'status': 'prepared', 'body': 'Public preview'}
        def get(self, plan_id):
            return {'plan_id': plan_id, 'status': 'prepared', 'body': 'Public preview'}
        def publish(self, *args, **kwargs):
            pytest.fail('Preview and inspection must never publish')
    def forbidden(*args, **kwargs):
        pytest.fail('Preview must not initialize a model or corpus')
    monkeypatch.setattr(publication, 'ReviewPublisher', Publisher)
    monkeypatch.setattr(cli, 'TriageDesk', forbidden)
    monkeypatch.setattr(cli, 'InvestigationService', forbidden)
    assert cli.main(['--root', str(tmp_path), 'prepare-publication', 'approved-case', '--pr', '10']) == 0
    assert actions == [('prepare', 'approved-case', 10)]
    assert json.loads(capsys.readouterr().out)['status'] == 'prepared'
    monkeypatch.setattr(cli, 'load_dotenv', forbidden)
    assert cli.main(['--root', str(tmp_path), 'show-publication', 'plan']) == 0
    assert json.loads(capsys.readouterr().out)['body'] == 'Public preview'
