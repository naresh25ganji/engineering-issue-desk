"""Run real isolated Bandit over registered text; never execute the supplied code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from triage_desk.agent_budget import BudgetExceeded, InvestigationBudget
from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.cases import CaseStore
from triage_desk import static_analysis


@pytest.fixture
def context(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case_id = store.create_case('Review registered Python statically')['case_id']
    registry = EvidenceRegistry(store, case_id)
    budget = InvestigationBudget(store, case_id)
    tool = static_analysis.build_static_analysis_tools(registry, budget)[0]
    return store, case_id, registry, budget, tool


def register(context, code, **metadata):
    return context[2].register_source(source_key='fixture:' + hashlib.sha256(code.encode()).hexdigest(),
                                     title='Registered code fixture', text=code,
                                     kind=metadata.pop('kind', 'code'), origin=metadata.pop('origin', 'user_code'), **metadata)


def invoke(context, source):
    return context[4].invoke({'evidence_id': source['evidence_id']})


def test_real_bandit_flags_are_attributed_and_saved_with_parent_provenance(context):
    code = 'import subprocess\nsubprocess.run(user_command, shell=True)\neval(user_text)\n'
    source = register(context, code)
    result = invoke(context, source)
    assert result['status'] == 'ok'
    analysis = result['analysis']
    assert analysis['analyzer'] == 'Bandit' and analysis['version'].startswith('1.')
    assert analysis['ruleset_version'] == static_analysis.RULESET_VERSION
    assert {item['rule_id'] for item in analysis['findings']} == {'B602', 'B307'}
    assert [item['line'] for item in analysis['findings']] == [2, 3]
    assert analysis['parent_evidence_id'] == source['evidence_id']
    assert analysis['code_sha256'] == hashlib.sha256(code.encode()).hexdigest()
    assert analysis['parent_sha256'] == source['sha256']
    derived = result['evidence'][0]
    saved = next(item for item in context[0].evidence(context[1]) if item['evidence_id'] == derived['evidence_id'])
    assert saved['origin'] == 'analysis' and saved['evidence_type'] == 'static_analysis'
    assert source['evidence_id'] in saved['text'] and analysis['fingerprint'] in saved['version_scope']
    assert 'attributed analyzer flags, not verified vulnerabilities' in saved['text']
    assert 'user_command' not in saved['text'] and 'user_text' not in saved['text']
    assert context[3].tool_calls == 1 and context[3].model_calls == 0
    events = context[0].events(context[1])
    assert events[-1]['name'] == 'run_static_checks' and events[-1]['status'] == 'ok'


def test_supplied_python_is_never_imported_executed_or_sent_to_a_shell(context, tmp_path):
    marker = tmp_path / 'should-not-exist'
    code = f'from pathlib import Path\nPath({str(marker)!r}).write_text("EXECUTED")\nimport os\nos.system("touch {marker}")\n'
    result = invoke(context, register(context, code))
    assert result['status'] == 'ok'
    assert any(item['rule_id'] == 'B605' for item in result['analysis']['findings'])
    assert not marker.exists()
    assert str(marker) not in json.dumps(result)
    assert context[4].args_schema.model_json_schema()['properties'].keys() == {'evidence_id'}


def test_no_flags_is_a_bounded_static_observation_not_verified_security(context):
    source = register(context, 'def add(left, right):\n    return left + right\n')
    result = invoke(context, source)
    assert result['status'] == 'ok' and result['analysis']['finding_count'] == 0
    assert 'No flags does not establish security or correctness' in result['evidence'][0]['text']
    again = invoke(context, source)
    assert again['analysis']['fingerprint'] == result['analysis']['fingerprint']
    assert again['evidence'][0]['evidence_id'] == result['evidence'][0]['evidence_id']


def test_syntax_error_is_recorded_without_a_false_clean_result(context):
    result = invoke(context, register(context, 'def incomplete(:\n    pass\n'))
    assert result['status'] == 'error' and result['error_code'] == 'syntax_error'
    assert result['analysis']['finding_count'] is None
    assert result['evidence'][0]['origin'] == 'analysis'
    assert 'Checks did not complete' in result['evidence'][0]['text']
    assert context[0].events(context[1])[-1]['error_code'] == 'syntax_error'


@pytest.mark.parametrize('header_kind', ['repository', 'supplied', 'cli'])
def test_header_line_mapping_preserves_declared_file_and_parent_evidence_lines(context, header_kind):
    code = 'import subprocess\nsubprocess.run(command, shell=True)\npass\n'
    digest = hashlib.sha256(code.encode()).hexdigest()
    if header_kind == 'repository':
        header = f'Repository snapshot: triage_desk/helper.py:50-52\nFile SHA-256: {"a" * 64}\nExcerpt SHA-256: {digest}\n\n'
        metadata, start, removed = {'origin': 'repository', 'truncated': True}, 50, 4
    elif header_kind == 'supplied':
        header = f'Supplied repository file: triage_desk/helper.py:50-52\nFile SHA-256: {"a" * 64}\n\n'
        metadata, start, removed = {}, 50, 3
    else:
        header, metadata, start, removed = '# Supplied file: helper.py\n', {}, 1, 1
    result = invoke(context, register(context, header + code, **metadata))
    assert result['status'] == 'ok'
    finding = result['analysis']['findings'][0]
    assert finding['line'] == start + 1 and finding['evidence_line'] == removed + 2
    assert result['analysis']['source_start_line'] == start
    assert result['analysis']['filename'].endswith('helper.py')
    assert result['analysis']['code_sha256'] == digest


@pytest.mark.parametrize('kind,origin', [('documentation', 'snapshot'), ('code', 'live'), ('code', 'analysis')])
def test_only_registered_user_or_repository_code_is_eligible(context, kind, origin):
    source = register(context, 'eval(text)\n', kind=kind, origin=origin,
                      **({'url': 'https://docs.streamlit.io/'} if origin == 'live' else {}))
    result = invoke(context, source)
    assert result['status'] == 'error' and result['error_code'] == 'source_not_code'
    assert not result['evidence']


def test_unknown_id_cannot_be_used_as_a_host_path(context):
    result = context[4].invoke({'evidence_id': '../../.env'})
    assert result['error_code'] == 'unknown_evidence' and not result['evidence']


def test_oversized_sources_are_rejected_before_starting_worker(context, monkeypatch):
    source = register(context, '# comment\n' * 2000)
    monkeypatch.setattr(static_analysis, '_run_worker', lambda code: pytest.fail('Source limit must precede worker launch'))
    result = invoke(context, source)
    assert result['error_code'] == 'source_limit'


@pytest.mark.parametrize('problem', ['mismatched_hash', 'invalid_path', 'wrong_bounds'])
def test_untrusted_provenance_headers_cannot_forge_hashes_or_line_bounds(context, problem):
    code = 'eval(text)\n'
    filename = '../../private.py' if problem == 'invalid_path' else 'triage_desk/example.py'
    digest = 'b' * 64 if problem == 'mismatched_hash' else hashlib.sha256(code.encode()).hexdigest()
    end = 12 if problem == 'wrong_bounds' else 10
    header = f'Repository snapshot: {filename}:10-{end}\nFile SHA-256: {"a" * 64}\nExcerpt SHA-256: {digest}\n\n'
    result = invoke(context, register(context, header + code, origin='repository'))
    assert result['error_code'] == 'invalid_header' and not result['evidence']


def test_nosec_and_local_configuration_cannot_disable_fixed_checks(context, tmp_path, monkeypatch):
    (tmp_path / '.bandit').write_text('[bandit]\nskips=B307\n')
    (tmp_path / 'bandit.py').write_text('raise RuntimeError("UNTRUSTED MODULE LOADED")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    result = invoke(context, register(context, 'eval(text)  # nosec B307\n'))
    assert result['status'] == 'ok'
    assert [item['rule_id'] for item in result['analysis']['findings']] == ['B307']


def test_worker_invocation_is_fixed_isolated_and_does_not_inherit_credentials(context, monkeypatch):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'PRIVATE CREDENTIAL')
    def run(arguments, **options):
        assert arguments[:2] == [sys.executable, '-I']
        assert Path(arguments[2]).name == '_static_analysis_worker.py'
        assert options['shell'] is False and options['timeout'] == static_analysis.WORKER_TIMEOUT_SECONDS
        assert 'FIREWORKS_API_KEY' not in options['env'] and 'PYTHONPATH' not in options['env']
        assert set(Path(options['cwd']).iterdir()) == {Path(options['cwd']) / 'result.json'}
        assert json.loads(options['input']) == {'code': 'value = 1\n'}
        options['stdout'].write(json.dumps({'version': '1.9.4', 'python_version': '3.12.14', 'status': 'ok',
                                          'findings': [], 'finding_count': 0, 'truncated': False}).encode())
        return subprocess.CompletedProcess(arguments, 0)
    monkeypatch.setattr(static_analysis.subprocess, 'run', run)
    assert invoke(context, register(context, 'value = 1\n'))['status'] == 'ok'


@pytest.mark.parametrize('failure', ['timeout', 'output_limit', 'invalid_output'])
def test_worker_failures_are_bounded_and_do_not_expose_output(context, monkeypatch, failure):
    def run(arguments, **options):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(['PRIVATE COMMAND'], 6, output=b'PRIVATE OUTPUT')
        if failure == 'output_limit':
            options['stdout'].write(b'X' * (static_analysis.MAX_OUTPUT_BYTES + 1))
        else:
            options['stdout'].write(json.dumps({'version': '1.9.4', 'python_version': '3.12.14', 'status': 'ok',
                                              'findings': [], 'finding_count': 0, 'truncated': False,
                                              'raw_output': 'PRIVATE OUTPUT'}).encode())
        return subprocess.CompletedProcess(arguments, 0)
    monkeypatch.setattr(static_analysis.subprocess, 'run', run)
    result = invoke(context, register(context, 'value = 1\n'))
    assert result['status'] == 'error' and result['error_code'] == failure
    assert 'PRIVATE' not in json.dumps(result) and not result['evidence']


def test_complexity_and_finding_limits_are_explicit(context):
    result = invoke(context, register(context, 'value = 1\n' * 1500))
    assert result['error_code'] == 'complexity_limit' and result['analysis']['finding_count'] is None
    many = invoke(context, register(context, 'eval(text)\n' * 65))
    assert many['analysis']['finding_count'] == 65
    assert len(many['analysis']['findings']) == 50 and many['analysis']['truncated'] is True


def test_budget_exhaustion_prevents_worker_launch(context, monkeypatch):
    source = register(context, 'eval(text)\n')
    budget = InvestigationBudget(context[0], context[1], max_tool_calls=0)
    tool = static_analysis.build_static_analysis_tools(context[2], budget)[0]
    monkeypatch.setattr(static_analysis, '_run_worker', lambda code: pytest.fail('No launch after budget exhaustion'))
    with pytest.raises(BudgetExceeded):
        tool.invoke({'evidence_id': source['evidence_id']})
