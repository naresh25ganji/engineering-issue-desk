"""Tool-backed playbooks and durable reviewed-case guidance, without providers."""
import hashlib
import json
from pathlib import Path

import pytest

from triage_desk.agent_budget import BudgetExceeded, InvestigationBudget
from triage_desk.cases import CaseStore
from triage_desk import review_extensions as extensions


@pytest.fixture
def case_context(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case = store.create_case('Current widget investigation', version='1.50')
    return store, case['case_id']


def tools_for(case_context, **kwargs):
    store, case_id = case_context
    budget = InvestigationBudget(store, case_id)
    tools = {tool.name: tool for tool in extensions.build_review_extension_tools(store, case_id, budget, **kwargs)}
    return tools, budget


def reviewed_case(store, *, statement='Widget state needs explicit handling.', notes='Confirmed the widget reproduction.',
                  status='approved', version='1.50', **extra_report):
    case = store.create_case('PRIVATE QUESTION SHOULD NOT LEAVE STORAGE', version=version,
                             code_context='PRIVATE RAW CODE SHOULD NOT LEAVE STORAGE')
    store.update_case(case['case_id'], status=status, review_notes=notes, report={
        'findings': [{'statement': statement, 'category': 'docs', 'severity': 'low',
                      'support': [{'quote': 'PRIVATE SOURCE QUOTE SHOULD NOT LEAVE STORAGE', 'evidence_id': 'source-id'}]}],
        'sources': [{'text': 'PRIVATE FULL SOURCE BODY SHOULD NOT LEAVE STORAGE'}],
        'next_steps': ['PRIVATE NEXT STEP SHOULD NOT LEAVE STORAGE'],
        **extra_report,
    })
    return case['case_id']


def test_playbooks_are_available_but_history_requires_explicit_opt_in(case_context):
    tools, budget = tools_for(case_context)
    assert set(tools) == {'list_review_playbooks', 'load_review_playbook'}
    enabled, _ = tools_for(case_context, history_enabled=True)
    assert set(enabled) == {'list_review_playbooks', 'load_review_playbook', 'search_review_history'}
    with pytest.raises(extensions.ReviewExtensionUnavailable):
        tools_for(case_context, history_enabled='true')


def test_maintained_playbooks_are_versioned_on_demand_and_hash_matches_file(case_context):
    tools, budget = tools_for(case_context)
    catalog = tools['list_review_playbooks'].invoke({})
    assert catalog['status'] == 'ok'
    assert catalog['delivery'] == 'on_demand_tool'
    assert {item['name'] for item in catalog['playbooks']} == {
        'python-security', 'python-testing', 'evidence-causality',
    }
    assert all('content' not in item for item in catalog['playbooks'])
    for item in catalog['playbooks']:
        loaded = tools['load_review_playbook'].invoke({'name': item['name']})
        playbook = loaded['playbook']
        assert loaded['status'] == 'ok'
        assert playbook['version'] == '1.0.0' == item['version']
        assert playbook['origin'] == 'maintainer_bundle'
        assert playbook['delivery'] == 'on_demand_tool'
        assert playbook['citable'] is False
        assert len(playbook['content'].encode()) <= extensions.MAX_PLAYBOOK_BYTES
        source = extensions._BUNDLE_ROOT / item['name'] / 'SKILL.md'
        assert playbook['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert loaded['evidence'] == []
    assert budget.tool_calls == 4
    store, case_id = case_context
    assert store.evidence(case_id) == []
    assert len([event for event in store.events(case_id) if event['phase'] == 'tool']) == 4


@pytest.mark.parametrize('name', ['../.env', '/etc/passwd', 'python-security/../../.env',
                                  'python-security/SKILL.md', 'https://evil.test', 'Python-Security', 'unknown'])
def test_model_cannot_supply_playbook_paths_or_unknown_names(case_context, name):
    tools, budget = tools_for(case_context)
    result = tools['load_review_playbook'].invoke({'name': name})
    assert result['status'] == 'error'
    assert result['evidence'] == []
    assert name not in result['error']


def test_playbook_loader_refuses_symlink_bundles_and_oversized_content(case_context, tmp_path, monkeypatch):
    bundle = tmp_path / 'bundles'
    bundle.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'SKILL.md').write_text('PRIVATE OUTSIDE CONTENT', encoding='utf-8')
    (bundle / 'python-security').symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(extensions, '_BUNDLE_ROOT', bundle)
    tools, budget = tools_for(case_context)
    result = tools['load_review_playbook'].invoke({'name': 'python-security'})
    assert result['status'] == 'error'
    assert 'PRIVATE' not in json.dumps(result)
    (bundle / 'python-security').unlink()
    (bundle / 'python-security').mkdir()
    (bundle / 'python-security' / 'SKILL.md').write_text('x' * (extensions.MAX_PLAYBOOK_BYTES + 1), encoding='utf-8')
    result = tools['load_review_playbook'].invoke({'name': 'python-security'})
    assert result['status'] == 'error'


def test_playbook_identity_must_match_maintained_catalog(case_context, tmp_path, monkeypatch):
    bundle = tmp_path / 'bundles'
    target = bundle / 'python-security'
    target.mkdir(parents=True)
    (target / 'SKILL.md').write_text('---\nname: wrong-name\nversion: 9.0.0\n---\nContent\n', encoding='utf-8')
    monkeypatch.setattr(extensions, '_BUNDLE_ROOT', bundle)
    tools, _ = tools_for(case_context)
    assert tools['load_review_playbook'].invoke({'name': 'python-security'})['status'] == 'error'


def test_history_reads_only_same_store_terminal_cases_and_excludes_current(case_context, tmp_path):
    store, current_id = case_context
    approved = reviewed_case(store, status='approved')
    rejected = reviewed_case(store, status='rejected', notes='Rejected the widget fix; investigate its prerequisites.')
    for status in ('new', 'running', 'awaiting_review', 'needs_revision', 'failed'):
        reviewed_case(store, status=status)
    # Even a terminal current case is not its own historical guidance.
    store.update_case(current_id, status='approved', review_notes='Current widget notes')
    other = CaseStore(tmp_path / 'other.sqlite')
    foreign = reviewed_case(other, notes='Foreign widget notes must stay in their own store.')
    tools, _ = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert result['status'] == 'ok'
    assert {item['case_id'] for item in result['historical_guidance']} == {approved, rejected}
    assert foreign not in json.dumps(result)
    assert current_id not in json.dumps(result)
    assert result['candidates_considered'] == 2
    assert result['scope'] == 'same_local_case_store'
    assert result['retrieval'] == 'literal_keywords'
    assert all(item['kind'] == 'historical_guidance' and item['citable'] is False
               for item in result['historical_guidance'])
    assert 'not factual ground truth' in result['notes']


def test_history_projects_notes_and_summaries_never_code_questions_or_supports(case_context):
    store, case_id = case_context
    historical_id = reviewed_case(store)
    tools, budget = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': 'widget'})
    guidance = result['historical_guidance'][0]
    assert guidance['case_id'] == historical_id
    assert guidance['review_notes'] == 'Confirmed the widget reproduction.'
    assert guidance['finding_summaries'] == ['Widget state needs explicit handling.']
    assert guidance['updated_at'] == store.get_case(historical_id)['updated_at']
    assert 'PRIVATE' not in json.dumps(result)
    assert result['evidence'] == []
    assert store.evidence(case_id) == []
    assert store.evidence(historical_id) == []
    assert budget.tool_calls == 1


def test_reviewed_history_survives_store_restart(case_context):
    store, case_id = case_context
    historical = reviewed_case(store)
    reopened = CaseStore(store.path)
    tools, _ = tools_for((reopened, case_id), history_enabled=True)
    assert tools['search_review_history'].invoke({'query': 'widget'})['historical_guidance'][0]['case_id'] == historical


@pytest.mark.parametrize('old_version,current_version,expected', [('1.49', '1.50', False), ('1.50', '1.50', True),
                                                               ('', '1.50', None)])
def test_version_labels_are_compared_without_implying_compatibility(case_context, old_version, current_version, expected):
    reviewed_case(case_context[0], version=old_version)
    tools, _ = tools_for(case_context, history_enabled=True, version=current_version)
    item = tools['search_review_history'].invoke({'query': 'widget'})['historical_guidance'][0]
    assert item['version_match'] is expected
    assert item['warning']
    if expected is False:
        assert 'Different version labels' in item['warning']
    elif expected is True:
        assert 'do not establish' in item['warning']
    else:
        assert 'unavailable' in item['warning']


def test_history_results_and_fields_are_bounded(case_context):
    store, _ = case_context
    for _ in range(8):
        reviewed_case(store, notes='Widget ' + 'plain review prose ' * 100,
                      findings=[{'statement': 'Widget ' + 'observed behavior ' * 100} for _ in range(10)])
    tools, _ = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert len(result['historical_guidance']) == 3
    assert result['truncated']
    for item in result['historical_guidance']:
        assert len(item['review_notes']) <= extensions.MAX_REVIEW_NOTES_CHARS
        assert len(item['finding_summaries']) <= extensions.MAX_FINDING_SUMMARIES
        assert all(len(summary) <= extensions.MAX_FINDING_CHARS for summary in item['finding_summaries'])


def test_history_candidate_limit_is_explicit(case_context, monkeypatch):
    for _ in range(6):
        reviewed_case(case_context[0])
    monkeypatch.setattr(extensions, 'MAX_HISTORY_CANDIDATES', 4)
    tools, _ = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert result['candidates_considered'] == 4
    assert result['truncated']


def test_literal_keywords_do_not_search_raw_code_or_invent_semantic_matches(case_context):
    store, _ = case_context
    reviewed_case(store, notes='Widget and state observation.', statement='Literal .* appears in this widget summary.')
    tools, _ = tools_for(case_context, history_enabled=True)
    assert tools['search_review_history'].invoke({'query': 'WIDGET state'})['historical_guidance']
    assert tools['search_review_history'].invoke({'query': '.*'})['historical_guidance']
    assert tools['search_review_history'].invoke({'query': 'memoization'})['historical_guidance'] == []
    assert tools['search_review_history'].invoke({'query': 'PRIVATE RAW CODE'})['historical_guidance'] == []


@pytest.mark.parametrize('query', [' ', 'one two three four five six seven eight nine', 'widget\nstate'])
def test_unsafe_or_oversized_history_queries_are_rejected_without_echo(case_context, query):
    tools, _ = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': query})
    assert result['status'] == 'error'
    assert result['evidence'] == []


def test_configured_secrets_are_checked_before_truncation_and_not_audited(case_context):
    store, case_id = case_context
    secret = 'configured-private-key-123456789'
    # The secret crosses a prospective output truncation boundary. Withholding
    # the complete projected field prevents leaking a partial credential prefix.
    notes = 'Widget ' + 'x' * (extensions.MAX_REVIEW_NOTES_CHARS - 10) + secret
    reviewed_case(store, notes=notes)
    good = reviewed_case(store, notes='A safe widget review.')
    tools, _ = tools_for(case_context, history_enabled=True, forbidden_values=[secret])
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert [item['case_id'] for item in result['historical_guidance']] == [good]
    assert secret not in json.dumps(result)
    assert secret[:12] not in json.dumps(result)
    denied = tools['search_review_history'].invoke({'query': secret})
    assert denied['status'] == 'error'
    assert secret not in json.dumps(denied)
    assert secret not in json.dumps(store.events(case_id))


def test_common_token_patterns_and_pasted_code_are_not_replayed(case_context):
    store, _ = case_context
    reviewed_case(store, notes='Widget token ghp_Abcdef123456789012345678901234567890')
    reviewed_case(store, notes='```python\nprivate_source = "widget"\n```', statement='def widget_source(): return "raw code"')
    good = reviewed_case(store)
    tools, _ = tools_for(case_context, history_enabled=True)
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert [item['case_id'] for item in result['historical_guidance']] == [good]
    assert 'ghp_' not in json.dumps(result)
    assert 'private_source' not in json.dumps(result)
    assert 'def widget_source' not in json.dumps(result)


def test_secret_values_in_excluded_source_bodies_are_never_loaded_or_returned(case_context):
    store, _ = case_context
    secret = 'source-only-secret-1234567890'
    historical = reviewed_case(store, sources=[{'text': secret}], next_steps=[secret])
    tools, _ = tools_for(case_context, history_enabled=True, forbidden_values=[secret])
    result = tools['search_review_history'].invoke({'query': 'widget'})
    assert [item['case_id'] for item in result['historical_guidance']] == [historical]
    assert secret not in json.dumps(result)


def test_playbook_content_with_configured_secret_is_withheld(case_context, tmp_path, monkeypatch):
    secret = 'configured-private-playbook-key-123456'
    bundle = tmp_path / 'bundles'
    folder = bundle / 'python-security'
    folder.mkdir(parents=True)
    (folder / 'SKILL.md').write_text(f'---\nname: python-security\nversion: 1.0.0\n---\n{secret}\n', encoding='utf-8')
    monkeypatch.setattr(extensions, '_BUNDLE_ROOT', bundle)
    tools, _ = tools_for(case_context, forbidden_values=[secret])
    result = tools['load_review_playbook'].invoke({'name': 'python-security'})
    assert result['status'] == 'error'
    assert secret not in json.dumps(result)


def test_shared_budget_blocks_further_history_or_bundle_reads(case_context):
    tools, budget = tools_for(case_context, history_enabled=True)
    budget.max_tool_calls = 1
    assert tools['list_review_playbooks'].invoke({})['status'] == 'ok'
    with pytest.raises(BudgetExceeded):
        tools['search_review_history'].invoke({'query': 'widget'})
    with pytest.raises(BudgetExceeded):
        tools['load_review_playbook'].invoke({'name': 'python-security'})


def test_unknown_current_case_cannot_borrow_another_stores_history(case_context, tmp_path):
    other = CaseStore(tmp_path / 'other.sqlite')
    with pytest.raises(KeyError):
        extensions.build_review_extension_tools(other, case_context[1], InvestigationBudget(other, other.create_case('Other')['case_id']))
