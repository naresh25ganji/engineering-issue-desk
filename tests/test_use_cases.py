"""Workflow versioning, persistence, and trusted prompt boundaries."""
from dataclasses import FrozenInstanceError
import sqlite3

import pytest

from triage_desk.cases import CaseStore
from triage_desk.deep_investigation import BOUNDARIES, ORCHESTRATOR_PROMPT, build_deep_investigator
from triage_desk.use_cases import USE_CASES, get_use_case


def test_workflow_registry_is_immutable_and_rejects_unknown_saved_versions():
    assert tuple(profile.id for profile in USE_CASES) == (
        'issue_investigation', 'incident_triage', 'change_review', 'test_planning')
    for profile in USE_CASES:
        assert get_use_case(profile.id, 1) is profile
        assert profile.examples and profile.required_details and profile.success_criteria
        assert isinstance(profile.examples, tuple)
        with pytest.raises(FrozenInstanceError):
            profile.instructions = 'Replace trusted workflow'
    for identifier, version in [('unknown', None), ('incident_triage', 2),
                                 ('change_review', True), ('test_planning', '1'), (None, None)]:
        with pytest.raises(ValueError):
            get_use_case(identifier, version)


def test_case_workflow_is_validated_immutable_and_survives_reopening(tmp_path):
    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    incident = store.create_case('Triage supplied logs', use_case_id='incident_triage')
    reopened = CaseStore(path).get_case(incident['case_id'])
    assert (reopened['use_case_id'], reopened['use_case_version']) == ('incident_triage', 1)
    for updates in ({'use_case_id': 'change_review'}, {'use_case_version': 2}):
        with pytest.raises(ValueError, match='Cannot update'):
            store.update_case(incident['case_id'], **updates)
    for options in ({'use_case_id': 'unknown'}, {'use_case_version': 2}, {'use_case_version': True}):
        with pytest.raises(ValueError):
            store.create_case('Invalid workflow must never be saved', **options)
    assert len(store.list_cases()) == 1


def test_existing_case_migration_defaults_to_original_workflow_and_preserves_review(tmp_path):
    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    case = store.create_case('Original issue investigation', code_context='original code')
    store.update_case(case['case_id'], review_notes='Saved review', report={'findings': []})
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE cases DROP COLUMN use_case_id')
        connection.execute('ALTER TABLE cases DROP COLUMN use_case_version')
    migrated = CaseStore(path).get_case(case['case_id'])
    assert migrated['use_case_id'] == 'issue_investigation' and migrated['use_case_version'] == 1
    assert migrated['code_context'] == 'original code'
    assert migrated['review_notes'] == 'Saved review' and migrated['report'] == {'findings': []}
    assert CaseStore(path).get_case(case['case_id']) == migrated


def test_workflow_guidance_reaches_orchestrator_and_every_specialist_without_changing_default(monkeypatch):
    import deepagents

    captured = []
    monkeypatch.setattr(deepagents, 'create_deep_agent', lambda **kwargs: captured.append(kwargs))
    build_deep_investigator(None, [], None, None)
    assert captured[-1]['system_prompt'] == ORCHESTRATOR_PROMPT
    defaults = [specialist['system_prompt'] for specialist in captured[-1]['subagents']]
    for profile in USE_CASES[1:]:
        build_deep_investigator(None, [], None, None, use_case=profile.id)
        assert profile.instructions in captured[-1]['system_prompt']
        assert captured[-1]['system_prompt'].startswith(ORCHESTRATOR_PROMPT)
        for original, specialist in zip(defaults, captured[-1]['subagents'], strict=True):
            assert specialist['system_prompt'].startswith(original)
            assert specialist['system_prompt'].startswith(BOUNDARIES)
            assert profile.instructions in specialist['system_prompt']
    with pytest.raises(ValueError):
        build_deep_investigator(None, [], None, None, use_case='unrecognized')
