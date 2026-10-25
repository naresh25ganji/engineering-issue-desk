"""Run the real Deep Agents tool loop and durable review graph entirely offline."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from pydantic import BaseModel
import pytest

from triage_desk.deep_investigation import AgentSettings, InvestigationService, checkpoint_saver


PASSAGE = 'Streamlit reruns the script from top to bottom after a widget interaction.'


class SnapshotDesk:
    """The real KB tool registers these passages, rather than a mocked tool reply."""
    def __init__(self):
        self.queries = []

    def search(self, query, *, mode, top_k):
        self.queries.append((query, mode, top_k))
        return [{
            'chunk_id': 'doc:reruns:0', 'source_id': 'doc:reruns',
            'title': 'Execution flow', 'text': PASSAGE,
            'url': 'https://docs.streamlit.io/develop/concepts/architecture',
            'kind': 'documentation', 'snapshot_at': '2026-10-20T00:00:00Z',
            'version_scope': 'Streamlit 1.50',
        }]


class ScriptedToolModel(BaseChatModel):
    """A provider substitute that still binds tools and runs LangChain messages."""
    replies: list[Any]
    calls: list[list[Any]] = Field(default_factory=list)
    bound_schemas: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'offline-scripted-tool-model'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_schemas.extend(convert_to_openai_tool(tool) for tool in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError('The graph made an unexpected additional model call')
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, str):
            result = AIMessage(content=reply)
        else:
            result = AIMessage(content='', tool_calls=reply)
        result.usage_metadata = {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5}
        return ChatResult(generations=[ChatGeneration(message=result)])


def tool_call(name, args, call_id='call-1'):
    return [{'name': name, 'args': args, 'id': call_id, 'type': 'tool_call'}]


def report_payload(evidence_id, quote_id=None, statement='Widget interactions rerun the script.'):
    return {
        'findings': [{'statement': statement, 'category': 'docs', 'severity': 'low',
                      'support': [{'evidence_id': evidence_id,
                                   'quote_id': quote_id or evidence_id + ':q1'}]}],
        'next_steps': ['Preserve uploaded data in session state when appropriate.'],
        'missing_information': ['Share the minimal reproduction and installed version.'],
    }


def source_from_messages(messages):
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == 'search_knowledge_base':
            return json.loads(message.content)['evidence'][0]
    raise AssertionError('The model must receive the real registered KB evidence')


def cite_kb(messages):
    source = source_from_messages(messages)
    return tool_call('InvestigationReport', report_payload(
        source['evidence_id'], source['quote_spans'][0]['quote_id']), 'report-1')


def read_registered_evidence(messages):
    request = next(json.loads(message.content) for message in messages if isinstance(message, HumanMessage))
    return tool_call('read_case_evidence', {'evidence_id': request['registered_evidence_ids'][0]}, 'repair-read')


def cite_read_evidence(messages):
    source = json.loads(next(message.content for message in reversed(messages)
                             if isinstance(message, ToolMessage) and message.name == 'read_case_evidence'))
    return tool_call('InvestigationReport', report_payload(
        source['evidence_id'], source['quote_spans'][0]['quote_id']), 'repaired-report')


def settings(**overrides):
    return AgentSettings(api_key='offline-test-key', model='offline-tool-model', **overrides)


def service(root: Path, model: ScriptedToolModel, *, desk=None, agent_settings=None):
    def checked(report, reviewer_model, budget):
        # The independent reviewer has its own strict provider-contract tests.
        # These tests focus on the real investigator/checkpoint/review graph.
        return {'report': {key: report[key] for key in ('findings', 'next_steps', 'missing_information')},
                'quality_review': {'status': 'completed', 'removed_findings': 0, 'removed_next_steps': 0}}
    return InvestigationService(root, desk or SnapshotDesk(), settings=agent_settings or settings(),
                                model_factory=lambda _: model, reviewer=checked)


def verdicts(keep_step=False):
    return tool_call('EvidenceVerdicts', {
        'finding_verdicts': [{'index': 0, 'supported': True, 'reason': 'The selected quote states that widget interactions rerun the script.'}],
        'next_step_verdicts': [{'index': 0, 'supported': keep_step,
                               'reason': 'The selected quote does not establish persistence of uploaded data.'}],
    }, 'evidence-verdicts')


def test_real_evidence_review_filters_before_human_checkpoint_and_counts_its_call(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}),
                                      cite_kb, verdicts()])
    engine = InvestigationService(tmp_path, SnapshotDesk(), settings(), model_factory=lambda _: model)
    pending = engine.start('Explain widget reruns using documentation.')
    assert pending['status'] == 'awaiting_review', engine.store.events(pending['case_id'])
    assert len(pending['report']['findings']) == 1
    assert pending['report']['next_steps'] == []
    assert pending['report']['quality_review']['removed_next_steps'] == 1
    assert len(model.calls) == 3
    calls = [e['name'] for e in engine.store.events(pending['case_id']) if e['phase'] == 'model']
    assert calls == ['investigator', 'investigator', 'evidence_reviewer']


def test_failed_evidence_review_retries_its_checkpoint_without_repeating_investigation(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb,
                                      RuntimeError('Private provider failure'), verdicts()])
    desk = SnapshotDesk()
    engine = InvestigationService(tmp_path, desk, settings(), model_factory=lambda _: model)
    failed = engine.start('Explain widget reruns.')
    assert failed['status'] == 'failed'
    assert 'independent evidence review' in failed['error']
    assert 'Private provider failure' not in failed['error']
    with pytest.raises(ValueError):
        engine.resume(failed['case_id'], 'approve')
    restarted = InvestigationService(tmp_path, desk, settings(), model_factory=lambda _: model)
    pending = restarted.retry(failed['case_id'])
    assert pending['status'] == 'awaiting_review', restarted.store.events(failed['case_id'])
    assert len(desk.queries) == 1 and len(model.calls) == 4
    assert pending['report']['quality_review']['status'] == 'completed'


def test_repository_tools_are_absent_without_explicit_case_opt_in(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb])
    pending = service(tmp_path, model).start('Explain widget reruns.')
    assert pending['repository_context'] is False
    names = {schema['function']['name'] for schema in model.bound_schemas}
    assert not names & {'list_repository_files', 'search_repository', 'read_repository_file'}


def test_opted_in_repository_read_is_cited_counted_once_and_resumed_after_failure(tmp_path):
    path = tmp_path / 'triage_desk' / 'configuration.py'
    path.parent.mkdir()
    path.write_text('def require_key(key):\n    if not key:\n        raise ValueError("Missing key")\n    return key\n')

    def cite_repository(messages):
        response = json.loads(next(message.content for message in reversed(messages)
                                   if isinstance(message, ToolMessage) and message.name == 'read_repository_file'))
        source = response['evidence'][0]
        return tool_call('InvestigationReport', report_payload(source['evidence_id'], source['quote_spans'][0]['quote_id'],
                         statement='The supplied require_key helper rejects an empty key.'))

    model = ScriptedToolModel(replies=[tool_call('read_repository_file', {'path': 'triage_desk/configuration.py'}),
                                      RuntimeError('simulated private provider failure'), cite_repository])
    first = service(tmp_path, model)
    failed = first.start('Inspect the key helper.', repository_context=True)
    assert failed['status'] == 'failed' and failed['repository_context'] is True
    restarted = service(tmp_path, model)
    pending = restarted.retry(failed['case_id'])
    assert pending['status'] == 'awaiting_review' and pending['repository_context'] is True
    sources = restarted.store.evidence(pending['case_id'])
    assert len(sources) == 1 and sources[0]['origin'] == 'repository'
    assert sources[0]['filename'] == 'triage_desk/configuration.py'
    assert 'raise ValueError("Missing key")' in pending['report']['findings'][0]['support'][0]['quote']
    reads = [event for event in restarted.store.events(pending['case_id'])
             if event.get('phase') == 'tool' and event.get('name') == 'read_repository_file']
    assert len(reads) == 1, 'Repository tools must not be double counted or repeated after a checkpointed read'
    assert len(model.calls) == 3


def test_review_failure_has_finite_safe_diagnostic_code_in_saved_events(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb,
        tool_call('EvidenceVerdicts', {'finding_verdicts': [], 'next_step_verdicts': []})])
    engine = InvestigationService(tmp_path, SnapshotDesk(), settings(), model_factory=lambda _: model)
    failed = engine.start('Explain widget reruns.')
    assert failed['status'] == 'failed'
    error = next(event for event in engine.store.events(failed['case_id']) if event.get('phase') == 'execution')
    assert error['error_code'] == 'incomplete_coverage'
    assert error['error_type'] == 'ReviewUnavailable'


def test_legacy_saved_draft_can_be_audited_without_repeating_its_research(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb,
                                      verdicts()])
    desk = SnapshotDesk()
    original = service(tmp_path, model, desk=desk)
    pending = original.start('Explain widget reruns.')
    legacy_report = dict(pending['report'])
    legacy_report.pop('quality_review')
    original.store.update_case(pending['case_id'], report=legacy_report)
    upgraded = InvestigationService(tmp_path, desk, settings(), model_factory=lambda _: model)
    with pytest.raises(ValueError, match='not passed evidence review'):
        upgraded.resume(pending['case_id'], 'approve')
    reviewed = upgraded.audit(pending['case_id'])
    assert reviewed['status'] == 'awaiting_review', upgraded.store.events(pending['case_id'])
    assert reviewed['report']['quality_review']['removed_next_steps'] == 1
    assert len(desk.queries) == 1 and len(model.calls) == 3


def test_evidence_reviewer_cannot_bypass_case_model_allowance(tmp_path):
    model = ScriptedToolModel(replies=[tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb,
                                      verdicts()])
    engine = InvestigationService(tmp_path, SnapshotDesk(), settings(max_model_calls=2), model_factory=lambda _: model)
    failed = engine.start('Explain widget reruns.')
    assert failed['status'] == 'failed'
    assert len(model.calls) == 2
    assert 'quality_review' not in failed['report']


def test_real_deep_agent_searches_kb_then_pauses_and_restart_approves_without_model(tmp_path):
    desk = SnapshotDesk()
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'file uploader resets'}), cite_kb,
    ])
    first = service(tmp_path, model, desk=desk)
    case = first.start('The uploaded CSV resets when I click a widget.', version='1.50')
    assert case['status'] == 'awaiting_review', first.store.events(case['case_id'])
    assert desk.queries == [('file uploader resets', 'keyword', 5)]
    assert len(model.calls) == 2
    evidence = first.store.evidence(case['case_id'])
    offered_source = source_from_messages(model.calls[1])
    assert case['report']['findings'][0]['support'][0] == {
        'evidence_id': evidence[0]['evidence_id'],
        'quote_id': offered_source['quote_spans'][0]['quote_id'], 'quote': PASSAGE,
    }
    assert evidence[0]['origin'] == 'snapshot'
    assert evidence[0]['sha256']
    names = {schema['function']['name'] for schema in model.bound_schemas}
    assert {'InvestigationReport', 'search_knowledge_base', 'task', 'write_todos'} <= names
    report_schema = next(schema['function']['parameters'] for schema in model.bound_schemas
                         if schema['function']['name'] == 'InvestigationReport')
    assert set(report_schema['required']) == {'findings', 'next_steps', 'missing_information'}
    assert 'quote_id' in json.dumps(report_schema)
    assert '"quote"' not in json.dumps(report_schema), 'The provider selects spans instead of rewriting quotes'

    def no_provider(_):
        raise AssertionError('Human approval must never create a model')

    restarted = InvestigationService(tmp_path, desk, settings=AgentSettings(), model_factory=no_provider)
    approved = restarted.resume(case['case_id'], 'approve', notes='Confirmed by the engineer.')
    assert approved['status'] == 'approved'
    assert approved['review_notes'] == 'Confirmed by the engineer.'
    assert approved['report'] == case['report']
    assert len(model.calls) == 2
    assert len(restarted.store.list_cases()) == 1
    with checkpoint_saver(first.directory / 'checkpoints.sqlite') as saver:
        payloads = [saved.checkpoint['channel_values']['structured_response'] for saved in saver.list(None)
                    if saved.checkpoint.get('channel_values', {}).get('structured_response') is not None]
    assert payloads
    assert all(isinstance(payload, dict) and not isinstance(payload, BaseModel) for payload in payloads)
    assert all(isinstance(payload['findings'][0], dict) for payload in payloads)


def test_revision_starts_a_new_agent_iteration_and_carries_reviewer_feedback(tmp_path):
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'reruns'}), cite_kb,
        tool_call('search_knowledge_base', {'query': 'widget rerun details'}, 'revision-search'), cite_kb,
    ])
    engine = service(tmp_path, model)
    case = engine.start('A widget interaction resets uploaded data.')
    assert case['status'] == 'awaiting_review'
    revised = service(tmp_path, model).resume(case['case_id'], 'revise', notes='Check widget identity changes.')
    assert revised['status'] == 'awaiting_review', engine.store.events(case['case_id'])
    assert revised['review_notes'] == 'Check widget identity changes.'
    assert len(model.calls) == 4
    revision_requests = [json.loads(message.content) for message in model.calls[2]
                         if isinstance(message, HumanMessage)]
    assert any(request.get('revision') == 1 and request.get('reviewer_feedback') == 'Check widget identity changes.'
               for request in revision_requests)
    assert any(request.get('previous_report') == case['report'] for request in revision_requests), (
        'A fresh revision thread needs the report that the human reviewed'
    )
    # Re-registering an identical snapshot keeps the same immutable evidence ID.
    assert len(engine.store.evidence(case['case_id'])) == 1
    assert len(engine.store.list_cases()) == 1


def test_specialist_registers_evidence_that_orchestrator_can_read_and_cite(tmp_path):
    import re

    def specialist_result(messages):
        source = source_from_messages(messages)
        return json.dumps({'evidence_id': source['evidence_id'],
                           'quote_id': source['quote_spans'][0]['quote_id']})

    def read_specialist_evidence(messages):
        returned = next(message.content for message in reversed(messages)
                        if isinstance(message, ToolMessage) and message.name == 'task')
        evidence_id = re.search(r'ev_[a-f0-9]{32}', returned).group()
        return tool_call('read_case_evidence', {'evidence_id': evidence_id}, 'shared-evidence')

    def cite_shared_source(messages):
        source = json.loads(next(message.content for message in reversed(messages)
                                 if isinstance(message, ToolMessage) and message.name == 'read_case_evidence'))
        return tool_call('InvestigationReport', report_payload(
            source['evidence_id'], source['quote_spans'][0]['quote_id']), 'shared-report')

    model = ScriptedToolModel(replies=[
        tool_call('task', {'description': 'Investigate rerun behavior and return literal supporting evidence.',
                           'subagent_type': 'correctness_security'}, 'delegate-1'),
        tool_call('search_knowledge_base', {'query': 'widget rerun behavior'}, 'specialist-search'),
        specialist_result, read_specialist_evidence, cite_shared_source,
    ])
    engine = service(tmp_path, model)
    case = engine.start('Review whether widget reruns explain this uploaded CSV behavior.')
    assert case['status'] == 'awaiting_review', engine.store.events(case['case_id'])
    events = engine.store.events(case['case_id'])
    assert len(model.calls) == 5
    model_roles = [event['name'] for event in events if event['phase'] == 'model']
    assert model_roles.count('correctness_security') == 2
    assert model_roles.count('investigator') == 3
    assert any(event.get('specialist') == 'correctness_security' for event in events)
    assert [event['name'] for event in events if event['phase'] == 'tool'].count('search_knowledge_base') == 1
    assert len(engine.store.evidence(case['case_id'])) == 1
    assert case['report']['findings'][0]['support'][0]['evidence_id'] == engine.store.evidence(case['case_id'])[0]['evidence_id']


def test_provider_failure_retry_resumes_saved_model_step_without_repeating_kb_tool(tmp_path):
    desk = SnapshotDesk()
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'rerun persistence'}),
        RuntimeError('Simulated provider failure with private response text'), cite_kb,
    ])
    engine = service(tmp_path, model, desk=desk)
    failed = engine.start('Uploaded CSV disappears after rerun.')
    assert failed['status'] == 'failed'
    assert failed['report'] is None
    assert 'private response text' not in failed['error']
    original_evidence = engine.store.evidence(failed['case_id'])
    assert len(original_evidence) == 1
    restarted = service(tmp_path, model, desk=desk)
    resumed = restarted.retry(failed['case_id'])
    assert resumed['status'] == 'awaiting_review', restarted.store.events(failed['case_id'])
    assert len(model.calls) == 3
    assert len(desk.queries) == 1, 'Retry must resume the pending model step, not repeat completed tools'
    assert restarted.store.evidence(failed['case_id']) == original_evidence
    assert len(restarted.store.list_cases()) == 1
    assert len([event for event in restarted.store.events(failed['case_id']) if event['phase'] == 'model']) == 3


@pytest.mark.parametrize('citation', ['unknown_evidence', 'unknown_quote', 'wrong_source_quote'])
def test_malformed_model_citations_fail_closed_before_human_review(tmp_path, citation):
    def invalid_report(messages):
        source = source_from_messages(messages)
        evidence_id = 'ev_invented' if citation == 'unknown_evidence' else source['evidence_id']
        quote_id = source['quote_spans'][0]['quote_id']
        if citation == 'unknown_quote':
            quote_id = source['evidence_id'] + ':q999999'
        elif citation == 'wrong_source_quote':
            request = next(json.loads(message.content) for message in messages if isinstance(message, HumanMessage))
            quote_id = request['user_code_evidence'][0] + ':q1'
        return tool_call('InvestigationReport', report_payload(evidence_id, quote_id), 'invalid-report')

    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'reruns'}), invalid_report,
    ])
    engine = service(tmp_path, model)
    case = engine.start('An uploaded CSV resets.', code_context='uploaded = st.file_uploader("Upload CSV")')
    assert case['status'] == 'failed'
    assert case['report'] is None
    assert not case['report_markdown']
    assert len(engine.store.evidence(case['case_id'])) == 2
    assert not any(event.get('name') == 'approved' for event in engine.store.events(case['case_id']))
    with pytest.raises(ValueError, match='not ready'):
        engine.resume(case['case_id'], 'approve')


def test_model_call_budget_is_shared_across_orchestrator_and_specialist(tmp_path):
    model = ScriptedToolModel(replies=[
        tool_call('task', {'description': 'Research rerun behavior.', 'subagent_type': 'correctness_security'}, 'delegate-cap'),
        tool_call('search_knowledge_base', {'query': 'rerun limits'}, 'specialist-cap-search'),
        'The specialist should not be allowed to make this third model call.',
    ])
    engine = service(tmp_path, model, agent_settings=settings(max_model_calls=2))
    case = engine.start('A widget resets uploaded data.')
    assert case['status'] == 'failed'
    assert len(model.calls) == 2, 'All specialist and orchestrator calls must consume one shared limit'
    assert case['report'] is None
    events = engine.store.events(case['case_id'])
    roles = [event['name'] for event in events if event['phase'] == 'model']
    assert roles == ['investigator', 'correctness_security']
    assert len(engine.store.evidence(case['case_id'])) == 1


def test_harness_summarization_cannot_bypass_the_shared_model_call_limit(tmp_path):
    # A small context profile forces the harness's actual compaction path.
    # Summary generation is a provider call too, even though it is not a turn
    # that directly returns an investigation tool call.
    empty_report = {'findings': [], 'next_steps': [], 'missing_information': ['Provide a minimal reproduction.']}
    model = ScriptedToolModel(
        replies=[tool_call('search_knowledge_base', {'query': 'reruns before compaction'}),
                 ContextOverflowError('Simulated provider context limit'),
                 'The user reports that uploaded data resets.',
                 tool_call('InvestigationReport', empty_report, 'after-summary')],
        profile={'max_input_tokens': 1000},
    )
    engine = service(tmp_path, model, agent_settings=settings(max_model_calls=2))
    case = engine.start('word ' * 850)
    assert len(model.calls) <= 2, 'Compaction model calls must consume the case-wide model budget'
    assert len([event for event in engine.store.events(case['case_id']) if event['phase'] == 'model']) == len(model.calls)


def test_retry_repairs_a_completed_report_with_invalid_citations_in_the_same_case(tmp_path):
    def invented_citation(messages):
        source_from_messages(messages)
        return tool_call('InvestigationReport', report_payload(
            'ev_invented', statement='Failed report text that must not enter the repair thread.'), 'bad-first-report')

    desk = SnapshotDesk()
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'rerun citation repair'}), invented_citation,
        read_registered_evidence, cite_read_evidence,
    ])
    engine = service(tmp_path, model, desk=desk)
    failed = engine.start('Uploaded CSV resets after widget interaction.', version='1.50')
    assert failed['status'] == 'failed'
    assert failed['report'] is None
    repaired = service(tmp_path, model, desk=desk).retry(failed['case_id'])
    assert repaired['status'] == 'awaiting_review', engine.store.events(failed['case_id'])
    assert len(model.calls) == 4, 'The fresh repair thread must reread saved evidence before citing it'
    assert len(desk.queries) == 1
    assert repaired['case_id'] == failed['case_id']
    support = repaired['report']['findings'][0]['support'][0]
    assert support['evidence_id'] == engine.store.evidence(failed['case_id'])[0]['evidence_id']
    assert support['quote'] == PASSAGE
    assert len(engine.store.list_cases()) == 1
    repair_request = next(json.loads(message.content) for message in model.calls[2] if isinstance(message, HumanMessage))
    assert repair_request['issue'] == failed['question']
    assert repair_request['reported_version'] == '1.50'
    assert repair_request['repair_number'] == 1
    assert repair_request['repair_feedback']
    assert not any(isinstance(message, (AIMessage, ToolMessage)) for message in model.calls[2])
    assert 'Failed report text' not in str(model.calls[2])
    repair_events = [event for event in engine.store.events(failed['case_id']) if event['phase'] == 'repair']
    assert len(repair_events) == 1
    assert repair_events[0]['revision'] == 0
    assert repair_events[0]['repair_number'] == 1
    assert len([event for event in engine.store.events(failed['case_id']) if event['phase'] == 'model']) == 4


def test_requesting_revision_does_not_reset_the_case_wide_model_budget(tmp_path):
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'rerun budget'}), cite_kb,
        'This third turn must never reach the provider.',
    ])
    engine = service(tmp_path, model, agent_settings=settings(max_model_calls=2))
    initial = engine.start('Uploaded data resets.')
    assert initial['status'] == 'awaiting_review'
    revised = service(tmp_path, model, agent_settings=settings(max_model_calls=2)).resume(
        initial['case_id'], 'revise', notes='Check widget identity.',
    )
    assert revised['status'] == 'failed'
    assert len(model.calls) == 2
    assert len([event for event in engine.store.events(initial['case_id']) if event['phase'] == 'model']) == 2


@pytest.mark.parametrize('decision, expected', [('approve', 'approved'), ('reject', 'rejected')])
def test_final_review_cannot_be_replayed_and_never_dispatches_a_model(tmp_path, decision, expected):
    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'review finality'}), cite_kb,
    ])
    engine = service(tmp_path, model)
    pending = engine.start('A widget interaction resets data.')
    assert pending['status'] == 'awaiting_review'

    def forbidden_provider(_):
        raise AssertionError('Final review must not construct a provider model')

    restarted = InvestigationService(tmp_path, SnapshotDesk(), settings=AgentSettings(),
                                     model_factory=forbidden_provider)
    final = restarted.resume(pending['case_id'], decision, notes='Human review recorded.')
    assert final['status'] == expected
    for repeat in ('approve', 'reject', 'revise'):
        # A revision additionally requires model configuration, but is still
        # forbidden once the case is final.
        with pytest.raises(ValueError):
            restarted.resume(pending['case_id'], repeat, notes='Attempted replay.')
    assert restarted.store.get_case(pending['case_id']) == final
    assert len(model.calls) == 2


def test_startup_recovers_an_actual_crashed_process_and_retries_its_saved_checkpoint(tmp_path):
    from triage_desk.cases import CaseStore

    script = '''
import importlib.util
import os
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("offline_investigation_test", sys.argv[2])
test = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = test
spec.loader.exec_module(test)

def crash_process(messages):
    os._exit(75)

model = test.ScriptedToolModel(replies=[
    test.tool_call("search_knowledge_base", {"query": "checkpoint before process crash"}),
    crash_process,
])
test.service(Path(sys.argv[1]), model).start("Uploaded CSV disappears after rerun.")
raise AssertionError("The offline test process should have crashed")
'''
    crashed = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(Path(__file__).resolve())],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True,
                             text=True, timeout=30, check=False)
    assert crashed.returncode == 75, crashed.stderr
    store = CaseStore(tmp_path / '.cache' / 'agent' / 'cases.sqlite')
    abandoned = store.list_cases()[0]
    assert abandoned['status'] == 'running'
    evidence_before = store.evidence(abandoned['case_id'])
    assert len(evidence_before) == 1
    assert len([event for event in store.events(abandoned['case_id']) if event['phase'] == 'model']) == 2

    model = ScriptedToolModel(replies=[cite_kb])
    desk = SnapshotDesk()
    restarted = service(tmp_path, model, desk=desk)
    recovered = restarted.store.get_case(abandoned['case_id'])
    assert recovered['status'] == 'failed'
    assert 'interrupted' in recovered['error']
    assert not model.calls, 'Startup recovery must only reconcile saved state'
    result = restarted.retry(abandoned['case_id'])
    assert result['status'] == 'awaiting_review', restarted.store.events(abandoned['case_id'])
    assert len(model.calls) == 1
    assert not desk.queries, 'The completed pre-crash KB tool must not run again'
    assert restarted.store.evidence(abandoned['case_id']) == evidence_before
    assert len(restarted.store.list_cases()) == 1


def test_fresh_report_repair_preserves_the_case_wide_model_limit(tmp_path):
    def invalid_report(messages):
        source_from_messages(messages)
        return tool_call('InvestigationReport', report_payload('ev_invented'), 'invalid-limited')

    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'bounded repair'}), invalid_report,
        read_registered_evidence, cite_read_evidence,
    ])
    limited = settings(max_model_calls=3)
    engine = service(tmp_path, model, agent_settings=limited)
    first = engine.start('A CSV upload resets after widget interaction.')
    assert first['status'] == 'failed'
    repaired = service(tmp_path, model, agent_settings=limited).retry(first['case_id'])
    assert repaired['status'] == 'failed'
    assert len(model.calls) == 3, 'Changing inner threads must not grant a new provider budget'
    assert len([event for event in engine.store.events(first['case_id']) if event['phase'] == 'repair']) == 1
    # The pending repair is resumed; an exhausted budget cannot create a second repair.
    again = service(tmp_path, model, agent_settings=limited).retry(first['case_id'])
    assert again['status'] == 'failed'
    assert len(model.calls) == 3
    assert len([event for event in engine.store.events(first['case_id']) if event['phase'] == 'repair']) == 1
    assert len(engine.store.list_cases()) == 1


def test_legacy_repair_event_seeds_its_new_thread_without_spending_another_repair(tmp_path):
    def invalid_report(messages):
        source_from_messages(messages)
        return tool_call('InvestigationReport', report_payload('ev_invented'), 'legacy-invalid')

    model = ScriptedToolModel(replies=[
        tool_call('search_knowledge_base', {'query': 'legacy repair state'}), invalid_report,
        read_registered_evidence, cite_read_evidence,
    ])
    engine = service(tmp_path, model)
    failed = engine.start('A CSV resets after rerun.')
    assert failed['status'] == 'failed'
    # Older application versions recorded the repair before attempting to append
    # feedback to the full original thread, without a revision or repair number.
    engine.store.add_event(failed['case_id'], {'phase': 'repair', 'name': 'invalid_report', 'status': 'started'})
    result = service(tmp_path, model).retry(failed['case_id'])
    assert result['status'] == 'awaiting_review', engine.store.events(failed['case_id'])
    request = next(json.loads(message.content) for message in model.calls[2] if isinstance(message, HumanMessage))
    assert request['revision'] == 0
    assert request['repair_number'] == 1
    assert request['issue'] == failed['question']
    assert len([event for event in engine.store.events(failed['case_id']) if event['phase'] == 'repair']) == 1
    assert len(model.calls) == 4


def test_crash_within_fresh_repair_resumes_that_thread_and_its_completed_tools(tmp_path):
    from triage_desk.cases import CaseStore

    script = '''
import importlib.util
import os
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("offline_repair_test", sys.argv[2])
test = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = test
spec.loader.exec_module(test)

def invalid_report(messages):
    test.source_from_messages(messages)
    return test.tool_call("InvestigationReport", test.report_payload("ev_invented"), "invalid-before-crash")

def crash_process(messages):
    os._exit(76)

model = test.ScriptedToolModel(replies=[
    test.tool_call("search_knowledge_base", {"query": "repair checkpoint before crash"}),
    invalid_report, test.read_registered_evidence, crash_process,
])
engine = test.service(Path(sys.argv[1]), model)
failed = engine.start("Uploaded CSV resets after widget interaction.")
assert failed["status"] == "failed"
engine.retry(failed["case_id"])
raise AssertionError("The offline repair process should have crashed")
'''
    crashed = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(Path(__file__).resolve())],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True,
                             text=True, timeout=30, check=False)
    assert crashed.returncode == 76, crashed.stderr
    store = CaseStore(tmp_path / '.cache' / 'agent' / 'cases.sqlite')
    abandoned = store.list_cases()[0]
    assert abandoned['status'] == 'running'
    events_before = store.events(abandoned['case_id'])
    assert len([event for event in events_before if event['phase'] == 'model']) == 4
    assert len([event for event in events_before if event['phase'] == 'repair']) == 1
    assert len([event for event in events_before if event['phase'] == 'tool']) == 2
    original_evidence = store.evidence(abandoned['case_id'])

    model = ScriptedToolModel(replies=[cite_read_evidence])
    desk = SnapshotDesk()
    restarted = service(tmp_path, model, desk=desk)
    assert restarted.store.get_case(abandoned['case_id'])['status'] == 'failed'
    repaired = restarted.retry(abandoned['case_id'])
    assert repaired['status'] == 'awaiting_review', restarted.store.events(abandoned['case_id'])
    assert len(model.calls) == 1
    assert not desk.queries
    assert restarted.store.evidence(abandoned['case_id']) == original_evidence
    events_after = restarted.store.events(abandoned['case_id'])
    assert len([event for event in events_after if event['phase'] == 'model']) == 5
    assert len([event for event in events_after if event['phase'] == 'repair']) == 1
    assert len([event for event in events_after if event['phase'] == 'tool']) == 2
