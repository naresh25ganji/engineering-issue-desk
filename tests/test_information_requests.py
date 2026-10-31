"""The observed credential-value request_text is guarded without rewriting evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
import pytest

from triage_desk.deep_investigation import AgentSettings, InvestigationService, checkpoint_saver
from triage_desk.information_requests import (
    POLICY_VERSION, SAFE_CONFIGURATION_REQUEST, SOURCE_REDACTION_NOTE, sanitize_missing_information,
)


OBSERVED_REQUEST = 'Specific values returned by _configuration() (key, base) and the expected embedding endpoint configuration.'


def report(*requests):
    return {'findings': [], 'next_steps': [], 'missing_information': list(requests)}


@pytest.mark.parametrize('request_text', [
    OBSERVED_REQUEST,
    'Please provide the API key value.',
    'Share your access token.',
    'Can you paste FIREWORKS_API_KEY?',
    'Actual values of the client secret and provider configuration.',
    'Output of _configuration() including key and base.',
    'Specific output of _runtime_key().',
    'API key value.',
    'Provide the source implementation and the actual API key values.',
    'Provide API key values and the helper implementation.',
    'What is the value of the API key?',
])
def test_explicit_credential_disclosure_requests_become_presence_and_redacted_configuration(request_text):
    original = report(request_text)
    guarded, count = sanitize_missing_information(original)
    assert count == 1 and guarded['missing_information'] == [SAFE_CONFIGURATION_REQUEST]
    assert original['missing_information'] == [request_text]
    assert sanitize_missing_information(guarded) == (guarded, 0)


@pytest.mark.parametrize('request_text', [
    'Which dictionary keys are accepted by the payload?',
    'Specific values of the dictionary keys used in this test.',
    'Specific values returned by _configuration() (timeout, base).',
    'Whether the API key is configured (yes/no).',
    'Share whether the API key has a value configured (yes/no).',
    'Share whether FIREWORKS_API_KEY is configured, without its value.',
    'Provide the API key variable name.',
    'The helper returns an API key for authentication.',
    'The API key value is never included in this error message.',
    'What is an API key used for in this helper?',
    'Installed version, operating system, and Python environment.',
    'The number of input and output tokens reported by the model.',
])
def test_presence_names_factual_mentions_dictionary_keys_and_environment_requests_are_preserved(request_text):
    original = report(request_text)
    assert sanitize_missing_information(original) == (original, 0)


@pytest.mark.parametrize('request_text', [
    'Implementation of _configuration().',
    'Source of _runtime_key().',
    'Provide source code for the credential-loading helper.',
    'Provide implementation of _configuration() returning key, base.',
])
def test_helper_source_requests_remain_allowed_with_explicit_redaction(request_text):
    guarded, count = sanitize_missing_information(report(request_text))
    assert count == 1
    assert guarded['missing_information'] == [request_text + ' ' + SOURCE_REDACTION_NOTE]
    assert sanitize_missing_information(guarded) == (guarded, 0)


def test_findings_quotes_proposals_and_existing_source_redaction_are_never_rewritten():
    finding = {'statement': 'The code returns key.', 'support': [{'quote': 'return key # literal source'}]}
    original = {'findings': [finding], 'next_steps': ['A proposed check.'],
                'missing_information': [OBSERVED_REQUEST, 'Source code with credentials redacted.']}
    before = deepcopy(original)
    guarded, count = sanitize_missing_information(original)
    assert count == 1 and original == before
    assert guarded['findings'] is original['findings']
    assert guarded['next_steps'] is original['next_steps']
    assert guarded['missing_information'][1] == original['missing_information'][1]


class RequestModel(BaseChatModel):
    payload: dict
    calls: int = 0
    observed: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'offline-information-request_text-contract'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        self.observed.append(messages)
        message = AIMessage(content='', tool_calls=[{
            'name': 'InvestigationReport', 'args': self.payload,
            'id': f'report-{self.calls}', 'type': 'tool_call'}])
        message.usage_metadata = {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.parametrize('reviewer_reintroduces_request', [False, True])
def test_real_graph_guards_observed_request_before_draft_storage_and_human_checkpoint(
        tmp_path, reviewer_reintroduces_request):
    model = RequestModel(payload=report(OBSERVED_REQUEST, 'Which dictionary keys are accepted?'))
    draft_reports = []

    def reviewer(draft, _model, _budget):
        draft_reports.append(deepcopy(draft))
        saved_draft = engine.store.get_case(case_id[0])['report']
        assert saved_draft['missing_information'][0] == SAFE_CONFIGURATION_REQUEST
        output = {key: deepcopy(draft[key]) for key in ('findings', 'next_steps', 'missing_information')}
        if reviewer_reintroduces_request:
            output['missing_information'][0] = OBSERVED_REQUEST
        return {'report': output, 'quality_review': {'status': 'completed'}}

    engine = InvestigationService(tmp_path, None, AgentSettings(api_key='offline-test-key', model='offline-model'),
                                  model_factory=lambda _: model, reviewer=reviewer)
    case_id = []
    original_create = engine.store.create_case

    def create(*args, **kwargs):
        result = original_create(*args, **kwargs)
        case_id.append(result['case_id'])
        return result
    engine.store.create_case = create
    pending = engine.start('Identify missing configuration without sharing credential values.')
    assert pending['status'] == 'awaiting_review', engine.store.events(pending['case_id'])
    assert model.calls == 1
    assert draft_reports[0]['missing_information'][0] == SAFE_CONFIGURATION_REQUEST
    assert pending['report']['missing_information'] == [SAFE_CONFIGURATION_REQUEST, 'Which dictionary keys are accepted?']
    assert pending['report']['quality_review']['information_request_safety'] == {
        'policy_version': POLICY_VERSION, 'draft_converted': 1,
        'review_converted': int(reviewer_reintroduces_request),
        'converted_count': 1 + int(reviewer_reintroduces_request)}
    events = [event for event in engine.store.events(pending['case_id']) if event.get('phase') == 'safety']
    assert [event['converted_count'] for event in events] == [1, int(reviewer_reintroduces_request)]
    assert OBSERVED_REQUEST not in json.dumps(events)
    assert all(event['policy_version'] == POLICY_VERSION for event in events)
    with checkpoint_saver(engine.directory / 'checkpoints.sqlite') as saver:
        checkpoint = saver.get_tuple({'configurable': {'thread_id': pending['case_id']}})
        human_draft = checkpoint.checkpoint['channel_values']['report']
    assert human_draft['missing_information'][0] == SAFE_CONFIGURATION_REQUEST
    assert OBSERVED_REQUEST not in pending['report_markdown']
