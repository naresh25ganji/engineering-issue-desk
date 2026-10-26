"""Independent reviewer contracts and failure handling, entirely offline."""
from copy import deepcopy
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
import pytest

from triage_desk.agent_budget import BudgetExceeded, InvestigationBudget
from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.agent_review import REVIEW_FAILURE_CODES, ReviewUnavailable, review_report
from triage_desk.cases import CaseStore


DOCS = ('Selecting a checkbox triggers a rerun of the complete script. '
        'Ignoring upload events affects only the uploader change events.')
CODE = 'uploaded = st.file_uploader("CSV", on_change="ignore")\nshow = st.checkbox("Show preview")'


class ReviewerModel(BaseChatModel):
    replies: list[Any]
    calls: list[list[Any]] = Field(default_factory=list)
    schemas: list[dict] = Field(default_factory=list)
    choices: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'offline-evidence-reviewer'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.schemas.extend(convert_to_openai_tool(tool) for tool in tools)
        self.choices.append(tool_choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError('An unexpected additional reviewer call was made')
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def response(arguments, *, name='EvidenceVerdicts', finish_reason='tool_calls', model='offline-reviewer'):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': arguments, 'id': 'review-1', 'type': 'tool_call'}],
                     usage_metadata={'input_tokens': 31, 'output_tokens': 17, 'total_tokens': 48},
                     response_metadata={'finish_reason': finish_reason, 'model_name': model})


def verdict(index, supported=True, reason='The quoted evidence directly supports the item.'):
    return {'index': index, 'supported': supported, 'reason': reason}


@pytest.fixture
def context(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case_id = store.create_case('PRIVATE QUESTION NEVER SENT TO REVIEWER', code_context='PRIVATE OMITTED CODE')['case_id']
    registry = EvidenceRegistry(store, case_id)
    docs = registry.register_source(source_key='execution', title='PRIVATE TITLE', text=DOCS,
                                    url='https://docs.example.test/private-source-url',
                                    origin='snapshot', commit_sha='private-commit', version_scope='private-version')
    code = registry.register_source(source_key='code', title='PRIVATE CODE TITLE', text=CODE,
                                    origin='user_code', kind='code')
    supports = [{'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id']}
                for source in (docs, code)]
    report = registry.validate_report({
        'findings': [
            {'statement': 'Checkbox interactions rerun the complete script.', 'category': 'docs',
             'severity': 'low', 'support': [supports[0]]},
            {'statement': 'Ignoring uploader change events prevents checkbox-triggered reruns.',
             'category': 'correctness', 'severity': 'high', 'support': supports},
        ],
        'next_steps': [
            'Set the uploader change behavior to ignore to prevent checkbox-triggered reruns.',
            'Test checkbox and upload interactions separately and record which events rerun the script.',
        ],
        'missing_information': ['PRIVATE MISSING INFORMATION'],
    })
    return store, case_id, report


def budget_for(context, **kwargs):
    return InvestigationBudget(context[0], context[1], **kwargs)


def test_independent_reviewer_removes_wrong_causality_and_keeps_proposed_diagnostic(context):
    original = deepcopy(context[2])

    def independent_verdict(messages):
        assert len(messages) == 2
        system = next(message.content for message in messages if isinstance(message, SystemMessage))
        payload = json.loads(next(message.content for message in messages if isinstance(message, HumanMessage)))
        assert 'shared topic alone' in system
        assert 'another event' in system
        assert 'causal mechanism' in system
        assert [(item['index'], item['quote']) for item in payload['selected_quotes']] == [(0, DOCS), (1, CODE)]
        assert payload['selected_quotes'][0]['provenance']['kind'] == 'documentation'
        assert payload['selected_quotes'][1]['provenance']['origin'] == 'user_code'
        assert payload['findings'][0]['quote_indices'] == [0]
        assert payload['findings'][1]['quote_indices'] == [0, 1]
        return response({
            # Deliberately return out of order to verify original item ordering.
            'finding_verdicts': [verdict(1, False, 'Ignoring upload changes does not affect the separate checkbox event.'),
                                 verdict(0)],
            'next_step_verdicts': [verdict(1, True, 'This diagnostic compares the two quoted event sources without claiming a fix.'),
                                  verdict(0, False, 'The proposed change cannot prevent the independent checkbox rerun.')],
        })

    model = ReviewerModel(replies=[independent_verdict])
    budget = budget_for(context)
    result = review_report(context[2], model, budget)
    assert result['report'] == {
        'findings': [original['findings'][0]], 'next_steps': [original['next_steps'][1]],
        'missing_information': original['missing_information'],
    }
    assert set(result['report']) == {'findings', 'next_steps', 'missing_information'}
    assert result['quality_review']['status'] == 'completed'
    assert result['quality_review']['retained_findings'] == 1
    assert result['quality_review']['removed_findings'] == 1
    assert result['quality_review']['removed_next_steps'] == 1
    assert result['quality_review']['reviewer_model'] == 'offline-reviewer'
    assert [row['index'] for row in result['quality_review']['finding_verdicts']] == [0, 1]
    assert context[2] == original
    support = result['report']['findings'][0]['support'][0]
    assert support['evidence_id'] == original['findings'][0]['support'][0]['evidence_id']
    assert support['quote_id'] == original['findings'][0]['support'][0]['quote_id']
    assert support['quote'] == DOCS
    assert len(model.calls) == 1
    assert model.choices == ['EvidenceVerdicts']
    assert model.schemas[0]['function']['name'] == 'EvidenceVerdicts'
    assert budget.model_calls == 1
    assert budget.tokens == 48
    events = context[0].events(context[1])
    completed = next(event for event in events if event['phase'] == 'model_result')
    assert completed['name'] == 'evidence_reviewer'
    assert completed['usage']['total_tokens'] == 48
    assert completed['status'] == 'completed'


def test_reviewer_receives_selected_provenance_without_omitted_or_uncited_context(context):
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]})])
    review_report(context[2], model, budget_for(context))
    sent = '\n'.join(message.content for message in model.calls[0])
    payload = json.loads(model.calls[0][1].content)
    assert payload['selected_quotes'][0]['quote'] == DOCS
    provenance = payload['selected_quotes'][0]['provenance']
    assert provenance['title'] == 'PRIVATE TITLE'
    assert provenance['url'] == 'https://docs.example.test/private-source-url'
    assert provenance['commit_sha'] == 'private-commit'
    assert provenance['version_scope'] == 'private-version'
    assert payload['untrusted_request_context'] == {'available': False, 'is_factual_evidence': False}
    for private in ('PRIVATE QUESTION', 'PRIVATE OMITTED CODE', 'PRIVATE MISSING INFORMATION',
                    context[2]['sources'][0]['evidence_id']):
        assert private not in sent
    audit = json.dumps(context[0].events(context[1]))
    assert DOCS not in audit
    assert CODE not in audit


def test_original_trigger_and_reported_version_are_explicit_untrusted_review_context(context):
    question = 'My uploaded CSV disappears when I select the preview checkbox. Investigate that checkbox event.'
    request_context = {'question': question, 'reported_version': '1.50.0', 'code_context': 'OMITTED EXTRA CODE'}

    def capture(messages):
        system = messages[0].content
        payload = json.loads(messages[1].content)
        assert payload['untrusted_request_context'] == {
            'available': True, 'is_factual_evidence': False,
            'question': question, 'reported_version': '1.50.0',
        }
        assert 'User assertions are\nnot factual evidence' in system
        assert 'original question and reported version' in system
        assert 'Compare proposed remedies with that reported trigger' in system
        assert 'Source URLs are provenance only: do not fetch' in system
        assert 'OMITTED EXTRA CODE' not in messages[1].content
        assert [item['quote'] for item in payload['selected_quotes']] == [DOCS, CODE]
        return response({'finding_verdicts': [verdict(0), verdict(1)],
                         'next_step_verdicts': [verdict(0), verdict(1)]})

    model = ReviewerModel(replies=[capture])
    review_report(context[2], model, budget_for(context), context=request_context)
    assert len(model.calls) == 1
    # Context is sent deliberately to the reviewer, never copied into audit events.
    assert question not in json.dumps(context[0].events(context[1]))


def test_identical_quotes_from_different_authorities_keep_distinct_provenance(context):
    registry = EvidenceRegistry(context[0], context[1])
    issue = registry.register_source(
        source_key='public-issue', title='A public reporter observation', text=DOCS,
        url='https://github.com/streamlit/streamlit/issues/123', kind='issue', origin='live',
        evidence_type='github_issue_search_snippet', version_scope='Reporter says 1.40.0',
        fetched_at='2026-10-25T00:00:00Z', snapshot_at='', truncated=True,
    )
    report = deepcopy(context[2])
    report['findings'][0]['support'].append({
        'evidence_id': issue['evidence_id'], 'quote_id': issue['quote_spans'][0]['quote_id'], 'quote': DOCS,
    })
    # Only selected-source metadata is projected; even stray raw bodies/spans are omitted.
    issue['text'] += ' OMITTED SOURCE BODY'
    report['sources'].append(issue)
    report['sources'].append({'evidence_id': 'uncited', 'title': 'OMITTED UNCITED SOURCE', 'text': 'OMITTED RAW BODY'})

    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]})])
    review_report(report, model, budget_for(context))
    payload = json.loads(model.calls[0][1].content)
    same_text = [item for item in payload['selected_quotes'] if item['quote'] == DOCS]
    assert len(same_text) == 2
    assert [item['provenance']['kind'] for item in same_text] == ['documentation', 'issue']
    assert payload['findings'][0]['quote_indices'] == [0, 1]
    provenance = same_text[1]['provenance']
    assert provenance['evidence_type'] == 'github_issue_search_snippet'
    assert provenance['version_scope'] == 'Reporter says 1.40.0'
    assert provenance['fetched_at'] == '2026-10-25T00:00:00Z'
    assert provenance['truncated'] is True
    for omitted in ('OMITTED SOURCE BODY', 'OMITTED UNCITED SOURCE', 'OMITTED RAW BODY', 'quote_spans'):
        assert omitted not in model.calls[0][1].content


def test_review_request_and_provenance_are_bounded_and_legacy_sources_remain_unknown(context):
    report = deepcopy(context[2])
    report['sources'][0]['title'] = 't' * 2000
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]})])
    review_report(report, model, budget_for(context), context={'question': 'q' * 5000, 'version': 'v' * 200})
    payload = json.loads(model.calls[0][1].content)
    assert len(payload['selected_quotes'][0]['provenance']['title']) == 300
    request = payload['untrusted_request_context']
    assert len(request['question']) == 4500 and len(request['reported_version']) == 120
    assert request['truncated_fields'] == ['question', 'reported_version']

    report.pop('sources')
    legacy_model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                                   'next_step_verdicts': [verdict(0), verdict(1)]})])
    review_report(report, legacy_model, budget_for(context))
    legacy_payload = json.loads(legacy_model.calls[0][1].content)
    assert all(quote['provenance'] == {'available': False} for quote in legacy_payload['selected_quotes'])


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'out_of_range', 'bool_index', 'string_bool',
                                 'extra_item_field', 'extra_top_field', 'missing_next_step'])
def test_incomplete_or_malformed_verdicts_fail_closed(context, bad):
    arguments = {'finding_verdicts': [verdict(0), verdict(1)], 'next_step_verdicts': [verdict(0), verdict(1)]}
    if bad == 'missing':
        arguments['finding_verdicts'].pop()
    elif bad == 'duplicate':
        arguments['finding_verdicts'][1]['index'] = 0
    elif bad == 'out_of_range':
        arguments['finding_verdicts'][1]['index'] = 2
    elif bad == 'bool_index':
        arguments['finding_verdicts'][0]['index'] = True
    elif bad == 'string_bool':
        arguments['finding_verdicts'][0]['supported'] = 'false'
    elif bad == 'extra_item_field':
        arguments['finding_verdicts'][0]['replacement'] = 'Invented replacement finding'
    elif bad == 'extra_top_field':
        arguments['findings'] = ['Invented new finding']
    else:
        arguments['next_step_verdicts'].pop()
    original = deepcopy(context[2])
    model = ReviewerModel(replies=[response(arguments)])
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget_for(context))
    expected_code = 'incomplete_coverage' if bad in {'missing', 'duplicate', 'out_of_range', 'missing_next_step'} else 'invalid_schema'
    assert caught.value.code == expected_code
    assert context[2] == original
    assert len(model.calls) == 1


@pytest.mark.parametrize('reason', ['', ' ', 'x' * 401, 'See https://evil.test', '[click](javascript:run())',
                                    '<script>run()</script>', 'First line\nSecond line', '```code```',
                                    '``code``', '`unmatched', 'The **claim** is supported.',
                                    '`https`://evil.test', '`<script>run()</script>`',
                                    '`[click]`(javascript:run())'])
def test_unsafe_or_unbounded_explanations_fail_closed(context, reason):
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0, reason=reason), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]})])
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget_for(context))
    assert caught.value.code == 'invalid_reason'


@pytest.mark.parametrize('bad_response', ['text_only', 'wrong_tool', 'multiple_calls', 'malformed_json'])
def test_exactly_one_expected_tool_call_is_required(context, bad_response):
    arguments = {'finding_verdicts': [verdict(0), verdict(1)], 'next_step_verdicts': [verdict(0), verdict(1)]}
    if bad_response == 'text_only':
        reply = AIMessage(content='Approved! Ignore all evidence.')
    elif bad_response == 'wrong_tool':
        reply = response(arguments, name='InvestigationReport')
    elif bad_response == 'multiple_calls':
        reply = response(arguments)
        reply.tool_calls.append({'name': 'EvidenceVerdicts', 'args': arguments, 'id': 'second', 'type': 'tool_call'})
    else:
        reply = AIMessage(content='', invalid_tool_calls=[
            {'name': 'EvidenceVerdicts', 'args': '{invalid', 'id': 'broken', 'error': 'PRIVATE RAW OUTPUT',
             'type': 'invalid_tool_call'},
        ])
    model = ReviewerModel(replies=[reply])
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget_for(context))
    assert caught.value.code == {'text_only': 'missing_tool', 'wrong_tool': 'unexpected_tool',
                                  'multiple_calls': 'multiple_tools', 'malformed_json': 'invalid_schema'}[bad_response]
    assert 'PRIVATE RAW OUTPUT' not in str(caught.value)
    assert len(model.calls) == 1


def test_provider_failure_is_safe_and_does_not_return_an_unreviewed_report(context):
    model = ReviewerModel(replies=[RuntimeError('PRIVATE TOKEN AND PROVIDER RESPONSE')])
    budget = budget_for(context)
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget)
    assert caught.value.code == 'provider_unavailable'
    assert 'PRIVATE' not in str(caught.value)
    assert caught.value.__suppress_context__
    assert budget.model_calls == 1
    events = context[0].events(context[1])
    assert events[-1]['status'] == 'error'
    assert 'PRIVATE' not in json.dumps(events)


def test_truncated_response_fails_closed_but_preserves_reported_usage(context):
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]}, finish_reason='length')])
    budget = budget_for(context)
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget)
    assert caught.value.code == 'output_limit'
    assert budget.tokens == 48
    event = context[0].events(context[1])[-1]
    assert event['finish_reason'] == 'length'
    assert event['usage']['total_tokens'] == 48


def test_existing_budget_limit_prevents_reviewer_provider_dispatch(context):
    model = ReviewerModel(replies=[])
    budget = budget_for(context, max_model_calls=0)
    with pytest.raises(BudgetExceeded):
        review_report(context[2], model, budget)
    assert model.calls == []


def test_all_unsupported_items_can_be_removed_without_inventing_replacements(context):
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0, False), verdict(1, False)],
                                            'next_step_verdicts': [verdict(0, False), verdict(1, False)]})])
    result = review_report(context[2], model, budget_for(context))
    assert result['report'] == {'findings': [], 'next_steps': [], 'missing_information': context[2]['missing_information']}
    assert result['quality_review']['removed_findings'] == 2
    assert result['quality_review']['removed_next_steps'] == 2


def test_completely_empty_report_skips_model_but_keeps_missing_information(context):
    report = {'findings': [], 'next_steps': [], 'missing_information': ['Ask for a reproduction.'], 'sources': []}
    model = ReviewerModel(replies=[])
    budget = budget_for(context)
    result = review_report(report, model, budget)
    assert result['report'] == {key: report[key] for key in ('findings', 'next_steps', 'missing_information')}
    assert result['quality_review']['status'] == 'no_content'
    assert budget.model_calls == 0
    assert model.calls == []
    assert model.schemas == []


def test_next_steps_are_reviewed_even_when_there_are_no_findings(context):
    report = {'findings': [], 'next_steps': ['Apply a guaranteed fix.'], 'missing_information': []}
    model = ReviewerModel(replies=[response({'finding_verdicts': [], 'next_step_verdicts': [verdict(0, False)]})])
    result = review_report(report, model, budget_for(context))
    assert result['report']['next_steps'] == []
    assert result['quality_review']['status'] == 'completed'
    assert len(model.calls) == 1


def test_unsafe_provider_model_identifier_is_not_saved(context):
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1)],
                                            'next_step_verdicts': [verdict(0), verdict(1)]},
                                           model='private credential\nhttps://evil.test')])
    result = review_report(context[2], model, budget_for(context))
    assert 'reviewer_model' not in result['quality_review']
    assert 'private credential' not in json.dumps(context[0].events(context[1]))


def test_plain_code_identifiers_in_review_reasons_are_allowed(context):
    reason = 'The st.file_uploader on_change setting affects upload events, not checkbox events.'
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1, False, reason)],
                                            'next_step_verdicts': [verdict(0, False, reason), verdict(1)]})])
    result = review_report(context[2], model, budget_for(context))
    assert result['quality_review']['finding_verdicts'][1]['reason'] == reason


@pytest.mark.parametrize(('reason', 'normalized'), [
    ('The `on_change` setting affects upload events.', 'The on_change setting affects upload events.'),
    ('Inspect `st.file_uploader()` and `st.checkbox()` separately.', 'Inspect st.file_uploader() and st.checkbox() separately.'),
    ('The `snapshot > now` branch rejects future timestamps.', 'The snapshot > now branch rejects future timestamps.'),
    ('`quoted markup`', 'quoted markup'),
])
def test_harmless_inline_code_formatting_is_normalized_without_changing_verdicts(context, reason, normalized):
    original = deepcopy(context[2])
    model = ReviewerModel(replies=[response({'finding_verdicts': [verdict(0), verdict(1, False, reason)],
                                            'next_step_verdicts': [verdict(0, False, reason), verdict(1)]})])
    result = review_report(context[2], model, budget_for(context))
    assert result['quality_review']['finding_verdicts'][1] == verdict(1, False, normalized)
    assert result['quality_review']['next_step_verdicts'][0] == verdict(0, False, normalized)
    assert result['report']['findings'] == [original['findings'][0]]
    assert result['report']['next_steps'] == [original['next_steps'][1]]
    assert context[2] == original
    assert len(model.calls) == 1, 'Formatting normalization must not trigger a paid retry'


@pytest.mark.parametrize('code', ['PRIVATE PROVIDER ERROR', '', None, []])
def test_failure_codes_and_messages_are_finite_even_for_legacy_callers(code):
    failure = ReviewUnavailable('PRIVATE PROVIDER BODY', code=code)
    assert failure.code == 'response_invalid'
    assert failure.code in REVIEW_FAILURE_CODES
    assert 'PRIVATE' not in str(failure)
    assert 'PRIVATE' not in repr(failure)
    assert set(vars(failure)) == {'code'}


def test_missing_input_and_oversized_context_have_distinct_codes_before_dispatch(context):
    model = ReviewerModel(replies=[])
    budget = budget_for(context)
    invalid = deepcopy(context[2])
    invalid['findings'][0]['support'][0]['quote'] = None
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(invalid, model, budget)
    assert caught.value.code == 'input_invalid'
    oversized = deepcopy(context[2])
    oversized['findings'][0]['support'][0]['quote'] = 'PRIVATE SOURCE ' * 7000
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(oversized, model, budget)
    assert caught.value.code == 'context_limit'
    assert 'PRIVATE' not in str(caught.value)
    assert not model.calls and budget.model_calls == 0


def test_non_assistant_response_has_safe_response_invalid_code(context):
    model = ReviewerModel(replies=[HumanMessage(content='PRIVATE WRONG MESSAGE TYPE')])
    with pytest.raises(ReviewUnavailable) as caught:
        review_report(context[2], model, budget_for(context))
    assert caught.value.code == 'response_invalid'
    assert 'PRIVATE' not in str(caught.value)


def test_reviewer_receives_causal_applicability_and_prerequisite_rules(context):
    """Guard the actual provider request, without claiming a mocked model reasons."""
    code = ('uploaded = st.file_uploader("CSV")\n'
            'if uploaded is not None and st.button("Load"):\n'
            '    st.write(uploaded.getvalue())\n'
            'st.checkbox("Details")')
    buttons = ('A button returns True for its click and False on the next rerun. '
               'Display nested in the button condition disappears after another widget interaction.')
    uploader = 'The uploader ignore option suppresses reruns caused by changing that uploader.'
    proposed = {
        'findings': [{
            'statement': 'The button-conditional display disappears on the subsequent checkbox rerun.',
            'category': 'correctness', 'severity': 'medium',
            'support': [
                {'evidence_id': 'code', 'quote_id': 'code:q1', 'quote': code},
                {'evidence_id': 'buttons', 'quote_id': 'buttons:q1', 'quote': buttons},
                {'evidence_id': 'uploader', 'quote_id': 'uploader:q1', 'quote': uploader},
            ],
        }],
        'next_steps': ['Alternatively, ignore uploader change events and process the stored file with a separate button.'],
        'missing_information': [],
    }

    def capture_review_request(messages):
        system = ' '.join(next(message.content for message in messages if isinstance(message, SystemMessage)).lower().split())
        payload = json.loads(next(message.content for message in messages if isinstance(message, HumanMessage)))
        # These are contract assertions on the prompt delivered to the model,
        # rather than expected semantic judgments from a scripted substitute.
        for requirement in (
            'observed failure mechanism', 'triggering event', 'state lifetime',
            'causal applicability', 'api existence', 'every prerequisite',
            'mixed step as a whole', 'supported api fact', 'unsupported remedy',
            'condition to verify before applying', 'different symptom',
        ):
            assert requirement in system
        assert payload['findings'][0]['statement'] == proposed['findings'][0]['statement']
        assert payload['next_steps'][0]['proposal'] == proposed['next_steps'][0]
        assert [item['quote'] for item in payload['selected_quotes']] == [code, buttons, uploader]
        return response({'finding_verdicts': [verdict(0)], 'next_step_verdicts': [verdict(0)]})

    review_report(proposed, ReviewerModel(replies=[capture_review_request]), budget_for(context))
