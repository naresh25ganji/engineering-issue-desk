"""Cross-specialist accounting, crash conservatism, and bounded execution."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
import pytest

from triage_desk.agent_budget import AgentOutputError, BudgetExceeded, BudgetMiddleware, InvestigationBudget
from triage_desk.cases import CaseStore


@pytest.fixture
def saved_case(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    return store, store.create_case('Why does this widget reset?')['case_id']


def response(_):
    return ModelResponse(result=[AIMessage(content='Done', usage_metadata={
        'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})])


def test_parent_and_specialists_share_model_and_token_limits(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id, max_model_calls=2)
    parent = BudgetMiddleware(budget)
    child = BudgetMiddleware(budget, 'correctness_security')
    parent.wrap_model_call(None, response)
    child.wrap_model_call(None, response)
    with pytest.raises(BudgetExceeded, match='model-call'):
        child.wrap_model_call(None, response)
    assert budget.tokens == 30
    events = store.events(case_id)
    assert [e['name'] for e in events if e['phase'] == 'model'] == ['investigator', 'correctness_security']
    assert sum(e.get('usage', {}).get('total_tokens', 0) for e in events) == 30


def test_restart_counts_dispatched_calls_even_if_process_did_not_record_result(saved_case):
    store, case_id = saved_case
    store.add_event(case_id, {'phase': 'model', 'name': 'investigator', 'status': 'started'})
    restarted = InvestigationBudget(store, case_id, max_model_calls=1)
    with pytest.raises(BudgetExceeded):
        BudgetMiddleware(restarted).wrap_model_call(None, response)


def test_parallel_tool_reservations_never_exceed_shared_limit(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id, max_tool_calls=3)

    def execute(_):
        try:
            with budget.tool_call('read_case_evidence'):
                return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(execute, range(12)))
    assert sum(results) == 3
    assert sum(e['phase'] == 'tool' for e in store.events(case_id)) == 3


def test_domain_tool_is_not_counted_twice_and_arguments_are_not_saved(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    request = SimpleNamespace(tool_call={'name': 'search_web', 'args': {'query': 'sensitive-code'}})

    def tool_handler(_):
        with budget.tool_call('search_web', {'query': 'sensitive-code'}):
            return ToolMessage(content='Found', tool_call_id='tool-1')

    BudgetMiddleware(budget).wrap_tool_call(request, tool_handler)
    assert budget.tool_calls == 1
    assert 'sensitive-code' not in str(store.events(case_id))


def test_failure_is_counted_and_exception_content_is_not_persisted(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)

    def fail(_):
        raise RuntimeError('private API key')

    with pytest.raises(RuntimeError):
        BudgetMiddleware(budget).wrap_model_call(None, fail)
    restarted = InvestigationBudget(store, case_id)
    assert restarted.model_calls == 1
    events = store.events(case_id)
    assert events[-1]['error_type'] == 'RuntimeError'
    assert events[-1]['status'] == 'error'
    assert 'private API key' not in str(events)


def test_reported_token_limit_and_active_deadline_block_further_dispatch(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id, max_tokens=15)
    BudgetMiddleware(budget).wrap_model_call(None, response)
    with pytest.raises(BudgetExceeded, match='token'):
        with budget.tool_call('read_case_evidence'):
            pytest.fail('Must not dispatch')
    expired = InvestigationBudget(store, case_id, max_seconds=0)
    with pytest.raises(BudgetExceeded, match='time'):
        BudgetMiddleware(expired).wrap_model_call(None, response)


def test_missing_usage_stays_explicit(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    BudgetMiddleware(budget).wrap_model_call(None, lambda _: ModelResponse(result=[AIMessage(content='Done')]))
    event = store.events(case_id)[-1]
    assert event['usage'] == {}
    assert event['usage_reported'] is False


def test_model_receives_quote_ids_without_duplicate_raw_source_text(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    source = {'evidence_id': 'ev_1', 'text': 'Original source text that is saved in the registry.',
              'origin': 'snapshot', 'quote_spans': [{'quote_id': 'ev_1:q1',
                   'quote': 'Original source text that is saved in the registry.'}]}
    request = SimpleNamespace(tool_call={'name': 'search_knowledge_base', 'args': {}})
    message = ToolMessage(content=json.dumps({'status': 'ok', 'evidence': [source]}), tool_call_id='read-1')
    shown = BudgetMiddleware(budget).wrap_tool_call(request, lambda _: message)
    model_source = json.loads(shown.content)['evidence'][0]
    assert model_source['quote_spans'] == source['quote_spans']
    assert model_source['origin'] == 'snapshot'
    assert 'text' not in model_source
    assert json.loads(message.content)['evidence'][0]['text'] == source['text']


def test_valid_tool_output_records_only_safe_metadata_and_keeps_response_unchanged(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    message = AIMessage(
        content='Private model explanation',
        tool_calls=[{'name': 'search_knowledge_base', 'args': {'query': 'private user code'}, 'id': 'call-1'}],
        response_metadata={'finish_reason': 'tool_calls', 'model_name': 'accounts/fireworks/models/gpt-oss-120b'},
        usage_metadata={'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30},
    )
    original = message.model_dump()
    output = ModelResponse(result=[message, ToolMessage(content='Private tool reply', tool_call_id='call-1')])
    returned = BudgetMiddleware(budget).wrap_model_call(None, lambda _: output)
    assert returned is output
    assert message.model_dump() == original
    event = store.events(case_id)[-1]
    assert event['finish_reason'] == 'tool_calls'
    assert event['returned_model'] == 'accounts/fireworks/models/gpt-oss-120b'
    assert event['tool_call_count'] == 1
    assert event['invalid_tool_call_count'] == 0
    assert event['usage']['total_tokens'] == 30
    assert 'Private' not in str(store.events(case_id))
    assert 'private user code' not in str(store.events(case_id))


@pytest.mark.parametrize('finish_reason', ['length', 'max_tokens', 'max_output_tokens'])
def test_truncated_model_output_fails_safely_after_usage_is_recorded(saved_case, finish_reason):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    message = AIMessage(
        content='Private incomplete output',
        response_metadata={'finish_reason': finish_reason, 'model': 'gpt-oss-120b'},
        usage_metadata={'input_tokens': 12, 'output_tokens': 2200, 'total_tokens': 2212},
    )
    output = ModelResponse(result=[message])
    with pytest.raises(AgentOutputError, match='output limit') as caught:
        BudgetMiddleware(budget).wrap_model_call(None, lambda _: output)
    assert caught.value.response is output
    assert message.content == 'Private incomplete output'
    assert 'Private' not in str(caught.value)
    assert budget.tokens == 2212
    event = store.events(case_id)[-1]
    assert event['status'] == 'error'
    assert event['error_type'] == 'AgentOutputError'
    assert event['finish_reason'] == finish_reason
    assert event['returned_model'] == 'gpt-oss-120b'
    assert event['usage']['total_tokens'] == 2212
    assert event['usage_reported'] is True
    assert InvestigationBudget(store, case_id).tokens == 2212
    assert 'Private incomplete output' not in str(store.events(case_id))


def test_fireworks_malformed_tool_arguments_fail_without_losing_usage_or_logging_arguments(saved_case):
    from langchain_fireworks.chat_models import _convert_dict_to_message

    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    # Use the installed Fireworks parser to reproduce a real invalid-tool turn.
    message = _convert_dict_to_message({
        'role': 'assistant', 'content': 'Private explanation',
        'tool_calls': [{'id': 'call-bad', 'type': 'function', 'function': {
            'name': 'search_knowledge_base', 'arguments': '{"query": "private snippet"',
        }}],
    })
    assert len(message.invalid_tool_calls) == 1
    message.response_metadata = {'finish_reason': 'tool_calls', 'model_name': 'gpt-oss-120b'}
    message.usage_metadata = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
    output = ModelResponse(result=[message])
    with pytest.raises(AgentOutputError, match='malformed tool') as caught:
        BudgetMiddleware(budget).wrap_model_call(None, lambda _: output)
    assert caught.value.response is output
    assert budget.tokens == 15
    event = store.events(case_id)[-1]
    assert event['tool_call_count'] == 0
    assert event['invalid_tool_call_count'] == 1
    assert event['usage']['total_tokens'] == 15
    assert 'private snippet' not in str(store.events(case_id))
    assert 'Private explanation' not in str(store.events(case_id))
    assert message.invalid_tool_calls[0]['args'] == '{"query": "private snippet"'


def test_unrecognized_response_metadata_cannot_copy_arbitrary_provider_text_to_audit(saved_case):
    store, case_id = saved_case
    budget = InvestigationBudget(store, case_id)
    message = AIMessage(content='Done', response_metadata={
        'finish_reason': 'private provider error and request body',
        'model_name': 'model\nAuthorization: private-key',
    })
    output = ModelResponse(result=[message])
    assert BudgetMiddleware(budget).wrap_model_call(None, lambda _: output) is output
    event = store.events(case_id)[-1]
    assert event['finish_reason'] == 'other'
    assert event['returned_model'] == ''
    assert event['tool_call_count'] == 0
    assert event['invalid_tool_call_count'] == 0
    assert 'private' not in str(store.events(case_id))
