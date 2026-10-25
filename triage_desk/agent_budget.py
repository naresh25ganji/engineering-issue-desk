"""Shared, thread-safe limits and audit events for a whole investigation.

All specialists share this object. Limits apply across retries and revisions;
human review time does not count toward the active invocation's deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import re
import threading
import time
import uuid

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessage


class BudgetExceeded(RuntimeError):
    """Stop execution before another provider or tool call is dispatched."""


class AgentOutputError(RuntimeError):
    """A safe failure with the unmodified provider response available privately.

    Response content is never part of the exception message or audit events.
    Retaining it lets a caller inspect or recover the failed turn deliberately.
    """

    def __init__(self, message: str, *, response=None):
        super().__init__(message)
        self.response = response


_FINISH_REASONS = frozenset({
    'stop', 'length', 'tool_calls', 'function_call', 'content_filter',
    'max_tokens', 'max_output_tokens', 'error', 'eos_token',
})
_TRUNCATED_REASONS = frozenset({'length', 'max_tokens', 'max_output_tokens'})
_MODEL_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z')


def _output_metadata(messages):
    """Keep only bounded identifiers, fixed labels, and counts from model output."""
    reasons, models = [], []
    tool_count, invalid_count = 0, 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        metadata = message.response_metadata
        reason = metadata.get('finish_reason')
        if reason is not None:
            label = reason if isinstance(reason, str) and reason in _FINISH_REASONS else 'other'
            if label not in reasons:
                reasons.append(label)
        model = metadata.get('model_name') or metadata.get('model')
        if isinstance(model, str) and _MODEL_IDENTIFIER.fullmatch(model) and model not in models:
            models.append(model)
        tool_count += len(message.tool_calls)
        invalid_count += len(message.invalid_tool_calls)
    return {
        'finish_reason': reasons[0] if len(reasons) == 1 else 'mixed' if reasons else 'unknown',
        'returned_model': models[0] if len(models) == 1 else 'multiple' if models else '',
        'tool_call_count': tool_count,
        'invalid_tool_call_count': invalid_count,
    }, bool(_TRUNCATED_REASONS.intersection(reasons))


class InvestigationBudget:
    def __init__(self, store, case_id: str, *, max_model_calls=18,
                 max_tool_calls=28, max_tokens=64000, max_seconds=180):
        self.store, self.case_id = store, case_id
        self.max_model_calls, self.max_tool_calls = max_model_calls, max_tool_calls
        self.max_tokens, self.max_seconds = max_tokens, max_seconds
        self.started = time.monotonic()
        self.lock = threading.RLock()
        prior = store.events(case_id)
        self.model_calls = sum(e.get('phase') == 'model' for e in prior)
        self.tool_calls = sum(e.get('phase') == 'tool' for e in prior)
        self.tokens = sum(e.get('usage', {}).get('total_tokens', 0) or 0 for e in prior)

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise BudgetExceeded('The investigation reached its active-time limit.')
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded('The investigation reached its reported-token limit.')

    def event(self, **event):
        self.store.add_event(self.case_id, event)

    @contextmanager
    def call(self, phase, name):
        with self.lock:
            self.check()
            field = 'model_calls' if phase == 'model' else 'tool_calls'
            maximum = self.max_model_calls if phase == 'model' else self.max_tool_calls
            if getattr(self, field) >= maximum:
                raise BudgetExceeded(f'The investigation reached its {phase}-call limit.')
            setattr(self, field, getattr(self, field) + 1)
        started = time.monotonic()
        # Do not persist tool arguments: they can include user code or credentials.
        call_id = uuid.uuid4().hex
        self.event(phase=phase, name=name, status='started', call_id=call_id)
        event = {'phase': phase + '_result', 'name': name, 'status': 'completed', 'call_id': call_id,
                 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            yield event
        except Exception as exc:
            event['status'] = 'limited' if isinstance(exc, BudgetExceeded) else 'error'
            event['error_type'] = type(exc).__name__
            raise
        finally:
            event['latency_ms'] = round((time.monotonic() - started) * 1000, 2)
            self.event(**event)

    def tool_call(self, name, arguments=None):
        return self.call('tool', name)


DOMAIN_TOOLS = {'search_knowledge_base', 'search_web', 'fetch_web_page',
                'read_github_issue', 'read_github_pr',
                'list_repository_files', 'search_repository', 'read_repository_file'}


def _present_evidence(result):
    """Give models the registered spans once, rather than duplicating raw text.

    The full immutable source remains in the case registry. Spans collectively
    cover the same source text and carry the IDs needed for report citations.
    """
    if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
        return result
    try:
        payload = json.loads(result.content)
    except (ValueError, TypeError):
        return result

    def compact(record):
        if isinstance(record, dict) and isinstance(record.get('quote_spans'), list):
            return {key: value for key, value in record.items() if key != 'text'}
        return record

    if isinstance(payload, dict) and isinstance(payload.get('evidence'), list):
        payload = {**payload, 'evidence': [compact(record) for record in payload['evidence']]}
    elif isinstance(payload, dict) and 'evidence_id' in payload:
        payload = compact(payload)
    return result.model_copy(update={'content': json.dumps(payload, ensure_ascii=False)})


class BudgetMiddleware(AgentMiddleware):
    """Instrument built-in harness tools and every model, including specialists."""

    def __init__(self, budget: InvestigationBudget, role='investigator'):
        self.budget, self.role = budget, role

    def wrap_model_call(self, request, handler):
        with self.budget.call('model', self.role) as event:
            response = handler(request)
            usage = [getattr(message, 'usage_metadata', None) for message in response.result]
            usage = [value for value in usage if isinstance(value, dict)]
            total = {}
            for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                values = [value[field] for value in usage
                          if isinstance(value.get(field), int) and not isinstance(value[field], bool)]
                if values:
                    total[field] = sum(values)
            event['usage'] = total
            event['usage_reported'] = bool(total)
            with self.budget.lock:
                self.budget.tokens += total.get('total_tokens', 0)
            metadata, truncated = _output_metadata(response.result)
            event.update(metadata)
            if truncated:
                raise AgentOutputError('The model response reached its output limit before completion.',
                                       response=response)
            if metadata['invalid_tool_call_count']:
                raise AgentOutputError('The model returned malformed tool arguments.', response=response)
            return response

    def wrap_tool_call(self, request, handler):
        name = request.tool_call['name']
        # Domain tools instrument their own errors and retries. Counting them
        # here as well would charge two tool calls for the same execution.
        if name in DOMAIN_TOOLS:
            return _present_evidence(handler(request))
        with self.budget.tool_call(name) as event:
            if name == 'task':
                role = request.tool_call.get('args', {}).get('subagent_type')
                if role in {'general-purpose', 'correctness_security', 'test_reproduction'}:
                    event['specialist'] = role
            result = handler(request)
            return _present_evidence(result) if name == 'read_case_evidence' else result


class BoundedContextMiddleware(AgentMiddleware):
    """Replace automatic model-powered summaries with a deterministic context cap.

    Deep Agents' automatic summarizer invokes a model outside wrap_model_call.
    This bounded workflow instead stops before its input grows too large, keeping
    every model call accounted for. The public same-name replacement hook is
    applied to the parent and each specialist independently.
    """

    @property
    def name(self):
        return 'SummarizationMiddleware'

    def before_model(self, state, runtime):
        messages = [message.model_dump(mode='json') for message in state.get('messages', [])]
        if len(json.dumps(messages, ensure_ascii=False)) > 80000:
            raise BudgetExceeded('The investigation reached its context-size limit.')
        return None
