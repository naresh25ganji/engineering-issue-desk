"""Streamlit workspace for saved, human-reviewed deep investigations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import streamlit as st


_ENV_NAMES = {'FIREWORKS_API_KEY', 'FIREWORKS_MODEL', 'KEENABLE_API_KEY'}
_STATUS_LABELS = {
    'running': 'Investigating',
    'awaiting_review': 'Awaiting your review',
    'approved': 'Approved',
    'rejected': 'Rejected',
    'failed': 'Investigation interrupted',
    'error': 'Investigation interrupted',
}


def _plain(value: Any, *, limit: int = 20000) -> str:
    """Only primitive content crosses from agent records into UI strings."""
    if isinstance(value, (str, int, float, bool)):
        return str(value)[:limit]
    return ''


def _local_time(value: Any) -> str:
    text = _plain(value, limit=200)
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if parsed.tzinfo is not None:
            return parsed.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%b %d, %Y %I:%M %p %Z')
    except ValueError:
        pass
    return text


def _https_url(value: Any) -> str:
    """External source links are explicit, credential-free HTTPS URLs."""
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return ''
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                and not parsed.password and not any(ord(char) < 32 for char in value)):
            return value
    except ValueError:
        pass
    return ''


def _progress_text(event: Any) -> str:
    if not isinstance(event, dict):
        return _plain(event, limit=1000) or 'Investigation progress recorded.'
    parts = []
    for key in ('stage', 'event', 'type', 'name', 'status', 'message', 'error_code'):
        value = _plain(event.get(key), limit=600)
        if value and value not in parts:
            parts.append(value)
    if isinstance(event.get('todos'), list):
        parts.extend(_plain(task.get('status'), limit=30) + ': ' + _plain(task.get('content'), limit=180)
                     for task in event['todos'][:10] if isinstance(task, dict))
    if isinstance(event.get('latency_ms'), (int, float)):
        parts.append(str(round(event['latency_ms'])) + ' ms')
    tokens = event.get('usage', {}).get('total_tokens') if isinstance(event.get('usage'), dict) else None
    if isinstance(tokens, int):
        parts.append(str(tokens) + ' reported tokens')
    return ' · '.join(parts) or 'Investigation progress recorded.'


def _run_with_progress(action, label: str) -> dict:
    with st.status(label, expanded=True) as progress_box:
        result = action(lambda event: st.text(_progress_text(event)))
        failed = result.get('status') in {'failed', 'error'}
        progress_box.update(
            label='Investigation interrupted' if failed else 'Investigation saved',
            state='error' if failed else 'complete',
            expanded=failed,
        )
    return result


def _markdown_report(case: dict) -> str:
    report = case.get('report') or {}
    supplied = case.get('report_markdown') or report.get('report_markdown')
    if isinstance(supplied, str):
        return supplied
    lines = ['# Engineering investigation', '', _plain(case.get('question')), '',
             'Status: ' + _plain(case.get('status')), '', '## Findings']
    for finding in report.get('findings', []):
        if not isinstance(finding, dict):
            continue
        lines.extend(['', '- ' + _plain(finding.get('statement'))])
        for support in finding.get('support', []):
            if isinstance(support, dict):
                lines.append('  - [' + _plain(support.get('evidence_id')) + '] '
                             + _plain(support.get('quote')))
    for key, heading in (('next_steps', 'Next steps'), ('missing_information', 'Missing information')):
        lines.extend(['', '## ' + heading])
        values = report.get(key, [])
        for value in values if isinstance(values, list) else [values]:
            text = _plain(value)
            if text:
                lines.append('- ' + text)
    return '\n'.join(lines) + '\n'


def _render_report(case: dict) -> None:
    report = case.get('report')
    if not isinstance(report, dict):
        return
    st.subheader('Investigation report')
    st.caption('Citations preserve exact source passages. Review whether each passage supports the finding.')
    quality = report.get('quality_review') or {}
    if quality.get('status') in {'completed', 'no_content'}:
        st.caption('AI evidence review: ' + str(quality.get('retained_findings', len(report.get('findings', []))))
                   + ' findings retained, ' + str(quality.get('removed_findings', 0))
                   + ' findings and ' + str(quality.get('removed_next_steps', 0))
                   + ' proposed steps removed. This review can make mistakes.')
        with st.expander('Evidence review decisions'):
            for label, key in [('Finding', 'finding_verdicts'), ('Proposed step', 'next_step_verdicts')]:
                for verdict in quality.get(key, []):
                    st.text(label + ' ' + str(verdict.get('index', 0) + 1) + ': '
                            + ('retained' if verdict.get('supported') else 'removed'))
                    st.text(_plain(verdict.get('reason'), limit=400))
    elif case.get('status') == 'failed':
        st.info('Evidence review has not completed. This draft cannot be approved; retry the saved case to continue.')
    else:
        st.info('This draft predates the evidence-review step. Review its saved evidence before approving it.')
    findings = report.get('findings') or []
    if not findings:
        st.info('No findings are available yet. Review the next steps and missing information.')
    for index, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            continue
        with st.container(border=True):
            st.caption('Finding ' + str(index))
            st.text(_plain(finding.get('statement')))
            details = [_plain(finding.get(key), limit=100) for key in ('category', 'severity')]
            st.text(' · '.join(part for part in details if part))
            for support in finding.get('support') or []:
                if isinstance(support, dict):
                    st.text('Evidence ' + _plain(support.get('evidence_id'), limit=100))
                    st.text(_plain(support.get('quote')))
    for key, heading in (('next_steps', 'Next steps'), ('missing_information', 'Missing information')):
        values = report.get(key) or []
        if values:
            st.subheader(heading)
            for value in values if isinstance(values, list) else [values]:
                text = _plain(value)
                if text:
                    st.text(text)
    left, right = st.columns(2)
    with left:
        st.download_button('Download report (.md)', _markdown_report(case),
                           file_name='investigation-report.md', mime='text/markdown',
                           key='deep_download_md')
    with right:
        st.download_button('Download case (.json)', json.dumps(case, indent=2, default=str),
                           file_name='investigation-case.json', mime='application/json',
                           key='deep_download_json')


def _render_evidence(records: list[dict]) -> None:
    with st.expander('Sources and provenance', expanded=False):
        if not records:
            st.text('No sources were saved for this case.')
        for record in records:
            if not isinstance(record, dict):
                continue
            st.text(_plain(record.get('title') or record.get('evidence_id') or 'Saved source'))
            for key, label in (
                ('evidence_id', 'Evidence'), ('source_key', 'Source'), ('kind', 'Kind'),
                ('evidence_type', 'Evidence type'), ('version_scope', 'Version scope'),
                ('provider', 'Provider'), ('origin', 'Origin'), ('version', 'Version'),
                ('retrieved_at', 'Retrieved'), ('fetched_at', 'Fetched'),
                ('snapshot_at', 'Snapshot'), ('commit_sha', 'Commit'),
                ('base_sha', 'Base commit'), ('filename', 'File'), ('truncated', 'Truncated'),
                ('published_at', 'Published'), ('acquired_at', 'Indexed'),
                ('content_hash', 'Content hash'), ('sha256', 'Content hash'),
            ):
                value = _plain(record.get(key), limit=1000)
                if value:
                    if key in {'retrieved_at', 'fetched_at', 'snapshot_at', 'published_at', 'acquired_at'}:
                        value = _local_time(value)
                    st.text(label + ': ' + value)
            provenance = record.get('provenance')
            if isinstance(provenance, dict):
                for key, value in provenance.items():
                    name, text = _plain(key, limit=100), _plain(value, limit=1000)
                    if name and text:
                        st.text(name + ': ' + text)
            quote = _plain(record.get('quote') or record.get('text') or record.get('content'))
            if quote:
                st.text(quote)
            url = _https_url(record.get('url'))
            if url:
                st.link_button('Open source', url)
            st.divider()


def _render_timeline(events: list[dict]) -> None:
    with st.expander('Recorded timeline', expanded=False):
        if not events:
            st.text('No progress events have been recorded yet.')
        for event in events:
            if isinstance(event, dict):
                timestamp = _local_time(event.get('created_at') or event.get('timestamp'))
                if timestamp:
                    st.text(timestamp)
            st.text(_progress_text(event))


def render_deep_investigation(root: Path, desk, *, service=None) -> None:
    """Render without running an agent until a user submits a form.

    ``service`` injection keeps UI tests offline and gives each session its own
    service instance; no stateful service is stored in Streamlit's shared cache.
    """
    if service is None:
        from triage_desk.deep_investigation import AgentSettings, InvestigationService
        settings = AgentSettings.from_env()
        service = InvestigationService(root, desk, settings=settings)
    else:
        settings = service.settings
    readiness = settings.readiness()
    ready = bool(readiness.get('ready'))
    st.subheader('Deep Investigation')
    st.caption('Investigate an issue with specialist research, supporting sources, and your review.')
    st.caption('Starting an investigation sends the report, supplied code, and selected evidence to Fireworks. '
               'Web research sends search queries and requested page URLs to Keenable.')
    with st.container(border=True):
        model = _plain(readiness.get('model'), limit=250) or 'Not selected'
        st.text('Fireworks: ' + ('Ready' if ready else 'Setup needed'))
        st.text('Model: ' + model)
        st.text('Keenable: ' + ('Available' if readiness.get('search_enabled') else 'Not configured'))
        st.text('Workspace: ' + root.name)
        if readiness.get('tracing_enabled'):
            st.caption('LangSmith tracing is enabled.')
    if not ready:
        missing = [name for name in readiness.get('missing', []) if name in _ENV_NAMES]
        message = 'Set ' + ', '.join(missing) if missing else 'Set FIREWORKS_API_KEY and FIREWORKS_MODEL'
        st.info(message + ' in the workspace environment to start an investigation.')
    if not readiness.get('search_enabled'):
        st.caption('Set KEENABLE_API_KEY to include current web sources.')
    notice = st.session_state.pop('deep_notice', None)
    if notice:
        st.success(notice)
    with st.form('deep_new_investigation'):
        question = st.text_area('Issue report', height=130, max_chars=4500,
                                placeholder='Describe the behavior, expected result, and steps to reproduce.',
                                key='deep_question')
        version = st.text_input('Streamlit version (optional)', max_chars=100, key='deep_version')
        code_context = st.text_area('Code or traceback (optional)', height=110, max_chars=16000,
                                   key='deep_code_context')
        repository_context = st.checkbox('Include related application code', value=False,
                                         key='deep_repository_context',
                                         help='Allow bounded static reads of app.py and Python files under triage_desk. Selected code is sent to Fireworks. No code is executed.')
        submitted = st.form_submit_button('Start deep investigation', type='primary', disabled=not ready)
    if submitted and ready:
        if not question.strip():
            st.warning('Enter an issue report to start an investigation.')
        else:
            try:
                case = _run_with_progress(
                    lambda progress: service.start(question.strip(), version=version.strip(),
                                                   code_context=code_context.strip(), progress=progress,
                                                   **({'repository_context': True} if repository_context else {})),
                    'Investigating your report…',
                )
                st.session_state['deep_case_picker'] = case['case_id']
            except Exception:
                st.error('The investigation could not finish. Check the saved cases for recorded progress and try again.')
    try:
        cases = service.store.list_cases()
    except Exception:
        st.error('Saved investigations could not be loaded. Try again after checking workspace access.')
        return
    by_id = {case['case_id']: case for case in cases}
    options = [''] + list(by_id)
    if st.session_state.get('deep_case_picker', '') not in options:
        st.session_state['deep_case_picker'] = ''

    def case_label(case_id: str) -> str:
        if not case_id:
            return 'Choose a saved investigation'
        case = by_id[case_id]
        question = ' '.join(_plain(case.get('question'), limit=100).split())
        status = _STATUS_LABELS.get(case.get('status'), _plain(case.get('status'), limit=100))
        return question[:65] + ' · ' + status + ' · ' + case_id[:8]

    selected = st.selectbox('Saved investigations', options, format_func=case_label, key='deep_case_picker')
    if not selected:
        return
    try:
        case = service.store.get_case(selected)
        if not case:
            st.warning('This saved investigation is no longer available.')
            return
        evidence = service.store.evidence(selected)
        events = service.store.events(selected)
    except Exception:
        st.error('This investigation could not be loaded. Try selecting it again.')
        return
    st.divider()
    st.text(_STATUS_LABELS.get(case.get('status'), _plain(case.get('status'), limit=100)))
    st.text('Case: ' + _plain(case.get('case_id'), limit=100))
    st.caption('Related application code: ' + ('Enabled for this case' if case.get('repository_context') else 'Not enabled for this case'))
    st.text(_plain(case.get('question')))
    if case.get('version'):
        st.text('Version: ' + _plain(case.get('version'), limit=100))
    if case.get('updated_at'):
        st.text('Updated: ' + _local_time(case.get('updated_at')))
    if case.get('code_context'):
        with st.expander('Submitted code or traceback'):
            st.text(_plain(case['code_context']))
    if case.get('error'):
        st.warning('This investigation stopped before completion.')
        st.text(_plain(case['error'], limit=2000))
        from .agent_review import REVIEW_FAILURE_CODES, ReviewUnavailable
        failure_codes = [event.get('error_code') for event in events
                         if event.get('phase') == 'execution' and event.get('status') == 'error'
                         and event.get('error_code') in REVIEW_FAILURE_CODES]
        if failure_codes:
            code = failure_codes[-1]
            st.caption('Evidence review detail [' + code + ']: ' + str(ReviewUnavailable(code=code)))
    calls, tools, tokens = st.columns(3)
    calls.metric('Model calls', sum(event.get('phase') == 'model' for event in events))
    tools.metric('Tool calls', sum(event.get('phase') == 'tool' for event in events))
    usages = [event.get('usage', {}) for event in events if event.get('phase') == 'model_result']
    recorded_tokens = [usage['total_tokens'] for usage in usages if isinstance(usage.get('total_tokens'), int)]
    tokens.metric('Reported tokens', sum(recorded_tokens) if recorded_tokens else 'Unavailable')
    if len(recorded_tokens) < sum(event.get('phase') == 'model' for event in events):
        st.caption('Token usage is incomplete; missing provider usage is not estimated.')
    if case.get('status') in {'failed', 'error'}:
        if st.button('Retry investigation', disabled=not ready, key='deep_retry_' + selected):
            try:
                result = _run_with_progress(
                    lambda progress: service.retry(selected, progress=progress),
                    'Continuing the saved investigation…',
                )
                if result.get('status') not in {'failed', 'error'}:
                    st.session_state['deep_notice'] = 'Investigation continued and saved.'
                st.rerun()
            except Exception:
                st.error('The investigation could not continue. Its recorded progress remains saved.')
    _render_report(case)
    quality = (case.get('report') or {}).get('quality_review', {})
    if case.get('status') == 'awaiting_review' and quality.get('status') not in {'completed', 'no_content'}:
        if st.button('Review saved evidence', disabled=not ready, key='deep_audit_' + selected):
            try:
                _run_with_progress(lambda progress: service.audit(selected, progress=progress),
                                   'Checking the saved findings against their evidence…')
                st.rerun()
            except Exception:
                st.error('Evidence review could not finish. The saved draft remains unapproved.')
    if case.get('review_notes'):
        st.subheader('Review notes')
        st.text(_plain(case['review_notes']))
    if case.get('status') == 'awaiting_review':
        st.subheader('Review this report')
        with st.form('deep_review_' + selected):
            reviewed = (case.get('report') or {}).get('quality_review', {}).get('status') in {'completed', 'no_content'}
            actions = ['Approve report', 'Request revision', 'Reject report'] if reviewed else ['Request revision', 'Reject report']
            action = st.radio('Review decision', actions,
                              key='deep_review_decision_' + selected)
            notes = st.text_area('Review notes', key='deep_review_notes_' + selected)
            review_submitted = st.form_submit_button('Submit review')
        if review_submitted:
            decision = {'Approve report': 'approve', 'Request revision': 'revise', 'Reject report': 'reject'}[action]
            if decision == 'revise' and not notes.strip():
                st.warning('Describe the changes you want in the review notes.')
            elif decision == 'revise' and not ready:
                st.warning('Configure FIREWORKS_API_KEY and FIREWORKS_MODEL before requesting a revision.')
            else:
                try:
                    result = _run_with_progress(
                        lambda progress: service.resume(selected, decision, notes=notes.strip(), progress=progress),
                        'Saving your review…' if decision != 'revise' else 'Revising the investigation…',
                    )
                    if result.get('status') not in {'failed', 'error'}:
                        st.session_state['deep_notice'] = {
                            'approve': 'Report approved and saved.',
                            'reject': 'Report rejected and saved.',
                            'revise': 'Revision saved. Review the updated investigation.',
                        }[decision]
                    st.rerun()
                except Exception:
                    st.error('Your review could not be saved. The report remains available for review.')
    _render_evidence(evidence)
    _render_timeline(events)
