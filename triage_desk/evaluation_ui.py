"""Read-only views of the Week 4 reviewer experiments."""
from __future__ import annotations

import streamlit as st


_METRICS = {
    'case_exact_match': 'Case verdict agreement',
    'unsupported_rejection': 'Unsupported items rejected',
    'supported_retention': 'Supported items retained',
    'p95_latency_seconds': 'Review latency p95',
    'mean_total_tokens': 'Mean reported tokens',
}


def _display(name, value):
    if value is None:
        return 'Not measured'
    if name in ('case_exact_match', 'unsupported_rejection', 'supported_retention'):
        return f'{value:.1%}'
    if name == 'p95_latency_seconds':
        return f'{value:.2f} s'
    return f'{value:,.0f}'


def render_week4_evaluation(document: dict) -> None:
    """Render exported evidence; viewing reports never initializes a provider."""
    paired = document.get('report_kind') == 'week4_paired_comparison'
    st.markdown('**Week 4 evidence reviewer evaluation**')
    st.caption('Component evaluation of the Week 3 evidence reviewer. Full-agent research, delegation, and end-to-end usefulness are outside these scores.')
    manifest = document.get('improved', {}) if paired else document.get('manifest', {})
    label_status = document.get('label_status', manifest.get('label_status', 'unknown'))
    if label_status != 'human_reviewed':
        st.info('Reference labels are AI-proposed and await human review. Scores measure agreement with those labels; they do not establish human-validated accuracy.')
    st.caption(f"Prompt: {manifest.get('prompt_version', 'unknown')} · Label status: {label_status}")
    st.caption('Dataset: ' + str(manifest.get('dataset_name', 'Not recorded'))
               + ' · version ' + str(manifest.get('dataset_version', 'unknown')))

    if paired:
        totals = document.get('summary', {})
        columns = st.columns(4)
        for column, key, label in zip(columns, ('fixed', 'regression', 'unchanged_pass', 'unchanged_failure'),
                                      ('Cases fixed', 'Cases regressed', 'Still passing', 'Still failing')):
            column.metric(label, totals.get(key, 0))
        scores = document.get('metrics', {})
        baseline = scores.get('baseline', {}).get('metrics', {})
        improved = scores.get('improved', {}).get('metrics', {})
        rows = []
        for name, label in _METRICS.items():
            before, after = baseline.get(name, {}), improved.get(name, {})
            rows.append({'Metric': label, 'Baseline': _display(name, before.get('value')),
                         'Improved': _display(name, after.get('value')),
                         'Pass bar': _display(name, after.get('threshold')),
                         'Result': after.get('status', 'not_measured')})
        st.dataframe(rows, hide_index=True, width='stretch')
        if document.get('clusters'):
            st.markdown('**Failure clusters**')
            st.caption('Human review time is an explicit rough assumption, not an observed business cost.')
            st.dataframe(document['clusters'], hide_index=True, width='stretch')
        if document.get('cases'):
            st.markdown('**Paired cases**')
            st.dataframe(document['cases'], hide_index=True, width='stretch')
        return

    summary = document.get('summary', {})
    columns = st.columns(3)
    columns[0].metric('Scheduled cases', summary.get('scheduled_cases', 0))
    columns[1].metric('Completed reviews', summary.get('completed_cases', 0))
    columns[2].metric('Review errors', summary.get('error_cases', 0))
    metrics = summary.get('metrics', {})
    st.dataframe([{'Metric': label, 'Observed': _display(name, metrics.get(name, {}).get('value')),
                   'Pass bar': _display(name, metrics.get(name, {}).get('threshold')),
                   'Result': metrics.get(name, {}).get('status', 'not_measured')}
                  for name, label in _METRICS.items()], hide_index=True, width='stretch')
    if not manifest.get('live'):
        st.caption('Validation-only inventory. No model was called and no quality scores were measured.')
    st.caption('Dollar cost is unknown unless explicit provider rates were supplied. Token counts include only returned usage.')
    rows = []
    for item in document.get('rows', []):
        scores = item.get('scores', {})
        rows.append({'Case': item.get('id'), 'Scenario': item.get('scenario'),
                     'Status': item.get('status'),
                     'Agreement': ('PASS' if scores.get('exact_match') else 'FAIL')
                     if item.get('status') == 'completed' else 'FAIL / no verdict'
                     if item.get('status') == 'error' else 'Not run',
                     'Unsupported retained': scores.get('false_positives', 0),
                     'Supported removed': scores.get('false_negatives', 0),
                     'Missing verdicts': scores.get('unknown_verdicts'),
                     'Failure mode': item.get('failure_mode'),
                     'Time': _display('p95_latency_seconds', item.get('latency_seconds')),
                     'Human label review': item.get('label_review', {}).get('status', 'pending')})
    if rows:
        st.dataframe(rows, hide_index=True, width='stretch')
        selected = st.selectbox('Inspect evaluation case', document['rows'],
                                format_func=lambda item: f"{item['id']} · {item.get('title', '')}",
                                key='week4_case')
        with st.expander('Input, reference labels, and actual verdicts', expanded=False):
            st.json({key: selected.get(key) for key in
                     ('id', 'context', 'source', 'annotations', 'quality_review', 'scores', 'error')})
