"""Read-only views of exported engineering and citation-integrity evaluations.

Only Streamlit is imported: inspecting an export never constructs a service,
loads provider configuration, or executes a model. Source and report text are
shown through plain-text and JSON widgets rather than interpreted as markup.
"""
from __future__ import annotations

import streamlit as st


_METRICS = (
    ('handoff_rate', 'Saved handoff rate', '>=', .95),
    ('contract_pass_rate', 'Structural contract pass', '>=', .90),
    ('quoted_anchor_recall', 'Quoted anchor recall', '>=', .85),
    ('latency_p95_seconds', 'Case latency p95', '<=', 90),
    ('mean_reported_tokens', 'Mean reported tokens', '<=', 20000),
)
_RATES = frozenset({'handoff_rate', 'contract_pass_rate', 'quoted_anchor_recall', 'acceptance_match_rate'})


def _object(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _value(name, value):
    if value is None:
        return 'Not measured'
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if name in _RATES:
        return f'{value:.1%}'
    if name.endswith('_seconds'):
        return f'{value:.2f} s'
    if name.endswith('_ms'):
        return f'{value:.3f} ms'
    return f'{value:,.0f}'


def _contract(value):
    return 'PASS' if value is True else 'FAIL' if value is False else 'Not measured'


def _citation(row):
    integrity = _object(row.get('automatic_citation_integrity'))
    status = integrity.get('status')
    if status == 'not_applicable':
        return 'N/A (no nonempty citations)'
    if status == 'passed':
        return 'PASS'
    if status == 'failed':
        return 'FAIL'
    return 'Not measured'


def _failure(row):
    return (bool(row.get('error')) or _object(row.get('scores')).get('contract_passed') is False
            or _object(row.get('automatic_citation_integrity')).get('status') == 'failed')


def _target_rows(summary, *, live=True):
    targets = _object(summary.get('targets'))
    result = []
    for name, label, operator, threshold in _METRICS:
        specification = _object(targets.get(name))
        passed = specification.get('passed') if live else None
        result.append({'Metric': label,
                       'Observed': _value(name, summary.get(name) if live else None),
                       'Target': f"{specification.get('operator', operator)} {_value(name, specification.get('target', threshold))}",
                       'Result': _contract(passed)})
    return result


def _manifest(document):
    with st.expander('Run identity and measurement limits', expanded=False):
        st.json({'manifest': document.get('manifest'), 'measurement_limits': document.get('measurement_limits')})


def _workflow_groups(groups, *, live):
    result = []
    for identity, raw in groups.items():
        summary = _object(raw)
        result.append({'Use case': identity, 'Scheduled': summary.get('scheduled_cases'),
                       'Attempted': summary.get('attempted_cases'), 'Saved handoffs': summary.get('completed_handoffs'),
                       'Errors': summary.get('failed_cases'),
                       'Handoff rate': _value('handoff_rate', summary.get('handoff_rate') if live else None),
                       'Contract pass': _value('contract_pass_rate', summary.get('contract_pass_rate') if live else None),
                       'Quoted anchor recall': _value('quoted_anchor_recall', summary.get('quoted_anchor_recall') if live else None),
                       'Citation passes': summary.get('citation_integrity_passed_cases'),
                       'Citation applicable cases': summary.get('citation_integrity_applicable_cases'),
                       'Reported tokens': summary.get('reported_tokens') if live else None,
                       'Human reviews pending': summary.get('human_rubric_pending_cases')})
    return result


def _case_filters(rows, *, prefix):
    profiles = sorted({str(row.get('use_case', 'unknown')) for row in rows})
    selected = st.selectbox('Filter by use case', ['All use cases', *profiles], key=f'{prefix}_profile')
    failures_only = st.toggle('Show failures only', value=False, key=f'{prefix}_failures')
    return [row for row in rows if (selected == 'All use cases' or row.get('use_case') == selected)
            and (not failures_only or _failure(row))]


def _inspect_workflow(rows):
    if not rows:
        st.info('No cases match the current filters.')
        return
    chosen = st.selectbox('Inspect workflow case', rows,
                          format_func=lambda row: f"{row.get('id', row.get('eval_id', 'unknown'))} · {row.get('title', '')}",
                          key='engineering_workflow_case')
    inputs = _object(chosen.get('input'))
    with st.expander('Question and supplied context', expanded=True):
        st.text('Question')
        st.text(chosen.get('question', inputs.get('question', 'Not exported')))
        st.text('Supplied context')
        st.text(inputs.get('context', 'Not exported'))
        st.json({'fixture': chosen.get('fixture'), 'input_sha256': chosen.get('input_sha256')})
    with st.expander('Actual report, reviewer output, and error', expanded=False):
        st.json({'status': chosen.get('status'), 'report': chosen.get('report'),
                 'independent_ai_review': chosen.get('independent_ai_review'), 'error': chosen.get('error')})
    with st.expander('Registered sources and citation checks', expanded=False):
        st.json({'sources': chosen.get('sources'),
                 'automatic_citation_integrity': chosen.get('automatic_citation_integrity')})
    with st.expander('Provisional expectations and human review', expanded=False):
        st.json({'expected': chosen.get('expected'), 'scores': chosen.get('scores'),
                 'label_review': chosen.get('label_review'), 'human_review': chosen.get('human_review')})
    with st.expander('Observed accounting and local traces', expanded=False):
        st.json({'accounting': chosen.get('accounting'), 'trace': chosen.get('trace'), 'events': chosen.get('events')})


def _render_workflow(document):
    st.markdown('**Engineering workflow evaluation**')
    st.caption('Complete investigation, independent AI evidence review, and a saved handoff for human review.')
    st.info('Structural contracts and quoted anchor coverage do not measure semantic quality. Human accuracy, relevance, and citation entailment remain pending; reference labels are AI-proposed.')
    summary = _object(document.get('summary'))
    manifest = _object(document.get('manifest'))
    live = document.get('execution_mode') == 'live' or manifest.get('live') is True
    if not live:
        st.caption('Validation-only inventory. No model was called; success rates and quality scores are not measured.')
    columns = st.columns(5)
    for column, (name, label, operator, threshold) in zip(columns, _METRICS, strict=True):
        column.metric(label, _value(name, summary.get(name) if live else None),
                      help=f'Target {operator} {_value(name, threshold)}. Automatic workflow measurement only.')
    st.dataframe(_target_rows(summary, live=live), hide_index=True, width='stretch')
    rows = _rows(document.get('results'))
    counts = st.columns(4)
    counts[0].metric('Scheduled cases', summary.get('scheduled_cases', len(rows)))
    counts[1].metric('Saved handoffs', summary.get('completed_handoffs', 0))
    counts[2].metric('Actual case errors', sum(bool(row.get('error')) for row in rows))
    passes = sum(_object(row.get('scores')).get('contract_passed') is True for row in rows)
    counts[3].metric('Cases passing contracts', f'{passes} / {len(rows)}' if live else 'Not measured')
    accounting = st.columns(3)
    accounting[0].metric('Observed model calls', summary.get('model_calls', 0))
    accounting[1].metric('Observed tool calls', summary.get('tool_calls', 0))
    accounting[2].metric('Observed reported tokens', summary.get('reported_tokens', 0))
    if live and summary.get('usage_complete') is not True:
        st.caption('Token usage is incomplete or unknown. Totals include only returned provider usage; the token target cannot pass and dollar cost is unknown.')
    else:
        st.caption('Token totals include returned provider usage only. Dollar cost is not inferred.')
    applicable = sum(_object(row.get('automatic_citation_integrity')).get('status') in {'passed', 'failed'} for row in rows)
    citation_passes = sum(_object(row.get('automatic_citation_integrity')).get('status') == 'passed' for row in rows)
    not_applicable = sum(_object(row.get('automatic_citation_integrity')).get('status') == 'not_applicable' for row in rows)
    st.text(f'Literal citation checks: {citation_passes} / {applicable} applicable cases passed; {not_applicable} N/A with no nonempty citations. N/A is not a citation pass.')
    st.caption('Provider errors and unstarted cases stay in scheduled rate denominators. Quoted anchor recall is literal fixture coverage inside valid quotes, not proof that a claim is correct.')
    groups = _workflow_groups(_object(document.get('by_use_case')), live=live)
    if groups:
        st.markdown('**Results by use case**')
        st.dataframe(groups, hide_index=True, width='stretch')
    _manifest(document)
    visible = _case_filters(rows, prefix='engineering_workflow')
    if visible:
        st.dataframe([{'Case': row.get('id', row.get('eval_id')), 'Use case': row.get('use_case'),
                       'Title': row.get('title'), 'Status': row.get('status'),
                       'Contract': _contract(_object(row.get('scores')).get('contract_passed')),
                       'Citation integrity': _citation(row),
                       'Quoted anchor recall': _value('quoted_anchor_recall', _object(row.get('scores')).get('quoted_anchor_recall')),
                       'Error': _object(row.get('error')).get('type', ''),
                       'Time': _value('elapsed_ms', row.get('elapsed_ms')), 'Human quality': 'Pending'}
                      for row in visible], hide_index=True, width='stretch')
    _inspect_workflow(visible)


def _pair_outcome(row):
    before, after = row.get('baseline_contract_passed'), row.get('candidate_contract_passed')
    if before is False and after is True:
        return 'Fixed'
    if before is True and after is False:
        return 'Regressed'
    if before is True and after is True:
        return 'Still passing'
    if before is False and after is False:
        return 'Still failing'
    return 'Not measured'


def _render_comparison(document):
    st.markdown('**Engineering workflow comparison**')
    st.info('Matched diagnostic cases with provisional labels. Contract changes and quoted anchor coverage do not establish human-validated accuracy; semantic quality review remains pending.')
    pairs = _rows(document.get('pairs'))
    columns = st.columns(4)
    for column, label, outcome in zip(columns,
                                     ('Cases fixed', 'Cases regressed', 'Still passing', 'Still failing'),
                                     ('Fixed', 'Regressed', 'Still passing', 'Still failing'), strict=True):
        column.metric(label, sum(_pair_outcome(row) == outcome for row in pairs))
    baseline = _object(document.get('baseline_summary'))
    candidate = _object(document.get('candidate_summary'))
    changes = _object(document.get('metric_deltas'))
    targets = _object(candidate.get('targets'))
    metrics = []
    for name, label, operator, threshold in _METRICS:
        target = _object(targets.get(name))
        metrics.append({'Metric': label, 'Baseline': _value(name, baseline.get(name)),
                        'Candidate': _value(name, candidate.get(name)), 'Delta': _value(name, changes.get(name)),
                        'Target': f"{target.get('operator', operator)} {_value(name, target.get('target', threshold))}",
                        'Candidate result': _contract(target.get('passed'))})
    st.dataframe(metrics, hide_index=True, width='stretch')
    for key, label in (('baseline_by_use_case', '**Baseline by use case**'),
                       ('candidate_by_use_case', '**Candidate by use case**')):
        groups = _workflow_groups(_object(document.get(key)), live=True)
        if groups:
            st.markdown(label)
            st.dataframe(groups, hide_index=True, width='stretch')
    with st.expander('Paired run identities and measurement limits', expanded=False):
        st.json({key: document.get(key) for key in ('baseline', 'candidate', 'dataset_sha256', 'measurement_limits')})
    profiles = sorted({str(row.get('use_case', 'unknown')) for row in pairs})
    selected = st.selectbox('Filter by use case', ['All use cases', *profiles], key='engineering_comparison_profile')
    failures_only = st.toggle('Show candidate failures only', value=False, key='engineering_comparison_failures')
    visible = [row for row in pairs if (selected == 'All use cases' or row.get('use_case') == selected)
               and (not failures_only or row.get('candidate_contract_passed') is False)]
    if not visible:
        st.info('No paired cases match the current filters.')
        return
    st.dataframe([{'Case': row.get('id'), 'Use case': row.get('use_case'), 'Change': _pair_outcome(row),
                   'Baseline contract': _contract(row.get('baseline_contract_passed')),
                   'Candidate contract': _contract(row.get('candidate_contract_passed')),
                   'Baseline anchor recall': _value('quoted_anchor_recall', row.get('baseline_quoted_anchor_recall')),
                   'Candidate anchor recall': _value('quoted_anchor_recall', row.get('candidate_quoted_anchor_recall'))}
                  for row in visible], hide_index=True, width='stretch')
    selected_pair = st.selectbox('Inspect paired case', visible,
                                format_func=lambda row: f"{row.get('id')} · {_pair_outcome(row)}",
                                key='engineering_comparison_case')
    with st.expander('Exported paired measurements', expanded=False):
        st.json(selected_pair)
    st.caption('Open each source run to inspect its actual question, context, report, evidence, and local trace.')


def _acceptance(value):
    return 'Accepted' if value is True else 'Rejected' if value is False else 'Not measured'


def _render_integrity(document):
    st.markdown('**Report and citation integrity checks**')
    st.info('These are structural and literal citation checks, not semantic accuracy. A matching citation attached to an unrelated claim is intentionally accepted by this layer. Human or semantic review must assess whether the quote supports the claim.')
    summary = _object(document.get('summary'))
    columns = st.columns(5)
    columns[0].metric('Acceptance matches', f"{summary.get('passed_cases', 0)} / {summary.get('scheduled_cases', 0)}")
    columns[1].metric('Expected accepts observed', f"{summary.get('valid_accepted', 0)} / {summary.get('valid_denominator', 0)}")
    columns[2].metric('Expected rejects observed', f"{summary.get('invalid_rejected', 0)} / {summary.get('invalid_denominator', 0)}")
    columns[3].metric('Model calls', summary.get('model_calls', _object(document.get('manifest')).get('model_calls')))
    columns[4].metric('Validator latency p95', _value('p95_latency_ms', summary.get('p95_latency_ms')))
    st.caption('Controlled synthetic fixtures, no model or network calls. Empty findings can be schema-valid while task completion and usefulness remain unmeasured.')
    _manifest(document)
    rows = _rows(document.get('results'))
    failures_only = st.toggle('Show mismatches only', value=False, key='engineering_integrity_failures')
    visible = [row for row in rows if not failures_only or row.get('passed') is False]
    if not visible:
        st.info('No acceptance mismatches match the current filter.')
        return
    st.dataframe([{'Case': row.get('id'), 'Check': row.get('title'), 'Category': row.get('category'),
                   'Expected': _acceptance(row.get('expected_acceptance')),
                   'Observed': _acceptance(row.get('observed_acceptance')),
                   'Acceptance match': _contract(row.get('passed')),
                   'Time': _value('elapsed_ms', row.get('elapsed_ms')),
                   'Validation error': row.get('validation_error') or ''} for row in visible],
                 hide_index=True, width='stretch')
    chosen = st.selectbox('Inspect integrity case', visible,
                          format_func=lambda row: f"{row.get('id')} · {row.get('title', '')}",
                          key='engineering_integrity_case')
    with st.expander('Actual validator input and output', expanded=True):
        st.json({'input_report': chosen.get('input_report'),
                 'output': {key: chosen.get(key) for key in ('expected_acceptance', 'observed_acceptance',
                                                           'passed', 'validation_error', 'elapsed_ms')}})
    with st.expander('Controlled source text', expanded=False):
        st.json(document.get('sources', []))


def render_engineering_evaluation(document: dict) -> None:
    """Display existing exports without executing any evaluation or provider."""
    kind = document.get('report_kind')
    if kind == 'use_case_workflow':
        _render_workflow(document)
    elif kind == 'use_case_comparison':
        _render_comparison(document)
    elif kind == 'evidence_integrity':
        _render_integrity(document)
    else:
        st.info('This engineering evaluation report format is not supported.')
