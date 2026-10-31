"""Actual structural checks, failed workflows, and unmeasured quality stay distinct."""
from copy import deepcopy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def workspace(document):
    from triage_desk.workflow_evaluation_ui import render_engineering_evaluation
    render_engineering_evaluation(document)


def workflow_report():
    return {
        'report_kind': 'use_case_workflow', 'execution_mode': 'live',
        'manifest': {'live': True, 'reviewer_prompt_version': 'baseline',
                     'label_status': 'ai_proposed_pending_human_review'},
        'summary': {'scheduled_cases': 3, 'completed_handoffs': 2, 'failed_cases': 1,
                    'handoff_rate': 2 / 3, 'contract_pass_rate': 1 / 3, 'quoted_anchor_recall': .5,
                    'latency_p95_seconds': 2.9, 'mean_reported_tokens': 400,
                    'model_calls': 6, 'tool_calls': 3, 'reported_tokens': 1200, 'usage_complete': False,
                    'targets': {'handoff_rate': {'operator': '>=', 'target': .95, 'measured': 2 / 3, 'passed': False},
                                'mean_reported_tokens': {'operator': '<=', 'target': 20000, 'measured': 400, 'passed': None}}},
        'by_use_case': {'incident_triage': {'scheduled_cases': 1, 'attempted_cases': 1, 'completed_handoffs': 1,
                                         'failed_cases': 0, 'handoff_rate': 1, 'contract_pass_rate': 1,
                                         'quoted_anchor_recall': 1, 'citation_integrity_passed_cases': 1,
                                         'citation_integrity_applicable_cases': 1, 'reported_tokens': 400,
                                         'human_rubric_pending_cases': 1}},
        'results': [
            {'id': 'WF-01', 'use_case': 'incident_triage', 'title': 'Reported timeout', 'status': 'awaiting_review',
             'question': '**Untrusted question** <script>not markup</script>',
             'input': {'context': '10:05 UTC: operator reported a timeout. **untrusted source**'},
             'scores': {'contract_passed': True, 'quoted_anchor_recall': 1},
             'automatic_citation_integrity': {'status': 'passed', 'checked_supports': 1, 'failed_supports': 0},
             'report': {'findings': [{'statement': 'The operator reported a timeout.'}], 'next_steps': []},
             'sources': [{'title': 'Actual note', 'evidence_id': 'ev_note'}],
             'independent_ai_review': {'status': 'completed', 'removed_findings': 1},
             'error': None, 'elapsed_ms': 1000, 'events': [{'phase': 'model', 'name': 'investigator'}],
             'trace': {'metadata': {'remote_tracing': False}}, 'accounting': {'model_calls': 2}},
            {'id': 'WF-02', 'use_case': 'change_review', 'title': 'Provider failed', 'status': 'failed',
             'question': 'Review this diff.', 'input': {'context': 'def unchanged(): pass'},
             'scores': {'contract_passed': False, 'quoted_anchor_recall': 0},
             'automatic_citation_integrity': {'status': 'not_applicable', 'checked_supports': 0, 'failed_supports': 0},
             'report': None, 'error': {'type': 'InvestigationFailed', 'message': 'Actual safe error'}, 'elapsed_ms': 3000},
            {'id': 'WF-03', 'use_case': 'test_planning', 'title': 'Citation mismatch', 'status': 'awaiting_review',
             'question': 'Plan tests.', 'input': {'context': 'return size <= 100'},
             'scores': {'contract_passed': False, 'quoted_anchor_recall': 0},
             'automatic_citation_integrity': {'status': 'failed', 'checked_supports': 1, 'failed_supports': 1},
             'report': {'findings': []}, 'error': None, 'elapsed_ms': 2000},
        ],
    }


def table_with(app, column):
    return next(frame.value for frame in app.dataframe if column in frame.value.columns)


def test_workflow_view_reports_errors_contracts_and_citation_applicability_without_running_models(monkeypatch):
    from triage_desk import deep_investigation

    def forbidden(*args, **kwargs):
        raise AssertionError('Viewing exported results must not create a provider or service')
    monkeypatch.setattr(deep_investigation, 'InvestigationService', forbidden)
    monkeypatch.setattr(deep_investigation, 'fireworks_model', forbidden)
    app = AppTest.from_function(workspace, args=(workflow_report(),)).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Saved handoff rate'] == '66.7%' and metrics['Structural contract pass'] == '33.3%'
    assert metrics['Actual case errors'] == '1' and metrics['Cases passing contracts'] == '1 / 3'
    assert metrics['Observed reported tokens'] == '1200'
    contracts = table_with(app, 'Citation integrity')
    assert list(contracts['Contract']) == ['PASS', 'FAIL', 'FAIL']
    assert list(contracts['Citation integrity']) == ['PASS', 'N/A (no nonempty citations)', 'FAIL']
    assert contracts.iloc[1]['Error'] == 'InvestigationFailed'
    assert any('1 / 2 applicable cases passed; 1 N/A' in text.value for text in app.text)
    assert any('semantic quality' in info.value and 'remain pending' in info.value for info in app.info)
    assert any('Token usage is incomplete' in caption.value for caption in app.caption)
    targets = table_with(app, 'Target')
    assert targets.iloc[0]['Target'] == '>= 95.0%' and targets.iloc[0]['Result'] == 'FAIL'
    assert targets.iloc[4]['Result'] == 'Not measured'
    assert table_with(app, 'Human reviews pending').iloc[0]['Scheduled'] == 1
    assert any('**Untrusted question** <script>' in text.value for text in app.text)
    assert not any('Untrusted question' in item.value for item in app.markdown)
    details = '\n'.join(item.value for item in app.json)
    for exported in ('ev_note', 'Actual note', 'removed_findings', 'remote_tracing', 'investigator'):
        assert exported in details


def test_profile_and_failure_filters_drill_into_actual_error_row():
    app = AppTest.from_function(workspace, args=(workflow_report(),)).run()
    app.selectbox(key='engineering_workflow_profile').select('change_review').run()
    app.toggle(key='engineering_workflow_failures').set_value(True).run()
    assert not app.exception
    shown = table_with(app, 'Citation integrity')
    assert list(shown['Case']) == ['WF-02']
    assert any('Actual safe error' in output.value for output in app.json)
    assert any(text.value == 'def unchanged(): pass' for text in app.text)


def test_inventory_preserves_null_scores_and_reports_no_model_calls():
    report = workflow_report()
    report['execution_mode'] = 'validation_only'
    report['manifest']['live'] = False
    report['summary'].update(handoff_rate=None, contract_pass_rate=None, quoted_anchor_recall=None,
                             latency_p95_seconds=None, mean_reported_tokens=None, model_calls=0,
                             tool_calls=0, reported_tokens=0, completed_handoffs=0)
    report['by_use_case'] = {}
    for row in report['results']:
        row.update(status='not_run', error=None, elapsed_ms=None,
                   scores={'contract_passed': None, 'quoted_anchor_recall': None},
                   automatic_citation_integrity={'status': 'not_applicable', 'checked_supports': 0})
    app = AppTest.from_function(workspace, args=(report,)).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Saved handoff rate'] == 'Not measured'
    assert metrics['Structural contract pass'] == 'Not measured'
    assert metrics['Mean reported tokens'] == 'Not measured'
    assert metrics['Cases passing contracts'] == 'Not measured' and metrics['Observed model calls'] == '0'
    assert set(table_with(app, 'Target')['Observed']) == {'Not measured'}
    assert set(table_with(app, 'Citation integrity')['Contract']) == {'Not measured'}
    assert any('No model was called' in caption.value for caption in app.caption)


def test_comparison_exposes_fixes_regressions_and_group_metrics():
    summary = workflow_report()['summary']
    report = {'report_kind': 'use_case_comparison', 'baseline': {'reviewer_prompt_version': 'baseline'},
              'candidate': {'reviewer_prompt_version': 'calibrated'},
              'baseline_summary': deepcopy(summary), 'candidate_summary': summary,
              'candidate_by_use_case': workflow_report()['by_use_case'],
              'metric_deltas': {'contract_pass_rate': .25, 'quoted_anchor_recall': .1},
              'pairs': [{'id': 'WF-01', 'use_case': 'incident_triage',
                         'baseline_contract_passed': False, 'candidate_contract_passed': True,
                         'baseline_quoted_anchor_recall': .5, 'candidate_quoted_anchor_recall': 1},
                        {'id': 'WF-02', 'use_case': 'change_review',
                         'baseline_contract_passed': True, 'candidate_contract_passed': False,
                         'baseline_quoted_anchor_recall': 1, 'candidate_quoted_anchor_recall': None}]}
    app = AppTest.from_function(workspace, args=(report,)).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Cases fixed'] == '1' and metrics['Cases regressed'] == '1'
    assert list(table_with(app, 'Change')['Change']) == ['Fixed', 'Regressed']
    assert table_with(app, 'Delta').iloc[1]['Delta'] == '25.0%'
    assert table_with(app, 'Human reviews pending').iloc[0]['Use case'] == 'incident_triage'
    assert any('human-validated accuracy' in info.value for info in app.info)
    app.toggle(key='engineering_comparison_failures').set_value(True).run()
    assert list(table_with(app, 'Change')['Case']) == ['WF-02']


def test_actual_integrity_export_displays_all_checks_and_preserves_semantic_scope_limit():
    report = json.loads((ROOT / 'evaluation' / 'integrity-results' / 'report.json').read_text())
    app = AppTest.from_function(workspace, args=(report,)).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Acceptance matches'] == '30 / 30'
    assert metrics['Expected accepts observed'] == '7 / 7'
    assert metrics['Expected rejects observed'] == '23 / 23'
    assert metrics['Model calls'] == '0'
    checks = table_with(app, 'Acceptance match')
    assert len(checks) == 30 and set(checks['Acceptance match']) == {'PASS'}
    unrelated = checks[checks['Case'] == 'CI-05'].iloc[0]
    assert unrelated['Observed'] == 'Accepted', 'This layer deliberately cannot establish entailment.'
    assert any('unrelated claim is intentionally accepted' in info.value for info in app.info)
    assert any('input_report' in output.value and 'observed_acceptance' in output.value for output in app.json)
    app.toggle(key='engineering_integrity_failures').set_value(True).run()
    assert not app.exception
    assert any('No acceptance mismatches' in info.value for info in app.info)
