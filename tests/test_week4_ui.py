"""Measured, provisional, and validation-only Week 4 results stay distinct."""
from streamlit.testing.v1 import AppTest


def workspace(document):
    from triage_desk.evaluation_ui import render_week4_evaluation
    render_week4_evaluation(document)


def test_inventory_has_no_false_zero_quality_score():
    document = {'report_kind': 'week4_evidence_review',
                'manifest': {'live': False, 'prompt_version': 'baseline', 'label_status': 'ai_proposed_pending_human_review'},
                'summary': {'scheduled_cases': 40, 'completed_cases': 0, 'error_cases': 0}, 'rows': []}
    app = AppTest.from_function(workspace, args=(document,)).run()
    assert not app.exception
    assert any('await human review' in info.value for info in app.info)
    assert set(app.dataframe[0].value['Observed']) == {'Not measured'}
    assert any('No model was called' in caption.value for caption in app.caption)


def test_paired_view_exposes_regressions_and_pending_labels():
    document = {'report_kind': 'week4_paired_comparison',
                'label_status': 'ai_proposed_pending_human_review',
                'improved': {'prompt_version': 'scope'},
                'summary': {'fixed': 3, 'regression': 1, 'unchanged_pass': 35, 'unchanged_failure': 1},
                'metrics': {'baseline': {'metrics': {'case_exact_match': {'value': .875}}},
                            'improved': {'metrics': {'case_exact_match': {'value': .925, 'threshold': .9, 'status': 'passed'}}}}}
    app = AppTest.from_function(workspace, args=(document,)).run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == 'Cases regressed').value == '1'
    table = app.dataframe[0].value
    assert table.iloc[0]['Baseline'] == '87.5%' and table.iloc[0]['Improved'] == '92.5%'
    assert any('human-validated accuracy' in info.value for info in app.info)


def test_review_errors_are_scheduled_failures_with_unknown_verdicts():
    document = {'report_kind': 'week4_evidence_review', 'manifest': {'live': True},
                'summary': {'scheduled_cases': 1, 'completed_cases': 0, 'error_cases': 1},
                'rows': [{'id': 'W4-01', 'status': 'error', 'title': 'Provider unavailable',
                          'scores': {'exact_match': False, 'unknown_verdicts': 3}}]}
    app = AppTest.from_function(workspace, args=(document,)).run()
    assert not app.exception
    row = app.dataframe[1].value.iloc[0]
    assert row['Agreement'] == 'FAIL / no verdict' and row['Missing verdicts'] == 3
