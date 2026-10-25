"""Display flat agent accounting and existing nested retrieval diagnostics."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def workspace(document):
    from unittest.mock import patch
    # Rendering an exported evaluation never needs local provider configuration.
    with patch('dotenv.load_dotenv', return_value=False):
        from app import render_evaluation_summary
    render_evaluation_summary(document)


def agent_report():
    return {
        'report_kind': 'deep_agent_workflow', 'execution_mode': 'live',
        'summary': {
            'scheduled_cases': 3, 'completed_handoffs': 2, 'failed_cases': 1, 'handoff_rate': 2 / 3,
            'citation_integrity_passed_cases': 2, 'ai_rejected_findings': 1, 'ai_rejected_next_steps': 2,
            'model_calls': 7, 'tool_calls': 4, 'reported_tokens': 1200, 'usage_complete': False,
            'model_calls_by_role': {'investigator': 5, 'evidence_reviewer': 2},
            'errors_by_role_or_tool': {'investigator': 1},
        },
        'results': [
            {'eval_id': 'W3-01', 'case_id': 'saved-case', 'status': 'awaiting_review',
             'attempted': True, 'handoff_observed': True, 'elapsed_ms': 1500,
             'automatic_citation_integrity': {'status': 'passed', 'checked_supports': 2, 'failed_supports': 0},
             'independent_ai_review': {'status': 'completed'}, 'error': None},
            {'eval_id': 'W3-02', 'case_id': 'failed-case', 'status': 'failed', 'attempted': True,
             'handoff_observed': False, 'elapsed_ms': 3000,
             'automatic_citation_integrity': {'status': 'not_applicable', 'checked_supports': 0, 'failed_supports': 0},
             'error': {'type': 'InvestigationFailed', 'message': 'Saved safe failure'}},
        ],
    }


def test_flat_agent_report_displays_observed_handoffs_and_unscored_quality():
    app = AppTest.from_function(workspace, args=(agent_report(),)).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Scheduled cases'] == '3' and metrics['Saved handoffs'] == '2'
    assert metrics['Failed cases'] == '1' and metrics['Handoff completion'] == '67%'
    assert metrics['Model calls'] == '7' and metrics['Reported tokens'] == '1200'
    assert metrics['Findings removed by AI review'] == '1'
    table = app.dataframe[0].value
    assert table.iloc[0]['Evaluation case'] == 'W3-01'
    assert table.iloc[0]['Citation integrity'] == 'passed'
    assert table.iloc[1]['Status'] == 'failed' and table.iloc[1]['Error'] == 'InvestigationFailed'
    assert set(table['Human quality']) == {'Unscored'}
    assert any('Token usage is incomplete' in caption.value for caption in app.caption)
    assert any('Human accuracy, relevance, and citation entailment are unscored' in info.value for info in app.info)
    assert not any('Expected source found' in str(frame.value.columns) for frame in app.dataframe)


def test_validation_inventory_is_not_presented_as_zero_percent_success():
    report = agent_report()
    report['summary'].update(handoff_rate=None, usage_complete=None, completed_handoffs=0)
    report['results'] = []
    app = AppTest.from_function(workspace, args=(report,)).run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == 'Handoff completion').value == 'Not run'
    assert any('Validation-only inventory' in caption.value for caption in app.caption)


def test_existing_retrieval_report_keeps_strategy_metrics_and_caption():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'evaluation' / 'results' / 'report.json').read_text())
    app = AppTest.from_function(workspace, args=(report,)).run()
    assert not app.exception
    assert not app.metric
    table = app.dataframe[0].value
    assert 'Expected source found' in table.columns and 'Warm retrieval p95' in table.columns
    assert len(table) >= 1
    assert any('diagnostic questions' in caption.value for caption in app.caption)
    assert not app.info
