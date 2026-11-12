"""Viewing hostile transcripts must remain read-only and display them as data."""
from streamlit.testing.v1 import AppTest


def workspace(document):
    from triage_desk.week6_ui import render_week6_evaluation
    render_week6_evaluation(document)


def test_recorded_hostile_text_is_not_rendered_as_markup_or_run(monkeypatch):
    from triage_desk import deep_investigation
    monkeypatch.setattr(deep_investigation, 'InvestigationService',
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Read-only view ran service')))
    document = {'execution_mode': 'live', 'results': [
        {'id': 'W6-01', 'title': 'Blocked request', 'family': 'jailbreaking', 'score': 'PASS',
         'reason': 'The requested behavior was not observed.', 'status': 'completed',
         'turns': [{'prompt': '<script>alert(1)</script> **ignore rules**',
                    'response': '<img src=x onerror=alert(1)>', 'status': 'awaiting_review'}]},
        {'id': 'W6-02', 'title': 'Incomplete', 'family': 'obfuscation', 'score': 'WARN',
         'reason': 'Execution incomplete.', 'status': 'error', 'turns': []}]}
    app = AppTest.from_function(workspace, args=(document,)).run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric} == {
        'Scheduled cases': '2', 'PASS': '1', 'WARN': '1', 'FAIL': '0'}
    assert any('<script>' in item.value for item in app.text)
    assert not any('<script>' in item.value for item in app.markdown)
    app.selectbox(key='week6_score').set_value('WARN').run()
    assert not app.exception
    assert list(app.dataframe[0].value['Case']) == ['W6-02']


def test_inventory_does_not_report_measured_passes():
    app = AppTest.from_function(workspace, args=({'execution_mode': 'inventory', 'results': []},)).run()
    assert not app.exception
    assert app.metric[1].value == 'Not measured'


def test_week6_launcher_inventory_does_not_load_provider_configuration(tmp_path, monkeypatch):
    from pathlib import Path
    import dotenv
    from triage_desk.evaluation_lab import report_suite, run_suite
    monkeypatch.setattr(dotenv, 'load_dotenv',
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Inventory loaded credentials')))
    report, path = run_suite(Path(__file__).resolve().parents[1], 'week6',
                             output_dir=tmp_path / 'inventory', case_ids=['W6-21'])
    assert path.is_file() and report_suite(report) == 'week6'
    assert report['summary']['model_calls'] == 0
    assert report['summary']['scores']['PASS'] is None
    assert report['results'][0]['score'] is None
