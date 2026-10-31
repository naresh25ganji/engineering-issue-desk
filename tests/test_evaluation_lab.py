"""Evaluation launchers validate locally and run models only on explicit action."""
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from triage_desk.evaluation_lab import list_reports, report_suite, run_suite

ROOT = Path(__file__).resolve().parents[1]


def test_new_suites_execute_offline_without_loading_provider_configuration(tmp_path, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: pytest.fail('Offline mode loaded configuration'))
    workflow, path = run_suite(ROOT, 'workflows', output_dir=tmp_path / 'workflow',
                               use_cases=['test_planning'], case_ids=['WF-31'])
    assert path.is_file() and workflow['summary']['scheduled_cases'] == 1
    assert workflow['summary']['handoff_rate'] is None and workflow['summary']['model_calls'] == 0
    assert workflow['results'][0]['use_case'] == 'test_planning'
    reviewer, _ = run_suite(ROOT, 'reviewer_regression', output_dir=tmp_path / 'reviewer',
                            case_ids=['W4-41'])
    assert reviewer['summary']['metrics']['case_exact_match']['value'] is None
    assert reviewer['rows'][0]['status'] == 'not_run'
    integrity, _ = run_suite(ROOT, 'integrity', output_dir=tmp_path / 'integrity')
    assert integrity['summary']['passed_cases'] == 30 and integrity['summary']['model_calls'] == 0
    unrelated = next(row for row in integrity['results'] if row['operation'] == 'unrelated_claim')
    assert unrelated['observed_acceptance'] is True and unrelated['category'] == 'scope_limit'


def test_suite_and_prompt_validation_precede_execution(tmp_path):
    for options in ({'suite_id': 'invented'}, {'suite_id': 'integrity', 'live': True},
                    {'suite_id': 'workflows', 'reviewer_prompt': 'invented'}):
        with pytest.raises(ValueError):
            run_suite(tmp_path, **options)
    assert not (tmp_path / 'evaluation').exists()


def test_report_filter_uses_dataset_identity_and_report_kind(tmp_path):
    directory = tmp_path / 'evaluation'
    directory.mkdir()
    raw = b'{"name":"fresh source"}'
    (directory / 'reviewer-regression-cases.json').write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    fixtures = {'workflows': {'report_kind': 'use_case_workflow'},
                'integrity': {'report_kind': 'evidence_integrity'},
                'reviewer_regression': {'report_kind': 'week4_evidence_review', 'manifest': {'dataset_sha256': digest}},
                'week4': {'report_kind': 'week4_evidence_review', 'manifest': {'dataset_sha256': 'old'}},
                'legacy': {'report_kind': 'deep_agent_workflow'}}
    for key, payload in fixtures.items():
        path = directory / key / 'report.json'
        path.parent.mkdir()
        path.write_text(json.dumps(payload))
    for key in fixtures:
        assert list_reports(tmp_path, key) == [directory / key / 'report.json']
    assert len(list_reports(tmp_path)) == 5
    assert report_suite({'report_kind': 'week4_paired_comparison', 'baseline': {'dataset_sha256': digest}},
                        regression_hash=digest) == 'reviewer_regression'


def _launcher(root):
    from pathlib import Path
    from triage_desk.evaluation_lab import render_evaluation_launcher
    render_evaluation_launcher(Path(root))


def test_browsing_launcher_does_not_run_models_and_click_forwards_explicit_mode(tmp_path, monkeypatch):
    from triage_desk import evaluation_lab
    evaluation = tmp_path / 'evaluation'
    evaluation.mkdir()
    (evaluation / 'workflow-cases.json').write_text(json.dumps({'cases': [{'id': 'WF-01'}]}))
    calls = []
    def selected_run(root, suite, **options):
        calls.append((suite, options))
        return {}, Path(root) / 'evaluation' / 'selected' / 'report.json'
    monkeypatch.setattr(evaluation_lab, 'run_suite', selected_run)
    app = AppTest.from_function(_launcher, args=(str(tmp_path),)).run()
    assert not app.exception and not calls
    assert app.radio(key='evaluation_execution_mode').value == 'Validate inputs'
    app.radio(key='evaluation_execution_mode').set_value('Run model evaluation').run()
    assert not calls
    app.button(key='run_evaluation_suite').click().run()
    assert not app.exception and len(calls) == 1
    assert calls[0][0] == 'workflows' and calls[0][1]['live'] is True
    assert calls[0][1]['reviewer_prompt'] == 'calibrated'
    assert app.selectbox(key='evaluation_suite_filter').value == 'workflows'
