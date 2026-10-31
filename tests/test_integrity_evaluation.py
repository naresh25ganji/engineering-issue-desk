"""Actual evidence-validator acceptance policy and its semantic limits."""
import json

import pytest

from evaluation import integrity


def test_integrity_contract_records_all_acceptances_and_rejections_without_provider(tmp_path, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *a, **k: pytest.fail('Integrity checks read provider config'))
    report = integrity.run_evaluation(tmp_path, tmp_path / 'results')
    assert report['summary']['scheduled_cases'] == report['summary']['passed_cases'] == 30
    assert report['summary']['invalid_rejected'] == 23
    assert report['summary']['valid_accepted'] == 7 and report['summary']['model_calls'] == 0
    assert report['manifest']['dataset_sha256'] and report['manifest']['validator_sha256']
    assert json.loads((tmp_path / 'results' / 'report.json').read_text()) == report
    unrelated = next(row for row in report['results'] if row['operation'] == 'unrelated_claim')
    assert unrelated['expected_acceptance'] is True and unrelated['observed_acceptance'] is True
    assert 'not factual accuracy' in report['measurement_limits'][0]
    with pytest.raises(ValueError, match='new or empty'):
        integrity.run_evaluation(tmp_path, tmp_path / 'results')


def test_integrity_html_neutralizes_hostile_fixture_markup(tmp_path):
    integrity.run_evaluation(tmp_path, tmp_path / 'results')
    page = (tmp_path / 'results' / 'report.html').read_text()
    assert '<img src="tracker">' not in page and '&lt;img' in page
