"""Offline runner contracts; model answers are test doubles, never live metrics."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
import pytest

from evaluation.week4 import (compare_runs, hydrate_report, load_cases, run_evaluation,
                              score_case, summarize)


class ReviewModel(BaseChatModel):
    calls: list[Any] = Field(default_factory=list)
    fail: bool = False
    malicious: bool = False

    @property
    def _llm_type(self):
        return 'week4-offline-test-double'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError('PROVIDER SECRET fw_abcdefghijk0123456789')
        payload = json.loads(messages[-1].content)
        assert 'expected_supported' not in messages[-1].content
        assert 'ANNOTATION DO NOT SEND' not in messages[-1].content
        args = {key: [{'index': item['index'], 'supported': item['index'] == 0,
                       'reason': 'The quoted evidence supports this decision.'}
                      for item in payload[kind]]
                for key, kind in (('finding_verdicts', 'findings'), ('next_step_verdicts', 'next_steps'))}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='', tool_calls=[{'name': 'EvidenceVerdicts', 'args': args, 'id': 'answer'}],
            usage_metadata={'input_tokens': 80, 'output_tokens': 20, 'total_tokens': 100},
            response_metadata={'model_name': 'week4-test-model', 'finish_reason': 'tool_calls'}))])


@pytest.fixture
def dataset(tmp_path):
    fixture = tmp_path / 'fixture.py'
    fixture.write_text('alpha = 1\nbeta = 2\n', encoding='utf-8')
    sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    cases = []
    for index in range(1, 31):
        cases.append({'id': f'W4-{index:02}', 'scenario': 'fixture observation',
                      'title': 'Frozen fixture', 'failure_mode': 'unsupported causality',
                      'source': {'path': 'fixture.py', 'start_line': 1, 'end_line': 2,
                                 'sha256': sha, 'quote': 'alpha = 1\nbeta = 2\n'},
                      'context': {'question': 'Does this literal code establish the claim?', 'reported_version': '1.0'},
                      'findings': [{'statement': 'The fixture assigns alpha to one.', 'expected_supported': True,
                                    'rationale': 'ANNOTATION DO NOT SEND: literal assignment.'},
                                   {'statement': 'The fixture ran successfully in production.', 'expected_supported': False,
                                    'rationale': 'ANNOTATION DO NOT SEND: no runtime observation.'}],
                      'next_steps': [{'proposal': 'Check the value of alpha in a controlled test.', 'expected_supported': True,
                                      'rationale': 'ANNOTATION DO NOT SEND: a diagnostic.'},
                                     {'proposal': 'Deploy alpha to fix every crash.', 'expected_supported': False,
                                      'rationale': 'ANNOTATION DO NOT SEND: an unsupported remedy.'}],
                      'label_review': {'status': 'pending', 'reviewer': None, 'notes': 'ANNOTATION DO NOT SEND'}})
    value = {'name': 'Test-only frozen fixtures', 'version': '1', 'label_status': 'provisional; pending human review',
             'source_policy': 'Literal pinned repository quotes.', 'cases': cases}
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return tmp_path, path, value


def run_fixture(dataset, output='run', **kwargs):
    root, path, _ = dataset
    return run_evaluation(root, path, root / output, progress=False, **kwargs)


def test_offline_inventory_never_constructs_model_or_scores(dataset):
    result = run_fixture(dataset, model_factory=lambda: pytest.fail('Offline must not reach a model'))
    assert result['report_kind'] == 'week4_evidence_review'
    assert result['summary']['scheduled_cases'] == 30
    assert result['summary']['not_run_cases'] == 30
    assert result['summary']['completion_rate'] is None
    assert result['summary']['all_pass_bars_met'] is None
    assert all(metric['value'] is None and metric['status'] == 'not_measured'
               for metric in result['summary']['metrics'].values())
    assert result['rows'][0]['scores']['exact_match'] is None
    assert result['rows'][0]['scores']['items'][0]['correct'] is None
    assert not (dataset[0] / '.cache' / 'week4').exists()
    for name in ('report.json', 'report.html', 'results.csv', 'label-review.csv', 'manifest.json', 'prompt.txt', 'dataset.json'):
        assert (dataset[0] / 'run' / name).exists()


def test_all_cases_validate_before_subset_and_source_mutation_never_refreshes_quote(dataset):
    root, path, value = dataset
    assert len(load_cases(root, path, ['W4-02'])[1]) == 1
    original = value['cases'][0]['source']['quote']
    (root / 'fixture.py').write_text('alpha = 999\nbeta = 2\n')
    with pytest.raises(ValueError, match='pin changed'):
        load_cases(root, path, ['W4-02'])
    assert json.loads(path.read_text())['cases'][0]['source']['quote'] == original


@pytest.mark.parametrize('mutation,match', [
    (lambda d: d['cases'].pop(), '30 to 50'),
    (lambda d: d['cases'][0]['findings'][0].update(expected_supported=1), 'strict boolean'),
    (lambda d: d['cases'][1].update(id='W4-01'), 'unique'),
    (lambda d: d['cases'][0]['source'].update(path='../outside'), 'inside'),
    (lambda d: d['cases'][0]['source'].update(quote='not the original quote'), 'does not match'),
])
def test_invalid_dataset_rejected(dataset, mutation, match):
    root, path, value = dataset
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        load_cases(root, path)


def test_hydration_excludes_all_golden_annotations(dataset):
    case = deepcopy(dataset[2]['cases'][0])
    case['title'] = 'ANNOTATION DO NOT SEND: answer-bearing evaluation title'
    case['failure_mode'] = 'ANNOTATION DO NOT SEND: expected failure taxonomy'
    report = hydrate_report(case)
    encoded = json.dumps(report)
    for forbidden in ('expected_supported', 'rationale', 'label_review', 'failure_mode', 'ANNOTATION DO NOT SEND'):
        assert forbidden not in encoded
    assert report['sources'][0]['title'] == case['source']['path']
    assert report['findings'][0]['support'][0]['quote'] == case['source']['quote']
    assert report['findings'][0]['support'][0]['evidence_id'] == report['sources'][0]['evidence_id']


def test_live_path_uses_production_review_accounting_isolation_and_redaction(dataset, monkeypatch):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'fw_exampleSuperSecret0123456789')
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    models = []
    def factory():
        model = ReviewModel()
        models.append(model)
        return model
    result = run_fixture(dataset, live=True, model_factory=factory, case_ids=['W4-01', 'W4-02'], workers=2,
                         input_usd_per_million=1, output_usd_per_million=2)
    assert len(models) == 2
    assert all(len(model.calls) == 1 for model in models)
    assert result['summary']['metrics']['case_exact_match']['value'] == 1
    assert result['summary']['metrics']['supported_retention']['value'] == 1
    assert result['summary']['metrics']['unsupported_rejection']['value'] == 1
    assert result['summary']['metrics']['mean_total_tokens']['value'] == 100
    assert result['summary']['estimated_cost_usd'] == pytest.approx(.00024)
    assert len({row['case_id'] for row in result['rows']}) == 2
    for row in result['rows']:
        assert row['status'] == 'completed'
        assert row['scores']['predicted'] == {'findings': [True, False], 'next_steps': [True, False]}
        assert row['accounting']['model_calls'] == 1
        assert row['accounting']['usage_complete']
        assert [event['phase'] for event in row['events']] == ['evaluation', 'model', 'model_result', 'evaluation_result']
        assert len(row['filtered_report']['findings']) == 1
    assert len(list((dataset[0] / '.cache' / 'week4').rglob('cases.sqlite'))) == 2
    assert os_environment_tracing() == 'true'
    assert 'fw_exampleSuperSecret0123456789' not in (dataset[0] / 'run' / 'report.json').read_text()


def os_environment_tracing():
    import os
    return os.environ.get('LANGSMITH_TRACING')


def test_errors_remain_unknown_and_count_in_all_scheduled_denominators(dataset):
    result = run_fixture(dataset, live=True, model_factory=lambda: ReviewModel(fail=True), case_ids=['W4-01'])
    row = result['rows'][0]
    assert row['status'] == 'error'
    assert row['error']['code'] == 'provider_unavailable'
    assert row['scores']['predicted'] == {'findings': [None, None], 'next_steps': [None, None]}
    assert row['scores']['false_positives'] == row['scores']['false_negatives'] == 0
    assert row['scores']['unknown_verdicts'] == 4
    for metric in ('case_exact_match', 'supported_retention', 'unsupported_rejection'):
        assert result['summary']['metrics'][metric]['value'] == 0
    assert result['summary']['metrics']['mean_total_tokens']['value'] is None
    assert result['summary']['estimated_cost_usd'] is None
    assert 'PROVIDER SECRET' not in (dataset[0] / 'run' / 'report.json').read_text()
    assert 'fw_abcdefghijk0123456789' not in (dataset[0] / 'run' / 'report.html').read_text()


def test_invalid_verdict_not_coerced_to_false(dataset):
    case = dataset[2]['cases'][0]
    quality = {'finding_verdicts': [{'index': 0, 'supported': True}, {'index': 1, 'supported': 'false'}],
               'next_step_verdicts': [{'index': 0, 'supported': True}, {'index': 1, 'supported': False}]}
    score = score_case(case, quality, completed=True)
    assert not score['valid']
    assert score['predicted']['findings'] == [None, None]
    assert score['false_positives'] == 0
    assert score['unknown_verdicts'] == 2
    assert not score['exact_match']


def test_resume_is_identical_only_and_preserves_completed_results(dataset, monkeypatch):
    result = run_fixture(dataset, case_ids=['W4-01'])
    root, path, value = dataset
    case_file = root / 'run' / 'cases' / 'W4-01.json'
    timestamp = case_file.stat().st_mtime_ns
    same = run_fixture(dataset, case_ids=['W4-01'], resume=True)
    assert same['manifest']['run_id'] == result['manifest']['run_id']
    assert case_file.stat().st_mtime_ns == timestamp
    with pytest.raises(ValueError, match='new or empty'):
        run_fixture(dataset, case_ids=['W4-01'])
    with pytest.raises(ValueError, match='Resume refused'):
        run_fixture(dataset, case_ids=['W4-02'], resume=True)
    value['cases'][0]['findings'][0]['rationale'] += ' changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Resume refused'):
        run_fixture(dataset, case_ids=['W4-01'], resume=True)


def test_compare_reports_fixed_regressions_and_same_case_denominators(dataset):
    root = dataset[0]
    run_fixture(dataset, output='baseline', live=True, model_factory=lambda: ReviewModel(fail=True), case_ids=['W4-01', 'W4-02'])
    run_fixture(dataset, output='improved', live=True, model_factory=ReviewModel, case_ids=['W4-01', 'W4-02'])
    comparison = compare_runs(root / 'baseline', root / 'improved', root / 'comparison')
    assert comparison['report_kind'] == 'week4_paired_comparison'
    assert comparison['summary'] == {'fixed': 2, 'regression': 0, 'unchanged_pass': 0, 'unchanged_failure': 0}
    assert comparison['clusters'][0]['baseline_error_cases'] == 2
    assert comparison['clusters'][0]['baseline_estimated_human_minutes'] == 10
    for filename in ('report.json', 'comparison.json', 'comparison.csv', 'clusters.csv', 'comparison.html'):
        assert (root / 'comparison' / filename).exists()
    run_fixture(dataset, output='subset', live=True, model_factory=ReviewModel, case_ids=['W4-01'])
    with pytest.raises(ValueError, match='same case IDs'):
        compare_runs(root / 'baseline', root / 'subset', root / 'bad-comparison')
    report_path = root / 'improved' / 'report.json'
    report = json.loads(report_path.read_text())
    report['manifest']['dataset_sha256'] = '0' * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='same frozen dataset'):
        compare_runs(root / 'baseline', root / 'improved', root / 'bad-comparison')


def test_csv_escapes_formulas_and_html_escapes_data(dataset):
    root, path, value = dataset
    value['cases'][0]['findings'][0]['statement'] = '=HYPERLINK("https://evil.test")'
    value['cases'][0]['title'] = '<script>alert(1)</script>'
    path.write_text(json.dumps(value))
    run_fixture(dataset, case_ids=['W4-01'])
    with (root / 'run' / 'results.csv').open(newline='') as handle:
        records = list(csv.DictReader(handle))
    assert records[0]['text'].startswith("'=HYPERLINK")
    page = (root / 'run' / 'report.html').read_text()
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page


def test_interrupted_resume_runs_only_missing_case_and_prompt_drift_is_refused(dataset, monkeypatch):
    import evaluation.week4 as module
    root = dataset[0]
    first = run_fixture(dataset, live=True, model_factory=ReviewModel, case_ids=['W4-01', 'W4-02'])
    old_case = (root / 'run' / 'cases' / 'W4-01.json').read_bytes()
    (root / 'run' / 'cases' / 'W4-02.json').unlink()
    models = []
    def factory():
        model = ReviewModel()
        models.append(model)
        return model
    resumed = run_fixture(dataset, live=True, model_factory=factory, case_ids=['W4-01', 'W4-02'], resume=True)
    assert len(models) == 1
    assert resumed['manifest']['run_id'] == first['manifest']['run_id']
    assert (root / 'run' / 'cases' / 'W4-01.json').read_bytes() == old_case
    monkeypatch.setattr(module, '_prompt', lambda _: 'Changed instructions')
    with pytest.raises(ValueError, match='Resume refused'):
        run_fixture(dataset, live=True, model_factory=factory, case_ids=['W4-01', 'W4-02'], resume=True)


def test_cli_cases_argument_maps_to_importable_runner(monkeypatch, tmp_path):
    import evaluation.week4 as module
    calls = []
    monkeypatch.setattr(module, 'run_evaluation', lambda **kwargs: calls.append(kwargs) or {'summary': {}})
    module.main(['run', '--cases', str(tmp_path / 'cases.json'), '--output-dir', str(tmp_path / 'out'),
                 '--case', 'W4-01', '--case', 'W4-02', '--workers', '2'])
    assert calls[0]['cases_path'] == tmp_path / 'cases.json'
    assert 'cases' not in calls[0]
    assert calls[0]['case_ids'] == ['W4-01', 'W4-02']

