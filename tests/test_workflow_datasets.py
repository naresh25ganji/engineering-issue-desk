"""Frozen expanded-workflow inputs must stay separate from evaluator labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pytest

from evaluation import week4

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'evaluation' / 'fixtures' / 'workflows'
WORKFLOW = json.loads((ROOT / 'evaluation' / 'workflow-cases.json').read_text(encoding='utf-8'))
REGRESSION = json.loads((ROOT / 'evaluation' / 'reviewer-regression-cases.json').read_text(encoding='utf-8'))


def test_full_workflow_dataset_covers_each_profile_and_scenario_class():
    cases = WORKFLOW['cases']
    assert {case['id'] for case in cases} == {f'WF-{number:02d}' for number in range(1, 41)}
    assert Counter(case['use_case'] for case in cases) == {
        'issue_investigation': 10, 'incident_triage': 10,
        'change_review': 10, 'test_planning': 10,
    }
    assert Counter(case['scenario'] for case in cases) == {
        'happy_path': 20, 'edge': 12, 'known_failure': 6, 'adversarial': 2,
    }
    assert all({case['scenario'] for case in cases if case['use_case'] == profile}
               >= {'happy_path', 'edge', 'known_failure'}
               for profile in {case['use_case'] for case in cases})


@pytest.mark.parametrize('case', WORKFLOW['cases'], ids=lambda case: case['id'])
def test_workflow_context_pins_exact_bounded_fixture_and_literal_anchors(case):
    context, expected = case['context'], case['expected']
    source = (ROOT / context['path']).resolve()
    assert source.is_relative_to(FIXTURES.resolve()) and source.is_file()
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == context['sha256']
    assert raw.decode('utf-8') == context['text']
    assert 20 <= len(context['text']) <= 3000
    assert 0 < len(case['question']) <= 4500 and len(case['version']) <= 120
    assert expected['required_tools'] == ['read_case_evidence']
    assert expected['evidence_anchors']
    assert all(isinstance(anchor, str) and anchor.strip() and anchor in context['text']
               for anchor in expected['evidence_anchors'])
    assert type(expected['min_findings']) is int and 0 <= expected['min_findings'] <= 12
    assert type(expected['min_next_steps']) is int and 0 <= expected['min_next_steps'] <= 8
    assert expected['behavior_notes'] and expected['human_rubric']
    # Evaluator outcomes and semantic rubrics are annotations, never source text.
    assert 'expected_supported' not in context['text']
    assert 'human_rubric' not in context['text']
    assert case['label_review'] == {'status': 'pending', 'reviewer': None, 'notes': ''}


def test_reviewer_regression_loads_through_production_week4_input_validation():
    dataset, cases, digest = week4.load_cases(ROOT, Path('evaluation/reviewer-regression-cases.json'))
    assert dataset['label_status'] == 'ai_proposed_pending_human_review'
    assert dataset['version'] == 1 and len(digest) == 64
    assert len(cases) == 30
    assert {case['id'] for case in cases} == {f'W4-{number}' for number in range(41, 71)}
    assert Counter(case['scenario'] for case in cases) == {
        'happy_path': 15, 'edge': 9, 'known_failure': 4, 'adversarial': 2,
    }
    verdicts = [item['expected_supported'] for case in cases
                for kind in ('findings', 'next_steps') for item in case[kind]]
    assert Counter(verdicts) == {True: 45, False: 45}


@pytest.mark.parametrize('case', REGRESSION['cases'], ids=lambda case: case['id'])
def test_reviewer_quotes_are_exact_complete_frozen_sources(case):
    source = case['source']
    path = (ROOT / source['path']).resolve()
    assert path.is_relative_to(FIXTURES.resolve())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    assert source['start_line'] == 1
    assert source['end_line'] == len(raw.decode('utf-8').splitlines())
    assert source['quote'] == raw.decode('utf-8')
    assert case['label_review'] == {'status': 'pending', 'reviewer': None, 'notes': ''}
    if source['evidence_type'].startswith('synthetic_'):
        assert re.search(r'(Synthetic|synthetic)', source['quote'])
    if source['evidence_type'] == 'unapplied_patch_from_repository_excerpt':
        assert source['quote'].startswith('# Proposed patch for static review only; not applied or executed.\n')
        assert '\n--- a/' in source['quote'] and '\n+++ b/' in source['quote']


def test_new_reviewer_claims_do_not_reuse_original_week4_claims():
    original = json.loads((ROOT / 'evaluation' / 'week4-cases.json').read_text(encoding='utf-8'))
    def claims(dataset):
        return {item[field] for case in dataset['cases']
                for kind, field in (('findings', 'statement'), ('next_steps', 'proposal'))
                for item in case[kind]}
    assert claims(REGRESSION).isdisjoint(claims(original))


def test_reviewer_request_projection_excludes_expected_labels_and_rationales():
    for case in REGRESSION['cases']:
        report = week4.hydrate_report(case)
        payload = json.dumps(report)
        assert 'expected_supported' not in payload and 'rationale' not in payload
        assert 'label_review' not in payload and 'failure_mode' not in payload
        assert [finding['statement'] for finding in report['findings']] == [
            item['statement'] for item in case['findings']]
