"""Publication previews and write gates, using only mocked HTTP transports."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import sqlite3
import threading

import httpx
import pytest

from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.cases import CaseStore
from triage_desk.review_publication import API, MAX_RESPONSE_BYTES, WEB, ReviewPublisher


HEAD, BASE, NUMBER = 'a' * 40, 'b' * 40, 88
PATCH = '+def answer():\n+    return 42\n'


@pytest.fixture
def approved(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case = store.create_case('PRIVATE QUESTION DO NOT PUBLISH', code_context='PRIVATE CODE DO NOT PUBLISH')
    registry = EvidenceRegistry(store, case['case_id'])
    registry.register_source(source_key=f'github:pr:{NUMBER}:{HEAD}', title='Public PR description',
        text='This public PR changes the returned constant value.', url=f'{WEB}/pull/{NUMBER}',
        origin='live', kind='issue', evidence_type='pr_description', commit_sha=HEAD, base_sha=BASE)
    filename = 'lib/streamlit/example.py'
    patch = registry.register_source(source_key=f'github:patch:{BASE}:{HEAD}:{filename}',
        title='Public Python patch', text=PATCH, origin='live', kind='code', evidence_type='code_patch',
        commit_sha=HEAD, base_sha=BASE, filename=filename,
        url=f'{WEB}/compare/{BASE}...{HEAD}#diff-' + hashlib.sha256(filename.encode()).hexdigest())
    payload = {'findings': [{'statement': 'The patch returns the constant value 42.', 'category': 'correctness',
                'severity': 'medium', 'support': [{'evidence_id': patch['evidence_id'],
                                                  'quote_id': patch['quote_spans'][0]['quote_id']}]}],
               'next_steps': ['PRIVATE PROPOSAL DO NOT PUBLISH'],
               'missing_information': ['PRIVATE FOLLOWUP DO NOT PUBLISH']}
    report = registry.validate_report(payload)
    report['quality_review'] = {'status': 'completed', 'retained_findings': 1}
    store.update_case(case['case_id'], status='approved', report=report, review_notes='PRIVATE REVIEW NOTES')
    return store, case['case_id'], registry, payload


def current_pr(**updates):
    return {'number': NUMBER, 'state': 'open', 'head': {'sha': HEAD},
            'base': {'sha': BASE, 'repo': {'full_name': 'streamlit/streamlit', 'private': False}}, **updates}


def published_response():
    return httpx.Response(200, json={'id': 456, 'state': 'COMMENTED', 'commit_id': HEAD,
                                    'html_url': f'{WEB}/pull/{NUMBER}#pullrequestreview-456'})


def mock_publisher(approved, handler, token='test-publish-token'):
    requests = []
    def capture(request):
        requests.append(request)
        return handler(request)
    client = httpx.Client(transport=httpx.MockTransport(capture))
    return ReviewPublisher(approved[0], token=token, session=client), requests


def test_prepare_needs_no_token_or_network_and_preview_is_immutable(approved, monkeypatch):
    monkeypatch.setattr(httpx, 'Client', lambda *a, **kw: pytest.fail('Prepare must never create an HTTP client'))
    publisher = ReviewPublisher(approved[0])
    plan = publisher.prepare(approved[1], NUMBER)
    assert plan['status'] == 'prepared'
    assert plan['event'] == 'COMMENT' and plan['commit_id'] == plan['head_sha'] == HEAD
    assert plan['base_sha'] == BASE and plan['repository'] == 'streamlit/streamlit'
    assert 'constant value 42' in plan['body'] and 'Public Python patch' in plan['body']
    assert 'PRIVATE' not in plan['body']
    for digest in ('report_sha256', 'evidence_sha256', 'body_sha256'):
        assert len(plan[digest]) == 64
    original = deepcopy(plan)
    plan['body'] = 'Changed caller dictionary'
    assert publisher.get(plan['plan_id']) == original
    assert publisher.prepare(approved[1], NUMBER)['plan_id'] == plan['plan_id']
    with sqlite3.connect(approved[0].path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute('UPDATE review_publication_plans SET plan_json=? WHERE plan_id=?', ('{}', plan['plan_id']))
    restarted = ReviewPublisher(CaseStore(approved[0].path))
    assert restarted.get(plan['plan_id']) == original


@pytest.mark.parametrize(('status', 'quality'), [
    ('awaiting_review', 'completed'), ('rejected', 'completed'), ('approved', 'failed'), ('approved', ''),
])
def test_case_approval_and_completed_evidence_review_are_required(approved, status, quality):
    report = approved[0].get_case(approved[1])['report']
    report['quality_review'] = {'status': quality}
    approved[0].update_case(approved[1], status=status, report=report)
    with pytest.raises(ValueError, match='evidence-reviewed'):
        ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)


@pytest.mark.parametrize('number', [0, -1, True, '88', 10_000_001])
def test_only_bounded_integer_pr_targets_are_accepted(approved, number):
    with pytest.raises(ValueError, match='positive'):
        ReviewPublisher(approved[0]).prepare(approved[1], number)


@pytest.mark.parametrize(('origin', 'kind'), [('user_code', 'code'), ('repository', 'code'),
                                            ('live', 'analysis'), ('live', 'private')])
def test_private_or_analysis_support_cannot_enter_public_preview(approved, origin, kind):
    source = approved[2].register_source(source_key='private-source', title='Private source', text='PRIVATE SOURCE BODY MATERIAL',
        url='https://docs.streamlit.io/develop/', origin=origin, kind=kind)
    payload = deepcopy(approved[3])
    payload['findings'][0]['support'] = [{'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id']}]
    report = approved[2].validate_report(payload)
    report['quality_review'] = {'status': 'completed'}
    approved[0].update_case(approved[1], report=report)
    with pytest.raises(ValueError, match='public source evidence'):
        ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)


def test_malformed_pr_provenance_and_missing_matching_patch_are_rejected(approved):
    publisher = ReviewPublisher(approved[0])
    approved[2].register_source(source_key=f'github:pr:{NUMBER}:invalid', title='Malformed context',
        text='A public but unpinned description.', url=f'{WEB}/pull/{NUMBER}', origin='live', kind='issue',
        evidence_type='pr_description', commit_sha='invalid', base_sha=BASE)
    with pytest.raises(ValueError, match='invalid public revision'):
        publisher.prepare(approved[1], NUMBER)
    newer = 'c' * 40
    approved[2].register_source(source_key=f'github:pr:{NUMBER}:{newer}', title='Newer PR context',
        text='A changed public description without matching patch evidence.', url=f'{WEB}/pull/{NUMBER}',
        origin='live', kind='issue', evidence_type='pr_description', commit_sha=newer, base_sha=BASE)
    with pytest.raises(ValueError, match='patch must match'):
        publisher.prepare(approved[1], NUMBER)


@pytest.mark.parametrize('source_kind', ['old_patch', 'other_pr', 'unpinned_pr', 'pr_issue_comment'])
def test_findings_cannot_publish_other_or_unpinned_pr_revisions_as_current_commit(approved, source_kind):
    if source_kind == 'old_patch':
        old_head, filename = 'c' * 40, 'lib/streamlit/old.py'
        source = approved[2].register_source(
            source_key=f'github:patch:{BASE}:{old_head}:{filename}', title='Old public patch',
            text='+def old_answer():\n+    return 99\n', origin='live', kind='code',
            evidence_type='code_patch', commit_sha=old_head, base_sha=BASE, filename=filename,
            url=f'{WEB}/compare/{BASE}...{old_head}#diff-' + hashlib.sha256(filename.encode()).hexdigest())
    elif source_kind == 'other_pr':
        source = approved[2].register_source(
            source_key=f'github:pr:99:{HEAD}', title='Different public PR', text='An observation from another pull request.',
            url=f'{WEB}/pull/99', origin='live', kind='issue', evidence_type='pr_description', commit_sha=HEAD, base_sha=BASE)
    elif source_kind == 'unpinned_pr':
        source = approved[2].register_source(
            source_key='keenable:page:mutable', title='Unpinned public PR page', text='A mutable pull request description snippet.',
            url=f'{WEB}/pull/{NUMBER}', origin='live', kind='issue', evidence_type='github_pull_request_page_content')
    else:
        source = approved[2].register_source(
            source_key='github:comment:42', title='An unpinned PR comment', text='A mutable observation in a PR comment.',
            url=f'{WEB}/issues/{NUMBER}#issuecomment-42', origin='live', kind='issue', evidence_type='issue_comment')
    payload = deepcopy(approved[3])
    payload['findings'][0]['support'] = [{'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id']}]
    report = approved[2].validate_report(payload)
    report['quality_review'] = {'status': 'completed'}
    approved[0].update_case(approved[1], report=report)
    with pytest.raises(ValueError, match='selected PR|selected target|pinned review provenance'):
        ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)


def test_related_public_issue_observations_can_remain_attributed_context(approved):
    source = approved[2].register_source(
        source_key='github:issue:99', title='A related public report', text='The reporter observed a constant result of 42.',
        url=f'{WEB}/issues/99', origin='live', kind='issue', evidence_type='issue_report')
    payload = deepcopy(approved[3])
    payload['findings'][0]['statement'] = 'The related issue report describes a constant result of 42.'
    payload['findings'][0]['support'] = [{'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id']}]
    report = approved[2].validate_report(payload)
    report['quality_review'] = {'status': 'completed'}
    approved[0].update_case(approved[1], report=report)
    plan = ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)
    assert plan['status'] == 'prepared' and f'{WEB}/issues/99' in plan['body']


@pytest.mark.parametrize(('path', 'evidence_type'), [
    ('blob/' + 'c' * 40 + '/lib/streamlit/old.py', 'page_content'),
    ('%62lob/' + 'c' * 40 + '/lib/streamlit/old.py', 'page_content'),
    (f'compare/{BASE}...{HEAD}', 'search_snippet'),
    ('raw/main/lib/streamlit/example.py', 'page_content'),
    ('tree/main/lib/streamlit', 'page_content'),
    ('commit/' + 'c' * 40, 'page_content'),
    ('commits/main', 'page_content'),
])
def test_legacy_documentation_labels_cannot_bypass_code_page_revision_guards(approved, path, evidence_type):
    legacy = approved[2].register_source(
        source_key='keenable:legacy-code-page', title='Legacy repository page',
        text='A public code excerpt from an older or unpinned repository page.',
        url=f'{WEB}/{path}', origin='live', kind='documentation', evidence_type=evidence_type)
    payload = deepcopy(approved[3])
    payload['findings'][0]['support'] = [{'evidence_id': legacy['evidence_id'],
                                          'quote_id': legacy['quote_spans'][0]['quote_id']}]
    report = approved[2].validate_report(payload)
    report['quality_review'] = {'status': 'completed'}
    approved[0].update_case(approved[1], report=report)
    publisher, requests = mock_publisher(approved, lambda _: pytest.fail('Invalid preview must remain offline'))
    with pytest.raises(ValueError, match='code support must match'):
        publisher.prepare(approved[1], NUMBER)
    assert not requests


@pytest.mark.parametrize('secret', ['custom_key_value_123456', 'github_pat_' + 'x' * 25])
def test_configured_and_common_credentials_are_refused_before_markdown_escaping_without_github_token(approved, secret):
    report = approved[0].get_case(approved[1])['report']
    report['findings'][0]['statement'] = f'The code mentions {secret}.'
    approved[0].update_case(approved[1], report=report)
    with pytest.raises(ValueError, match='credential-like'):
        ReviewPublisher(approved[0], forbidden_values=(secret,)).prepare(approved[1], NUMBER)


def test_newly_configured_forbidden_value_is_rechecked_before_publication(approved):
    secret = 'custom_key_value_123456'
    report = approved[0].get_case(approved[1])['report']
    report['findings'][0]['statement'] = f'The code mentions {secret}.'
    approved[0].update_case(approved[1], report=report)
    preview = ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)
    publisher, requests = mock_publisher(approved, lambda _: pytest.fail('Credential match must block HTTP'))
    publisher = ReviewPublisher(approved[0], token='test-publish-token', session=publisher.session, forbidden_values=(secret,))
    result = publisher.publish(preview['plan_id'], confirmation=preview['plan_id'])
    assert result['status'] == 'failed' and not requests


def test_overlong_public_summary_is_refused_instead_of_truncated(approved):
    payload = deepcopy(approved[3])
    finding = payload['findings'][0]
    finding['statement'] = 'A bounded factual statement. ' * 40
    payload['findings'] = [deepcopy(finding) for _ in range(12)]
    report = approved[2].validate_report(payload)
    report['quality_review'] = {'status': 'completed'}
    approved[0].update_case(approved[1], report=report)
    with pytest.raises(ValueError, match='12000'):
        ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)


def test_publish_sends_exact_confirmed_comment_payload_once(approved):
    def handler(request):
        assert request.url.host == 'api.github.com'
        assert request.headers['authorization'] == 'Bearer test-publish-token'
        if request.method == 'GET':
            assert str(request.url) == f'{API}/pulls/{NUMBER}'
            return httpx.Response(200, json=current_pr())
        assert str(request.url) == f'{API}/pulls/{NUMBER}/reviews'
        assert json.loads(request.content) == {'commit_id': HEAD, 'event': 'COMMENT', 'body': plan['body']}
        return published_response()

    publisher, requests = mock_publisher(approved, handler)
    plan = publisher.prepare(approved[1], NUMBER)
    result = publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert result['status'] == 'published'
    assert result['github_review_id'] == 456
    assert result['review_url'] == f'{WEB}/pull/{NUMBER}#pullrequestreview-456'
    assert [request.method for request in requests] == ['GET', 'POST']
    assert publisher.prepare(approved[1], NUMBER)['plan_id'] == plan['plan_id']
    with pytest.raises(ValueError, match='already attempted'):
        publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert len(requests) == 2
    assert 'test-publish-token' not in json.dumps(result)


@pytest.mark.parametrize('confirmation', [None, '', 'approve', 'another-plan'])
def test_exact_preview_confirmation_precedes_any_http_or_claim(approved, confirmation):
    publisher, requests = mock_publisher(approved, lambda _: pytest.fail('No request authorized'))
    plan = publisher.prepare(approved[1], NUMBER)
    with pytest.raises(ValueError, match='exact preview plan ID'):
        publisher.publish(plan['plan_id'], confirmation=confirmation)
    assert publisher.get(plan['plan_id'])['status'] == 'prepared' and not requests


def test_token_is_required_only_at_explicit_publication(approved):
    publisher, requests = mock_publisher(approved, lambda _: pytest.fail('No token'), token='')
    plan = publisher.prepare(approved[1], NUMBER)
    with pytest.raises(ValueError, match='GitHub token'):
        publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert not requests and publisher.get(plan['plan_id'])['status'] == 'prepared'


@pytest.mark.parametrize(('changed', 'code'), [
    ({'head': {'sha': 'c' * 40}}, 'stale_revision'),
    ({'base': {'sha': 'c' * 40, 'repo': {'full_name': 'streamlit/streamlit', 'private': False}}}, 'stale_revision'),
    ({'base': {'sha': BASE, 'repo': {'full_name': 'streamlit/streamlit', 'private': True}}}, 'public_pr_unconfirmed'),
    ({'state': 'closed'}, 'public_pr_unconfirmed'),
    ({'base': None}, 'public_pr_unconfirmed'),
])
def test_remote_revision_and_public_target_revalidation_block_post(approved, changed, code):
    publisher, requests = mock_publisher(approved, lambda _: httpx.Response(200, json=current_pr(**changed)))
    plan = publisher.prepare(approved[1], NUMBER)
    result = publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert result['status'] == 'failed' and result['error_code'] == code
    assert [request.method for request in requests] == ['GET']


@pytest.mark.parametrize('change', ['approval', 'report', 'evidence'])
def test_saved_report_approval_or_evidence_changes_invalidate_preview_before_http(approved, change):
    publisher, requests = mock_publisher(approved, lambda _: pytest.fail('Changed case must not dispatch'))
    plan = publisher.prepare(approved[1], NUMBER)
    if change == 'approval':
        approved[0].update_case(approved[1], status='awaiting_review')
    elif change == 'report':
        report = approved[0].get_case(approved[1])['report']
        report['findings'][0]['statement'] = 'A changed reviewed statement.'
        approved[0].update_case(approved[1], report=report)
    else:
        approved[2].register_source(source_key='new', title='Additional context', text='A newly registered public source.',
                                   url='https://docs.streamlit.io/develop/', origin='live')
    result = publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert result['status'] == 'failed' and result['error_code'] == 'case_changed'
    assert not requests


@pytest.mark.parametrize('failure', ['timeout', 'server', 'redirect', 'malformed', 'oversized', 'wrong_review'])
def test_unknown_post_outcome_is_persisted_and_cannot_be_retried_via_another_preview(approved, failure):
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json=current_pr())
        if failure == 'timeout':
            raise httpx.ReadTimeout('PRIVATE TOKEN PROVIDER FAILURE', request=request)
        if failure == 'server':
            return httpx.Response(503, text='PRIVATE PROVIDER RESPONSE')
        if failure == 'redirect':
            return httpx.Response(307, headers={'Location': 'https://unapproved.test/'})
        if failure == 'malformed':
            return httpx.Response(200, text='PRIVATE INVALID JSON')
        if failure == 'oversized':
            return httpx.Response(200, content=b' ' * (MAX_RESPONSE_BYTES + 1))
        return httpx.Response(200, json={'id': 456, 'state': 'APPROVED', 'commit_id': HEAD})
    publisher, requests = mock_publisher(approved, handler)
    plan = publisher.prepare(approved[1], NUMBER)
    result = publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert result['status'] == 'uncertain'
    assert 'PRIVATE' not in json.dumps(result)
    reopened = ReviewPublisher(CaseStore(approved[0].path), token='different-token', session=publisher.session)
    assert reopened.prepare(approved[1], NUMBER)['plan_id'] == plan['plan_id']
    with pytest.raises(ValueError, match='already attempted'):
        reopened.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert [request.method for request in requests] == ['GET', 'POST']


def test_definitive_github_rejection_is_safe_and_allows_a_fresh_explicit_plan(approved):
    publisher, requests = mock_publisher(approved, lambda request: httpx.Response(200, json=current_pr())
        if request.method == 'GET' else httpx.Response(403, text='test-publish-token PRIVATE RESPONSE'))
    plan = publisher.prepare(approved[1], NUMBER)
    result = publisher.publish(plan['plan_id'], confirmation=plan['plan_id'])
    assert result['status'] == 'failed' and result['error_code'] == 'github_rejected'
    assert 'PRIVATE' not in json.dumps(result) and 'test-publish-token' not in json.dumps(result)
    replacement = publisher.prepare(approved[1], NUMBER)
    assert replacement['status'] == 'prepared' and replacement['plan_id'] != plan['plan_id']
    assert len(requests) == 2


def test_concurrent_publishers_cannot_dispatch_duplicate_reviews(approved):
    entered, release = threading.Event(), threading.Event()
    def handler(request):
        if request.method == 'GET':
            entered.set()
            assert release.wait(3)
            return httpx.Response(200, json=current_pr())
        return published_response()
    first, requests = mock_publisher(approved, handler)
    second = ReviewPublisher(CaseStore(approved[0].path), token='test-publish-token', session=first.session)
    plan = first.prepare(approved[1], NUMBER)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.publish, plan['plan_id'], confirmation=plan['plan_id'])
        assert entered.wait(3)
        try:
            with pytest.raises(ValueError, match='already running'):
                second.publish(plan['plan_id'], confirmation=plan['plan_id'])
        finally:
            release.set()
        assert future.result(timeout=3)['status'] == 'published'
    assert [request.method for request in requests] == ['GET', 'POST']


def test_abandoned_publication_recovers_as_uncertain_without_retry_or_new_plan(approved):
    plan = ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)
    with sqlite3.connect(approved[0].path) as connection:
        connection.execute("UPDATE review_publication_plans SET status='publishing' WHERE plan_id=?", (plan['plan_id'],))
    restarted = ReviewPublisher(CaseStore(approved[0].path))
    recovered = restarted.get(plan['plan_id'])
    assert recovered['status'] == 'uncertain' and recovered['error_code'] == 'interrupted_publication'
    assert restarted.prepare(approved[1], NUMBER)['plan_id'] == plan['plan_id']


def test_restart_does_not_mark_an_active_publication_uncertain(approved):
    plan = ReviewPublisher(approved[0]).prepare(approved[1], NUMBER)
    with approved[0].execution_lock(approved[1]):
        with sqlite3.connect(approved[0].path) as connection:
            connection.execute("UPDATE review_publication_plans SET status='publishing' WHERE plan_id=?", (plan['plan_id'],))
        other = ReviewPublisher(CaseStore(approved[0].path))
        assert other.get(plan['plan_id'])['status'] == 'publishing'

