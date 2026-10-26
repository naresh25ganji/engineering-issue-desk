from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from uuid import UUID
import sqlite3

import pytest

from triage_desk.cases import CASE_STATUSES, CaseExecutionBusy, CaseStore


def _hold_case_in_process(path, case_id, connection):
    store = CaseStore(path)
    with store.execution_lock(case_id):
        assert store.claim_case(case_id, ('new',))
        connection.send('locked')
        connection.recv()


def test_case_survives_restart_with_report_and_review(tmp_path):
    path = tmp_path / 'nested' / 'cases.sqlite'
    store = CaseStore(path)
    case = store.create_case('Why does my widget reset?', '1.2.3', 'st.button("Run")')
    assert str(UUID(case['case_id'])) == case['case_id']
    assert case['status'] == 'new'
    assert case['report'] is None
    assert 'report_json' not in case
    report = {'findings': [], 'next_steps': ['Reproduce with a widget key.']}
    store.update_case(case['case_id'], status='awaiting_review', report=report,
                      review_notes='Check the reproduction.', report_markdown='A report', error='')
    reopened = CaseStore(path)
    saved = reopened.get_case(case['case_id'])
    assert saved['question'] == case['question']
    assert saved['version'] == '1.2.3'
    assert saved['code_context'] == 'st.button("Run")'
    assert saved['report'] == report
    assert saved['review_notes'] == 'Check the reproduction.'
    assert saved['report_markdown'] == 'A report'
    assert saved['status'] == 'awaiting_review'
    assert saved['updated_at'] >= saved['created_at']


def test_repository_access_is_explicit_immutable_and_survives_restart(tmp_path):
    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    original = store.create_case('No repository access')
    enabled = store.create_case('Inspect related helpers', repository_context=True)
    assert original['repository_context'] is False
    assert CaseStore(path).get_case(enabled['case_id'])['repository_context'] is True
    with pytest.raises(ValueError, match='Cannot update'):
        store.update_case(original['case_id'], repository_context=True)
    for invalid in ('yes', 1, None):
        with pytest.raises(ValueError, match='boolean'):
            store.create_case('Invalid access option', repository_context=invalid)


def test_legacy_case_database_migration_preserves_data_and_defaults_access_off(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    store = CaseStore(path)
    case = store.create_case('Saved before repository context')
    store.update_case(case['case_id'], report={'findings': []}, review_notes='Keep this note.')
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE cases DROP COLUMN repository_context')
    migrated = CaseStore(path).get_case(case['case_id'])
    assert migrated['repository_context'] is False
    assert migrated['report'] == {'findings': []}
    assert migrated['review_notes'] == 'Keep this note.'
    assert CaseStore(path).get_case(case['case_id']) == migrated


def test_updates_are_allowlisted_and_validate_before_writing(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case = store.create_case('Investigate')
    for fields in [
        {'created_at': 'replace'}, {'status': 'published'},
        {'question': ''}, {'report': ['bad']}, {'report': {'confidence': float('nan')}},
    ]:
        with pytest.raises(ValueError):
            store.update_case(case['case_id'], **fields)
    with pytest.raises(TypeError):
        store.update_case(case['case_id'], **{'case_id': 'replace'})
    assert store.get_case(case['case_id'])['status'] == 'new'
    assert store.get_case(case['case_id'])['report'] is None
    for status in CASE_STATUSES:
        assert store.update_case(case['case_id'], status=status)['status'] == status


def test_atomic_claim_allows_one_execution_across_connections(tmp_path):
    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    case = store.create_case('Investigate')
    # The stores are created outside threads to isolate claim contention from DB setup.
    stores = [CaseStore(path) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=12) as executor:
        claimed = list(executor.map(lambda instance: instance.claim_case(case['case_id'], ('new',)), stores))
    assert claimed.count(True) == 1
    assert store.get_case(case['case_id'])['status'] == 'running'
    assert not store.claim_case(case['case_id'], ('running', 'new'))
    store.update_case(case['case_id'], status='failed')
    assert store.claim_case(case['case_id'], ('failed',))


def test_audit_events_append_without_mutating_caller_data(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case_id = store.create_case('Investigate')['case_id']
    event = {'phase': 'tool', 'name': 'search', 'status': 'success', 'counts': {'calls': 1}}
    store.add_event(case_id, event)
    store.add_event(case_id, {'phase': 'stage', 'status': 'awaiting_review'})
    events = CaseStore(store.path).events(case_id)
    assert len(events) == 2
    assert events[0]['event_id'] < events[1]['event_id']
    assert events[0]['name'] == 'search'
    assert events[0]['case_id'] == case_id
    assert events[0]['created_at']
    assert 'event_id' not in event
    with pytest.raises(ValueError):
        store.add_event(case_id, {'event_id': 12})
    assert len(store.events(case_id)) == 2


def test_immutable_evidence_idempotent_content_and_case_isolation(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    first = store.create_case('First')['case_id']
    second = store.create_case('Second')['case_id']
    record = {'source_key': 'doc-a', 'title': 'A', 'text': 'Original text', 'origin': 'snapshot',
              'url': 'https://example.test/a', 'fetched_at': '2026-10-22T12:00:00Z', 'commit_sha': 'abc'}
    original = store.put_evidence(first, record)
    duplicate = store.put_evidence(first, {**record, 'title': 'Changed', 'url': 'https://other.test'})
    assert duplicate == original
    changed = store.put_evidence(first, {**record, 'text': 'Updated text'})
    other_case = store.put_evidence(second, record)
    assert len({original['evidence_id'], changed['evidence_id'], other_case['evidence_id']}) == 3
    assert original['sha256'] != changed['sha256']
    assert original['sha256'] == other_case['sha256']
    assert len(store.evidence(first)) == 2
    assert len(store.evidence(second)) == 1


def test_unknown_case_cannot_receive_events_evidence_or_claim(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    operations = [
        lambda: store.get_case('missing'), lambda: store.update_case('missing', status='failed'),
        lambda: store.add_event('missing', {'phase': 'tool'}), lambda: store.events('missing'),
        lambda: store.put_evidence('missing', {'source_key': 'a', 'text': 'text'}),
        lambda: store.evidence('missing'), lambda: store.claim_case('missing', ('new',)),
    ]
    for operation in operations:
        with pytest.raises(KeyError):
            operation()


def test_case_listing_limits_and_orders_recent_updates(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    first = store.create_case('First')
    store.create_case('Second')
    store.update_case(first['case_id'], review_notes='Most recent')
    assert [row['question'] for row in store.list_cases(limit=1)] == ['First']
    for limit in (0, -1, True, 1001):
        with pytest.raises(ValueError):
            store.list_cases(limit=limit)


def test_execution_lock_blocks_other_instances_and_releases_after_exception(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    other = CaseStore(store.path)
    case_id = store.create_case('Investigate')['case_id']
    with pytest.raises(RuntimeError, match='simulated failure'):
        with store.execution_lock(case_id):
            with pytest.raises(CaseExecutionBusy):
                with other.execution_lock(case_id):
                    pytest.fail('A second session acquired the same case')
            raise RuntimeError('simulated failure')
    with other.execution_lock(case_id):
        assert other.claim_case(case_id, ('new',))


def test_recovery_skips_real_process_lock_and_recovers_after_process_is_killed(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case_id = store.create_case('Investigate')['case_id']
    context = multiprocessing.get_context('spawn')
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_hold_case_in_process, args=(str(store.path), case_id, child_connection))
    process.start()
    try:
        assert parent_connection.poll(10), 'Child did not acquire its execution lock'
        assert parent_connection.recv() == 'locked'
        restarted = CaseStore(store.path)
        assert restarted.recover_interrupted_cases() == []
        assert restarted.get_case(case_id)['status'] == 'running'
        with pytest.raises(CaseExecutionBusy):
            with restarted.execution_lock(case_id):
                pytest.fail('Acquired a lock held by another process')
        process.terminate()
        process.join(timeout=5)
        assert not process.is_alive()
        assert restarted.recover_interrupted_cases() == [case_id]
        assert restarted.recover_interrupted_cases() == []
        recovered = restarted.get_case(case_id)
        assert recovered['status'] == 'failed'
        assert recovered['error'] == 'Previous execution was interrupted; retry to resume saved progress.'
        assert restarted.events(case_id)[-1]['name'] == 'interrupted_execution'
        with restarted.execution_lock(case_id):
            assert restarted.claim_case(case_id, ('failed',))
            restarted.update_case(case_id, status='awaiting_review')
        assert restarted.recover_interrupted_cases() == []
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_connection.close()
        child_connection.close()


def test_recovery_skips_same_process_execution_and_never_touches_review_cases(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    active_id = store.create_case('Active')['case_id']
    review_id = store.create_case('Review')['case_id']
    store.update_case(review_id, status='awaiting_review')
    with store.execution_lock(active_id):
        assert store.claim_case(active_id, ('new',))
        assert CaseStore(store.path).recover_interrupted_cases() == []
    assert store.recover_interrupted_cases() == [active_id]
    assert store.get_case(review_id)['status'] == 'awaiting_review'


def test_execution_lock_rejects_unknown_cases_and_path_traversal(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    for invalid in ('../outside', '/tmp/elsewhere', '', 'a' * 32):
        with pytest.raises(ValueError):
            with store.execution_lock(invalid):
                pytest.fail('Acquired an invalid case lock')
    with pytest.raises(KeyError):
        with store.execution_lock('00000000-0000-0000-0000-000000000000'):
            pytest.fail('Acquired a nonexistent case lock')
    assert not (tmp_path / 'locks').exists()


def test_review_memory_consent_is_immutable_durable_and_legacy_defaults_off(tmp_path):
    path = tmp_path / 'cases.sqlite'
    store = CaseStore(path)
    off = store.create_case('No earlier reviews')
    on = store.create_case('Recall prior reviews', memory_context=True)
    assert off['memory_context'] is False
    assert CaseStore(path).get_case(on['case_id'])['memory_context'] is True
    with pytest.raises(ValueError, match='Cannot update'):
        store.update_case(off['case_id'], memory_context=True)
    for invalid in ('yes', 1, None):
        with pytest.raises(ValueError, match='boolean'):
            store.create_case('Invalid consent', memory_context=invalid)
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE cases DROP COLUMN memory_context')
    assert CaseStore(path).get_case(on['case_id'])['memory_context'] is False
