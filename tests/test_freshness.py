from datetime import datetime, timezone

from triage_desk.freshness import corpus_freshness


def test_oldest_source_controls_snapshot_age():
    now = datetime(2026, 10, 18, tzinfo=timezone.utc)
    sources = [{'snapshot_at': '2026-10-17T00:00:00Z'}, {'snapshot_at': '2026-10-10T00:00:00Z'}]
    result = corpus_freshness(sources, now=now)
    assert result['status'] == 'stale'
    assert result['age_days'] == 8


def test_missing_and_future_timestamps_cannot_claim_current_snapshot():
    now = datetime(2026, 10, 18, tzinfo=timezone.utc)
    assert corpus_freshness([{'snapshot_at': '2026-10-17T00:00:00Z'}, {}], now=now)['status'] == 'unknown'
    assert corpus_freshness([{'snapshot_at': '2026-10-19T00:00:00Z'}], now=now)['status'] == 'unknown'


def test_recent_snapshot_is_current_without_claiming_upstream_verification():
    result = corpus_freshness([{'snapshot_at': '2026-10-17T00:00:00Z'}],
                              now=datetime(2026, 10, 18, tzinfo=timezone.utc))
    assert result['status'] == 'current'
    assert result['age_days'] == 1
    assert result['refresh_mode'] == 'manual'
