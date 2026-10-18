"""Snapshot-age disclosure; a recent fetch does not verify upstream facts."""
from __future__ import annotations

from datetime import datetime, timezone


def corpus_freshness(sources: list[dict], *, now: datetime | None = None, max_age_days: int = 7) -> dict:
    if not isinstance(max_age_days, int) or isinstance(max_age_days, bool) or max_age_days < 1:
        raise ValueError('Maximum snapshot age must be a positive number of days')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Snapshot age requires a timezone-aware current time')
    snapshots = []
    missing = 0
    for source in sources:
        try:
            snapshot = datetime.fromisoformat(str(source.get('snapshot_at', '')).replace('Z', '+00:00'))
            if snapshot.tzinfo is None or snapshot > now:
                raise ValueError('Invalid snapshot time')
            snapshots.append(snapshot)
        except (ValueError, TypeError):
            missing += 1
    oldest = min(snapshots) if snapshots else None
    age_days = (now - oldest).total_seconds() / 86400 if oldest else None
    status = 'unknown' if missing or not oldest else 'stale' if age_days >= max_age_days else 'current'
    return {'status': status, 'oldest_snapshot_at': oldest.isoformat() if oldest else None,
            'age_days': round(age_days, 2) if age_days is not None else None,
            'max_age_days': max_age_days, 'missing_or_invalid_timestamps': missing,
            'refresh_mode': 'manual',
            'note': 'Snapshot age measures fetch time, not whether an upstream claim or version is still correct.'}
