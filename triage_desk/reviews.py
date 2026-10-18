"""Local user outcome labels, storing no report text, answers or credentials."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading

OUTCOMES = {'resolved', 'needs_followup', 'human_handoff'}
_LOCK = threading.Lock()


def save_outcome(path: Path, run_id: str, outcome: str, *, first_contact: bool = True) -> None:
    if not re.fullmatch(r'[a-f0-9]{32}', run_id) or outcome not in OUTCOMES:
        raise ValueError('Choose a valid run and outcome')
    record = {'run_id': run_id, 'outcome': outcome, 'first_contact': bool(first_contact),
              'reviewed_at': datetime.now(timezone.utc).isoformat()}
    path = Path(path)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')


def outcome_summary(path: Path) -> dict:
    latest = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
                if record.get('outcome') in OUTCOMES:
                    latest[record['run_id']] = record
            except (ValueError, TypeError, KeyError):
                continue
    first = [row for row in latest.values() if row.get('first_contact')]
    resolved = sum(row['outcome'] == 'resolved' for row in first)
    return {'reviewed_runs': len(latest), 'reviewed_first_contacts': len(first),
            'user_reported_resolved_first_contacts': resolved,
            'user_reported_first_contact_resolution': resolved / len(first) if first else None,
            'note': 'User outcome labels on reviewed first contacts only; not an automated quality score or a claim about unreviewed users.'}
