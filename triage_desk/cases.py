"""Durable local investigation cases, evidence snapshots, and audit events.

Each operation owns a short-lived SQLite connection.  This makes a store safe to
reuse across Streamlit sessions and prevents a connection from crossing threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import UUID, uuid4


DEFAULT_CASE_DB = Path('.cache/agent/cases.sqlite')
CASE_STATUSES = frozenset({
    'new', 'running', 'awaiting_review', 'approved', 'rejected', 'failed', 'needs_revision',
})
_CASE_FIELDS = frozenset({
    'question', 'version', 'code_context', 'status', 'report', 'review_notes', 'error', 'report_markdown',
})
_ORIGINS = frozenset({'snapshot', 'live', 'user_code', 'repository', 'analysis'})


class CaseExecutionBusy(ValueError):
    """Another process or session owns the execution lock for this case."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: object, name: str, *, limit: int = 200_000, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f'{name} must be {"nonempty " if required else ""}text of at most {limit} characters')
    return value


def _json(value: object, name: str, *, limit: int = 500_000) -> str:
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain only JSON values') from exc
    if len(result) > limit:
        raise ValueError(f'{name} is too large')
    return result


class CaseStore:
    """Local SQLite case storage; an execution claim is atomic across processes."""

    def __init__(self, path: str | Path = DEFAULT_CASE_DB):
        if str(path) == ':memory:':
            raise ValueError('CaseStore needs a file path for durable, independent connections')
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '',
                    code_context TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'new',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    report_json TEXT,
                    review_notes TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    report_markdown TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    source_key TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(case_id, source_key, sha256)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    created_at TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cases_updated ON cases(updated_at DESC);
                CREATE INDEX IF NOT EXISTS evidence_case ON evidence(case_id);
                CREATE INDEX IF NOT EXISTS events_case ON events(case_id, event_id);
            ''')
            # Serialize the additive migration across simultaneous app sessions.
            # Older saved cases never acquire repository access implicitly.
            connection.execute('BEGIN IMMEDIATE')
            columns = {row['name'] for row in connection.execute('PRAGMA table_info(cases)')}
            if 'memory_context' not in columns:
                connection.execute('ALTER TABLE cases ADD COLUMN memory_context INTEGER NOT NULL DEFAULT 0 CHECK(memory_context IN (0,1))')
            if 'repository_context' not in columns:
                connection.execute('ALTER TABLE cases ADD COLUMN repository_context INTEGER NOT NULL DEFAULT 0 CHECK(repository_context IN (0,1))')

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=15000')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _case(row: sqlite3.Row) -> dict:
        result = dict(row)
        report_json = result.pop('report_json')
        result['report'] = json.loads(report_json) if report_json else None
        result['repository_context'] = bool(result['repository_context'])
        result['memory_context'] = bool(result['memory_context'])
        return result

    @staticmethod
    def _require_case(connection: sqlite3.Connection, case_id: str) -> None:
        if connection.execute('SELECT 1 FROM cases WHERE case_id=?', (case_id,)).fetchone() is None:
            raise KeyError(f'Unknown investigation case: {case_id}')

    def create_case(self, question: str, version: str = '', code_context: str = '', *, repository_context: bool = False, memory_context: bool = False) -> dict:
        question = _text(question, 'question', limit=20_000, required=True)
        version = _text(version, 'version', limit=200)
        code_context = _text(code_context, 'code_context')
        if not isinstance(repository_context, bool):
            raise ValueError('repository_context must be a boolean')
        if not isinstance(memory_context, bool):
            raise ValueError('memory_context must be a boolean')
        case_id, timestamp = str(uuid4()), _now()
        with self._connection() as connection:
            connection.execute(
                'INSERT INTO cases(case_id,question,version,code_context,created_at,updated_at,repository_context,memory_context) VALUES(?,?,?,?,?,?,?,?)',
                (case_id, question, version, code_context, timestamp, timestamp, int(repository_context), int(memory_context)),
            )
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM cases WHERE case_id=?', (case_id,)).fetchone()
        if row is None:
            raise KeyError(f'Unknown investigation case: {case_id}')
        return self._case(row)

    def list_cases(self, limit: int = 30) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError('limit must be between 1 and 1000')
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM cases ORDER BY updated_at DESC, case_id LIMIT ?', (limit,),
            ).fetchall()
        return [self._case(row) for row in rows]

    def update_case(self, case_id: str, **allowed_fields: object) -> dict:
        unknown = set(allowed_fields) - _CASE_FIELDS
        if unknown:
            raise ValueError(f'Cannot update case fields: {", ".join(sorted(unknown))}')
        values = dict(allowed_fields)
        for name, value in list(values.items()):
            if name == 'report':
                if value is not None and not isinstance(value, dict):
                    raise ValueError('report must be a JSON object or None')
                values['report_json'] = None if value is None else _json(value, 'report')
                del values['report']
            else:
                limit = {'version': 200, 'question': 20_000, 'review_notes': 20_000,
                         'error': 20_000}.get(name, 200_000)
                _text(value, name, limit=limit, required=name == 'question')
        if 'status' in values and values['status'] not in CASE_STATUSES:
            raise ValueError('Unknown investigation case status')
        if values:
            values['updated_at'] = _now()
            with self._connection() as connection:
                self._require_case(connection, case_id)
                # Column names come only from the fixed allowlist above.
                connection.execute(
                    f'UPDATE cases SET {", ".join(f"{name}=?" for name in values)} WHERE case_id=?',
                    (*values.values(), case_id),
                )
        return self.get_case(case_id)

    def claim_case(self, case_id: str, allowed_statuses: tuple[str, ...]) -> bool:
        """Acquire execution once; a running case can never be claimed again."""
        if not isinstance(allowed_statuses, tuple) or any(
            not isinstance(status, str) or status not in CASE_STATUSES for status in allowed_statuses
        ):
            raise ValueError('allowed_statuses must be a tuple of case statuses')
        allowed = tuple(status for status in allowed_statuses if status != 'running')
        with self._connection() as connection:
            self._require_case(connection, case_id)
            if not allowed:
                return False
            placeholders = ','.join('?' for _ in allowed)
            cursor = connection.execute(
                f'UPDATE cases SET status=?, updated_at=? WHERE case_id=? AND status IN ({placeholders})',
                ('running', _now(), case_id, *allowed),
            )
            return cursor.rowcount == 1

    @contextmanager
    def execution_lock(self, case_id: str) -> Iterator[None]:
        """Hold a nonblocking, process-safe lock across a case execution.

        The caller must acquire this before claiming a case and hold it until
        the final case state is saved. The OS releases the advisory lock after
        a process crash. Lock files are retained to avoid inode replacement races.
        """
        try:
            if not isinstance(case_id, str) or str(UUID(case_id)) != case_id:
                raise ValueError('Invalid investigation case ID')
        except (AttributeError, ValueError) as exc:
            raise ValueError('Invalid investigation case ID') from exc
        self.get_case(case_id)
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError('Process-safe case execution requires macOS or Linux file locking') from exc
        directory = self.path.parent / 'locks'
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / f'{case_id}.lock', os.O_CREAT | os.O_RDWR, 0o600)
        locked = False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CaseExecutionBusy('This investigation is already running in another session.') from exc
            locked = True
            yield
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def recover_interrupted_cases(self) -> list[str]:
        """Recover only running cases whose OS execution lock is no longer held."""
        with self._connection() as connection:
            running = connection.execute("SELECT case_id FROM cases WHERE status='running'").fetchall()
        recovered = []
        reason = 'Previous execution was interrupted; retry to resume saved progress.'
        for row in running:
            case_id = row['case_id']
            try:
                with self.execution_lock(case_id):
                    with self._connection() as connection:
                        timestamp = _now()
                        cursor = connection.execute(
                            "UPDATE cases SET status='failed', error=?, updated_at=? WHERE case_id=? AND status='running'",
                            (reason, timestamp, case_id),
                        )
                        if cursor.rowcount:
                            connection.execute(
                                'INSERT INTO events(case_id,created_at,event_json) VALUES(?,?,?)',
                                (case_id, timestamp, _json({'phase': 'execution', 'name': 'interrupted_execution',
                                                          'status': 'recovered'}, 'event')),
                            )
                            recovered.append(case_id)
            except CaseExecutionBusy:
                continue
        return recovered

    def add_event(self, case_id: str, event: dict) -> None:
        if not isinstance(event, dict) or set(event) & {'event_id', 'case_id', 'created_at'}:
            raise ValueError('event must be an object without reserved audit metadata')
        payload = _json(event, 'event', limit=100_000)
        with self._connection() as connection:
            self._require_case(connection, case_id)
            connection.execute(
                'INSERT INTO events(case_id,created_at,event_json) VALUES(?,?,?)', (case_id, _now(), payload),
            )

    def events(self, case_id: str) -> list[dict]:
        with self._connection() as connection:
            self._require_case(connection, case_id)
            rows = connection.execute(
                'SELECT * FROM events WHERE case_id=? ORDER BY event_id', (case_id,),
            ).fetchall()
        return [{**json.loads(row['event_json']), 'event_id': row['event_id'],
                 'case_id': case_id, 'created_at': row['created_at']} for row in rows]

    def put_evidence(self, case_id: str, record: dict) -> dict:
        """Insert an immutable source snapshot, or return its existing original."""
        if not isinstance(record, dict):
            raise ValueError('evidence must be an object')
        source_key = _text(record.get('source_key'), 'source_key', limit=1000, required=True)
        text = _text(record.get('text'), 'text', limit=200_000, required=True)
        origin = record.get('origin', 'snapshot')
        if not isinstance(origin, str) or origin not in _ORIGINS:
            raise ValueError('Unknown evidence origin')
        if not isinstance(record.get('truncated', False), bool):
            raise ValueError('truncated must be a boolean')
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        identity = f'{case_id}\0{source_key}\0{content_hash}'
        evidence_id = 'ev_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]
        result = {
            'evidence_id': evidence_id, 'case_id': case_id, 'source_key': source_key,
            'title': _text(record.get('title', source_key), 'title', limit=2000, required=True),
            'text': text, 'sha256': content_hash,
            'url': _text(record.get('url', ''), 'url', limit=4000),
            'kind': _text(record.get('kind', 'documentation'), 'kind', limit=100, required=True),
            'origin': origin,
            'fetched_at': _text(record.get('fetched_at') or (
                record.get('snapshot_at', '') if origin == 'snapshot' else _now()
            ), 'fetched_at', limit=200),
            'commit_sha': _text(record.get('commit_sha', ''), 'commit_sha', limit=200),
            'snapshot_at': _text(record.get('snapshot_at', ''), 'snapshot_at', limit=200),
            'version_scope': _text(record.get('version_scope', ''), 'version_scope', limit=2000),
            'base_sha': _text(record.get('base_sha', ''), 'base_sha', limit=200),
            'filename': _text(record.get('filename', ''), 'filename', limit=500),
            'truncated': record.get('truncated', False),
            'evidence_type': _text(record.get('evidence_type', ''), 'evidence_type', limit=100),
            'published_at': _text(record.get('published_at', ''), 'published_at', limit=200),
            'acquired_at': _text(record.get('acquired_at', ''), 'acquired_at', limit=200),
        }
        with self._connection() as connection:
            self._require_case(connection, case_id)
            connection.execute(
                'INSERT OR IGNORE INTO evidence(evidence_id,case_id,source_key,sha256,record_json) VALUES(?,?,?,?,?)',
                (evidence_id, case_id, source_key, content_hash, _json(result, 'evidence')),
            )
            original = connection.execute(
                'SELECT record_json FROM evidence WHERE case_id=? AND source_key=? AND sha256=?',
                (case_id, source_key, content_hash),
            ).fetchone()
        return json.loads(original['record_json'])

    def evidence(self, case_id: str) -> list[dict]:
        with self._connection() as connection:
            self._require_case(connection, case_id)
            rows = connection.execute(
                'SELECT record_json FROM evidence WHERE case_id=? ORDER BY rowid', (case_id,),
            ).fetchall()
        return [json.loads(row['record_json']) for row in rows]
