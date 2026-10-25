"""Opt-in, bounded, read-only context from this project's Python source files.

Only ``app.py`` and nonhidden Python files beneath ``triage_desk/`` are eligible.
No code is imported or executed. Source bodies become evidence only after an
explicit read; inventory and literal search results are discovery leads.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Annotated, Iterable

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .agent_budget import BudgetExceeded


MAX_FILES = 80
MAX_DIRECTORY_ENTRIES = 400
MAX_DIRECTORIES = 20
MAX_DEPTH = 5
MAX_FILE_BYTES = 256_000
MAX_SEARCH_BYTES = 512_000
MAX_SEARCH_CHARS = 100_000
MAX_SEARCH_RESULTS = 12
MAX_SEARCH_LINE_CHARS = 400
MAX_EXCERPT_CHARS = 8_000
MAX_READ_LINES = 160
_EXCLUDED = frozenset({'tests', 'test', 'data', 'node_modules', 'venv', '__pycache__'})
_TOKEN = re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|'
                    r'github_pat_[A-Za-z0-9_]{20,}|fw_[A-Za-z0-9_-]{20,})')
_LITERAL_CREDENTIAL = re.compile(
    r'''(?im)\b(?:api_?key|access_?token|auth_?token|password|secret)\s*(?::\s*str\s*)?=\s*["']([^"'\r\n]{8,})["']''',
)
RepositoryPath = Annotated[str, Field(min_length=1, max_length=500)]
RepositoryQuery = Annotated[str, Field(min_length=1, max_length=160)]
RepositoryLine = Annotated[int, Field(strict=True, ge=1, le=100_000)]


class RepositoryContextError(ValueError):
    """A controlled error with no absolute path, source text, or credentials."""

    def __init__(self, message: str, *, bytes_read: int = 0):
        super().__init__(message)
        self.bytes_read = bytes_read


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class RepositoryContext:
    """Confine all filesystem access to an explicit project-source allowlist."""

    def __init__(self, root: str | Path, *, forbidden_values: Iterable[str] = ()):
        supplied = Path(root).expanduser().absolute()
        if supplied.is_symlink() or not supplied.is_dir():
            raise RepositoryContextError('Repository context requires an existing nonsymlink project directory.')
        self.root = supplied.resolve()
        # Do not read environment variables or files to discover credentials.
        self._forbidden = tuple(value for value in forbidden_values
                                if isinstance(value, str) and 8 <= len(value) <= 4096)

    def _sensitive(self, text: str) -> bool:
        return (any(value in text for value in self._forbidden)
                or bool(_TOKEN.search(text)) or bool(_LITERAL_CREDENTIAL.search(text)))

    def _parts(self, path: str) -> tuple[str, ...]:
        if (not isinstance(path, str) or not path or len(path) > 500 or '\\' in path
                or any(ord(character) < 32 or ord(character) == 127 for character in path)):
            raise RepositoryContextError('Use a project-relative Python source path from the repository inventory.')
        parts = tuple(path.split('/'))
        if (PurePosixPath(path).is_absolute() or len(parts) > MAX_DEPTH + 1
                or any(not part or part in {'.', '..'} for part in parts)
                or any(part.startswith('.') or part in _EXCLUDED for part in parts)
                or self._sensitive(path)):
            raise RepositoryContextError('The requested path is outside the repository source allowlist.')
        if path != 'app.py' and not (len(parts) >= 2 and parts[0] == 'triage_desk' and parts[-1].endswith('.py')):
            raise RepositoryContextError('Only app.py and triage_desk Python source files are available.')
        return parts

    @contextmanager
    def _root_fd(self):
        descriptor = None
        try:
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            yield descriptor
        except OSError:
            raise RepositoryContextError('Repository source is unavailable or contains a symlink.') from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def _file_fd(self, parts: tuple[str, ...]):
        opened = []
        with self._root_fd() as root_fd:
            parent = root_fd
            try:
                for part in parts[:-1]:
                    descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    opened.append(descriptor)
                    parent = descriptor
                descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                opened.append(descriptor)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RepositoryContextError('The requested source is not a regular file.')
                yield descriptor
            except OSError:
                raise RepositoryContextError('Repository source is unavailable or contains a symlink.') from None
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)

    def _read(self, path: str, *, byte_limit: int = MAX_FILE_BYTES) -> dict:
        parts = self._parts(path)
        raw = bytearray()
        try:
            with self._file_fd(parts) as descriptor:
                before = os.fstat(descriptor)
                if before.st_nlink != 1:
                    raise RepositoryContextError('Source files with multiple hard links are withheld.')
                if before.st_size > byte_limit:
                    raise RepositoryContextError('The source exceeds the bounded file or search size limit.')
                while len(raw) < byte_limit:
                    chunk = os.read(descriptor, min(65_536, byte_limit - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                after = os.fstat(descriptor)
        except RepositoryContextError as exc:
            exc.bytes_read = max(exc.bytes_read, len(raw))
            raise
        if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RepositoryContextError('The source changed while being read; retry for a consistent snapshot.', bytes_read=len(raw))
        try:
            text = raw.decode('utf-8')
        except UnicodeError:
            raise RepositoryContextError('Repository source must be readable UTF-8 text.', bytes_read=len(raw)) from None
        if '\x00' in text or self._sensitive(text):
            raise RepositoryContextError('The source is withheld because it may contain credential or binary material.',
                                         bytes_read=len(raw))
        return {'path': path, 'text': text, 'file_sha256': _sha(bytes(raw)), 'size_bytes': len(raw)}

    def inventory(self) -> dict:
        files, entries_seen, directories_seen, truncated = [], 0, 0, False

        def add(path, info):
            nonlocal truncated
            try:
                self._parts(path)
            except RepositoryContextError:
                return
            if stat.S_ISREG(info.st_mode):
                if len(files) < MAX_FILES:
                    files.append({'path': path, 'size_bytes': info.st_size})
                else:
                    truncated = True

        def walk(descriptor, prefix, depth):
            nonlocal entries_seen, directories_seen, truncated
            if depth > MAX_DEPTH or directories_seen >= MAX_DIRECTORIES:
                truncated = True
                return
            directories_seen += 1
            selected = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    if entries_seen >= MAX_DIRECTORY_ENTRIES:
                        truncated = True
                        break
                    entries_seen += 1
                    selected.append(entry.name)
            for name in sorted(selected):
                if (name.startswith('.') or name in _EXCLUDED or '/' in name or '\\' in name
                        or any(ord(character) < 32 or ord(character) == 127 for character in name)):
                    continue
                path = f'{prefix}/{name}'
                try:
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                        try:
                            walk(child, path, depth + 1)
                        finally:
                            os.close(child)
                    elif name.endswith('.py'):
                        add(path, info)
                except OSError:
                    continue

        with self._root_fd() as descriptor:
            try:
                add('app.py', os.stat('app.py', dir_fd=descriptor, follow_symlinks=False))
            except OSError:
                pass
            try:
                child = os.open('triage_desk', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except OSError:
                child = None
            if child is not None:
                try:
                    walk(child, 'triage_desk', 1)
                finally:
                    os.close(child)
        return {'files': sorted(files, key=lambda item: item['path']), 'truncated': truncated,
                'limits': {'files': MAX_FILES, 'directory_entries': MAX_DIRECTORY_ENTRIES,
                           'directories': MAX_DIRECTORIES, 'depth': MAX_DEPTH},
                'notes': 'Bounded source inventory only. Missing paths are not proof of absence. No source bodies were read.'}

    def search(self, query: str) -> dict:
        if (not isinstance(query, str) or not query.strip() or len(query) > 160
                or '\n' in query or '\r' in query or self._sensitive(query)):
            raise RepositoryContextError('Use a nonempty literal source query of at most 160 characters without credentials.')
        inventory = self.inventory()
        matches, files_scanned, chars_scanned, bytes_scanned = [], 0, 0, 0
        truncated = inventory['truncated']
        for file_index, item in enumerate(inventory['files']):
            if chars_scanned >= MAX_SEARCH_CHARS or bytes_scanned >= MAX_SEARCH_BYTES or len(matches) >= MAX_SEARCH_RESULTS:
                truncated = True
                break
            try:
                snapshot = self._read(item['path'], byte_limit=min(MAX_FILE_BYTES, MAX_SEARCH_BYTES - bytes_scanned))
            except RepositoryContextError as exc:
                bytes_scanned += exc.bytes_read
                truncated = True
                continue
            bytes_scanned += snapshot['size_bytes']
            files_scanned += 1
            remaining = MAX_SEARCH_CHARS - chars_scanned
            # Give every indexed module an opportunity. A large app or early
            # alphabetic module must not consume the budget before an imported
            # helper's module is even inspected.
            remaining_files = len(inventory['files']) - file_index
            allowance = max(1, remaining // remaining_files)
            text = snapshot['text'][:allowance]
            chars_scanned += len(text)
            truncated |= len(text) < len(snapshot['text'])
            for number, line in enumerate(text.splitlines(keepends=True), 1):
                match_at = line.casefold().find(query.casefold())
                if match_at < 0:
                    continue
                left = max(0, min(match_at - 80, len(line) - MAX_SEARCH_LINE_CHARS))
                matches.append({'path': item['path'], 'line_number': number,
                                'line': line[left:left + MAX_SEARCH_LINE_CHARS], 'column_start': left + 1,
                                'line_truncated': len(line) > MAX_SEARCH_LINE_CHARS,
                                'file_sha256': snapshot['file_sha256']})
                if len(matches) >= MAX_SEARCH_RESULTS:
                    truncated = True
                    break
        return {'matches': matches, 'scanned_files': files_scanned, 'scanned_characters': chars_scanned,
                'scanned_bytes': bytes_scanned, 'truncated': truncated,
                'notes': 'Case-insensitive literal search over bounded file prefixes, fairly sharing the character budget across indexed modules. These are discovery leads, not registered evidence. Read the relevant lines before citing. Later lines, bounded files, or withheld files may be absent.'}

    def read(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        if (not isinstance(start_line, int) or isinstance(start_line, bool) or not 1 <= start_line <= 100_000
                or (end_line is not None and (not isinstance(end_line, int) or isinstance(end_line, bool)
                                              or not start_line <= end_line <= 100_000))):
            raise RepositoryContextError('Choose valid positive inclusive source line bounds.')
        requested_end = end_line if end_line is not None else start_line + 119
        if requested_end - start_line + 1 > MAX_READ_LINES:
            raise RepositoryContextError('A source read may select at most 160 lines.')
        snapshot = self._read(path)
        lines = snapshot.pop('text').splitlines(keepends=True)
        if not lines or start_line > len(lines):
            raise RepositoryContextError('The requested line range is outside the source file.')
        selected, count = [], 0
        for line in lines[start_line - 1:min(requested_end, len(lines))]:
            if count + len(line) > MAX_EXCERPT_CHARS:
                break
            selected.append(line)
            count += len(line)
        if not selected:
            raise RepositoryContextError('The selected source line exceeds the context character limit.')
        excerpt = ''.join(selected)
        actual_end = start_line + len(selected) - 1
        return {**snapshot, 'start_line': start_line, 'end_line': actual_end, 'total_lines': len(lines),
                'excerpt': excerpt, 'excerpt_sha256': _sha(excerpt.encode('utf-8')),
                'truncated': actual_end < min(requested_end, len(lines)),
                'partial_file': start_line > 1 or actual_end < len(lines)}


def build_repository_tools(root, registry, budget, *, forbidden_values: Iterable[str] = ()) -> list[BaseTool]:
    """Create explicitly enabled, self-accounted tools; no provider calls or writes."""
    context = RepositoryContext(root, forbidden_values=forbidden_values)

    def run(name, action):
        with budget.tool_call(name) as event:
            try:
                result = {'status': 'ok', 'evidence': [], **action()}
            except BudgetExceeded:
                raise
            except (RepositoryContextError, OSError) as exc:
                message = str(exc) if isinstance(exc, RepositoryContextError) else 'Repository source could not be read.'
                result = {'status': 'error', 'error': message, 'evidence': []}
            event.update(status=result['status'], evidence_count=len(result['evidence']))
            if result['status'] == 'error':
                event['error_type'] = 'RepositoryContextUnavailable'
            return result

    @tool
    def list_repository_files() -> dict:
        """List bounded eligible local Python source paths: app.py and triage_desk/. Excludes hidden files, tests, data, symlinks and other host paths. Inventory only; read source before citing."""
        return run('list_repository_files', context.inventory)

    @tool
    def search_repository(query: RepositoryQuery) -> dict:
        """Search eligible local Python files using a case-insensitive literal string, not a regex. Returns bounded line leads and full-file hashes. Read the relevant source before citing; no code is executed."""
        return run('search_repository', lambda: context.search(query))

    @tool
    def read_repository_file(path: RepositoryPath, start_line: RepositoryLine = 1,
                             end_line: RepositoryLine | None = None) -> dict:
        """Read and register up to 160 inclusive lines of an eligible project-relative Python path. Defaults to 120 lines, capped at 8000 characters. Returns saved literal quotes, file/excerpt hashes and line provenance; never executes code."""
        def action():
            snapshot = context.read(path, start_line, end_line)
            timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
            location = f'{snapshot["path"]}:{snapshot["start_line"]}-{snapshot["end_line"]}'
            header = (f'Repository snapshot: {location}\nFile SHA-256: {snapshot["file_sha256"]}\n'
                      f'Excerpt SHA-256: {snapshot["excerpt_sha256"]}\n\n')
            record = registry.register_source(
                source_key=f'repository:{location}:{snapshot["file_sha256"]}', title=location,
                text=header + snapshot['excerpt'], kind='code', origin='repository', url='',
                fetched_at=timestamp, snapshot_at=timestamp, evidence_type='repository_excerpt',
                filename=snapshot['path'], truncated=snapshot['truncated'] or snapshot['partial_file'],
                version_scope=f'Local working-tree snapshot; lines {snapshot["start_line"]}-{snapshot["end_line"]} '
                              f'of {snapshot["total_lines"]}; full-file SHA-256 {snapshot["file_sha256"]}. '
                              'Static inspection only; no tests or code were executed.',
            )
            return {'evidence': [record], 'file': {key: value for key, value in snapshot.items() if key != 'excerpt'},
                    'notes': 'Source bytes were saved as a bounded literal snapshot. The working tree may change; file and excerpt hashes identify this observation.'}
        return run('read_repository_file', action)

    return [list_repository_files, search_repository, read_repository_file]
