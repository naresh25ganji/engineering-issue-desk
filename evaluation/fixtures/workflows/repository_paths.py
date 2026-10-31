# Frozen source excerpt: triage_desk/repository_context.py:74-86
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
