"""Build a reviewable course source archive without local credentials or caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

from dotenv import dotenv_values


ROOT_FILES = {'.env.example', '.gitignore', '.python-version', 'README.md', 'app.py',
              'pyproject.toml', 'requirements.txt', 'uv.lock'}
SOURCE_DIRECTORIES = {'triage_desk', 'data', 'evaluation', 'scripts', 'tests', 'docs', 'agent_skills', '.github', '.streamlit'}
PRIVATE_PARTS = {'.git', '.venv', '.cache', '__pycache__', '.pytest_cache', 'node_modules', 'artifacts'}


def build_archive(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    # A known local credential must never be included, regardless of its filename.
    settings = dotenv_values(root / '.env') if (root / '.env').is_file() else {}
    secrets = [str(value).encode() for name, value in settings.items()
               if value and any(part in name.upper() for part in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET'))]
    entries = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path == output or PRIVATE_PARTS.intersection(relative.parts):
            continue
        workbook = (relative.parts[:2] == ('outputs', 'week4') and len(relative.parts) == 3
                    and relative.suffix.lower() == '.xlsx')
        allowed = ((len(relative.parts) == 1 and relative.name in ROOT_FILES)
                   or relative.parts[0] in SOURCE_DIRECTORIES or workbook)
        if not allowed or not path.is_file():
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        fixture_log = relative.parts[:-1] == ('evaluation', 'fixtures', 'workflows') and path.suffix.lower() == '.log'
        if (path.name == 'secrets.toml' or path.suffix.lower() in {'.pyc', '.zip'}
                or path.suffix.lower() == '.log' and not fixture_log or path.name == '.DS_Store'):
            continue
        if path.is_symlink():
            raise ValueError(f'Refusing a symbolic link in source assets: {relative}')
        content = path.read_bytes()
        contains_secret = any(secret in content for secret in secrets)
        # Office files compress their XML; scanning only the outer ZIP bytes
        # would miss a configured credential stored inside a workbook cell.
        if secrets and path.suffix.lower() in {'.xlsx', '.docx', '.pptx'} and zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as office:
                contains_secret = contains_secret or any(
                    any(secret in office.read(member) for secret in secrets)
                    for member in office.namelist() if not member.endswith('/'))
        if contains_secret:
            raise ValueError('A configured local credential appears in a candidate asset. No archive was written.')
        entries.append((relative.as_posix(), content))
    if not entries:
        raise ValueError('No source assets found')
    manifest = {
        'project': 'Engineering Issue Desk',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Local course asset archive; not a GitHub upload or course submission. Local credentials, environments and caches are excluded.',
        'files': [{'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
                  for name, content in entries],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries:
                archive.writestr('engineering-issue-desk/' + name, content)
            archive.writestr('engineering-issue-desk/SUBMISSION-MANIFEST.json', json.dumps(manifest, indent=2))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive': str(output), 'source_files': len(entries), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/engineering-issue-desk-week2.zip'))
    args = parser.parse_args()
    print(json.dumps(build_archive(Path(__file__).resolve().parents[1], args.output), indent=2))


if __name__ == '__main__':
    main()
