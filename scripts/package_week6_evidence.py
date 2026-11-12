"""Package only the frozen Week 6 observations and synthetic case definitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
RUNS = ('week6-baseline', 'week6-defended', 'week6-defense-controls', 'week6-pilot')


def build(output: Path) -> dict:
    entries = [('evaluation/week6-cases.json', (ROOT / 'evaluation/week6-cases.json').read_bytes())]
    for run in RUNS:
        directory = ROOT / 'evaluation' / run
        if not (directory / 'report.json').is_file():
            raise ValueError(f'The recorded run is not complete: {run}')
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.html', '.csv', '.py'}:
                if path.is_symlink():
                    raise ValueError('Evidence cannot contain symbolic links.')
                entries.append((path.relative_to(ROOT).as_posix(), path.read_bytes()))
    for path in sorted((ROOT / 'outputs/week6/screenshots').glob('*.png')):
        if path.is_symlink():
            raise ValueError('Evidence cannot contain symbolic links.')
        entries.append((path.relative_to(ROOT).as_posix(), path.read_bytes()))
    settings = {**dotenv_values(ROOT / '.env'), **os.environ}
    secrets = [str(value).encode() for key, value in settings.items()
               if value and len(str(value)) >= 8 and any(word in key.upper()
                   for word in ('API_KEY', 'TOKEN', 'PASSWORD', 'SECRET', 'CREDENTIAL'))]
    if any(secret in content for _, content in entries for secret in secrets):
        raise ValueError('Configured credential detected in evidence. No archive was written.')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Frozen live observations, separate deterministic guard controls, and synthetic input definitions. No local checkpoints or credentials.',
        'files': [{'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
                  for name, content in entries]}
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries:
                archive.writestr(name, content)
            archive.writestr('EVIDENCE-MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive': str(output), 'files': len(entries), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/week6-evidence.zip')
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
