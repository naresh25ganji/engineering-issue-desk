"""Package only the course source, data, prepared records, and local baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md", "SOURCE.md", "RESULTS.md", "LAUNCH.md", "pyproject.toml", "uv.lock", ".gitignore", "requirements-colab.txt"}
SOURCE_DIRS = {"router_lab", "scripts", "tests", "data", "notebooks", "configs"}


def build(output):
    paths = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or {".venv", ".cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}.intersection(relative.parts):
            continue
        source = relative.parts[0] in SOURCE_DIRS
        evidence = relative.parts[:2] in (("artifacts", "prepared"), ("artifacts", "baselines")) or relative.as_posix() == "artifacts/token-audit.json"
        if not (relative.as_posix() in ROOT_FILES or source or evidence):
            continue
        if path.suffix in (".zip", ".pyc") or path.name.startswith(".env"):
            continue
        if path.is_symlink():
            raise ValueError(f"Do not package symbolic links: {relative}")
        paths.append((relative, path.read_bytes()))
    receipt = {"project": "Week 5 support ticket router", "files": [
        {"path": rel.as_posix(), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()} for rel, content in paths
    ], "note": "Contains reproducible source and CPU baseline evidence. Actual Colab run evidence and trained adapter weights are delivered separately. Credentials and virtual environments are excluded."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, content in paths:
            archive.writestr("week5-ticket-router/" + relative.as_posix(), content)
        archive.writestr("week5-ticket-router/PACKAGE-MANIFEST.json", json.dumps(receipt, indent=2))
    return {"path": str(output.resolve()), "files": len(paths), "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/week5-ticket-router.zip")
    print(json.dumps(build(parser.parse_args().output), indent=2))
