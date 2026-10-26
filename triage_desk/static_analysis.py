"""Bounded Bandit checks over code already registered in this investigation.

Bandit examines Python ASTs; it does not verify vulnerabilities or execute the
supplied program. The model selects an evidence ID, never a path or command.
Official reference: https://bandit.readthedocs.io/en/latest/
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .agent_budget import BudgetExceeded


MAX_SOURCE_CHARS = 16000
MAX_OUTPUT_BYTES = 65536
WORKER_TIMEOUT_SECONDS = 6
RULESET_VERSION = 'issue-desk-bandit-v1'
RULE_LABELS = {
    'B102': 'Dynamic exec API pattern', 'B301': 'Potentially unsafe deserialization API pattern',
    'B307': 'Dynamic eval API pattern', 'B324': 'Weak hash configuration pattern',
    'B506': 'YAML load API pattern', 'B602': 'Shell-enabled subprocess pattern',
    'B605': 'Shell-capable process API pattern', 'B608': 'SQL string-construction pattern',
}
_HEX = r'[0-9a-f]{64}'
_REPOSITORY_HEADER = re.compile(
    rf'\ARepository snapshot: (?P<file>[^\r\n]+):(?P<start>\d+)-(?P<end>\d+)\r?\n'
    rf'File SHA-256: {_HEX}\r?\nExcerpt SHA-256: (?P<excerpt>{_HEX})\r?\n\r?\n')
_SUPPLIED_HEADER = re.compile(
    rf'\ASupplied repository file: (?P<file>[^\r\n]+):(?P<start>\d+)-(?P<end>\d+)\r?\n'
    rf'File SHA-256: {_HEX}\r?\n\r?\n')
_CLI_HEADER = re.compile(r'\A# Supplied file: (?P<file>[^\r\n]+)\r?\n')
_VERSION = re.compile(r'\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]{0,40})\Z')
EvidenceID = Annotated[str, Field(min_length=1, max_length=100)]


class StaticAnalysisUnavailable(ValueError):
    """A fixed label suitable for a tool result, never raw analyzer output."""

    def __init__(self, code: str):
        messages = {
            'unknown_evidence': 'Select an existing code evidence ID from this case.',
            'source_not_code': 'Static checks accept only registered user or repository code.',
            'source_limit': 'Registered code must contain 1 to 16000 characters.',
            'invalid_header': 'Code provenance header is invalid or inconsistent with the saved excerpt.',
            'unsupported_encoding': 'Static checks require UTF-8 Python source.',
            'timeout': 'Static checks reached their isolated-worker time limit.',
            'output_limit': 'Static checks exceeded the bounded result size.',
            'worker_unavailable': 'The isolated static analyzer could not finish.',
            'invalid_output': 'The isolated static analyzer returned an invalid result.',
        }
        self.code = code if code in messages else 'worker_unavailable'
        super().__init__(messages[self.code])


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _safe_label(value: str) -> str:
    if not isinstance(value, str):
        raise StaticAnalysisUnavailable('invalid_header')
    path = PurePosixPath(value)
    if (len(value) > 500 or not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', value)
            or path.is_absolute() or any(part in {'.', '..'} or part.startswith('.') for part in value.split('/'))):
        raise StaticAnalysisUnavailable('invalid_header')
    return value


def _code_snapshot(source: dict) -> dict:
    if source.get('origin') not in {'user_code', 'repository'} or source.get('kind') != 'code':
        raise StaticAnalysisUnavailable('source_not_code')
    text = source.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SOURCE_CHARS:
        raise StaticAnalysisUnavailable('source_limit')
    code, filename, source_start, removed_lines, partial = text, '', 1, 0, bool(source.get('truncated', False))
    header = _REPOSITORY_HEADER.match(text) or _SUPPLIED_HEADER.match(text)
    if header:
        filename = _safe_label(header['file'])
        source_start, end = int(header['start']), int(header['end'])
        code = text[header.end():]
        if not 1 <= source_start <= end <= 100000 or len(code.splitlines()) != end - source_start + 1:
            raise StaticAnalysisUnavailable('invalid_header')
        if 'excerpt' in header.groupdict() and _sha(code) != header['excerpt']:
            raise StaticAnalysisUnavailable('invalid_header')
        removed_lines = text[:header.end()].count('\n')
        partial = partial or source_start > 1
    elif text.startswith(('Repository snapshot:', 'Supplied repository file:')):
        raise StaticAnalysisUnavailable('invalid_header')
    else:
        header = _CLI_HEADER.match(text)
        if header:
            filename, code, removed_lines = _safe_label(header['file']), text[header.end():], 1
        elif source.get('filename'):
            filename = _safe_label(source['filename'])
    if not code.strip():
        raise StaticAnalysisUnavailable('source_limit')
    # Reject contradictory byte-encoding declarations instead of allowing the
    # string AST precheck and Bandit's byte parser to see different programs.
    for line in code.splitlines()[:2]:
        cookie = re.search(r'^[ \t\f]*#.*?coding[:=]\s*([-\w.]+)', line)
        if cookie and cookie[1].lower().replace('_', '-') not in {'utf-8', 'utf8'}:
            raise StaticAnalysisUnavailable('unsupported_encoding')
    return {'code': code, 'filename': filename, 'source_start_line': source_start,
            'removed_header_lines': removed_lines, 'partial_source': partial,
            'code_sha256': _sha(code), 'parent_sha256': _sha(text)}


def _run_worker(code: str) -> dict:
    worker = Path(__file__).with_name('_static_analysis_worker.py').resolve()
    with TemporaryDirectory(prefix='issue-desk-static-') as directory:
        output_path = Path(directory) / 'result.json'
        try:
            with output_path.open('wb') as output:
                completed = subprocess.run(
                    [sys.executable, '-I', str(worker)], input=json.dumps({'code': code}).encode('utf-8'),
                    stdout=output, stderr=subprocess.DEVNULL, cwd=directory,
                    env={'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'},
                    timeout=WORKER_TIMEOUT_SECONDS, shell=False, check=False, close_fds=True,
                )
            if output_path.stat().st_size > MAX_OUTPUT_BYTES:
                raise StaticAnalysisUnavailable('output_limit')
            if completed.returncode != 0:
                raise StaticAnalysisUnavailable('worker_unavailable')
            result = json.loads(output_path.read_bytes())
        except StaticAnalysisUnavailable:
            raise
        except subprocess.TimeoutExpired:
            raise StaticAnalysisUnavailable('timeout') from None
        except (OSError, ValueError):
            raise StaticAnalysisUnavailable('worker_unavailable') from None
    try:
        if (not isinstance(result, dict) or set(result) - {'version', 'python_version', 'status', 'findings', 'finding_count', 'truncated', 'error_line'}
                or not _VERSION.fullmatch(str(result.get('version', '')))
                or not _VERSION.fullmatch(str(result.get('python_version', '')))
                or result.get('status') not in {'ok', 'syntax_error', 'complexity_limit', 'analysis_error'}
                or not isinstance(result.get('findings'), list) or len(result['findings']) > 50
                or not isinstance(result.get('finding_count'), int) or isinstance(result['finding_count'], bool)
                or not len(result['findings']) <= result['finding_count'] <= 6000 or not isinstance(result.get('truncated'), bool)
                or result['truncated'] != (result['finding_count'] > len(result['findings']))):
            raise StaticAnalysisUnavailable('invalid_output')
        for finding in result['findings']:
            if (not isinstance(finding, dict) or set(finding) != {'rule_id', 'line', 'severity', 'confidence'}
                    or finding.get('rule_id') not in RULE_LABELS
                    or not isinstance(finding.get('line'), int) or isinstance(finding['line'], bool)
                    or not 1 <= finding['line'] <= len(code.splitlines())
                    or finding.get('severity') not in {'LOW', 'MEDIUM', 'HIGH'}
                    or finding.get('confidence') not in {'LOW', 'MEDIUM', 'HIGH'}):
                raise StaticAnalysisUnavailable('invalid_output')
    except (KeyError, TypeError, ValueError):
        raise StaticAnalysisUnavailable('invalid_output') from None
    return result


def build_static_analysis_tools(registry, budget) -> list[BaseTool]:
    """Expose one self-accounted tool with no path, command, or rule arguments."""
    @tool
    def run_static_checks(evidence_id: EvidenceID) -> dict:
        """Run a fixed Bandit rule subset on already registered user_code or repository Python code. Only evidence_id is accepted; no host paths, shell commands, source execution, or tests. Findings are attributed analyzer flags, not verified vulnerabilities. Incomplete Python excerpts can fail syntax checks."""
        with budget.tool_call('run_static_checks', {'evidence_id': evidence_id}) as event:
            try:
                parent = next((source for source in registry.list_sources() if source['evidence_id'] == evidence_id), None)
                if parent is None:
                    raise StaticAnalysisUnavailable('unknown_evidence')
                snapshot = _code_snapshot(parent)
                checked = _run_worker(snapshot['code'])
                findings = []
                for finding in checked['findings']:
                    line = finding['line']
                    findings.append({**finding, 'line': snapshot['source_start_line'] + line - 1,
                                     'evidence_line': snapshot['removed_header_lines'] + line,
                                     'label': RULE_LABELS[finding['rule_id']]})
                fingerprint = _sha(json.dumps({'parent': evidence_id, 'parent_sha256': snapshot['parent_sha256'],
                    'code_sha256': snapshot['code_sha256'], 'bandit_version': checked['version'],
                    'python_version': checked['python_version'], 'ruleset': RULESET_VERSION}, sort_keys=True))
                lines = [f'Bandit {checked["version"]} static analysis; ruleset {RULESET_VERSION}.',
                    f'Parent evidence ID: {evidence_id}. Parent SHA-256: {snapshot["parent_sha256"]}.',
                    f'Analyzed code SHA-256: {snapshot["code_sha256"]}. Python parser: {checked["python_version"]}.',
                    f'Analysis status: {checked["status"]}. Rules: {", ".join(RULE_LABELS)}.',
                    f'Source label: {snapshot["filename"] or "registered code"}; declared first line: {snapshot["source_start_line"]}; partial source: {snapshot["partial_source"]}.',
                    'Findings below are attributed analyzer flags, not verified vulnerabilities. No supplied code or tests were executed.']
                if checked['status'] == 'ok':
                    lines.append(f'Bandit reported {checked["finding_count"]} flags; {len(findings)} shown. No flags does not establish security or correctness.')
                    lines.extend(f'{finding["rule_id"]} at source line {finding["line"]} (evidence line {finding["evidence_line"]}): '
                                 f'{finding["label"]}; severity {finding["severity"]}, confidence {finding["confidence"]}.' for finding in findings)
                else:
                    lines.append('Checks did not complete; do not interpret an empty findings list as a clean result. Supply a complete parseable Python excerpt when applicable.')
                record = registry.register_source(
                    source_key='analysis:bandit:' + fingerprint, title='Bandit static checks for registered code',
                    text='\n'.join(lines), url='', kind='code', origin='analysis', evidence_type='static_analysis',
                    fetched_at=datetime.now(timezone.utc).isoformat(), filename=snapshot['filename'],
                    truncated=checked['truncated'] or snapshot['partial_source'],
                    version_scope=f'Bandit {checked["version"]}; {RULESET_VERSION}; parent {evidence_id}; fingerprint {fingerprint}. Static flags only.',
                )
                analysis = {'analyzer': 'Bandit', 'version': checked['version'], 'ruleset_version': RULESET_VERSION,
                    'rule_ids': list(RULE_LABELS), 'parent_evidence_id': evidence_id,
                    'parent_sha256': snapshot['parent_sha256'], 'code_sha256': snapshot['code_sha256'],
                    'fingerprint': fingerprint, 'filename': snapshot['filename'],
                    'source_start_line': snapshot['source_start_line'], 'partial_source': snapshot['partial_source'],
                    'status': checked['status'], 'finding_count': checked['finding_count'] if checked['status'] == 'ok' else None,
                    'findings': findings, 'truncated': checked['truncated']}
                result = {'status': 'ok' if checked['status'] == 'ok' else 'error',
                          'analysis': analysis, 'evidence': [record],
                          'notes': 'Bandit flags require contextual human review. Omitted code was not analyzed; no supplied program or tests were run.'}
                if checked['status'] != 'ok':
                    result.update(error_code=checked['status'], error='Static checks could not complete on this saved code excerpt.')
            except BudgetExceeded:
                raise
            except StaticAnalysisUnavailable as exc:
                result = {'status': 'error', 'error_code': exc.code, 'error': str(exc), 'evidence': []}
            event.update(status=result['status'], evidence_count=len(result['evidence']))
            if result['status'] == 'error':
                event.update(error_type='StaticAnalysisUnavailable', error_code=result['error_code'])
            return result
    return [run_static_checks]
