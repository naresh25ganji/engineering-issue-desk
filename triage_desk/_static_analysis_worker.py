"""Trusted isolated Bandit worker; input source is parsed, never executed.

Invoked only by static_analysis.py with an isolated interpreter and temporary
working directory. No project configuration or source plugin is loaded.
"""
from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
import sys


RULE_IDS = ('B102', 'B301', 'B307', 'B324', 'B506', 'B602', 'B605', 'B608')
MAX_INPUT_CHARS = 16000
MAX_AST_NODES = 6000
MAX_FINDINGS = 50
MAX_OUTPUT_BYTES = 65536


def main() -> int:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        raw = sys.stdin.buffer.read(MAX_INPUT_CHARS * 6 + 1024)
        payload = json.loads(raw)
        code = payload['code']
        if not isinstance(code, str) or not code.strip() or len(code) > MAX_INPUT_CHARS:
            return 2
        result = {'version': importlib.metadata.version('bandit'),
                  'python_version': '.'.join(map(str, sys.version_info[:3])),
                  'status': 'ok', 'findings': [], 'finding_count': 0, 'truncated': False}
        try:
            tree = ast.parse(code, filename='registered_evidence.py')
            if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
                result['status'] = 'complexity_limit'
        except SyntaxError as exc:
            result.update(status='syntax_error', error_line=exc.lineno or 1)
        except (RecursionError, MemoryError):
            result['status'] = 'complexity_limit'
        if result['status'] == 'ok':
            from bandit.core.config import BanditConfig
            from bandit.core.manager import BanditManager
            # This filename and profile are trusted constants, never model args.
            target = Path('registered_evidence.py')
            target.write_text(code, encoding='utf-8')
            manager = BanditManager(BanditConfig(), 'file', quiet=True,
                                    profile={'include': set(RULE_IDS), 'exclude': set()},
                                    ignore_nosec=True)
            manager.discover_files([str(target)])
            manager.run_tests()
            if manager.skipped:
                result['status'] = 'analysis_error'
            else:
                issues = sorted(manager.get_issue_list(), key=lambda issue: (issue.lineno, issue.test_id))
                result['finding_count'] = len(issues)
                result['truncated'] = len(issues) > MAX_FINDINGS
                for issue in issues[:MAX_FINDINGS]:
                    # Bandit issue_text/code can contain source values. Return
                    # identifiers and bounded numeric positions only.
                    if issue.test_id not in RULE_IDS:
                        return 2
                    result['findings'].append({'rule_id': issue.test_id, 'line': issue.lineno,
                                               'severity': issue.severity, 'confidence': issue.confidence})
        output = json.dumps(result, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(output) > MAX_OUTPUT_BYTES:
            return 2
        sys.stdout.buffer.write(output)
        return 0
    except Exception:
        # Parent emits a fixed failure label; never expose paths, code or errors.
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
