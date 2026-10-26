#!/usr/bin/env python3
"""Run, inspect, and review saved Deep Investigation cases.

Examples from the repository root:
    uv run python scripts/investigate_deep.py start --question "Why did my widget reset?"
    uv run python scripts/investigate_deep.py list
    uv run python scripts/investigate_deep.py show CASE_ID
    uv run python scripts/investigate_deep.py review CASE_ID --decision approve

Start, retry, and revision may call configured providers. Approval and rejection
only resume local review state. Publishing requires the separate publish-review command,
a prepared preview, and exact plan-ID confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from triage_desk.agent_budget import InvestigationBudget
from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.agent_tools import build_investigation_tools
from triage_desk.cases import CaseStore
from triage_desk.core import TriageDesk
from triage_desk.deep_investigation import AgentSettings, InvestigationService


class _CLIError(ValueError):
    """A controlled message containing no provider response or file contents."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (default: this script\'s repository).')
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start', help='Start a provider-backed investigation and pause for human review.')
    question = start.add_mutually_exclusive_group(required=True)
    question.add_argument('--question')
    question.add_argument('--question-file', type=Path, help='UTF-8 issue report, at most 4500 characters.')
    start.add_argument('--version', default='')
    start.add_argument('--code-file', type=Path, help='Optional UTF-8 reproduction or code, at most 16000 characters.')
    start.add_argument('--repository-context', action='store_true',
                       help='Allow bounded static reads of app.py and triage_desk Python files; selected content is sent to Fireworks.')
    start.add_argument('--review-memory', action='store_true',
                       help='Allow bounded recall of earlier approved/rejected case summaries and review notes; selected guidance is sent to Fireworks.')
    listing = commands.add_parser('list', help='List saved case summaries without provider calls.')
    listing.add_argument('--limit', type=int, choices=range(1, 1001), default=30, metavar='1..1000')
    show = commands.add_parser('show', help='Read a saved report, provenance, and audit events without provider calls.')
    show.add_argument('case_id')
    review = commands.add_parser('review', help='Approve, reject, or request a revision to a saved report.')
    review.add_argument('case_id')
    review.add_argument('--decision', choices=('approve', 'reject', 'revise'), required=True)
    review.add_argument('--notes', default='')
    retry = commands.add_parser('retry', help='Retry a failed investigation from its saved checkpoint.')
    retry.add_argument('case_id')
    audit = commands.add_parser('audit', help='Review a saved draft against its evidence without repeating research.')
    audit.add_argument('case_id')
    prepare = commands.add_parser('prepare-publication', help='Prepare a local preview for an approved public-source PR report; no network calls.')
    prepare.add_argument('case_id')
    prepare.add_argument('--pr', type=int, required=True)
    publication = commands.add_parser('show-publication', help='Read a saved immutable publication preview; no network calls.')
    publication.add_argument('plan_id')
    publish = commands.add_parser('publish-review', help='Explicitly publish a prepared summary COMMENT after checking the current PR head and base.')
    publish.add_argument('plan_id')
    publish.add_argument('--confirm', required=True, help='Exact plan ID from the preview. Requires GITHUB_TOKEN with permission to write PR reviews.')
    commands.add_parser('probe', help='Explicitly test Fireworks tool calling and Keenable access using small live requests.')
    return parser


def _read_text(path: Path, maximum: int, label: str) -> str:
    try:
        with path.open('rb') as stream:
            raw = stream.read(maximum * 4 + 1)
        if len(raw) > maximum * 4:
            raise _CLIError(f'{label} exceeds its {maximum}-character limit.')
        text = raw.decode('utf-8')
    except (OSError, UnicodeError) as exc:
        raise _CLIError(f'{label} must be a readable UTF-8 text file.') from exc
    if len(text) > maximum:
        raise _CLIError(f'{label} exceeds its {maximum}-character limit.')
    return text


def _summary(case: dict) -> dict:
    report = case.get('report') or {}
    return {'case_id': case['case_id'], 'status': case['status'],
            'error': case.get('error', ''), 'findings_count': len(report.get('findings', []))}


def _show(store: CaseStore, case_id: str) -> dict:
    case = store.get_case(case_id)
    # Literal source bodies and user input remain in local storage. The report
    # contains only its quoted support; URLs originate in registered evidence.
    source_fields = ('evidence_id', 'title', 'url', 'kind', 'origin', 'fetched_at',
                     'snapshot_at', 'version_scope', 'commit_sha', 'sha256',
                     'evidence_type', 'published_at', 'acquired_at', 'base_sha', 'filename', 'truncated')
    event_fields = ('event_id', 'phase', 'name', 'status', 'started_at', 'latency_ms',
                    'error_type', 'error_code', 'evidence_count', 'specialist', 'usage', 'usage_reported', 'findings', 'revision')
    return {**_summary(case), 'repository_context': case.get('repository_context', False), 'memory_context': case.get('memory_context', False), 'report': case.get('report'),
            'provenance': [{key: source.get(key, '') for key in source_fields}
                           for source in store.evidence(case_id)],
            'events': [{key: event[key] for key in event_fields if key in event}
                       for event in store.events(case_id)]}


def _probe_model(settings: AgentSettings):
    from langchain_fireworks import ChatFireworks
    return ChatFireworks(model=settings.model, api_key=settings.api_key, temperature=0,
                         max_tokens=512, max_retries=0, timeout=15)


def _probe(settings: AgentSettings) -> dict:
    """Run only on explicit invocation; discard source and model output text."""
    @tool
    def readiness_probe() -> str:
        """Return a fixed marker to verify that model tool calling works."""
        return 'engineering_issue_desk_probe_ok'

    fireworks = {'status': 'error', 'tool_call_roundtrip': False}
    keenable = {'status': 'error', 'evidence_count': 0}
    with TemporaryDirectory(prefix='issue-desk-probe-') as temporary:
        store = CaseStore(Path(temporary) / 'probe.sqlite')
        case = store.create_case('Explicit provider readiness probe')
        registry = EvidenceRegistry(store, case['case_id'])
        budget = InvestigationBudget(store, case['case_id'], max_model_calls=2,
                                     max_tool_calls=2, max_tokens=1024, max_seconds=60)
        missing = settings.readiness()['missing']
        if missing:
            fireworks['missing'] = missing
        else:
            try:
                model = _probe_model(settings)
                message = HumanMessage(content='Call readiness_probe exactly once, then briefly confirm receiving its result.')
                with budget.call('model', 'fireworks_probe'):
                    response = model.bind_tools([readiness_probe], tool_choice='readiness_probe').invoke([message])
                calls = response.tool_calls
                if (len(calls) != 1 or calls[0].get('name') != 'readiness_probe'
                        or calls[0].get('args') != {} or not calls[0].get('id')):
                    raise _CLIError('Model did not return the expected tool call.')
                with budget.tool_call('readiness_probe'):
                    marker = readiness_probe.invoke({})
                with budget.call('model', 'fireworks_probe_result'):
                    final = model.invoke([message, response, ToolMessage(
                        content=marker, tool_call_id=calls[0]['id'], name='readiness_probe')])
                if not isinstance(final.content, str) or not final.content.strip():
                    raise _CLIError('Model did not complete the tool-result round trip.')
                fireworks.update(status='ok', tool_call_roundtrip=True)
            except Exception:
                fireworks['error'] = 'Tool-call round trip failed; check model support and provider configuration.'

        if not settings.keenable_api_key:
            keenable['missing'] = ['KEENABLE_API_KEY']
        else:
            try:
                search = next(tool for tool in build_investigation_tools(
                    None, registry, budget, keenable_api_key=settings.keenable_api_key)
                    if tool.name == 'search_web')
                result = search.invoke({'query': 'official Streamlit Session State documentation'})
                keenable.update(status=result['status'], evidence_count=len(result.get('evidence', [])))
            except Exception:
                keenable['error'] = 'Search probe failed; check provider configuration.'
    return {'status': 'ok' if fireworks['status'] == keenable['status'] == 'ok' else 'error',
            'fireworks': fireworks, 'keenable': keenable}


def dispatch(args: argparse.Namespace) -> dict | list:
    root = args.root.expanduser().resolve()
    if args.command in {'list', 'show'}:
        store = CaseStore(root / '.cache' / 'agent' / 'cases.sqlite')
        if args.command == 'list':
            return [_summary(case) for case in store.list_cases(args.limit)]
        return _show(store, args.case_id)

    if args.command in {'prepare-publication', 'show-publication', 'publish-review'}:
        from triage_desk.review_publication import ReviewPublisher
        if args.command != 'show-publication':
            load_dotenv(root / '.env', override=False)
        store = CaseStore(root / '.cache' / 'agent' / 'cases.sqlite')
        publisher = ReviewPublisher(store, token=AgentSettings.from_env().github_token if args.command == 'publish-review' else '',
            forbidden_values=tuple(os.getenv(name, '') for name in (
                'FIREWORKS_API_KEY', 'KEENABLE_API_KEY', 'TRIAGE_API_KEY', 'NEBIUS_API_KEY', 'LANGSMITH_API_KEY')))
        try:
            if args.command == 'prepare-publication':
                return publisher.prepare(args.case_id, args.pr)
            if args.command == 'show-publication':
                return publisher.get(args.plan_id)
            return publisher.publish(args.plan_id, confirmation=args.confirm)
        except ValueError as exc:
            raise _CLIError(str(exc)) from None

    if args.command == 'probe':
        load_dotenv(root / '.env', override=False)
        return _probe(AgentSettings.from_env())

    needs_model = args.command in {'start', 'retry', 'audit'} or args.command == 'review' and args.decision == 'revise'
    if needs_model:
        load_dotenv(root / '.env', override=False)
    settings = AgentSettings.from_env() if needs_model else AgentSettings()
    if needs_model and not settings.readiness()['ready']:
        raise _CLIError('Configure FIREWORKS_API_KEY and FIREWORKS_MODEL before starting, retrying, or revising a case.')

    if args.command == 'review' and args.decision == 'revise' and not args.notes.strip():
        raise _CLIError('Add --notes describing the requested revision.')
    question, code = '', ''
    if args.command == 'start':
        question = (_read_text(args.question_file, 4500, 'Question file')
                    if args.question_file else args.question)
        if not question.strip() or len(question) > 4500:
            raise _CLIError('Issue report must contain 1 to 4500 characters.')
        if args.code_file:
            code = _read_text(args.code_file, 16000, 'Code file')
            # Retain the supplied path as part of the evidence so static findings
            # can identify it. Never read any additional file automatically.
            code = f'# Supplied file: {args.code_file.name}\n{code}'
            if len(code) > 16000:
                raise _CLIError('Code file plus its filename must fit within 16000 characters.')

    desk = TriageDesk(data_dir=root / 'data', cache_dir=root / '.cache' / 'retrieval') if needs_model else None
    service = InvestigationService(root, desk, settings)
    if args.command == 'start':
        options = {'repository_context': True} if args.repository_context else {}
        if args.review_memory:
            options['memory_context'] = True
        case = service.start(question, args.version, code, **options)
    elif args.command == 'retry':
        case = service.retry(args.case_id)
    elif args.command == 'audit':
        case = service.audit(args.case_id)
    else:
        case = service.resume(args.case_id, args.decision, args.notes)
    return _summary(case)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except _CLIError as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}), file=sys.stderr)
        return 1
    except KeyError:
        print(json.dumps({'status': 'error', 'error': 'No saved case with that ID was found.'}), file=sys.stderr)
        return 1
    except Exception:
        # Raw provider exceptions, file paths, and tracebacks can reveal secrets.
        print(json.dumps({'status': 'error', 'error': 'The command could not finish. Check configuration, case state, and the saved audit events.'}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    failed = isinstance(result, dict) and (
        result.get('status') in {'failed', 'uncertain'} and args.command not in {'show', 'show-publication'}
        or args.command == 'probe' and result.get('status') == 'error')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
