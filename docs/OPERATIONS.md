# Engineering Issue Desk operations

Version date: October 25, 2026.

This runbook operates one local personal workspace. Commands are instructions for an operator; writing this guide does not run provider calls, stop processes, restore files, or publish anything. Read the [architecture](ARCHITECTURE.md) for data flow and the [design record](DESIGN.md) for the authority and recovery rules.

## Local setup

Use Python 3.12 or 3.13, `uv`, and macOS/Linux file locking. From the repository root:

```bash
uv sync --frozen
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8503
```

Open `http://127.0.0.1:8503`. Restart the server after Python module changes if a browser refresh still shows old controls. The Deep Investigation tab can show readiness and saved cases without starting an agent. The service is created per render/session, rather than being held in a shared Streamlit resource cache. The original retrieval modes remain available; first use of local embedding models may require downloads/index preparation, so keyword mode is the simplest offline baseline.

The eighth tab, **Project guide**, displays and downloads the two authored SVG diagrams and fixed architecture, design, operations, validation, and submission documents. Opening it reads only trusted repository assets and makes no provider call.

Configure server-side values in the ignored local `.env`, then restart Streamlit. Do not paste keys into reports, code samples, screenshots, or committed files.

| Variable | Purpose | Required for |
| --- | --- | --- |
| `FIREWORKS_API_KEY` | Fireworks account access | Start, retry, revision, or saved evidence audit |
| `FIREWORKS_MODEL` | A model ID with tool calling | Same model-backed operations |
| `KEENABLE_API_KEY` | Current public search/page fetch | Keenable tools; optional for the rest of the agent |
| `GITHUB_TOKEN` | Optional public API allowance; a write-capable token is needed only for separately confirmed publication | Reads do not publish; preparing a preview needs no token |
| `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` | Optional tracing | Trace collection; may retain model/tool content |

The live baseline used `accounts/fireworks/models/gpt-oss-120b`. Account entitlement and provider availability must be checked explicitly rather than inferred from this recorded model ID. Existing Nebius variables affect the separate original retrieval/generated-answer path.

GitHub reads pin `X-GitHub-Api-Version: 2026-03-10`, a supported version listed in the [official API version documentation](https://docs.github.com/en/rest/about-the-rest-api/api-versions). Live issue/PR reads are checked separately from model-generated investigations.

## Connectivity check

This command intentionally spends a small amount of configured provider credit:

```bash
uv run python scripts/investigate_deep.py probe
```

The probe attempts at most two small Fireworks calls and one Keenable search. It reports booleans/status/counts rather than raw keys or provider text. A missing optional Keenable key yields an incomplete probe even though snapshot/GitHub investigation may remain usable. A successful probe does not establish report quality or sufficient credit for a long run.

## Start and inspect a case

```bash
uv run python scripts/investigate_deep.py start --question "Investigate documented Streamlit rerun behavior and identify missing reproduction details."
uv run python scripts/investigate_deep.py list
uv run python scripts/investigate_deep.py show CASE_ID
```

Replace `CASE_ID` with an actual returned UUID. `list` and `show` use saved local data without provider calls. Add `--version` when known, or `--code-file path/to/sample.py` to deliberately supply one UTF-8 file. The client does not read neighboring files implicitly in the baseline mode. Issue text is limited to 4,500 characters, code plus its supplied filename to 16,000, version to 120, and review notes to 2,000.

Starting sends the report, supplied code, and selected evidence to Fireworks. Keenable receives search queries/page URLs only if those tools run. A model may select a different tool order on another run. Do not narrate an expected tool sequence as if it was observed.

### Include related application code

Use `start --repository-context`, or select the default-off **Include related application code** checkbox in the UI, to let the agent locate related helpers within `app.py` and `triage_desk/**/*.py`. Selected code can be sent to Fireworks. The choice is immutable for that case and inherited by retry/revision; start a new case to change it.

Inventory/search are discovery leads. Only an explicit excerpt read becomes citation evidence. Search is literal and bounded across file prefixes; missing matches do not prove absent behavior. Reads preserve at most 160 lines and 8,000 code characters from a file of at most 256,000 bytes. Each saved excerpt records path, line range, hashes, timestamp, and static-inspection provenance. A changed working tree can produce a new snapshot on a later read. The [architecture bounds](ARCHITECTURE.md#opt-in-repository-context) describe the complete limits and path/credential guards; no file is executed or modified.

## Optional guidance and static checks

Select **Use earlier human reviews**, or add `--review-memory` to CLI `start`, only when earlier approved/rejected prose from this local workspace may be sent to Fireworks. The choice defaults off and is immutable for that case. Retry/revision inherit it. Start a new case to change consent. Pending/current cases, raw source/code, questions, and support quotes are excluded; bounded literal lookup returns hints, not factual evidence or prior approval authority. This is not Mem0 or semantic retrieval.

The agent can list/load three maintained review playbooks on demand. They are versioned guidance, not native `SkillsMiddleware` or citable sources. It can also call `run_static_checks` on a registered code evidence ID. Fixed Bandit rules parse saved Python in a trusted isolated worker; supplied code and tests are never executed. Inspect the analysis status, parent hash, partial-source flag, and original/evidence line mapping. A syntax/complexity/timeout error is not a clean scan, and no flags does not establish security.

## Human review

Inspect the cited text, source type/date/version, retained findings, proposed steps, missing details, and independent AI verdicts before deciding:

```bash
uv run python scripts/investigate_deep.py review CASE_ID --decision approve --notes "Reviewed the retained claims and their exact supporting passages."
uv run python scripts/investigate_deep.py review CASE_ID --decision reject --notes "The selected evidence does not establish the proposed cause."
uv run python scripts/investigate_deep.py review CASE_ID --decision revise --notes "Check the configuration helper before claiming the validation is missing."
```

Choose one command appropriate to the case. Approval/rejection require no provider configuration and do not publish externally. A revision requires model configuration and explicit notes; it receives the original request and prior report. Final cases cannot be reviewed again as another decision.

Older pending drafts may lack the independent evidence audit. Run the explicit audit, or request a revision, before approving such a draft:

```bash
uv run python scripts/investigate_deep.py audit CASE_ID
```

The audit spends a model call against the same case allowance and does not repeat research. An AI audit can retain an unsupported claim; completion is a prerequisite for approval, not proof of correctness.

## Separate approved PR summary preview

Only an approved evidence-reviewed case with a registered public PR description and matching commit-pinned patch can prepare a preview for `streamlit/streamlit`. The published findings must use allowed public source evidence. Reports supported by local code or review history cannot be published through this path.

```bash
uv run python scripts/investigate_deep.py prepare-publication CASE_ID --pr PR_NUMBER
uv run python scripts/investigate_deep.py show-publication PLAN_ID
```

Preparation is offline and needs no write token. Inspect the exact body, target, observed base/head, and saved plan ID. The UI exposes the same preview only for approved cases. Publication is a separate external action requiring exact plan-ID confirmation and a deliberately configured write-capable `GITHUB_TOKEN`; investigation or local approval alone cannot trigger it.

```bash
uv run python scripts/investigate_deep.py publish-review PLAN_ID --confirm PLAN_ID
```

That command can create a public summary review `COMMENT`. It is an operator instruction, not an action performed by this runbook. No live POST or write token was used during submission verification. Fresh PR checks stop if the target is not an open public fixed-repository PR or its base/head changed. The plan also binds the original approved report and evidence hashes. There are no inline comments or PR approval/request-changes actions.

An atomically claimed plan cannot be replayed. A `failed` preflight requires inspecting the saved safe error before preparing anything new. An `uncertain` result, or a plan left `publishing` after interruption, must be checked on GitHub manually; never retry a potentially completed POST by editing the database or regenerating plans. No automatic remote reconciliation is implemented.

## Failure and recovery

| Observation | Check | Recovery |
| --- | --- | --- |
| Start disabled or CLI configuration error | Required variable names/model support; do not print key values | Correct local configuration, restart app, run explicit probe |
| Keenable unavailable/rate limited | Sanitized tool result and saved events | Use snapshot/GitHub evidence or explicitly retry later; do not invent current context |
| Provider timeout, truncated output, malformed tool arguments | Safe execution error, model result metadata, recorded usage | Retry the failed case after correcting the external condition |
| Citation/schema failure after research | Saved evidence and invalid-report repair events | Explicit retry can start a fresh repair thread, at most twice, with unchanged case limits |
| Independent evidence review fails | Failed case and unapproved draft | Retry resumes the pending review step; it must not turn the draft into an approved report |
| Budget/context limit | Dispatch count, reported usage, active invocation time, and context guard | Existing evidence remains inspectable; repeated retry cannot restore exhausted calls/tokens |
| Another session owns the case | Nonblocking process lock/atomic status claim | Wait for that execution to finish; do not edit the database or remove its lock file |
| Process crashed while `running` | Restart service; startup recovery checks the OS lock | Abandoned case becomes `failed`, then explicit retry resumes its checkpoint |

```bash
uv run python scripts/investigate_deep.py retry CASE_ID
```

Retry is for failed cases, not approved/rejected ones. Completed reads are not automatically replayed when the pending checkpoint can resume. A report repair is a new inner conversation, not a new case or budget. Default limits are 18 model calls, 28 tools, two revisions, two report repairs, and a 64,000 reported-token threshold. A 180-second active invocation deadline and an 80,000-character message guard also apply. Time/token checks occur before dispatch; in-flight requests can finish beyond them. Missing usage is unknown.

### Evidence-review diagnostics

The saved error event exposes a finite `error_code`, not raw provider text. All failures leave the report unapproved; retry remains explicit and consumes the same case allowance.

| Code | Meaning and operator response |
| --- | --- |
| `input_invalid` | Draft lacks valid reviewable evidence; inspect its sources and schema |
| `context_limit` | Selected review context exceeds its bound; request a narrower investigation |
| `provider_unavailable` | Review call could not finish; check provider readiness before retry |
| `response_invalid`, `missing_tool`, `unexpected_tool`, `multiple_tools` | Provider did not return the single required verdict tool response |
| `invalid_schema`, `incomplete_coverage` | Verdict types/indices do not cover every original item exactly once |
| `invalid_reason` | Explanation is invalid or contains prohibited markup/links |
| `output_limit` | Provider stopped before complete verdict output |

Harmless paired single inline backticks around unchanged code text are normalized; unsafe formatting is still rejected. A safe failure code diagnoses format/control behavior, not the quality of the findings.

## Private state and backup

Private case state is in `.cache/agent/`. `cases.sqlite` contains issue/code text, source snapshots, events, and reports; `checkpoints.sqlite` contains conversation and workflow state. Neither belongs in Git or the submission ZIP. These files are not an encrypted secrets vault. An optional trace service is another content-retention destination.

Back up both databases together **after stopping Streamlit and every case-execution CLI process**. This matters because the two stores do not share an atomic cross-file transaction. A stopped workspace also avoids concurrent graph writes while snapshots are taken. The following example creates a new timestamped backup; it does not modify the live database contents:

```bash
uv run python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

source = Path('.cache/agent')
backup = Path('.cache/backups/engineering-issue-desk') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup.mkdir(parents=True, exist_ok=False)
for name in ('cases.sqlite', 'checkpoints.sqlite'):
    path = source / name
    if not path.is_file():
        raise SystemExit(f'Missing required database: {name}')
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True) as live:
        with sqlite3.connect(backup / name) as saved:
            live.backup(saved)
            result = saved.execute('PRAGMA integrity_check').fetchone()[0]
            if result != 'ok':
                raise SystemExit(f'Backup integrity check failed: {name}')
print(backup)
PY
```

Keep the backup private and outside the source submission. The example uses ignored `.cache/backups/`; preserve that exclusion if moving the destination. Retain the repository version/lockfile and the `data/manifest.json` that correspond to the saved checkpoints. A backup inside `.cache` is excluded from Git but remains vulnerable to local cache deletion; copy the paired snapshots to private durable storage as appropriate.

### Restore rehearsal

1. Stop all execution and preserve the current databases as a separate backup.
2. Use a separate working copy of the matching repository version. Restore the paired database snapshots into its `.cache/agent/`; do not merge tables or copy only one store.
3. Run SQLite `PRAGMA integrity_check` against both restored files. Open `list` and `show` without credentials and compare case IDs, report/source counts, and event history with the backup inventory.
4. Start the service so lock-aware startup recovery can identify abandoned `running` cases. Retained lock filenames are not proof that a process is alive; the advisory OS lock is the authority.
5. Verify a saved `awaiting_review` case remains pending. Test approval/rejection or a provider-backed retry only on an intentionally selected case, because those actions change its local state or spend credit.

This guide does not perform a restore. Never hand-edit statuses to bypass a missing checkpoint, audit, budget, or review guard. Migration compatibility must be tested before replacing a newer workspace with older state.

## Validation matrix

Run the full offline suite after code changes:

```bash
uv run pytest -q
uv run python scripts/fetch_corpus.py --check
```

| Boundary or feature | Automated evidence | Additional verification |
| --- | --- | --- |
| Case identity, immutable evidence, lock ownership | [test_cases.py](../tests/test_cases.py), [test_agent_evidence.py](../tests/test_agent_evidence.py) | Reopen persisted sources and compare hashes |
| Tool allowlists, provider errors, no credential arguments | [test_agent_tools.py](../tests/test_agent_tools.py) | Explicit Fireworks/Keenable probe for connectivity |
| Real Deep Agents loop and specialist evidence sharing | [test_deep_investigation.py](../tests/test_deep_investigation.py) | Observe actual tool/delegation events in a live run |
| Crash/restart, repair thread, review replay, prior-report revision | [test_deep_investigation.py](../tests/test_deep_investigation.py) | Restore rehearsal in an isolated workspace |
| Shared model/tool/token budgets and output metadata | [test_agent_budget.py](../tests/test_agent_budget.py) | Compare dispatch/result pairs; missing usage stays unknown |
| Evidence verdict coverage, filtering, fail-closed behavior | [test_agent_review.py](../tests/test_agent_review.py) | Human assessment of causal support and omitted context |
| User forms, saved cases, safe text/links, review actions | [test_agent_ui.py](../tests/test_agent_ui.py), [test_agent_cli.py](../tests/test_agent_cli.py) | Browser check of progress, source display, and final status |
| Reproducible evaluation and displayed metrics | [test_agent_evaluation.py](../tests/test_agent_evaluation.py), [test_agent_evaluation_ui.py](../tests/test_agent_evaluation_ui.py) | Confirm denominator includes failed scheduled cases |
| Source packaging excludes private state | [test_submission.py](../tests/test_submission.py) | Inspect archive manifest and contents before sharing |
| Opt-in paths, race/credential/bounds guards and immutable snapshots | [test_repository_context.py](../tests/test_repository_context.py), [test_deep_investigation.py](../tests/test_deep_investigation.py), [test_agent_evaluation_context.py](../tests/test_agent_evaluation_context.py) | Inspect actual read events and saved source hashes; bounded search may omit later code |
| Finite review errors and inline-code presentation normalization | [test_agent_review.py](../tests/test_agent_review.py) | Inspect safe error code; keep semantic quality separate |
| Static Project guide assets and no provider work on opening | [test_project_guide.py](../tests/test_project_guide.py) | Open/download both diagrams and inspect readable layout |
| Versioned playbooks and opt-in literal review guidance | [test_review_extensions.py](../tests/test_review_extensions.py) | Distinguish history hints from new factual evidence |
| Real static analyzer and source/line/bounds preservation | [test_static_analysis.py](../tests/test_static_analysis.py) | Review attributed flags and omitted context; do not claim tests ran |
| Approved public-summary preview, stale target and replay guards | [test_review_publication.py](../tests/test_review_publication.py) | Mocked HTTP only; live POST and write credentials unvalidated |

See [VALIDATION.md](VALIDATION.md) for the final test count, live extension results, browser checks, and limitations recorded for this revision.

Final software checks passed 603 tests and the source environment check found 117 packages current. Use a repository checkout with `uv`; default `uv build` discovery is not configured, so this runbook does not promise a working wheel or `pip install` distribution. The latest [verified report](../evaluation/week3-verified-results/report.json), separate [extension exercise](../evaluation/week3-verified-results/extension-exercise.json), and [Codex-assisted inspection](../evaluation/week3-verified-results/engineering-review.md) are distinct records. They do not establish human approval or accuracy.

The frozen [Week 3 report](../evaluation/week3-results/report.json) records three static code samples: two saved handoffs, one failed evidence-review response, nine model calls, three saved-evidence reads, and 44,382 reported tokens. Literal quotes passed for all three drafts, including the failed case. No human quality score was collected. It did not exercise repository context, delegation, or Keenable. See the [retained false positives](../evaluation/week3-results/engineering-review.md); exact references and an AI review are not a correctness guarantee.

The separate [repository-context report](../evaluation/week3-context-results/report.json) records three of three saved handoffs, zero failures, 12 model calls, six tools, and 70,046 reported tokens with complete returned usage. Its tool calls were two saved-evidence reads, one inventory, and three repository reads; one overlong line range failed safely before two valid reads. This run did not invoke repository search or inspect `_configuration`, and did not use Keenable or delegation. The October 23 suite passed 473 tests; these operational observations are not a human accuracy score.

New live evaluation spends provider credit and must use a new or empty output directory:

```bash
uv run python -m evaluation.agent_workflow --live --output-dir evaluation/week3-results-new-run
```

Omit `--live` to validate fixtures and export an unscored inventory without loading provider credentials or calling providers. Never overwrite a frozen result to make a new implementation appear to have produced it.

## Deployment limits

Bind the demonstration server to loopback. This build has no multi-user login/authorization layer, tenant separation, distributed execution lease, container sandbox for reproductions, or encrypted case storage. `fcntl` case locks require suitable local filesystem semantics and do not make SQLite a multi-host queue. It does not integrate Mem0, execute supplied programs/tests, publish autonomously, or submit course forms. The separate confirmed GitHub summary path exists but has not been live-validated. Those deployment and execution capabilities require separate design, authority, tests, and operational procedures.
