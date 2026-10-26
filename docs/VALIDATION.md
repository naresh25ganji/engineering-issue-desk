# Validation record

Verified October 25, 2026, in the local Python 3.12 workspace. Checks establish the behavior observed on these cases and controlled tests, not human accuracy, production readiness, or general reliability.

## Final software checks

| Check | Observed result | Evidence and boundary |
| --- | --- | --- |
| Full regression suite | **603 passed**, 13.77 seconds | Actual Deep Agents harness, SQLite crashes/recovery, review/revision/repair, budgets, tools, UI/CLI, evaluation, and new extensions |
| Locked source environment | `uv sync --frozen --check` found 117 packages current | Source-checkout execution supported; default `uv build` package discovery is not configured, so no wheel/pip-install claim |
| Corpus integrity | 65 records; SHA-256 unchanged | `3caa3fbd41e436469b6bdc63f5435eb9ec8f5eecfa49064e02a77fedc618d503`; October 17 historical snapshot, not all current material |
| Playbooks and literal review guidance | Fixed catalog/version/hash, path/credential controls, budgets, immutable consent and bounded projections | [Extension tests](../tests/test_review_extensions.py), graph/UI/CLI tests; positive rejected-history lookup exercised with a controlled model |
| Real static analyzer | Actual Bandit worker, eight fixed rules, registered source only, bounds, parse failure and line/hash preservation | [Static tests](../tests/test_static_analysis.py); no supplied program or tests executed |
| Confirmed summary publication | 55 controlled HTTP/state tests, including stale and mixed revisions and legacy code-page classification bypasses | [Publication tests](../tests/test_review_publication.py); no live POST, write token, inline review, or PR approval tested |
| Report comparison guard | Plain comparison syntax no longer rejected as HTML; regression tests retain markup rejection | A fresh live run exposed the false rejection; saved review-only recovery is recorded separately |
| Browser | Default-off history/repository controls, saved real case, sources, review and downloads inspected | Real local browser; case remained pending, no approval or external post |
| Final Word artifacts | Nine writeup pages and three demo-guide pages; title sanitizer checks passed and all 12 final PNGs visually inspected | Six writeup tables and one architecture image; demo targets 4:45 and does not claim a recorded video |
| Final native Google Doc | Private final import passed metadata/content readback; all nine exported native pages visually inspected against the local Word render | Page/section breaks, six tables, one image, headings and metrics retained; title wraps to two lines and body wraps differ without clipping or overlap |
| Final diagrams and guide | Updated component/sequence SVGs rendered with bundled Sharp and visually inspected; current guide/readiness selector loaded in browser | Separate publication authority and guidance/evidence distinctions are shown; no provider call from opening authored docs |
| Source/docs hygiene | Corpus check and `git diff --check` passed; local Markdown links resolved | Final source ZIP is rebuilt after artifacts freeze; private state and credentials remain excluded |

## Latest frozen live run

The [verified three-case report](../evaluation/week3-verified-results/report.json) records **3/3 saved human-review handoffs**, zero failures, 12 model calls (nine investigator, three reviewer), six tools, and **70,003 reported tokens** with complete returned usage. Actual tools were two `read_case_evidence`, one `list_repository_files`, and three `read_repository_file`; one overlong read failed safely before smaller reads succeeded.

Only W3-02 retained factual findings: two static freshness observations, with literal quote integrity passed. W3-01 and W3-03 retained no factual findings; their citation status is **not applicable**, not a pass. The AI review removed four draft findings and three proposed steps. Empty output can reflect overconservative filtering and is not an accuracy achievement. The third sample established no defect. The [inspection notes](../evaluation/week3-verified-results/engineering-review.md) are Codex-assisted observations, not user approval or human quality scores. No evaluation case was automatically approved.

No repository search, configuration-helper read, Keenable, delegation, playbook, history, or Bandit invocation occurred in this three-case run. Availability and controlled tests must not be narrated as observed live use. Raw report SHA-256: `4582cab5e70b1cb0f6bb470f581c0c5b82b776246a55d69e5cf56ca74a01b1d4`.

## Separate extension and recovery observations

The [additional extension exercise](../evaluation/week3-verified-results/extension-exercise.json) over the actual 30-line freshness sample reached `awaiting_review` with six model calls, four tools, and **34,120 reported tokens**. Its observed calls were evidence list, evidence read, `load_review_playbook`, and real `run_static_checks`, once each. It registered one derived analysis source and retained one finding plus two test proposals. Memory was opted in but `search_review_history` was not invoked. Positive history recall is covered offline, not by this live result. This exercise is outside the three-case denominator and token total.

The earlier [October 25 run](../evaluation/week3-final-results/report.json) remains frozen at 2/3 handoffs, 11 model calls, six tools, and 64,580 reported tokens. A plain comparison expression was incorrectly caught as HTML. After the guard fix, [review-only recovery](../evaluation/week3-final-results/comparison-guard-recovery.json) used one reviewer call, zero tools, and 3,144 reported tokens; research was not repeated. The subsequent verified run is a new observation rather than an overwrite.

The [fresh provider probe](../evaluation/week3-final-results/provider-probe.json) verified a Fireworks tool round trip and five Keenable results. It is separate from report evaluation, extension use, and billing. Optional LangSmith configuration is not a live tracing validation. GitHub publication remains mocked only.

## Historical October 23 baseline

The following checks and run summaries preserve the prior implementation's observations. Its 473-test count and three-case results are historical, not the final count.

## Checks completed

| Check | Observed result | Evidence |
| --- | --- | --- |
| Full offline regression suite | **473 passed**, 10.66 seconds in the final run | `uv run pytest -q`; tests cover the actual Deep Agents factory with controlled tool-calling models, SQLite state, specialist delegation, process crashes, revisions, repair, tools, UI, and evaluation |
| Corpus integrity | All 65 source records passed; original corpus hash preserved | `uv run python scripts/fetch_corpus.py --check`; SHA-256 `3caa3fbd41e436469b6bdc63f5435eb9ec8f5eecfa49064e02a77fedc618d503` |
| Repository boundaries | 37 focused tests passed, including traversal, symlinks/races, hardlinks, nonregular files, credentials, concurrent edits, bounded reads/search, immutable evidence, and late-file discovery | [Repository tests](../tests/test_repository_context.py) |
| Opt-in and migration | Older cases default off; only a strict Boolean opt-in enables tools; stored choice survives restart and cannot be changed through report updates | [Case tests](../tests/test_cases.py), [graph integration](../tests/test_deep_investigation.py), [evaluation context tests](../tests/test_agent_evaluation_context.py) |
| Resumed repository read | Actual controlled harness read a source, failed on the next model call, and resumed without repeating or double-counting the read | [Graph integration](../tests/test_deep_investigation.py) |
| Reviewer diagnostics | Finite safe codes, static messages, strict verdict coverage/schema, safe inline-code normalization, and failure without approval | [Reviewer tests](../tests/test_agent_review.py), [UI checks](../tests/test_agent_ui.py) |
| Current provider probe | Fireworks tool-call round trip succeeded; Keenable returned five results | [Probe record](../evaluation/week3-context-results/provider-probe.json) |
| Direct domain reads | Snapshot search returned five passages; Keenable fetched one page; GitHub issue reader returned six records; PR reader returned context and one commit-pinned patch | [Direct tool record](../evaluation/week3-context-results/domain-tool-smoke.json); no model calls in this check |
| Fresh live code evaluation | Three scheduled cases reached saved human review; no case approved | [Readable report](../evaluation/week3-context-results/report.html), [raw JSON](../evaluation/week3-context-results/report.json) |
| Actual failed-case recovery | Original failed W3-02 resumed with one reviewer call, zero tools, zero investigator calls; saved pending review | [Recovery record](../evaluation/week3-context-results/checkpoint-recovery.json) |
| Browser | Eight views loaded; default-off repository checkbox and saved-case enabled flag were visible; sources/review controls survived server restart; Project guide and live evaluation displays loaded | Actual Chrome checks at `http://127.0.0.1:8503`; no browser approval or publishing performed |
| Diagrams | Both detailed SVGs rendered with bundled Sharp and were visually inspected; labels/arrows remain readable | [System architecture](week3-system-architecture.svg), [investigation sequence](week3-investigation-sequence.svg) |
| Documentation | New local Markdown link targets and SVG structure checked; Project guide displays and downloads fixed authored assets | [Guide tests](../tests/test_project_guide.py), architecture/design/operations/submission docs |
| Source handoff | Archive rebuilt with a per-file SHA-256 manifest; manifest/content and exclusion of credentials, caches, backups, checkpoints, and environments checked | `uv run python scripts/package_submission.py --output artifacts/engineering-issue-desk-week3.zip`; [packaging tests](../tests/test_submission.py) |
| Change hygiene | `git diff --check` passed | Historical validation snapshot; baseline code later became public as commit 6e87e4b; no course submission performed |

The browser found that the previously running Streamlit process retained an old imported module. The development server was restarted after confirming that no case was running; the new checkbox then appeared. Restart the server after Python module changes if a browser refresh still shows old controls.

## Live evaluation accounting

The fresh repository-context run recorded 12 model dispatches and six tool dispatches across three cases, with **70,046 reported tokens** and usage present for every model call. Tool dispatches were two `read_case_evidence`, one `list_repository_files`, and three `read_repository_file` calls. One repository read exceeded the 160-line bound and returned a safe error; two smaller reads succeeded. Tool errors are included in accounting even when the overall case completes.

All three retained reports passed literal source/quote integrity checks. The AI reviewer removed one finding and two proposed steps. The run did not invoke repository search, read the `_configuration` helper, use Keenable, delegate to specialists, execute supplied code, or call the embeddings provider. Offline tests and the separate connectivity probe establish different properties. The [post-run inspection](../evaluation/week3-context-results/engineering-review.md) identifies remaining wording and quality limits; human quality fields remain null.

The independent recovery experiment used one additional reviewer call and **2,716 reported tokens**, excluded from the three-case total. The provider probe is also separate; no missing probe usage or billing total is inferred. Recovery reached `awaiting_review`; it did not approve the report.

Direct domain validation is another separate check. Its first harness attempt set a zero reported-token threshold, so the budget guard blocked every tool before dispatch. That [setup failure](../evaluation/week3-context-results/domain-tool-smoke-setup-failure.json) is retained, rather than counted as provider failure or success. The corrected harness kept model dispatch disabled, allowed four bounded tool calls, and all four completed successfully. Web search was already covered by the separate provider probe. One earlier public HTTP request discovered the PR number; it is not an agent tool/model dispatch or part of the three-case total.

## Preserved observations

| Artifact | SHA-256 | Meaning |
| --- | --- | --- |
| [Original Week 3 raw report](../evaluation/week3-results/report.json) | `ae73ff16315b04656842d5d79c8d627bc3d8ea75277b190f5b0f943c7651c4cd` | Initial 2/3 handoff result, one failed review, retained false positives |
| [Repository-context raw report](../evaluation/week3-context-results/report.json) | `cf6c29b0bbef8dc02a8c01b250ba28199b344f67ebcb9ebb91e57a31eea9a23c` | Later 3/3 handoff observation with repository options enabled |

The original failed case's private state later advanced through the separately recorded retry. The earlier exported failure remains a valid historical observation and was not overwritten. The runs differ in several tool/prompt/reviewer conditions; their handoff counts do not isolate a causal quality improvement.

Private SQLite backups were created before the additive case-option migration, and both backup databases passed `PRAGMA integrity_check`. They remain under ignored `.cache/backups/` and are excluded from the submission. A full restore rehearsal in a separate workspace was not performed; the [runbook](OPERATIONS.md#restore-rehearsal) describes that future check.

## Limits that remain

- Human usefulness, faithfulness, citation entailment, and the proposed eight-of-ten success target remain unmeasured. AI review can accept incorrect claims or reject useful ones.
- This is a personal local workspace with read-only research. Hosted identity, tenant isolation, distributed execution, encrypted case storage, semantic memory/Mem0, code execution, and GitHub publishing are not implemented.
- Credential-pattern guards are heuristic, and bounded prefix search can miss later code. Explicit reads identify the working-tree snapshot at read time; the repository is not frozen at case creation.
- macOS/Linux file locking is required. Only this local Python 3.12 configuration was exercised; Python 3.13, Windows, multi-host storage, and production load were not validated.
- The video, Week 3 GitHub push, reviewer access, and course-form submission remain separate tasks. The final Word/native Doc refresh and link access are tracked in the submission readiness matrix; earlier exports preserve historical observations.
