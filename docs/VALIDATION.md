# Validation record

Verified October 23, 2026, in the local Python 3.12 workspace. These results apply to this implementation and these examples. They do not establish human answer accuracy, production readiness, or general agent reliability.

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
| Change hygiene | `git diff --check` passed | Changes remain local; no GitHub push or course submission performed |

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
- The video, Week 3 GitHub push, reviewer access, and course-form submission remain separate tasks. Earlier Word/Google Docs assets describe the initial Week 3 version; these technical docs record the later update.
