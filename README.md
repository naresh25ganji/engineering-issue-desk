# Engineering Issue Desk

Investigate Streamlit behavior and prepare a reviewable issue draft from a bounded snapshot of official documentation, release notes, contribution guidance, and public issue reports. Inspect exact supporting passages, identify missing local details, and keep source authority and version scope visible.

This independent course project does not establish a private application's root cause, prove a duplicate, promise a fix, inspect arbitrary host files, or execute supplied programs. Week 3 adds a Deep Agents investigator with Fireworks, Keenable, saved cases, and resumable human review. The investigator cannot publish; an approved report can use a separate human-confirmed PR summary workflow.

For the Week 3 walkthrough, open the [system architecture](docs/week3-system-architecture.svg), [investigation sequence](docs/week3-investigation-sequence.svg), and [architecture documentation](docs/ARCHITECTURE.md). The app's **Project guide** tab displays and downloads these assets. The original retrieval [diagram](docs/architecture.svg) and [PNG](docs/architecture.png) remain available for the earlier workflow.

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://localhost:8502. The 65-source snapshot is included. Keyword evidence mode needs no model or provider key. Semantic/hybrid retrieval uses real local FastEmbed `BAAI/bge-small-en-v1.5` embeddings by default; prepare model files and indexes before a fully offline demonstration. First-use downloads/indexing can take much longer than a warm query. A failed model never silently selects a different retrieval mode.

The guided workflow defaults to structured passages. Fixed/structured indexes and optional hosted indexes are fingerprinted and cached separately in ignored `.cache/`; the bundled historical base-pipeline index does not replace those separate indexes. Changed corpus/model identities invalidate stale vectors.

## Eight working views

| View | What it provides |
| --- | --- |
| Investigate | Structured reporter details, cited or extracted evidence, related public reports, missing details, workflow trace, and downloadable draft/JSON |
| Deep Investigation | Fireworks-powered planning and specialist delegation, Keenable/GitHub tools, immutable source evidence, saved cases, and approve/reject/revise review |
| Chat | Follow-up investigations with at most the last three user requests used only to resolve explicit references; prior assistant answers are never source evidence |
| Knowledge base | Inspect the included source text, upstream URL, snapshot, source type, and version scope |
| Relationships | Explore a provenance-bearing graph of literal API mentions and explicit reported versions, kinds, and issue states |
| Retrieval lab | Compare fixed and structured passages and optional real FlashRank reranking for the same query |
| Evaluation | Inspect recorded experiment reports and separately labeled user outcome summaries |
| Project guide | View/download system and sequence diagrams, architecture, design decisions, operations, validation, and submission docs |

The original investigation's LangGraph path is `prepare → retrieve → grade → analyze → handoff → finish`, with conditional clarification/abstention branches. LangChain Core `Document` and `RunnableLambda` carry evidence and wrap nodes. Each eligible request retrieves once. Guards can request local diagnostics before retrieval. This original path remains stateless; the new Deep Investigation path has separate durable checkpoints.

## Week 3 Deep Investigation

The new mode helps an engineer investigate a Streamlit report, compare documentation and public repository evidence, inspect supplied reproduction code, and prepare a cited report for review. A real `deepagents.create_deep_agent` loop chooses tools and delegates to three specialists: documentation/issue research (`general-purpose`), correctness/security, and tests/reproduction. Planning uses `TodoListMiddleware` explicitly. The model controls research and delegation; deterministic Python checks source identities and literal quotes, manages review, and renders exports.

Configure the local `.env` file with `FIREWORKS_API_KEY`, `FIREWORKS_MODEL`, and `KEENABLE_API_KEY`, then restart Streamlit. Select a Fireworks model that supports tool calling. The local probe verified `accounts/fireworks/models/gpt-oss-120b` with the configured account; other accounts must verify their own model access. The Fireworks key and model are required to start. Keenable is optional: unavailable web tools return an explicit error while local snapshot and public GitHub tools remain usable. Existing Nebius configuration continues to serve the original mode.

Starting a deep investigation sends the issue, supplied code, and selected evidence to Fireworks. Keenable receives search queries and requested page URLs. Optional `GITHUB_TOKEN` increases public API allowance. `LANGSMITH_TRACING=true` with `LANGSMITH_API_KEY` enables tracing; traces may retain model/tool content. Keep all credentials in `.env`, which is excluded from Git and submission assets.

The five domain tools are:

- `search_knowledge_base`: bounded keyword search of the included corpus, with its original snapshot date and version scope.
- `search_web`: Keenable discovery restricted to Streamlit, Python, and LangChain documentation and `streamlit/streamlit` GitHub pages; snippets are labeled as snippets.
- `fetch_web_page`: fetch an approved page through Keenable, preserving its literal bounded content and retrieval timestamp.
- `read_github_issue`: read a public Streamlit issue and bounded comments, attributed as reports rather than confirmed diagnoses.
- `read_github_pr`: read PR context and up to five file patches using immutable base/head commit comparison; truncated patches are labeled.

Specialists access evidence through a shared case registry rather than relying on filesystem visibility. Deep Agents' filesystem is a virtual `StateBackend`; the agent has no host shell or code execution tool. Source records preserve exact text, hashes, source URLs, evidence type, timestamps, version scope, and commit references. Each source has deterministic quote IDs for literal spans of up to 1,000 characters. The model selects an evidence ID and one of that source's quote IDs; the application resolves the saved text, so punctuation, Unicode, and whitespace cannot be rewritten by the model. Legacy saved literal citations remain supported. Model-facing tool results show spans without duplicating the raw source text. **These checks establish citation integrity, not entailment or factual correctness.** A separate Fireworks evidence reviewer assesses complete findings and proposed steps against selected quotes, removes unsupported items, and records its verdicts before the human checkpoint. It cannot author replacements or approve the report. Its one model call shares the case budget, and a failed review leaves the case unapproved. AI review can miss unsupported advice or reject useful content; a human must still assess the evidence and causal applicability.

The surrounding LangGraph flow is `investigate → evidence_review → human_review → finish`, with a revision branch back to investigation. SQLite checkpoints persist both the Deep Agents loop and the human-review state under stable case/thread IDs. Approval and rejection resume without calling a model. Requesting revision supplies the review notes and prior report to a new agent iteration. Approval saves the local report; a separate explicit preview/confirmation is required for a public PR summary. Saved cases can be reopened from the workspace selector.

Case records, immutable evidence, audit events, checkpoints, and process locks live under ignored `.cache/agent/`. This is one local personal workspace, not a multi-user hosted service. Independent case locks prevent simultaneous executions, and startup recovery only marks an abandoned running case as failed after acquiring its process lock. Retry resumes the unfinished checkpoint; completed reads and pending review are not automatically repeated. An invalid completed report can be repaired on explicit retry, at most twice. A report repair starts a fresh inner conversation with the original request and saved source registry; its calls still consume the same case-wide allowance. Human-requested revisions receive the previous report as well as review notes.

Each case shares limits across the orchestrator, specialists, retries, and revisions: 18 model calls, 28 tools, and two human-requested revisions. The 64,000 reported-token threshold and 180-second active-invocation deadline are checked before dispatch; in-flight calls have separate timeouts and may finish after a threshold. Missing token usage remains unknown. Automatic model-powered summarization is replaced with a deterministic 80,000-character message-context guard so summary calls cannot bypass accounting. Large cases stop with retained evidence rather than silently expanding the allowance. Truncated or malformed model tool output stops before further tool execution, with reported usage preserved. Provider errors are sanitized; the plan, calls, reported usage, and saved provenance remain inspectable.

The CLI uses the same saved cases and review graph. After configuring `.env`, explicitly check provider connectivity with the small probe before a full investigation:

```bash
uv run python scripts/investigate_deep.py probe
uv run python scripts/investigate_deep.py start --question "My uploaded CSV resets after a widget interaction. Investigate the documentation and propose a focused reproduction."
uv run python scripts/investigate_deep.py list
uv run python scripts/investigate_deep.py show CASE_ID
uv run python scripts/investigate_deep.py audit CASE_ID
uv run python scripts/investigate_deep.py review CASE_ID --decision approve --notes "Reviewed the cited sources."
```

The probe makes at most two small Fireworks calls and one Keenable search when the respective configuration is present. It returns status/booleans/counts, not keys or raw model/search text. `list` and `show` need no credentials. To review a file, add `--code-file triage_desk/reviews.py` to `start`; only that explicitly selected file is read. A revision uses `review CASE_ID --decision revise --notes "Describe the change"`; retry uses `retry CASE_ID`. Legacy drafts without an evidence review must use `audit CASE_ID` before approval; auditing reviews saved evidence without repeating research.

Run all offline regression checks with `uv run pytest -q`. The new tests exercise the actual Deep Agents harness with a scripted tool-calling model, real tools, real SQLite checkpoints, specialist delegation, malformed-citation rejection, cross-specialist limits, restart/review, revision, and failure recovery. These tests establish workflow behavior with controlled responses. Provider connectivity and live evaluation have separate recorded evidence; none establishes human investigation quality. The earlier manual plan in `evaluation/agent-cases.json` remains unscored. The executable three-code-sample dataset is `evaluation/week3-cases.json`.

## Week 3 deliverables and recorded evaluation

The [submission handoff](docs/WEEK3_SUBMISSION.md) maps the handout to the agent, datasets, development iterations, required links, and remaining work. The [readiness matrix](docs/SUBMISSION_READINESS.md) distinguishes required deliverables from stretch options. The [final Week 3 write-up](docs/Engineering%20Issue%20Desk%20Week%203%20Writeup.docx), [final native Google Doc](https://docs.google.com/document/d/1ImRAJn28t61YaaqlSdyZIDpqb_UEK_oRYHHsDFbrWdI), [4:45 demo guide](docs/Engineering%20Issue%20Desk%20Week%203%20Demo%20Guide.docx), and [agent architecture](docs/week3-architecture.svg) accompany the code. The private native Doc passed metadata/content readback and visual inspection of all nine converted pages; reviewer access still needs confirmation. The validated October 25 implementation is public in [commit `7400304`](https://github.com/naresh25ganji/engineering-issue-desk/commit/740030480909d1e38b4697dda6810dde5821b52d); its [GitHub CI](https://github.com/naresh25ganji/engineering-issue-desk/actions/runs/40247965771) passed on Ubuntu with 603 tests. The user confirmed no existing video, and no course form has been submitted.

## October 25 final verification and stretch options

The latest frozen [verified three-code-sample report](evaluation/week3-verified-results/report.html) and [raw JSON](evaluation/week3-verified-results/report.json) record **3/3 saved handoffs**, zero failed cases, 12 model calls, six tools, and 70,003 reported tokens with complete returned usage. There were nine investigator calls and three evidence-review calls. Tool use was two saved-evidence reads, one repository inventory, and three repository reads, including one safe line-bound error followed by smaller reads.

Only W3-02 retained factual findings: two static observations about `freshness.py`, with exact citation integrity passed. W3-01 and W3-03 retained no factual findings, so citation checks are **not applicable**, rather than passes or accuracy successes. AI review removed four draft findings and three proposed steps. Human quality remains unscored; none was automatically approved. The third sample established no defect. This run did not exercise playbook/history lookup, Bandit, Keenable, repository search, or specialist delegation.

Four stretch options extend the agent; the handout has no official extra-credit checklist or point values:

- **Maintained playbooks:** list/load three versioned, hashed guidance bundles on demand. This is tool delivery, not native `SkillsMiddleware` or factual evidence.
- **Earlier human reviews:** default-off immutable opt-in to bounded literal prose lookup from approved/rejected cases in this local SQLite workspace. Results are untrusted guidance, not source evidence, semantic retrieval, or Mem0.
- **Static checks:** real Bandit with eight fixed rules over registered Python code in a bounded trusted parser worker. No supplied program or tests execute; flags are not verified vulnerabilities.
- **Approved public PR summary:** a separate offline preview and exact plan-ID confirmation can publish a summary `COMMENT` after unchanged case/evidence and fresh base/head checks. The agent cannot publish. Atomic plan claims prevent replay; uncertain POSTs cannot auto-retry. Only mocked HTTP/state validation was performed, with no live POST or write token used.

A separate [live extension exercise](evaluation/week3-verified-results/extension-exercise.json) observed one playbook load and real Bandit call on the 30-line freshness sample, alongside one evidence list and one evidence read. It used six model calls, four tools, and 34,120 reported tokens and saved one finding plus two test proposals pending human review. These totals are outside the three-case denominator. Review memory was opted in but never invoked; positive history recall is covered by controlled harness tests.

The [readiness matrix](docs/SUBMISSION_READINESS.md) and [validation record](docs/VALIDATION.md) distinguish implemented scope, controlled tests, actual live observations, and remaining deliverables. The October 17 corpus remains a dated snapshot on October 25; live evidence has separate acquisition timestamps.

Final software verification passed **603 tests** in 13.77 seconds, the 65-source integrity check, and source-environment validation. The [machine-readable validation record](evaluation/week3-verified-results/validation.json) and [qualitative inspection](evaluation/week3-verified-results/engineering-review.md) preserve the exact boundary. The latter is Codex-assisted analysis, not human approval. Supported execution is a source checkout with `uv`; default wheel build discovery is not configured.

### Preserved initial Week 3 run

The initial [three-code-sample report](evaluation/week3-results/report.html) and [raw JSON](evaluation/week3-results/report.json) preserve an earlier run of `evaluation/week3-cases.json`:

| Observed measure | Result |
| --- | --- |
| Saved human-review handoffs | 2/3 scheduled cases |
| Failed cases | 1; independent evidence review response did not pass validation |
| Literal quote/source integrity | 3/3 drafts, including the failed case; this does not establish entailment |
| AI filtering on completed reviews | 1 finding and 2 proposed steps removed |
| Dispatch and usage | 9 model calls, 3 tools, 44,382 reported tokens; usage present for all calls |
| Human accuracy, usefulness, citation entailment | Unscored |

The static samples required only supplied code: the observed tools were three `read_case_evidence` calls, with no specialist delegation or live web research in this run. Other scripted harness tests exercise delegation; a separate actual provider probe verified Fireworks tool calling and Keenable search. The [post-run code inspection](evaluation/week3-results/engineering-review.md) records incorrect retained claims. AI review did not establish investigation quality, and no case was automatically approved.

Validate fixtures without provider calls, or explicitly run another live evaluation into a new directory:

```bash
uv run python -m evaluation.agent_workflow --output-dir .cache/week3-input-inventory
uv run python -m evaluation.agent_workflow --live --output-dir evaluation/week3-results-rerun
uv run python scripts/package_submission.py --output artifacts/engineering-issue-desk-week3.zip
```

The initial Week 3 verification passed **401 tests** and the 65-source integrity check. The source package excludes local credentials, private saved cases, environments, caches, and logs, and includes a per-file hash manifest.

## October 23 architecture and reliability update

The [architecture](docs/ARCHITECTURE.md), [design decisions](docs/DESIGN.md), [operations runbook](docs/OPERATIONS.md), and [validation record](docs/VALIDATION.md) document the implemented system, state model, trust boundaries, recovery, and deployment limits. The [archived initial native Google Doc](https://docs.google.com/document/d/1gOxafWnF3u6OI5Rp1OtHMQyF0VYTPsBnm_NzWH3Dzus) preserves the first Week 3 observations; final Word/native Doc assets and these technical docs include the later extensions.

**Related application code is optional.** Enable **Include related application code** when creating a case, or add `--repository-context` to CLI `start`. Three additional tools list, search, and read only `app.py` and Python files under `triage_desk/`. The immutable case option defaults off, including migrated older cases; retry/revision inherit it. Selected context goes to Fireworks. No code is executed.

Reads preserve whole lines, relative paths, file/excerpt hashes, timestamps, and immutable case evidence. Inventory/search are bounded discovery operations; a source must be read before citation. Symlinks, hardlinks, nonregular files, hidden paths, and paths outside the allowlist are rejected. Configured credential values and common credential patterns cause the whole file to be withheld; this heuristic does not classify every possible secret. Search shares a bounded character allowance across file prefixes, so omitted matches are not proof of absence. Exact bounds are in the architecture document.

Evidence-review failures now have finite safe codes in the saved timeline and CLI `show`. Harmless paired inline-code backticks in reviewer reasons normalize to plain text; schema, coverage, URLs, HTML, and unsafe formatting checks remain enforced. No automatic model retry was added.

The [new live report](evaluation/week3-context-results/report.html) recorded **3/3 saved handoffs**, 12 model calls, six tools, and 70,046 reported tokens across all three cases, with usage available for every model call. One overlong repository read was rejected; smaller reads succeeded. It used repository listing/reading, but no repository search, helper read, Keenable, or specialist delegation. The [post-run inspection](evaluation/week3-context-results/engineering-review.md) records the remaining quality limits. This run changes multiple implementation/prompt conditions and does not isolate a causal quality improvement.

A separate [live checkpoint recovery](evaluation/week3-context-results/checkpoint-recovery.json) resumed the original failed W3-02 case using only one evidence-review call, zero tools, and 2,716 reported tokens. It reached `awaiting_review`, repeated no investigator call, and did not approve the report. The frozen original 2/3 report remains unchanged. A separate [provider probe](evaluation/week3-context-results/provider-probe.json) again verified a Fireworks tool round trip and five Keenable results.

October 23 validation passed **473 tests**, the 65-source integrity check, actual browser checks, diagram rendering, and source-package checks. See the validation record for the exact scope and unmeasured outcomes.

```bash
uv run python scripts/investigate_deep.py start --question "Review the embedding response and its configuration helpers." --code-file triage_desk/hosted_embeddings.py --repository-context
uv run python -m evaluation.agent_workflow --live --repository-context --output-dir evaluation/week3-context-rerun
```

## Evidence and optional Nebius models

**Evidence mode** returns literal retrieved passages, heuristic type/component suggestions, and a draft. It does not generate an LLM answer, and offline directness grading is explicitly unscored.

For **generated answers**, copy `.env.example` to `.env`, configure the server-side key and model IDs, restart the app, and enable the generation toggle:

```bash
cp .env.example .env
```

The measured configuration used these non-secret settings:

```dotenv
TRIAGE_BASE_URL=https://api.tokenfactory.nebius.com/v1
TRIAGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
TRIAGE_VERIFIER_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
TRIAGE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

Set `TRIAGE_API_KEY` locally; `NEBIUS_API_KEY` is also supported. Never commit `.env`. The guided generated path can make one batched evidence-directness call, one synthesis call, and one separate selected-quote support audit. The audit may omit unsupported claims; partial answers are labeled. Invalid/unavailable checks fall back safely without an automatic repair loop. Fallbacks and partial generated answers require human handoff review. Source IDs and exact quotes establish reference integrity, not entailment or correctness.

Local BGE remains the default embedding backend. Selecting **Nebius course credits** explicitly uses `Qwen/Qwen3-Embedding-8B` with a separate 1,024-dimensional document/query index; source passages and questions are sent to the provider for embedding. Generation sends the request and retrieved passages to the configured chat provider only when enabled. Keyword mode makes no embedding request.

FlashRank `ms-marco-MiniLM-L-12-v2` reranking and graph augmentation are opt-in. The lab retrieves up to 18 candidates and returns six chunks with source caps. Graph augmentation filters known sources by the selected source kinds before adding at most four sources, within an eight-chunk total budget. The graph's relationships do not imply causality, duplicates, compatibility, or fixes.

## Measured results and limits

The latest recorded expanded workflow regression used 20 previously authored AI validation cases: 14 expected-evidence and six boundary cases. It combined structured hybrid retrieval, hosted Qwen embeddings, FlashRank, optional graph augmentation, 30B directness/synthesis, and a 235B support audit. These results do not isolate the benefit of any one feature.

| Measure | Recorded result |
| --- | --- |
| Expected routing/behavior | 19/20; all six boundary cases passed |
| Generated answer coverage | 10/14 expected-evidence cases; four accepted answers were partial |
| Separately reviewed selected-quote support | 19/22 retained claims judged fully supported by an AI reviewer (86.4% AI judgment, not measured accuracy) |
| Mean AI quality ratings | Faithfulness 1.70/2; answer relevance 1.10/2, over the 10 generated answers |
| Answerable-case backend p95 | 24.079 seconds; the 20-second target is not met, before browser transport/rendering |
| Actual workflow provider usage | 55 calls; 62,215 reported total tokens |

The 55 calls comprise 16 query embeddings, 16 directness grades, 13 synthesis requests, and 10 support audits. Three drafts failed structural/formatting checks and safely fell back; one expected-evidence case requested clarification. Separate cold hosted indexing used 113,064 reported tokens and is excluded from the run total. Missing usage is never inferred. **Human faithfulness, relevance, and entailment review fields remain blank.** AI acceptance and AI review can share errors; the proposed 95% claim-faithfulness target has not been established.

The frozen workflow raw report is [evaluation/workflow-results/raw-report.json](evaluation/workflow-results/raw-report.json), SHA-256 `827db28bfe99f0d42eecd84bcaebf70c4ba1bab21a3f206a99e20c1a5d8dd6cb`. The [readable report](evaluation/workflow-results/report.html) and [separate AI review](evaluation/workflow-results/ai-review.json) preserve that run. A later handoff-flag policy fix was regression-tested separately and does not rewrite the frozen observations or imply another live evaluation.

The [local retrieval comparison](evaluation/retrieval-lab-results/report.html) used 378 fixed and 688 structured chunks. On 14 validation evidence cases, gold-source precision was 0.402 fixed, 0.410 structured, and 0.310 structured plus reranking; family recall was 0.964, 0.964, and 0.893. Labels are incomplete source-selection proxies, not RAGAS or answer-quality scores. Reranking showed no consistent advantage and added latency, so it remains opt-in. The [standalone graph comparison](evaluation/graph-comparison-results/report.html) hit an expected family for 5/14 original and 3/14 validation cases, versus 14/14 for hybrid source selection. Exact-seed graph retrieval has low recall here; graph exploration/augmentation remains optional.

## Reproduce checks and experiments

Use new output directories to preserve frozen reports. The following local paths request no hosted model:

```bash
uv run pytest -q
uv run python scripts/fetch_corpus.py --check
uv run python -m evaluation.workflow --questions evaluation/questions.json --mode keyword --output-dir evaluation/workflow-local-rerun
uv run python -m evaluation.retrieval_lab --questions evaluation/questions.json --questions evaluation/validation-questions.json --output-dir evaluation/retrieval-lab-rerun
uv run python -m evaluation.graph_comparison --questions evaluation/questions.json --questions evaluation/validation-questions.json --output-dir evaluation/graph-comparison-rerun
```

Dense/reranking experiments require local model caches or initial downloads. The original core keyword/semantic/hybrid comparison is also available through `uv run python -m evaluation.run`; it does not exercise the expanded graph workflow.

With configured course credits, this explicitly requests hosted embeddings and generation:

```bash
uv run python -m evaluation.workflow --questions evaluation/validation-questions.json --embedding-provider nebius --rerank --graph --include-generation --output-dir evaluation/workflow-live-rerun
```

The workflow CLI loads `.env` only when generation or hosted embeddings are explicitly selected. It runs once per case without hidden warmups or a second search, counts embedding/grading/synthesis/verification calls once, and labels final-selection metrics as post-grade rather than raw retrieval. These reused question sets are regression diagnostics, not new unseen validation data.

## Outcomes, data, and refresh

A user can label an investigation **Resolved from the evidence**, **Needs follow-up**, or **Human handoff needed**. The local `.cache/user-outcomes.jsonl` log stores only run ID, label, first-contact flag, and timestamp. Its summary uses the latest label per reviewed run and reports user-reported resolution only over labeled first contacts. No automatic first-contact resolution or helpdesk effectiveness claim is made.

[data/manifest.json](data/manifest.json) and each source preserve allowlist, hashes, provenance, authority, snapshot, and version scope. The corpus has 38 documentation/API records, 24 public issue reports, two contribution records, and one bounded release excerpt. Source licenses and attribution are in `data/licenses/`.

```bash
uv run python scripts/fetch_corpus.py
```

Refresh is manual. Documentation/release pins do not automatically advance to the latest release. Cached immutable source timestamps retain their original fetch time; reading cache bytes does not make them fresh. The UI warns when the oldest valid source is at least seven days old and flags invalid timestamps as unknown. Review changed sources, validate hashes, rebuild matching indexes, and rerun evaluation before publishing a refreshed snapshot.

A rebuildable source ZIP with per-file SHA-256 manifest can be prepared with:

```bash
uv run python scripts/package_submission.py --output artifacts/engineering-issue-desk-week2.zip
```

The script excludes credentials, environments, caches, and local outcome logs; it checks candidate assets against configured local credentials before writing. The local handoff includes a generated source ZIP at `artifacts/engineering-issue-desk-week2.zip`, with a per-file manifest inside it.

## Historical Week 2 handoff

- [Private project repository](https://github.com/naresh25ganji/engineering-issue-desk)
- [Private Google Docs write-up](https://docs.google.com/document/d/1C1OZmA_590xQs1ItVmqQq2ljM9BM2dvxlKilXt9vH44)
- [Verified six-page Word write-up](docs/Engineering%20Issue%20Desk%20Week%202%20Writeup.docx)
- [Four minute and forty second recording guide](docs/Engineering%20Issue%20Desk%20Video%20Recording%20Guide.docx) with narration, screen cues, and recording controls; also available as a [private Google Doc](https://docs.google.com/document/d/1wPnj_PcB3pk73QN6mVSyr9hAz0TVUiJd076T1sUF408)
- [Latest workflow report and AI review](evaluation/workflow-results/report.html)

The Week 2 handoff verification passed 200 tests at that point, the 65-source integrity check, real browser investigation/chat checks, and all six converted Google Docs pages. At that handoff the repository and Google Doc were private. The repository is now public; access to older private Docs is a separate concern. The video and course form remain outstanding deliverables.

## Course expansion

The backend is independent of Streamlit. Continue this project with read-only engineering tools/MCP, stronger evaluation and observability, reviewed issue-routing training data/local model comparisons, and deliberate action/permission controls as later handouts require. See [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md) for requirement mapping, prompt locations, four-guide adaptations, historical runs, and the five-minute demo sequence.

```text
app.py                       Eight-view Streamlit workspace
triage_desk/                  Core RAG, workflow, retrieval lab, graph, providers, outcomes
scripts/fetch_corpus.py       Bounded source snapshot and offline validation
scripts/package_submission.py Source ZIP and per-file manifest builder
data/                        Corpus, provenance, licenses, historical base index
evaluation/                  Question sets, CLI runners, frozen reports and AI reviews
tests/                       Behavior, provider-contract and interface checks
docs/                        Assignment mapping and demo guide
```
