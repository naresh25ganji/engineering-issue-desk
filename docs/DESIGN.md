# Engineering Issue Desk design

Version date: October 23, 2026.

The design favors an inspectable local investigation over unrestricted automation. An engineer should be able to identify what the agent read, what it inferred, what failed, and which decision was actually recorded. This document records the decisions behind the [architecture](ARCHITECTURE.md), including consequences that remain relevant when adding features.

## Invariants

- A factual finding must resolve to evidence registered in the same case and an exact saved quote span.
- A matching quote does not prove that the finding follows from it. AI support review can remove content, and a human still decides whether to accept it.
- A source, pasted code, tool result, or model response cannot expand permissions or publish anything.
- Model/tool dispatch consumes the same case allowance across specialists, retries, repairs, and revisions. Changing a thread cannot reset it.
- Approval and rejection are local state transitions that do not call an investigator model. Final cases cannot be replayed as another decision.
- Provider/tool failure cannot silently become successful research, a completed audit, or an approved report.
- A source snapshot remains immutable. Working-tree or upstream changes do not rewrite evidence already used by a case.

## ADR 1 Use Deep Agents inside an application controlled graph

**Decision.** The inner Deep Agents harness chooses research steps, plans with todos, and delegates to specialists. The outer LangGraph service owns input validation, persistence, review stages, final status, and error recovery.

**Reason.** Tool choice benefits from adaptation, while review authority and execution limits need explicit control. A single prompt cannot reliably enforce the application's state transitions or authorize an external action.

**Consequence.** There are two checkpointed layers to reason about. Stable case/iteration/repair thread IDs and integration tests are necessary. The existing Week 2 retrieval graph remains a separate mode; it is not retroactively given autonomous tools or the new case lifecycle.

## ADR 2 Keep durable case state in SQLite

**Decision.** Use `cases.sqlite` for application records and a separate SQLite saver for graph checkpoints. Keep case evidence and events beside the report, and serialize new structured outputs as JSON dictionaries.

**Reason.** A local workspace needs restart/resume, precise source snapshots, and simple backup. Semantic memory is a different operation: retrieving a prior conclusion does not reconstruct a pending tool call or human interrupt.

**Consequence.** Mem0 is not part of the current implementation. Future resolution recall must scope projects, preserve approval/provenance/version context, and avoid treating an old answer as new source evidence. Two database files require a quiescent, consistent backup; they are not a distributed transaction or a hosted service.

## ADR 3 Save evidence before accepting citations

**Decision.** Tools register literal evidence and provenance in the case store. The application derives deterministic quote IDs; the model selects IDs instead of transcribing quotations. Rendering resolves and checks saved text again.

**Reason.** A live model modified quotation text even though its report looked plausible. Exact string matching appropriately rejected it. Source-assigned span IDs remove that transcription failure without weakening citation integrity.

**Consequence.** A quote can be longer than the smallest useful sentence, and a bounded span may omit relevant context. Re-reading a source is allowed; inventing missing context is not. Legacy literal citations remain readable, but new model report schemas require quote IDs.

## ADR 4 Review support independently without giving the reviewer authorship

**Decision.** A separate Fireworks call reviews the selected findings and next steps against selected quotes only. It returns one indexed supported/rejected verdict and reason per original item. Application code can retain or remove items; the reviewer cannot author replacements or approve the report.

**Reason.** Exact citations did not prevent an inapplicable uploader workaround. Keeping filtering separate from report authorship makes the intervention inspectable and prevents an audit response from adding unchecked facts.

**Consequence.** The reviewer has intentionally limited context and may miss causality, assume omitted code, or reject a useful diagnostic. The frozen W3-03 result retained unsupported empty-key and error-body-leak claims; W3-02 failed structured reviewer validation. A completed AI audit is not a human quality score. A failed audit leaves the draft unapproved and resumable.

Verdict explanations remain plain text. Paired single inline backticks are normalized as presentation around unchanged identifiers or expressions; links, HTML, controls, fences, bold markup, and unmatched/doubled backticks are still rejected. Eleven finite failure codes distinguish response/schema/coverage/context/provider errors without logging raw provider explanations. This improves diagnosability without accepting incomplete verdicts or retrying automatically.

## ADR 5 Make research read only and keep write authority local

**Decision.** Domain tools read the bundled corpus, approved public pages, and the fixed Streamlit GitHub repository. Agent scratch files use a virtual backend. Approval saves a local decision; no tool posts a GitHub issue or review.

**Reason.** Investigation can proceed within a bounded read scope without granting credentials for external writes or a host shell. Human review is meaningful when the application enforces the resulting authority boundary.

**Consequence.** Proposed reproductions and tests are suggestions, not execution results. Private application root causes, verified duplicates, exploitability, and working fixes cannot be established from public similarity alone.

## ADR 6 Account before dispatch and stop visibly

**Decision.** Persist model/tool dispatch records before requests, pair terminal records by call ID, and reconstruct cumulative usage after restart. Default case limits are 18 model calls, 28 tool calls, 64,000 reported tokens, two revisions, and two report repairs. Each active service invocation checks a 180-second deadline before dispatch.

**Reason.** Counting only successful returns would refund crashed or failed requests. Independent specialist limits would let delegation multiply the allowance. Automatic summarization can make hidden model calls, so it is replaced with an 80,000-character serialized-message guard in every role.

**Consequence.** A dispatched request may finish after a token/time threshold; this is not a hard cancellation guarantee. Missing usage stays unknown. No billing total or dollar estimate is inferred. Truncated output and malformed tool arguments stop with safe diagnostics after usage is recorded.

## ADR 7 Coordinate case execution with a process lock and atomic claim

**Decision.** Acquire a nonblocking `fcntl` advisory lock before the database status claim, and hold it throughout execution. Keep lock files in place. Startup recovery only marks an abandoned `running` case failed after acquiring its lock.

**Reason.** A status flag alone cannot distinguish a crashed run from a live process, and deleting/recreating lock files can cause inode races. Atomic status updates also prevent invalid review/retry transitions.

**Consequence.** The service requires macOS/Linux file locking and a local filesystem with suitable lock semantics. It is not a multi-host work queue. Recovery identifies unfinished work; an engineer explicitly retries it. Tests crash actual subprocesses during normal and repair runs to exercise OS lock release and checkpoint reuse.

## ADR 8 Separate revision from report repair

**Decision.** A revision starts a fresh inner thread and carries the original request, prior report, and human notes. A report repair uses another fresh thread, original context, and the existing source registry after completed output fails schema/source checks. Both retain case budgets.

**Reason.** Notes such as correcting a specific finding are meaningless if the new thread cannot see the prior report. Appending corrective prompts to failed history can also preserve distracting or oversized context.

**Consequence.** Repair events persist the active revision and global repair number so a crash can resume that thread. A pending tool/model step is resumed rather than treated as another repair. Legacy repair events use a documented compatibility mapping. A user cannot gain more calls by repeatedly requesting a new inner conversation.

## ADR 9 Make repository context explicit and bounded

**Decision.** Offer three static list/search/read tools only when the engineer opts in at case creation. Store the immutable choice with the case. Restrict reads to `app.py` and Python files inside `triage_desk/`; persist explicit read excerpts as ordinary case evidence.

**Reason.** A concrete failure inferred missing behavior from an excerpt without its configuration helper. Finding related code can improve context, while silently exposing an entire working tree would grant more authority than a pasted sample. The application owns path validation, descriptor-based symlink/race defenses, credential guards, and byte/line/call bounds.

**Consequence.** Search returns bounded leads, not evidence or an absence proof. Its character allowance is shared across file prefixes so one large early file does not consume all discovery context. Explicit reads create immutable hashed excerpts; the whole tree is not frozen. Current code may differ from later or earlier reads, credential detection is heuristic, and static inspection cannot establish runtime correctness. The extension adds neither Mem0, providers, code execution, nor publishing. Its live evaluation is separate from the frozen pre-extension run.

## What the measurements mean

| Observation | What it can support | What it cannot support |
| --- | --- | --- |
| Offline harness and SQLite tests | Particular tool/state/recovery invariants with controlled providers | Live account access or general answer quality |
| Fireworks/Keenable probe | Small live connectivity/tool-call round trip for that configuration | A complete useful investigation |
| Exact quote validation | The selected literal text belongs to a saved case source | Entailment, causal relevance, or factual correctness |
| Completed independent AI review | A model returned valid verdicts and application filtering completed | Human accuracy or absence of false positives |
| `awaiting_review` | A saved case reached the human handoff | Approval, resolution, or a verified fix |
| Human approval | An engineer accepted that local report | External publication or a measured population success rate |

The proposed usable-handoff target remains a target. Validation changes should produce new result directories; a new feature must not rewrite an earlier live run into a better observation.
