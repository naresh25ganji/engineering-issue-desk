# Week 2 assignment: Engineering Issue Desk

## Project and user

Engineering Issue Desk helps a developer investigate a Streamlit question or prepare a useful issue report. It retrieves official documentation and potentially related public reports, preserves the distinction between them, and prepares a draft for the developer to review. A similar issue is evidence to inspect; it is not proof of a duplicate, root cause, or working fix.

This is a new project rather than an extension of Campaign Compass or the legal PDF extractor. The initial domain is deliberately one framework and a small source snapshot. It can grow into an engineering support workflow across the course without becoming an assistant for every task.

The handout's deadline for **Builder of the Week eligibility** is **October 18, 2026, 11:59 PM Pacific**. It explicitly permits later project submissions through the same form for course/certification completion, without Builder of the Week consideration.

## Primer: exact one-liner

My RAG app helps Streamlit developers answer framework-behavior and related-issue questions from a versioned English corpus of 38 official documentation records, 24 public issue reports, two contribution records, and one release-note excerpt in a Streamlit investigation workspace, targeting **95% claim faithfulness** and **warm end-to-end p95 latency of at most 20 seconds**.

These are design targets, not observed quality claims. Claim faithfulness means the percentage of generated factual claims that a reviewer judges fully supported and correctly qualified by the selected cited evidence; the denominator is all factual claims in accepted generated answers, with answer coverage and fallback rates reported separately so abstention cannot silently improve the result.

The latency target covers submission to visible answer or explicit fallback, including query embedding, retrieval, fusion, generation, validation, and response rendering. It assumes the model and index are already loaded; first-use download/index preparation is reported separately, and an enforced request-wide timeout or UI timing measurement must not be claimed unless implemented and measured.

## The six-field RAG framework

| Field | Decision (one or two sentences) |
| --- | --- |
| Use case | A developer asks about Streamlit behavior or a similar public report in a local Streamlit investigation workspace and receives cited context plus a reviewable issue draft. The workspace does not establish a private application's root cause or a verified duplicate. |
| Corpus | The snapshot contains 65 English records: 38 official documentation/API records, 24 public issue reports, two contribution/template records, and one bounded release-note excerpt, normalized from Markdown/MDX, generated API docstrings, and issue text. Streamlit owns the official documentation, public reporters supply unverified observations, and this project maintains the allowlist, snapshot, and provenance. |
| Ingestion + cleaning | A reproducible allowlist-based script fetches commit-pinned documentation and dated issue snapshots, strips frontmatter/display wrappers and issue boilerplate, decodes generated HTML, and preserves technical prose, tables, headings, code, and source links. It removes obvious contact/credential strings and never executes source content. |
| Ingestion + freshness | The maintainer reviews documentation/release pins and refreshes snapshots weekly, targeting publication before the oldest valid snapshot reaches seven days and reviewed documentation updates within seven days of a chosen upstream release. Refresh before version-sensitive demonstrations; the interface warns at an oldest-snapshot age of seven days or more and treats invalid timestamps as unknown, while refresh itself remains manual. |
| Chunking + embedding | The new retrieval lab compares fixed 180-word windows with 35-word overlap against deterministic Markdown/paragraph/sentence-aware blocks capped at 180 words and 2,400 characters, preserving bounded code and table blocks and marking oversized splits. Both use actual local 384-dimensional `BAAI/bge-small-en-v1.5` embeddings by default; the structured strategy is **not embedding-based semantic chunking**, and the earlier heading-aware index remains available for historical reproduction. |
| Retrieve | Persisted, fingerprint-validated NumPy vectors support dense cosine retrieval, BM25 supplies the sparse baseline, and reciprocal rank fusion combines their rankings; the lab can rerank 18 candidates locally with FlashRank before returning six chunks with at most two per source. An optional exact-API/reported-version graph can add bounded source candidates, and a real LangGraph workflow grades source directness before cited synthesis and a separate claim-support audit or a clarification/handoff. |

## Required scope versus optional extensions

This is the handout's **bring-your-own use case**. The common Week 2 requirements are a scoped framework and one-liner, corpus ingestion, chunking, embedding, storage, retrieval, cited generation, and the final documentation/demo/project assets; the generated-answer evaluation makes faithfulness and remaining failures inspectable. The implementation began as direct Python RAG under the handout's allowance for other frameworks and now adds actual LangGraph orchestration and LangChain Core document/runnable interfaces.

The suggested projects contain additional requirements for their own selected use cases: the financial track compares two chunking strategies and reranking, GraphRAG builds 20+ nodes and compares graph/vector results, and the support-KB track measures first-contact resolution and uses escalation logic. These were not automatically required of this engineering use case, and a chat UI is explicitly a bonus add-on and is implemented here. The expanded project deliberately adapts selected ideas from all four supplied solution guides: retrieval experiments, an inspectable source-fact graph, bounded workflow branches, and clarification/handoff with short user follow-ups. It does not claim financial-domain RAGAS results, helpdesk resolution rates, automatic ticket escalation, or organizational relationships absent from this corpus.

## Assignment mapping

| Requirement | Implementation and evidence |
| --- | --- |
| A document-grounded use case | Streamlit engineering questions and issue preparation using official docs plus selected real GitHub issue reports |
| Ingestion | The snapshot script normalizes source text and preserves source ID, URL, kind, snapshot time, commit, and version scope in `data/sources.json` |
| Chunking | Historical heading-aware index plus a new fixed-window versus deterministic structured-block lab; literal source spans and split-block metadata preserve inspectable provenance |
| Embeddings | Actual local 384-dimensional `BAAI/bge-small-en-v1.5` embeddings through FastEmbed by default; optional real Nebius `Qwen/Qwen3-Embedding-8B` with a separate fingerprinted 1,024-dimensional document/query index |
| Persistent vector retrieval | Normalized vectors stored in a content-addressed NumPy index; cosine similarity search; corpus/model identity prevents stale cache reuse |
| Retrieval comparison | Historical keyword/semantic/hybrid comparison plus completed fixed-hybrid, structured-hybrid, and structured-hybrid-plus-FlashRank runs over both 20-case sets; source-selection metrics and initialization/warm timings remain separate from answer quality |
| Workflow and source graph | Compiled LangGraph `prepare → retrieve → grade → analyze → handoff → finish` workflow using LangChain Core `Document` and `RunnableLambda`; separate NetworkX source-fact graph has 178 nodes and 456 explicit-provenance edges |
| Grounded answer and citations | Evidence-only mode quotes retrieved passages; optional compatible chat synthesis uses supplied evidence and returns source IDs plus exact support quotes |
| Evaluation | Original 20-question diagnostic and a separately authored 20-case validation set; repeatable CLI, frozen raw outputs, hash-bound AI reviews, blank human-review fields, and readable reports; the expanded workflow run and separate AI review are in `evaluation/workflow-results/` |
| Faithfulness and answer relevance | Generated answers have a separately labeled AI assessment against selected supporting passages; human rubric fields remain unscored until a person completes them |
| Demonstration and write-up | This guide, repository README, runnable app, and a five-minute demo sequence below; the student still creates and submits the requested recording and final course documents |

LangGraph now controls the investigation branches, and LangChain Core carries text with source metadata as `Document` objects and wraps node callables as `RunnableLambda` instances. Retrieval, citation validation, and source-fact extraction remain inspectable Python components. The persistent NumPy cache is a **local vector index**, not a separate hosted database service.

## Architecture

```text
Official docs + selected GitHub issue snapshots
            │ normalize, provenance, pinned/datable snapshot
            ▼
        data/sources.json
            ├── fixed / structured literal chunks
            │       ├── BM25
            │       └── local BGE / hosted Qwen → separate vectors
            └── NetworkX source-fact graph
                    literal APIs / explicit reported versions / kind / state

Question + bounded prior USER requests for reference resolution
            ▼
   LangGraph prepare ── missing/private/local-state request ──┐
            ▼                                              │
         retrieve                                          │
   keyword / semantic / hybrid                             │
   optional local FlashRank + exact-seed graph candidates   │
            ▼                                              │
   LangChain Documents → grade ── insufficient/failed ───────┤
            ▼                                              │
         analyze                                           │
   evidence-only OR cited draft → structural checks         │
            → separate claim-support audit                  │
            └──────────────────────────────────────────────┤
                                                           ▼
                                               handoff → finish
                                               answer / clarification
                                               + reviewable issue draft
```

Hybrid retrieval combines keyword and dense rankings using reciprocal rank fusion with rank constant 60. The retrieval lab takes up to 18 first-stage candidates with a cap of three per source, optionally applies the actual local FlashRank `ms-marco-MiniLM-L-12-v2` model, and returns six passages with a cap of two per source. First-stage and reranker scores remain separate; neither is a calibrated probability or proof that a passage answers the question. Reranking can reorder candidates but cannot recover a passage absent from the candidate pool.

The two lab chunking strategies use the same sources and preserve literal contiguous text offsets. Fixed windows ignore structure and overlap by 35 words; structured chunks group Markdown headings, paragraphs, lists, fenced code, and tables within word/character limits, preferring complete lines or sentences when an oversized block must split. Such splits retain part/group metadata and are explicitly partial. This comparison changes boundary handling and overlap together, so it is a configuration comparison rather than an isolated causal test of one setting. No embedding-distance breakpoint chunker has been implemented.

The bundled 645-chunk vector index belongs to the historical core baseline. The default guided workflow uses the separate 688-chunk structured lab index and must compute it on first use when no matching derived cache exists; fixed and hosted indexes also have separate identities. All vectors come from actual embeddings. The loader accepts only a matching corpus/model fingerprint and valid matrix; a stale or corrupt bundle/cache is rebuilt in the writable derived cache. Query embeddings still require the selected local model or explicit hosted provider.

Search finds candidates; analysis applies a conservative relevance check and handles missing details and selected unsupported requests. Type and component labels are **heuristics** in Week 2. They are not a trained classifier and are not evidence that a developer's bug has been diagnosed.

### Bounded workflow, source graph, and follow-ups

`triage_desk/workflow.py` compiles a real `StateGraph` with conditional edges. Preparation can request concrete diagnostics or reject requests that public sources cannot answer before retrieval. The retrieve node resolves existing chunks, passes LangChain `Document` objects into grading, and the analyze node consumes those selected passages once; it does not secretly repeat retrieval. Stage traces retain decisions, reasons, evidence counts, and timings. Without generation, directness is explicitly unscored; with generation, one batched relevance-grading call can select useful source IDs or route to clarification/handoff. A malformed or unavailable grade fails closed. The nominal generated path can therefore make up to three chat-model calls: directness grade, synthesis, and support audit. Hosted document/query embedding requests are counted separately; earlier exits make fewer calls, and there is no autonomous retry loop.

`triage_desk/knowledge_graph.py` builds a distinct NetworkX `MultiDiGraph` over the 65 source records. Its observed 178 nodes comprise 65 sources, 90 literal `st.<identifier>` API symbols, 17 explicitly reported Streamlit versions, four source kinds, and two issue states; its 456 edges comprise 347 `MENTIONS_API`, 20 `REPORTS_VERSION`, 65 `HAS_KIND`, and 24 `HAS_STATE` relations. Every edge carries source provenance. API mentions retain whether they came from documentation or reports; an exact API documentation title can identify a documented subject, while a literal mention alone does not establish an official API definition. Version edges come only from an issue's explicit `reported_versions` field, never from arbitrary numbers, documentation API versions, release titles, or prose inference.

Graph inspection starts from exact known API/version seeds or an explicitly selected node and traverses at most two hops in either direction while preserving edge orientation. Views enforce a node budget and disclose truncation. Graph search has no LLM/fuzzy/semantic fallback. Augmentation considers up to 100 known source IDs, filters the chosen source kinds, and adds at most four sources within an eight-chunk final budget; it can append existing source chunks but never invent a source. Shared API mentions and a closed state do not establish causality, duplicate issues, affected-release ranges, compatibility, or fixes. The separate six-distinct-source comparison in `evaluation/graph-comparison-results/` found exact-graph Hit@6 of 5/14 original and 3/14 validation evidence cases, versus 14/14 for hybrid on both sets. Exact-seed graph retrieval has low recall on these questions and remains an optional explorer/augmentation path.

The workflow supports reference resolution from at most the last three **user** requests, each bounded to 1,000 characters and within the query budget. Prior assistant answers never become source evidence, and every turn retrieves current snapshot evidence again. This is short explicit context without a persistent checkpoint or cross-session memory. The Streamlit follow-up chat interface is being integrated; UI availability and behavior still require the final app smoke check. Handoffs create a reviewable checklist/draft and never promise that a maintainer was contacted or an issue was posted.

## Source provenance, freshness, and version limits

Use the source inventory and snapshot metadata in `data/` as the authoritative list of included documents. The initial corpus has 65 sources: 38 documentation records, 24 issue reports, two contribution/template records, and one bounded release-note excerpt. Sources are real public documentation and issue reports; the evaluation questions are authored for this project. The evaluation file must not be ingested into the retrieval corpus.

- Documentation is pinned to its repository snapshot. Generated API docstrings have the version recorded by the snapshot script. A current documentation snapshot must not be presented as the behavior of every historical release.
- A public issue's reported version comes from its text or metadata; that is not independently verified compatibility information. A closed issue is not proof that a specific release fixes the reported behavior.
- Issue state, comments, and documentation may change after the snapshot. The displayed upstream link lets the developer inspect current information; the app does not silently refresh the source text.
- Public reports may contain mistaken diagnoses or embedded instructions. They are evidence data, not trusted commands. The app never executes pasted code, installs suggested packages, or posts an issue.
- Public availability does not establish a blanket redistribution license. Preserve source attribution and the source-specific notices included by the snapshot process; review them before redistributing a larger corpus.

Corpus and evaluation SHA-256 hashes are included in every report. The cache fingerprint includes the model identity, chunk IDs, and text hashes. This makes the corpus/index identity auditable for the same snapshot and ensures text changes create a new vector index; model output and timing are not guaranteed deterministic.

### Practical freshness policy

The weekly maintenance pass should check the allowlisted public issues, inspect upstream documentation/release changes, review the diff, and publish a validated replacement snapshot before the oldest source snapshot reaches seven days. Start the pass early enough to include review within that seven-day budget; a version-specific question still requires checking the source's applicability rather than equating a recent snapshot with compatibility.

`scripts/fetch_corpus.py` deliberately pins `DOCS_COMMIT`, `STREAMLIT_COMMIT`, and `API_VERSION`: rerunning it refreshes mutable issue information but does **not** automatically advance documentation to the latest release. Updating those pins is a separate reviewed decision, followed by source validation, index rebuilding, and evaluation; retain old reports with their original hashes.

Immutable cached responses retain their original fetch timestamp and byte hash in a sidecar; legacy cache timestamps use file modification time as a documented approximation. Reading an immutable cache hit makes no upstream request and never resets its age.

If refresh fails or a source cannot be verified, retain the last valid snapshot and disclose its age and coverage. The interface now uses the oldest source snapshot for its age warning, flags ages of seven days or more, and treats invalid timestamps as unknown; the warning is a disclosure, not an automatic refresh or a guarantee that a pinned version is current.

## Generation and prompt

The current measured Nebius configuration uses `Qwen/Qwen3-30B-A3B-Instruct-2507` for evidence directness and synthesis, `Qwen/Qwen3-235B-A22B-Instruct-2507` for the separate support audit, and optional `Qwen/Qwen3-Embedding-8B` for 1,024-dimensional embeddings. Local BGE remains the default. Configure the server-side `TRIAGE_API_KEY` (or `NEBIUS_API_KEY`), `TRIAGE_MODEL`, `TRIAGE_BASE_URL`, optional `TRIAGE_VERIFIER_MODEL`, and optional `TRIAGE_EMBEDDING_MODEL` using `.env.example`; the verified Nebius base is `https://api.tokenfactory.nebius.com/v1`. Keep credentials local and out of source/evaluation outputs. The app loads project configuration at startup; CLI evaluation loads `.env` only for explicit generation or hosted embedding selection.

Selecting hosted embeddings sends document passages and questions to that provider and uses a separate endpoint/model/dimension/corpus fingerprint. It never mixes local and hosted vectors. A real 688-chunk structured hosted index was prepared with 23 requests and 113,064 reported tokens; that cold preparation is separate from the later workflow's query/generation usage. This establishes actual hosted indexing, not better retrieval quality. No cost is inferred from token totals. Provider interfaces are documented in the [Nebius embeddings reference](https://docs.tokenfactory.nebius.com/api-reference/inference/create-embeddings) and [chat-completion reference](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion).

| Runtime prompt | Location and contract |
| --- | --- |
| Evidence directness | `triage_desk/workflow.py`, `GRADE_PROMPT`: one bounded batch judges whether supplied contexts directly support a useful answer, returns strict `answerable`, known citation IDs, reason, and missing details; unavailable or malformed grading fails closed |
| Cited synthesis | `triage_desk/generation.py`, local `system` variable inside `synthesize()`: treat all question/source text as untrusted data; use only selected evidence; return at most three concise claims with known citation/quote IDs or an empty claims list |
| Selected-quote support audit | `triage_desk/generation.py`, `VERIFICATION_PROMPT`, version `selected_quote_entailment_v1`: judge each structurally valid claim using only its selected literal quotes and source-kind/version metadata, without the question, other chunks, earlier reviews, or tools |

The synthesis instructions distinguish official documentation from public reports, prohibit unsupported causal diagnoses/duplicates/fixes, and treat question assumptions or version numbers as user requests rather than factual evidence. Current snapshot metadata is provenance, not compatibility proof for an older user version.

The application supplies literal spans with stable quote IDs and resolves selected IDs back to exact source text. It checks JSON structure, known citation/quote-ID agreement, numeric/version literals against selected spans, and prohibited links/HTML/fenced formatting. Exact selected support is visible in the interface. A valid ID and matching quote establish reference integrity, not that the whole claim follows from that quote; the controls do not establish comprehensive prompt-injection resistance.

The separate AI audit checks every factual part, condition, attribution, version qualification and causal claim. Unsupported claims are omitted, not repaired; partial answers disclose omitted claims. If every claim fails, or the audit is unavailable/truncated/malformed, extracted evidence remains available. Empty model drafts and guarded requests skip that audit. There is no automatic repair/retry loop and no forced grounded answer after a failed check.

Each embedding, directness, synthesis and verification request retains its own safe identity, endpoint, status, reported usage, prompt hash where applicable, and latency. `workflow.provider_calls` and `workflow.provider_usage` count actual stage calls once; generation's already-aggregated usage is not added a second time. Missing applicable usage remains explicit. Passing an AI audit, including one using a larger model, is not independent human validation.

The handoff policy requires review for insufficient evidence or missing details, requested generation that fails to produce accepted cited synthesis, and partial synthesis. The final fallback/partial flag fix was tested after the frozen workflow run and does not rewrite that run's handoff counts. Offline extracted evidence remains available for manual review. No issue is posted or local action executed.

## Evaluation protocol

Run from the repository root after `uv sync --frozen`. Use a new output directory for every rerun so frozen reports remain untouched.

```bash
# Historical core comparison; this does not exercise the expanded workflow.
uv run python -m evaluation.run --modes keyword semantic hybrid --top-k 6 --output-dir evaluation/core-rerun

# Actual graph workflow, local keyword evidence only; no provider or model download.
uv run python -m evaluation.workflow --questions evaluation/questions.json --mode keyword --output-dir evaluation/workflow-local-rerun

# Local dense chunk/reranker comparison over both existing sets.
uv run python -m evaluation.retrieval_lab --questions evaluation/questions.json --questions evaluation/validation-questions.json --output-dir evaluation/retrieval-lab-rerun

# Six-distinct-source hybrid versus exact-seed graph comparison, no generation.
uv run python -m evaluation.graph_comparison --questions evaluation/questions.json --questions evaluation/validation-questions.json --output-dir evaluation/graph-comparison-rerun

# Explicit hosted embeddings + complete live generated workflow.
uv run python -m evaluation.workflow --questions evaluation/validation-questions.json --embedding-provider nebius --rerank --graph --include-generation --output-dir evaluation/workflow-live-rerun
```

Dense/reranker paths need real cached models or an initial download; missing models never silently choose a different retrieval mode. The final command uses configured credits, can incur separate cold document-index preparation, and intentionally loads local `.env`. Hosted embedding selection loads configuration independently of generation. Importing the evaluation module or selecting the default local evidence path does not read `.env`.

The workflow runner invokes the graph once per case without a hidden warmup or second raw search. It scores final selected evidence after grading/core relevance checks, not the first-stage candidate pool. Failed or negative grading may leave candidates visible for manual inspection, but those candidates do not earn post-grade gold hits. Reference integrity still checks all returned references against the actual lab chunk IDs and corpus. Whole-run timing includes first-use work when it occurs; local retrieval-only experiments time warm queries after separately recorded initialization.

| Measurement | Meaning and denominator |
| --- | --- |
| Retrieval-only Hit@6 | At least one labeled source family among the six returned chunks on completed expected-evidence cases; failed-case counts are also explicit |
| Final workflow source-family hit | Same expected-family rule on the variable final selected set, within an eight-chunk graph budget; not a fixed-six raw-retrieval score |
| Source-family recall | Fraction of labeled required source families found; duplicate chunks do not inflate coverage |
| Gold-source precision | Fraction of distinct selected sources in incomplete gold families; a conservative source-selection proxy, not passage relevance/RAGAS |
| Reference integrity | Known source/URL/chunk IDs and literal quote existence; not claim entailment |
| Expected behavior/routing | Evidence, missing details, or abstention as specified by each diagnostic; separate from generated quality |
| Generated coverage | Accepted generated answers over all scheduled expected-evidence cases; fallback/partial/abstention counts remain visible |
| Full backend p50/p95 | Complete per-case workflow duration, with answerable-case and all-case denominators distinguished; excludes browser transport/rendering |
| Human faithfulness/relevance/entailment | Manual rubric fields, left blank until a person reviews actual generated claims/answers |
| Separate AI quality review | Explicit AI judgments on accepted generated answers and exact selected support, hash-bound to the frozen run; not human accuracy |

The original and separately authored validation files each contain 20 AI-authored cases: 14 expected-evidence and six boundary cases. They are public diagnostic/regression sets, not an external blinded benchmark. Reusing them after development does not make them fresh unseen validation; further tuning needs new cases and human review.

### Manual review rubric and targets

Populate the JSON's `manual_review` fields for each actual generated answer. Score each dimension 0, 1, or 2:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Faithfulness | Invented or contradicted material; unsupported diagnosis or fix | Mostly grounded with a material unsupported inference or missing qualification | Every factual statement is supported and correctly qualified |
| Answer relevance | Does not answer the question or falsely claims knowledge of the user's environment | Partly addresses the request; misses a useful distinction or needed clarification | Directly answers the supported portion or appropriately requests missing details |
| Citation entailment | Citations are absent or do not support the statements | Some claims have incomplete or weak support | Each factual claim is linked to evidence that supports it |

Proposed acceptance targets are **goals, not results**: claim faithfulness at least **95%** on a fresh held-out set; expected-source Hit@6 at least 0.85; reference integrity 1.0; all adversarial/private-data cases abstain; and warm **end-to-end p95 at most 20 seconds** from submission to visible answer/fallback. The prior diagnostic targets of average human faithfulness/relevance at least 1.8/2 and warm retrieval p95 under two seconds remain separate measures; ordinal ratings must not be converted into a faithfulness percentage, and retrieval time must not substitute for full-pipeline latency.

For the percentage target, have a reviewer label each factual generated claim fully supported or not, checking attribution, version scope, and all clauses of compound statements against its selected evidence. Report the claim numerator/denominator, generated-answer coverage, rejected outputs, and abstentions alongside the score; no held-out human-reviewed 95% result has been established by the existing diagnostic.

If a source supplies only a related report, a good answer says so. A response that confidently labels the user's issue a duplicate or promises a fix fails faithfulness even if it cites a genuine issue number.

### Earlier quote-ID Nebius run and limitations

The earlier quote-ID hybrid run completed all 20 diagnostic cases. Of 14 provider calls, 13 responses passed the application’s structural checks and one safely fell back because a numeric/version detail was absent from its selected source spans. The remaining six cases were handled before generation. The 14 calls reported 29,633 prompt tokens and 2,802 completion tokens (32,435 total); provider latency was 6.04 seconds at the median and 9.40 seconds at p95. This run’s token total excludes other setup and iteration calls, and is not a dollar-cost estimate.

The initial generation run is preserved in `evaluation/nebius-baseline/`. Literal-quote copying caused four rejected responses, so the application switched to preassigned quote IDs and stricter support instructions. That revised run is saved in `evaluation/nebius-results/`, including an untouched `raw-report.json`, separate AI annotations, and reviewable answers. This was iteration on the same public diagnostic set; it does not demonstrate improvement on unseen questions.

Some structurally accepted answers still contain weak citation selection or unsupported wording. Examples include an issue observation cited only to documentation, a callback claim supported by a non-callback example, and a persistence recommendation phrased as a requirement. The AI assessment records these gaps instead of treating acceptance as factual correctness. Human ratings and the held-out quality targets remain unverified. At that stage, the next quality steps were human review and a separately authored question set; the completed AI-authored validation iteration is recorded below, while human review remains outstanding.

The separate AI reviewer judged 23 of 33 generated claims fully supported by their selected quotations; ten had partial or missing support. Its mean ordinal ratings were faithfulness **1.62/2**, answer relevance **1.85/2**, and citation entailment **1.46/2**. These are AI judgments, not measured accuracy or human validation. The reviewer also helped author the diagnostic questions, so the review was not blinded. Annotations are bound to the unchanged raw report and exact answers with hashes in `evaluation/nebius-results/ai-review.json`.

The frozen run also records the full backend `analyze` duration, which includes retrieval, generation, validation, and draft preparation. Across the 14 provider-attempt cases, the observed backend median was **6.06 seconds**, p95 **9.41 seconds**, and maximum **9.51 seconds**; across all 20 cases, backend p95 was **9.37 seconds**, with fast guard cases included. These figures are computed from `analysis_ms` in the unchanged raw report, include provider-request time but exclude browser rendering and browser-to-server transport, and do not establish the proposed submission-to-visible-answer SLA.

### Separately authored validation: support-audit iteration, before the four-guide expansion

The completed next run used a new 20-case set written by a different AI agent: 14 answerable questions and six boundary cases. It evaluated the two-call core synthesis/support-audit pipeline **before** the new LangGraph relevance grader, retrieval lab, FlashRank, source graph, follow-up UI, hosted embedding option, or separate 235B verifier. Its results must not be attributed to those later features or compared with the older diagnostic as a controlled quality improvement.

The frozen raw report is `evaluation/validation-results/raw-report.json`, SHA-256 **`20b1f4333d98fbc57bb5f5ba89000dd45054ebb8cd26bc514ff19cd554910766`**. It records 18 pipeline attempts and 30 provider calls: 18 drafts and 12 support audits. Twelve answers were accepted, including four labeled partial answers; three structural failures fell back to evidence, three model outputs abstained, and two cases stopped at guards. Eleven of the 14 answerable cases produced a generated answer; the twelfth accepted answer came from boundary case `VB01`, which should have requested details. Expected behavior passed on **18/20** cases: `VB01` generated a generic answer to an underspecified personal diagnosis, and `VB02` safely abstained instead of requesting the expected local details.

An AI reviewer who did not author this new set or implement the inline verifier inspected all 12 accepted answers and 30 retained claims against selected quotes and expected answer checks. The reviewer judged **26/30** claims fully supported, with mean ordinal scores of faithfulness **1.75/2**, answer relevance **1.42/2**, and citation entailment **1.75/2**. This is an **AI-supported-claim fraction**, not human-validated accuracy, and it is below the 95% claim-faithfulness target. The reviewer did not treat the inline verifier's verdict as ground truth. This remained an AI-team review with visible questions/evidence, not an external blinded human study; human-review fields remain blank.

The review found unsupported HTML-rendering guidance, an unsupported outdated/mismatched-package causal characterization, and two public-issue observations generalized without attribution in `VB01`. It also found material answer omissions after support filtering in `V01`, `V04`, `V06`, `V08`, and `V11`. Removing an unsupported claim can improve grounding while leaving an incomplete answer, so partial-answer notices do not substitute for relevance or coverage. All eight nongenerated outcomes stay in coverage/outcome counts and are excluded from generated-quality means, rather than being assigned perfect faithfulness.

The annotations in `evaluation/validation-results/ai-review.json` bind to the unchanged raw-report SHA-256, each answer, each retained statement, and its selected-support list. Claim indices are zero-based and each claim has a binary fully-supported judgment with concrete gap notes; ordinal ratings are not converted into percentages. This preserves the exact evidence reviewed rather than attaching ratings to a later rerun.

Observed answerable-case backend p95 was **24.90 seconds**, which already exceeds the **20-second** warm submit-to-visible-answer target before browser transport/rendering is included. It is an unmet goal, not an enforced ceiling or an achieved SLA. This historical result remains preserved; the expanded workflow and its separate latency/coverage measurements are recorded below.

### Current expanded workflow: recorded regression and separate AI review

The completed expanded regression ran once over the existing 20-case validation set with structured hybrid retrieval, hosted Qwen embeddings, FlashRank reranking, graph augmentation, 30B directness/synthesis and a 235B selected-quote support audit. It is a new configuration on previously used questions, not an unseen test or a controlled estimate of any one feature's benefit.

| Observed measure | Result and denominator |
| --- | --- |
| Completed/error cases | 20 completed, zero errors |
| Expected routing/behavior | 19/20 overall; all six boundary cases passed |
| Final selected evidence source-family hit | 13/14 expected-evidence cases |
| Generated answer coverage | 10/14 expected-evidence cases; four accepted answers were partial |
| Answer-pipeline outcomes | 13 synthesis attempts, three structural/format fallbacks, one expected-evidence case clarified |
| Separate AI selected-quote support | 19/22 retained factual claims fully supported (86.4% AI judgment, not measured accuracy) |
| Mean AI answer ratings | Faithfulness 1.70/2 and relevance 1.10/2 over 10 accepted generated answers |
| Answerable-case backend p95 | 24.079 seconds; the 20-second target is not met before browser transport/rendering |
| Actual workflow provider use | 55 calls and 62,215 reported total tokens; excludes separate 113,064-token cold hosted indexing |

The 55 calls are 16 query embedding requests, 16 evidence-directness grades, 13 synthesis requests and 10 support audits. Three paid drafts safely fell back after citation/quote-ID/format failures; the inline AI audit retained 22 claims and omitted five. Missing or rejected generated answers remain in outcome/coverage denominators and do not receive perfect quality scores. The larger verifier did not establish human-level correctness: three retained claims still lacked full selected-quote support according to the separate AI reviewer, and relevance remains weak after omissions. Human quality fields remain blank and the 95% claim-faithfulness target is unverified.

The untouched raw report is `evaluation/workflow-results/raw-report.json`, SHA-256 **`827db28bfe99f0d42eecd84bcaebf70c4ba1bab21a3f206a99e20c1a5d8dd6cb`**. `evaluation/workflow-results/ai-review.json` binds the separate AI assessment to its answers and selected support; `report.json`/`report.html` attach that assessment and explain derived metadata corrections. A later handoff-flag fix requires review for fallback/partial generated outputs even when evidence exists. It was regression-tested separately; the frozen live report still records the policy loaded for that run and has not been presented as a new live evaluation.

### Retrieval and relationship experiments

The local BGE comparison in `evaluation/retrieval-lab-results/` used the same source labels and top-six/18-candidate settings for fixed hybrid, structured hybrid, and structured plus FlashRank. Fixed creates 378 chunks with 180-word windows and 35-word overlap; structured creates 688 chunks, including 33 explicit oversized-block splits. On the 14 validation evidence cases, gold-source precision was **0.402 / 0.410 / 0.310**, family recall **0.964 / 0.964 / 0.893**, and warm retrieval p95 **20.49 / 17.50 / 891.95 ms**, respectively. All three hit at least one expected family for every evidence case in both sets. Reranking showed no consistent advantage and added latency, so it remains opt-in. Boundary cases contribute retrieval timing, not an answer-behavior score, in this retrieval-only experiment.

The standalone comparison in `evaluation/graph-comparison-results/` gives both methods six distinct source slots. Exact graph retrieval hit expected families for **5/14** original and **3/14** validation evidence cases, versus **14/14** hybrid on both sets. Many conceptual questions have no exact entity seed. This low standalone recall does not negate inspectable graph relationships, but it does not support replacing hybrid retrieval or claiming augmentation improves answer quality. Both graph augmentation and FlashRank remain optional.

### Working interface and user-reviewed outcomes

The six implemented top-level tabs are **Investigate**, **Chat**, **Knowledge base**, **Relationships**, **Retrieval lab**, and **Evaluation**. Investigate collects reporter details, displays exact support/source authority/version scope and stage traces, and downloads drafts/JSON. Chat runs new eligible investigations while using at most the last three USER requests only to resolve explicit references; previous assistant text is never factual evidence. Relationships explores the source-fact graph, while the lab and evaluation views expose configuration differences and recorded limits.

A user can save **Resolved from the evidence**, **Needs follow-up**, or **Human handoff needed** for a run. `triage_desk/reviews.py` writes only run ID, outcome, first-contact flag and timestamp to local `.cache/user-outcomes.jsonl`, with the latest label per reviewed run used by the summary. User-reported first-contact resolution is computed only over labeled first contacts. No real helpdesk study, unreviewed-user resolution rate, or automatic resolution claim follows from this capability.

### Rebuildable source package

```bash
uv run python scripts/fetch_corpus.py --check
uv run python scripts/package_submission.py --output artifacts/engineering-issue-desk-week2.zip
```

The snapshot check validates source counts/hashes offline. The package builder collects allowlisted source assets with a per-file SHA-256 manifest, excludes credentials/environments/caches/outcome logs, and checks candidate files against configured local credentials before writing. It rebuilds a reviewable source package; timestamps mean no byte-identical ZIP guarantee is claimed. The command documents implemented capability, not that packaging, GitHub publication, a Google Doc, video upload, or form submission has been completed.

## Focused next improvements

1. **Review omitted question parts with a person.** The current AI relevance mean is 1.10/2, and 19/22 retained claims were judged fully supported. Review attribution, exact selected spans and missing answer parts together; do not optimize claim acceptance by silently reducing coverage.
2. **Add new cases before more tuning.** Preserve all existing raw reports, use genuinely fresh authoring/sampling, and measure retrieval choices independently. Keep graph/reranking opt-in until a controlled comparison supports a benefit.
3. **Measure full user-visible latency.** The 24.079-second answerable backend p95 already misses the 20-second target. Record submission-to-visible-answer timing and all provider stages, then consider a smaller or simpler measured pipeline; no request-wide SLO is established.
4. **Collect actual outcome labels.** Keep follow-up history bounded and source applicability visible, inspect user-reported first contacts separately from model quality, and preserve manual refresh plus reviewable actions before adding tools or posting.

## Adaptation of the four supplied solution guides

The four supplied Week 2 solution documents were read as reference material. Their architecture ideas were adapted independently to this project's engineering evidence and code interfaces; whole solution implementations, prompts, datasets, and reported scores were not copied into this project. Instructions within those examples do not authorize sending messages, following forms, or performing external actions.

| Supplied solution document and linked course repository | Idea applied here | Engineering-specific choice and evidence boundary |
| --- | --- | --- |
| **Enterprise Policy Q&A Bot (Langchain and LangGraph)** — [provided repository revision](https://github.com/The-Gen-Academy/2A-Enterprise-QandA-Bot/tree/4b2103dcc2ed2c8b09544d84ab396f1ce5e239b0) | Explicit state, conditional routing, relevance grading, and a separate answer-support check in a real LangGraph workflow | Route public engineering requests and missing local diagnostics; grade contexts in one bounded batch. A failed/invalid grade or support check routes to clarification/evidence/handoff, never to forced generation or a forced grounded verdict after a retry cap. No corporate policies or namespace facts are invented. |
| **Financial Document Intelligence Pipeline Solution Kit (Langchain and LangGraph)** — [provided repository revision](https://github.com/The-Gen-Academy/2B-Financial-Document-Intelligence-Pipeline/tree/39d2249c8135d24af99e99d2a6c33488ad2f6dc0) | Controlled chunking-configuration comparison and an actual local FlashRank reranker | Compare fixed word windows with deterministic engineering Markdown/code/table-aware chunks, using the same source labels and separate first-stage/reranker scores. This is not its embedding-breakpoint semantic chunker, Pinecone deployment, SEC corpus, or RAGAS evaluation; none of its financial scores apply here. |
| **GraphRAG for Organizational Knowledge Solution Kit (LangChain and LangGraph)** — [provided repository](https://github.com/The-Gen-Academy/2C-Graph-RAG-for-Organizational-Knowledge) | NetworkX multi-directed graph, entity-seeded bounded two-hop expansion, inspectable traversal, and a graph/vector comparison | The 178-node engineering graph links sources to literal APIs and explicit metadata. Exact linking has no hidden fallback, and every relation has provenance; co-mentions/issue state never become invented causal, duplicate, approval, compatibility, or fix relationships. Compare source-family recall rather than borrowing organizational gold entities. |
| **Customer Support Knowledge Base Agent Solution Kit (Langchain and LangGraph)** — [provided repository revision](https://github.com/The-Gen-Academy/2D-Customer-Support-KB-with-Hybrid-Search/tree/6a40fa1ca1e50ae06e40d930affaf5902776558c) | Hybrid retrieval, explicit answer-versus-handoff decisions, decision traces, and bounded user follow-ups | Use a concrete reporter checklist and source-review draft without promising contact by a human agent. Similarity/reranker scores are not calibrated confidence, its numeric cutoffs are not imported, and user outcome labels are implemented, but first-contact resolution remains a user-reported labeled-first-contact measure with no real helpdesk outcome study. |

Primary technical references for the implemented interfaces are the [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), LangChain Core [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) and [`RunnableLambda`](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda), [NetworkX `MultiDiGraph`](https://networkx.org/documentation/stable/reference/classes/multidigraph.html), the [FlashRank maintainer repository](https://github.com/PrithivirajDamodaran/FlashRank), and [FastEmbed's maintainer repository](https://github.com/qdrant/fastembed). These explain the actual libraries; the repository's frozen reports and tests supply this project's observations.

## Five-minute demo sequence

1. **0:00–0:35 — User, AI coding tools, and sources.** State the engineering support task, briefly explain how AI coding tools helped build/test it and what was validated, and show the source snapshot/version context. Distinguish official documentation from public reports.
2. **0:35–1:25 — A documentation question.** Ask when to choose `st.cache_data` versus `st.cache_resource`. Open a citation and verify an exact passage. Show that evidence-only mode is available without provider credentials.
3. **1:25–2:15 — A real issue report.** Ask about custom-class caching after the reported 1.43-to-1.44 upgrade or checkbox state after `st.switch_page`. Filter to issues. Explain why similar symptoms and closed status do not establish a duplicate or fix.
4. **2:15–2:55 — Boundaries and draft.** Ask a vague question such as “My app does not work. What is the fix?” Show the request for details. Inspect the issue draft and its missing-environment checklist; nothing is posted automatically.
5. **2:55–3:40 — Workflow and retrieval choices.** Inspect one stage trace and the same query's retrieval configuration. Show fixed/structured chunks and actual local reranking if its model is prepared; inspect a small `st.cache_data` graph view and explain that an edge is a source fact, not a diagnosis. Use the Chat tab for an explicit follow-up and show that prior assistant text is not evidence.
6. **3:40–4:30 — Evaluation and a failure.** Open a frozen HTML report, state the exact pipeline/dataset measured, and show an unsupported or omitted answer detail. Explain the current 19/22 AI-supported claims, 10/14 answerable generated coverage, AI relevance 1.10/2, blank human fields, and 24.079-second answerable backend p95 versus the unmet 20-second target. Identify the frozen run and later handoff-only fix.
7. **4:30–5:00 — Generated answer and next steps.** Show a prepared cited answer and its exact support spans, partial-answer/fallback status, and reviewable draft. Distinguish structural checks, the inline AI audit, separate AI assessment, and any human review. Explain how the next week can add read-only engineering tools to this bounded workflow.

## Course expansion plan

This roadmap follows the published course themes reviewed for planning. Check each actual weekly handout before treating a feature as a required deliverable.

| Week/theme | Coherent extension of this project |
| --- | --- |
| Week 2: RAG | Source snapshots, fixed/structured and reranker experiments, source-fact graph, bounded LangGraph investigation, citations, and evaluation |
| Week 3: agents, tools, MCP | Extend the existing workflow with read-only tools for repository/environment metadata, issue search, and reproduction-checklist preparation; a separate draft action remains reviewable before posting |
| Week 4: evaluation and observability | Held-out cases, stage traces, citation/answer review, failure taxonomy, retrieval/generation latency and cost monitoring |
| Week 5: local models and specialization | Evaluate a local response model and fine-tune issue routing or component classification using reviewed historical labels; compare against current heuristic and prompted baselines |
| Week 6: guardrails and production | Permission boundaries, secret/PII handling, prompt-injection tests, tenant/source isolation where relevant, rate/cost controls, and deployment monitoring |

For Week 5, historical issue labels are noisy, imbalanced supervision rather than unquestionable truth. Remove labels and boilerplate from model inputs, split by time and issue family, and keep related issues out of both train and test. Do not fine-tune changing documentation facts into a model when retrieval can supply the current source snapshot.

## Remaining submission work

| Required deliverable | Readiness evidenced by this audit | Remaining action |
| --- | --- | --- |
| Working RAG application | Local app and real model runs are present; 200 tests, the 65-source integrity check, and real browser investigation/chat checks passed | Use the prepared demo environment and disclose the observed quality limits |
| Project documentation as a Google Doc | A six-page write-up was authored, visually verified, imported as a native private Google Doc, read back, and checked on all six exported pages | Configure intended reviewer access when ready to submit |
| Video of five minutes or less | The timed script above covers the app, sources, evaluation, and AI coding-tool use | Record the final result live, verify duration and access, and obtain the submission link |
| GitHub project assets | Code, tests, write-up, source provenance, and frozen evaluation artifacts are in the private project repository linked below | Configure intended reviewer access when ready to submit |
| Evaluation report document | The write-up includes current measured results, sample sizes, fallback counts, limitations, and AI-versus-human distinctions; detailed HTML/JSON reports are in the repository | Complete human review when available; retain the existing blank human fields until then |
| Course form submission | The handout specifies the form and deadline | Submit the requested document, video, and GitHub links through the course form after checking access |

The supplied handout's form is [the Week 2 submission form](https://forms.gle/Cxad8AnPHBB4sNzw5). Its Builder of the Week deadline is **Sunday, October 18, 2026, 11:59 PM Pacific**; it explicitly permits later submissions through the same form for course completion, without Builder of the Week consideration.

The [project repository](https://github.com/naresh25ganji/engineering-issue-desk) and [native Google Docs write-up](https://docs.google.com/document/d/1C1OZmA_590xQs1ItVmqQq2ljM9BM2dvxlKilXt9vH44) are private. The verified local Word copy is `docs/Engineering Issue Desk Week 2 Writeup.docx`. A source ZIP and per-file hash manifest are prepared locally under `artifacts/`; the packaging command above rebuilds it. No video, reviewer-sharing change, or course form submission has been completed.

Human-review fields remain blank until a person completes them; AI ratings are not human validation.
