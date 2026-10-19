# Engineering Issue Desk

Investigate Streamlit behavior and prepare a reviewable issue draft from a bounded snapshot of official documentation, release notes, contribution guidance, and public issue reports. Inspect exact supporting passages, identify missing local details, and keep source authority and version scope visible.

This independent course project does not establish a private application's root cause, prove a duplicate, promise a fix, inspect a developer's machine, execute pasted commands, or post issues.

For the demo walkthrough, open the standalone [architecture diagram](docs/architecture.svg) or its [PNG copy](docs/architecture.png). Read top to bottom: knowledge-base preparation, the LangGraph investigation workflow, and the generated-answer checks inside Analyze. Both files work offline.

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://localhost:8502. The 65-source snapshot is included. Keyword evidence mode needs no model or provider key. Semantic/hybrid retrieval uses real local FastEmbed `BAAI/bge-small-en-v1.5` embeddings by default; prepare model files and indexes before a fully offline demonstration. First-use downloads/indexing can take much longer than a warm query. A failed model never silently selects a different retrieval mode.

The guided workflow defaults to structured passages. Fixed/structured indexes and optional hosted indexes are fingerprinted and cached separately in ignored `.cache/`; the bundled historical base-pipeline index does not replace those separate indexes. Changed corpus/model identities invalidate stale vectors.

## Six working views

| View | What it provides |
| --- | --- |
| Investigate | Structured reporter details, cited or extracted evidence, related public reports, missing details, workflow trace, and downloadable draft/JSON |
| Chat | Follow-up investigations with at most the last three user requests used only to resolve explicit references; prior assistant answers are never source evidence |
| Knowledge base | Inspect the included source text, upstream URL, snapshot, source type, and version scope |
| Relationships | Explore a provenance-bearing graph of literal API mentions and explicit reported versions, kinds, and issue states |
| Retrieval lab | Compare fixed and structured passages and optional real FlashRank reranking for the same query |
| Evaluation | Inspect recorded experiment reports and separately labeled user outcome summaries |

The actual LangGraph path is `prepare → retrieve → grade → analyze → handoff → finish`, with conditional clarification/abstention branches. LangChain Core `Document` and `RunnableLambda` carry evidence and wrap nodes. Each eligible request retrieves once. Guards can request local diagnostics before retrieval; no persistent checkpoint or cross-session memory is implemented.

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

## Course handoff

- [Private project repository](https://github.com/naresh25ganji/engineering-issue-desk)
- [Private Google Docs write-up](https://docs.google.com/document/d/1C1OZmA_590xQs1ItVmqQq2ljM9BM2dvxlKilXt9vH44)
- [Verified six-page Word write-up](docs/Engineering%20Issue%20Desk%20Week%202%20Writeup.docx)
- [Four minute and forty second recording guide](docs/Engineering%20Issue%20Desk%20Video%20Recording%20Guide.docx) with narration, screen cues, and recording controls; also available as a [private Google Doc](https://docs.google.com/document/d/1wPnj_PcB3pk73QN6mVSyr9hAz0TVUiJd076T1sUF408)
- [Latest workflow report and AI review](evaluation/workflow-results/report.html)

Local verification passed 200 tests, the 65-source integrity check, real browser investigation/chat checks, and all six converted Google Docs pages. The video and course form submission remain separate deliverables. The repository and Google Doc are private; reviewer access has not been configured.

## Course expansion

The backend is independent of Streamlit. Continue this project with read-only engineering tools/MCP, stronger evaluation and observability, reviewed issue-routing training data/local model comparisons, and deliberate action/permission controls as later handouts require. See [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md) for requirement mapping, prompt locations, four-guide adaptations, historical runs, and the five-minute demo sequence.

```text
app.py                       Six-view Streamlit workspace
triage_desk/                  Core RAG, workflow, retrieval lab, graph, providers, outcomes
scripts/fetch_corpus.py       Bounded source snapshot and offline validation
scripts/package_submission.py Source ZIP and per-file manifest builder
data/                        Corpus, provenance, licenses, historical base index
evaluation/                  Question sets, CLI runners, frozen reports and AI reviews
tests/                       Behavior, provider-contract and interface checks
docs/                        Assignment mapping and demo guide
```
