# Week 3 submission handoff

Engineering Issue Desk extends the existing Streamlit evidence workspace with a Deep Agents investigation loop, Fireworks tool calling, Keenable research, specialist delegation, durable LangGraph checkpoints, and human review. The Week 3 handout permits an independent use case; this project adapts the code-review examples to engineering issue investigation.

The supplied solution documents informed design choices. Instructions in those documents do not authorize external actions, messaging, account changes, or form submission.

## October 23 technical extension

The initial Word/Google Docs write-up remains a record of the first Week 3 version. The [architecture](ARCHITECTURE.md), [design decisions](DESIGN.md), [operations](OPERATIONS.md), and [validation record](VALIDATION.md) document the later extension, with detailed [system](week3-system-architecture.svg) and [sequence](week3-investigation-sequence.svg) diagrams. The app's eighth **Project guide** tab displays and downloads these files.

The extension adds explicit per-case repository context, bounded list/search/read tools, immutable code evidence, safe review failure codes, and harmless reviewer-format normalization. Current verification passed 473 tests. The [new live evaluation](../evaluation/week3-context-results/report.html) reached human review for 3/3 cases, while the original 2/3 result is preserved. Handoff completion is not measured accuracy; [inspection notes](../evaluation/week3-context-results/engineering-review.md) retain the remaining quality issues. A separate saved-review retry and provider probe are recorded in the validation document.

## Required deliverables

| Handout requirement | Prepared artifact | Remaining submission work |
| --- | --- | --- |
| Google Doc covering project overview, datasets, coding prompts, iterations, and learnings | [Native Google Doc](https://docs.google.com/document/d/1gOxafWnF3u6OI5Rp1OtHMQyF0VYTPsBnm_NzWH3Dzus), converted from the visually verified `docs/Engineering Issue Desk Week 3 Writeup.docx` | Confirm the intended reviewers can open the private document |
| Video of five minutes or less | `docs/Engineering Issue Desk Week 3 Demo Guide.docx`: narration and screen cues targeting 4:45 | Record, check duration/audio, and obtain a video link |
| GitHub code link | Existing repository: https://github.com/naresh25ganji/engineering-issue-desk | Commit and push the Week 3 changes, then confirm reviewer access |
| Supporting project assets | Live results in `evaluation/week3-results/`, diagram `docs/week3-architecture.svg`, source ZIP built by `scripts/package_submission.py` | Include the required links in the course form |

The handout lists October 25, 2026 for Builder of the Week eligibility and November 11, 2026 for certification. The course form is https://forms.gle/waBYFkkf5xxmcouV9. Neither a form submission nor a recorded video is claimed here.

## Agent framework mapping

| Framework field | Implemented behavior |
| --- | --- |
| Goal | Investigate a reported Streamlit behavior and produce cited findings, missing information, and proposed checks |
| User surface | Deep Investigation tab and `scripts/investigate_deep.py` CLI |
| Ordered workflow | Investigate → deterministic citation validation → independent AI evidence review → human review → approve/reject or revise |
| Available tools | Bounded local knowledge search, Keenable search/page fetch, public Streamlit issue reader, and immutable PR patch reader |
| Memory/state | SQLite cases, immutable evidence, event records, and parent/inner LangGraph checkpoints; Mem0 and cross-case semantic recall are future work |
| Boundaries | Virtual agent filesystem, no shell/code execution or GitHub publishing, public-source allowlists, source attribution, and shared execution budgets |
| Human involvement | Inspect quotes and causality; approve, reject, or request up to two revisions with notes |
| Failure behavior | Sanitized tool/provider errors, retained evidence, saved retries, bounded repair threads, and no automatic approval |
| Proposed success criterion | A human judges the handoff usable within ten minutes in eight of ten representative cases; this target remains unmeasured |

## Evidence and evaluation

The executable dataset `evaluation/week3-cases.json` selects three real repository samples: outcome labels/concurrency, freshness timestamps, and hosted embedding response/cache handling. The runner records file and excerpt hashes, actual provider/tool calls, reported usage, retained reports, citation integrity, AI filtering, and persisted human-review handoffs. It never runs the supplied code or approves a report. Human quality fields remain blank.

Run with a **new** output directory to preserve prior observations:

```bash
uv run python -m evaluation.agent_workflow --live --output-dir evaluation/week3-results-rerun
```

Without `--live`, the same command validates fixtures and writes an unscored inventory without provider calls. Operational handoff completion, literal citation integrity, AI judgments, and human accuracy measure different things. A matching quote does not prove entailment; the separate AI reviewer can also miss unsupported advice.

An exploratory case with valid citations recommended suppressing uploader callbacks for a checkbox-triggered rerun. Manual inspection flagged that causal mismatch. An added AI review removed some content but still retained the inapplicable recommendation. The reviewer instructions were strengthened, and the limitation remains documented. That case is unapproved; it is not a verified fix.

## Reproduce and package

```bash
uv sync --frozen
uv run pytest -q
uv run python scripts/fetch_corpus.py --check
uv run python scripts/investigate_deep.py probe
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8503
uv run python scripts/package_submission.py --output artifacts/engineering-issue-desk-week3.zip
```

The probe and live evaluation use configured provider credits. `.env` contains local credentials and is excluded from Git and the source ZIP. The archive also excludes environments, private saved cases, checkpoints, caches, and outcome logs, and includes a per-file SHA-256 manifest. Opening the source ZIP does not restore private live case state; the exported evaluation reports preserve bounded shareable observations.

The supplied guides also discuss Mem0 and alternative providers. This implementation uses the confirmed Fireworks and Keenable access. It does not claim those other services or optional features were integrated.
