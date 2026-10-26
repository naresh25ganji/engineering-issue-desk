# Week 3 submission handoff

Engineering Issue Desk extends the existing Streamlit evidence workspace with a Deep Agents investigation loop, Fireworks tool calling, Keenable research, specialist delegation, durable LangGraph checkpoints, and human review. It supports engineering issue investigation and code review from intake through a saved, cited report.

## October 25 final refresh

The refreshed Word writeup and demo guide accompany the [architecture](ARCHITECTURE.md), [design decisions](DESIGN.md), [operations](OPERATIONS.md), [validation record](VALIDATION.md), and [readiness matrix](SUBMISSION_READINESS.md). Detailed [system](week3-system-architecture.svg) and [sequence](week3-investigation-sequence.svg) diagrams are available through the app's eighth **Project guide** tab. Repository and earlier-review context are default-off immutable case options.

The final stretch options add on-demand maintained playbooks, optional local literal review guidance, real bounded Bandit checks, and a separate approved-case public PR summary preview/confirmation flow. The handout has no official extra-credit checklist or scores. The [verified live evaluation](../evaluation/week3-verified-results/report.html) reached human review for 3/3 cases; only the nonempty freshness report has an applicable exact-citation check. Human quality remains unscored. All earlier failed and successful exports remain separate observations.

After the knowledge-refresh follow-up, software checks passed 615 tests. The [recorded validation](../evaluation/week3-verified-results/validation.json) and [qualitative inspection](../evaluation/week3-verified-results/engineering-review.md) distinguish controlled evidence, live tool use, and unmeasured human outcomes. The separate extension exercise observed playbook/Bandit use; history lookup was not invoked live. No GitHub POST was performed.

The [October 25 corpus refresh](../evaluation/corpus-refresh-20261025/refresh-summary.json) acquired all 65 sources again, keeping commit pins and API version 1.66.0. The sidebar now reports the oldest October 25 fetch, with no stale warning. Frozen code-sample results still refer to the original evaluation snapshot.

## Required deliverables

| Handout requirement | Prepared artifact | Remaining submission work |
| --- | --- | --- |
| Google Doc covering project overview, datasets, coding prompts, iterations, and learnings | [Final private native Doc](https://docs.google.com/document/d/1ImRAJn28t61YaaqlSdyZIDpqb_UEK_oRYHHsDFbrWdI), imported from final Word source; readback and all nine converted pages verified | Confirm intended reviewer access to the private document |
| Video of five minutes or less | `docs/Engineering Issue Desk Week 3 Demo Guide.docx`: narration and screen cues targeting 4:45 | Record, check duration/audio, upload to Google Drive, and confirm reviewer access to the video link |
| GitHub code link | Public [implementation commit `7400304`](https://github.com/naresh25ganji/engineering-issue-desk/commit/740030480909d1e38b4697dda6810dde5821b52d) in the [repository](https://github.com/naresh25ganji/engineering-issue-desk) | [GitHub CI](https://github.com/naresh25ganji/engineering-issue-desk/actions/runs/40247965771) passed; use the public repository link in the form |
| Supporting project assets | Latest `evaluation/week3-verified-results/`, preserved historical reports, detailed diagrams, source ZIP manifest | Final package includes the refreshed documents, three playbooks, and per-file hashes; include required links in the form |

The handout lists October 25, 2026 for Builder of the Week eligibility and November 11, 2026 for certification. The course form is https://forms.gle/waBYFkkf5xxmcouV9. Neither a form submission nor a recorded video is claimed here. The live form was read without submitting: all five fields are required—Full Name, Email Address, GitHub Repository Link, Video Demo Drive Link, and Project Documentation Drive Link. Upload the recorded video to Drive and use the final native Doc link for documentation.

The [initial native Doc](https://docs.google.com/document/d/1gOxafWnF3u6OI5Rp1OtHMQyF0VYTPsBnm_NzWH3Dzus) remains a historical archive of the first version, not the final submission link.

## Agent framework mapping

| Framework field | Implemented behavior |
| --- | --- |
| Goal | Investigate a reported Streamlit behavior and produce cited findings, missing information, and proposed checks |
| User surface | Deep Investigation tab and `scripts/investigate_deep.py` CLI |
| Ordered workflow | Investigate → deterministic citation validation → independent AI evidence review → human review → approve/reject or revise |
| Available tools | Five domain reads, optional repository reads/history, maintained playbooks, and fixed Bandit checks on registered code |
| Memory/state | SQLite cases, evidence/events, parent/inner checkpoints, and opt-in literal approved/rejected prose recall; no Mem0 or semantic memory |
| Boundaries | Virtual scratch files, no host shell or supplied-program execution, source allowlists, attribution, shared budgets; publication requires a separate confirmed approved-case preview |
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

The implementation uses Fireworks and Keenable. Review history is local, bounded literal recall; Mem0 and other unconfigured providers are outside the implemented scope.
