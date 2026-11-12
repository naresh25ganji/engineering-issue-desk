# Week 6 submission

This submission uses **Path B**, red-teaming Engineering Issue Desk. The handout's deadline is **Sunday, November 15, 2026, at 11:59 PM Pacific Time**. Testing and documenting findings are required; implementing defense code is optional.

The live submission form was inspected read-only on November 12, 2026. It adds a **required video demo Drive link** and **required GitHub repository link**, and says that all submitted links must be publicly accessible. The handout does not list a video as a required deliverable, so follow the form's required fields when preparing the final submission.

## Deliverables

The final run directory, results, Google Doc link, and verification status will be added here after the live evaluation. An input inventory is not a completed attack run and must not be submitted as observed results.

The evidence layout separates the original live findings, targeted follow-up, and deterministic control checks:

| Artifact | Scope |
| --- | --- |
| `evaluation/week6-baseline/report.html` and `report.json` | Original full live evaluation, retaining observed failures |
| `evaluation/week6-defended/report.html` and `report.json` | Separate targeted live rerun after the credential-request control change; not a repeat of the whole suite |
| `evaluation/week6-defense-controls/report.json` | Direct before-and-after sanitizer comparison with zero model and network calls; not live agent outcomes |
| `outputs/week6/Engineering Issue Desk Week 6 Demo.webm` | Captioned capture of the actual application and saved evidence |
| `outputs/week6/screenshots/` | Actual report and application screenshots used as evidence |

Use the generated final files and their recorded counts when assessing completion. A missing artifact in this layout is not an observed result.

| Handout requirement | Prepared evidence |
| --- | --- |
| Test ARIA or your own project | Production Engineering Issue Desk investigation service, Path B |
| Try several attack families | Frozen Week 6 dataset and the family summary in the final report |
| Copy prompts and observed responses | Full per-turn transcripts in the saved HTML and JSON; selected complete examples in the findings document |
| Score PASS, WARN, or FAIL with reasoning | Per-case score, reason, boundary checks, and separate human-review fields |
| Capture important evidence | Saved report screenshots plus the complete text records |
| Connect findings to defenses | Attack-to-defense mapping naming the control and its enforcement point |
| Submit a short Google Doc | Native Google Doc imported from the visually verified findings DOCX |
| Submit the repository link required by the form | GitHub project repository containing the evaluation implementation and evidence |
| Submit the video Drive link required by the form | Recorded demonstration uploaded to Drive |

## Exact form fields

The form is titled **Mastering Agentic AI: August 2026 Cohort: Week 6 Project Submissions**. Each field below is marked required:

| Field | What to prepare |
| --- | --- |
| Email | Google Forms records the response email address |
| Full Name | User-provided submitter name |
| Email Address | User-provided contact email, separate from the collected response email |
| GitHub Repository Link | Publicly accessible project repository URL |
| Video Demo Drive Link | Publicly accessible Drive URL for the completed demonstration video |
| Project Documentation Drive Link | Publicly accessible Google Doc URL for the findings document |

No form fields were filled and no response was submitted during inspection. Personal field answers must come from the submitter; they are not inferred from repository metadata.

## Reproduction

The evaluation runner is `evaluation/week6.py`; use its `--help` for the final command syntax. Live evaluation must be explicitly selected and requires the project's configured model provider. Use a new output directory to preserve the submitted run. Keep `.env`, API keys, local caches, and unrelated saved user cases out of the submission.

The findings document builder consumes a recorded `report.json`:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 scripts/build_week6_submission.py --report evaluation/week6-baseline/report.json --retest-report evaluation/week6-defended/report.json --guard-report evaluation/week6-defense-controls/report.json
```

Before the first DOCX authoring operation in a session, run the document skill's required operation marker. The builder sanitizes the title for Google Docs. Render the final DOCX with the packaged document renderer, inspect every page, and import that verified file into native Google Docs.

The recording script uses packaged Playwright and the installed Chrome executable. It opens the actual local Streamlit **Evaluation** tab and saved HTML reports, records the rendered pages, adds explanatory captions, and saves a video identity file beside the WebM. It does not start a new investigation:

```bash
~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node scripts/record_week6_demo.mjs --report evaluation/week6-baseline/report.json --retest-report evaluation/week6-defended/report.json
```

The recorder requires Playwright's small FFmpeg support package. If unavailable, install it with the packaged Node runtime and packaged `playwright/cli.js install ffmpeg`. The bundled encoder produces VP8 WebM, without a narration track.

## Final handoff

- Confirm the Google Doc opens and the complete evidence links are accessible to the intended grader.
- Review the findings, especially ambiguous results and any remaining human-review fields.
- Record the required video demonstration using [the recording guide](WEEK6_DEMO.md), upload it to Drive, and verify the submitted link's access.
- Submit the document URL through the [course submission form](https://forms.gle/4oBSZ7nq8Zy5W51P8).

The course form is not submitted by creating the document or evidence package.
