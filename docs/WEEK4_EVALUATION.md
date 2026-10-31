# Week 4 evaluation of the Engineering Issue Desk

This project evaluates the independent evidence reviewer used by the Week 3 Deep Investigation agent. The reviewer decides which cited findings and proposed next steps reach a human. Unsupported advice can waste an engineer's time; rejecting supported observations can make the investigation unhelpful. This is a component evaluation, and does not measure the full investigator's research, delegation, or end-to-end usefulness.

## Evaluation framework

I will measure exact case-level verdict agreement, unsupported-item rejection, supported-item retention, p95 review latency, and mean reported tokens on the Engineering Issue Desk evidence reviewer using 40 fixed cases derived from repository code, existing test fixtures, and recorded Week 3 failures. Code compares model verdicts with proposed reference labels. The pass bars are 90% exact case agreement, 95% unsupported-item rejection, 90% supported-item retention, p95 latency at most 30 seconds, and mean reported tokens at most 4,000 per case. Baseline and experimental prompts use the same dataset, model, evidence, and output schema.

The reference labels are AI-proposed and require human review. Agreement with them is a development diagnostic, not human-validated accuracy. No report is automatically approved or published. The spreadsheet route is used; local case traces are supporting execution evidence, not LangSmith traces.

| Field | Decision |
| --- | --- |
| Agent under test | The actual `review_report` evidence-review stage in `triage_desk/agent_review.py`, using the configured Fireworks model. |
| Target user and outcome | Engineers receive supported observations and useful diagnostics, with unsupported causal fixes and invented execution results removed before review. |
| Judges | Exact boolean comparison for findings and proposals; separate trajectory/schema/error checks; editable human semantic review in the workbook. The model under test is not also the ground-truth judge. |
| Dataset | 40 fixed cases: 20 happy paths, 12 edge cases, 6 known failures, and 2 adversarial cases. Literal source excerpts retain paths, line ranges and source hashes. |
| Reference-label provenance | Cases and labels are adapted by a coding assistant from real repository seeds. No case has human signoff by default. This limitation prevents calling the dataset a completed human-reviewed golden benchmark. |
| Pass bars | Case agreement ≥90%; unsupported rejection ≥95%; supported retention ≥90%; p95 latency ≤30 s; mean reported tokens ≤4,000. All targets were selected before the baseline run. |
| Instrumentation | One local case trace and one real LLM child call per attempted case, with case ID, dataset/prompt hashes, version, model, expected and predicted verdicts, scores, errors, observed token counts and latency. |
| Comparison controls | Immutable output directories, identical case IDs and dataset bytes, prompt snapshots, matched model settings, no reference labels in reviewer inputs, and no automatic retries of bad responses. |
| Cost | Tokens are an observed resource metric. Dollar cost stays unknown unless explicit provider rates are supplied; an estimate is not a billing record. |

## Read the recorded results

<!-- BEGIN GENERATED WEEK4 RESULTS -->

All 5 stages below are real calls to the configured production evidence-review component on the same corrected 40-case version 2 dataset. The model is `accounts/fireworks/models/gpt-oss-120b`. Reference labels remain **AI-proposed and pending human review**; these scores measure agreement with those labels, not human-validated accuracy.

| Stage | Exact ≥90% | Reject unsupported ≥95% | Retain supported ≥90% | p95 ≤30 s | Mean tokens ≤4,000 | FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 80.00% (failed) | 95.35% (passed) | 89.47% (failed) | 4.54 (passed) | 2,187.85 (passed) | 2 / 8 |
| diagnostic | 85.00% (failed) | 97.67% (passed) | 93.42% (passed) | 5.40 (passed) | 2,329.68 (passed) | 1 / 5 |
| scope | 85.00% (failed) | 95.35% (passed) | 92.11% (passed) | 5.33 (passed) | 2,443.90 (passed) | 1 / 4 |
| authority | 87.50% (failed) | 100.00% (passed) | 93.42% (passed) | 6.15 (passed) | 2,565.72 (passed) | 0 / 5 |
| calibrated | 97.50% (passed) | 100.00% (passed) | 98.68% (passed) | 6.50 (passed) | 2,932.70 (passed) | 0 / 1 |

baseline: 40/40 completed, 0 error cases, 0 unknown items; diagnostic: 40/40 completed, 0 error cases, 0 unknown items; scope: 39/40 completed, 1 error case, 3 unknown items; authority: 40/40 completed, 0 error cases, 0 unknown items; calibrated: 40/40 completed, 0 error cases, 0 unknown items. All scheduled cases/items remain in the score denominators.

Recorded execution/validation errors: scope W4-33: `invalid_reason`. Their item verdicts remain unknown and count against the scheduled agreement, rejection, and retention denominators. A successful call in another stage does not rewrite these preserved failures.

**Selected experimental candidate: calibrated.** Selection maximizes exact case agreement among variants with unsupported rejection at least 95% and supported retention at least 90%; lower p95 latency breaks ties. All five pass bars met: yes. The production default remains baseline pending review.

Against baseline, exact agreement changes +17.50 percentage points, unsupported rejection +4.65 pp, supported retention +9.21 pp, p95 latency +1.96 s, and mean reported tokens +744.85. Case fixes: W4-02, W4-13, W4-18, W4-26, W4-34, W4-35, W4-36. Case regressions: none. Item fixes: 9; item regressions: 0; a case fix requires every item in that case to agree.

### 4 hypotheses: predicted and measured

The first three predictions came from the preserved v1 baseline before the corrected v2 baseline and experiments. The fourth calibration hypothesis was registered after the authority result and before its own run. Measured deltas below compare each cumulative stage with the immediately preceding stage.

| Addition and target | Predicted exact Δ | Measured exact Δ | Predicted target-rate Δ | Measured target-rate Δ | Paired case fixes / regressions |
| --- | --- | --- | --- | --- | --- |
| diagnostic: Over-rejection of useful diagnostics | +12.50 pp | +5.00 pp | +6.58 pp supported retention | +3.95 pp | W4-13, W4-18, W4-26, W4-34, W4-35 / W4-06, W4-30, W4-33 |
| scope: Wrong comparison strictness and condition scope | +2.50 pp | +0.00 pp | +2.33 pp unsupported rejection | -2.33 pp | W4-06, W4-30 / W4-18, W4-35 |
| authority: Over-rejection of code and fixture observations | +7.50 pp | +2.50 pp | +3.95 pp supported retention | +1.32 pp | W4-33, W4-35 / W4-30 |
| calibrated: Remaining over-rejection of grounded diagnostics and literal fixture observations | +5.00 pp | +10.00 pp | +2.63 pp supported retention | +5.26 pp | W4-02, W4-18, W4-30, W4-36 / none |

Every incremental metric delta, item fix, item regression, and actual before/after model reason is preserved in [week4-analysis.json](../evaluation/week4-analysis.json). Negative deltas remain visible; a cumulative addition can lose ground. The variants are cumulative and each stage makes one real model call per case. Serving nondeterminism and repeated use of this development set prevent isolated causal attribution or a generalization claim.

### Baseline failure roots and explicit time assumption

The baseline has 10 wrong items across 8 unique cases. These operational buckets classify error direction rather than proving a model-internal cause. Cases can appear in several buckets.

| Root bucket | Wrong items | Affected cases | Case IDs | Hypothetical minutes | Actual model reason example |
| --- | --- | --- | --- | --- | --- |
| Supported diagnostic removed | 6 | 6 | W4-02, W4-13, W4-17, W4-18, W4-26, W4-35 | 30 | W4-02:next_steps:0: Proposed fixture relies on 'accepted' records, which the code does not distinguish; premise unsupported. |
| Supported static/fixture finding removed | 2 | 2 | W4-02, W4-36 | 10 | W4-02:findings:0: Code overwrites any record per run_id, not specifically accepted records; no 'accepted' filter shown. |
| Unsupported item retained | 2 | 2 | W4-34, W4-35 | 60 | W4-34:findings:0: Code shows no check for empty key before Authorization header, confirming lack of verification. |
| Unknown verdict or execution error | 0 | 0 | none | unpriced | none |

**Hypothetical assumption only:** rejecting a supported finding/proposal costs 5 minutes of triage effort; retaining an unsupported item costs 30 minutes. These are illustrative weights, not observed business cost or measured time saved. Unknown/error items stay unpriced. The baseline weights are 30 + 10 + 60 = 100 minutes. The calibrated candidate totals 5 hypothetical minutes (baseline minus candidate: +95 minutes under the same assumption). Dollar cost is unknown because provider billing rates were not supplied.

The paired comparison artifact separately uses a flat five-minute-per-case illustration. That case weight is distinct from these item-based 5/30-minute weights; do not combine the two estimates.

### Remaining candidate disagreements

The candidate has 1 item disagreement across 1 case. Each item below still needs semantic review against its frozen quote.

| Case/item | Bucket | Expected → actual | Actual model reason |
| --- | --- | --- | --- |
| W4-17:next_steps:0 | Supported diagnostic removed | True → False | Quote gives no evidence about adding live labeled examples or measuring semantic judgment. |

### Provenance and deliverables

Dataset SHA-256: `49b5fa4b1c28b650bda580b24c8c2495317deb019afcbb673ffdfed2c86f098f`. Runner SHA-256 at analysis generation: `ce01886f7753ca1282fa375c1775b2d328f702d9619f357a8e3adfe303100e10`. Every run also preserves its prompt text/hash, dataset bytes, case-local root and model events, expected/predicted verdicts, error state, returned token usage, and latency. These are local execution records; no LangSmith project is claimed.

Version 1 runs remain preserved under `evaluation/week4-v1-*` and are excluded from the final deltas. Version 2 extends the W4-07/30/38 source spans and tightens W4-10/16/21/29 wording without changing boolean labels. Reviewer source titles are neutral so evaluation-authored scenario titles cannot cue the verdict.

Recorded artifacts: [baseline](../evaluation/week4-baseline/report.html), [diagnostic](../evaluation/week4-diagnostic/report.html), [scope](../evaluation/week4-scope/report.html), [authority](../evaluation/week4-authority/report.html), [calibrated](../evaluation/week4-calibrated/report.html), [paired comparison](../evaluation/week4-comparison/comparison.html), [machine-readable analysis](../evaluation/week4-analysis.json), [editable workbook](../outputs/week4/Engineering%20Issue%20Desk%20Week%204%20Evaluation.xlsx), [demo guide](WEEK4_DEMO.md), and [submission packager](../scripts/package_submission.py).

Next, complete human review of all 119 item labels in the workbook, resolve disagreements, and evaluate the selected prompt on a separately authored held-out set with repeated calls. Keep production promotion pending that review. This component study does not establish full-investigator usefulness or observed engineering time saved.

<!-- END GENERATED WEEK4 RESULTS -->

## Reproduce the evaluation

```bash
# Validate inputs and write an unscored inventory without credentials or network calls.
uv run python -m evaluation.week4 run --output-dir .cache/week4-inventory

# Execute the production reviewer. This sends the public/repository case inputs
# to the configured Fireworks account and records local results.
uv run python -m evaluation.week4 run --live --variant baseline --workers 4 --output-dir evaluation/week4-baseline-new

# Execute the selected experimental candidate, then compare matched runs.
uv run python -m evaluation.week4 run --live --variant calibrated --workers 4 --output-dir evaluation/week4-improved-new
uv run python -m evaluation.week4 compare --baseline evaluation/week4-baseline-new --improved evaluation/week4-improved-new --output-dir evaluation/week4-comparison-new
```

Use new directories for reruns. `--resume` is for interrupted runs with the same manifest, rather than overwriting completed experiment evidence. The runner disables LangSmith recording for this spreadsheet evaluation, even if ambient tracing is enabled. Local `.env` and cached provider state never belong in the submission.

## Human review and submission

Review each expected finding/proposal against its frozen quote, mark the human decision and rationale in the workbook, and resolve disagreements before describing the labels as golden. If labels change, create a new version and rescore both runs under that same version; preserve the original experiment. A pending label is not a failed model verdict or a human-approved result.

Record a short Loom using the Week 4 demo guide. The handout gives a Sunday, November 1, 2026 deadline for Builder of the Week. No video is fabricated and no course form is submitted by the code.

## Production monitoring

Keep agent, prompt, rubric and dataset versions on every review. Sample real cases for human review weekly. Alert on a greater than 5 percentage-point seven-day drop in human-reviewed unsupported rejection or supported retention, more than 5% review errors in an hour, p95 latency above 30 seconds for more than 5% of rolling windows, or p95 tokens rising more than 25% over the previous day. Keep missing telemetry explicit. Investigate alerts using case IDs and saved evidence, and rerun the frozen suite before promoting a prompt.

## Software verification

The final offline suite passed 723 tests. The actual paired and case-level Streamlit views rendered without exceptions. The workbook passed visual inspection of all three sheets and its longest source rows, plus native LibreOffice recalculation with zero formula errors; all 40 frozen excerpts matched and all 119 human-label rows remained pending. The offline CLI inventory records no model execution or quality scores. See the [verification record](../evaluation/week4-validation.json).

Verification also exposed an intermittent pre-existing crash-recovery race: a LangGraph snapshot could have empty `next` while task writes were still pending. Recovery now checks both fields before treating a thread as finished, preserving its saved tools. A deterministic regression and all Deep Investigation tests pass. This repair was made after the reviewer experiments and is not included in their measured prompt gains.
