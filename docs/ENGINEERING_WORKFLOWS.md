# Engineering workflows and expanded evaluations

The app now supports incident triage, change review, and test planning alongside issue investigation. All four use the existing investigator, registered evidence, independent evidence reviewer, and saved human-review checkpoint. The expanded evaluation exercises the complete graph as well as the reviewer component and deterministic citation checks.

| Workflow | What to supply | Intended result |
| --- | --- | --- |
| Issue investigation | Trigger, expected behavior, version, reproduction or code | Supported observations, focused diagnostics, and explicit information gaps |
| Incident triage | Timestamped notes or logs, reported impact, recent changes | Attributed signals separated from possible causes, with verification before proposed remediation |
| Change review | Focused diff or source, paths, intended behavior, relevant helpers | Static observations and concrete regression risks, with proposed verification |
| Test planning | Relevant code branches, behavior, or acceptance criteria | Proposed positive, boundary, and negative checks with supported expected results |

Open **Deep Investigation**, choose **Workflow**, and review its input guidance. A starter example can populate the request; Try a sample with evidence can load a complete frozen request and context without starting a model call. Paste the corresponding reproduction, incident notes, diff, or acceptance criteria, then submit **Start deep investigation**. Saved cases show their workflow and version and can be reopened, revised, or resumed. Reports remain subject to human review; supplied code and diffs are inspected rather than run or merged.

Workflow definitions live in [use_cases.py](../triage_desk/use_cases.py). Each case saves an immutable `use_case_id` and `use_case_version`; old cases receive issue investigation version 1. The selected application-owned instructions enter the actual investigator prompt and remain stable across retry and revision. Traces carry workflow and case identities. The shared report schema still separates findings, proposed next steps, and missing information. Repository access and earlier-review memory remain explicit case options.

The available knowledge base and external source allowlist retain their existing Streamlit, Python, LangChain, and streamlit/streamlit scope. Supplying another engineering excerpt does not add a new domain corpus or establish the behavior of omitted helpers, current deployments, or external services.

In **Evaluation**, use **Show evaluation suite** to browse full engineering workflows, fresh reviewer regression, citation integrity, or earlier experiments. **Run a new evaluation** offers suite selection, case filters, workflow filters where applicable, baseline/calibrated evidence-review selection, and one to four parallel cases. **Validate inputs** writes an unscored inventory without a model call. **Run model evaluation** explicitly sends the selected requests and frozen context to the configured Fireworks model. Citation checks always run locally without a model.

The workflow report shows targets, per-workflow results, accounting, and case filters. Its case inspector exposes the supplied question/context, actual report, reviewer decisions, error, registered sources, provisional expectations, and local trace. The reviewer view preserves item verdicts and paired changes. The integrity view separates expected acceptance from rejection and exposes the validator inputs.

Three suites provide different evidence:

| Suite | Cases | What is measured |
| --- | --- | --- |
| [Full engineering workflows](../evaluation/workflow-cases.json) | 40, ten per workflow | Actual investigator/tool/reviewer execution, persisted handoff, structural contract, literal citation and anchor coverage, latency, returned token usage |
| [Fresh reviewer regression](../evaluation/reviewer-regression-cases.json) | 30, containing 90 balanced reference items | Actual reviewer verdict agreement with 45 proposed supported and 45 proposed unsupported labels |
| [Citation and report integrity](../evaluation/integrity.py) | 30 deterministic checks | Actual schema and registered-quote acceptance/rejection behavior, with zero model calls |

The forty workflow cases contain 20 happy paths, 12 edge cases, six known failure classes, and two adversarial sources. The thirty reviewer cases contain 15 happy paths, nine edge cases, four known failure classes, and two adversarial sources. Twenty [frozen fixtures](../evaluation/fixtures/workflows) contain ten repository/test excerpts, five proposed unapplied diffs based on repository excerpts, and five explicitly labeled synthetic log/adversarial examples. Each context is pinned by SHA-256 and is at most 1,494 characters. Evaluator titles, expected labels, rubrics, and failure descriptions stay outside the investigator inputs. Labels and semantic rubrics remain AI-proposed and pending human review. These are development regressions, not a human-validated held-out benchmark.

The completed full-workflow run used `accounts/fireworks/models/gpt-oss-120b` with the **calibrated** independent evidence-review prompt, four workers, repository context off, and review memory off. The interactive app retains **baseline** evidence review by default. This experiment does not promote the candidate or establish the same results for baseline. The recorded run was created October 30, 2026, at 3:23 p.m. PDT.

| Full-workflow measure | Recorded result | Target |
| --- | --- | --- |
| Saved human-review handoffs | 37/40, **92.5%** | ≥95%, not met |
| Structural contract | 26/40, **65%** | ≥90%, not met |
| Literal anchors covered inside valid quotes | 42/45, **93.33%** | ≥85%, met |
| Cases with exact citations | 38/38 applicable; two empty reports N/A | Literal integrity only |
| Case latency p95 | **16.15 seconds** | ≤90 seconds, met |
| Mean reported tokens per scheduled case | **16,930.85** | ≤20,000, met |
| Recorded execution | 128 model calls, 49 tool calls, **677,234 reported tokens** | Complete returned usage for recorded calls |

| Workflow | Saved handoffs | Structural contract |
| --- | --- | --- |
| Issue investigation | 8/10 | 5/10 |
| Incident triage | 10/10 | 8/10 |
| Change review | 10/10 | 8/10 |
| Test planning | 9/10 | 5/10 |

The contract requires the saved workflow identity, persisted handoff, completed independent review, configured minimum finding/step counts, successful required evidence-read tool, and no citation-integrity or execution error. Anchor recall is a separate lexical metric. Handoffs and contract rates retain all forty scheduled cases in their denominators. Valid citations may remain on failed cases, so 38 citation passes do not imply 38 completed reviews. Neither exact quotes nor structural item counts establish entailment, correct diagnosis, or useful advice; all forty human semantic reviews remain pending.

The fourteen contract failures have three distinct observable forms:

- **Three execution failures:** WF-04 reached its shared token budget before independent review; WF-08 recorded `ReviewUnavailable` with `invalid_schema`; WF-31 recorded `ReviewUnavailable` with `invalid_reason`. These finite diagnostics and their local traces establish the boundary reached, without proving a model-internal root cause.
- **Nine reports below the proposed-step minimum:** WF-02, WF-09, WF-11, WF-16, WF-22, WF-23, WF-34, WF-36, and WF-40. WF-22 authored no steps; the others lost steps during evidence review. Some removals address unsupported premises or expected outcomes, so this group cannot be counted as nine semantic reviewer errors.
- **Two reports with no retained findings:** WF-10 and WF-39. Their selected evidence and reviewer reasons need human review before deciding whether the original finding was supported, the quote span was sufficient, or abstention was appropriate.

WF-02 also requested specific configuration-helper key/base values in `missing_information`. No credential was supplied or exposed in the recorded run. Follow-up work narrows sensitive information requests and adds shared guidance; it is separate from the historical forty-case experiment.

A separate [post-fix WF-02 smoke run](../evaluation/workflow-guard-smoke/report.json) reached human review with no credential-value request and an empty missing-information list. It made three model calls and one evidence read, reporting 13,364 tokens in 6.05 seconds. No guard conversion occurred in that sample; deterministic real-graph tests exercise the exact historical unsafe request and preserve the converted count through review. The sample still missed the proposed-step minimum, so its structural contract remains failed. The guard is a narrow pattern check on missing-information requests, not a general secret detector or a forty-case quality improvement claim. Final software verification passed 879 tests.

The [full JSON record](../evaluation/use-case-live/report.json), [readable report](../evaluation/use-case-live/report.html), and [analysis with hashes and failure groups](../evaluation/engineering-workflows-analysis.json) preserve the original outputs. A later guard smoke does not rewrite that run or establish a new forty-case result.

The new reviewer comparison used the same thirty cases and model for each prompt, without further prompt tuning on these results:

| Reviewer measure | Baseline | Calibrated |
| --- | --- | --- |
| Exact case agreement | 26/30, **86.67%** | 27/30, **90%** |
| Rejection of proposed unsupported items | 45/45, **100%** | 45/45, **100%** |
| Retention of proposed supported items | 41/45, **91.11%** | 42/45, **93.33%** |
| False positive / false negative against proposed labels | 0 / 4 | 0 / 3 |
| Review latency p95 | 3.85 seconds | 5.13 seconds |
| Mean reported tokens | 2,123.70 | 2,767.13 |

Both prompts completed 30/30 cases. Calibrated fixed W4-51 and W4-69, regressed W4-50, and retained disagreements on W4-53 and W4-57. The W4-50 disagreement concerns whether “specialist” appropriately attributes the `correctness_security` role in the bounded log; its label and the model reason need semantic review. The net change is +3.33 percentage points of case agreement. This is one paired development run, with provisional labels and serving variability; it does not establish human accuracy, robust generalization, or an isolated causal effect of prompt changes. See [baseline](../evaluation/reviewer-regression-baseline/report.json), [calibrated](../evaluation/reviewer-regression-calibrated/report.json), and [paired comparison](../evaluation/reviewer-regression-comparison/comparison.html).

The deterministic integrity suite matched all **30/30** expected outcomes: seven accepted inputs and 23 rejected inputs, with zero model calls. CI-05 deliberately attaches a real registered quote to an unrelated claim and expects acceptance. That pass demonstrates the layer's limit: schema and literal citation integrity do not establish entailment. The [recorded integrity report](../evaluation/integrity-results/report.json) preserves that example and every input.

Run the app with `uv run streamlit run app.py`. Fireworks runs require the existing local `FIREWORKS_API_KEY` and `FIREWORKS_MODEL` configuration. Keenable remains optional. Use fresh output directories to preserve recorded experiments:

```bash
# Frozen-input inventories: no model or provider call.
uv run python -m evaluation.use_cases run --output-dir .cache/workflows-inventory
uv run python -m evaluation.week4 run --cases evaluation/reviewer-regression-cases.json --output-dir .cache/reviewer-regression-inventory

# Full graph, matching the recorded experiment's reviewer choice.
uv run python -m evaluation.use_cases run --live --review-prompt calibrated --workers 4 --output-dir evaluation/workflows-live-new

# Select one workflow or repeat --case to select individual cases.
uv run python -m evaluation.use_cases run --live --review-prompt calibrated --use-case incident_triage --output-dir evaluation/incidents-live-new

# Paired production-reviewer component evaluation on the new thirty cases.
uv run python -m evaluation.week4 run --cases evaluation/reviewer-regression-cases.json --live --variant baseline --workers 4 --output-dir evaluation/reviewer-baseline-new
uv run python -m evaluation.week4 run --cases evaluation/reviewer-regression-cases.json --live --variant calibrated --workers 4 --output-dir evaluation/reviewer-calibrated-new
uv run python -m evaluation.week4 compare --baseline evaluation/reviewer-baseline-new --improved evaluation/reviewer-calibrated-new --output-dir evaluation/reviewer-comparison-new

# Deterministic validator execution: no credentials or model required.
uv run python -m evaluation.integrity --output-dir evaluation/integrity-new

# Start one interactive incident workflow from an explicit context file.
uv run python scripts/investigate_deep.py start --use-case incident_triage --question "Triage these supplied records and identify the next verification step." --code-file evaluation/fixtures/workflows/incident_review.log
```

`evaluation.use_cases` also supports `--resume` for an interrupted compatible run and `compare` for matched live workflow reports. It gives each case an isolated local workspace and preserves fixture/input hashes, profile versions/hashes, prompt hashes, code hashes, model identity, budgets, local root/child events, errors, and returned usage. Remote tracing is disabled for these evaluation runs; local records are not LangSmith traces. Dataset v1 hashes are `0c816f4693609be82e579ba6f1a0e6c775544e8b134a7c9fecb4851ddccb75ba` for workflows and `1b7b2fac74ee74f4257986effb2d57070f99085a57e02f02fa07762e804c7b70` for reviewer regression. The full manifest retains the exact prompt/profile/code versions that produced the recorded measurements, including versions preceding follow-up fixes.

Dollar cost and engineering time saved remain unknown. The next priorities are human semantic review of outputs and labels; reliable recovery from invalid structured reviewer output while reserving enough budget for review; calibration of diagnostic proposals against verified fixes; and a separately authored, human-reviewed held-out set with repeated calls to the actual prompt/model combination before candidate promotion. The original Week 4 study and workbook remain separate preserved deliverables.

The [software verification record](../evaluation/engineering-workflows-validation.json) records 879 passing tests, working app report filters and case views, and zero model calls while browsing them. The source archive includes all twenty frozen fixtures, including the explicitly synthetic log files.
