# Week 4 video script

About 2 minutes. Read the narration below; screen cues are in italics. Use saved results without starting a new model run.

In Evaluation, set **Show evaluation suite** to **Original Week 4 reviewer experiments**.

**0:00–0:20 · Architecture**

*Show the [Week 4 diagram](week4-architecture.png).*

“Engineering Issue Desk helps engineers investigate issues and prepare cited reports. An independent evidence reviewer filters findings and proposed steps before human approval. My Week 4 project evaluates that reviewer.”

**0:20–0:40 · Dataset**

*Show the [workbook](../outputs/week4/Engineering%20Issue%20Desk%20Week%204%20Evaluation.xlsx), Summary and Cases sheets.*

“I tested 40 fixed repository-derived cases with 119 item decisions, including happy paths, edge cases, known failures, and adversarial inputs. Runs share the same evidence and model. Expected labels stay outside the model inputs.”

**0:40–1:00 · One failure**

*Choose `evaluation/week4-baseline/report.json`, select W4-13 under **Inspect evaluation case**, and expand **Input, reference labels, and actual verdicts**. Repeat with `evaluation/week4-calibrated/report.json`.*

“Here, the baseline rejected a useful error-message test because the sentinel should be absent. But checking absence is exactly the diagnostic. The calibrated prompt correctly keeps it. Across four prompt additions, I clarified diagnostics, boundary conditions, static evidence, and rubric examples.”

**1:00–1:30 · Results**

*Show `evaluation/week4-comparison/report.json` and the workbook's Summary sheet.*

“Across five recorded stages, exact agreement rose from 80 to 97.5 percent: seven case fixes and no final regressions. The selected prompt rejected all 43 unsupported items and retained 75 of 76 supported items. One intermediate stage had an error, so the changes did not improve every measure at every step. Final p95 latency was 6.5 seconds, with about 2,933 tokens per case.”

**1:30–2:00 · Trace and limits**

*Show the `events` in the [saved W4-13 case](../evaluation/week4-calibrated/cases/W4-13.json), then the workbook's Item review sheet.*

“Each run saves verdicts, reasons, prompt versions, latency, and token usage. Dollar cost is unknown. These scores measure agreement with AI-proposed labels; human review is pending. One supported diagnostic is still rejected. Next are human label review and fresh held-out testing. Production still uses the baseline prompt.”

Before submitting: review the labels, record and upload the video, and check reviewer access to the submitted links. The [evaluation report](WEEK4_EVALUATION.md) contains the full evidence and remaining work.
