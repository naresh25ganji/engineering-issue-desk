# Inspection of the repository-context run

Codex inspected the exported findings and code after this live run. These are AI-assisted engineering notes, not human accuracy labels. Human fields in `report.json` remain null; none of the cases was approved.

| Case | Observed behavior | Remaining review need |
| --- | --- | --- |
| W3-01 outcome labels | One retained finding identifies the accepted labels and user-reported meaning; the reviewer removed a cross-process defect claim and retained a proposed diagnostic | The selected code shows the validation and disclosure. Cross-process guarantees still need primary threading/filesystem context or actual tests before becoming a defect claim. |
| W3-02 freshness | The evidence review completed and retained two observations plus proposed boundary tests | The phrase “earlier than the current time” is stricter than the code: `snapshot > now` rejects the future but accepts equality. An engineer should correct that wording. Proposed tests were not executed by the agent. |
| W3-03 hosted embeddings | The model listed files and attempted three repository reads. One exceeded the 160-line limit and failed; two smaller reads registered immutable excerpts. It described response validation, cache identity, and sanitized error handling instead of repeating the earlier missing-key/leak claims | It did not read the `_configuration` helper or call repository search. Avoiding an unsupported claim does not establish complete helper analysis. Broad wording such as “thoroughly” is a judgment; the specific observed checks and their code evidence are what can be assessed. |

The 3/3 handoff result establishes saved workflow completion on these three cases. It does not establish human accuracy, usefulness, a resolved defect, or general reliability. The run differs from the baseline in tools and prompt/reviewer behavior; it is not an isolated experiment proving which change improved the result. Live web research and specialist delegation were not used in this run; controlled harness tests and a separate provider probe cover different properties.

The original failed W3-02 case was retried separately after this run. `checkpoint-recovery.json` records one reviewer call, no repeated research/tools, and a pending human handoff. That later private case transition does not rewrite the historical failure recorded in `evaluation/week3-results/report.json`.
