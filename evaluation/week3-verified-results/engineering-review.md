# Engineering inspection of the final three-case run

Coding-assistant inspection, October 25, 2026. This is a qualitative audit of the exported reports and actual source files. It is not a user approval, independent human rating, answer-accuracy score, or execution of the supplied programs.

The frozen report is `report.json`, SHA-256 `4582cab5e70b1cb0f6bb470f581c0c5b82b776246a55d69e5cf56ca74a01b1d4`. All three scheduled cases reached saved human review. Only W3-02 retained findings, so one case passed exact quotation checks and two were `not_applicable`. These measures must remain distinct.

| Case | Inspection and limits |
| --- | --- |
| W3-01, reviews.py | The separate reviewer removed both findings and both remedies. The selected quotations did not show all filtering logic claimed, and the proposed fixes assumed a multi-process deployment that was not established. The empty report asks for deployment details. This demonstrates filtering and abstention; it does not establish that the module has no defects. The reviewer may be overly conservative about standard threading.Lock semantics when no authoritative reference is selected. |
| W3-02, freshness.py | The retained statements match the inspected checks: positive non-boolean integer age, timezone-aware current time, invalid naive/future snapshots counted as missing, and stale age at or above the threshold. A snapshot equal to now is accepted. Proposed boundary tests are not claimed as run by the agent. Findings are sorted medium before low; AI verdict indices refer to the original draft order. |
| W3-03, hosted_embeddings.py | Both draft findings were removed because the selected excerpts did not establish every HTTP/response validation or cache-invalidation claim. The remaining proposal is to document the observed fingerprint composition. No missing-key, error-leak, cache-defect, or successful runtime behavior is established by this handoff. |

The run observed saved-evidence reads and bounded repository reads, including one rejected oversized line range followed by successful smaller reads. It did not invoke Keenable, specialists, playbooks, static analysis, or review-history search. Those capabilities have separate checks; available tools must not be reported as tools actually used.

## Preserved failure and recovery

The immediately preceding `../week3-final-results/report.json` reached 2/3 handoffs. W3-02 failed because the application HTML detector misclassified the plain comparison sequence `oldest < max_age_days ... >= max_age_days` as an HTML tag. A regression test and narrower tag detection corrected this. Explicit retry reused the completed investigator checkpoint and made only one evidence-review call (3,144 reported tokens, zero tools), recorded in `../week3-final-results/comparison-guard-recovery.json`. The original failed-run report was preserved. The separate fresh final run above reached 3/3; these are separate observations, not a rewritten success rate.

## Stretch-feature exercise

`extension-exercise.json` records a separate live case over the actual freshness.py module. It invoked list/read case evidence, the evidence-causality playbook, and bounded Bandit analysis, reached human review with one retained finding, and used six model calls, four tools, and 34,120 reported tokens. It did not execute supplied code. The history option was enabled but the agent did not invoke search_review_history, so positive history recall is established only by offline harness tests. No real GitHub review was published; publication transport tests use mocks.

Human relevance, completeness, false-positive rate, time saved, and investigation usability remain unscored. The final video must demonstrate the actual application and explain these limits.
