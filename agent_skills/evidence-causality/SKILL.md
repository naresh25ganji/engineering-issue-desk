---
name: evidence-causality
version: 1.0.0
description: Assess claim entailment, omitted context, causal applicability, and prerequisites for engineering remedies.
---

# Evidence and causal applicability

This maintained playbook is review guidance, not citable case evidence. Loading
it adds no execution, publishing, filesystem, or external-system permission.

1. State one factual claim at a time. Select the registered source and quote IDs
   that support that complete claim. Exact quotation establishes provenance; it
   does not by itself establish entailment, causality, or correctness.
2. Distinguish code observations, attributed issue reports, documented guarantees,
   hypotheses, and proposed tests. Do not combine a supported observation with an
   unsupported diagnosis or runtime result.
3. Inspect called helpers and the relevant caller before asserting a missing
   safeguard. Name omitted context. Absence from a bounded passage is not proof
   that a feature, check, or condition is absent from the implementation.
4. For a remedy, trace the observed trigger through its execution path and state
   lifetime to the symptom. Identify the specific part that the remedy changes.
   A true API capability may be irrelevant to that path or affect a different
   event. Calling it an alternative does not establish applicability.
5. Check every prerequisite: persisted data, changed code placement, triggering
   callback, installed version, and state lifetime. The proposal must establish
   missing prerequisites or explicitly make them conditions to verify first.
   A true fact combined with an unsupported remedy makes the whole step suspect.
6. Preserve contradictory evidence and uncertainty. Use a diagnostic to test a
   missing causal link; do not label that diagnostic a verified fix. Ask for the
   environment or reproduction details that would resolve the uncertainty.

Prior human approval is a historical label, not factual ground truth. Revalidate
reused conclusions against newly registered sources and the current version.
Retain the human review handoff even after an independent AI review completes.
