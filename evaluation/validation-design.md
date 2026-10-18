# Frozen validation set design

This is a new, independently AI-authored validation set for Engineering Issue Desk. It is **not** a human benchmark, an external hidden test, or a measure of production reliability. The author also participated in earlier corpus and integration reviews, so this is not a claim of complete blindness to the project's architecture. During this authoring task, only the fixed source corpus and the original question file were read: no implementation or prompt inspection, retrieval experiments, provider calls, environment-file access, or final-repository writes were performed.

## Freeze record

- Frozen at UTC: `2026-10-18T15:52:33+00:00`.
- Dataset: `evaluation/validation-questions.json`.
- Dataset SHA-256: `1beedd20ab48292707fe659d25bf9c51b41150aee2493e44166b557d17793b2f`.
- Corpus SHA-256: `3caa3fbd41e436469b6bdc63f5435eb9ec8f5eecfa49064e02a77fedc618d503`.
- Original diagnostic SHA-256: `2cf023de47117bcd7fc02b45219937803185ed9493faaa51705960b8b79883aa`.

`freeze.json` contains machine-readable counts and the same provenance hashes. The adjacent `.sha256` file provides a convenient file-integrity check. Keep this dataset byte-for-byte unchanged after freeze. If a genuine annotation problem is found, record it separately and version any correction; do not silently rewrite labels after observing results.

## Composition and novelty

There are 20 cases: 14 seek supported evidence and six test boundaries. The evidence cases comprise 12 documentation-oriented cases and two mixed issue-report cases. The six boundaries comprise two ambiguous cases requiring clarification, two requests for absent evidence, and two adversarial requests.

Five cases require more than one expected source family. Across the answerable cases, 19 distinct expected source IDs are named; 14 were absent from the original diagnostic's gold source IDs. The new prompts target different mechanisms or decisions when they reuse a source family. There are no exact question duplicates with the original 20 cases. Those original questions were read to avoid duplication, rather than used as examples for rewriting each old case.

The set includes source authority, report-versus-diagnosis distinctions, version scope, conditions and exceptions, multi-source support, omitted evidence, and unsupported operational claims. It deliberately includes questions whose requested exact result cannot be supplied from a frozen public snapshot.

## Schema and pre-run checks

The original `questions.json` schema is retained: case ID, kind, expected behavior, question, expected source families and answer checks. `evidence` is used for 14 cases, `needs_details` for two, and `abstain` for four. Every expected source ID was checked against the corpus; each named expected URL matches its source record. Case IDs are unique, all prompts are nonempty and below 6,000 characters, answerable cases have gold families, and boundary cases have none.

Gold families are relevant examples, not exhaustive relevance labels. Multiple source IDs within one family mean alternatives; multiple families represent separate evidence needs. A retrieval miss does not by itself prove that an answer is unsupported, and a source-family hit does not prove that its selected quote supports every claim.

## Evaluation protocol

Keep the question content and expected checks withheld from the implementer until the implementation changes are complete and evaluation is explicitly requested. Reveal only the paths, hashes and aggregate counts before then. Do not index this dataset or this design note into the source corpus.

Run the frozen set once against the frozen corpus and record the implementation commit, model identity, synthesis protocol, token counts, failures, fallback behavior and latency. Preserve the original report if a later implementation is evaluated again. Repeated tuning against this set makes subsequent results diagnostic rather than independent validation.

Review accepted generated answers claim by claim against their **selected** support passages. Record factual support separately from relevance and completeness. In particular, check conditional language, source authority, version applicability, numerical details and whether an unverified report has become a general product claim. Exact quote integrity is necessary but does not establish entailment. Safe fallbacks must be reported as fallbacks, not counted as accepted model answers.

For boundary cases, judge whether the requested unsupported fact or action was declined and whether missing details were requested appropriately. General guidance or a pointer to a status service must not be mistaken for having inspected an environment or fetched live telemetry. Record discrepancies between automated status labels and a semantically appropriate partial refusal rather than quietly changing the expected labels.

No performance result, acceptance score or quality score is asserted by this design file. Those remain unmeasured until the frozen set is evaluated.
