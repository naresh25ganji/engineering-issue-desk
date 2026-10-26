---
name: python-testing
version: 1.0.0
description: Plan focused regression tests and state honest limits of static Python review.
---

# Python testing review

This maintained playbook supplies review guidance through an on-demand tool. It
does not execute code, run a test framework, or establish factual case evidence.

1. Identify the behavior in question, the expected behavior, the triggering input,
   and the relevant function or state boundary. Read helpers that control the
   outcome before diagnosing the caller from an incomplete excerpt.
2. Prefer a small test that distinguishes the suspected mechanism from a plausible
   alternative. Explain the setup, trigger, observation, and expected result in
   plain language. Keep a diagnostic proposal separate from a proven fix.
3. Include meaningful boundaries when the code warrants them: missing input,
   invalid input, conflicting state, provider timeout, denied access, malformed
   output, or persistence across a restart. Do not add unrelated checklists.
4. For concurrency, identify the exact shared resource and process or thread
   boundary. A thread test does not establish process safety. A happy-path test
   does not demonstrate recovery from an interrupted operation.
5. For report and tool contracts, check allowed fields, identifiers, bounds,
   failure behavior, and whether untrusted content can gain action authority.
   Test the outcome rather than merely duplicating the implementation.
6. Record what is known from code, what remains a hypothesis, and what would need
   actual execution. Never describe a proposed test as completed or infer a
   passing result from source inspection.

Use current registered sources for factual findings. Historical reviews can guide
which test to propose, but they cannot supply proof, execution results, or human
approval for the current case.
