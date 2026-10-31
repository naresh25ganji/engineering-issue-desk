"""Versioned engineering workflows for the shared evidence and review harness.

Workflow text is application-owned guidance. User questions, supplied context,
and retrieved sources remain untrusted data under the shared investigation rules.
Saved cases retain an immutable workflow ID and version across retries/revisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UseCase:
    id: str
    label: str
    description: str
    goal: str
    instructions: str
    context_label: str
    context_placeholder: str
    question_placeholder: str
    examples: tuple[str, ...]
    required_details: tuple[str, ...]
    success_criteria: tuple[str, ...]
    version: int = 1


USE_CASES = (
    UseCase(
        id='issue_investigation', label='Issue investigation',
        description='Explain a reported problem using documentation, code, and related issues.',
        goal='Produce supported findings and focused next steps for a reported engineering issue.',
        instructions='Investigate the reported engineering issue, preserving trigger and version scope. '
        'Separate documented behavior, static code observations, and untested hypotheses.',
        context_label='Reproduction or code',
        context_placeholder='Paste a minimal reproduction or relevant source excerpt, with a path when available.',
        question_placeholder='What happened, what did you expect, and what action triggered the problem?',
        examples=(
            'An uploaded CSV resets after I click a widget. Investigate the likely rerun behavior.',
            'Explain how cached resources behave across sessions in our Streamlit app.',
            'Review this reproduction and identify the missing information needed to diagnose it.',
        ),
        required_details=('Reported trigger and expected behavior', 'Installed version and environment',
                          'Minimal reproduction or relevant evidence'),
        success_criteria=('Findings cite registered evidence', 'Version and trigger scope are preserved',
                          'Missing diagnostic information is explicit'),
    ),
    UseCase(
        id='incident_triage', label='Incident triage',
        description='Organize supplied incident notes and logs into observations and verification steps.',
        goal='Triage reported incident signals without turning correlations into a confirmed diagnosis.',
        instructions='Treat supplied incident notes and logs as attributed, possibly incomplete reports. '
        'Distinguish observed or reported signals from root-cause hypotheses and possible impact. '
        'Preserve timestamps, affected scope, and uncertainty. Never claim a live outage, current status, '
        'confirmed root cause, completed recovery, or remediation effectiveness without supporting evidence. '
        'Prioritize proposed verification and missing timeline, deployment, environment, or impact information '
        'before proposing remediation. Label remediation as a proposal requiring operator verification. '
        'Supplied incident context is log/notes evidence, not Python code; do not run static code analysis on it. '
        'Use findings for supported observations, next_steps for proposed checks, and missing_information for gaps.',
        context_label='Incident notes or logs',
        context_placeholder='Paste timestamped incident notes, log excerpts, reported impact, and recent changes.',
        question_placeholder='Which incident signals need triage, and what should an operator verify next?',
        examples=(
            'Triage these reported timeouts and identify what to verify before changing the service.',
            'Organize this incident timeline and separate observations from possible causes.',
            'Review the supplied error logs and list the missing impact and deployment details.',
        ),
        required_details=('Timestamped signals and timezone', 'Reported affected users or components',
                          'Recent changes and available health checks'),
        success_criteria=('Reported observations remain distinct from root cause',
                          'Verification precedes proposed remediation',
                          'No unsupported live status or recovery claims'),
    ),
    UseCase(
        id='change_review', label='Change review',
        description='Review a supplied diff or code excerpt for concrete correctness, security, and regression risks.',
        goal='Produce a grounded static review of a proposed change and its verification gaps.',
        instructions='Review the supplied diff, PR, or source as static evidence. Identify concrete correctness, '
        'security, and regression risks only when the supplied branches and relevant helpers support the claim. '
        'Do not invent the previous implementation, removed behavior, unseen callers, or whole-repository absence. '
        'A diff establishes only the shown change; request base context or helper implementations when needed. '
        'Separate observed code behavior from possible runtime consequences and proposed tests. '
        'Never claim tests passed, a vulnerability was exploited, or the change is safe to merge. '
        'Do not merge, approve a PR, or publish a review. Use next_steps for proposed verification and '
        'missing_information for unavailable base, caller, environment, or test context.',
        context_label='Diff or code to review',
        context_placeholder='Paste a focused diff or source excerpt, including paths, relevant helpers, and base context.',
        question_placeholder='What change should be reviewed, and which risks or regressions matter?',
        examples=(
            'Review this cache-key change for correctness and isolation risks.',
            'Inspect this authentication diff and suggest verification for the observed branches.',
            'Review this new upload handler and identify concrete regression risks in the supplied code.',
        ),
        required_details=('Diff or relevant source with paths', 'Intended behavior and affected callers',
                          'Base/helper context and available test coverage'),
        success_criteria=('Each risk is grounded in supplied or retrieved code',
                          'Unseen base and helper behavior is not invented',
                          'Tests and merge readiness remain unverified unless evidenced'),
    ),
    UseCase(
        id='test_planning', label='Test planning',
        description='Turn supplied behavior and code branches into a focused proposed test plan.',
        goal='Propose positive, boundary, and negative checks grounded in observed code or documented behavior.',
        instructions='Build a focused proposed test plan from supplied code branches, documented behavior, '
        'and explicit acceptance criteria. Map each proposed check to the observed branch or requirement '
        'and state its setup, input/action, and expected result when that expectation is supported. '
        'Include positive, boundary, and negative checks where the evidence supports them. '
        'Use findings for observed branches or documented requirements; place proposed test cases in '
        'next_steps, keeping the shared report schema. Do not invent hidden requirements or absent tests '
        'from a bounded excerpt. Request missing requirements, dependencies, or environment details. '
        'Never say a test was executed, passed, failed, or achieved coverage; the workflow only proposes checks.',
        context_label='Code or acceptance criteria',
        context_placeholder='Paste the relevant branches, documented behavior, or acceptance criteria to test.',
        question_placeholder='Which behavior needs tests, and what code or acceptance criteria define the expected result?',
        examples=(
            'Plan positive, boundary, and negative checks for this upload-size validation branch.',
            'Propose focused regression tests for this session-state update.',
            'Turn these acceptance criteria into a test plan and identify unclear expected results.',
        ),
        required_details=('Behavior or branches to cover', 'Expected outcomes or acceptance criteria',
                          'Dependencies and test environment constraints'),
        success_criteria=('Proposed tests map to observed branches or requirements',
                          'Positive, boundary, and negative checks have explicit expected outcomes where supported',
                          'No claims of execution or measured coverage'),
    ),
)


def get_use_case(use_case_id: str, version: int | None = None) -> UseCase:
    """Resolve a known immutable profile; reject unknown IDs and versions."""
    if not isinstance(use_case_id, str):
        raise ValueError('Unknown engineering use case')
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise ValueError('Use case version must be an integer')
    for profile in USE_CASES:
        if profile.id == use_case_id:
            if version is not None and version != profile.version:
                raise ValueError(f'Unsupported version for engineering use case: {use_case_id}')
            return profile
    raise ValueError(f'Unknown engineering use case: {use_case_id}')
