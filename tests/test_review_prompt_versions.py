"""Experimental prompt selection must not silently change production review."""
import hashlib

import pytest

from triage_desk.agent_review import _REVIEW_PROMPT, review_prompt, review_report


def test_baseline_is_preserved_and_experiments_are_explicit_and_cumulative():
    assert review_prompt() == review_prompt('baseline') == _REVIEW_PROMPT
    versions = ['baseline', 'diagnostic', 'scope', 'authority', 'calibrated']
    prompts = [review_prompt(version) for version in versions]
    assert len({hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts}) == 5
    assert all(later.startswith(earlier) for earlier, later in zip(prompts, prompts[1:]))


def test_unknown_prompt_refuses_before_provider_or_budget_dispatch():
    with pytest.raises(ValueError, match='Unknown evidence-review prompt version'):
        review_report({}, None, None, prompt_version='unknown')
