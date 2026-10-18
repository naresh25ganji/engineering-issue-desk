import pytest

from triage_desk.reviews import outcome_summary, save_outcome


def test_outcomes_count_latest_review_once_and_exclude_followups(tmp_path):
    path = tmp_path / 'outcomes.jsonl'
    assert outcome_summary(path)['user_reported_first_contact_resolution'] is None
    save_outcome(path, 'a' * 32, 'resolved')
    save_outcome(path, 'a' * 32, 'needs_followup')
    save_outcome(path, 'b' * 32, 'resolved', first_contact=False)
    result = outcome_summary(path)
    assert result['reviewed_runs'] == 2
    assert result['reviewed_first_contacts'] == 1
    assert result['user_reported_first_contact_resolution'] == 0
    assert 'question' not in path.read_text()


def test_invalid_outcome_cannot_write_a_record(tmp_path):
    path = tmp_path / 'outcomes.jsonl'
    with pytest.raises(ValueError):
        save_outcome(path, 'a' * 32, 'guaranteed_correct')
    assert not path.exists()
