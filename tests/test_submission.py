import zipfile

import pytest

from scripts.package_submission import build_archive


def test_archive_excludes_private_local_state_and_preserves_source(tmp_path):
    (tmp_path / '.env').write_text('TRIAGE_API_KEY=local-credential-placeholder\n')
    (tmp_path / '.env.example').write_text('TRIAGE_API_KEY=replace-me\n')
    (tmp_path / 'app.py').write_text('print("demo")\n')
    (tmp_path / '.cache').mkdir()
    (tmp_path / '.cache' / 'token.txt').write_text('private')
    (tmp_path / '.streamlit').mkdir()
    (tmp_path / '.streamlit' / 'secrets.toml').write_text('private')
    output = tmp_path / 'artifacts' / 'submission.zip'
    build_archive(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert names == {'engineering-issue-desk/app.py', 'engineering-issue-desk/.env.example',
                         'engineering-issue-desk/SUBMISSION-MANIFEST.json'}


def test_archive_rejects_configured_credential_inside_an_asset(tmp_path):
    secret = 'local-credential-placeholder'
    (tmp_path / '.env').write_text('TRIAGE_API_KEY=' + secret)
    (tmp_path / 'app.py').write_text('credential = ' + repr(secret))
    output = tmp_path / 'submission.zip'
    with pytest.raises(ValueError, match='credential'):
        build_archive(tmp_path, output)
    assert not output.exists()


def test_archive_includes_versioned_application_playbooks(tmp_path):
    playbook = tmp_path / 'agent_skills' / 'python-security' / 'SKILL.md'
    playbook.parent.mkdir(parents=True)
    playbook.write_text('---\nname: python-security\nversion: 1.0.0\n---\nReview guidance.')
    output = tmp_path / 'submission.zip'
    build_archive(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        assert 'engineering-issue-desk/agent_skills/python-security/SKILL.md' in archive.namelist()
