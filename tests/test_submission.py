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


def test_archive_includes_only_final_week4_workbooks_from_outputs(tmp_path):
    outputs = tmp_path / 'outputs' / 'week4'
    outputs.mkdir(parents=True)
    (outputs / 'evaluation.xlsx').write_bytes(b'final-workbook')
    (outputs / 'builder.mjs').write_text('intermediate')
    (outputs / 'preview.png').write_bytes(b'preview')
    other = tmp_path / 'outputs' / 'unrelated'
    other.mkdir()
    (other / 'private.xlsx').write_bytes(b'private')
    output = tmp_path / 'submission.zip'
    build_archive(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert 'engineering-issue-desk/outputs/week4/evaluation.xlsx' in names
        assert not any(name.endswith(('.mjs', '.png', 'private.xlsx')) for name in names)


def test_archive_rejects_credential_inside_compressed_workbook_xml(tmp_path):
    secret = 'local-credential-placeholder'
    (tmp_path / '.env').write_text('FIREWORKS_API_KEY=' + secret)
    workbook = tmp_path / 'outputs' / 'week4' / 'evaluation.xlsx'
    workbook.parent.mkdir(parents=True)
    with zipfile.ZipFile(workbook, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('xl/sharedStrings.xml', '<t>' + secret + '</t>')
    assert secret.encode() not in workbook.read_bytes()
    with pytest.raises(ValueError, match='credential'):
        build_archive(tmp_path, tmp_path / 'submission.zip')


def test_archive_preserves_authored_log_fixtures_but_excludes_runtime_logs(tmp_path):
    fixture = tmp_path / 'evaluation' / 'fixtures' / 'workflows' / 'incident.log'
    fixture.parent.mkdir(parents=True)
    fixture.write_text('Synthetic controlled incident record; no private telemetry.')
    (tmp_path / 'evaluation' / 'runtime.log').write_text('private runtime diagnostics')
    output = tmp_path / 'submission.zip'
    build_archive(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        assert 'engineering-issue-desk/evaluation/fixtures/workflows/incident.log' in archive.namelist()
        assert not any(name.endswith('runtime.log') for name in archive.namelist())
