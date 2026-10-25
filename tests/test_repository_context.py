"""Real local-file boundaries and durable evidence for opted-in repository context."""
import hashlib
import json
import os

import pytest

from triage_desk.agent_budget import InvestigationBudget
from triage_desk.agent_evidence import EvidenceRegistry
from triage_desk.cases import CaseStore
from triage_desk import repository_context as repository


HELPER = ('from pathlib import Path\r\n\r\n'
          'def _configuration():\r\n'
          '    api_key = configured_key()\r\n'
          '    if not api_key:\r\n'
          '        raise ValueError("Missing key")\r\n'
          '    return api_key\r\n')


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'app.py').write_text('from triage_desk.generation import _configuration\n', encoding='utf-8')
    package = root / 'triage_desk'
    package.mkdir()
    (package / 'generation.py').write_bytes(HELPER.encode('utf-8'))
    nested = package / 'nested'
    nested.mkdir()
    (nested / 'helpers.py').write_text('def answer():\n    return "literal .* example"\n', encoding='utf-8')
    for directory in (root / 'tests', root / 'data', root / '.cache', package / 'tests', package / '.private'):
        directory.mkdir()
        (directory / 'hidden.py').write_text('do_not_disclose = True\n', encoding='utf-8')
    (root / '.env').write_text('PRIVATE_FAKE_CREDENTIAL=never-read\n', encoding='utf-8')
    (root / 'unrelated.py').write_text('do_not_disclose = True\n', encoding='utf-8')
    return root


def setup_tools(project, tmp_path, **kwargs):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case_id = store.create_case('Inspect the omitted configuration helper')['case_id']
    registry = EvidenceRegistry(store, case_id)
    budget = InvestigationBudget(store, case_id)
    tools = {tool.name: tool for tool in repository.build_repository_tools(project, registry, budget, **kwargs)}
    return tools, store, case_id, registry, budget


def test_inventory_is_a_deterministic_source_allowlist(project):
    context = repository.RepositoryContext(project)
    listing = context.inventory()
    assert [item['path'] for item in listing['files']] == [
        'app.py', 'triage_desk/generation.py', 'triage_desk/nested/helpers.py',
    ]
    assert not listing['truncated']
    assert context.inventory() == listing
    assert listing['files'][1]['size_bytes'] == len(HELPER.encode('utf-8'))


@pytest.mark.parametrize('path', [
    '../outside.py', '/etc/passwd', '/tmp/app.py', '.env', '.git/config', '.cache/private.py',
    'tests/hidden.py', 'data/hidden.py', 'unrelated.py', 'triage_desk/../.env', './app.py',
    'triage_desk//generation.py', 'triage_desk\\generation.py', 'triage_desk/generation.py\n',
    'triage_desk/.private/hidden.py', 'triage_desk/tests/hidden.py', 'triage_desk/readme.md',
    'triage_desk/nested/../../app.py',
])
def test_read_rejects_traversal_absolute_hidden_and_nonallowlisted_paths(project, path):
    with pytest.raises(repository.RepositoryContextError):
        repository.RepositoryContext(project).read(path)


def test_symlink_files_and_directory_components_cannot_escape_source_root(project, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'private.py').write_text('PRIVATE_OUTSIDE_BODY\n', encoding='utf-8')
    (project / 'triage_desk' / 'linked.py').symlink_to(outside / 'private.py')
    (project / 'triage_desk' / 'linked_dir').symlink_to(outside, target_is_directory=True)
    context = repository.RepositoryContext(project)
    assert not any('linked' in item['path'] for item in context.inventory()['files'])
    for path in ('triage_desk/linked.py', 'triage_desk/linked_dir/private.py'):
        with pytest.raises(repository.RepositoryContextError):
            context.read(path)
    (tmp_path / 'linked_root').symlink_to(project, target_is_directory=True)
    with pytest.raises(repository.RepositoryContextError):
        repository.RepositoryContext(tmp_path / 'linked_root')


def test_symlink_substitution_after_inventory_is_rejected_at_open(project, tmp_path):
    context = repository.RepositoryContext(project)
    assert any(item['path'] == 'triage_desk/generation.py' for item in context.inventory()['files'])
    target = tmp_path / 'private.py'
    target.write_text('PRIVATE_OUTSIDE_BODY\n', encoding='utf-8')
    (project / 'triage_desk' / 'generation.py').unlink()
    (project / 'triage_desk' / 'generation.py').symlink_to(target)
    with pytest.raises(repository.RepositoryContextError):
        context.read('triage_desk/generation.py')


def test_hardlinks_and_nonregular_files_are_not_source_read_candidates(project, tmp_path):
    outside = tmp_path / 'private.py'
    outside.write_text('PRIVATE_OUTSIDE_BODY\n', encoding='utf-8')
    os.link(outside, project / 'triage_desk' / 'linked.py')
    os.mkfifo(project / 'triage_desk' / 'pipe.py')
    context = repository.RepositoryContext(project)
    for path in ('triage_desk/linked.py', 'triage_desk/pipe.py'):
        with pytest.raises(repository.RepositoryContextError):
            context.read(path)


def test_read_preserves_crlf_and_full_file_excerpt_hashes_and_lines(project):
    snapshot = repository.RepositoryContext(project).read('triage_desk/generation.py', 3, 6)
    expected = ''.join(HELPER.splitlines(keepends=True)[2:6])
    assert snapshot['excerpt'] == expected
    assert snapshot['file_sha256'] == hashlib.sha256(HELPER.encode()).hexdigest()
    assert snapshot['excerpt_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
    assert snapshot['start_line'] == 3
    assert snapshot['end_line'] == 6
    assert snapshot['total_lines'] == 7
    assert snapshot['partial_file']
    assert not snapshot['truncated']


def test_file_modified_during_read_cannot_be_registered_as_consistent_snapshot(project, monkeypatch):
    context = repository.RepositoryContext(project)
    original = os.read
    changed = False

    def change_after_first_read(*args):
        nonlocal changed
        chunk = original(*args)
        if chunk and not changed:
            changed = True
            with (project / 'triage_desk' / 'generation.py').open('ab') as stream:
                stream.write(b'# concurrent modification\r\n')
        return chunk

    monkeypatch.setattr(repository.os, 'read', change_after_first_read)
    with pytest.raises(repository.RepositoryContextError, match='changed while being read'):
        context.read('triage_desk/generation.py')


def test_oversized_file_is_refused_before_body_is_read(project, monkeypatch):
    (project / 'triage_desk' / 'huge.py').write_bytes(b'x' * (repository.MAX_FILE_BYTES + 1))
    calls = []
    original = os.read

    def track_read(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(repository.os, 'read', track_read)
    with pytest.raises(repository.RepositoryContextError, match='size limit'):
        repository.RepositoryContext(project).read('triage_desk/huge.py')
    assert calls == []


def test_read_line_and_character_bounds_are_explicit(project):
    context = repository.RepositoryContext(project)
    for start, end in ((0, 5), (True, 5), (4, 2), (1, 161), (8, 9)):
        with pytest.raises(repository.RepositoryContextError):
            context.read('triage_desk/generation.py', start, end)
    (project / 'triage_desk' / 'long.py').write_text(('x' * 3000 + '\n') * 4, encoding='utf-8')
    snapshot = context.read('triage_desk/long.py')
    assert snapshot['end_line'] == 2
    assert snapshot['truncated']
    assert len(snapshot['excerpt']) <= repository.MAX_EXCERPT_CHARS
    (project / 'triage_desk' / 'single.py').write_text('x' * (repository.MAX_EXCERPT_CHARS + 1), encoding='utf-8')
    with pytest.raises(repository.RepositoryContextError, match='character limit'):
        context.read('triage_desk/single.py')


def test_search_is_literal_case_insensitive_and_lines_hashes_match(project):
    context = repository.RepositoryContext(project)
    matches = context.search('DEF _CONFIGURATION')['matches']
    assert len(matches) == 1
    assert matches[0]['path'] == 'triage_desk/generation.py'
    assert matches[0]['line_number'] == 3
    assert matches[0]['line'] == 'def _configuration():\r\n'
    assert matches[0]['file_sha256'] == hashlib.sha256(HELPER.encode()).hexdigest()
    regex_like = context.search('.*')['matches']
    assert len(regex_like) == 1
    assert regex_like[0]['path'] == 'triage_desk/nested/helpers.py'
    assert 'literal .* example' in regex_like[0]['line']


def test_search_results_files_entries_and_characters_are_bounded(project, monkeypatch):
    for index in range(20):
        (project / 'triage_desk' / f'item{index:02}.py').write_text(('needle here\n' * 10), encoding='utf-8')
    context = repository.RepositoryContext(project)
    result = context.search('needle')
    assert len(result['matches']) == repository.MAX_SEARCH_RESULTS
    assert result['truncated']
    monkeypatch.setattr(repository, 'MAX_SEARCH_CHARS', 70)
    result = context.search('missing')
    assert result['scanned_characters'] <= 70
    assert result['truncated']
    monkeypatch.setattr(repository, 'MAX_FILES', 3)
    listing = context.inventory()
    assert len(listing['files']) == 3
    assert listing['truncated']
    monkeypatch.setattr(repository, 'MAX_DIRECTORY_ENTRIES', 2)
    listing = context.inventory()
    assert len(listing['files']) <= 3
    assert listing['truncated']


def test_search_has_total_byte_cap_even_without_matches(project, monkeypatch):
    monkeypatch.setattr(repository, 'MAX_SEARCH_BYTES', 200)
    result = repository.RepositoryContext(project).search('missing')
    assert result['scanned_bytes'] <= 200
    assert result['truncated']


def test_large_early_module_cannot_starve_imported_helper_discovery(project, monkeypatch):
    (project / 'app.py').write_text('# large application\n' * 1000, encoding='utf-8')
    monkeypatch.setattr(repository, 'MAX_SEARCH_CHARS', 1000)
    result = repository.RepositoryContext(project).search('def _configuration')
    assert [(item['path'], item['line_number']) for item in result['matches']] == [
        ('triage_desk/generation.py', 3),
    ]
    assert result['scanned_characters'] <= 1000
    assert result['truncated']


def test_search_long_line_shows_bounded_literal_match_at_reported_column(project):
    line = 'x' * 2000 + 'literal_target' + 'z' * 2000 + '\n'
    (project / 'triage_desk' / 'long.py').write_text(line, encoding='utf-8')
    match = repository.RepositoryContext(project).search('literal_target')['matches'][0]
    assert 'literal_target' in match['line']
    assert len(match['line']) <= repository.MAX_SEARCH_LINE_CHARS
    start = match['column_start'] - 1
    assert match['line'] == line[start:start + repository.MAX_SEARCH_LINE_CHARS]
    assert match['line_truncated']


@pytest.mark.parametrize('secret', ['configured-credential-123456789', 'sk-liveAbcdef12345678901234567890'])
def test_sensitive_file_is_withheld_entirely_from_read_search_and_evidence(project, tmp_path, secret):
    content = f'# ordinary discovery term\nconfigured = {secret!r}\n'
    (project / 'triage_desk' / 'sensitive.py').write_text(content, encoding='utf-8')
    tools, store, case_id, registry, budget = setup_tools(project, tmp_path, forbidden_values=[secret])
    search = tools['search_repository'].invoke({'query': 'ordinary discovery term'})
    assert search['matches'] == []
    assert search['truncated']
    result = tools['read_repository_file'].invoke({'path': 'triage_desk/sensitive.py'})
    assert result['status'] == 'error'
    assert result['evidence'] == []
    assert secret not in json.dumps(result)
    assert secret not in json.dumps(store.events(case_id))
    assert registry.list_sources() == []
    assert budget.tool_calls == 2


def test_literal_credential_assignment_is_withheld_without_environment_discovery(project):
    (project / 'triage_desk' / 'credentials.py').write_text('api_key = "unconfigured-sensitive-literal"\n', encoding='utf-8')
    context = repository.RepositoryContext(project)
    with pytest.raises(repository.RepositoryContextError, match='credential'):
        context.read('triage_desk/credentials.py')


def test_only_explicit_read_registers_repository_snapshot_and_counts_once(project, tmp_path):
    tools, store, case_id, registry, budget = setup_tools(project, tmp_path)
    assert set(tools) == {'list_repository_files', 'search_repository', 'read_repository_file'}
    assert tools['list_repository_files'].invoke({})['status'] == 'ok'
    assert tools['search_repository'].invoke({'query': '_configuration'})['status'] == 'ok'
    assert registry.list_sources() == []
    result = tools['read_repository_file'].invoke({'path': 'triage_desk/generation.py', 'start_line': 3, 'end_line': 7})
    assert result['status'] == 'ok'
    record = result['evidence'][0]
    assert record['origin'] == 'repository'
    assert record['kind'] == 'code'
    assert record['filename'] == 'triage_desk/generation.py'
    assert record['evidence_type'] == 'repository_excerpt'
    assert record['url'] == ''
    assert 'triage_desk/generation.py:3-7' in record['text']
    assert result['file']['file_sha256'] in record['text']
    assert result['file']['excerpt_sha256'] in record['text']
    assert 'def _configuration():\r\n' in record['text']
    assert record['quote_spans']
    assert all(span['quote'] in record['text'] for span in record['quote_spans'])
    assert len(registry.list_sources()) == 1
    repeated = tools['read_repository_file'].invoke({'path': 'triage_desk/generation.py', 'start_line': 3, 'end_line': 7})
    assert repeated['evidence'][0]['evidence_id'] == record['evidence_id']
    assert len(registry.list_sources()) == 1
    assert budget.tool_calls == 4
    assert len([event for event in store.events(case_id) if event['phase'] == 'tool']) == 4


def test_changed_file_produces_new_hash_and_immutable_evidence(project, tmp_path):
    tools, store, case_id, registry, budget = setup_tools(project, tmp_path)
    first = tools['read_repository_file'].invoke({'path': 'triage_desk/generation.py'})
    (project / 'triage_desk' / 'generation.py').write_bytes((HELPER + '# changed working tree\r\n').encode())
    second = tools['read_repository_file'].invoke({'path': 'triage_desk/generation.py'})
    assert first['file']['file_sha256'] != second['file']['file_sha256']
    assert first['evidence'][0]['evidence_id'] != second['evidence'][0]['evidence_id']
    assert len(registry.list_sources()) == 2
    assert '# changed working tree' not in registry.list_sources()[0]['text']


def test_tool_error_is_safe_and_budget_limits_prevent_further_file_reads(project, tmp_path):
    tools, store, case_id, registry, budget = setup_tools(project, tmp_path)
    result = tools['read_repository_file'].invoke({'path': '/etc/passwd'})
    assert result['status'] == 'error'
    assert str(project) not in json.dumps(result)
    assert registry.list_sources() == []
    budget.max_tool_calls = 1
    from triage_desk.agent_budget import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        tools['read_repository_file'].invoke({'path': 'triage_desk/generation.py'})
    assert registry.list_sources() == []
