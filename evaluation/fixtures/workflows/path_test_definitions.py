# Frozen source excerpt: tests/test_repository_context.py:61-70
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
