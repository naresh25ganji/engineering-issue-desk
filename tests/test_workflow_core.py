import pytest

from triage_desk import TriageDesk


def test_workflow_passages_cannot_replace_source_authority(corpus_dir, tmp_path):
    desk = TriageDesk(corpus_dir, tmp_path / 'cache')
    evidence = desk.search('cache_resource shared database connection', mode='keyword')
    evidence[0]['title'] = 'A fabricated official guarantee'
    evidence[0]['url'] = 'https://untrusted.example/leak'
    result = desk.analyze('cache_resource shared database connection', mode='keyword', _evidence=evidence)
    assert result['evidence'][0]['title'] != 'A fabricated official guarantee'
    assert result['evidence'][0]['url'].startswith('https://docs.example.test/')


def test_workflow_cannot_supply_an_invented_source_passage(corpus_dir, tmp_path):
    desk = TriageDesk(corpus_dir, tmp_path / 'cache')
    evidence = desk.search('cache_resource shared database connection', mode='keyword')
    evidence[0]['text'] = 'This invented quote says your application is definitely fixed.'
    with pytest.raises(ValueError, match='literal passage'):
        desk.analyze('cache_resource shared database connection', mode='keyword', _evidence=evidence)
