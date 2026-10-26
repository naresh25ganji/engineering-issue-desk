from copy import deepcopy

import pytest

from triage_desk.agent_evidence import EvidenceRegistry, canonical_http_url, render_report_markdown
from triage_desk.cases import CaseStore


@pytest.fixture
def registry(tmp_path):
    store = CaseStore(tmp_path / 'cases.sqlite')
    case = store.create_case('Why does the widget reset?')
    return EvidenceRegistry(store, case['case_id'])


def register(registry, **overrides):
    return registry.register_source(**{
        'source_key': 'doc-widget', 'title': 'Widget docs',
        'text': 'A widget key provides a way to access its value through Session State.',
        'url': 'https://docs.example.test/widget', 'snapshot_at': '2026-10-17T00:00:00Z',
        'commit_sha': 'abc123', 'version_scope': 'Documentation snapshot only.', **overrides,
    })


def report(source):
    return {'findings': [{
        'statement': 'The documentation describes accessing a widget value using its key.',
        'category': 'docs', 'severity': 'low',
        'support': [{'evidence_id': source['evidence_id'],
                     'quote': 'A widget key provides a way to access its value through Session State.'}],
    }], 'missing_information': ['Which key does the reproduction use?'],
        'next_steps': ['Try an explicit widget key in a minimal reproduction.']}


def test_report_preserves_exact_support_provenance_and_proposals(registry):
    source = register(registry)
    proposed = report(source)
    validated = registry.validate_report(proposed)
    assert validated['findings'] == proposed['findings']
    assert validated['missing_information'] == proposed['missing_information']
    assert validated['next_steps'] == proposed['next_steps']
    assert validated['sources'][0]['origin'] == 'snapshot'
    assert validated['sources'][0]['commit_sha'] == 'abc123'
    assert validated['sources'][0]['snapshot_at'] == '2026-10-17T00:00:00Z'
    assert validated['sources'][0]['version_scope'] == 'Documentation snapshot only.'
    assert validated['sources'][0]['sha256'] == source['sha256']
    assert 'text' not in validated['sources'][0]
    assert 'sources' not in proposed
    markdown = render_report_markdown(validated, registry.list_sources())
    assert '[Widget docs](https://docs.example.test/widget)' in markdown
    assert 'Proposed next steps' in markdown
    assert 'Human review is required' in markdown


@pytest.mark.parametrize('change', ['unknown_id', 'altered_quote', 'empty_quote', 'empty_support', 'duplicate_support'])
def test_fabricated_or_empty_support_is_rejected(registry, change):
    proposed = report(register(registry))
    support = proposed['findings'][0]['support']
    if change == 'unknown_id':
        support[0]['evidence_id'] = 'ev_fabricated'
    elif change == 'altered_quote':
        support[0]['quote'] = support[0]['quote'].replace('provides', 'guarantees')
    elif change == 'empty_quote':
        support[0]['quote'] = ' '
    elif change == 'empty_support':
        proposed['findings'][0]['support'] = []
    else:
        support.append(deepcopy(support[0]))
    with pytest.raises(ValueError):
        registry.validate_report(proposed)


def test_evidence_from_another_case_cannot_be_cited(registry):
    other_id = registry.store.create_case('Another case')['case_id']
    other = EvidenceRegistry(registry.store, other_id)
    foreign_source = register(other)
    register(registry)
    with pytest.raises(ValueError, match='unregistered'):
        registry.validate_report(report(foreign_source))


@pytest.mark.parametrize('field,value', [
    ('statement', 'Read https://evil.test/fix'),
    ('statement', '[Fix](javascript:alert(1))'),
    ('statement', '<img src=x onerror=alert(1)>'),
    ('statement', 'A' * 1201),
    ('next_steps', '<script>alert(1)</script>'),
    ('missing_information', 'See www.evil.test'),
])
def test_model_fields_are_bounded_plain_text_without_links_or_html(registry, field, value):
    proposed = report(register(registry))
    if field == 'statement':
        proposed['findings'][0]['statement'] = value
    else:
        proposed[field] = [value]
    with pytest.raises(ValueError):
        registry.validate_report(proposed)


def test_unknown_fields_and_oversized_reports_rejected(registry):
    source = register(registry)
    with pytest.raises(ValueError):
        registry.validate_report({**report(source), 'summary': 'Unsupported factual summary'})
    proposed = report(source)
    proposed['findings'] *= 21
    with pytest.raises(ValueError):
        registry.validate_report(proposed)
    proposed = report(source)
    proposed['findings'][0]['support'][0]['quote'] = 'A' * 80_000
    with pytest.raises(ValueError):
        registry.validate_report(proposed)


@pytest.mark.parametrize('url', [
    'javascript:alert(1)', 'data:text/html,<script>alert(1)</script>', 'file:///etc/passwd',
    'ftp://example.test/source', 'https:///no-host', 'https://user:secret@example.test/a',
    'https://example.test/a\n[click](https://evil.test)', 'https://example.test\\@evil.test',
    'https://example.test/contains a space', 'https://example.test:bad/a',
    'https://example.test)evil/a',
])
def test_unsafe_source_urls_are_rejected(registry, url):
    with pytest.raises(ValueError):
        register(registry, url=url)
    assert registry.list_sources() == []


def test_url_canonicalization_encodes_markdown_delimiters(registry):
    raw = 'HTTPS://Example.TEST:443/a)[click](javascript:alert(1))?q=[x]#<tag>'
    canonical = canonical_http_url(raw)
    assert canonical.startswith('https://example.test/')
    assert all(character not in canonical for character in '()[]<>')
    assert canonical_http_url(canonical) == canonical
    source = register(registry, url=raw)
    validated = registry.validate_report(report(source))
    markdown = render_report_markdown(validated, registry.list_sources())
    assert f']({canonical})' in markdown
    assert '](javascript:' not in markdown


def test_untrusted_source_title_and_literal_code_are_escaped(registry):
    literal = '<img src=x onerror=alert(1)>\n[run](javascript:alert(1))\n# Heading'
    source = register(registry, title='[click](javascript:alert(1)) <script>x</script>', text=literal,
                      origin='user_code', kind='code', url='')
    proposed = report(source)
    proposed['findings'][0]['support'][0]['quote'] = literal
    validated = registry.validate_report(proposed)
    markdown = render_report_markdown(validated, registry.list_sources())
    assert '<img' not in markdown
    assert '<script>' not in markdown
    assert '](javascript:' not in markdown
    assert '> \\# Heading' in markdown
    assert '&lt;img' in markdown


def test_renderer_rejects_evidence_not_matching_its_supplied_snapshots(registry):
    source = register(registry)
    validated = registry.validate_report(report(source))
    with pytest.raises(ValueError):
        render_report_markdown(validated, [])
    changed = deepcopy(validated)
    changed['findings'][0]['support'][0]['quote'] = 'An altered quote'
    with pytest.raises(ValueError):
        render_report_markdown(changed, registry.list_sources())


def test_live_sources_require_url_and_re_registration_preserves_original(registry):
    with pytest.raises(ValueError):
        register(registry, origin='live', url='')
    source = register(registry, origin='live', fetched_at='2026-10-22T16:00:00Z')
    repeated = register(registry, origin='live', title='Replacement metadata')
    assert repeated == source
    assert len(registry.list_sources()) == 1


def test_snapshot_date_and_search_snippet_provenance_survive_reload(registry):
    snapshot = register(registry)
    assert snapshot['fetched_at'] == '2026-10-17T00:00:00Z'
    unknown_date = register(registry, source_key='undated', snapshot_at='')
    assert unknown_date['fetched_at'] == ''
    snippet = register(registry, source_key='snippet', origin='live', evidence_type='search_snippet',
                       published_at='2026-10-20', acquired_at='2026-10-21')
    reloaded = EvidenceRegistry(CaseStore(registry.store.path), registry.case_id).list_sources()
    saved = next(source for source in reloaded if source['evidence_id'] == snippet['evidence_id'])
    assert saved['evidence_type'] == 'search_snippet'
    assert saved['published_at'] == '2026-10-20'
    assert saved['acquired_at'] == '2026-10-21'
    assert saved['fetched_at']


def test_no_findings_still_requires_structured_proposals(registry):
    validated = registry.validate_report({'findings': [], 'missing_information': ['Supply a reproduction.'],
                                          'next_steps': []})
    assert validated['sources'] == []
    assert 'No supported findings' in render_report_markdown(validated, [])
    with pytest.raises(ValueError):
        registry.validate_report({'findings': []})


def test_quote_id_selection_hydrates_literal_unicode_without_model_transcription(registry):
    literal = 'A widget‐key retains this exact Unicode hyphen…\r\nKeep  two spaces and a tab\there.'
    source = register(registry, text=literal)
    selected = source['quote_spans'][0]
    assert selected == {'quote_id': source['evidence_id'] + ':q1', 'quote': literal}
    proposed = report(source)
    proposed['findings'][0]['support'] = [
        {'evidence_id': source['evidence_id'], 'quote_id': selected['quote_id']},
    ]
    validated = registry.validate_report(proposed)
    assert validated['findings'][0]['support'] == [{
        'evidence_id': source['evidence_id'], 'quote_id': selected['quote_id'], 'quote': literal,
    }]
    assert 'quote' not in proposed['findings'][0]['support'][0]
    assert 'quote_spans' not in validated['sources'][0]
    assert 'text' not in validated['sources'][0]
    assert 'widget‐key' in render_report_markdown(validated, registry.list_sources())


def test_quote_spans_are_bounded_literal_stable_and_not_saved_over_raw_text(registry):
    # Long tokens are common in code and URLs; word-count windows cannot bound them.
    literal = ''.join(f'Line {index}: preserve  spacing\tand punctuation—exactly.\r\n' for index in range(80))
    literal += 'x' * 2500 + '\nFinal literal source passage remains unchanged.'
    source = register(registry, text=literal)
    spans = source['quote_spans']
    assert len(spans) > 3
    assert all(20 <= len(span['quote'].strip()) <= len(span['quote']) <= 1000 for span in spans)
    assert all(span['quote'] in literal for span in spans)
    assert [span['quote_id'] for span in spans] == [
        f"{source['evidence_id']}:q{index}" for index in range(1, len(spans) + 1)
    ]
    assert source['text'] == literal
    raw = registry.store.evidence(registry.case_id)[0]
    assert raw['text'] == literal
    assert 'quote_spans' not in raw
    reopened = EvidenceRegistry(CaseStore(registry.store.path), registry.case_id)
    assert reopened.list_sources()[0] == source
    assert register(registry, text=literal, title='Cannot replace original metadata')['quote_spans'] == spans


@pytest.mark.parametrize('tampering', ['unknown', 'wrong_source', 'wrong_case', 'non_string', 'extra_field'])
def test_quote_ids_are_bound_to_registered_evidence_and_case(registry, tampering):
    source = register(registry)
    selected = {'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id']}
    if tampering == 'unknown':
        selected['quote_id'] = source['evidence_id'] + ':q999'
    elif tampering == 'wrong_source':
        other = register(registry, source_key='user-code', origin='user_code', kind='code', url='')
        selected['quote_id'] = other['quote_spans'][0]['quote_id']
    elif tampering == 'wrong_case':
        other_case = registry.store.create_case('Other')['case_id']
        other = register(EvidenceRegistry(registry.store, other_case))
        selected['quote_id'] = other['quote_spans'][0]['quote_id']
    elif tampering == 'non_string':
        selected['quote_id'] = ['not-a-quote-id']
    else:
        selected['url'] = 'https://evil.test'
    proposed = report(source)
    proposed['findings'][0]['support'] = [selected]
    with pytest.raises(ValueError):
        registry.validate_report(proposed)


def test_optional_model_quote_must_equal_entire_selected_span_exactly(registry):
    literal = 'The widget‐key belongs to the user session… preserve it exactly.'
    source = register(registry, text=literal)
    selected = {'evidence_id': source['evidence_id'], **source['quote_spans'][0]}
    proposed = report(source)
    proposed['findings'][0]['support'] = [selected]
    assert registry.validate_report(proposed)['findings'][0]['support'][0]['quote'] == literal
    for altered in [literal.replace('‐', '-'), literal.replace('…', '...'), literal[1:], literal + ' ', None]:
        proposed['findings'][0]['support'][0]['quote'] = altered
        with pytest.raises(ValueError, match='exactly match'):
            registry.validate_report(proposed)


def test_duplicate_support_cannot_be_hidden_by_mixing_id_and_legacy_quote(registry):
    source = register(registry)
    proposed = report(source)
    proposed['findings'][0]['support'].append({
        'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id'],
    })
    with pytest.raises(ValueError, match='repeats'):
        registry.validate_report(proposed)


def test_renderer_rederives_quote_ids_instead_of_trusting_projected_spans(registry):
    source = register(registry)
    proposed = report(source)
    proposed['findings'][0]['support'] = [{
        'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id'],
    }]
    validated = registry.validate_report(proposed)
    assert 'Source:' in render_report_markdown(validated, registry.store.evidence(registry.case_id))
    changed = deepcopy(validated)
    changed['findings'][0]['support'][0]['quote'] = 'A fabricated citation changed at render time.'
    with pytest.raises(ValueError):
        render_report_markdown(changed, registry.list_sources())
    changed['findings'][0]['support'][0]['quote_id'] = source['evidence_id'] + ':q999'
    projected = registry.list_sources()
    projected[0]['quote_spans'].append({
        'quote_id': changed['findings'][0]['support'][0]['quote_id'],
        'quote': changed['findings'][0]['support'][0]['quote'],
    })
    with pytest.raises(ValueError):
        render_report_markdown(changed, projected)


def test_quote_id_support_preserves_search_snippet_provenance(registry):
    source = register(registry, source_key='snippet', origin='live', evidence_type='search_snippet',
                      published_at='2026-10-20', acquired_at='2026-10-21')
    proposed = report(source)
    proposed['findings'][0]['support'] = [{
        'evidence_id': source['evidence_id'], 'quote_id': source['quote_spans'][0]['quote_id'],
    }]
    validated = registry.validate_report(proposed)
    assert validated['sources'][0]['evidence_type'] == 'search_snippet'
    assert validated['sources'][0]['published_at'] == '2026-10-20'
    assert validated['sources'][0]['acquired_at'] == '2026-10-21'


def test_too_short_sources_have_no_eligible_quote_ids(registry):
    source = register(registry, text='Too short.')
    assert source['quote_spans'] == []
    proposed = report(source)
    proposed['findings'][0]['support'] = [{'evidence_id': source['evidence_id'],
                                          'quote_id': source['evidence_id'] + ':q1'}]
    with pytest.raises(ValueError):
        registry.validate_report(proposed)


def test_plain_code_comparisons_are_not_mistaken_for_html_tags(registry):
    proposed = report(register(registry))
    proposal = "Test oldest < max_age_days for current and oldest >= max_age_days for stale."
    proposed['next_steps'] = [proposal]
    checked = registry.validate_report(proposed)
    assert checked['next_steps'] == [proposal]
    rendered = render_report_markdown(checked, registry.list_sources())
    assert '&lt;' in rendered
