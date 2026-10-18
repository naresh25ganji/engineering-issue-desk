"""Exercise actual user-facing form flows without a browser or network."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_loads_and_requires_a_report():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    assert not app.exception
    assert 'Investigate a Streamlit issue' in app.title[0].value
    next(b for b in app.button if b.label == 'Investigate report').click().run()
    assert not app.exception
    assert any('Enter a report' in w.value for w in app.warning)


def test_example_report_can_be_investigated_with_keyword_search():
    assert (ROOT / 'data' / 'sources.json').exists(), 'Bundled real corpus must be included'
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    next(b for b in app.button if b.label == 'CSV upload').click().run()
    assert 'st.file_uploader' in app.text_area(key='report').value
    next(s for s in app.selectbox if s.label == 'Search strategy').select('Keyword').run()
    next(b for b in app.button if b.label == 'Investigate report').click().run()
    assert not app.exception
    assert not app.error
    analysis = app.session_state['analysis']
    assert analysis['status'] == 'evidence_found'
    assert analysis['evidence']
    assert '## Retrieved context' in analysis['draft_markdown']


def test_empty_source_selection_does_not_execute_a_search():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
    next(b for b in app.button if b.label == 'Widget state').click().run()
    next(s for s in app.multiselect if s.label == 'Evidence sources').set_value([]).run()
    next(b for b in app.button if b.label == 'Investigate report').click().run()
    assert not app.exception
    assert any('at least one evidence source' in w.value for w in app.warning)
