from streamlit.testing.v1 import AppTest


def workspace(root):
    from pathlib import Path
    from triage_desk.project_guide import render_project_guide
    render_project_guide(Path(root))


def test_guide_displays_authored_assets_and_switches_without_provider_calls(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'week3-system-architecture.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Architecture</text></svg>')
    (docs / 'week3-investigation-sequence.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Sequence</text></svg>')
    (docs / 'ARCHITECTURE.md').write_text('# Components\nSee [design](DESIGN.md) and [primary docs](https://docs.python.org/3/).\n![Components](week3-system-architecture.svg)')
    (docs / 'OPERATIONS.md').write_text('# Recovery\nRetry saved cases explicitly.')
    app = AppTest.from_function(workspace, args=(str(tmp_path),)).run()
    assert not app.exception
    assert any('Components' in element.value for element in app.markdown)
    assert any('design (`DESIGN.md`)' in element.value for element in app.markdown)
    assert any('[primary docs](https://docs.python.org/3/)' in element.value for element in app.markdown)
    assert any('Diagram reference: Components' in element.value for element in app.markdown)
    app.selectbox(key='project_diagram').select('Investigation sequence')
    app.selectbox(key='project_document').select('Operations').run()
    assert not app.exception
    assert any('Retry saved cases explicitly.' in element.value for element in app.markdown)


def test_missing_assets_have_clear_messages_instead_of_crashing(tmp_path):
    app = AppTest.from_function(workspace, args=(str(tmp_path),)).run()
    assert not app.exception
    assert len(app.info) == 2
