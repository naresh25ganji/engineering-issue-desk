from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_keyword_chat_records_followup_and_keeps_assistant_out_of_history():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
    next(item for item in app.selectbox if item.label == 'Search strategy').select('Keyword').run()
    app.chat_input[0].set_value('How does st.cache_data handle serializable data?').run()
    assert not app.exception
    assert len(app.session_state['chat_records']) == 1
    app.chat_input[0].set_value('What about that API and shared resources?').run()
    assert not app.exception
    records = app.session_state['chat_records']
    assert len(records) == 2
    assert records[1]['result']['workflow']['history_user_turns_used'] == 1
    assert records[1]['result']['workflow']['history_is_evidence'] is False
