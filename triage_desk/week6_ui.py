"""Read-only inspection of recorded red-team cases; never dispatches a model."""
from __future__ import annotations

import streamlit as st


def render_week6_evaluation(document: dict) -> None:
    st.markdown('**Week 6 red-team evaluation**')
    live = document.get('execution_mode') == 'live'
    rows = document.get('results', [])
    rows = [row for row in rows if isinstance(row, dict)]
    st.info('Scores are provisional checks of the recorded application contract. A PASS means the specified failure was not observed. Review the actual responses; this run does not establish general security.')
    if not live:
        st.caption('Input inventory only. No model was called and no attack outcome was measured.')
    counts = st.columns(4)
    counts[0].metric('Scheduled cases', len(rows))
    for column, score in zip(counts[1:], ('PASS', 'WARN', 'FAIL'), strict=True):
        column.metric(score, sum(row.get('score') == score for row in rows) if live else 'Not measured')
    families = sorted({row.get('family', 'unknown') for row in rows})
    family = st.selectbox('Attack family', ['All families', *families], key='week6_family')
    score = st.selectbox('Result', ['All results', 'PASS', 'WARN', 'FAIL', 'Not scored'], key='week6_score')
    visible = [row for row in rows if (family == 'All families' or row.get('family') == family)
               and (score == 'All results' or (row.get('score') or 'Not scored') == score)]
    if not visible:
        st.info('No recorded cases match these filters.')
        return
    st.dataframe([{'Case': row.get('id'), 'Family': row.get('family'), 'Title': row.get('title'),
                   'Status': row.get('status'), 'Score': row.get('score') or 'Not scored',
                   'Reason': row.get('reason', ''), 'Turns': len(row.get('turns', []))}
                  for row in visible], hide_index=True, width='stretch')
    chosen = st.selectbox('Inspect red-team case', visible,
                          format_func=lambda row: f"{row.get('id')} · {row.get('title')}", key='week6_case')
    st.text(chosen.get('reason', 'Not scored'))
    st.text('Expected boundary: ' + str(chosen.get('expected_boundary', 'Not exported')))
    for index, turn in enumerate(chosen.get('turns', []), 1):
        with st.expander(f'Turn {index} · {turn.get("status", "recorded")}', expanded=True):
            st.text('Prompt')
            st.text(turn.get('prompt', 'Not recorded'))
            context = turn.get('context', '') or (chosen.get('input', {}).get('context', '') if index == 1 else '')
            if context:
                st.text('Supplied context')
                st.text(context)
            st.text('Actual saved response' if turn.get('status') == 'awaiting_review'
                    else 'Last saved draft; this turn did not complete a new response')
            st.text(turn.get('response', 'No response recorded'))
    with st.expander('Evidence, checks, and local trace', expanded=False):
        st.json(chosen)
    with st.expander('Run identity and measurement limits', expanded=False):
        st.json({'manifest': document.get('manifest'), 'summary': document.get('summary'),
                 'measurement_limits': document.get('measurement_limits')})
    st.caption('Public tool names are documented capabilities. Cross-case tests use synthetic fixtures. Saved revisions test escalation through the existing review workflow; they are not ordinary chat turns.')
