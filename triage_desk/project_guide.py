"""Browse the project's authored design and validation assets without providers."""
from __future__ import annotations

from pathlib import Path
import re

import streamlit as st


_DOCUMENTS = {
    'Architecture': 'ARCHITECTURE.md',
    'Design decisions': 'DESIGN.md',
    'Operations': 'OPERATIONS.md',
    'Validation': 'VALIDATION.md',
    'Week 3 submission': 'WEEK3_SUBMISSION.md',
    'Submission readiness': 'SUBMISSION_READINESS.md',
}
_DIAGRAMS = {
    'System architecture': 'week3-system-architecture.svg',
    'Investigation sequence': 'week3-investigation-sequence.svg',
}
_LOCAL_LINK = re.compile(r'(!?)\[([^\]\n]+)\]\((?!https?://|#)([^)\n]+)\)')


def render_project_guide(root: Path) -> None:
    """Read only fixed, authored project assets; never inspect a case or .env."""
    docs = Path(root) / 'docs'
    st.subheader('Project guide')
    st.caption('Architecture, design decisions, operating instructions, and recorded validation for this local engineering workspace.')
    diagram = st.selectbox('Diagram', list(_DIAGRAMS), key='project_diagram')
    diagram_path = docs / _DIAGRAMS[diagram]
    try:
        content = diagram_path.read_text(encoding='utf-8')
        st.image(content, caption=diagram, width='stretch')
        st.download_button('Download diagram', content, file_name=diagram_path.name, mime='image/svg+xml')
    except (OSError, UnicodeError):
        st.info('This project diagram is not available in this checkout.')
    st.divider()
    selected = st.selectbox('Project document', list(_DOCUMENTS), key='project_document')
    path = docs / _DOCUMENTS[selected]
    try:
        content = path.read_text(encoding='utf-8')
        st.download_button('Download document', content, file_name=path.name, mime='text/markdown')
        # Relative repository links are file references here, not Streamlit routes.
        display = _LOCAL_LINK.sub(lambda match: ('Diagram reference: ' if match[1] else '')
                                  + match[2] + ' (`' + match[3] + '`)', content)
        st.markdown(display)
    except (OSError, UnicodeError):
        st.info('This project document is not available in this checkout.')
