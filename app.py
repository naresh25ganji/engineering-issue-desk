"""Operator workspace for a bounded, evidence-backed issue investigation."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

from triage_desk.freshness import corpus_freshness

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)

st.set_page_config(page_title='Issue Desk · Engineering workspace', page_icon='◈', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.block-container {padding-top:4.2rem;padding-bottom:3rem;max-width:1500px}
[data-testid="stSidebar"] {background:#142B2B;color:#E5EFEB}
[data-testid="stSidebar"] * {color:inherit}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#B5C8C1}
[data-testid="stSidebar"] hr {border-color:#36504B}
[data-testid="stSidebar"] [data-baseweb="select"] {color:#172532}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {color:#172532!important}
[data-testid="stSidebar"] input[role="combobox"] {color:#172532!important;-webkit-text-fill-color:#172532!important}
[data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] {color:#172532!important}
[data-testid="stSidebar"] [data-testid="stToggle"] p {color:#E5EFEB!important}
[data-testid="stSidebar"] [data-testid="stExpander"] {border-color:#36504B}
h1 {font-size:2.6rem!important;font-weight:700!important;letter-spacing:-1.4px!important}
h2 {font-size:1.4rem!important;letter-spacing:-.4px!important}
h3 {font-size:1.08rem!important;letter-spacing:-.1px!important}
[data-testid="stMetricValue"] {font-family:'IBM Plex Mono',monospace;font-size:1.5rem}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:12px}
button {border-radius:7px!important}
.eyebrow {color:#51736A;font:500 11px 'IBM Plex Mono',monospace;letter-spacing:2px;margin:0 0 5px}
.brand {font-size:23px;font-weight:700;letter-spacing:-.7px;margin-top:8px}
.brandmark {display:inline-flex;align-items:center;justify-content:center;width:37px;height:37px;background:#CBF0DC;color:#142B2B;border-radius:9px;margin-right:8px;font:500 22px monospace}
.sidebar-label {color:#AFC5BD;font:500 10px 'IBM Plex Mono',monospace;letter-spacing:1.6px;margin-top:28px}
.workspace-repo {font:400 13px 'IBM Plex Mono',monospace;padding:10px 0;color:#E4F0EA}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:22px}
[data-testid="stTabs"] [data-baseweb="tab"] {font-size:14px;font-weight:600;padding:0 3px 10px}
textarea {font-size:15px!important;line-height:1.6!important}
[data-testid="stCodeBlock"] pre {font-family:'IBM Plex Mono',monospace;font-size:12px}
a {text-underline-offset:3px}
</style>''', unsafe_allow_html=True)

EXAMPLES = {
    'CSV upload': 'My Streamlit app accepts a CSV using st.file_uploader. After I interact with another widget, the uploaded data seems to reset. How should I preserve the data across reruns, and what details should I include in a bug report?',
    'Widget state': 'A widget disappears when I switch pages and its value is lost. Is this expected Streamlit widget behavior? Which documentation should I check before reporting a bug?',
    'Caching error': 'I used st.cache_data on a function that takes an object and now I get an UnhashableParamError. How do I investigate the error and prepare a useful report?',
}


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def safe_url(url: str) -> str:
    parsed = urlparse(url or '')
    return url if parsed.scheme in {'http', 'https'} and parsed.netloc else ''


def report_summaries(value: dict, prefix: str = ''):
    """Handle both single-run and dataset/configuration comparison reports."""
    if not isinstance(value, dict):
        return
    if any(key in value for key in ('hit_at_k', 'hit_at_6', 'family_recall_at_k', 'questions')):
        yield prefix, value
        return
    for name, child in value.items():
        if isinstance(child, dict):
            yield from report_summaries(child, (prefix + ' / ' if prefix else '') + name)


def render_evaluation_summary(document: dict):
    """Display measured agent handoffs or the existing retrieval diagnostics."""
    if document.get('report_kind') == 'deep_agent_workflow':
        summary = document.get('summary', {})
        st.markdown('**Deep Agents workflow run**')
        st.caption('Saved human-review handoffs across all scheduled cases, including failures. Handoff completion does not measure answer accuracy.')
        st.caption('Related application code option: ' + ('Enabled' if document.get('repository_context_enabled') else 'Not enabled'))
        columns = st.columns(4)
        columns[0].metric('Scheduled cases', summary.get('scheduled_cases', 0))
        columns[1].metric('Saved handoffs', summary.get('completed_handoffs', 0))
        columns[2].metric('Failed cases', summary.get('failed_cases', 0))
        rate = summary.get('handoff_rate')
        columns[3].metric('Handoff completion', f'{rate:.0%}' if rate is not None else 'Not run')
        checks = st.columns(3)
        checks[0].metric('Cases with exact citations', summary.get('citation_integrity_passed_cases', 0))
        checks[1].metric('Findings removed by AI review', summary.get('ai_rejected_findings', 0))
        checks[2].metric('Next steps removed by AI review', summary.get('ai_rejected_next_steps', 0))
        calls = st.columns(3)
        calls[0].metric('Model calls', summary.get('model_calls', 0))
        calls[1].metric('Tool calls', summary.get('tool_calls', 0))
        calls[2].metric('Reported tokens', summary.get('reported_tokens', 0))
        usage_complete = summary.get('usage_complete')
        if usage_complete is False:
            st.caption('Token usage is incomplete. Displayed tokens include only reported counts; missing usage and cost are not estimated.')
        elif usage_complete is True:
            st.caption('Token totals sum reported model usage. Call counts reflect recorded dispatches; no cost is inferred.')
        else:
            st.caption('Validation-only inventory. No provider execution or quality score is recorded.')
        st.info('Exact citation checks establish saved quotes and source identities. AI evidence review remains a model judgment. Human accuracy, relevance, and citation entailment are unscored.')
        rows = []
        for item in document.get('results', []):
            integrity = item.get('automatic_citation_integrity', {})
            review = item.get('independent_ai_review') or (item.get('report') or {}).get('quality_review') or {}
            error = item.get('error') or {}
            latency = item.get('elapsed_ms')
            rows.append({
                'Evaluation case': item.get('eval_id', item.get('id', '')),
                'Saved case': item.get('case_id') or 'Not created',
                'Status': item.get('status', 'unknown'),
                'Human handoff': 'Saved' if item.get('handoff_observed') else 'Not observed' if item.get('attempted') else 'Not run',
                'Citation integrity': integrity.get('status', 'not_applicable'),
                'Supports checked': integrity.get('checked_supports', 0),
                'Failed supports': integrity.get('failed_supports', 0),
                'AI evidence review': review.get('status', 'Not recorded'),
                'Related application code': 'Enabled' if item.get('repository_context') else 'Not enabled',
                'Time': f'{latency / 1000:.1f} s' if latency is not None else 'Not run',
                'Error': (error.get('type', 'Recorded error') + (': ' + error['code'] if error.get('code') else '')) if error else '',
                'Human quality': 'Unscored',
            })
        if rows:
            st.dataframe(rows, hide_index=True, width='stretch')
        with st.expander('Agent call accounting'):
            st.json({key: summary.get(key) for key in (
                'model_calls_by_role', 'tool_calls_by_name', 'errors_by_role_or_tool',
                'reported_token_usage', 'model_calls_missing_usage_by_field',
                'unfinished_call_records', 'accounting_unknown_error_rows')}, expanded=True)
        return

    summary_rows = []
    for mode, scores in report_summaries(document.get('summary', {})):
        def percent(key, alias=''):
            value = scores.get(key, scores.get(alias))
            return f'{value * 100:.0f}%' if value is not None else 'Not scored'
        latency = scores.get('retrieval_latency_p95_ms')
        mrr = scores.get('mean_reciprocal_rank')
        summary_rows.append({'Strategy': mode.title(), 'Expected source found': percent('hit_at_k', 'hit_at_6'), 'Gold source precision': percent('gold_source_precision_at_k', 'gold_source_precision_at_6'), 'First expected source (MRR)': f'{mrr:.2f}' if mrr is not None else 'Not scored', 'Behavior checks': percent('behavior_pass_rate'), 'Reference integrity': percent('reference_integrity_pass_rate'), 'Warm retrieval p95': f'{latency:.0f} ms' if latency is not None else 'Not scored'})
    if summary_rows:
        st.dataframe(summary_rows, hide_index=True, width='stretch')
        unit = 'sources' if document.get('report_kind') == 'graph_source_retrieval_comparison' else 'chunks'
        count = document.get('question_count', sum(item.get('total_cases', 0) for item in document.get('datasets', [])))
        st.caption(f"{count} diagnostic questions · top {document.get('top_k', document.get('source_budget', 6))} {unit}. Source labels cover selected expected references. These scores do not measure model faithfulness.")


@st.cache_resource(show_spinner=False)
def engine(corpus_mtime: float):
    from triage_desk import TriageDesk
    return TriageDesk(data_dir=ROOT / 'data', cache_dir=ROOT / '.cache' / 'retrieval')


@st.cache_resource(show_spinner=False)
def knowledge_graph(corpus_mtime: float):
    from triage_desk.knowledge_graph import EngineeringKnowledgeGraph
    return EngineeringKnowledgeGraph(read_json(ROOT / 'data' / 'sources.json', []))


@st.cache_resource(show_spinner=False)
def guided_engine(corpus_mtime: float, embedding_provider: str):
    from triage_desk.retrieval_lab import RetrievalLab
    from triage_desk.workflow import InvestigationWorkflow
    factory = None
    if embedding_provider == 'Nebius course credits':
        from triage_desk.hosted_embeddings import HostedDenseIndex
        factory = HostedDenseIndex
    desk = engine(corpus_mtime)
    lab = RetrievalLab(desk.sources, ROOT / '.cache' / 'retrieval', dense_factory=factory)
    return InvestigationWorkflow(desk, lab, knowledge_graph(corpus_mtime)), lab


def investigate(question: str, settings: dict, *, history=None, version: str = '') -> dict:
    timestamp = (ROOT / 'data' / 'sources.json').stat().st_mtime
    if settings['guided']:
        workflow, _ = guided_engine(timestamp, settings['embedding_provider'])
        result = workflow.run(question, mode=settings['mode'], source_kinds=settings['kinds'],
                              version=version, include_generation=settings['generation'],
                              strategy=settings['strategy'], rerank=settings['rerank'],
                              use_graph=settings['use_graph'], history=history)
    else:
        result = engine(timestamp).analyze(question, mode=settings['mode'], source_kinds=settings['kinds'],
                                          version=version, include_generation=settings['generation'])
    result['run_id'] = uuid.uuid4().hex
    result['first_contact'] = not history
    return result


def select_example(label: str):
    st.session_state['report'] = EXAMPLES[label]
    st.session_state.pop('analysis', None)


def main():
    sources = read_json(ROOT / 'data' / 'sources.json', [])
    manifest = read_json(ROOT / 'data' / 'manifest.json', {})
    configured = bool((os.getenv('TRIAGE_API_KEY') or os.getenv('NEBIUS_API_KEY')) and (os.getenv('TRIAGE_MODEL') or os.getenv('NEBIUS_MODEL')))
    freshness = corpus_freshness(sources)
    snapshot_raw = str(freshness.get('oldest_snapshot_at') or manifest.get('snapshot_at') or manifest.get('fetched_at') or manifest.get('generated_at') or '')
    try:
        snapshot = datetime.fromisoformat(snapshot_raw.replace('Z', '+00:00')).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%b %d, %Y PT')
    except ValueError:
        snapshot = snapshot_raw[:10] or 'Preparing corpus'

    with st.sidebar:
        st.markdown('<div class="brand"><span class="brandmark">◈</span>Issue Desk</div>', unsafe_allow_html=True)
        st.caption('Engineering issue triage')
        st.markdown('<div class="sidebar-label">WORKSPACE</div><div class="workspace-repo">streamlit / streamlit</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('**Investigation settings**')
        st.caption('These settings apply to Investigate and Chat. Deep Investigation has its own provider setup.')
        mode_label = st.selectbox('Search strategy', ['Hybrid', 'Semantic', 'Keyword'], help='Hybrid combines local semantic embeddings with BM25 keyword search. Keyword is the offline comparison baseline.')
        mode = mode_label.lower()
        kinds = st.multiselect('Evidence sources', ['documentation', 'issue', 'release', 'contributing'], default=['documentation', 'issue', 'release', 'contributing'], format_func=lambda s: {'documentation': 'Documentation', 'issue': 'Issue history', 'release': 'Release notes', 'contributing': 'Contribution guide'}[s])
        use_generation = st.toggle('Nebius generated answer', value=False, disabled=not configured, help='Send this report and retrieved passages to Nebius. A second model call checks claims against selected quotes; unsupported claims are removed. Requires a server-side API key and model ID.')
        with st.expander('Retrieval and workflow options'):
            guided = st.checkbox('Guided LangGraph workflow', value=True, help='Route, retrieve, assess evidence, generate, verify support, and prepare a human handoff.')
            strategy_label = st.selectbox('Chunking strategy', ['Structured passages', 'Fixed word baseline'], disabled=not guided)
            embedding_provider = st.selectbox('Embedding backend', ['Local BGE', 'Nebius course credits'], disabled=not guided)
            rerank = st.checkbox('Rerank with FlashRank', value=False, disabled=not guided, help='Scores a wider candidate pool with a real local cross-encoder. Downloads the model once.')
            use_graph = st.checkbox('Add API and version relationships', value=False, disabled=not guided, help='Add bounded source candidates linked by literal API mentions or reported-version metadata.')
            st.caption('Nebius embeddings send source passages and questions to your course account when that backend is selected. Keyword mode does not embed.')
        if configured:
            st.caption('Nebius configured. Generation runs only when enabled.')
            st.caption('Model: ' + (os.getenv('TRIAGE_MODEL') or os.getenv('NEBIUS_MODEL') or '')[:200])
        else:
            st.caption('Evidence mode is ready. Add a Nebius key and model ID to .env to enable generated answers.')
        with st.expander('Model setup'):
            st.markdown('Create a [Nebius API key](https://tokenfactory.nebius.com/), then configure the project’s local `.env` file using `.env.example`. Restart the app after changing settings.')
            st.code('TRIAGE_API_KEY=…\nTRIAGE_BASE_URL=https://api.tokenfactory.nebius.com/v1\nTRIAGE_MODEL=your-model-id', language='bash')
            st.caption('API keys stay on the server. Do not add .env to Git.')
        st.divider()
        st.markdown('**Knowledge snapshot**')
        st.caption(f'{len(sources)} sources · {snapshot}')
        if freshness['status'] == 'stale':
            st.warning('This snapshot is over 7 days old. Refresh and review it before relying on version-specific guidance.')
        elif freshness['status'] == 'unknown':
            st.warning('Snapshot age cannot be verified for every source. Review source timestamps before relying on this snapshot.')
        else:
            st.caption(f"Oldest snapshot: {freshness['age_days']:.1f} days old. Manual refresh is due weekly.")
        with st.expander('Refresh instructions'):
            st.caption('Fetch every source again with this maintainer command, then reload the app. Commit pins and API version stay explicit; a new fetch date does not verify every upstream claim.')
            st.code('uv run python scripts/fetch_corpus.py --refresh-all', language='bash')
        st.caption('A bounded source snapshot. Issue reports describe observed behavior; matches need human review.')

    st.markdown('<p class="eyebrow">ENGINEERING WORKSPACE / STREAMLIT</p>', unsafe_allow_html=True)
    head, state = st.columns([4, 1])
    with head:
        st.title('Investigate a Streamlit issue')
        st.caption('Bring the symptoms. Find supporting evidence, capture missing details, and draft a clear handoff.')
    with state:
        st.write('')
        st.badge('Evidence workspace', icon=':material/fact_check:', color='green')

    settings = {'mode': mode, 'kinds': kinds, 'generation': use_generation, 'guided': guided,
                'strategy': 'structured' if strategy_label == 'Structured passages' else 'fixed',
                'embedding_provider': embedding_provider, 'rerank': rerank, 'use_graph': use_graph}
    investigate_tab, deep_tab, chat_tab, knowledge_tab, graph_tab, lab_tab, evaluation_tab, guide_tab = st.tabs(
        ['Investigate', 'Deep Investigation', 'Chat', 'Knowledge base', 'Relationships', 'Retrieval lab', 'Evaluation', 'Project guide'])

    with guide_tab:
        from triage_desk.project_guide import render_project_guide
        render_project_guide(ROOT)

    with deep_tab:
        from triage_desk.agent_ui import render_deep_investigation
        render_deep_investigation(ROOT, engine((ROOT / 'data' / 'sources.json').stat().st_mtime))

    with investigate_tab:
        left, right = st.columns([2.05, 1], gap='large')
        with left:
            with st.container(border=True):
                st.subheader('What’s going wrong?')
                st.caption('Paste a question or report. Include the exact error when you have it.')
                choices = st.columns(3)
                for col, label in zip(choices, EXAMPLES):
                    col.button(label, on_click=select_example, args=(label,), width='stretch')
                if 'report' not in st.session_state:
                    st.session_state.report = ''
                with st.form('investigation'):
                    report = st.text_area('Issue report', key='report', height=180, max_chars=4500, placeholder='For example: My uploaded CSV disappears after a widget interaction. How can I investigate this?')
                    with st.expander('Add environment and reproduction details'):
                        a, b, c = st.columns(3)
                        version = a.text_input('Streamlit version', placeholder='e.g. 1.50.0', max_chars=120)
                        python = b.text_input('Python version', placeholder='e.g. 3.12', max_chars=40)
                        os_name = c.text_input('Operating system', placeholder='e.g. macOS', max_chars=60)
                        reproduction = st.text_area('Steps to reproduce', height=85, max_chars=1000, placeholder='1. Start the app…\n2. Upload a file…')
                    submitted = st.form_submit_button('Investigate report', type='primary', width='stretch')
                if submitted:
                    if not report.strip():
                        st.warning('Enter a report or choose an example to start.')
                    elif not kinds:
                        st.warning('Choose at least one evidence source in the sidebar.')
                    elif not sources:
                        st.warning('The corpus is still being prepared. Try again after ingestion completes.')
                    else:
                        question = report.strip()
                        details = '\n'.join(f'{k}: {v}' for k, v in [('Python version', python), ('Operating system', os_name), ('Steps to reproduce', reproduction)] if v.strip())
                        if details:
                            question += '\n\n' + details
                        try:
                            with st.spinner('Searching source evidence and preparing the handoff…'):
                                result = investigate(question, settings, version=version)
                            st.session_state.analysis = result
                        except Exception as exc:
                            st.error(f'Investigation could not finish: {exc}')
                            st.caption('For a first semantic search, the local embedding model must download successfully. Keyword search can run without that download.')
        with right:
            with st.container(border=True):
                st.subheader('A useful issue starts with evidence')
                st.markdown('**01 · Find the context**  \nSearch documentation, release notes, and related reports.')
                st.markdown('**02 · Fill the gaps**  \nCapture versions, reproduction steps, and observed behavior.')
                st.markdown('**03 · Prepare the handoff**  \nDownload a draft with source links for maintainer review.')
                st.caption('Routing and related-issue suggestions are provisional. The draft stays yours to review.')

        result = st.session_state.get('analysis')
        if result:
            render_result(result)
        else:
            st.markdown('')
            c1, c2, c3 = st.columns(3)
            c1.metric('Documentation & guidance', sum(s.get('kind') in {'documentation', 'contributing', 'release'} for s in sources))
            c2.metric('Historical issues', sum(s.get('kind') == 'issue' for s in sources))
            c3.metric('Sources in this workspace', len(sources))

    with chat_tab:
        st.subheader('Ask a follow-up with the evidence in view')
        st.caption('Explicit follow-ups can use the last three user requests to resolve references. Previous answers are never treated as source evidence.')
        if st.button('Clear chat'):
            st.session_state.chat_records = []
        records = st.session_state.setdefault('chat_records', [])
        for index, record in enumerate(records):
            with st.chat_message('user'):
                st.text(record['question'])
            with st.chat_message('assistant'):
                answer = record['result']
                if answer.get('metrics', {}).get('generation_mode', '').startswith('workflow_'):
                    st.text(answer['answer'])
                else:
                    st.markdown(answer['answer'])
                with st.expander('Sources, workflow and handoff'):
                    for item in answer.get('evidence', []):
                        st.text('[' + item.get('citation_id', '?') + '] ' + item.get('title', 'Source'))
                        for quote in item.get('support_quotes') or [item.get('quote', '')]:
                            st.text(quote)
                        if safe_url(item.get('url', '')):
                            st.link_button('Open source', safe_url(item['url']))
                    st.json(answer.get('workflow', {}).get('provider_usage', {}))
                    st.download_button('Download handoff draft', answer.get('draft_markdown', ''), file_name='issue-handoff.md', key=f'chat-draft-{index}')
        prompt = st.chat_input('Ask about a Streamlit API, issue, or the previous report…', max_chars=4500)
        if prompt:
            if not kinds:
                st.warning('Choose at least one evidence source in the sidebar.')
            else:
                try:
                    history = [{'role': 'user', 'content': record['question']} for record in records]
                    with st.spinner('Investigating the follow-up and checking the available evidence…'):
                        result = investigate(prompt, settings, history=history)
                    records.append({'question': prompt, 'result': result})
                    st.session_state.analysis = result
                    st.rerun()
                except Exception as exc:
                    st.error(f'Chat investigation could not finish: {type(exc).__name__}. Review the model and retrieval settings.')

    with knowledge_tab:
        st.subheader('Inspect the knowledge behind an answer')
        st.caption('Each entry retains its upstream link and snapshot metadata. Refreshing the corpus is a separate, explicit operation.')
        needle = st.text_input('Find a source', placeholder='Search titles or source IDs')
        filtered = [s for s in sources if needle.casefold() in (s.get('title', '') + ' ' + s.get('source_id', '')).casefold()]
        if filtered:
            label_to_source = {f"{s['title']} · {s['source_id']}": s for s in filtered}
            selected = st.selectbox('Source document', list(label_to_source))
            source = label_to_source[selected]
            a, b = st.columns([2.4, 1], gap='large')
            with a:
                with st.container(border=True):
                    st.subheader(source['title'])
                    url = safe_url(source.get('url', ''))
                    if url:
                        st.link_button('Open original source ↗', url)
                    st.text(source.get('text', '')[:40000])
            with b:
                st.markdown('**Source record**')
                st.json({k: v for k, v in source.items() if k != 'text'}, expanded=True)
        else:
            st.info('No matching sources.')
        with st.expander('Corpus manifest and licensing'):
            st.json(manifest)
            st.caption('See data/licenses/ for retained upstream license and attribution notices.')

    with graph_tab:
        st.subheader('Explore the relationships recorded in the sources')
        graph = knowledge_graph((ROOT / 'data' / 'sources.json').stat().st_mtime)
        summary = graph.summary()
        columns = st.columns(3)
        columns[0].metric('Graph nodes', summary['node_count'])
        columns[1].metric('Provenance edges', summary['edge_count'])
        columns[2].metric('Source records', summary['source_count'])
        graph_query = st.text_input('API or reported version', value='st.file_uploader', help='Exact API identifiers, reported versions, or source IDs. This graph does not infer fixes or affected-version ranges.')
        graph_limit = st.slider('Graph node limit', 5, 60, 8)
        view = graph.explore(graph_query, max_nodes=graph_limit)
        if view['nodes']:
            st.graphviz_chart(graph.to_dot(graph_query, max_nodes=graph_limit), width='stretch', height=600)
            st.caption(f"Showing {len(view['nodes'])} nodes. Increase the limit to inspect more relationships; the JSON keeps full source labels and provenance.")
            with st.expander('Inspect graph facts and provenance'):
                st.json(view, expanded=False)
        else:
            st.info('No exact API/version entity matched. Try an API such as st.cache_data.')
        st.caption('Edges record literal mentions and explicit source metadata. They do not establish causes, duplicates, compatibility, or confirmed fixes.')
        st.download_button('Download graph view JSON', json.dumps(view, indent=2), file_name='engineering-relationships.json')

    with lab_tab:
        st.subheader('Compare chunking and reranking on the same question')
        st.caption('Three runs: fixed chunks, structured chunks, and structured chunks with FlashRank. Search results are candidates; the reranker score is not a probability or a correctness rating.')
        lab_question = st.text_input('Comparison question', value='How does st.cache_data handle an unhashable connection argument?')
        if st.button('Compare retrieval configurations'):
            try:
                _, lab = guided_engine((ROOT / 'data' / 'sources.json').stat().st_mtime, embedding_provider)
                rows = []
                for label, strategy, should_rerank in [('Fixed', 'fixed', False), ('Structured', 'structured', False), ('Structured + rerank', 'structured', True)]:
                    started = time.perf_counter()
                    matches = lab.search(lab_question, strategy=strategy, mode=mode, source_kinds=kinds, rerank=should_rerank)
                    rows.append({'configuration': label, 'latency_ms': round((time.perf_counter() - started) * 1000, 1), 'evidence': matches})
                st.session_state.retrieval_comparison = rows
            except Exception as exc:
                st.error(f'Comparison could not finish ({type(exc).__name__}). No reranking scores were substituted.')
        if st.session_state.get('retrieval_comparison'):
            for col, record in zip(st.columns(3), st.session_state.retrieval_comparison):
                with col:
                    st.markdown('**' + record['configuration'] + '**')
                    st.caption(str(record['latency_ms']) + ' ms, including initialization if needed')
                    for item in record['evidence'][:4]:
                        with st.expander(item['title']):
                            st.text(item['quote'])
                            st.caption(f"{item['source_id']} · rerank score: {item.get('rerank_score', 'not used')}")
            st.download_button('Download retrieval comparison', json.dumps(st.session_state.retrieval_comparison, indent=2), file_name='retrieval-comparison.json')

    with evaluation_tab:
        st.subheader('Evaluate evidence and investigation handoffs')
        st.caption('Compare retrieval diagnostics and observed Deep Agents runs. Citation integrity, saved handoffs, and human answer-quality judgments are measured separately.')
        from triage_desk.reviews import outcome_summary
        reviews = outcome_summary(ROOT / '.cache' / 'user-outcomes.jsonl')
        if reviews['reviewed_runs']:
            st.markdown('**User-reviewed outcomes**')
            st.json(reviews)
        else:
            st.caption('No user outcome labels recorded. A first-contact resolution rate cannot be inferred from generated answers.')
        reports = sorted(set((ROOT / 'evaluation').glob('*report*.json')) | set((ROOT / 'evaluation').glob('*/report.json')), key=lambda p: p.stat().st_mtime, reverse=True)
        if reports:
            path = st.selectbox('Evaluation report', reports, format_func=lambda p: str(p.relative_to(ROOT)))
            document = read_json(path, {})
            render_evaluation_summary(document)
            if document.get('generation_requested'):
                st.markdown('**Generated-answer run**')
                st.json({mode: {key: value for key, value in scores.items() if key.startswith('generation_')} for mode, scores in document.get('summary', {}).items()}, expanded=True)
            if document.get('ai_review_summary'):
                st.markdown('**AI review of generated answers**')
                st.caption('An AI assessment against retrieved passages. Human-review fields remain separate.')
                st.json(document['ai_review_summary'], expanded=True)
            with st.expander('Full benchmark record'):
                st.json(document, expanded=False)
            st.download_button('Download evaluation JSON', json.dumps(document, indent=2), file_name=path.name, mime='application/json')
            if path.with_suffix('.html').exists():
                st.download_button('Download readable evaluation report', path.with_suffix('.html').read_text(), file_name='evaluation-report.html', mime='text/html')
        else:
            st.info('Run a retrieval benchmark or Deep Investigation evaluation to create a report.')
        st.code('uv run python -m evaluation.run\nuv run python -m evaluation.agent_workflow --live --output-dir evaluation/week3-results-new', language='bash')
        st.markdown('**Review dimensions:** source retrieval, citation validity, supported claims, saved human handoffs, answer relevance, observed calls, and latency. Automatic checks and AI review leave human quality judgments unscored.')


def render_result(result: dict):
    st.divider()
    heading, timing = st.columns([4, 1])
    with heading:
        st.subheader('Investigation workspace')
        st.text(result.get('question', '')[:500])
    metrics = result.get('metrics', {})
    st.caption(f"Results from {metrics.get('mode', 'unknown')} search. Run the investigation again after changing the report or filters.")
    with timing:
        st.metric('Investigation time', f"{metrics.get('latency_ms', 0) / 1000:.2f}s")
    status = result.get('status', '')
    if status == 'insufficient_evidence':
        st.warning('The snapshot does not contain enough evidence for a supported answer.')
    if metrics.get('generation_mode', '') in {'llm', 'generated', 'nebius', 'cited_model_synthesis'}:
        st.badge('Generated from retrieved evidence', color='blue')
        model = metrics.get('generation_returned_model') or metrics.get('generation_requested_model')
        if model:
            st.caption('Model: ' + str(model))
        verification = metrics.get('generation_verification') or {}
        if verification.get('attempted'):
            st.caption('AI support check: ' + str(verification.get('accepted_claims', 0)) + ' retained, ' + str(verification.get('rejected_claims', 0)) + ' removed. This check can make mistakes; inspect the cited quotes.')
            if verification.get('rejected_claims'):
                st.warning('Partial answer: some draft claims were removed because the AI check could not support them. Review whether the remaining answer addresses your question.')
    elif metrics.get('generation_mode') == 'model_abstained':
        st.badge('Model requested more evidence', color='orange')
    else:
        st.badge('Evidence mode · extracted passages and triage heuristics', color='green')
    if metrics.get('generation_error'):
        st.warning(metrics['generation_error'])
    workflow = result.get('workflow')
    if workflow:
        with st.expander('Workflow trace and model usage'):
            st.caption('Route: ' + workflow.get('route', 'unknown') + '. Confidence labels describe the decision process; they are not calibrated probabilities.')
            st.dataframe(workflow.get('trace', []), hide_index=True, width='stretch')
            st.json(workflow.get('provider_usage', {}), expanded=True)
            st.json(workflow.get('handoff', {}), expanded=False)
        st.download_button('Download workflow trace', json.dumps(workflow, indent=2), file_name='workflow-trace.json')
    answer_col, triage_col = st.columns([2.05, 1], gap='large')
    with answer_col:
        with st.container(border=True):
            st.markdown('### Evidence summary')
            if metrics.get('generation_mode', '').startswith('workflow_'):
                st.text(result.get('answer') or 'Review the evidence below before proceeding.')
            else:
                st.markdown(result.get('answer') or 'Review the evidence below before proceeding.')
        st.markdown('### Supporting sources')
        evidence = result.get('evidence', [])
        for i, item in enumerate(evidence):
            ref = item.get('citation_id') or item.get('citation') or f'S{i + 1}'
            title = item.get('title') or item.get('source_id', 'Source')
            with st.expander(f'[{ref}] {title}', expanded=i < 2):
                st.caption(f"{str(item.get('kind', '')).title()} · {item.get('source_id', '')} · {item.get('section', '')}")
                supports = item.get('support_quotes', [])
                if supports:
                    st.markdown('**Exact passages cited by the generated answer**')
                    for quote in supports:
                        st.text(quote)
                    with st.expander('Retrieved excerpt'):
                        st.text(item.get('quote') or item.get('text') or '')
                else:
                    st.text(item.get('quote') or item.get('text') or '')
                url = safe_url(item.get('url', ''))
                if url:
                    st.link_button('View source ↗', url)
                scope = item.get('version_scope') or 'Check applicability to your installed version.'
                st.caption(f"Version scope: {scope}")
                if item.get('snapshot_at'):
                    st.caption(f"Snapshot: {item['snapshot_at']}")
        if not evidence:
            st.caption('No supported source passages were selected.')
    with triage_col:
        with st.container(border=True):
            st.markdown('### Proposed routing')
            st.markdown(f"**{result.get('suggested_type', 'Needs review')}**")
            st.caption(f"Component: {result.get('component', 'Needs review')}")
            st.caption('Heuristic suggestion for a human maintainer.')
            st.divider()
            st.markdown('**Information to collect**')
            missing = result.get('missing_details', [])
            if missing:
                for item in missing:
                    st.text('• ' + str(item))
            else:
                st.caption('Core details were detected. Review the reproduction before filing.')
        related = result.get('related_issues', [])
        if related:
            with st.container(border=True):
                st.markdown('### Related reports to review')
                for item in related:
                    title = item.get('title', 'Related issue')
                    url = safe_url(item.get('url', ''))
                    st.text(title)
                    if url:
                        st.link_button('Review related report ↗', url)
                    st.caption(f"State at snapshot: {item.get('state', 'unknown')}")
                st.caption('Similarity does not confirm a duplicate or a resolved bug.')
    draft_tab, record_tab = st.tabs(['Issue draft', 'Investigation record'])
    with draft_tab:
        st.caption('Review and complete the placeholders before sharing this draft.')
        draft = result.get('draft_markdown', '')
        st.code(draft, language='markdown')
        st.download_button('Download issue draft', draft, file_name='issue-draft.md', mime='text/markdown', type='primary')
    with record_tab:
        st.json(result, expanded=False)
        st.download_button('Download investigation JSON', json.dumps(result, indent=2), file_name='investigation.json', mime='application/json')
    if result.get('run_id'):
        from triage_desk.reviews import save_outcome
        with st.expander('Record whether this investigation helped'):
            label = st.selectbox('Your outcome', ['Needs follow-up', 'Resolved from the evidence', 'Human handoff needed'], key='outcome-' + result['run_id'])
            st.caption('Only the run ID, outcome, first-contact flag and timestamp are saved locally. Reports and answers are not stored in the outcome log.')
            if st.button('Save outcome', key='save-outcome-' + result['run_id']):
                outcome = {'Needs follow-up': 'needs_followup', 'Resolved from the evidence': 'resolved', 'Human handoff needed': 'human_handoff'}[label]
                save_outcome(ROOT / '.cache' / 'user-outcomes.jsonl', result['run_id'], outcome, first_contact=result.get('first_contact', True))
                st.success('Outcome saved locally.')


if __name__ == '__main__':
    main()
