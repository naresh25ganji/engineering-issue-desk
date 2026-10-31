"""Select and execute bounded evaluation suites from the local workspace."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st


@dataclass(frozen=True)
class EvaluationSuite:
    id: str
    label: str
    description: str
    dataset: str | None
    supports_live: bool


SUITES = (
    EvaluationSuite('workflows', 'Full engineering workflows',
                    'Runs the investigator, tools, evidence review, and saved human handoff across four workflows.',
                    'evaluation/workflow-cases.json', True),
    EvaluationSuite('reviewer_regression', 'Fresh reviewer regression',
                    'Tests evidence-review verdicts on 30 new cases about incidents, proposed changes, and tests.',
                    'evaluation/reviewer-regression-cases.json', True),
    EvaluationSuite('integrity', 'Citation and report integrity',
                    'Checks 30 structural acceptance and rejection cases without a model or provider.', None, False),
)
_LABELS = {'all': 'All recorded runs', **{suite.id: suite.label for suite in SUITES},
           'week4': 'Original Week 4 reviewer experiments', 'legacy': 'Retrieval and earlier agent runs'}


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return {}


def report_suite(document: dict, *, regression_hash: str | None = None) -> str:
    kind = document.get('report_kind')
    if kind in {'use_case_workflow', 'use_case_comparison'}:
        return 'workflows'
    if kind == 'evidence_integrity':
        return 'integrity'
    if kind in {'week4_evidence_review', 'week4_paired_comparison'}:
        manifest = document.get('manifest', document.get('baseline', {}))
        if regression_hash and manifest.get('dataset_sha256') == regression_hash:
            return 'reviewer_regression'
        return 'week4'
    return 'legacy'


def list_reports(root: Path, suite: str = 'all') -> list[Path]:
    """Index exported reports only; caches and unfinished case files stay separate."""
    if suite not in _LABELS:
        raise ValueError('Unknown evaluation suite.')
    root = Path(root)
    regression = root / 'evaluation' / 'reviewer-regression-cases.json'
    digest = hashlib.sha256(regression.read_bytes()).hexdigest() if regression.is_file() else None
    paths = set((root / 'evaluation').glob('*report*.json')) | set((root / 'evaluation').glob('*/report.json'))
    records = [path for path in paths if suite == 'all' or report_suite(_read(path), regression_hash=digest) == suite]
    return sorted(records, key=lambda path: path.stat().st_mtime, reverse=True)


def run_suite(root: Path, suite_id: str, *, output_dir=None, live=False,
              reviewer_prompt='calibrated', workers=2, case_ids=None, use_cases=None):
    """Model execution happens only through this explicit action, never browsing."""
    root = Path(root).resolve()
    suite = next((item for item in SUITES if item.id == suite_id), None)
    if suite is None:
        raise ValueError('Choose an executable evaluation suite.')
    if live and not suite.supports_live:
        raise ValueError('The integrity suite does not use a model.')
    if reviewer_prompt not in {'baseline', 'calibrated'}:
        raise ValueError('Choose baseline or calibrated evidence review.')
    if output_dir is None:
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
        output_dir = root / 'evaluation' / f'{suite_id}-{"live" if live else "local"}-{suffix}'
    else:
        output_dir = Path(output_dir)
        output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    if suite_id == 'workflows':
        from evaluation.use_cases import run_evaluation
        result = run_evaluation(root, Path(suite.dataset), output_dir, live=live, workers=workers,
                                case_ids=case_ids, use_cases=use_cases, reviewer_prompt=reviewer_prompt)
    elif suite_id == 'reviewer_regression':
        from evaluation.week4 import run_evaluation
        result = run_evaluation(root, Path(suite.dataset), output_dir, live=live, workers=workers,
                                case_ids=case_ids, variant=reviewer_prompt)
    else:
        if case_ids or use_cases:
            raise ValueError('The integrity suite runs all thirty contract checks.')
        from evaluation.integrity import run_evaluation
        result = run_evaluation(root, output_dir)
    return result, output_dir / 'report.json'


def render_evaluation_launcher(root: Path) -> str:
    """Return a browsing filter while keeping execution an explicit form action."""
    pending_suite = st.session_state.pop('evaluation_pending_suite', None)
    if pending_suite in _LABELS:
        st.session_state['evaluation_suite_filter'] = pending_suite
    selected = st.selectbox('Show evaluation suite', list(_LABELS),
                            format_func=_LABELS.get, key='evaluation_suite_filter')
    with st.expander('Run a new evaluation', expanded=False):
        suite_id = st.selectbox('Suite to run', [item.id for item in SUITES],
                                format_func=lambda value: next(item.label for item in SUITES if item.id == value),
                                key='evaluation_suite_to_run')
        suite = next(item for item in SUITES if item.id == suite_id)
        st.caption(suite.description)
        dataset = _read(Path(root) / suite.dataset) if suite.dataset else {}
        cases = dataset.get('cases', [])
        if suite.dataset and not cases:
            st.warning('This suite dataset is not available in the checkout.')
        if cases:
            st.caption(f'{len(cases)} fixed cases · labels and semantic rubrics await human review.')
        live, prompt, workers, identities, profiles = False, 'calibrated', 2, [], []
        if suite.supports_live:
            live = st.radio('Execution mode', ['Validate inputs', 'Run model evaluation'],
                             key='evaluation_execution_mode', horizontal=True) == 'Run model evaluation'
            prompt = st.selectbox('Evidence review prompt', ['calibrated', 'baseline'],
                                   key='evaluation_review_prompt')
            workers = st.number_input('Parallel cases', min_value=1, max_value=4, value=2, step=1,
                                       key='evaluation_workers')
            if suite_id == 'workflows':
                from .use_cases import USE_CASES, get_use_case
                profiles = st.multiselect('Workflows to evaluate', [item.id for item in USE_CASES],
                                           format_func=lambda value: get_use_case(value).label,
                                           help='Leave empty to run all four workflows.', key='evaluation_workflows')
            identities = st.multiselect('Cases to evaluate', [case['id'] for case in cases],
                                         help='Leave empty to use every case allowed by the workflow filter.',
                                         key='evaluation_cases_' + suite_id)
            if live:
                st.caption('This action sends the selected requests and public fixture context to the configured Fireworks model. Local traces and scored results are saved; reports await human review.')
            else:
                st.caption('Input validation checks frozen fixtures and writes an unscored inventory. No model or provider is called.')
        else:
            st.caption('All 30 checks run locally. Citation integrity does not establish factual correctness or usefulness.')
        label = 'Run model evaluation' if live else 'Validate suite inputs' if suite.supports_live else 'Run citation checks'
        if st.button(label, key='run_evaluation_suite', type='primary', disabled=bool(suite.dataset and not cases)):
            try:
                with st.spinner('Running the selected evaluation…'):
                    _, report_path = run_suite(root, suite_id, live=live, reviewer_prompt=prompt,
                                               workers=int(workers), case_ids=identities or None,
                                               use_cases=profiles or None)
                st.session_state['evaluation_report_path'] = str(report_path.relative_to(root))
                st.session_state['evaluation_pending_suite'] = suite_id
                st.session_state['evaluation_saved_notice'] = 'Evaluation saved. Inspect the report below.'
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error('The evaluation could not start or save its results: ' + str(exc))
            except Exception:
                st.error('The evaluation was interrupted. Inspect its saved case records before starting another run.')
    notice = st.session_state.pop('evaluation_saved_notice', None)
    if notice:
        st.success(notice)
    return selected
