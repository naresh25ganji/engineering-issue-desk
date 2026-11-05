"""Build the concise Week 4 submission with the bundled document runtime."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from build_week3_docs import DEFAULT_SKILL, base_doc, heading, new_page, paragraph, table


ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/naresh25ganji/engineering-issue-desk'
WORKBOOK = 'https://docs.google.com/spreadsheets/d/1Lj9Q6fWfSeAPdyUegOaphcscCncEr4_3/edit?usp=drivesdk&ouid=117743947283173099816&rtpof=true&sd=true'
VIDEO = 'https://drive.google.com/file/d/1HKnO5cZ26HwhxMy414UAMDZKW8rhGyHP/view?usp=drivesdk'


def link(p, label, url):
    relation = p.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), relation)
    run = OxmlElement('w:r')
    properties = OxmlElement('w:rPr')
    style = OxmlElement('w:rStyle')
    style.set(qn('w:val'), 'Hyperlink')
    properties.append(style)
    run.append(properties)
    text = OxmlElement('w:t')
    text.text = label
    run.append(text)
    hyperlink.append(run)
    p._p.append(hyperlink)


def links(doc, items):
    p = doc.add_paragraph()
    for index, (label, url) in enumerate(items):
        if index:
            p.add_run('  |  ')
        link(p, label, url)
    return p


def build():
    doc = base_doc('Engineering Issue Desk Week 4 Evaluation', 'Naresh Ganji  |  November 3 2026  |  Own agent spreadsheet track')
    doc.core_properties.subject = 'The Gen Academy Week 4 evaluation submission'
    doc.styles['Title'].font.size = Pt(22)
    doc.styles['Heading 1'].font.size = Pt(16)
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)

    paragraph(doc, 'I evaluated the independent evidence reviewer in my engineering investigation agent. Across 40 fixed cases, the selected experimental prompt increased exact verdict agreement from 80% to 97.5%, with seven case fixes and no final regressions. The 119 reference labels are AI-proposed and still await human review; these results are development diagnostics.')
    heading(doc, 'Purpose and architecture')
    paragraph(doc, 'Engineers use the app to investigate reports and prepare cited findings and diagnostic steps. The reviewer checks that selected quotes support each item before a human approves, revises, or rejects the report. Week 4 isolates this reviewer; it does not score the full investigator.')
    picture = doc.add_paragraph().add_run().add_picture(str(ROOT / 'docs/week4-architecture.png'), width=Inches(6.9))
    picture._inline.docPr.set('descr', 'Application flow from engineer to investigator, evidence reviewer and human review, plus the separate Week 4 evaluation runner and scoring flow.')
    heading(doc, 'Evaluation framework')
    paragraph(doc, 'I measure exact case agreement, unsupported-item rejection, supported-item retention, p95 review latency, and mean reported tokens using the actual review_report component and Fireworks gpt-oss-120b. All five runs use the same corrected version 2 dataset, model settings, output schema, and evidence. Expected labels are excluded from model inputs.')
    paragraph(doc, 'The dataset contains 20 happy paths, 12 edge cases, 6 known failures, and 2 adversarial cases. Frozen repository excerpts retain paths, line ranges, and source hashes. There are 76 supported and 43 unsupported item labels.')
    paragraph(doc, 'Scoring compares strict boolean verdicts with reference labels. A case passes only if every finding and proposal agrees. Missing verdicts and execution errors remain in the denominators and fail correctness. Source matching checks citation integrity; semantic support still requires review.')

    new_page(doc, 'Measured results and failures')
    paragraph(doc, 'Pass bars were set before the baseline: exact agreement at least 90%, unsupported rejection at least 95%, supported retention at least 90%, p95 latency at most 30 seconds, and mean reported tokens at most 4,000.')
    table(doc,
          ['Prompt', 'Exact', 'Reject', 'Retain', 'p95 s', 'Tokens', 'Errors'],
          [['baseline', '80.0%', '95.35%', '89.47%', '4.54', '2,188', '0'],
           ['diagnostic', '85.0%', '97.67%', '93.42%', '5.40', '2,330', '0'],
           ['scope', '85.0%', '95.35%', '92.11%', '5.33', '2,444', '1'],
           ['authority', '87.5%', '100%', '93.42%', '6.15', '2,566', '0'],
           ['calibrated', '97.5%', '100%', '98.68%', '6.50', '2,933', '0']],
          [1.05, .80, .91, .91, .73, .85, .69])
    paragraph(doc, 'The calibrated candidate meets all five development pass bars: 43/43 unsupported items rejected and 75/76 supported items retained. Versus baseline, agreement improved 17.5 percentage points, rejection 4.65 points, and retention 9.21 points. p95 increased 1.96 seconds and mean tokens increased 744.85. Dollar cost is unknown because provider rates were not supplied.')
    heading(doc, 'Baseline failure modes')
    table(doc, ['Failure', 'Wrong items', 'Example cases'],
          [['Supported diagnostics removed', '6', 'W4-13, W4-18, W4-26'],
           ['Supported static or fixture findings removed', '2', 'W4-02, W4-36'],
           ['Unsupported statements retained', '2', 'W4-34, W4-35']],
          [3.55, 1.0, 1.95])
    paragraph(doc, 'Ten wrong items affect eight baseline cases, which can overlap categories. Assuming 5 minutes per supported item removed and 30 per unsupported item retained, baseline cluster costs are 30, 10, and 60 minutes; the candidate totals 5 minutes. These are illustrative weights, not measured effort or proven internal causes.')
    heading(doc, 'One concrete improvement')
    paragraph(doc, 'W4-13 proposes a fake HTTP error containing a sentinel, then inspects the raised message. The baseline rejects the proposal because the sentinel would not appear. The calibrated reviewer recognizes that checking its absence is a useful diagnostic. Its saved verdict changes from false to true and the whole case passes.')
    paragraph(doc, 'The final candidate fixes W4-02, W4-13, W4-18, W4-26, W4-34, W4-35, and W4-36, with zero final case regressions. W4-17 still rejects one supported diagnostic. The intermediate scope run also preserves an invalid_reason error on W4-33; later success does not erase that failed run.')

    new_page(doc, 'Prompt changes and submission materials')
    paragraph(doc, 'The four additions are cumulative prompt changes. Each row compares a stage with its predecessor; pp means percentage points. Predicted and measured deltas below refer to exact case agreement.')
    table(doc, ['Change', 'Targeted failure', 'Predicted', 'Measured'],
          [['Diagnostic guidance', 'Useful tests rejected as unproven outcomes', '+12.5 pp', '+5.0 pp'],
           ['Boundary audit', 'Incorrect strictness and compound conditions', '+2.5 pp', '+0.0 pp'],
           ['Static evidence scope', 'Literal code and fixture observations rejected', '+7.5 pp', '+2.5 pp'],
           ['Rubric examples', 'Remaining diagnostic and fixture over-rejection', '+5.0 pp', '+10.0 pp']],
          [1.55, 2.95, 1.0, 1.0])
    paragraph(doc, 'The scope addition reduced unsupported rejection by 2.33 pp and supported retention by 1.32 pp; intermediate regressions remain visible in the workbook and analysis. The first three hypotheses were registered from the preserved v1 baseline before the v2 experiments; calibration was registered after authority and before its own run.')
    heading(doc, 'Relevant prompt excerpts')
    paragraph(doc, 'The diagnostic addition says: “Inspecting for a sentinel includes checking that it is absent; do not require the sentinel to appear for the test to be useful.” The boundary addition says: “Do not silently repair the author\'s claim to a nearby correct statement in your reason.” Full frozen prompts and the evaluator are linked below.')
    heading(doc, 'Limits and next steps')
    paragraph(doc, 'All labels remain pending human signoff. These cases were reused for development, with one call per case per stage; serving variation and cumulative changes prevent isolated causal attribution or a generalization claim. Complete human label review, resolve disagreements, and test a separately authored held-out set with repeated calls before promoting the candidate. Production still defaults to baseline.')
    paragraph(doc, 'Local traces save case IDs, dataset and prompt hashes, verdicts, reasons, errors, latency, and returned tokens. LangSmith was not used for this spreadsheet track. Proposed production alerts cover a quality drop over 5 pp in seven days, review errors over 5% in an hour, p95 latency over 30 seconds, or p95 tokens up over 25% in a day. Engineering time saved has not been measured.')
    heading(doc, 'Submission links')
    links(doc, [('Evaluation workbook', WORKBOOK), ('Demo video', VIDEO), ('Project repository', REPO)])
    links(doc, [('Full evaluation report', REPO + '/blob/main/docs/WEEK4_EVALUATION.md'),
                ('Case dataset', REPO + '/blob/main/evaluation/week4-cases.json'),
                ('Paired results', REPO + '/blob/main/evaluation/week4-comparison/report.json')])
    links(doc, [('Baseline prompt', REPO + '/blob/main/evaluation/week4-baseline/prompt.txt'),
                ('Calibrated prompt', REPO + '/blob/main/evaluation/week4-calibrated/prompt.txt'),
                ('Registered hypotheses', REPO + '/blob/main/evaluation/week4-hypotheses.json')])
    links(doc, [('Reviewer code', REPO + '/blob/main/triage_desk/agent_review.py'),
                ('Evaluator and rubric', REPO + '/blob/main/evaluation/week4.py'),
                ('Case trace example', REPO + '/blob/main/evaluation/week4-calibrated/cases/W4-13.json')])
    destination = ROOT / 'docs/Engineering Issue Desk Week 4 Writeup.docx'
    doc.save(destination)
    sanitized = ROOT / '.cache/week4-submission/Engineering Issue Desk Week 4 Writeup.docx'
    sanitized.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(DEFAULT_SKILL / 'scripts/google_docs_title_sanitize.py'),
                    str(destination), '--out', str(sanitized)], check=True)
    shutil.copyfile(sanitized, destination)
    print(destination)


if __name__ == '__main__':
    build()
