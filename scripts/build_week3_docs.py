"""Build the Week 3 submission documents with the bundled artifact runtime.

Run with ~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12.
The authoring-operation marker is a session responsibility and is not repeated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path.home() / '.cache/codex-runtimes/codex-primary-runtime'
DEFAULT_SKILL = Path.home() / '.codex/plugins/cache/openai-primary-runtime/documents/26.909.12148/skills/documents'
WRITEUP = 'Engineering Issue Desk Week 3 Writeup.docx'
DEMO = 'Engineering Issue Desk Week 3 Demo Guide.docx'
BLACK = RGBColor(0, 0, 0)


def paragraph(doc, text='', *, bold_lead=None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def heading(doc, text, level=1):
    return doc.add_heading(text, level)


def bullet(doc, text):
    return paragraph(doc, text, style='List Bullet')


def new_page(doc, title):
    p = heading(doc, title)
    p.paragraph_format.page_break_before = True


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for column, width in zip(t.columns, widths):
        column.width = Inches(width)
    for cell, label, width in zip(t.rows[0].cells, headers, widths):
        cell.width = Inches(width)
        cell.text = label
    for values in rows:
        cells = t.add_row().cells
        for cell, value, width in zip(cells, values, widths):
            cell.width = Inches(width)
            cell.text = str(value)
    for row_index, row in enumerate(t.rows):
        row_pr = row._tr.get_or_add_trPr()
        no_split = OxmlElement('w:cantSplit')
        row_pr.append(no_split)
        if row_index == 0:
            repeated = OxmlElement('w:tblHeader')
            row_pr.append(repeated)
        for col_index, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
                item = OxmlElement('w:' + edge)
                item.set(qn('w:val'), 'single')
                item.set(qn('w:sz'), '4')
                item.set(qn('w:color'), 'D9D9D9')
                borders.append(item)
            tc_pr.append(borders)
            margins = OxmlElement('w:tcMar')
            for edge, value in [('top', '95'), ('bottom', '95'), ('left', '110'), ('right', '110')]:
                item = OxmlElement('w:' + edge)
                item.set(qn('w:w'), value)
                item.set(qn('w:type'), 'dxa')
                margins.append(item)
            tc_pr.append(margins)
            if row_index == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'E8EDF1')
                tc_pr.append(shade)
            for p in cell.paragraphs:
                if headers[col_index] == 'Records':
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    r.font.size = Pt(10)
                    r.font.color.rgb = BLACK
                    r.bold = row_index == 0
    paragraph(doc).paragraph_format.space_after = Pt(1)
    return t


def base_doc(title, subtitle):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(.72), Inches(.72)
    section.left_margin, section.right_margin = Inches(.78), Inches(.78)
    for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3', 'List Bullet'):
        style = doc.styles[name]
        style.font.name = 'Arial'
        style.font.color.rgb = BLACK
        style.font.underline = False
        style._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), 'Arial')
        style._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), 'Arial')
    normal = doc.styles['Normal']
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.widow_control = True
    for name, size, before, after in [('Title', 24, 0, 12), ('Subtitle', 11, 0, 14),
                                      ('Heading 1', 17, 12, 9), ('Heading 2', 12, 10, 5),
                                      ('Heading 3', 11, 8, 4)]:
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = name != 'Subtitle'
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    doc.styles['List Bullet'].paragraph_format.space_after = Pt(5)
    doc.core_properties.author = 'Naresh Ganji'
    doc.core_properties.title = title
    doc.core_properties.subject = 'The Gen Academy Week 3 agent project'
    doc.core_properties.comments = ''
    doc.add_paragraph(title, 'Title')
    doc.add_paragraph(subtitle, 'Subtitle')
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run('Page ')
    run.font.name, run.font.size = 'Arial', Pt(9)
    page = OxmlElement('w:fldSimple')
    page.set(qn('w:instr'), 'PAGE')
    footer._p.append(page)
    return doc


def architecture(root, qa, runtime):
    source = root / 'docs/week3-architecture.svg'
    source.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="340" viewBox="0 0 1200 340">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>
<rect width="1200" height="340" fill="white"/>
<g fill="#f3f5f6" stroke="#999" stroke-width="1.5"><rect x="25" y="20" width="330" height="90" rx="8"/><rect x="435" y="20" width="330" height="90" rx="8"/><rect x="845" y="20" width="330" height="90" rx="8"/><rect x="435" y="220" width="330" height="90" rx="8"/><rect x="845" y="220" width="330" height="90" rx="8"/></g>
<g fill="black" font-family="Arial,sans-serif" font-size="26" text-anchor="middle"><text x="190" y="58">Investigate</text><text x="190" y="89" font-size="20">Plan tools and specialists</text><text x="600" y="58">Validate citations</text><text x="600" y="89" font-size="20">Resolve saved quote IDs</text><text x="1010" y="58">Review evidence</text><text x="1010" y="89" font-size="20">Separate model support audit</text><text x="600" y="258">Human review</text><text x="600" y="289" font-size="20">Approve reject or revise</text><text x="1010" y="258">Finish</text><text x="1010" y="289" font-size="20">Save local decision</text></g>
<g fill="none" stroke="#333" stroke-width="2" marker-end="url(#arrow)"><path d="M355 65 H430"/><path d="M765 65 H840"/><path d="M1010 110 V160 H600 V215"/><path d="M765 265 H840"/><path d="M435 265 H190 V115"/></g>
<g fill="black" font-family="Arial,sans-serif" font-size="20"><text x="210" y="220">Revision with feedback</text><text x="210" y="246">and the prior report</text></g>
</svg>''')
    png = qa / 'architecture.png'
    js = 'const sharp=require(process.argv[1]); sharp(process.argv[2],{density:180}).png().toFile(process.argv[3]);'
    subprocess.run([str(runtime / 'dependencies/node/bin/node'), '-e', js,
                    str(runtime / 'dependencies/node/node_modules/sharp'), str(source), str(png)], check=True)
    return png


def read_metrics(path):
    return json.loads(path.read_text()) if path and path.exists() else None


def counted(value, singular):
    return f'{value:,} {singular}' + ('' if value == 1 else 's')


def metric_rows(metrics):
    if not metrics:
        return [('Final Week 3 evaluation', 'Awaiting recorded verification')]
    rows = []
    cases = metrics.get('cases', metrics.get('results', []))
    if isinstance(cases, dict):
        cases = [{'id': key, **value} for key, value in cases.items() if isinstance(value, dict)]
    for case in cases:
        if not isinstance(case, dict):
            continue
        label = str(case.get('id') or case.get('case_id') or case.get('scenario') or 'Recorded case')
        sample = case.get('sample', {}).get('path')
        if sample:
            label += ' ' + Path(sample).name
        status = case.get('status') or case.get('execution_status') or 'Recorded'
        counts = case.get('accounting', case.get('metrics', case))
        parts = ['Awaiting human review' if status == 'awaiting_review' else str(status)]
        for key, suffix in [('model_calls', 'model call'), ('tool_calls', 'tool call'),
                            ('total_tokens', 'reported token')]:
            value = counts.get(key)
            if key == 'total_tokens':
                value = counts.get('reported_token_usage', {}).get('total_tokens', value)
            if isinstance(value, (int, float)):
                parts.append(counted(value, suffix))
        if isinstance(case.get('report'), dict):
            report = case['report']
            reviewed = report.get('quality_review', {}).get('status') in {'completed', 'no_content'}
            label_kind = 'retained finding' if reviewed else 'draft finding'
            parts.append(counted(len(report.get('findings', [])), label_kind))
        rows.append((label[:50], '; '.join(parts)))
    return rows or [('Recorded evaluation', 'See evaluation/week3-results/report.json')]


def metric_summary(metrics):
    if not metrics:
        return 'Final Week 3 case results are awaiting recorded verification.'
    summary = metrics.get('summary', {})
    usage_note = ('Reported usage was available for every recorded model call. '
                  if summary.get('usage_complete') is True else
                  'Unavailable usage remains unknown; totals include only reported tokens. ')
    return (
        f'The recorded run scheduled {summary.get("scheduled_cases", "unreported")} cases and reached '
        f'{summary.get("completed_handoffs", "unreported")} saved human review handoffs, with '
        f'{counted(summary.get("failed_cases", 0), "failed case")}. It recorded '
        f'{summary.get("model_calls", "unreported")} model calls, '
        f'{summary.get("tool_calls", "unreported")} tool calls, and '
        f'{summary.get("reported_tokens", 0):,} reported tokens. Recorded AI review removals were '
        f'{counted(summary.get("ai_rejected_findings", 0), "finding")} and '
        f'{counted(summary.get("ai_rejected_next_steps", 0), "proposed next step")}. ' + usage_note +
        'A saved handoff is pending human judgment, not an approved answer or measured accuracy.'
    )


def build_writeup(args, diagram, metrics):
    doc = base_doc('Engineering Issue Desk Week 3 Project Writeup', 'Naresh Ganji\nOctober 22 2026')
    paragraph(doc, 'Engineering Issue Desk extends a Streamlit evidence workspace into a saved investigation agent. An engineer submits a behavior report and optional code; the agent plans research, chooses bounded tools, delegates specialist work, and prepares findings with sources for human review.')
    paragraph(doc, 'The Week 3 contribution is a complete investigation and review workflow built with Deep Agents, Fireworks, Keenable, and LangGraph. The existing retrieval workspace remains available. This submission documents the implemented behavior, the development iterations, and the limits of the verification; a cited report still requires an engineer to judge whether its evidence supports the conclusion.')
    heading(doc, 'Project one liner')
    paragraph(doc, 'Engineering Issue Desk helps a Streamlit engineer investigate an issue in a web workspace, replacing manual documentation searches and public issue comparison; it researches and delegates with five domain read tools, saves the evidence, and hands the report to an engineer for approval, with a proposed target of a usable handoff within ten minutes in eight of ten representative cases.')
    paragraph(doc, 'That workflow target is proposed. The project has not measured an eight of ten human success rate, established ten minute end to end performance, or timed the manual workflow baseline.')
    heading(doc, 'User and scope')
    paragraph(doc, 'The user is a developer or engineering lead deciding what a Streamlit report means, what information is missing, and which reproduction checks to try next. The project adapts the handout\'s code review pattern to engineering issue investigation, using documentation and attributed public reports alongside explicitly supplied code.')
    bullet(doc, 'Research can proceed autonomously within the approved documentation and public repository scope.')
    bullet(doc, 'Correctness and security review, and test or reproduction planning, can be delegated when useful.')
    bullet(doc, 'The engineer can reopen the case, inspect exact evidence, request a revision, and record a final local decision.')
    paragraph(doc, 'The application does not inspect the user\'s installed environment, execute reproduction code, confirm a private root cause, or publish a GitHub issue or review. Three real sample investigations and automated failure checks are assessed separately from any future human quality study.')
    source = paragraph(doc, 'Assignment source: Week 3 Project Handout October 2026. Submission: https://forms.gle/waBYFkkf5xxmcouV9. Tool access: https://tga-cohort-credits-hub.vercel.app/.')
    for run in source.runs:
        run.font.size = Pt(9)

    new_page(doc, 'Agent framework')
    fields = [
        ('Agent goal', 'Produce a prioritized, evidence based investigation report and missing information list for one Streamlit engineering issue. The report remains a draft until an engineer reviews it.'),
        ('Where people use it', 'The Deep Investigation tab in the existing Streamlit application is the main surface. A command line client uses the same saved case and review service for reproducible runs.'),
        ('Steps in order', 'Accept the issue and optional version or code, plan research, choose tools and specialists, validate saved citation IDs, and run a separate evidence support review. Pause for a human decision, then save approval or rejection or investigate again with feedback and the prior report.'),
        ('Available actions', 'Five domain tools search the bundled knowledge base, search the web, fetch an approved page, read a public issue, and read commit pinned PR patches. These tools read information; approval changes the local case record and does not publish externally.'),
        ('What it remembers', 'SQLite preserves the case, immutable evidence, audit events, and graph checkpoints across application restarts. This is durable case state; semantic recall across cases and Mem0 integration are not implemented.'),
        ('Hard limits', 'The agent cannot execute host commands, post to GitHub, or treat a public issue as a verified diagnosis of the user\'s application. Sources and pasted code are untrusted data, and provider or execution limits must stop further dispatch.'),
        ('Human involvement', 'The engineer reviews retained findings, exact source passages, proposed checks, and missing details before approving or rejecting the local report. A revision requires explicit notes and receives the previous report; up to two revisions are allowed.'),
        ('When a step fails', 'Unavailable web tools return a safe error so other evidence or missing information can guide the investigation. A failed saved case can resume an unfinished checkpoint; invalid completed reports can use a fresh repair conversation, at most twice, without resetting the case budget.'),
        ('Success measure', 'The proposed measure is a human judged usable investigation handoff within ten minutes for eight of ten representative cases. Operational completion, source integrity, AI support review, and human usefulness must be reported as separate measures.'),
    ]
    for label, text in fields:
        heading(doc, label, 2)
        paragraph(doc, text)

    new_page(doc, 'Architecture and tools')
    paragraph(doc, 'Deep Agents runs the adaptive research loop on a Fireworks tool calling model. LangGraph wraps it with durable state and an interrupt for human review. Python checks citation identities and resolves source text before the independent evidence review; the model chooses research steps, while deterministic application code controls saved state and final decisions.')
    p = doc.add_paragraph()
    p.add_run().add_picture(str(diagram), width=Inches(6.86))
    paragraph(doc, 'A revision returns to investigation with the reviewer\'s notes and the prior report. Approval and rejection resume the saved review state without another investigation model call. The evidence review is an AI check, not a measured human accuracy score.')
    paragraph(doc, 'Older drafts require a completed evidence review before approval. The saved evidence review action can audit an older report without repeating its research.')
    table(doc, ['Tool', 'Read scope', 'What the result establishes'], [
        ('Knowledge base search', '65 source snapshot', 'Literal passages with snapshot and version scope'),
        ('Keenable search', 'Approved documentation and Streamlit repository pages', 'Discovery snippets that need further inspection'),
        ('Keenable page fetch', 'Approved public HTTPS pages', 'Bounded page content with retrieval time'),
        ('GitHub issue read', 'streamlit/streamlit issue and bounded comments', 'Attributed public reports and observed issue state'),
        ('GitHub PR read', 'Observed base and head commits and up to five patches', 'Partial static code evidence tied to immutable commits'),
    ], [1.5, 2.15, 3.21])
    paragraph(doc, 'Three specialist roles cover documentation and issue research, correctness and security, and tests or reproduction. They share a case evidence registry and budget. The Deep Agents filesystem uses a virtual StateBackend and does not expose the host filesystem or a working shell.')
    paragraph(doc, 'Case state and semantic memory serve different purposes. Checkpoints resume work that is already in progress; an evidence registry preserves what a source actually said. Future Mem0 work would retrieve selected approved resolutions across cases, with provenance and project isolation, and would need its own evaluation.')

    new_page(doc, 'Datasets and source provenance')
    paragraph(doc, 'The bundled corpus contains 65 English records selected for Streamlit engineering questions. The manifest records a snapshot completed October 17 2026 Pacific and an API reference version of 1.66.0. An individual report must still preserve each source\'s date and version scope; the corpus is a bounded sample rather than a complete history.')
    table(doc, ['Source type', 'Records', 'How it is used'], [
        ('Official documentation and API records', '38', 'Framework behavior and supported API context'),
        ('Public GitHub issue reports', '24', 'Attributed symptoms and reproduction material'),
        ('Contribution and template records', '2', 'Issue reporting and contribution guidance'),
        ('Bounded release excerpt', '1', 'Version and release context'),
        ('Total', '65', 'Curated allowlist with per source provenance'),
    ], [2.9, .85, 3.11])
    paragraph(doc, 'The reproducible ingestion script normalizes Markdown, MDX, API docstrings, and public issue text while retaining technical prose, code, headings, and source links. It removes display wrappers, issue boilerplate, and obvious contact or credential strings. The documentation commit begins 7a0a34742a9f and the Streamlit repository commit begins 4278206f51bd; full pins, hashes, allowlists, and licenses are stored with the corpus.')
    heading(doc, 'Live evidence and supplied code', 2)
    paragraph(doc, 'Keenable adds current search snippets and fetched page content, while GitHub tools add public issue context or commit pinned patch excerpts. These records are labeled as live evidence with retrieval timestamps. Supplied code is registered as user code; static inspection does not mean the code was executed or that a failure was reproduced.')
    heading(doc, 'Citation integrity and support', 2)
    paragraph(doc, 'Every saved source has a stable evidence ID, exact text, a content hash, and provenance fields. The application derives quote IDs from literal source spans. The model selects an evidence ID and a quote ID; the application resolves the original characters instead of accepting a rewritten quotation.')
    paragraph(doc, 'A valid source identifier proves that the cited text exists. A separate Fireworks evidence reviewer assesses whether the claim follows from that text and filters statements or workarounds it judges unsupported before human review. It can miss an incorrect recommendation even when the cited text exists. Measured human faithfulness and end to end task success remain separate, unestablished outcomes.')
    paragraph(doc, 'Primary project records are data/sources.json, data/manifest.json, data/licenses, evaluation/week3-cases.json, and the recorded Week 3 evaluation report. Historical Week 2 retrieval and generated answer results remain separate and are not reused as Week 3 agent quality scores.')
    paragraph(doc, 'The Week 3 code review evaluation uses three real repository files: reviews.py, freshness.py, and hosted_embeddings.py. Each sample preserves its original file hash and line range so a reviewer can identify exactly which code was inspected.')

    new_page(doc, 'Prompts and development iterations')
    paragraph(doc, 'Naresh used Codex to extend the existing repository. The initial request asked for a review of the Week 3 handout and supplied solution kits, then for a feature plan that expanded Engineering Issue Desk with Deep Agents and team provided tools. Naresh confirmed access to Keenable and Fireworks and instructed Codex to start building. These requests are paraphrased here; they are development direction, not agent source evidence.')
    heading(doc, 'Development prompts used', 2)
    paragraph(doc, 'The first two summaries capture user direction. The final summary describes a Codex implementation check used during development.')
    bullet(doc, 'Review the Week 3 project and solution examples, then identify how Engineering Issue Desk can grow into an agent workflow.')
    bullet(doc, 'Use the confirmed Fireworks and Keenable access with Deep Agents, preserve the existing application, and implement the new investigation mode.')
    bullet(doc, 'Exercise the real tool calling harness offline, persist human review across restarts, and make failures and provider usage inspectable.')
    paragraph(doc, 'Codex then generated implementation and test changes, ran independent code reviews, and revised the design in response to observed failures. The runtime prompt tells the investigator to treat code and sources as data, select registered quote IDs, attribute public reports, request missing details, and separate proposed tests from observed results.')
    heading(doc, 'Changes driven by observed failures', 2)
    paragraph(doc, 'Literal quotes were initially copied into the model\'s report. A live run modified source text and failed validation, so support now uses deterministic quote IDs and the application supplies the exact saved text.')
    paragraph(doc, 'Process locks and startup recovery were added after reviewing the difference between a saved case and an actively running execution. Actual subprocess crash tests now verify that a retry resumes a completed tool\'s checkpoint instead of repeating the read.')
    paragraph(doc, 'The default automatic summarizer could call a model outside the shared budget. It was replaced with an explicit context size cap for the investigator and each specialist. New completion metadata records truncation and malformed tool counts while preserving reported usage and omitting raw arguments from audit events.')
    paragraph(doc, 'Invalid completed reports now use a fresh repair thread with the original issue and saved evidence registry. Repairs retain cumulative case limits. Structured output checkpoints use JSON dictionaries, and revisions carry the previous report so feedback such as correcting a specific finding has enough context.')
    paragraph(doc, 'A later sample produced valid citations but included an incorrect uploader recommendation involving on_change set to ignore. A separate evidence review removed one finding and one proposed step but retained a recommendation that did not address the checkbox triggered rerun. The prompt was tightened to check causal applicability; AI review still requires human judgment.')
    paragraph(doc, 'An early run failed literal quote validation and reported 78,956 tokens after its final in flight call exceeded the 64,000 pre dispatch threshold. The later uploader case reported 57,361 tokens for research and 60,748 including its extra audit. It remained unapproved. These exploratory runs informed changes and are separate from the final three sample evaluation.')

    new_page(doc, 'Verification results and remaining work')
    paragraph(doc, f'The regression suite has {args.test_count} passing tests, including the real Deep Agents factory with an offline model, SQLite checkpoints, specialist evidence sharing, citation rejection, shared limits, review replay protection, revisions, and subprocess crash recovery. These tests verify execution behavior rather than live model quality. A separate provider probe verified a Fireworks tool round trip and five Keenable search results.')
    paragraph(doc, metric_summary(metrics))
    table(doc, ['Recorded Week 3 case', 'Observed result'], metric_rows(metrics), [2.0, 4.86])
    paragraph(doc, 'All three drafts passed literal citation checks, including the failed W3-02 draft. W3-02 stopped because its evidence review response failed validation. No evaluation report was human approved. This static code run used three saved evidence reads; it did not exercise delegation or Keenable, whose connectivity was tested separately.')
    paragraph(doc, 'W3-03 retained unsupported claims about a missing empty key check and an error body leak. The configuration helper was outside the supplied excerpt, and the visible exception path sanitized provider content. This concrete false positive shows that an AI audit can miss scope and causal applicability. Human quality scores remain unmeasured.')
    heading(doc, 'Known limits', 2)
    bullet(doc, 'Review is static. Proposed tests were not executed, and incomplete public evidence cannot establish a private fix or duplicate.')
    bullet(doc, 'The case limits are 18 model calls, 28 tools, two revisions, and two repairs. The 64,000 reported token threshold and 180 second active invocation deadline are checked before dispatch; an in flight request can finish beyond them. Missing usage is unknown.')
    heading(doc, 'Submission requirements', 2)
    paragraph(doc, 'Submit a Google Doc, a video of five minutes or less, and a GitHub code link. Builder eligibility closes October 25 2026; certification closes November 11 2026. The demo guide targets 4 minutes 45 seconds. The video and final Week 3 repository push remain outstanding submission actions.')
    return doc


def cue(doc, timing, screen, narration):
    heading(doc, timing, 2)
    paragraph(doc, 'Screen cue: ' + screen, bold_lead='Screen cue:')
    paragraph(doc, 'Narration: ' + narration, bold_lead='Narration:')


def build_demo(args, metrics):
    doc = base_doc('Engineering Issue Desk Week 3 Demo Guide', 'Naresh Ganji\nFour minutes and forty five seconds')
    paragraph(doc, 'Record a concise walkthrough of the working investigation and human review flow. The target runtime is 4 minutes 45 seconds, leaving 15 seconds below the handout\'s five minute limit. This is a recording script and preparation guide; it does not claim a video has already been recorded.')
    heading(doc, 'Before recording')
    bullet(doc, 'Start the Streamlit app and confirm the Deep Investigation tab can open. Keep credentials and .env outside the recording surface.')
    bullet(doc, 'Choose a small public issue question. Have one previously recorded real case ready as a latency fallback, with its actual date, findings, and review state visible.')
    bullet(doc, 'Open the architecture diagram and recorded Week 3 results in separate tabs. Use a readable zoom level and test the microphone.')
    bullet(doc, 'Avoid pre approving the selected case. The recording should show a real local review decision or explicitly show that the case remains pending.')
    heading(doc, 'Suggested live input', 2)
    paragraph(doc, 'My Streamlit CSV upload appears to reset after another widget changes. Investigate the documented rerun and session state behavior. Identify which version and minimal reproduction details are missing. Keep findings limited to what the saved sources directly support.')
    paragraph(doc, 'Leave the version blank unless it is known. Add code only when it is the code actually being discussed. The run may choose different tools from a previous run; narrate observed events rather than promising a fixed sequence.')
    cue(doc, 'Opening through 25 seconds', 'Show the issue workspace, then the Deep Investigation tab.',
        'Engineering Issue Desk helps a developer investigate a Streamlit issue and prepare a report for review. Week 3 adds an agent that chooses research steps and can delegate specialist work. The engineer still decides whether the evidence supports the proposed conclusion.')
    cue(doc, '25 seconds through 1 minute 5 seconds', 'Enter the suggested issue and start a genuine investigation. Keep the status panel visible.',
        'The report starts an investigation rather than a single answer call. Fireworks powers the model, and Keenable supplies current public research when needed. The local knowledge base and public GitHub readers are also available. This run saves its sources and progress as it works.')

    new_page(doc, 'Show the investigation and evidence')
    cue(doc, '1 minute 5 seconds through 2 minutes 15 seconds', 'Show the live plan and completed tool events. Briefly open the architecture diagram while the run continues.',
        'Deep Agents decides what to read next. It can ask documentation research, correctness and security, or test and reproduction specialists for help. Each role shares the same case evidence and usage allowance. LangGraph saves the research state and pauses for human review. The tools read public information; they do not execute my code or publish a GitHub issue. If a tool fails, that failure is recorded and the agent must use other evidence or say what is missing.')
    cue(doc, '2 minutes 15 seconds through 3 minutes 25 seconds', 'Open the resulting report and one source record. Show a finding, its exact quote, and provenance. If the live run is still working, use the fallback below.',
        'The application assigns quote IDs to saved source passages. The model selects those IDs, and the application supplies the original text. This prevents a paraphrase from being presented as an exact quotation. A separate model check reviews whether the evidence supports each finding before the report reaches human review. That extra check can still make mistakes. I inspect the actual passage, its date and version scope, and whether it just reports a public symptom or establishes documented behavior.')
    heading(doc, 'If the provider run is slow', 2)
    paragraph(doc, 'Say: This live investigation is still running. I will open a previously completed real case to demonstrate the report and review steps; its recorded events show what actually happened. Switch visibly to that saved case. Do not imply its result belongs to the new run or remove evidence of an error to suggest a successful run.')
    heading(doc, 'If a tool or model fails', 2)
    paragraph(doc, 'Show the saved failure and its recorded steps. Explain that retry resumes an unfinished checkpoint, while invalid completed reports can use a separate repair conversation with the same case evidence and cumulative limits. Demonstrate a retry only if the remaining recording time permits it; otherwise show a real saved case for the review interaction.')
    paragraph(doc, 'Do not read raw provider content or private code aloud. A sanitized error and the visible saved timeline are enough to demonstrate the recovery behavior.')

    new_page(doc, 'Complete human review and close')
    cue(doc, '3 minutes 25 seconds through 4 minutes 20 seconds', 'Show missing information and proposed checks. Enter a genuine review note and approve or reject, or request a revision if the report needs correction. Reopen the saved case.',
        'The proposed checks are not tests that this agent has executed. I can approve the local report, reject it, or request a revision with notes. Approval and rejection do not rerun the investigator and do not post anything externally. The case remains available after a restart, with its evidence and review decision. A revision gets the prior report and my feedback so it can address the specific problem. Persistent case state is implemented; Mem0 based recall across cases is future work.')
    cue(doc, '4 minutes 20 seconds through 4 minutes 45 seconds', 'Show the recorded evaluation summary and the repository or writeup title.',
        'Codex helped implement and test the tool loop, saved review, and failure recovery. Live testing led to quote IDs and a separate evidence check. That AI check still missed an incorrect recommendation, so human review remains essential. Automated tests verify workflow behavior; the quality target needs human evaluation.')
    heading(doc, 'Evidence to mention accurately', 2)
    paragraph(doc, f'The current regression suite contains {args.test_count} passing tests. Provider connectivity was verified with a Fireworks tool round trip and a Keenable search. Describe the recorded case outcomes below as handoffs pending human judgment; AI review does not establish human accuracy.')
    table(doc, ['Recorded case', 'Result to describe'], metric_rows(metrics), [2.0, 4.86])
    heading(doc, 'After recording', 2)
    bullet(doc, 'Check that the finished video is five minutes or less, readable, audible, and free of credentials.')
    bullet(doc, 'Attach the native Google Doc, the recorded video link, and the pushed GitHub code link to the submission form.')
    bullet(doc, 'Confirm link access from the intended reviewer account. Builder consideration closes October 25 2026; certification closes November 11 2026.')
    paragraph(doc, 'The video recording and final Week 3 GitHub push are outstanding actions. This guide does not mark either action complete.')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-json', type=Path, default=ROOT / 'evaluation/week3-results/report.json')
    parser.add_argument('--test-count', type=int, default=401)
    parser.add_argument('--runtime-root', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--skill-root', type=Path, default=DEFAULT_SKILL)
    parser.add_argument('--qa-dir', type=Path, default=ROOT / '.cache/week3-docs-qa')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    runtime_python = args.runtime_root / 'dependencies/python/bin/python3.12'
    if Path(sys.executable).resolve() != runtime_python.resolve():
        raise SystemExit('Run this builder using the selected bundled artifact Python runtime.')
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / 'docs').mkdir(exist_ok=True)
    diagram = architecture(ROOT, args.qa_dir, args.runtime_root)
    metrics = read_metrics(args.metrics_json)
    documents = [(WRITEUP, build_writeup(args, diagram, metrics), 'writeup'),
                 (DEMO, build_demo(args, metrics), 'demo')]
    sanitizer = args.skill_root / 'scripts/google_docs_title_sanitize.py'
    for filename, doc, label in documents:
        raw = args.qa_dir / ('raw-' + filename)
        destination = ROOT / 'docs' / filename
        doc.save(raw)
        subprocess.run([str(runtime_python), str(sanitizer), str(raw), '--out', str(destination)], check=True)
        subprocess.run([str(runtime_python), str(sanitizer), str(destination), '--check'], check=True)
        if args.render:
            render_dir = args.qa_dir / label
            render_dir.mkdir(exist_ok=True)
            for old_page in render_dir.glob('page-*.png'):
                old_page.unlink()
            subprocess.run([str(runtime_python), str(args.skill_root / 'render_docx.py'), str(destination),
                            '--output_dir', str(render_dir), '--emit_pdf'], check=True)
        print(destination)


if __name__ == '__main__':
    main()
