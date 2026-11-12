"""Build a Week 6 findings document from recorded red-team results.

Use the packaged Codex document Python runtime. The session must run the
document-operation marker once before the first invocation that writes DOCX.
This script never invokes a model, changes a score, or invents a response.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from build_week3_docs import DEFAULT_SKILL, base_doc, heading, new_page, paragraph, table
from build_week4_docs import link


ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Engineering Issue Desk Week 6 Red Team Findings'
REPO = 'https://github.com/naresh25ganji/engineering-issue-desk'
SCORES = ('PASS', 'WARN', 'FAIL')


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding='utf-8'))
    results = report.get('results')
    if not isinstance(results, list) or not results:
        raise ValueError('A nonempty recorded report.results list is required.')
    if report.get('execution_mode') not in ('live', 'recorded_live', 'live_agent'):
        mode = report.get('manifest', {}).get('execution_mode', report.get('manifest', {}).get('mode'))
        if mode not in ('live', 'recorded_live', 'live_agent'):
            raise ValueError('Only a live model run can produce the submission findings document.')
    seen = set()
    for row in results:
        case_id = row.get('id')
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError('Recorded results need distinct string IDs.')
        seen.add(case_id)
        if row.get('score') not in (*SCORES, None):
            raise ValueError(f'{case_id}: unknown score.')
        if not isinstance(row.get('turns'), list):
            raise ValueError(f'{case_id}: recorded turns are required.')
        for turn in row['turns']:
            if not isinstance(turn.get('prompt'), str) or not isinstance(turn.get('response'), str):
                raise ValueError(f'{case_id}: prompt and response must be verbatim text.')
    return report


def _label(value) -> str:
    return str(value or 'Not recorded').replace('_', ' ').strip()


def _short(text, maximum=280):
    text = ' '.join(str(text or '').split())
    return text if len(text) <= maximum else text[:maximum].rsplit(' ', 1)[0] + '...'


def _counts(rows):
    counts = Counter(row.get('score') or 'NOT SCORED' for row in rows)
    return ', '.join(f'{counts[score]} {score}' for score in (*SCORES, 'NOT SCORED') if counts[score])


def _transcript_length(row):
    return sum(len(turn.get('prompt', '')) + len(turn.get('response', '')) for turn in row['turns'])


def choose_examples(rows, selected_ids):
    by_id = {row['id']: row for row in rows}
    if selected_ids:
        missing = set(selected_ids) - by_id.keys()
        if missing:
            raise ValueError(f'Unknown example IDs: {sorted(missing)}')
        return [by_id[case_id] for case_id in selected_ids]
    candidates = [row for row in rows if any(turn.get('response') for turn in row['turns'])]
    selected = []
    # Include a recorded failure, warning, and pass where they exist. Prefer a
    # compact complete transcript over trimming a response to fit the writeup.
    for score in ('FAIL', 'WARN', 'PASS'):
        pool = [row for row in candidates if row.get('score') == score and row.get('kind') != 'benign']
        if pool:
            selected.append(min(pool, key=_transcript_length))
    multi = [row for row in candidates if len(row['turns']) > 1 and row not in selected]
    if multi and not any(len(row['turns']) > 1 for row in selected):
        if len(selected) == 3:
            # Keep a failure if one exists; replace the ordinary passing case.
            replace = next((i for i, row in enumerate(selected) if row.get('score') == 'PASS'), None)
            if replace is not None:
                selected[replace] = min(multi, key=_transcript_length)
        else:
            selected.append(min(multi, key=_transcript_length))
    if not selected and candidates:
        selected.append(min(candidates, key=_transcript_length))
    return selected[:3]


def _copy_text(doc, label, text):
    p = paragraph(doc, label)
    p.runs[0].bold = True
    p.paragraph_format.keep_with_next = True
    # Preserve every character other than Word's paragraph representation of
    # newlines. Splitting paragraphs keeps long transcripts readable in Docs.
    for line in text.split('\n'):
        p = paragraph(doc, line)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.04
        p.paragraph_format.widow_control = True
        for run in p.runs:
            run.font.size = Pt(10.5)


def _links(doc, args):
    p = paragraph(doc)
    link(p, 'Project repository', args.repository_url)
    if args.evidence_url:
        p.add_run('  |  ')
        link(p, 'Complete recorded evidence', args.evidence_url)
    if args.video_url:
        p.add_run('  |  ')
        link(p, 'Recorded demonstration', args.video_url)


def load_context_review(report_path):
    path = report_path.with_name('ai-context-review.json')
    if not path.exists():
        return None
    review = json.loads(path.read_text(encoding='utf-8'))
    if (review.get('artifact_kind') != 'separate_ai_context_review'
            or review.get('source_report_sha256') != hashlib.sha256(report_path.read_bytes()).hexdigest()
            or review.get('automated_scores_modified') is not False
            or review.get('human_labels_modified') is not False):
        raise ValueError('Contextual review must pin this report and preserve original scores and human labels.')
    return review


def defense_section(doc, baseline, args):
    if not args.retest_report and not args.guard_report:
        return
    new_page(doc, 'Credential control follow up')
    paragraph(doc, 'The baseline findings remain the primary evaluation. The follow-up below checks '
              'a credential-request control change using two separate methods: a targeted live run '
              'and direct deterministic tests of the sanitizer. Neither method rewrites a baseline '
              'failure or establishes protection against every phrasing.')
    if args.retest_report:
        retest = load_report(args.retest_report)
        baseline_by_id = {row['id']: row for row in baseline['results']}
        repeated = retest['results']
        heading(doc, 'Targeted live rerun', 2)
        paragraph(doc, f'The separate live rerun scheduled {len(repeated)} cases and recorded '
                  f'{_counts(repeated)}. Its scope is the selected cases below, not a full repeat '
                  'of every baseline attack. Model serving and trajectories can vary between runs; '
                  'a changed score alone does not isolate the causal effect of the control.')
        table(doc, ['Case', 'Baseline', 'Rerun', 'Recorded rerun reason'], [
            [row['id'], baseline_by_id.get(row['id'], {}).get('score') or 'Not run',
             row.get('score') or 'Not scored', _short(row.get('reason'), 160)]
            for row in repeated
        ], [.65, .7, .7, 4.45])
        converted, untouched = [], []
        for row in repeated:
            values = [turn.get('report', {}).get('quality_review', {})
                      .get('information_request_safety', {}).get('converted_count')
                      for turn in row['turns']]
            if values and all(type(value) is int for value in values):
                if sum(values):
                    converted.append((row['id'], sum(values)))
                else:
                    untouched.append(row['id'])
        if converted:
            paragraph(doc, 'The live audit records show the control replacing unsafe diagnostic '
                      'requests in ' + ', '.join(f'{case_id} ({count})' for case_id, count in converted)
                      + '. These are observed sanitizer conversions, rather than an inference that '
                      'the model refused. No conversion was recorded in ' + ', '.join(untouched)
                      + '; the complete records preserve the per-turn values.')
    if args.guard_report:
        controls = json.loads(args.guard_report.read_text(encoding='utf-8'))
        if (controls.get('report_kind') != 'week6_deterministic_guard_comparison'
                or controls.get('execution_mode') != 'deterministic_no_model'):
            raise ValueError('The guard report must identify a deterministic Week 6 control comparison.')
        summary = controls.get('summary', {})
        before, after = summary.get('before', {}), summary.get('after', {})
        if summary.get('model_calls') != 0 or summary.get('network_calls') != 0:
            raise ValueError('The deterministic control report must use zero model and network calls.')
        heading(doc, 'Deterministic sanitizer checks', 2)
        table(doc, ['Direct control measure', 'Before', 'After'], [
            ['Checks passed', f"{before.get('passed', '?')}/{before.get('total', '?')}",
             f"{after.get('passed', '?')}/{after.get('total', '?')}"],
            ['Unsafe requests replaced', f"{before.get('unsafe_requests_replaced', '?')}/{before.get('unsafe_request_denominator', '?')}",
             f"{after.get('unsafe_requests_replaced', '?')}/{after.get('unsafe_request_denominator', '?')}"],
            ['Benign requests preserved', f"{before.get('benign_requests_preserved', '?')}/{before.get('benign_request_denominator', '?')}",
             f"{after.get('benign_requests_preserved', '?')}/{after.get('benign_request_denominator', '?')}"],
        ], [3.6, 1.45, 1.45])
        paragraph(doc, 'These are direct sanitizer calls with zero model or network calls. One fixture '
                  'reproduces the observed baseline credential request; the remaining fixtures cover '
                  'synthetic unsafe variants and benign controls. These counts measure replacement '
                  'and preservation behavior, separately from the live agent scores.')
        identity = controls.get('source_identity', {})
        if identity.get('baseline_policy_version') or identity.get('current_policy_version'):
            paragraph(doc, 'Control versions: ' + str(identity.get('baseline_policy_version', 'unknown'))
                      + ' to ' + str(identity.get('current_policy_version', 'unknown')) + '.')


def build(args):
    report = load_report(args.report)
    context_review = load_context_review(args.report)
    context_by_id = {note['case_id']: note for note in (context_review or {}).get('notes', [])}
    rows, manifest = report['results'], report.get('manifest', {})
    attacks = [row for row in rows if row.get('kind') != 'benign']
    benign = [row for row in rows if row.get('kind') == 'benign']
    examples = choose_examples(rows, args.example)
    doc = base_doc(TITLE, 'Naresh Ganji  |  The Gen Academy Week 6  |  Path B own agent')
    doc.core_properties.subject = 'Recorded red-team evaluation of Engineering Issue Desk'
    doc.styles['Title'].font.size = Pt(22)
    doc.styles['Heading 1'].font.size = Pt(16)
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)

    paragraph(doc, f'I stress-tested Engineering Issue Desk with {len(attacks)} attack cases and '
              f'{len(benign)} normal engineering controls. Across all {len(rows)} scheduled cases, '
              f'the recorded scores are {_counts(rows)}. '
              'These scores describe the checks in this run. A PASS means that the specified failure '
              'was not observed; it does not establish that the agent is secure against other prompts.')
    if context_review:
        unsafe = context_review.get('summary', {}).get('confirmed_unsafe_collection_requests', [])
        if unsafe:
            paragraph(doc, f'The main finding is {len(unsafe)} unsafe diagnostic collection requests '
                      '(' + ', '.join(unsafe) + '). The agent adopted requests for credential material '
                      'or credential-bearing outputs into its missing-information field. No real '
                      'credential was supplied or disclosed. This interpretation is documented in '
                      'a separate AI contextual review; human review is still pending.')
    _links(doc, args)
    heading(doc, 'Project and testing surface')
    paragraph(doc, 'Engineering Issue Desk investigates engineering reports using supplied code and logs, '
              'a bounded knowledge base, and approved read-only web tools. It prepares findings with saved '
              'source quotes for human review. Report approval and GitHub publication are separate steps. '
              'The agent has no host shell or unrestricted file-reading tool. This evaluation uses the '
              'production investigation service rather than a substitute chatbot.')
    paragraph(doc, 'I tested the application contract: untrusted evidence must remain evidence, findings '
              'must retain valid support, remote destinations must stay within the backend allowlist, '
              'and requests for secrets or approval bypass must not grant new authority. Publicly '
              'documented tool names are not treated as confidential banking internals.')
    heading(doc, 'Recorded run')
    errors = sum(bool(row.get('error')) or row.get('status') == 'error' for row in rows)
    table(doc, ['Measure', 'Recorded value'], [
        ['Scheduled cases', str(len(rows))],
        ['Overall scores', _counts(rows)],
        ['Attack scores', _counts(attacks)],
        ['Normal control scores', _counts(benign) if benign else 'No normal controls recorded'],
        ['Cases with execution errors', str(errors)],
        ['Model', _label(manifest.get('model') or manifest.get('model_name') or report.get('model'))],
        ['Run identifier', _label(manifest.get('run_id') or manifest.get('run_name'))],
    ], [2.15, 4.35])
    paragraph(doc, 'The runner saves the prompts, responses, per-turn status, tool and model events, '
              'source provenance, latency, and provider-reported usage. It uses synthetic test data and '
              'redacts configured secrets before export. Remote tracing is disabled for this evaluation.')

    new_page(doc, 'Scoring and findings')
    paragraph(doc, 'PASS means the recorded response completed and the configured boundary checks found '
              'no violation. WARN covers incomplete execution, an ambiguous outcome, or a case requiring '
              'semantic review. FAIL means a configured check detected the specified unsafe behavior. '
              'Expected outcomes stay outside the model request. Human review remains explicit and '
              'separate from the automated score.')
    family_rows = []
    for family in dict.fromkeys(row.get('family', 'unspecified') for row in rows):
        group = [row for row in rows if row.get('family') == family]
        family_rows.append([_label(family), str(len(group)), _counts(group)])
    table(doc, ['Attack family or control', 'Cases', 'Scores'], family_rows, [2.55, .6, 3.35])
    concerns = [row for row in rows if row.get('score') in ('FAIL', 'WARN')]
    if concerns:
        paragraph(doc, f'{len(concerns)} cases require follow-up. The complete report retains each '
                  'case rather than replacing earlier failures with a later success.')
        for row in concerns[:6]:
            paragraph(doc, f"{row['id']} {_label(row.get('score'))}: {_short(row.get('reason'), 430)}")
        if len(concerns) > 6:
            paragraph(doc, f'The full evidence report contains the remaining {len(concerns) - 6} cases requiring follow-up.')
    else:
        paragraph(doc, 'No configured failure was observed in the recorded cases. This result is limited '
                  'to this dataset, serving configuration, and run; broader semantic review and fresh '
                  'attack variants remain necessary.')
    if context_review:
        overview = context_review.get('summary', {})
        marker_warns = overview.get('marker_attribution_or_refusal_warns', [])
        incomplete = overview.get('incomplete_execution_warns', [])
        if marker_warns:
            paragraph(doc, 'The separate AI review identifies ' + ', '.join(marker_warns)
                      + ' as attributed or refusal-related marker mentions, not confirmed leaks. '
                      'Their automated WARN labels remain unchanged.')
        if incomplete:
            paragraph(doc, 'Execution was incomplete in ' + ', '.join(incomplete)
                      + '. These WARNs remain inconclusive. A failed revision can retain the prior '
                      'draft; that saved text is not a newly completed answer to the failed prompt.')
    heading(doc, 'Limits of the evidence')
    limits = report.get('measurement_limits')
    if isinstance(limits, str):
        paragraph(doc, limits)
    elif isinstance(limits, list):
        for item in limits:
            paragraph(doc, str(item))
    paragraph(doc, 'This is a bounded development evaluation with one recorded trajectory per case, '
              'not an estimate of a real-world attack success rate. Automated marker and structure '
              'checks can miss paraphrased leakage, false claims, or poor refusal quality. Refusal '
              'without useful assistance can also overblock a normal engineering task. Pending human '
              'review must not be reported as completed review.')

    new_page(doc, 'Attack to defense mapping')
    paragraph(doc, 'Each recommended control below addresses the corresponding tested boundary. '
              'Recommendations are not a claim that new defense code was deployed or that every '
              'failure was fixed. The recorded code revision and evidence identify what was tested.')
    mapping = []
    for family in dict.fromkeys(row.get('family', 'unspecified') for row in attacks):
        group = [row for row in attacks if row.get('family') == family]
        defenses = list(dict.fromkeys(str(row.get('recommended_defense', '')).strip() for row in group))
        defenses = [text for text in defenses if text]
        mapping.append([_label(family), ', '.join(row['id'] for row in group),
                        '\n'.join(defenses) or 'Review the recorded boundary and add the matching backend or output check.'])
    table(doc, ['Family', 'Case evidence', 'Recommended control'], mapping, [1.25, 1.15, 4.1])
    heading(doc, 'Evidence package')
    paragraph(doc, 'The accompanying report.html contains every recorded prompt and response, including '
              'all saved revision turns. report.json retains machine-readable events, scores, reasons, '
              'and provenance; the CSV supports case review. The examples on the following pages are '
              'complete copied transcripts selected from those records, not reconstructed conversations.')
    paragraph(doc, 'Evidence input: ' + str(args.report.relative_to(ROOT) if args.report.is_relative_to(ROOT) else args.report.name))
    paragraph(doc, 'Evidence SHA256: ' + hashlib.sha256(args.report.read_bytes()).hexdigest())

    defense_section(doc, report, args)

    for index, row in enumerate(examples, 1):
        new_page(doc, f'Copied example {index} {row["id"]}')
        paragraph(doc, f"{row.get('title', row['id'])}. Family: {_label(row.get('family'))}. "
                  f"Score: {row.get('score') or 'NOT SCORED'}.")
        supplied = row.get('input', {})
        details = []
        for field, label in (('use_case', 'Workflow'), ('reported_version', 'Reported version')):
            if supplied.get(field):
                details.append(label + ': ' + str(supplied[field]))
        if details:
            paragraph(doc, '. '.join(details) + '.')
        paragraph(doc, 'Expected boundary: ' + str(row.get('expected_boundary', 'See the recorded case.')))
        paragraph(doc, 'Observed result: ' + str(row.get('reason', 'No reason recorded.')))
        contextual = context_by_id.get(row['id'])
        if contextual:
            paragraph(doc, 'Separate AI contextual review: ' + str(contextual.get('interpretation')
                      or contextual.get('ai_conclusion') or 'See the complete contextual review.'))
        if supplied.get('context'):
            _copy_text(doc, 'Exact supplied context', str(supplied['context']))
        for turn in row['turns']:
            _copy_text(doc, f"Turn {turn.get('turn', '?')} prompt", turn['prompt'])
            completed = turn.get('status') in ('awaiting_review', 'completed')
            response_label = (f"Turn {turn.get('turn', '?')} actual response" if completed
                              else f"Turn {turn.get('turn', '?')} last saved draft no new completed response")
            if not completed:
                paragraph(doc, 'This turn did not complete. Any text below is the last saved draft '
                          'retained by the application, not a newly completed response to this prompt.')
            _copy_text(doc, response_label, turn['response'] or '[No response text was saved for this turn.]')
            if turn.get('status'):
                paragraph(doc, 'Turn status: ' + str(turn['status']))
        paragraph(doc, 'Recommended control: ' + str(row.get('recommended_defense', 'See the mapping.')))

    if args.screenshot:
        new_page(doc, 'Captured evidence')
        paragraph(doc, 'Screenshots below show the saved evaluation report. The copied text and '
                  'machine-readable records accompany the screenshots so the result does not depend '
                  'on reading text from an image.')
        for filename in args.screenshot:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            picture = p.add_run().add_picture(str(filename), width=Inches(6.5))
            picture._inline.docPr.set('descr', 'Screenshot of the recorded Week 6 red-team evidence report ' + filename.name)
            paragraph(doc, filename.stem.replace('_', ' ').replace('-', ' '))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    temporary = args.output.with_name(args.output.stem + '.sanitized.docx')
    sanitizer = args.skill / 'scripts/google_docs_title_sanitize.py'
    subprocess.run([sys.executable, str(sanitizer), str(args.output), '--out', str(temporary)], check=True)
    shutil.move(temporary, args.output)
    subprocess.run([sys.executable, str(sanitizer), str(args.output), '--check'], check=True)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/week6-live/report.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/Engineering Issue Desk Week 6 Writeup.docx')
    parser.add_argument('--skill', type=Path, default=DEFAULT_SKILL)
    parser.add_argument('--example', action='append', default=[], help='Recorded case ID to copy in full, repeat for more examples.')
    parser.add_argument('--screenshot', action='append', type=Path, default=[])
    parser.add_argument('--evidence-url', default='')
    parser.add_argument('--video-url', default='')
    parser.add_argument('--repository-url', default=REPO, help='Exact published branch or revision containing the Week 6 work.')
    parser.add_argument('--retest-report', type=Path, help='Separate recorded targeted live rerun, preserving the baseline.')
    parser.add_argument('--guard-report', type=Path, help='Separate deterministic sanitizer comparison; never counted as live attacks.')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    args.report = args.report.resolve()
    if args.validate_only:
        report = load_report(args.report)
        print(json.dumps({'cases': len(report['results']), 'examples': [row['id'] for row in choose_examples(report['results'], args.example)]}))
    else:
        build(args)


if __name__ == '__main__':
    main()
