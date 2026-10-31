"""Build the final Week 4 analysis from matched, completed v2 runs.

Only the recorded-results section of the existing solution document is replaced.
This script makes no model calls and does not alter labels or run artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'diagnostic', 'scope', 'authority', 'calibrated')
METRICS = ('case_exact_match', 'unsupported_rejection', 'supported_retention',
           'p95_latency_seconds', 'mean_total_tokens')
ROOTS = ('diagnostic_over_rejection', 'static_artifact_over_rejection',
         'unsupported_extrapolation', 'unknown_or_error')
ROOT_LABELS = {
    'diagnostic_over_rejection': 'Supported diagnostic removed',
    'static_artifact_over_rejection': 'Supported static/fixture finding removed',
    'unsupported_extrapolation': 'Unsupported item retained',
    'unknown_or_error': 'Unknown verdict or execution error',
}
START = '<!-- BEGIN GENERATED WEEK4 RESULTS -->'
END = '<!-- END GENERATED WEEK4 RESULTS -->'
TIME_ASSUMPTION = {
    'supported_item_rejection_minutes': 5,
    'unsupported_item_acceptance_minutes': 30,
    'unknown_or_error_minutes': None,
    'meaning': 'Hypothetical triage-time weighting, not observed business cost or measured time saved. '
               'Both supported findings and supported proposals receive the 5-minute rejection weight. '
               'Unknown/error items are counted separately and remain unpriced.',
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def reason(row, item):
    key = 'finding_verdicts' if item['kind'] == 'findings' else 'next_step_verdicts'
    verdicts = (row.get('quality_review') or {}).get(key, [])
    return next((v.get('reason', '') for v in verdicts if v.get('index') == item['index']),
                (row.get('error') or {}).get('message', 'No verdict returned.'))


def item_key(case_id, item):
    return f"{case_id}:{item['kind']}:{item['index']}"


def wrong_item(row, item):
    if row['status'] != 'completed' or item['predicted'] is None or item['error_type'] == 'unknown':
        category = 'unknown_or_error'
    elif item['expected']:
        category = 'diagnostic_over_rejection' if item['kind'] == 'next_steps' else 'static_artifact_over_rejection'
    else:
        category = 'unsupported_extrapolation'
    annotated = row['annotations'][item['kind']][item['index']]
    return {
        'case_id': row['id'], 'kind': item['kind'], 'index': item['index'],
        'item_key': item_key(row['id'], item), 'expected': item['expected'],
        'predicted': item['predicted'], 'category': category,
        'statement_or_proposal': annotated.get('statement', annotated.get('proposal')),
        'reference_rationale': annotated['rationale'], 'actual_reason': reason(row, item),
        'error': row.get('error'), 'case_status': row['status'],
        'source': {field: row['source'][field] for field in ('path', 'start_line', 'end_line', 'sha256')},
    }


def analyze_run(variant, report, path):
    rows = report['rows']
    failures = [wrong_item(row, item) for row in rows for item in row['scores']['items']
                if not item['correct']]
    clusters = []
    for category in ROOTS:
        items = [item for item in failures if item['category'] == category]
        unit = None if category == 'unknown_or_error' else 30 if category == 'unsupported_extrapolation' else 5
        clusters.append({
            'category': category, 'item_count': len(items),
            'case_count': len({item['case_id'] for item in items}),
            'case_ids': sorted({item['case_id'] for item in items}),
            'hypothetical_minutes': len(items) * unit if unit is not None else None,
            'example_actual_reasons': [{'item_key': item['item_key'], 'reason': item['actual_reason']}
                                       for item in items[:2]],
        })
    token_fields = ('input_tokens', 'output_tokens', 'total_tokens')
    token_totals = {
        field: sum(row['accounting']['reported_token_usage'][field] for row in rows)
        if all(row['accounting']['model_calls_missing_usage_by_field'][field] == 0 for row in rows) else None
        for field in token_fields
    }
    latencies = [row['latency_seconds'] for row in rows if row.get('latency_seconds') is not None]
    returned_models = sorted({event['returned_model'] for row in rows for event in row.get('events', [])
                              if isinstance(event.get('returned_model'), str)})
    return {
        'variant': variant, 'report_path': str(path.relative_to(ROOT)), 'report_sha256': digest(path),
        'manifest': report['manifest'], 'summary': report['summary'],
        'model': report['manifest']['model'], 'returned_models': returned_models,
        'prompt_sha256': report['manifest']['prompt_sha256'],
        'model_calls': sum(row['accounting']['model_calls'] for row in rows),
        'tool_calls': sum(row['accounting']['tool_calls'] for row in rows),
        'calls_per_case': dict(Counter(row['accounting']['model_calls'] for row in rows)),
        'reported_token_totals': token_totals,
        'usage_complete_cases': sum(row['accounting']['usage_complete'] for row in rows),
        'latency_seconds': {'minimum': min(latencies) if latencies else None,
                            'median': statistics.median(latencies) if latencies else None,
                            'mean': statistics.mean(latencies) if latencies else None,
                            'maximum': max(latencies) if latencies else None,
                            'p95': report['summary']['metrics']['p95_latency_seconds']['value'],
                            'observed_cases': len(latencies)},
        'failure_item_count': len(failures), 'failure_case_count': len({item['case_id'] for item in failures}),
        'failure_case_ids': sorted({item['case_id'] for item in failures}),
        'failure_clusters': clusters, 'wrong_items': failures,
        'hypothetical_known_error_minutes': sum(c['hypothetical_minutes'] or 0 for c in clusters),
        'unpriced_unknown_items': sum(item['category'] == 'unknown_or_error' for item in failures),
    }


def paired(previous_name, current_name, reports):
    before, after = reports[previous_name], reports[current_name]
    old_rows = {row['id']: row for row in before['rows']}
    new_rows = {row['id']: row for row in after['rows']}
    fixes, regressions, cases = [], [], []
    for case_id, old in old_rows.items():
        new = new_rows[case_id]
        old_items = {(item['kind'], item['index']): item for item in old['scores']['items']}
        new_items = {(item['kind'], item['index']): item for item in new['scores']['items']}
        for identity, item_before in old_items.items():
            item_after = new_items[identity]
            if item_before['correct'] == item_after['correct']:
                continue
            detail = {'case_id': case_id, 'kind': identity[0], 'index': identity[1],
                      'item_key': item_key(case_id, item_before), 'expected': item_before['expected'],
                      'previous_prediction': item_before['predicted'], 'current_prediction': item_after['predicted'],
                      'previous_actual_reason': reason(old, item_before),
                      'current_actual_reason': reason(new, item_after)}
            (fixes if item_after['correct'] else regressions).append(detail)
        cases.append({'id': case_id, 'previous_exact': old['scores']['exact_match'],
                      'current_exact': new['scores']['exact_match'],
                      'previous_wrong_items': sum(not item['correct'] for item in old_items.values()),
                      'current_wrong_items': sum(not item['correct'] for item in new_items.values())})
    deltas = {}
    for name in METRICS:
        old_value = before['summary']['metrics'][name]['value']
        new_value = after['summary']['metrics'][name]['value']
        multiplier = 100 if name in METRICS[:3] else 1
        deltas[name] = {'previous': old_value, 'current': new_value,
                        'delta': (new_value - old_value) * multiplier
                        if old_value is not None and new_value is not None else None,
                        'unit': 'percentage_points' if multiplier == 100 else
                                'seconds' if name == 'p95_latency_seconds' else 'reported_tokens'}
    return {
        'previous_variant': previous_name, 'current_variant': current_name, 'metric_deltas': deltas,
        'fixed_case_ids': [c['id'] for c in cases if not c['previous_exact'] and c['current_exact']],
        'regressed_case_ids': [c['id'] for c in cases if c['previous_exact'] and not c['current_exact']],
        'unchanged_correct_case_ids': [c['id'] for c in cases if c['previous_exact'] and c['current_exact']],
        'unchanged_failing_case_ids': [c['id'] for c in cases if not c['previous_exact'] and not c['current_exact']],
        'item_fixes': fixes, 'item_regressions': regressions, 'all_case_pairs': cases,
    }


def load_inputs(root):
    dataset_path = root / 'evaluation/week4-cases.json'
    dataset = read_json(dataset_path)
    if dataset['version'] != 2:
        raise ValueError('The final report requires the corrected dataset version 2.')
    ids, dataset_sha = [case['id'] for case in dataset['cases']], digest(dataset_path)
    reports, paths = {}, {}
    for variant in VARIANTS:
        path = root / f'evaluation/week4-{variant}/report.json'
        if not path.is_file():
            raise ValueError(f'Wait for all {len(VARIANTS)} final runs: missing {path.relative_to(root)}.')
        report = read_json(path)
        manifest = report['manifest']
        if manifest['dataset_sha256'] != dataset_sha or manifest['dataset_version'] != 2:
            raise ValueError(f'{variant}: dataset hash/version does not match the frozen v2 inputs.')
        if not manifest['live'] or manifest['prompt_version'] != variant:
            raise ValueError(f'{variant}: expected its completed live prompt variant.')
        if manifest['case_ids'] != ids or [row['id'] for row in report['rows']] != ids:
            raise ValueError(f'{variant}: scheduled case IDs differ.')
        if report['summary']['not_run_cases'] or any(row['status'] == 'not_run' for row in report['rows']):
            raise ValueError(f'{variant}: some scheduled cases have not run.')
        if digest(path.parent / 'dataset.json') != dataset_sha:
            raise ValueError(f'{variant}: saved dataset bytes differ.')
        if digest(path.parent / 'prompt.txt') != manifest['prompt_sha256']:
            raise ValueError(f'{variant}: saved prompt does not match its manifest hash.')
        reports[variant], paths[variant] = report, path
    if len({report['manifest']['model'] for report in reports.values()}) != 1:
        raise ValueError('All runs must use the same configured model.')
    hypotheses = read_json(root / 'evaluation/week4-hypotheses.json')
    if hypotheses['dataset_sha256'] != dataset_sha:
        raise ValueError('Hypotheses must identify the same corrected dataset.')
    return dataset, reports, paths, hypotheses, dataset_sha


def build_analysis(root):
    dataset, reports, paths, hypotheses, dataset_sha = load_inputs(root)
    runs = {variant: analyze_run(variant, reports[variant], paths[variant]) for variant in VARIANTS}
    eligible = [variant for variant in VARIANTS[1:]
                if reports[variant]['summary']['metrics']['unsupported_rejection']['value'] >= .95
                and reports[variant]['summary']['metrics']['supported_retention']['value'] >= .9]
    selected = min(eligible, key=lambda variant: (
        -reports[variant]['summary']['metrics']['case_exact_match']['value'],
        reports[variant]['summary']['metrics']['p95_latency_seconds']['value'],
        VARIANTS.index(variant))) if eligible else None
    changes = []
    if len(hypotheses['changes']) != len(VARIANTS) - 1:
        raise ValueError('Every experimental variant requires a registered hypothesis.')
    for previous, variant, hypothesis in zip(VARIANTS[:-1], VARIANTS[1:], hypotheses['changes'], strict=True):
        if hypothesis['variant'] != variant:
            raise ValueError('Hypotheses must follow the cumulative experimental-stage order.')
        changes.append({'hypothesis': hypothesis, 'observed_incremental': paired(previous, variant, reports)})
    return {
        'analysis_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'label_status': dataset['label_status'], 'dataset_version': dataset['version'],
        'dataset_sha256': dataset_sha, 'model': reports['baseline']['manifest']['model'],
        'provenance': {'runner_path': 'evaluation/week4.py', 'runner_sha256': digest(root / 'evaluation/week4.py'),
                       'reviewer_path': 'triage_desk/agent_review.py',
                       'reviewer_sha256': digest(root / 'triage_desk/agent_review.py'),
                       'analysis_builder_sha256': digest(Path(__file__)),
                       'hypotheses_sha256': digest(root / 'evaluation/week4-hypotheses.json'),
                       'hash_note': 'Runner/reviewer file hashes record the source present at analysis generation; '
                                    'run manifests independently pin dataset/model/prompt and saved input bytes.'},
        'selection': {'eligible_variants': eligible, 'selected_variant': selected,
                       'rule': 'Among experimental variants with unsupported rejection >=95% and supported retention >=90%, '
                               'maximize exact case agreement; break ties by lower p95 latency, then stage order.',
                       'promotion_status': 'Experimental candidate only; production default remains baseline pending human review.'},
        'runs': runs, 'incremental_experiments': changes,
        'baseline_to_candidate': paired('baseline', selected, reports) if selected else None,
        'human_time_assumption': TIME_ASSUMPTION,
        'classification_note': 'Operational error-direction buckets, not verified causal diagnoses. '
                               'Every wrong item is included; a case can occur in more than one category.',
        'dataset_revision': hypotheses['dataset_revision_note'],
        'excluded_artifacts': ['evaluation/week4-v1-baseline', 'evaluation/week4-v1-diagnostic'],
        'limitations': [
            'Exact quality metrics are agreement with AI-proposed labels pending human review, not human accuracy.',
            'One real model call per case per stage; no repeat-run uncertainty estimate. Serving nondeterminism remains possible.',
            'Cumulative prompt development used the same development set; incremental deltas do not isolate causal effects.',
            'The first three hypotheses were registered from the v1 baseline before the corrected v2 runs; '
            'the calibration hypothesis was registered after authority and before its own run.',
            'Component-only evaluation: no full-investigator or observed business-outcome measurement.',
            'Dollar cost remains unknown without provider rates; hypothetical minute weights are not observed cost or time saved.',
            'Human label review and an independently labeled held-out set are required before promotion/generalization claims.',
        ],
    }


def number(value, percentage=False):
    return 'unknown' if value is None else f'{value * 100:.2f}%' if percentage else f'{value:,.2f}'


def signed(value, suffix=''):
    return 'unknown' if value is None else f'{value:+.2f}{suffix}'


def names(values):
    return ', '.join(values) if values else 'none'


def counted(value, noun):
    return f"{value} {noun}{'' if value == 1 else 's'}"


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render_results(analysis):
    runs, selected = analysis['runs'], analysis['selection']['selected_variant']
    baseline, candidate = runs['baseline'], runs.get(selected)
    lines = [START, '',
        f'All {len(VARIANTS)} stages below are real calls to the configured production evidence-review component on the same corrected '
        '40-case version 2 dataset. The model is `' + analysis['model'] + '`. Reference labels remain **AI-proposed and pending human review**; '
        'these scores measure agreement with those labels, not human-validated accuracy.', '',
        '| Stage | Exact ≥90% | Reject unsupported ≥95% | Retain supported ≥90% | p95 ≤30 s | Mean tokens ≤4,000 | FP / FN |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for variant in VARIANTS:
        run = runs[variant]
        cells = []
        for metric in METRICS:
            result = run['summary']['metrics'][metric]
            cells.append(f"{number(result['value'], metric in METRICS[:3])} ({result['status']})")
        lines.append('| ' + ' | '.join([variant, *cells,
                                       f"{run['summary']['false_positives']} / {run['summary']['false_negatives']}"]) + ' |')
    completion = '; '.join(f"{variant}: {runs[variant]['summary']['completed_cases']}/40 completed, "
                           f"{counted(runs[variant]['summary']['error_cases'], 'error case')}, "
                           f"{runs[variant]['summary']['unknown_verdicts']} unknown items" for variant in VARIANTS)
    lines.extend(['', completion + '. All scheduled cases/items remain in the score denominators.', ''])
    error_cases = {(variant, item['case_id'], item['error'].get('code', 'execution_failed'))
                   for variant, run in runs.items() for item in run['wrong_items'] if item.get('error')}
    if error_cases:
        lines.extend(['Recorded execution/validation errors: ' + '; '.join(
            f'{variant} {case_id}: `{code}`' for variant, case_id, code in sorted(error_cases)) + '. '
            'Their item verdicts remain unknown and count against the scheduled agreement, rejection, and retention denominators. '
            'A successful call in another stage does not rewrite these preserved failures.', ''])
    if selected:
        pair = analysis['baseline_to_candidate']
        values = pair['metric_deltas']
        lines.extend([
            f"**Selected experimental candidate: {selected}.** Selection maximizes exact case agreement among variants with unsupported rejection "
            'at least 95% and supported retention at least 90%; lower p95 latency breaks ties. '
            f"All five pass bars met: {'yes' if candidate['summary']['all_pass_bars_met'] else 'no'}. The production default remains baseline pending review.", '',
            f"Against baseline, exact agreement changes {signed(values['case_exact_match']['delta'], ' percentage points')}, "
            f"unsupported rejection {signed(values['unsupported_rejection']['delta'], ' pp')}, supported retention "
            f"{signed(values['supported_retention']['delta'], ' pp')}, p95 latency {signed(values['p95_latency_seconds']['delta'], ' s')}, "
            f"and mean reported tokens {signed(values['mean_total_tokens']['delta'])}. "
            f"Case fixes: {names(pair['fixed_case_ids'])}. Case regressions: {names(pair['regressed_case_ids'])}. "
            f"Item fixes: {len(pair['item_fixes'])}; item regressions: {len(pair['item_regressions'])}; "
            'a case fix requires every item in that case to agree.', '',
        ])
    else:
        lines.extend(['**No experimental candidate meets both safety/retention eligibility bars.** Keep the baseline and investigate before promotion.', ''])
    lines.extend([f'### {len(VARIANTS) - 1} hypotheses: predicted and measured', '',
        'The first three predictions came from the preserved v1 baseline before the corrected v2 baseline and experiments. '
        'The fourth calibration hypothesis was registered after the authority result and before its own run. '
        'Measured deltas below compare each cumulative stage with the immediately preceding stage.', '',
        '| Addition and target | Predicted exact Δ | Measured exact Δ | Predicted target-rate Δ | Measured target-rate Δ | Paired case fixes / regressions |',
        '| --- | --- | --- | --- | --- | --- |'])
    for experiment in analysis['incremental_experiments']:
        hypothesis, pair = experiment['hypothesis'], experiment['observed_incremental']
        rate = 'supported_retention' if 'predicted_supported_retention_delta_pp' in hypothesis else 'unsupported_rejection'
        predicted_rate = hypothesis[f'predicted_{rate}_delta_pp']
        lines.append('| ' + ' | '.join([
            f"{hypothesis['variant']}: {hypothesis['target']}",
            signed(hypothesis['predicted_case_agreement_delta_pp'], ' pp'),
            signed(pair['metric_deltas']['case_exact_match']['delta'], ' pp'),
            signed(predicted_rate, ' pp') + ' ' + rate.replace('_', ' '),
            signed(pair['metric_deltas'][rate]['delta'], ' pp'),
            names(pair['fixed_case_ids']) + ' / ' + names(pair['regressed_case_ids']),
        ]) + ' |')
    lines.extend(['', 'Every incremental metric delta, item fix, item regression, and actual before/after model reason is preserved in '
                  '[week4-analysis.json](../evaluation/week4-analysis.json). Negative deltas remain visible; a cumulative addition can lose ground. '
                  'The variants are cumulative and each stage makes one real model call per case. Serving nondeterminism and repeated use of this '
                  'development set prevent isolated causal attribution or a generalization claim.', '',
                  '### Baseline failure roots and explicit time assumption', '',
                  f"The baseline has {baseline['failure_item_count']} wrong items across {baseline['failure_case_count']} unique cases. "
                  'These operational buckets classify error direction rather than proving a model-internal cause. Cases can appear in several buckets.', '',
                  '| Root bucket | Wrong items | Affected cases | Case IDs | Hypothetical minutes | Actual model reason example |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for cluster in baseline['failure_clusters']:
        example = cluster['example_actual_reasons'][0] if cluster['example_actual_reasons'] else None
        lines.append('| ' + ' | '.join([
            ROOT_LABELS[cluster['category']], str(cluster['item_count']), str(cluster['case_count']), names(cluster['case_ids']),
            str(cluster['hypothetical_minutes']) if cluster['hypothetical_minutes'] is not None else 'unpriced',
            escape(f"{example['item_key']}: {example['reason']}") if example else 'none',
        ]) + ' |')
    lines.extend(['', '**Hypothetical assumption only:** rejecting a supported finding/proposal costs 5 minutes of triage effort; '
                  'retaining an unsupported item costs 30 minutes. These are illustrative weights, not observed business cost or measured time saved. '
                  'Unknown/error items stay unpriced. The baseline weights are 30 + 10 + 60 = 100 minutes. '
                  + (f"The {selected} candidate totals {candidate['hypothetical_known_error_minutes']} hypothetical minutes "
                     f"(baseline minus candidate: {baseline['hypothetical_known_error_minutes'] - candidate['hypothetical_known_error_minutes']:+d} minutes under the same assumption). "
                     if candidate else '') + 'Dollar cost is unknown because provider billing rates were not supplied.', ''])
    lines.extend(['The paired comparison artifact separately uses a flat five-minute-per-case illustration. '
                  'That case weight is distinct from these item-based 5/30-minute weights; do not combine the two estimates.', ''])
    if candidate:
        lines.extend(['### Remaining candidate disagreements', '',
                      f"The candidate has {counted(candidate['failure_item_count'], 'item disagreement')} across "
                      f"{counted(candidate['failure_case_count'], 'case')}. "
                      'Each item below still needs semantic review against its frozen quote.', '',
                      '| Case/item | Bucket | Expected → actual | Actual model reason |',
                      '| --- | --- | --- | --- |'])
        for item in candidate['wrong_items']:
            lines.append('| ' + ' | '.join([item['item_key'], ROOT_LABELS[item['category']],
                                            f"{item['expected']} → {item['predicted']}", escape(item['actual_reason'])]) + ' |')
        lines.append('')
    lines.extend(['### Provenance and deliverables', '',
                  f"Dataset SHA-256: `{analysis['dataset_sha256']}`. Runner SHA-256 at analysis generation: "
                  f"`{analysis['provenance']['runner_sha256']}`. Every run also preserves its prompt text/hash, "
                  'dataset bytes, case-local root and model events, expected/predicted verdicts, error state, returned token usage, and latency. '
                  'These are local execution records; no LangSmith project is claimed.', '',
                  'Version 1 runs remain preserved under `evaluation/week4-v1-*` and are excluded from the final deltas. '
                  'Version 2 extends the W4-07/30/38 source spans and tightens W4-10/16/21/29 wording without changing boolean labels. '
                  'Reviewer source titles are neutral so evaluation-authored scenario titles cannot cue the verdict.', '',
                  'Recorded artifacts: ' + ', '.join(f'[{variant}](../evaluation/week4-{variant}/report.html)' for variant in VARIANTS) + ', '
                  '[paired comparison](../evaluation/week4-comparison/comparison.html), '
                  '[machine-readable analysis](../evaluation/week4-analysis.json), '
                  '[editable workbook](../outputs/week4/Engineering%20Issue%20Desk%20Week%204%20Evaluation.xlsx), '
                  '[demo guide](WEEK4_DEMO.md), and [submission packager](../scripts/package_submission.py).', '',
                  'Next, complete human review of all 119 item labels in the workbook, resolve disagreements, and evaluate the selected prompt on '
                  'a separately authored held-out set with repeated calls. Keep production promotion pending that review. '
                  'This component study does not establish full-investigator usefulness or observed engineering time saved.', '', END])
    return '\n'.join(lines)


def replace_results(document, block):
    if START in document or END in document:
        if document.count(START) != 1 or document.count(END) != 1:
            raise ValueError('Expected one complete generated results block.')
        before, remainder = document.split(START, 1)
        _, after = remainder.split(END, 1)
        return before + block + after
    heading, following = '## Read the recorded results\n', '\n## Reproduce the evaluation'
    if heading not in document or following not in document:
        raise ValueError('Existing report is missing its recorded-results/reproduction boundaries.')
    before, remainder = document.split(heading, 1)
    _, after = remainder.split(following, 1)
    return before + heading + '\n' + block + '\n' + following + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate and report the selected candidate without writing artifacts.')
    args = parser.parse_args()
    try:
        analysis = build_analysis(ROOT)
        docs_path = ROOT / 'docs/WEEK4_EVALUATION.md'
        updated = replace_results(docs_path.read_text(encoding='utf-8'), render_results(analysis))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f'Week 4 report not written: {exc}\n')
    if not args.check:
        (ROOT / 'evaluation/week4-analysis.json').write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        docs_path.write_text(updated, encoding='utf-8')
    print(json.dumps({'selected_variant': analysis['selection']['selected_variant'],
                      'eligible_variants': analysis['selection']['eligible_variants'],
                      'writes_performed': not args.check}, indent=2))


if __name__ == '__main__':
    main()
