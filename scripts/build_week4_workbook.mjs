/**
 * Build the Week 4 evaluation workbook from frozen live reports.
 * Uses the Codex primary runtime's @oai/artifact-tool, not repo dependencies.
 * Run through a conversation-specific symlink with --preserve-symlinks-main.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const root = process.cwd();
const args = new Map(process.argv.slice(2).map(arg => {
  const split = arg.indexOf('=');
  return split < 0 ? [arg.replace(/^--/, ''), true] : [arg.slice(2, split), arg.slice(split + 1)];
}));
const variants = ['baseline', 'diagnostic', 'scope', 'authority', 'calibrated'];
const selected = args.get('selected') || 'authority';
assert(variants.slice(1).includes(selected), 'Select diagnostic, scope, authority, or calibrated.');
const outputDir = path.resolve(root, args.get('output-dir') || 'outputs/week4');
const qaDir = path.resolve(root, '.cache/week4-workbook/qa');
const datasetPath = path.resolve(root, args.get('cases') || 'evaluation/week4-cases.json');
const datasetRaw = await fs.readFile(datasetPath);
const dataset = JSON.parse(datasetRaw);
const datasetHash = crypto.createHash('sha256').update(datasetRaw).digest('hex');
const reports = {};
const reportPaths = {};
for (const variant of variants) {
  reportPaths[variant] = args.get(variant) || `evaluation/week4-${variant}/report.json`;
  reports[variant] = JSON.parse(await fs.readFile(path.resolve(root, reportPaths[variant]), 'utf8'));
  const report = reports[variant];
  assert.equal(report.manifest.dataset_sha256, datasetHash, `${variant}: dataset changed`);
  assert.equal(report.manifest.live, true, `${variant}: no live evaluation`);
  assert.equal(report.rows.length, dataset.cases.length, `${variant}: incomplete scheduled population`);
  assert.equal(report.summary.not_run_cases, 0, `${variant}: cases not run`);
  assert.deepEqual(report.rows.map(r => r.id), dataset.cases.map(c => c.id), `${variant}: case identity/order`);
}
assert.equal(new Set(variants.map(v => reports[v].manifest.model)).size, 1, 'Model must match across variants.');
const baseById = new Map(reports.baseline.rows.map(row => [row.id, row]));
const improvedById = new Map(reports[selected].rows.map(row => [row.id, row]));

const wb = Workbook.create();
const summary = wb.worksheets.add('Summary');
const casesSheet = wb.worksheets.add('Cases');
const itemsSheet = wb.worksheets.add('Item review');
const navy = '#20354C', light = '#EEF2F6', ink = '#243246', amber = '#FFF1CC', red = '#FCE4E4';
const sFirst = 7, sLast = sFirst + dataset.cases.length - 1;
const itemCount = dataset.cases.reduce((n, c) => n + c.findings.length + c.next_steps.length, 0);
const iFirst = 7, iLast = iFirst + itemCount - 1;
const cRange = col => `'Cases'!$${col}$${sFirst}:$${col}$${sLast}`;
const iRange = col => `'Item review'!$${col}$${iFirst}:$${col}$${iLast}`;
function literal(value) { return typeof value === 'string' && value.startsWith('=') ? "'" + value : value; }
function cell(sheet, ref, value) { sheet.getRange(ref).values = [[literal(value)]]; }
function formula(sheet, ref, value) { sheet.getRange(ref).formulas = [[value]]; }
function fmt(sheet, ref, attrs) { sheet.getRange(ref).format = attrs; }
function header(sheet, range) {
  fmt(sheet, range, { fill: navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true, rowHeight: 36,
    borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } } });
}
function base(sheet, end) {
  sheet.showGridLines = false;
  fmt(sheet, `A1:${end}`, { font: { name: 'Arial', size: 10, color: ink }, verticalAlignment: 'center', rowHeight: 24 });
  fmt(sheet, 'A2', { font: { name: 'Arial', size: 14, bold: true, color: ink }, rowHeight: 28 });
}
function passCues(sheet, range) {
  sheet.getRange(range).conditionalFormats.add('containsText', { text: 'FAIL', format: { fill: red, font: { bold: true, color: '#9C2525' } } });
  sheet.getRange(range).conditionalFormats.add('containsText', { text: 'Pending', format: { fill: amber } });
}
function verdict(value) { return value === true ? 'Supported' : value === false ? 'Unsupported' : 'Unknown'; }
function reason(row, kind, index) {
  const key = kind === 'findings' ? 'finding_verdicts' : 'next_step_verdicts';
  return row.quality_review?.[key]?.find(v => v.index === index)?.reason || row.error?.message || 'No verdict returned.';
}
function failureFormula(label, prediction) {
  return `=IF(${prediction}="Unknown","Unknown verdict",IF(${prediction}=${label},"None",IF(${prediction}="Supported","Unsupported retained","Supported removed")))`;
}
function token(row) { return row.accounting?.usage_complete && row.accounting?.model_calls ? row.accounting.reported_token_usage.total_tokens : null; }
const sourceRef = c => `${c.source.path}:${c.source.start_line}-${c.source.end_line}`;

base(itemsSheet, `T${iLast}`);
itemsSheet.tabColor = '#D7C88E';
cell(itemsSheet, 'A2', 'Item verdicts and human labels');
cell(itemsSheet, 'A3', `Baseline versus ${selected}. AI-proposed labels are pending human review.`);
cell(itemsSheet, 'A4', 'Complete the amber Human label, Reviewer, and Human rationale fields for each item. Full source excerpts are on Cases.');
fmt(itemsSheet, 'A3:A4', { font: { name: 'Arial', size: 10, italic: true, color: '#53667C' } });
itemsSheet.getRange('A6:T6').values = [[
  'Case ID', 'Item', 'Proposed finding or next step', 'AI-proposed label', 'AI-proposed rationale',
  'Baseline verdict', 'Baseline reason', `${selected} verdict`, `${selected} reason`,
  'Baseline agreement', `${selected} agreement`, 'Baseline failure', `${selected} failure`,
  'Human label', 'Reviewer', 'Human rationale', 'Review status', 'Human baseline agreement', 'Human improved agreement', 'Source citation',
]];
header(itemsSheet, 'A6:T6');
let itemRow = iFirst;
const itemSpans = new Map();
for (const c of dataset.cases) {
  const first = itemRow;
  for (const kind of ['findings', 'next_steps']) for (let index = 0; index < c[kind].length; index++) {
    const item = c[kind][index], b = baseById.get(c.id), improved = improvedById.get(c.id);
    const bPrediction = b.scores.predicted[kind][index], aPrediction = improved.scores.predicted[kind][index];
    const r = itemRow++;
    itemsSheet.getRange(`A${r}:T${r}`).values = [[c.id, `${kind === 'findings' ? 'F' : 'N'}${index + 1}`, item.statement || item.proposal,
      verdict(item.expected_supported), item.rationale, verdict(bPrediction), reason(b, kind, index), verdict(aPrediction), reason(improved, kind, index),
      null, null, null, null, null, null, null, null, null, null, sourceRef(c)].map(literal)];
    formula(itemsSheet, `J${r}`, `=IF(AND(F${r}<>"Unknown",F${r}=D${r}),"PASS","FAIL")`);
    formula(itemsSheet, `K${r}`, `=IF(AND(H${r}<>"Unknown",H${r}=D${r}),"PASS","FAIL")`);
    formula(itemsSheet, `L${r}`, failureFormula(`D${r}`, `F${r}`));
    formula(itemsSheet, `M${r}`, failureFormula(`D${r}`, `H${r}`));
    formula(itemsSheet, `Q${r}`, `=IF(AND(OR(N${r}="Supported",N${r}="Unsupported"),LEN(TRIM(O${r}))>0,LEN(TRIM(P${r}))>0),"Reviewed","Pending")`);
    formula(itemsSheet, `R${r}`, `=IF(Q${r}<>"Reviewed","Pending",IF(AND(F${r}<>"Unknown",F${r}=N${r}),"PASS","FAIL"))`);
    formula(itemsSheet, `S${r}`, `=IF(Q${r}<>"Reviewed","Pending",IF(AND(H${r}<>"Unknown",H${r}=N${r}),"PASS","FAIL"))`);
    const lines=Math.max(Math.ceil((item.statement || item.proposal).length/55),Math.ceil(item.rationale.length/55),
      Math.ceil(reason(b,kind,index).length/45),Math.ceil(reason(improved,kind,index).length/45));
    fmt(itemsSheet, `A${r}:T${r}`, { rowHeight: Math.max(48,lines*12+12), wrapText: true, verticalAlignment: 'top', fill: r % 2 ? '#FFFFFF' : '#F6F8FA' });
  }
  itemSpans.set(c.id, [first, itemRow - 1]);
}
const itemWidths = { A: 12, B: 8, C: 55, D: 18, E: 55, F: 18, G: 45, H: 18, I: 45, J: 16, K: 16, L: 22, M: 22, N: 18, O: 22, P: 50, Q: 14, R: 20, S: 20, T: 52 };
for (const [col, width] of Object.entries(itemWidths)) fmt(itemsSheet, `${col}6:${col}${iLast}`, { columnWidth: width });
fmt(itemsSheet, `N${iFirst}:P${iLast}`, { fill: amber, font: { name: 'Arial', size: 10, color: '#8B5700' } });
itemsSheet.getRange(`N${iFirst}:N${iLast}`).dataValidation = { rule: { type: 'list', values: ['Supported', 'Unsupported'] } };
passCues(itemsSheet, `J${iFirst}:M${iLast}`);
passCues(itemsSheet, `Q${iFirst}:S${iLast}`);
itemsSheet.tables.add(`A6:T${iLast}`, true, 'Week4ItemReview');
itemsSheet.freezePanes.freezeRows(6);
itemsSheet.freezePanes.freezeColumns(2);

base(casesSheet, `AC${sLast}`);
casesSheet.tabColor = '#72849B';
cell(casesSheet, 'A2', 'Case results and frozen inputs');
cell(casesSheet, 'A3', 'PASS means every proposed item agrees with the AI-proposed reference. Errors and unknown verdicts fail. Human scores remain pending.');
cell(casesSheet, 'A4', `Selected improvement: ${selected}. The frozen question, source excerpt, and expected behavior appear to the right of the results.`);
fmt(casesSheet, 'A3:A4', { font: { name: 'Arial', size: 10, italic: true, color: '#53667C' } });
casesSheet.getRange('A6:AC6').values = [[
  'Case ID', 'Scenario', 'Case title', 'Baseline', selected, 'Change', 'Baseline FP', 'Improved FP', 'Baseline FN', 'Improved FN',
  'Baseline unknown', 'Improved unknown', 'Human review', 'Human baseline', 'Human improved', 'Baseline seconds', 'Improved seconds',
  'Baseline tokens', 'Improved tokens', 'Baseline status', 'Improved status', 'Failure mode seeded',
  'Input question', 'Reported version', 'Source citation', 'Frozen source excerpt', 'Expected behavior', 'Source file SHA-256', 'Dataset source',
]];
header(casesSheet, 'A6:AC6');
dataset.cases.forEach((c, index) => {
  const r = sFirst + index, [first, last] = itemSpans.get(c.id), b = baseById.get(c.id), improved = improvedById.get(c.id);
  const expectedCount=last-first+1;
  const countFor = (col,value) => `COUNTIFS(${iRange('A')},A${r},${iRange(col)},"${value}")`;
  const expected = ['Findings:', ...c.findings.map((x,i)=>`F${i+1}: ${verdict(x.expected_supported)}. ${x.statement}`),
    'Next steps:', ...c.next_steps.map((x,i)=>`N${i+1}: ${verdict(x.expected_supported)}. ${x.proposal}`)].join('\n');
  casesSheet.getRange(`A${r}:AC${r}`).values = [[c.id, c.scenario, c.title, null, null, null, null, null, null, null, null, null, null, null, null,
    b.latency_seconds, improved.latency_seconds, token(b), token(improved), b.status, improved.status, c.failure_mode,
    c.context.question,c.context.reported_version || 'Not specified',sourceRef(c),c.source.quote,expected,c.source.sha256,'evaluation/week4-cases.json'].map(literal)];
  formula(casesSheet, `D${r}`, `=IF(AND(COUNTIFS(${iRange('A')},A${r})=${expectedCount},${countFor('J','FAIL')}=0),"PASS","FAIL")`);
  formula(casesSheet, `E${r}`, `=IF(AND(COUNTIFS(${iRange('A')},A${r})=${expectedCount},${countFor('K','FAIL')}=0),"PASS","FAIL")`);
  formula(casesSheet, `F${r}`, `=IF(D${r}=E${r},IF(D${r}="PASS","Unchanged pass","Unchanged failure"),IF(E${r}="PASS","Fixed","Regression"))`);
  for (const [target, source, value] of [['G','L','Unsupported retained'],['H','M','Unsupported retained'],['I','L','Supported removed'],['J','M','Supported removed'],['K','L','Unknown verdict'],['L','M','Unknown verdict']])
    formula(casesSheet, `${target}${r}`, `=${countFor(source,value)}`);
  formula(casesSheet, `M${r}`, `=IF(${countFor('Q','Reviewed')}=${expectedCount},"Reviewed","Pending")`);
  formula(casesSheet, `N${r}`, `=IF(M${r}<>"Reviewed","Pending",IF(${countFor('R','FAIL')}=0,"PASS","FAIL"))`);
  formula(casesSheet, `O${r}`, `=IF(M${r}<>"Reviewed","Pending",IF(${countFor('S','FAIL')}=0,"PASS","FAIL"))`);
  const displayedLines = c.source.quote.split('\n').reduce((n,line)=>n+Math.max(1,Math.ceil(line.length/115)),0);
  fmt(casesSheet, `A${r}:AC${r}`, { rowHeight: Math.min(409,Math.max(156,displayedLines*12+12)), wrapText: true, verticalAlignment: 'top', fill: r % 2 ? '#FFFFFF' : '#F6F8FA' });
});
casesSheet.tables.add(`A6:AC${sLast}`, true, 'Week4CaseResults');
passCues(casesSheet, `D${sFirst}:O${sLast}`);
casesSheet.getRange(`F${sFirst}:F${sLast}`).conditionalFormats.add('containsText', { text: 'Regression', format: { fill: red, font: { bold: true, color: '#9C2525' } } });
fmt(casesSheet, `P${sFirst}:Q${sLast}`, { numberFormat: '0.00' });
fmt(casesSheet, `R${sFirst}:S${sLast}`, { numberFormat: '#,##0' });
const caseWidths = { A: 12, B: 17, C: 42, D: 15, E: 15, F: 22, G: 14, H: 14, I: 14, J: 14, K: 14, L: 14, M: 14, N: 16, O: 16, P: 16, Q: 16, R: 16, S: 16, T: 16, U: 16, V: 60, W: 55, X: 18, Y: 55, Z: 145, AA: 95, AB: 70, AC: 40 };
for (const [col,width] of Object.entries(caseWidths)) fmt(casesSheet, `${col}6:${col}${sLast}`, { columnWidth: width });
casesSheet.freezePanes.freezeRows(6);
casesSheet.freezePanes.freezeColumns(1);

base(summary, 'I55');
summary.tabColor = navy;
cell(summary, 'A2', 'Engineering Issue Desk Week 4 evaluation');
cell(summary, 'A3', 'AI-proposed labels pending human review. Scores measure agreement with these labels, not human-validated accuracy.');
cell(summary, 'A4', `Production evidence reviewer only. Same ${dataset.cases.length} frozen cases in every run. Selected improvement: ${selected}.`);
fmt(summary, 'A3:A4', { font: { name: 'Arial', size: 10, italic: true, color: '#53667C' } });
summary.getRange('A6:I6').values = [['Metric', 'Pass bar', 'Baseline', 'Diagnostic', 'Scope', 'Authority', 'Calibrated', 'Selected delta', 'Selected status']];
header(summary, 'A6:I6');
const metricDefs = [
  ['case_exact_match','Exact case verdict agreement',.90,'>=90%',true],
  ['unsupported_rejection','Unsupported item rejection',.95,'>=95%',true],
  ['supported_retention','Supported item retention',.90,'>=90%',true],
  ['p95_latency_seconds','p95 latency (seconds)',30,'<=30 sec',false],
  ['mean_total_tokens','Mean tokens per case',4000,'<=4,000',false],
];
for (let i=0;i<metricDefs.length;i++) {
  const [key,label,target,bar,higher] = metricDefs[i],r=7+i;
  cell(summary,`A${r}`,label); cell(summary,`B${r}`,target);
  for (let v=0;v<variants.length;v++) {
    assert.equal(reports[variants[v]].summary.metrics[key].threshold,target,`${variants[v]} ${key}: threshold mismatch`);
    cell(summary,`${String.fromCharCode(67+v)}${r}`,reports[variants[v]].summary.metrics[key].value);
  }
  const selectedCol=String.fromCharCode(67+variants.indexOf(selected));
  formula(summary,`H${r}`,`=IF(OR(${selectedCol}${r}="",C${r}=""),"Not measured",(${selectedCol}${r}-C${r})${i<3?'*100':''})`);
  formula(summary,`I${r}`,`=IF(${selectedCol}${r}="","Not measured",IF(${selectedCol}${r}${higher?'>=':'<='}B${r},"PASS","FAIL"))`);
  fmt(summary,`B${r}:G${r}`,{numberFormat:i<3?'0.0%':i===3?'0.00':'#,##0'});
  fmt(summary,`H${r}`,{numberFormat:i<3?'0.0" pp"':i===3?'0.00':'#,##0'});
}
cell(summary, 'A12', 'Quality rates must meet or exceed the target. Latency and tokens must remain at or below the target.');
cell(summary, 'A13', 'Run metrics are measured snapshots. Delta = selected minus baseline. Rate deltas are percentage points. p95 uses nearest rank.');
cell(summary, 'A14', 'All scheduled cases and items count. Missing verdicts fail correctness. Token prices were not supplied, so monetary cost is unknown.');
fmt(summary,'A12:A14',{font:{name:'Arial',size:10,italic:true,color:'#53667C'}});
summary.getRange('A16:H16').values=[['Run','Completed / scheduled','Errors','False positives','False negatives','Unknown verdicts','All bars met','Run ID']];
header(summary,'A16:H16');
variants.forEach((variant,index)=>{
  const r=17+index,report=reports[variant],s=report.summary;
  summary.getRange(`A${r}:H${r}`).values=[[variant,`${s.completed_cases} / ${s.scheduled_cases}`,s.error_cases,s.false_positives,s.false_negatives,s.unknown_verdicts,null,report.manifest.run_id]];
  const col=String.fromCharCode(67+index);
  formula(summary,`G${r}`,`=IF(COUNT(${col}7:${col}11)<5,"Not measured",IF(AND(${col}7>=$B$7,${col}8>=$B$8,${col}9>=$B$9,${col}10<=$B$10,${col}11<=$B$11),"PASS","FAIL"))`);
});
cell(summary,'A23','Human label review'); fmt(summary,'A23',{font:{name:'Arial',size:12,bold:true}});
summary.getRange('A25:D25').values=[['Items reviewed','Items pending','Cases fully reviewed','Human scores available']];header(summary,'A25:D25');
formula(summary,'A26',`=COUNTIFS(${iRange('Q')},"Reviewed")`);
formula(summary,'B26',`=${itemCount}-A26`);
formula(summary,'C26',`=COUNTIFS(${cRange('M')},"Reviewed")`);
formula(summary,'D26',`=IF(B26=0,"Yes","Pending review")`);
summary.getRange('A28:C28').values=[['Human-reviewed metric','Baseline',selected]];header(summary,'A28:C28');
cell(summary,'A29','Exact case verdict agreement');cell(summary,'A30','Unsupported item rejection');cell(summary,'A31','Supported item retention');
for (const [col,caseCol,prediction] of [['B','N','F'],['C','O','H']]) {
  formula(summary,`${col}29`,`=IF($B$26>0,"Pending review",COUNTIFS(${cRange(caseCol)},"PASS")/${dataset.cases.length})`);
  formula(summary,`${col}30`,`=IF($B$26>0,"Pending review",IF(COUNTIFS(${iRange('N')},"Unsupported")=0,"n.a.",COUNTIFS(${iRange('N')},"Unsupported",${iRange(prediction)},"Unsupported")/COUNTIFS(${iRange('N')},"Unsupported")))`);
  formula(summary,`${col}31`,`=IF($B$26>0,"Pending review",IF(COUNTIFS(${iRange('N')},"Supported")=0,"n.a.",COUNTIFS(${iRange('N')},"Supported",${iRange(prediction)},"Supported")/COUNTIFS(${iRange('N')},"Supported")))`);
}
fmt(summary,'B29:C31',{numberFormat:'0.0%'});
cell(summary,'A33','Rubric: mark an item Supported only when the complete claim or proposed diagnostic follows from the selected evidence and request scope.');
cell(summary,'A34','Unsupported covers wrong causality, omitted prerequisites, overbroad scope, unmeasured runtime claims, and instructions embedded in evidence.');
cell(summary,'A35','Exact case agreement requires every finding and next-step verdict to match. F = finding. N = next step. Unknown never equals Unsupported.');
fmt(summary,'A33:A35',{font:{name:'Arial',size:10,italic:true,color:'#53667C'}});
cell(summary,'A38','Failure impact');fmt(summary,'A38',{font:{name:'Arial',size:12,bold:true}});
summary.getRange('A40:G40').values=[['Failure category','Baseline items','Improved items','Baseline cases','Improved cases','Baseline review minutes','Improved review minutes']];header(summary,'A40:G40');
cell(summary,'H39','Minutes per affected case');cell(summary,'H40',5);fmt(summary,'H40',{fill:amber});
const clusters=[['Unsupported retained','G','H'],['Supported removed','I','J'],['Unknown verdict','K','L']];
clusters.forEach(([label,b,a],index)=>{
  const r=41+index;cell(summary,`A${r}`,label);
  formula(summary,`B${r}`,`=SUM(${cRange(b)})`);formula(summary,`C${r}`,`=SUM(${cRange(a)})`);
  formula(summary,`D${r}`,`=COUNTIFS(${cRange(b)},">0")`);formula(summary,`E${r}`,`=COUNTIFS(${cRange(a)},">0")`);
  formula(summary,`F${r}`,`=D${r}*$H$40`);formula(summary,`G${r}`,`=E${r}*$H$40`);
});
cell(summary,'A45','Review minutes are an editable planning assumption, not measured labor or verified savings. Categories can overlap; do not sum affected cases.');
cell(summary,'A47','Scenario mix');summary.getRange('A48:C48').values=[['Scenario','Cases','Share']];header(summary,'A48:C48');
['happy_path','edge','known_failure','adversarial'].forEach((scenario,index)=>{const r=49+index;cell(summary,`A${r}`,scenario);formula(summary,`B${r}`,`=COUNTIFS(${cRange('B')},A${r})`);formula(summary,`C${r}`,`=B${r}/${dataset.cases.length}`);});
fmt(summary,'C49:C52',{numberFormat:'0.0%'});
cell(summary,'E48','Measured report sources');
variants.forEach((v,index)=>cell(summary,`E${49+index}`,`${v}: ${reportPaths[v]}`));
cell(summary,'E54',`Dataset SHA-256: ${datasetHash}`);
for (const [col,width] of Object.entries({A:38,B:18,C:16,D:16,E:18,F:18,G:18,H:28,I:18})) fmt(summary,`${col}1:${col}55`,{columnWidth:width});
fmt(summary,'A7:I11',{rowHeight:28});fmt(summary,'A17:H21',{rowHeight:30,wrapText:true});
fmt(summary,'A40:G43',{wrapText:true,rowHeight:36});
passCues(summary,'I7:I11');passCues(summary,'G17:G21');passCues(summary,'D26');passCues(summary,'B29:C31');

// Verify the human-review prerequisite cannot be satisfied by a label alone.
const testRow=iFirst;
cell(itemsSheet,`N${testRow}`,'Supported');
assert.equal(itemsSheet.getRange(`Q${testRow}`).values[0][0],'Pending');
cell(itemsSheet,`O${testRow}`,'Verification only');cell(itemsSheet,`P${testRow}`,'Verification only');
assert.equal(itemsSheet.getRange(`Q${testRow}`).values[0][0],'Reviewed');
cell(itemsSheet,`N${testRow}`,null);cell(itemsSheet,`O${testRow}`,null);cell(itemsSheet,`P${testRow}`,null);
wb.recalculate();
assert.equal(summary.getRange('A26').values[0][0],0);
assert.equal(summary.getRange('B26').values[0][0],itemCount);
assert.equal(summary.getRange('B29').values[0][0],'Pending review');
assert.equal(casesSheet.getRange(`M${sFirst}`).values[0][0],'Pending');
const baselines=casesSheet.getRange(`D${sFirst}:D${sLast}`).values.flat();
const improvedCases=casesSheet.getRange(`E${sFirst}:E${sLast}`).values.flat();
assert.equal(baselines.filter(v=>v==='PASS').length,reports.baseline.rows.filter(r=>r.scores.exact_match).length);
assert.equal(improvedCases.filter(v=>v==='PASS').length,reports[selected].rows.filter(r=>r.scores.exact_match).length);
for (let index=0;index<dataset.cases.length;index++) {
  const r=sFirst+index,id=dataset.cases[index].id;
  assert.equal(baselines[index],baseById.get(id).scores.exact_match?'PASS':'FAIL',`${id}: baseline formula`);
  assert.equal(improvedCases[index],improvedById.get(id).scores.exact_match?'PASS':'FAIL',`${id}: improved formula`);
  assert.deepEqual(casesSheet.getRange(`G${r}:L${r}`).values[0],[baseById.get(id).scores.false_positives,improvedById.get(id).scores.false_positives,
    baseById.get(id).scores.false_negatives,improvedById.get(id).scores.false_negatives,baseById.get(id).scores.unknown_verdicts,improvedById.get(id).scores.unknown_verdicts]);
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Formula error scan'});
await fs.mkdir(qaDir,{recursive:true});
await fs.writeFile(path.join(qaDir,'formula-scan.ndjson'),errors.ndjson);
await fs.writeFile(path.join(qaDir,'summary-inspection.ndjson'),(await wb.inspect({kind:'table',range:'Summary!A6:I31',include:'values,formulas',tableMaxRows:26,tableMaxCols:9,maxChars:16000})).ndjson);
for (const [name,range,label] of [['Summary','A1:I35','summary'],['Summary','A38:I54','summary-failures'],['Cases','A1:F8','cases-results'],['Cases','Y6:AA7','cases-inputs'],['Cases','Y23:AA23','cases-long-source-w17'],['Cases','Y25:AA25','cases-long-source-w19'],['Cases','Y39:AA39','cases-long-expected-w33'],['Item review','A1:I9','items-verdicts'],['Item review','N6:T10','items-human']]) {
  const image=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(qaDir,`${label}.png`),new Uint8Array(await image.arrayBuffer()));
}
await fs.mkdir(outputDir,{recursive:true});
const output=await SpreadsheetFile.exportXlsx(wb);
const outputPath=path.join(outputDir,'Engineering Issue Desk Week 4 Evaluation.xlsx');
await output.save(outputPath);
await fs.writeFile(path.join(qaDir,'build-metadata.json'),JSON.stringify({output:outputPath,datasetHash,selected,reportPaths,cases:dataset.cases.length,items:itemCount,aiLabelStatus:dataset.label_status,humanReviewedItems:0},null,2)+'\n');
console.log(JSON.stringify({output:outputPath,selected,cases:dataset.cases.length,items:itemCount,qaDir}));
