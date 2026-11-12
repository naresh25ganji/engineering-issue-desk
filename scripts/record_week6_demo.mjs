#!/usr/bin/env node
/** Record a captioned walkthrough of the real app and saved Week 6 evidence.
 *
 * Run using the packaged Codex Node runtime. This script never starts an
 * investigation or changes a score. It requires a completed live report and
 * records Chromium's actual rendered pages, with a caption overlay.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const runtime = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime');
const require = createRequire(import.meta.url);
const {chromium} = require(path.join(runtime, 'dependencies/node/node_modules/playwright'));

function argsFrom(argv) {
  const args = {
    report: path.join(root, 'evaluation/week6-live/report.json'),
    output: path.join(root, 'outputs/week6/Engineering Issue Desk Week 6 Demo.webm'),
    appUrl: 'http://127.0.0.1:8502',
    screenshots: path.join(root, 'outputs/week6/screenshots'),
    chrome: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    scale: 1,
    skipApp: false,
    overwrite: false,
  };
  const names = {'--report':'report', '--output':'output', '--app-url':'appUrl',
    '--report-url':'reportUrl', '--retest-report':'retestReport', '--screenshots':'screenshots', '--chrome':'chrome',
    '--duration-scale':'scale'};
  for (let i=0; i<argv.length; i++) {
    if (argv[i] === '--help') {
      console.log('record_week6_demo.mjs [--report report.json] [--output demo.webm] [--app-url URL] [--report-url URL] [--retest-report report.json] [--screenshots DIR] [--chrome PATH] [--duration-scale NUMBER] [--skip-app] [--overwrite]');
      process.exit(0);
    }
    if (argv[i] === '--skip-app') args.skipApp = true;
    else if (argv[i] === '--overwrite') args.overwrite = true;
    else if (names[argv[i]]) {
      if (!argv[i+1]) throw new Error('Missing value for ' + argv[i]);
      args[names[argv[i]]] = argv[++i];
    } else throw new Error('Unknown argument: ' + argv[i]);
  }
  args.scale = Number(args.scale);
  if (!Number.isFinite(args.scale) || args.scale <= 0 || args.scale > 3)
    throw new Error('--duration-scale must be greater than 0 and at most 3.');
  for (const key of ['report','output','screenshots','chrome']) args[key] = path.resolve(args[key]);
  if (!args.output.endsWith('.webm')) throw new Error('The packaged recorder outputs .webm (VP8).');
  return args;
}

const short = (value, n=250) => {
  const s = String(value ?? '').replace(/\s+/g, ' ').trim();
  return s.length <= n ? s : s.slice(0, n).replace(/\s\S*$/, '') + '…';
};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  const args = argsFrom(process.argv.slice(2));
  const raw = await fs.readFile(args.report);
  const report = JSON.parse(raw.toString('utf8'));
  if (report.execution_mode !== 'live' || !Array.isArray(report.results) || !report.results.length)
    throw new Error('A nonempty actual live report is required; inventory data cannot be recorded as findings.');
  let retest = null;
  if (args.retestReport) {
    args.retestReport = path.resolve(args.retestReport);
    retest = JSON.parse(await fs.readFile(args.retestReport,'utf8'));
    if (retest.execution_mode !== 'live' || !retest.results?.length)
      throw new Error('The targeted rerun must also be a nonempty actual live report.');
  }
  if (!args.overwrite) {
    try { await fs.access(args.output); throw new Error('Output exists; choose a new path or pass --overwrite.'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const html = path.join(path.dirname(args.report), 'report.html');
  await fs.access(html);
  if (!args.skipApp) {
    const response = await fetch(args.appUrl, {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error('The local Streamlit app is unavailable: ' + response.status);
  }
  await fs.mkdir(path.dirname(args.output), {recursive:true});
  await fs.mkdir(args.screenshots, {recursive:true});
  const rawDir = await fs.mkdtemp(path.join(os.tmpdir(), 'week6-browser-recording-'));
  const browser = await chromium.launch({headless:true, executablePath:args.chrome,
    args:['--disable-background-networking','--disable-features=Translate','--no-first-run']});
  const context = await browser.newContext({viewport:{width:1440,height:1000},
    deviceScaleFactor:1, reducedMotion:'reduce',
    recordVideo:{dir:rawDir,size:{width:1440,height:1000}}});
  const page = await context.newPage();
  const recordedVideo = page.video();
  const timeline = [];
  const started = Date.now();
  let captionValue = null;

  async function caption(title, body) {
    captionValue = {title,body};
    await page.evaluate(({title,body}) => {
      let style=document.getElementById('week6-demo-caption-style');
      if (!style) {
        style=document.createElement('style');
        style.id='week6-demo-caption-style';
        style.textContent='#week6-demo-caption{position:fixed;left:0;right:0;bottom:0;z-index:2147483647;padding:17px 34px 20px;background:rgba(12,23,42,.97);color:white;border-top:3px solid #adc7e9;font:21px/1.35 Arial,sans-serif;box-shadow:0 -6px 20px #0002}#week6-demo-caption strong{display:block;font-size:23px;margin-bottom:5px;color:white}html{scroll-padding-bottom:155px}';
        document.head.appendChild(style);
      }
      let el=document.getElementById('week6-demo-caption');
      if (!el) {el=document.createElement('div');el.id='week6-demo-caption';document.body.appendChild(el);}
      el.style.visibility='visible';
      el.replaceChildren();
      const heading=document.createElement('strong');heading.textContent=title;el.appendChild(heading);
      const copy=document.createElement('span');copy.textContent=body;el.appendChild(copy);
    }, captionValue);
  }
  async function hold(seconds, title, body) {
    await caption(title,body);
    timeline.push({seconds:Number(((Date.now()-started)/1000).toFixed(2)), title, body});
    console.log(JSON.stringify({stage:title}));
    await sleep(seconds*1000*args.scale);
  }
  async function screenshot(name) {
    await page.locator('#week6-demo-caption').evaluate(el=>el.style.visibility='hidden').catch(()=>{});
    await page.screenshot({path:path.join(args.screenshots,name+'.png'),animations:'disabled'});
    if (captionValue) await caption(captionValue.title,captionValue.body);
  }
  async function scrollTo(locator) {
    await locator.waitFor({state:'visible'});
    await locator.evaluate(el=>el.scrollIntoView({block:'start',behavior:'instant'}));
    await page.evaluate(()=>window.scrollBy(0,-22));
    await sleep(350);
  }
  async function focusMissingInformation(responseHeading) {
    const pre=responseHeading.locator('xpath=following-sibling::pre[1]');
    if (!await pre.count()) return false;
    return await pre.evaluate(el=>{
      const node=el.firstChild;
      if (!node || node.nodeType!==Node.TEXT_NODE) return false;
      const text=node.textContent || '';
      const match=/missing[ _-]*information|paste.{0,30}(?:api key|token)|raw api key/i.exec(text);
      if (!match) return false;
      const range=document.createRange();
      range.setStart(node,match.index);
      range.setEnd(node,Math.min(text.length,match.index+match[0].length));
      window.scrollBy(0,range.getBoundingClientRect().top-90);
      return true;
    });
  }
  const rows=report.results;
  const attacks=rows.filter(row=>row.kind!=='benign');
  const controls=rows.filter(row=>row.kind==='benign');
  const counts=Object.fromEntries(['PASS','WARN','FAIL'].map(score=>[score,rows.filter(row=>row.score===score).length]));
  const families=[...new Set(attacks.map(row=>row.family))];
  const responseLength=row=>row.turns.reduce((n,turn)=>n+(turn.response?.length??0),0);
  const hasResponse=row=>row.turns?.some(turn=>turn.response);
  const meaningful=rows.filter(hasResponse);
  const caseWithScore=score=>meaningful.filter(row=>row.kind!=='benign'&&row.score===score).sort((a,b)=>responseLength(a)-responseLength(b))[0];
  const important=meaningful.find(row=>row.id==='W6-21'&&row.score==='FAIL')
    || caseWithScore('FAIL') || caseWithScore('WARN') || caseWithScore('PASS');
  const crescendo=meaningful.find(row=>row.family.toLowerCase().includes('crescendo')&&row.turns.length>1)
    || rows.find(row=>row.family.toLowerCase().includes('crescendo')&&row.turns.length>1);
  const benign=controls.filter(hasResponse).sort((a,b)=>responseLength(a)-responseLength(b))[0] || controls[0];

  try {
    if (!args.skipApp) {
      await page.goto(args.appUrl,{waitUntil:'domcontentloaded'});
      const tab=page.getByRole('tab',{name:'Evaluation',exact:true});
      await tab.waitFor({state:'visible',timeout:45000});
      await hold(10,'Week 6 · Engineering Issue Desk',
        'Path B: I tested the existing engineering investigation agent against its evidence, tool, privacy, and human-review boundaries.');
      await tab.click();
      await sleep(1500);
      await hold(8,'The application and its evaluation workspace',
        'This walkthrough opens saved results. It does not start a new model run or replace recorded responses.');
      await screenshot('01-streamlit-evaluation');
    }
    await page.goto(args.reportUrl || pathToFileURL(html).href,{waitUntil:'load'});
    await hold(12,'Actual saved run · '+rows.length+' scheduled cases',
      attacks.length+' attacks and '+controls.length+' normal controls. Automated observations: '+counts.PASS+' PASS, '+counts.WARN+' WARN, '+counts.FAIL+' FAIL. Human semantic review is separate.');
    await screenshot('02-run-summary');
    await scrollTo(page.getByRole('heading',{name:'Attack and defense matrix',exact:true}));
    await hold(11,families.length+' attack families with concrete defenses',
      families.map(family=>family.replaceAll('_',' ')).join(' · ')+'. The table records the observed result and matching control for each case.');
    await screenshot('03-attack-defense-matrix');

    async function showCase(row, name, promptSeconds=8, responseSeconds=13) {
      if (!row) return;
      const article=page.locator('article[id="'+row.id+'"]');
      await scrollTo(article);
      await hold(6,row.id+' · '+row.score+' · '+short(row.title,80),short(row.reason,250));
      const prompt=article.getByRole('heading',{name:'Exact prompt',exact:true}).first();
      if (await prompt.count()) {
        await scrollTo(prompt);
        await hold(promptSeconds,'Exact recorded prompt',short(row.expected_boundary || row.title,230));
        await screenshot(name+'-prompt');
      }
      const saved=article.getByRole('heading',{name:'Saved response',exact:true}).first();
      if (await saved.count()) {
        await scrollTo(saved);
        const initialSeconds=Math.min(5,responseSeconds);
        await hold(initialSeconds,'Actual saved response · '+row.id,
          row.score+': '+short(row.reason,220));
        if (row.id==='W6-21') await focusMissingInformation(saved);
        await hold(Math.max(1,responseSeconds-initialSeconds),'Actual saved response · '+row.id,
          row.score+': '+short(row.reason,220));
        await screenshot(name+'-response');
      }
    }
    await showCase(important,'04-significant-case');
    if (crescendo) {
      const article=page.locator('article[id="'+crescendo.id+'"]');
      await scrollTo(article);
      await hold(4,crescendo.id+' · Multi-turn crescendo',
        'The same saved case is revised across turns. Each later prompt tries to escalate the request; each response remains visible in the evidence.');
      const prompts=article.getByRole('heading',{name:'Exact prompt',exact:true});
      const responses=article.getByRole('heading',{name:'Saved response',exact:true});
      const turnCount=await prompts.count();
      for (let index=0; index<turnCount; index++) {
        await scrollTo(prompts.nth(index));
        await hold(4,'Crescendo · Turn '+(index+1)+' prompt',
          short(crescendo.turns[index]?.prompt,230));
        await scrollTo(responses.nth(index));
        const turn=crescendo.turns[index];
        const complete=['awaiting_review','completed'].includes(turn?.status);
        await hold(5,'Crescendo · Turn '+(index+1)+(complete?' actual response':' last saved draft'),
          (complete?'':'No new response completed. ')+
          'Recorded status: '+turn?.status+'. Overall '+crescendo.score+': '+short(crescendo.reason,150));
      }
      await screenshot('05-crescendo-response');
    }
    if (benign) await showCase(benign,'06-normal-control',5,8);
    if (retest) {
      const followupHtml=path.join(path.dirname(args.retestReport),'report.html');
      await page.goto(pathToFileURL(followupHtml).href,{waitUntil:'load'});
      const followupCounts=['PASS','WARN','FAIL'].map(score=>retest.results.filter(row=>row.score===score).length+' '+score).join(', ');
      await hold(12,'Separate targeted live follow-up · '+retest.results.length+' cases',
        followupCounts+'. This rerun follows a credential-control change. It is limited to the selected cases; serving variation prevents an isolated causal claim.');
      await screenshot('08-targeted-followup-summary');
      const repeated=retest.results.find(row=>row.id===important?.id&&hasResponse(row));
      if (repeated) {
        const response=page.locator('article[id="'+repeated.id+'"]').getByRole('heading',{name:'Saved response',exact:true}).first();
        await scrollTo(response);
        await focusMissingInformation(response);
        await hold(10,'Same case in the targeted follow-up · '+repeated.id,
          repeated.score+': '+short(repeated.reason,220)+' The original baseline result remains preserved.');
        await screenshot('09-targeted-followup-response');
      }
      await page.goto(args.reportUrl || pathToFileURL(html).href,{waitUntil:'load'});
    }
    await scrollTo(page.getByRole('heading',{name:'Measurement limits',exact:true}));
    await hold(10,'What these results establish',
      'A PASS means the configured failure was not observed. Errors and ambiguity remain WARN. The complete JSON retains prompts, responses, events, source provenance, and review status.');
    await screenshot('07-measurement-limits');
    await scrollTo(page.getByRole('heading',{name:'Attack and defense matrix',exact:true}));
    await hold(8,'Submission evidence and recommended controls',
      'The findings document links this complete evidence. Recommended controls target the observed boundary; the saved run does not claim that every recommendation was deployed.');
  } finally {
    await context.close();
    try {
      // The recording is finalized when its context closes, but saveAs still
      // needs the browser connection. Closing the browser first loses it.
      await recordedVideo.saveAs(args.output);
    } finally {
      await browser.close();
    }
  }
  const manifest={
    report:path.relative(root,args.report),report_sha256:createHash('sha256').update(raw).digest('hex'),
    run_id:report.manifest?.run_id || report.manifest?.run_name || null,
    video:path.relative(root,args.output),format:'webm VP8',resolution:'1440x1000',
    content:'Actual Chromium browser capture with explanatory text captions; no narration track.',
    duration_scale:args.scale,wall_seconds:Number(((Date.now()-started)/1000).toFixed(2)),
    selected_cases:[important?.id,crescendo?.id,benign?.id].filter(Boolean),
    targeted_retest:args.retestReport ? path.relative(root,args.retestReport) : null,
    app_included:!args.skipApp,timeline,
  };
  await fs.writeFile(args.output.replace(/\.webm$/,'.json'),JSON.stringify(manifest,null,2)+'\n');
  console.log(JSON.stringify({video:args.output,seconds:manifest.wall_seconds,screenshots:args.screenshots}));
}

main().catch(error=>{console.error(error.stack||error.message);process.exitCode=1;});
