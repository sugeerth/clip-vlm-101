// flowviz.js — the LIVE structured-agent-graph trace. Play one case through the
// flow.js DAG and watch it happen: sub-agents fan out, contracts are checked,
// nodes go pending → running → ok / quarantined / skipped, the graph escalates or
// fails closed. Page-only, like trace.js/motion.js — there is no "live" in a batch
// script. The graph and the span waterfall are driven by flow.js executing for
// real (the byte-identical twin of flow.py); nothing here is faked.
//
// Design follows how production agent-observability tools (LangSmith, Arige
// Phoenix, LangGraph Studio) render an agent run: a node-graph for topology +
// a span waterfall for order, semantic status colors (running/ok/error/skipped),
// live streaming of each step, and drill-down on the sub-agents.
import { runVerdict, RUBRICS } from './flow.js';

const wait = ms => new Promise(r => setTimeout(r, ms));
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const dwell = ms => (reduced ? Promise.resolve() : wait(ms));
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

// the graph is essentially linear (panel → council → decide) with the fan-out
// living inside `panel`, so we lay it out as tiers left→right and show the
// sub-agents as chips — faithful to the topology, and readable without a layout engine.
// `needs`/`contract` mirror flow.py exactly; `span` maps each node to its OpenTelemetry
// GenAI operation, the emerging standard vocabulary for agent traces.
const TIERS = [
  { key: 'input', name: 'input', meta: 'query · result', needs: [], contract: ['query', 'result'], span: 'invoke_workflow' },
  { key: 'panel', name: 'panel', meta: 'fan-out: 3 sub-agents', needs: [], contract: ['workers'], span: 'invoke_agent ×N (parallel)' },
  { key: 'council', name: 'council', meta: 'fan-in: aggregate', needs: ['panel'], contract: ['decision', 'reason'], span: 'chat / aggregate' },
  { key: 'decide', name: 'decide', meta: 'rule / escalate', needs: ['council', 'panel'], contract: ['decision', 'reason', 'via'], span: 'handoff → debate' },
];
const INFO = Object.fromEntries(TIERS.map(t => [t.key, t]));

let ITEMS = [];
const by = s => ITEMS.find(it => (it.file || it.path || '').includes(s));
const ROGUE = ['rogue', () => ({ name: 'rogue', opinion: 'trust me' })];
const CASES = [
  { id: 'catdog', label: 'cat → dog · easy', q: '004_cat', r: '005_dog' },
  { id: 'catparrot', label: 'cat → parrot · panel rules', q: '004_cat', r: '009_parrot' },
  { id: 'applepizza', label: 'apple → pizza · escalates to debate', q: '000_apple', r: '010_pizza' },
  { id: 'rogue', label: 'apple → pizza + a rogue sub-agent', q: '000_apple', r: '010_pizza', rogue: true },
];

const graphBox = document.getElementById('fgraph');
const waterBox = document.getElementById('fwaterfall');
const casesBox = document.getElementById('fcases');
const detailBox = document.getElementById('fdetails');
let nodes = {}, running = false, token = 0;
let lastOutputs = {}, lastStatus = {}, selected = null;

function buildGraph() {
  graphBox.replaceChildren();
  nodes = {};
  TIERS.forEach((t, i) => {
    if (i) graphBox.append(el('div', 'fedge', '›'));
    const tier = el('div', 'ftier');
    const node = el('div', 'fnode is-pending');
    node.append(el('div', 'fname', `<span class="fdot"></span>${t.name}`));
    node.append(el('div', 'fmeta', t.meta));
    node.append(el('div', 'fcontract', `⊢ ${t.contract.join(', ')}`));   // the output contract, always visible
    if (t.key === 'panel') { const subs = el('div', 'fsubs'); subs.id = 'fsubs'; node.append(subs); }
    node.addEventListener('click', () => select(t.key));
    tier.append(node);
    graphBox.append(tier);
    nodes[t.key] = node;
  });
  graphBox.append(el('div', 'fedge', '⇒'));
  const v = el('div', 'ftier fverdict');
  const pill = el('div', 'fpill', 'verdict');
  pill.id = 'fpill';
  v.append(pill);
  graphBox.append(v);
}

function setState(key, state, meta) {
  const n = nodes[key];
  if (!n) return;
  const sel = n.classList.contains('sel');
  n.className = `fnode ${state}${sel ? ' sel' : ''}`;
  lastStatus[key] = state;
  if (meta != null) n.querySelector('.fmeta').textContent = meta;
}

function waterRow(key, name, mark, tag) {
  const row = el('div', 'fwrow');
  row.dataset.key = key;
  row.append(el('div', 'fwname', `<span class="fmark ${mark}">${mark === 'ok' ? '✓' : mark === 'bad' ? '✗' : '·'}</span>${name}`));
  const right = el('div');
  right.style.display = 'flex'; right.style.alignItems = 'center'; right.style.gap = '10px';
  const bar = el('div', `fwbar ${mark}`);
  bar.style.width = (mark === 'skip' ? 26 : 60 + Math.random() * 40) + 'px';
  right.append(bar, el('span', 'fwtag', tag));
  row.append(right);
  if (key) { row.style.cursor = 'pointer'; row.addEventListener('click', () => select(key)); }
  waterBox.append(row);
}

// linked selection: click a node OR its span row → highlight both, show its
// structured detail (needs / contract / output). The two-view drill-down every
// production agent-observability tool ships.
function select(key) {
  selected = key;
  const info = INFO[key];
  if (!info) return;
  Object.entries(nodes).forEach(([k, n]) => n.classList.toggle('sel', k === key));
  waterBox.querySelectorAll('.fwrow').forEach(r => r.style.background = r.dataset.key === key ? 'var(--accent-wash)' : '');
  const status = lastStatus[key] || 'is-pending';
  const mark = status === 'is-ok' ? '<span class="ok">✓ ok</span>'
    : status === 'is-bad' ? '<span class="bad">✗ off-contract</span>'
    : status === 'is-skip' ? '· skipped' : '· pending';
  const out = lastOutputs[key];
  const met = info.contract.map(k =>
    (out && k in out) ? `<span class="ok">${k}&nbsp;✓</span>` : `<span class="bad">${k}&nbsp;✗</span>`).join('&nbsp;&nbsp;');
  let outStr = out ? JSON.stringify(out, (k, v) => typeof v === 'number' ? +v.toFixed(3) : v) : '—';
  if (outStr.length > 220) outStr = outStr.slice(0, 217) + '…';
  detailBox.innerHTML =
    `<div class="fdh">${info.name} <span class="fspan">otel: ${info.span}</span> ${mark}</div>`
    + `<div class="fdgrid">`
    + `<span class="fdk">needs</span><span class="fdv">${info.needs.length ? info.needs.join(', ') : '— (reads workflow input)'}</span>`
    + `<span class="fdk">contract</span><span class="fdv">${met}</span>`
    + `<span class="fdk">output</span><span class="fdv">${outStr}</span>`
    + `</div>`;
}

async function play(caseDef) {
  if (running) return;
  running = true;
  const mine = ++token;
  document.querySelectorAll('.fcase').forEach(b => b.classList.toggle('on', b.dataset.id === caseDef.id));
  buildGraph();
  waterBox.replaceChildren();
  lastStatus = {}; selected = null;

  const extra = caseDef.rogue ? [ROGUE] : [];
  const res = runVerdict(by(caseDef.q), by(caseDef.r), extra);
  const out = res.outputs;
  // the workflow input isn't a flow.js node; synthesize its output for drill-down
  lastOutputs = { input: { query: caseDef.q.split('_')[1], result: caseDef.r.split('_')[1] }, ...out };

  const alive = () => mine === token;

  // 1) input
  setState('input', 'is-running'); await dwell(360);
  if (!alive()) return;
  setState('input', 'is-ok', `${caseDef.q.split('_')[1]} → ${caseDef.r.split('_')[1]}`);
  waterRow('input', 'input', 'ok', 'query · result ready');

  // 2) panel — stream the sub-agents fanning out
  setState('panel', 'is-running'); await dwell(300);
  const subsBox = document.getElementById('fsubs');
  const spawn = [...RUBRICS, ...(caseDef.rogue ? ['rogue'] : [])];
  const kept = new Map((out.panel?.workers || []).map(w => [w.name, w]));
  const dropped = new Set(out.panel?.dropped || []);
  let keptN = 0;
  for (const name of spawn) {
    if (!alive()) return;
    const w = kept.get(name);
    const chip = el('div', 'fsub');
    const drop = dropped.has(name);
    chip.classList.toggle('dropped', drop);
    chip.innerHTML = `<span>${drop ? '✗' : '◦'}</span><span>${name}</span>`
      + `<span class="fsv">${drop ? 'no score → dropped' : w.score.toFixed(2)}</span>`;
    subsBox.append(chip);
    await dwell(200);
    chip.classList.add('in');
    if (!drop) keptN++;
  }
  const panelMeta = dropped.size
    ? `${keptN} kept · ${dropped.size} dropped (contract)`
    : `${keptN} sub-agents · all valid`;
  setState('panel', 'is-ok', panelMeta);
  waterRow('panel', 'panel', dropped.size ? 'bad' : 'ok', dropped.size ? `dropped: ${[...dropped].join(', ')}` : 'fan-out: 3 valid votes');
  await dwell(240);

  // 3) council — fan-in
  if (!alive()) return;
  setState('council', 'is-running'); await dwell(420);
  const c = out.council;
  const cmeta = c.decision === 'relevant' || c.decision === 'not relevant'
    ? `${c.decision} · consensus ${(c.consensus ?? 0).toFixed(2)}`
    : `${c.reason} · consensus ${(c.consensus ?? 0).toFixed(2)}`;
  setState('council', 'is-ok', cmeta);
  waterRow('council', 'council', 'ok', cmeta);
  await dwell(240);

  // 4) decide — rule or escalate
  if (!alive()) return;
  setState('decide', 'is-running'); await dwell(460);
  const d = out.decide;
  const via = d.via === 'debate' ? 'escalated → debate' : 'ruled at the council';
  setState('decide', 'is-ok', `${d.decision} · ${via}`);
  waterRow('decide', 'decide', d.decision === 'abstain' ? 'bad' : 'ok',
    d.via === 'debate' ? (d.factions ? `debate: ${d.factions.map(f => '{' + f.join(', ') + '}').join(' | ')}` : 'debate: consensus') : 'council ruled');

  // 5) verdict pill
  if (!alive()) return;
  const pill = document.getElementById('fpill');
  pill.textContent = d.decision + (d.reason ? ` · ${d.reason}` : '');
  pill.classList.add('show', d.decision === 'abstain' ? 'abstain' : 'relevant');

  running = false;
  if (alive()) select('decide');       // leave the terminal node's detail open for inspection
}

async function main() {
  ITEMS = (await fetch('db.json').then(r => r.json())).items;
  CASES.forEach((cd, i) => {
    const b = el('button', 'fcase' + (i === 0 ? ' on' : ''), cd.label);
    b.dataset.id = cd.id;
    b.addEventListener('click', () => play(cd));
    casesBox.append(b);
  });
  buildGraph();
  play(CASES[0]);
}

main();
