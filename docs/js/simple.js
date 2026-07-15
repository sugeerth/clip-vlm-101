// CLIP·search — the simple page. One box, real inference, results.
//
// All the complexity this repo teaches is deliberately invisible here:
// the model loads once on first use (with a quiet progress bar), queries
// are embedded live as you type, and ranking is the same fused dot
// product as search.py. The explorable at explore.html shows the insides.
import { dot, fuse } from './rank.js';
import { hermesSearch, MIN_MARGIN } from './hermes.js';
import { getTextEncoder, getImageEncoder, getActiveModel, setActiveModel } from './clip.js';
import { MODELS, DEFAULT_MODEL } from './models.js';
import { discover } from './crawler.js';
import { explain, explainWithLLM } from './explain.js';
import { councilWithLLM } from './judge.js';
import { compose, gateTrust, conformalTrust, councilTrust, marginTrust, WEIGHTS } from './trust.js';
import { debate, fromCouncil } from './debate.js';
import { STRONG, MODERATE, WEAK } from './explain.js';
import { OnlineRanker } from './learn.js';
import { looScores, calibrate } from './conformal.js';
import { flip } from './motion.js';

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];
const TOP_K = 8;

const DB = await (await fetch('db.json')).json();

// the on-device personal ranker (learn.js) — restored from localStorage, a
// model of a few floats that never leaves the machine. And a conformal
// threshold (conformal.js) calibrated ONCE on the gallery for the 80 % set.
const RANK_KEY = 'personalRanker.v1';
const ranker = new OnlineRanker();
try { const st = JSON.parse(localStorage.getItem(RANK_KEY)); if (st) ranker.loadState(st); } catch { /* ignore */ }
const persistRanker = () => localStorage.setItem(RANK_KEY, JSON.stringify(ranker.toState()));
// Calibrate in the SAME modality a live search queries in: text_emb (caption,
// the stand-in query) over image_emb. Image→image cosines sit at ~0.5–1.0 but
// text→image cosines at ~0.15–0.30 (CLIP's modality gap), so a threshold from
// the image-side band would never fire on a real text query.
const CAL = looScores(DB.items, 'image_emb', 'text_emb');
const TAU80 = 1 - calibrate(CAL, 0.2);        // cos ≥ TAU80  → the 80%-coverage set
let candCtx = null;                            // { query, cand:[{item,features,...}] }

let encodeText = null;      // resolves once per brain, then ~instant
let encodeTextModel = null; // WHICH brain the cached encoder belongs to
let imageQuery = null;      // { emb } — set by camera / drop / paste
let searching = false, rerun = false, debounce = 0;

const status = t => { $('status').textContent = t; };
const progress = p => {
  if (p.status === 'progress' && p.total) {
    $('barTrack').classList.remove('hidden');
    $('bar').style.width = Math.round(100 * p.loaded / p.total) + '%';
  }
};
const hideBar = () => $('barTrack').classList.add('hidden');

// ---------------------------------------------------------------- results --
function show(entries, note = '', feedback = false) {
  document.body.classList.add('searched');
  $('results').replaceChildren(...entries.map(({ item }, i) => {
    const fig = document.createElement('figure');
    fig.dataset.key = item.file;                 // FLIP identity for re-rank
    fig.style.animationDelay = `${i * 45}ms`;
    if (feedback) fig.append(feedbackButtons(item));
    fig.append(
      Object.assign(document.createElement('img'),
        { src: item.file, alt: item.caption, loading: 'lazy' }),
      Object.assign(document.createElement('figcaption'),
        { textContent: item.tags.slice(0, 3).join(' · ') }));
    // details later: any result opens its full dissection in the Lab
    const idx = DB.items.indexOf(item);
    if (idx >= 0) {
      fig.classList.add('linked');
      fig.tabIndex = 0;
      fig.title = 'see why this matched — open it in the Lab';
      const openLab = () => { location.href = `explore.html?pick=${idx}#lab`; };
      fig.addEventListener('click', openLab);
      fig.addEventListener('keydown', e => { if (e.key === 'Enter') openLab(); });
    }
    return fig;
  }));
  $('detailsHint').classList.toggle('hidden', !entries.length);
  status(note);
}

// details on demand: one line on what Hermes did, the full trace one tap away
function showTrace(out) {
  $('trace').classList.remove('hidden');
  $('traceSummary').textContent = out.satisfied
    ? `hermes chose “${out.chose}”`
    : 'hermes: no phrasing was decisive — answered with their ensemble';
  $('traceBody').replaceChildren(...out.rounds.map(r =>
    Object.assign(document.createElement('div'), {
      className: 'round' + (r.margin >= MIN_MARGIN ? ' ok' : ''),
      textContent: `${r.margin >= MIN_MARGIN ? '✓' : '·'} “${r.phrasing}” — margin ${r.margin >= 0 ? '+' : ''}${r.margin.toFixed(3)}`,
    })));
}
const hideTrace = () => $('trace').classList.add('hidden');

// model unavailable? still answer: plain keyword match over tags + captions
function keywordResults(query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return DB.items.map(item => ({
    item,
    score: words.filter(w => item.tags.some(t => t.includes(w))
      || item.caption.includes(w)).length,
  })).filter(e => e.score > 0 || !words.length)
    .sort((a, b) => b.score - a.score).slice(0, TOP_K);
}

// ------------------------------------------------------------- the brain --
// Different models live in different vector spaces — they never mix (the
// repo's own lesson). The shipped gallery vectors are CLIP B/32; choosing
// another brain re-embeds all 14 gallery images + captions right here in
// the browser, once per session, before any ranking happens.
const galleryCache = { [DEFAULT_MODEL]: DB.items.map(it =>
  ({ image_emb: it.image_emb, text_emb: it.text_emb, fused_emb: it.fused_emb })) };

const brainName = key => MODELS[key].label.split(' · ')[0];

async function ensureBrainReady() {
  const key = getActiveModel();
  if (!galleryCache[key]) {
    const encT = await getTextEncoder(status, progress);
    const encI = await getImageEncoder(status, progress);
    hideBar();
    const snap = [];
    for (let i = 0; i < DB.items.length; i++) {
      status(`re-embedding the gallery with ${brainName(key)} — ${i + 1}/${DB.items.length}…`);
      const blob = await (await fetch(DB.items[i].file)).blob();
      const image_emb = await encI(blob);
      const [text_emb] = await encT([DB.items[i].caption]);
      snap.push({ image_emb, text_emb, fused_emb: fuse(image_emb, text_emb) });
    }
    galleryCache[key] = snap;
    status('');
  }
  DB.items.forEach((it, i) => Object.assign(it, galleryCache[key][i]));
}

async function switchBrain(key) {
  const prev = getActiveModel();
  if (key === prev) return;
  setActiveModel(key);
  encodeText = null; encodeTextModel = null;
  clearImageQuery();        // an old-space image query cannot carry over
  clearWeb();               // web embeddings are per-model too
  try {
    // load the new brain HERE, where a failure is catchable — search()
    // swallows its own errors into the keyword fallback and never throws
    encodeText = await getTextEncoder(status, progress);
    encodeTextModel = key;
    hideBar();
    await ensureBrainReady();
    localStorage.setItem('brain', key);       // persist only what worked
    status(`brain: ${MODELS[key].label}`);
    if ($('q').value.trim()) await search();
  } catch (err) {
    console.error(err);
    hideBar();
    setActiveModel(prev);
    encodeText = null; encodeTextModel = null;
    if (galleryCache[prev]) DB.items.forEach((it, i) => Object.assign(it, galleryCache[prev][i]));
    $('brainSel').value = prev;
    status(`could not load ${brainName(key)} — staying on ${brainName(prev)}`);
  }
}

// ----------------------------------------------------------------- search --
async function search() {
  const query = $('q').value.trim();
  if (!query && !imageQuery) return;
  if (searching) { rerun = true; return; }
  searching = true;
  try {
    if (imageQuery && imageQuery.model !== getActiveModel()) {
      clearImageQuery();                      // an old-space photo can't be reused
    }
    if (imageQuery) {                         // image → image, like search.py --image
      hideTrace();
      clearWeb();
      $('explain').classList.add('hidden');
      $('learnPanel').classList.add('hidden'); candCtx = null;
      await ensureBrainReady();
      show(DB.items
        .map(item => ({ item, score: dot(item.image_emb, imageQuery.emb) }))
        .sort((a, b) => b.score - a.score).slice(0, TOP_K));
    } else {
      // the encoder must belong to the ACTIVE brain — if the user switches
      // mid-download, the loop re-resolves instead of caching a stale one
      while (!encodeText || encodeTextModel !== getActiveModel()) {
        status('loading the model — one time, cached after…');
        const key = getActiveModel();
        encodeText = await getTextEncoder(status, progress);
        encodeTextModel = key;
      }
      hideBar();
      await ensureBrainReady();
      // Hermes works the query: propose phrasings, critique each by its
      // retrieval margin, ensemble if none is decisive — then answer.
      const out = await hermesSearch(query, encodeText, DB.items, TOP_K);
      // rank on the SAME embedding Hermes chose (best phrasing or ensemble), so
      // the results match the "🪽 hermes chose …" trace instead of the raw query.
      personalize(query, out.qvec);               // learned re-rank + 👍/👎 + confidence
      showTrace(out);
      // explain reads .score as a real cosine (its strength bands + the gate's
      // number check depend on it), so pass the fused cosine in DISPLAY order —
      // not the ranker's min-max-normalized blend score (whose top is ≈1.0).
      renderExplain(query, ranker.rank(candCtx.cand, { k: TOP_K })
        .map(c => ({ item: c.item, score: c.base_score })));  // why (gated)
      webPhase(query);               // fire-and-forget: local results never wait
    }
  } catch (err) {
    console.error(err);
    hideBar();
    hideTrace();
    clearWeb();
    $('explain').classList.add('hidden');
    $('learnPanel').classList.add('hidden'); candCtx = null;
    const hits = keywordResults(query);
    show(hits, hits.length ? 'model unavailable here — showing keyword matches'
                           : 'model unavailable and no keyword matches');
  }
  searching = false;
  if (rerun) { rerun = false; search(); }
}

// ------------------------------------------------------ personalization --
// 👍/👎 on a result trains the on-device ranker (learn.js) and re-ranks live.
function feedbackButtons(item) {
  const wrap = document.createElement('div');
  wrap.className = 'fb';
  for (const [label, glyph] of [[1, '👍'], [0, '👎']]) {
    const b = Object.assign(document.createElement('button'),
      { type: 'button', textContent: glyph, title: label ? 'more like this' : 'less like this' });
    b.setAttribute('aria-label', label ? 'thumbs up' : 'thumbs down');
    b.addEventListener('click', e => { e.stopPropagation(); giveFeedback(item, label); });
    wrap.append(b);
  }
  return wrap;
}

function personalize(query, qvec) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const cand = DB.items.map(item => {
    const cos_image = dot(item.image_emb, qvec), cos_text = dot(item.text_emb, qvec);
    const tag_overlap = item.tags.filter(t => words.some(w => t.includes(w) || w.includes(t))).length;
    return { item, cos_image, cos_text, tag_overlap, base_score: (cos_image + cos_text) / 2 };
  });
  cand.sort((a, b) => b.base_score - a.base_score);
  cand.forEach((c, i) => { c.rank_prior = 1 / (i + 1);
    c.features = [c.cos_image, c.cos_text, c.tag_overlap, c.rank_prior]; });
  candCtx = { query, cand };
  renderPersonalized();
}

function renderPersonalized() {
  if (!candCtx) return;
  show(ranker.rank(candCtx.cand, { k: TOP_K }), '', true);
  renderLearnPanel();
}

function giveFeedback(item, label) {
  const c = candCtx?.cand.find(c => c.item === item);
  if (!c) return;
  ranker.feedback(c.features, label);
  persistRanker();
  flip($('results'), () => renderPersonalized());   // re-rank plays as motion
}

function renderLearnPanel() {
  const el = $('learnPanel');
  el.classList.remove('hidden');
  const beta = ranker.rank(candCtx.cand)[0]?.beta ?? 0;
  const imp = ranker.importance().slice().sort((a, b) => b.importance - a.importance);
  const row = document.createElement('div'); row.className = 'row';
  row.append(Object.assign(document.createElement('span'), {
    innerHTML: ranker.n
      ? `learning from <b>${ranker.n}</b> 👍/👎 (${ranker.nPairs()} pairs) — personal weight <b>${(beta * 100 | 0)}%</b>`
      : 'tip: 👍/👎 a result — a tiny ranker learns your taste, on your device' }));
  const reset = Object.assign(document.createElement('button'),
    { type: 'button', className: 'reset', textContent: 'reset' });
  reset.addEventListener('click', () => {
    localStorage.removeItem(RANK_KEY);
    ranker.buffer.length = 0; ranker._refit();   // back to the untrained w=[1,0,0,0]
    flip($('results'), () => renderPersonalized());
  });
  row.append(reset);
  el.replaceChildren(row);
  if (ranker.n) {                                    // interpretable learned weights
    const bars = document.createElement('div'); bars.className = 'bars';
    const max = Math.max(...imp.map(f => f.importance), 1e-6);
    for (const f of imp) {
      const feat = document.createElement('span'); feat.className = 'feat';
      const bar = document.createElement('i'); bar.style.width = `${Math.round(46 * f.importance / max)}px`;
      feat.append(`${f.name.replace('_', ' ')}`, bar);
      bars.append(feat);
    }
    el.append(bars);
  }
  renderConfidence(el);
}

// conformal.js: how many results clear the gallery-calibrated 80% bar.
function renderConfidence(el) {
  const inSet = candCtx.cand.filter(c => c.cos_image >= TAU80).length;
  const conf = document.createElement('div');
  conf.className = 'conf' + (inSet ? '' : ' abstain');
  conf.title = 'Split-conformal: τ was calibrated cross-modally on the gallery '
    + '(each caption a held-out query over the images, leave-one-out) so a same-'
    + 'tag match lands in the set ≥80% of the time — a marginal guarantee; loose '
    + 'natural-language queries drift from the caption calibration, so it errs '
    + 'toward abstaining.';
  conf.innerHTML = inSet
    ? `🎯 <b>80%-confidence set</b>: ${inSet} result${inSet > 1 ? 's' : ''} clear the calibrated bar (cos ≥ ${TAU80.toFixed(2)})`
    : `🎯 <b>abstain</b>: nothing clears the 80% bar (cos ≥ ${TAU80.toFixed(2)}) — no confident match`;
  el.append(conf);
}

// ------------------------------------------------------- the explanation --
// Say WHY these matched, from verifiable facts only, and offer an optional
// language model that must pass the SAME hallucination gate (explain.js).
// Paints ONLY the explanation body (into its own sub-container). The LLM button
// re-paints this body in place; the council panel is a separate sibling, so
// upgrading the explanation never destroys a rendered verdict, and vice-versa.
function paintExplain(body, res, query) {
  const why = document.createElement('div');
  why.className = 'why';
  why.textContent = res.explanation;
  const row = document.createElement('div');
  row.className = 'row2';
  const btn = Object.assign(document.createElement('button'),
    { type: 'button', className: 'llm-btn', textContent: '✨ explain with a language model' });
  const gate = document.createElement('span');
  gate.className = 'gate';
  if (res.stripped && res.stripped.length) {
    gate.classList.add('stripped');
    gate.textContent = `gate removed ${res.stripped.length} unsupported claim${res.stripped.length > 1 ? 's' : ''}`;
    gate.title = res.stripped.map(s => s.reasons.join('; ')).join(' · ');
  } else if (res.source) {
    gate.textContent = res.source.startsWith('llm') ? '✓ gated — every claim checks out' : res.source;
  }
  btn.addEventListener('click', async () => {
    const gen = explainGen;            // snapshot this render generation
    btn.disabled = true; gate.className = 'gate'; gate.textContent = '';
    const out = await explainWithLLM(query, lastRanked,
      t => { if (gen === explainGen) gate.textContent = t; });
    if (gen !== explainGen || !body.isConnected) return;   // a newer search replaced the panel
    paintExplain(body, out, query);    // repaints ONLY the body; the council stays put
  });
  row.append(btn, gate);
  body.replaceChildren(why, row);
}

// A second, independent honesty layer: a COUNCIL of LLM judges rules on the top
// result (judge.js). Lives in its own container so the explanation's LLM
// upgrade can't clobber it; guarded by explainGen AND DOM-attachment so a slow
// council call that resolves after a new search is dropped, not mis-rendered.
function mountCouncil(wrap, query) {
  const cbtn = Object.assign(document.createElement('button'),
    { type: 'button', className: 'llm-btn', textContent: '⚖️ convene a council of LLM judges' });
  const cbox = document.createElement('div'); cbox.className = 'council hidden';
  cbtn.addEventListener('click', async () => {
    const gen = explainGen;
    cbtn.disabled = true; cbox.classList.remove('hidden');
    cbox.textContent = 'convening the council…';
    const top = lastRanked[0];
    if (!top) { cbox.textContent = 'no result to judge.'; return; }
    const words = (query || '').toLowerCase().split(/\s+/).filter(Boolean);
    const tags = top.item.tags || [];
    const ev = { tags, topScore: top.score ?? 0,
      shared: tags.filter(t => words.some(w => t.includes(w) || w.includes(t))) };
    const res = await councilWithLLM(query, ev,
      t => { if (gen === explainGen && cbox.isConnected) cbox.textContent = t; });
    if (gen !== explainGen || !cbox.isConnected) return;   // superseded by a newer search
    renderCouncil(cbox, res, top.item);
    renderTrust(res);                  // fold the council's verdict into the trust capstone
  });
  wrap.replaceChildren(cbtn, cbox);
}

// The council's ruling, with each judge's GATE made visible: its raw utterance
// → a ✓/⊘ gate glyph → the score bar. A judge whose text carries no number in
// [0,1] is BLOCKED by the gate (⊘) and abstains — the same "drop it, don't
// guess" stance as the hallucination gate. Then the weighted verdict + a
// consensus meter, or an honest abstain (no quorum / hung jury).
function renderCouncil(box, res, item) {
  box.replaceChildren();
  const passed = res.perJudge.filter(j => j.score !== null && j.score !== undefined).length;
  const blocked = res.perJudge.length - passed;
  const head = document.createElement('div'); head.className = 'chead';
  head.innerHTML = `⚖️ <b>council of ${res.nTotal} LLM judges</b> on the top result `
    + `<i>“${(item.tags || []).slice(0, 3).join(' · ')}”</i> — `
    + `<span class="gatesum">🚪 gate: ${passed} passed${blocked ? ` · <b>${blocked} blocked</b>` : ''}</span>`;
  box.append(head);
  for (const j of res.perJudge) {
    const abst = j.score === null || j.score === undefined;
    const m = /said "([^"]*)"/.exec(j.rationale || '');
    const raw = m ? m[1] : (abst ? '—' : (j.rationale || ''));
    const row = document.createElement('div'); row.className = 'jrow' + (abst ? ' blocked' : '');
    row.append(Object.assign(document.createElement('span'), { className: 'jname', textContent: j.name }));
    // the GATE: raw utterance → ✓ / ⊘
    const chip = document.createElement('span'); chip.className = 'gatechip ' + (abst ? 'block' : 'pass');
    chip.innerHTML = `<span class="raw">“${raw}”</span><span class="arrow">→</span>`
      + `<span class="glyph">${abst ? '⊘' : '✓'}</span>`;
    chip.title = abst ? 'no number in [0,1] → blocked by the gate, this judge abstains'
      : `parsed to ${j.score.toFixed(2)} — cleared the score gate`;
    row.append(chip);
    // the score track + value
    const track = document.createElement('span'); track.className = 'jtrack';
    const fill = document.createElement('i');
    fill.style.width = abst ? '0%' : `${Math.round(100 * j.score)}%`;
    track.append(fill);
    row.append(track, Object.assign(document.createElement('span'),
      { className: 'jscore', textContent: abst ? 'abstains' : j.score.toFixed(2) }));
    box.append(row);
  }
  const v = document.createElement('div');
  v.className = 'verdict ' + (res.decision === 'relevant' ? 'ok'
    : res.decision === 'abstain' ? 'abstain' : 'no');
  const glyph = res.decision === 'relevant' ? '✅' : res.decision === 'abstain' ? '⚖️' : '🚫';
  const pct = Math.round((res.consensus ?? 0) * 100);
  if (res.decision === 'abstain') {
    v.innerHTML = res.reason === 'hung jury'
      ? `${glyph} <b>hung jury</b> — the judges span ${res.spread?.toFixed(2)}; the council abstains`
      : `${glyph} <b>abstain</b> — no quorum (too few judges cleared the gate)`;
  } else {
    v.innerHTML = `${glyph} <b>${res.decision}</b> · weighted mean <b>${res.mean?.toFixed(2)}</b>`
      + `<span class="meter" title="consensus ${pct}%"><i style="width:${pct}%"></i></span>`
      + `consensus <b>${pct}%</b>`;
  }
  box.append(v);

  // DELIBERATION: the judges voted independently — now let them TALK (debate.js).
  // Bounded-confidence dynamics, deterministic, instant: no extra model calls.
  const seat = fromCouncil(res.perJudge);
  if (seat.opinions.length >= 2) {
    const dbtn = Object.assign(document.createElement('button'),
      { type: 'button', className: 'llm-btn', textContent: '🗣️ let them debate' });
    const dbox = document.createElement('div'); dbox.className = 'debate hidden';
    dbtn.addEventListener('click', () => {
      dbtn.disabled = true; dbox.classList.remove('hidden'); renderDebate(dbox, seat);
    });
    box.append(dbtn, dbox);
  }
}

// Render a multi-agent debate: each seated judge is a line; watch them converge
// to one consensus or split into factions across the rounds (debate.js).
const AGENT_COLORS = ['#2a78d6', '#c0392b', '#2a8a4a', '#8e44ad', '#b06a1a'];
function renderDebate(box, seat) {
  const d = debate(seat.opinions, seat.weights);
  const R = d.trajectory.length - 1, W = 260, H = 96, PAD = 22;
  const x = r => PAD + (R ? r / R : 0) * (W - 2 * PAD);
  const y = o => H - PAD - o * (H - 2 * PAD);
  const lines = seat.names.map((name, i) => {
    const pts = d.trajectory.map((pos, r) => `${x(r).toFixed(1)},${y(pos[i]).toFixed(1)}`).join(' ');
    const c = AGENT_COLORS[i % AGENT_COLORS.length];
    return `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2"/>`
      + `<circle cx="${x(R).toFixed(1)}" cy="${y(d.final[i]).toFixed(1)}" r="2.6" fill="${c}"/>`;
  }).join('');
  const mid = y(0.5).toFixed(1);
  const svg = `<svg viewBox="0 0 ${W} ${H}" class="dtraj">`
    + `<line x1="${PAD}" y1="${mid}" x2="${W - PAD}" y2="${mid}" class="half"/>${lines}</svg>`;
  const legend = seat.names.map((n, i) =>
    `<span class="ag"><i style="background:${AGENT_COLORS[i % AGENT_COLORS.length]}"></i>${n}</span>`).join('');
  const camps = d.factions.map(g => '{' + g.map(i => seat.names[i]).join(', ') + '}').join(' vs ');
  const flipped = d.flips.length ? ` · ${d.flips.map(i => seat.names[i]).join(', ')} changed their mind` : '';
  const summary = d.consensus
    ? `🤝 <b>consensus</b> after ${d.rounds} round${d.rounds > 1 ? 's' : ''} — they talked it out to <b>${d.score.toFixed(2)}</b> (${d.verdict})${flipped}`
    : `⚔️ <b>contested</b> after ${d.rounds} round${d.rounds > 1 ? 's' : ''} — the panel split: ${camps}. Deliberation couldn't move them${flipped}`;
  box.innerHTML = `<div class="dhead">🗣️ the judges debate — each moves only toward peers within reach</div>`
    + svg + `<div class="dlegend">${legend}</div>`
    + `<div class="dsum ${d.consensus ? 'ok' : 'no'}">${summary}</div>`;
}

let lastRanked = [], explainGen = 0, trustLine = null;
function renderExplain(query, ranked) {
  explainGen++;                        // invalidate any in-flight LLM explanation / council
  lastRanked = ranked;
  const el = $('explain');
  el.classList.remove('hidden');
  // the CAPSTONE first: one trust verdict composed from every honesty lens,
  // then two independent sub-panels (the explanation body, the council) that
  // each re-paint on their OWN button without touching the other.
  trustLine = document.createElement('div'); trustLine.className = 'trust';
  const body = document.createElement('div'); body.className = 'ebody';
  const cwrap = document.createElement('div'); cwrap.className = 'cwrap';
  el.replaceChildren(trustLine, body, cwrap);
  renderTrust(null);                   // gate + conformal + margin now; council abstains until convened
  paintExplain(body, explain(query, ranked), query);
  mountCouncil(cwrap, query);
}

// trust.js: compose the four honesty lenses on the top result into ONE verdict.
// gate (match strength) + conformal (clears the calibrated bar?) + margin
// (decisively ahead?) are known immediately; council stays null until convened,
// then folds in. A council of gates — high only when the lenses agree, abstain
// when they split.
function liveTrust(councilVerdict) {
  const top = lastRanked[0];
  if (!top || !candCtx) return null;
  const c = candCtx.cand.find(x => x.item === top.item);
  const scores = lastRanked.map(r => r.score);
  const signals = [
    { name: 'gate', trust: gateTrust(top.score, STRONG, MODERATE, WEAK), weight: WEIGHTS.gate },
    { name: 'conformal', trust: conformalTrust(c ? c.cos_image : -1, TAU80), weight: WEIGHTS.conformal },
    { name: 'council', trust: councilTrust(councilVerdict), weight: WEIGHTS.council },
    { name: 'margin', trust: marginTrust(scores, 0.05), weight: WEIGHTS.margin },  // text→image margins are small
  ];
  return { v: compose(signals), signals };
}

const TRUST_GLYPH = { high: '🔒', medium: '🔓', low: '⚠️', abstain: '🤔' };
function renderTrust(councilVerdict) {
  if (!trustLine) return;
  const t = liveTrust(councilVerdict);
  if (!t) { trustLine.classList.add('hidden'); return; }
  trustLine.classList.remove('hidden');
  const { v, signals } = t;
  const head = document.createElement('div'); head.className = 'thead';
  const label = v.level === 'abstain'
    ? `trust: <b>can't say</b> — ${v.reason === 'split decision' ? 'the lenses split' : 'too few lenses'}`
    : `trust: <b>${v.level}</b>${v.reason.startsWith('capped') ? ' <span class="cap">(capped — lenses abstained)</span>' : ''}`;
  head.innerHTML = `${TRUST_GLYPH[v.level]} ${label} `
    + `<span class="tsub">· composed from ${v.nValid}/${v.nTotal} honesty lenses`
    + `${v.score != null ? ` · score ${v.score.toFixed(2)}` : ''}</span>`;
  trustLine.replaceChildren(head);
  const lenses = document.createElement('div'); lenses.className = 'lenses';
  for (const s of signals) {
    const abst = s.trust === null || s.trust === undefined;
    const pill = document.createElement('span');
    pill.className = 'lens' + (abst ? ' out' : '');
    pill.innerHTML = `${abst ? '—' : '✓'} ${s.name}${abst ? '' : ` <b>${s.trust.toFixed(2)}</b>`}`;
    pill.title = abst ? `${s.name}: this lens abstained` : `${s.name}: ${s.trust.toFixed(2)}`;
    lenses.append(pill);
  }
  trustLine.append(lenses);
}

// ----------------------------------------------------- the web phase --
// Every text search also CRAWLS: ask Commons for fresh matches, embed the
// thumbnails right here with the vision tower, rank, and show them under
// the gallery results with attribution. Toggleable; never blocks local
// results (it runs after them); scales are kept separate on purpose —
// cross-modal cosines and fused scores don't mix.
const webOn = () => localStorage.getItem('websearch') !== 'off';
let webToken = 0;
const webCache = new Map();          // thumb_url -> { emb, obj } per session

function clearWeb() {
  webToken++;
  $('webHead').classList.add('hidden');
  $('webResults').replaceChildren();
}

// failures get a way back — a dead web phase should cost one tap, not a reload
const retryBtn = query => {
  const b = Object.assign(document.createElement('button'),
    { type: 'button', className: 'retry', textContent: '↻ try again' });
  b.addEventListener('click', () => webPhase(query));
  return b;
};

async function webPhase(query) {
  if (!webOn()) { clearWeb(); return; }
  const token = ++webToken;
  const head = $('webHead'), grid = $('webResults');
  head.classList.remove('hidden');
  head.textContent = '🌐 searching the web — 5 sources in parallel…';
  grid.replaceChildren();
  try {
    const [q] = await encodeText([query]);
    const { records: recs, tried } = await discover(query, 4);
    if (token !== webToken) return;
    const oks = tried.filter(t => t.ok && t.count);
    const fails = tried.filter(t => !t.ok);
    const ledgerLine = oks.map(t => `${t.provider} ${t.count}`).join(' · ')
      + (fails.length ? `  (down: ${fails.map(t => `${t.provider}: ${t.error}`).join(' · ')})` : '');
    if (!recs.length) {          // say exactly what happened — never vanish
      const detail = tried.map(t =>
        t.ok ? `${t.provider}: 0 results` : `${t.provider}: ${t.error}`).join(' · ');
      head.replaceChildren(
        fails.length === tried.length
          ? `🌐 no web source answered — likely a network hiccup (${detail}) `
          : `🌐 web search came back empty — ${detail} `,
        retryBtn(query));
      return;
    }
    const encodeImage = await getImageEncoder(() => {}, progress);
    hideBar();
    const scored = [];
    for (let i = 0; i < recs.length; i++) {
      if (token !== webToken) return;
      head.textContent = `🌐 from the web — embedding ${i + 1}/${recs.length} in your browser… (${ledgerLine})`;
      const rec = recs[i];
      try {
        const cacheKey = `${getActiveModel()}|${rec.thumb_url}`;
        if (!webCache.has(cacheKey)) {
          const blob = await (await fetch(rec.thumb_url)).blob();
          webCache.set(cacheKey,
            { emb: await encodeImage(blob), obj: URL.createObjectURL(blob) });
        }
        const c = webCache.get(cacheKey);
        scored.push({ rec, obj: c.obj, score: dot(q, c.emb) });
      } catch (e) { console.error('web thumb failed:', rec.thumb_url, e); }
    }
    if (token !== webToken) return;
    scored.sort((a, b) => b.score - a.score);
    head.textContent = scored.length
      ? `🌐 from the web — ${scored.length} images ranked just now, locally · ${ledgerLine}`
      : '🌐 found web results but none of their thumbnails could be fetched';
    grid.replaceChildren(...scored.map(({ rec, obj }, i) => {
      const fig = document.createElement('figure');
      fig.className = 'linked';
      fig.style.animationDelay = `${i * 45}ms`;
      fig.tabIndex = 0;
      fig.title = 'open the source page — attribution and license';
      fig.append(
        Object.assign(document.createElement('img'), { src: obj, alt: rec.name }),
        Object.assign(document.createElement('figcaption'),
          { textContent: `${rec.license || 'license: see source'} · ${rec.provider} ↗` }));
      const open = () => window.open(rec.source, '_blank', 'noopener');
      fig.addEventListener('click', open);
      fig.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
      return fig;
    }));
  } catch (err) {
    console.error(err);
    if (token === webToken)          // even unexpected failures stay visible
      head.replaceChildren(`🌐 web search failed: ${err?.message ?? err} `, retryBtn(query));
  }
}

// live search: instant once the model is warm, Enter always works
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
$('q').addEventListener('input', () => {
  if (!encodeText) return;                    // never auto-download the model
  clearTimeout(debounce);
  debounce = setTimeout(search, 250);
});
for (const ex of EXAMPLES) {
  const b = Object.assign(document.createElement('button'),
    { type: 'button', textContent: ex });
  b.addEventListener('click', () => { clearImageQuery(); $('q').value = ex; search(); });
  $('chips').append(b);
}

// the web toggle rides with the example chips; the preference persists
const webBtn = Object.assign(document.createElement('button'),
  { type: 'button', className: 'toggle' });
const paintWebBtn = () => {
  webBtn.textContent = webOn() ? '🌐 web results: on' : '🌐 web results: off';
  webBtn.classList.toggle('on', webOn());
  webBtn.setAttribute('aria-pressed', String(webOn()));
};
paintWebBtn();
webBtn.addEventListener('click', () => {
  localStorage.setItem('websearch', webOn() ? 'off' : 'on');
  paintWebBtn();
  const q = $('q').value.trim();
  if (!webOn()) clearWeb();
  else if (q && encodeText && !imageQuery) webPhase(q);
});
$('chips').append(webBtn);

// the brain picker: newer models are one dropdown away. The saved choice
// restores on load but nothing downloads until the first search.
const brainSel = Object.assign(document.createElement('select'),
  { className: 'brain', id: 'brainSel' });
brainSel.setAttribute('aria-label', 'Model');
for (const [key, m] of Object.entries(MODELS)) {
  brainSel.append(Object.assign(document.createElement('option'),
    { value: key, textContent: `🧠 ${m.label} · ${m.accuracy} · ${m.size}` }));
}
const savedBrain = localStorage.getItem('brain');
if (savedBrain && MODELS[savedBrain]) setActiveModel(savedBrain);
brainSel.value = getActiveModel();
brainSel.addEventListener('change', () => switchBrain(brainSel.value));
$('chips').append(brainSel);

// ---------------------------------------------------- search by image --
// Camera button, drop anywhere, or paste a copied image — same result:
// the photo is embedded locally and ranked image-to-image.
async function imageSearch(file) {
  if (!file || !file.type.startsWith('image/')) return;
  try {
    status('loading the vision model — one time, cached after…');
    const key = getActiveModel();
    const encodeImage = await getImageEncoder(status, progress);
    hideBar();
    await ensureBrainReady();
    status('');
    const emb = await encodeImage(file);
    if (getActiveModel() !== key) {           // brain changed mid-embed
      status('brain changed while embedding — drop the photo again');
      return;
    }
    imageQuery = { emb, model: key };
    $('imgChipThumb').src = URL.createObjectURL(file);
    $('imgChip').classList.remove('hidden');
    $('q').value = '';
    $('q').placeholder = 'searching by your image — ✕ to clear';
    search();
  } catch (err) {
    console.error(err);
    hideBar();
    status('could not run the vision model in this browser — try Chrome, Edge or Safari 17+');
  }
}

function clearImageQuery() {
  imageQuery = null;
  $('imgChip').classList.add('hidden');
  $('q').placeholder = 'Try “a fluffy animal” — or drop a photo anywhere';
}

$('cam').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', () => imageSearch($('file').files[0]));
$('imgChipClear').addEventListener('click', () => { clearImageQuery(); if ($('q').value.trim()) search(); });

addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('dragging'); });
addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
addEventListener('drop', e => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  imageSearch(e.dataTransfer.files[0]);
});
addEventListener('paste', e => {
  const f = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (f) imageSearch(f.getAsFile());
});

// a shared ?q= link searches immediately — that's the point of this page
const q0 = new URLSearchParams(location.search).get('q');
if (q0) { $('q').value = q0; search(); }
$('q').addEventListener('change', () => {
  const p = new URLSearchParams(location.search);
  const q = $('q').value.trim();
  q ? p.set('q', q) : p.delete('q');
  history.replaceState(null, '', p.size ? '?' + p : location.pathname);
});
