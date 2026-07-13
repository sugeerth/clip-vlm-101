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

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];
const TOP_K = 8;

const DB = await (await fetch('db.json')).json();

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
function show(entries, note = '') {
  document.body.classList.add('searched');
  $('results').replaceChildren(...entries.map(({ item }, i) => {
    const fig = document.createElement('figure');
    fig.style.animationDelay = `${i * 45}ms`;
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
      show(out.ranked);
      showTrace(out);
      webPhase(query);               // fire-and-forget: local results never wait
    }
  } catch (err) {
    console.error(err);
    hideBar();
    hideTrace();
    clearWeb();
    const hits = keywordResults(query);
    show(hits, hits.length ? 'model unavailable here — showing keyword matches'
                           : 'model unavailable and no keyword matches');
  }
  searching = false;
  if (rerun) { rerun = false; search(); }
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
