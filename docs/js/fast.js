// The front page — one box, results on the first keystroke.
//
// Three layers, fastest first; each keystroke takes the first that can answer:
//   1. cache     a query CLIP already read → its vector (LRU, ~0 ms)
//   2. preview   no model yet → precomputed word scores (lexicon.json), < 0.1 ms
//   3. CLIP      the real text encoder, in a Web Worker (fast-worker.js),
//                latest-wins: a burst of keystrokes costs ONE extra encode
// Ranking itself is 14 dot products over a packed Float32Array (instant.js).
// The model starts downloading the moment the page opens, not on Enter.
import { imageVectors, scoreAll, topK, loadLexicon, preview, LRU } from './instant.js';

const $ = id => document.getElementById(id);
const K = 8;
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe', 'something delicious',
  'outer space', 'water in nature', 'a bird that talks', 'two wheels', 'snowy peaks'];

// page state — declared first: worker messages can arrive while the module
// is still awaiting db.json below
let textState = 'loading', loadPct = 0, encEMA = 0;  // CLIP text tower: 'loading' | 'ready' | 'failed'
let current = '', shownExact = false, last = null;   // last = what the status line describes
let photo = null, photoSeq = 0, ex = 0;
let inflight = false, queued = null;

// ------------------------------------------------------------ engine --
// CLIP in a module worker; if the browser can't run one, the same encoders
// on the main thread. Either way: call('text'|'image', x) → { vec, ms }.
const bar = $('bar');
const files = new Map();
let barTimer = 0;
function onProgress({ file, loaded, total }) {
  files.set(file, [loaded, total]);
  let l = 0, t = 0;
  for (const [a, b] of files.values()) { l += a; t += b; }
  loadPct = t ? Math.round(100 * l / t) : 0;
  bar.style.opacity = '1';
  bar.style.transform = `scaleX(${t ? l / t : 0})`;
  clearTimeout(barTimer);
  if (l >= t) barTimer = setTimeout(() => { bar.style.opacity = '0'; }, 250);
  if (textState === 'loading' && current && !shownExact) paintStatus();
}

function onReady(what) {
  if (what !== 'text') return;
  textState = 'ready';
  if (current && !shownExact) requestExact(current);
  else if (current) paintStatus();
}
function onFailed(what) {
  if (what !== 'text') return;
  textState = 'failed';
  if (current) paintStatus();
}

const engine = (() => {
  let worker = null, seq = 0;
  const pending = new Map();
  async function local(type, x) {
    const C = await import('./clip.js');
    const enc = await (type === 'image' ? C.getImageEncoder : C.getTextEncoder)(() => {}, onProgress);
    if (type === 'warm') return;
    const t0 = performance.now();
    const vec = type === 'text' ? (await enc([x]))[0] : await enc(x);
    return { vec: Float32Array.from(vec), ms: performance.now() - t0 };
  }
  const localWarm = what => local(what, null).then(() => onReady(what), () => onFailed(what));
  try {
    worker = new Worker(new URL('./fast-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data: m }) => {
      if (m.type === 'progress') return onProgress(m);
      if (m.type === 'ready') { pending.delete(`w${m.what}`); return onReady(m.what); }
      if (m.type === 'failed') { pending.delete(`w${m.what}`); return onFailed(m.what); }
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.ok ? p.resolve(m) : p.reject(new Error(m.error));
    };
    worker.onerror = e => {                       // no module workers here → main thread
      e.preventDefault();
      worker.terminate(); worker = null;
      for (const p of pending.values()) {
        if (p.type === 'warm') localWarm(p.x);
        else local(p.type, p.x).then(p.resolve, p.reject);
      }
      pending.clear();
    };
  } catch { worker = null; }
  return {
    warm(what) {
      if (!worker) return localWarm(what);
      pending.set(`w${what}`, { type: 'warm', x: what, resolve() {}, reject() {} });
      worker.postMessage({ type: 'warm', what });
    },
    call(type, x) {
      if (!worker) return local(type, x);
      const id = ++seq;
      return new Promise((resolve, reject) => {
        pending.set(id, { type, x, resolve, reject });
        worker.postMessage(type === 'text' ? { id, type, text: x } : { id, type, blob: x });
      });
    },
  };
})();
// start CLIP now (unless the visitor asked the browser to save data)
const saveData = navigator.connection?.saveData;
if (!saveData) engine.warm('text');

// ------------------------------------------------------------- data --
const [db, lexJson] = await Promise.all([
  fetch('db.json').then(r => r.json()),
  fetch('lexicon.json').then(r => (r.ok ? r.json() : null)).catch(() => null),
]);
const items = db.items;
const IMG = imageVectors(items);
const lex = loadLexicon(lexJson, items);          // null if stale → just wait for CLIP
const BIAS = lex?.bias ?? null;
const cache = new LRU(256);

// one figure per image, built once; re-ranking only MOVES these nodes, so
// an image never reloads or re-decodes between keystrokes
const figs = items.map((it, i) => {
  const fig = document.createElement('figure');
  fig.dataset.key = i;
  fig.tabIndex = 0;
  fig.title = 'more like this';
  const img = new Image();
  img.src = it.file; img.alt = it.caption; img.decoding = 'async';
  const cap = document.createElement('figcaption');
  cap.textContent = it.tags.slice(0, 3).join(' · ');
  fig.append(img, cap);
  fig.addEventListener('click', () => moreLike(i));
  fig.addEventListener('keydown', e => { if (e.key === 'Enter') moreLike(i); });
  return fig;
});

// ------------------------------------------------------------ render --
const results = $('results'), statusEl = $('status');
const HINT = statusEl.textContent;                  // the empty page's one line
const fmt = ms => (ms < 0.1 ? '<0.1 ms' : ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`);

function place(order) {
  const before = new Map([...results.children].map(el => [el, el.getBoundingClientRect()]));
  results.replaceChildren(...order.map(i => figs[i]));
  if (!matchMedia('(prefers-reduced-motion: no-preference)').matches) return;
  for (const el of results.children) {
    const r0 = before.get(el);
    if (!r0) { el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 140 }); continue; }
    const r1 = el.getBoundingClientRect(), dx = r0.left - r1.left, dy = r0.top - r1.top;
    if (dx || dy) el.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: 'none' }],
      { duration: 170, easing: 'cubic-bezier(.2,.7,.2,1)' });
  }
}

// info.ms = time spent scoring; "ranked in" = scoring + top-k, never DOM work
function show(scores, info, skip = -1) {
  const t0 = performance.now();
  const order = topK(scores, K, skip);
  const ms = info.ms + performance.now() - t0;
  place(order);
  document.body.classList.add('searched');
  last = { ...info, ms, q: current };              // q: which query the grid shows
  paintStatus();
}

function paintStatus() {
  if (!last) { statusEl.textContent = HINT; return; }
  const n = items.length, head = `${n} images ranked in ${fmt(last.ms)}`;
  let tail = '';
  if (last.kind === 'clip') tail = `CLIP read your words in ${fmt(last.enc)}`;
  else if (last.kind === 'cached') tail = 'already read — cached';
  else if (last.kind === 'similar') tail = 'more like this — no model needed';
  else if (last.kind === 'photo') tail = `CLIP looked at your photo in ${fmt(last.enc)}`;
  else if (last.kind === 'preview') {
    tail = `instant preview of “${last.words.join(' ')}”`;
    if (textState === 'loading') tail += ` · CLIP loading${loadPct ? ` ${loadPct}%` : '…'}`;
    else if (textState === 'failed') tail += ' · CLIP could not load here';
  }
  statusEl.textContent = `${head} · ${tail}`;
}

function clearAll() {
  results.replaceChildren();
  document.body.classList.remove('searched');
  last = null; current = ''; shownExact = false;
  paintStatus();
}

// ------------------------------------------------------------ search --
function requestExact(q) {
  if (textState === 'failed' || cache.get(q)) return;
  if (inflight) { queued = q; return; }
  inflight = true;
  engine.call('text', q).then(({ vec, ms }) => {
    cache.set(q, vec);
    encEMA = encEMA ? 0.7 * encEMA + 0.3 * ms : ms;
    if (q === current && !photo) {
      const t0 = performance.now();
      const s = scoreAll(IMG, vec, BIAS);
      shownExact = true;
      show(s, { kind: 'clip', enc: ms, ms: performance.now() - t0 });
      syncURL(q);
    }
  }, err => { console.error(err); }).finally(() => {
    inflight = false;
    if (queued !== null) { const nx = queued; queued = null; if (nx === current) requestExact(nx); }
  });
}

function onType() {
  const raw = $('q').value, q = raw.trim();
  if (photo) clearPhoto();
  current = q; shownExact = false;
  if (!q) { queued = null; clearAll(); syncURL(''); return; }
  const t0 = performance.now();
  const hit = cache.get(q);
  if (hit) {
    shownExact = true;
    show(scoreAll(IMG, hit, BIAS), { kind: 'cached', ms: performance.now() - t0 });
    syncURL(q);
    return;
  }
  // a warm, fast CLIP answers within a frame — skip the preview so the grid
  // re-orders once, not twice. Cold or slow CLIP: the preview fills the gap.
  const fastCLIP = textState === 'ready' && encEMA && encEMA < 40;
  if (!fastCLIP) {
    const p = preview(lex, raw);
    if (p) show(p.scores, { kind: 'preview', words: p.words, ms: performance.now() - t0 });
    else if (textState === 'loading') {
      statusEl.textContent = `CLIP loading${loadPct ? ` ${loadPct}%` : '…'} — results appear the moment it’s ready`;
    }
  }
  if (saveData && textState === 'loading' && !engine.started) { engine.started = true; engine.warm('text'); }
  requestExact(q);
}

// ------------------------------------------------- search by an image --
// click a result → "more like this" from its stored image_emb (no model);
// drop / paste / 📷 a photo → CLIP's vision tower, in the worker.
function setPhoto(src, label) {
  photo = { src };
  $('chipImg').src = src; $('chipImg').alt = label;
  $('chip').hidden = false;
  $('q').value = ''; current = '';
  $('q').placeholder = 'type to search by words instead';
}
function clearPhoto() {
  photo = null; photoSeq++;
  $('chip').hidden = true;
  $('q').placeholder = EXAMPLES[ex];
}

function moreLike(i) {
  setPhoto(items[i].file, items[i].caption);
  const t0 = performance.now();
  show(scoreAll(IMG, items[i].image_emb), { kind: 'similar', ms: performance.now() - t0 }, i);
  syncURL('');
}

async function photoSearch(file) {
  if (!file || !file.type?.startsWith('image/')) return;
  setPhoto(URL.createObjectURL(file), 'your photo');
  const mine = ++photoSeq;
  document.body.classList.add('searched');
  statusEl.textContent = 'loading CLIP’s vision tower — one time, cached after…';
  try {
    const { vec, ms } = await engine.call('image', file);
    if (mine !== photoSeq) return;
    const t0 = performance.now();
    show(scoreAll(IMG, vec), { kind: 'photo', enc: ms, ms: performance.now() - t0 });
  } catch (err) {
    console.error(err);
    if (mine === photoSeq) statusEl.textContent = 'could not run the vision model in this browser';
  }
}

// ------------------------------------------------------------- wiring --
$('q').addEventListener('input', onType);
$('q').addEventListener('keydown', e => { if (e.key === 'Escape') { $('q').value = ''; clearPhoto(); onType(); } });
$('chipX').addEventListener('click', () => { clearPhoto(); clearAll(); $('q').focus(); });
$('cam').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', () => { photoSearch($('file').files[0]); $('file').value = ''; });
addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement !== $('q')) { e.preventDefault(); $('q').focus(); }
});
addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('drop'); });
addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('drop'); });
addEventListener('drop', e => {
  e.preventDefault(); document.body.classList.remove('drop');
  photoSearch(e.dataTransfer.files[0]);
});
addEventListener('paste', e => {
  const f = [...(e.clipboardData?.items || [])].find(x => x.type.startsWith('image/'));
  if (f) photoSearch(f.getAsFile());
});

// rotating placeholder: what to try, without a single extra element
setInterval(() => {
  if ($('q').value || photo) return;
  ex = (ex + 1) % EXAMPLES.length;
  $('q').placeholder = EXAMPLES[ex];
}, 2600);

let urlTimer = 0;
function syncURL(q) {
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    history.replaceState(null, '', q ? `?q=${encodeURIComponent(q)}` : location.pathname);
  }, 300);
}

// a shared ?q= link searches immediately
const q0 = new URLSearchParams(location.search).get('q');
if (q0) { $('q').value = q0; onType(); }

// for the E2E test and the curious
window.__fast = { get state() { return { textState, loadPct, encEMA, last, current, lex: !!lex }; } };
