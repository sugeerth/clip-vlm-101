// The page itself: wires the mirrored pipeline modules to the DOM.
//   templates.js (prompts) · clip.js (encoders) · rank.js (dot products)
//   labels.js (multi-label) · agent.js (propose ⇄ critique) · viz.js (pictures)
// Sections below match the page: theme, matrix, gallery, map, search, lab.
import { VOCAB, tagPrompts, NEUTRAL_PROMPT } from './templates.js';
import { dot, rank, topTags, fuse } from './rank.js';
import { labelProbs } from './labels.js';
import { runAgent, MIN_ALIGNED, MIN_CONFIDENT } from './agent.js';
import { recommend, userVector } from './recsys.js';
import { flip, tweenNumber, motionOK } from './motion.js';
import { startTour } from './tour.js';
import { getTextEncoder, getImageEncoder } from './clip.js';
import { drawMatrix, drawMap, markUpload, project, stripRow, redrawStrips } from './viz.js';

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];

const DB = await (await fetch('db.json')).json();

// ------------------------------------------------------------------ theme --
// prefers-color-scheme sets the default; the toggle stamps data-theme on
// <html> (persisted), which overrides it. Canvas/heatmap colors are painted
// in JS, so a flip re-colorizes them too.
let recolorMatrix = () => {};
$('themeToggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
  requestAnimationFrame(() => { recolorMatrix(); redrawStrips(); });
});

// --------------------------------------------------- contrastive matrix --
// The readout keeps its elements so the value COUNTS toward each new cell
// as you sweep the grid, instead of flickering.
const mxVal = Object.assign(document.createElement('div'), { className: 'val', textContent: '0.000' });
const mxPair = Object.assign(document.createElement('div'), { className: 'pair' });
const mxNote = Object.assign(document.createElement('div'), { className: 'pair' });
recolorMatrix = drawMatrix($('matrix'), DB.items, (imgItem, capItem, score, isDiag) => {
  const readout = $('matrixReadout');
  if (!mxVal.isConnected) readout.replaceChildren(mxVal, mxPair, mxNote);
  tweenNumber(mxVal, score);
  mxPair.textContent = `image “${imgItem.tags[0]}” × caption “${capItem.caption}”`;
  mxNote.textContent = isDiag
    ? 'This image scored against its own caption — the diagonal the training objective brightens.'
    : 'An off-diagonal pair — contrastive training pushes these apart.';
});

// Table twin of the heatmap (built lazily on first open): every value the
// colors encode, readable without color at all.
$('matrixTable').addEventListener('toggle', () => {
  const body = $('matrixTable').querySelector('.table-scroll');
  if (!$('matrixTable').open || body.children.length) return;
  const table = document.createElement('table');
  const head = table.insertRow();
  head.append(Object.assign(document.createElement('th'), { textContent: 'image \\ caption' }),
    ...DB.items.map((_, j) => Object.assign(document.createElement('th'), { textContent: j + 1 })));
  DB.items.forEach((a, i) => {
    const tr = table.insertRow();
    tr.append(Object.assign(document.createElement('th'), { textContent: `${i + 1} · ${a.tags[0]}` }),
      ...DB.items.map(b => Object.assign(document.createElement('td'),
        { textContent: dot(a.image_emb, b.text_emb).toFixed(3) })));
  });
  body.append(table);
}, { once: false });

// ---------------------------------------------------------------- gallery --
for (const item of DB.items) {
  const fig = document.createElement('figure');
  fig.tabIndex = 0;
  const img = Object.assign(document.createElement('img'),
    { src: item.file, alt: item.caption, loading: 'lazy' });
  const cap = document.createElement('figcaption');
  cap.append(...item.tags.map(t =>
    Object.assign(document.createElement('span'), { className: 'pill', textContent: t })));
  fig.append(img, cap);
  const open = () => { setSubject(galSubject(item)); location.hash = '#lab'; };
  fig.addEventListener('click', open);
  fig.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
  $('gallery').append(fig);
}

// -------------------------------------------------------------------- map --
// Nearest neighbours in the FULL 512-d space (not the 2-D squash) — the
// hover lines show what PCA had to distort to fit the screen.
const nn = new Map(DB.items.map(a => [a,
  DB.items.filter(b => b !== a)
    .map(b => ({ b, s: dot(a.image_emb, b.image_emb) }))
    .sort((x, y) => y.s - x.s).slice(0, 3).map(x => x.b)]));

const tip = document.createElement('div');
tip.className = 'map-tip hidden';
$('mapWrap').append(tip);

if (DB.pca) drawMap($('map'), DB.items, {
  nnOf: item => nn.get(item),
  onPick: item => { setSubject(galSubject(item)); location.hash = '#lab'; },
  onHover: (item, cx, cy) => {
    if (!item) { tip.classList.add('hidden'); return; }
    tip.classList.remove('hidden');
    tip.replaceChildren(
      Object.assign(document.createElement('b'), { textContent: item.caption }),
      Object.assign(document.createElement('span'), {
        className: 't2', textContent: 'lines → its 3 nearest neighbours in 512-d · click to open in the Lab' }));
    const r = $('mapWrap').getBoundingClientRect();
    tip.style.left = Math.min(cx - r.left + 14, r.width - 250) + 'px';
    tip.style.top = (cy - r.top + 14) + 'px';
  },
});

// -------------------------------------------------------------- shared UI --
// Progress bars for model downloads (transformers.js progress events).
const progressTo = (track, fill) => p => {
  if (p.status === 'progress' && p.total) {
    $(track).classList.remove('hidden');
    $(fill).style.width = Math.round(100 * p.loaded / p.total) + '%';
  }
};
const searchProgress = progressTo('loadTrack', 'loadFill');
const labProgress = progressTo('labTrack', 'labFill');

// Memoized text encoding — the agent re-uses each template's 60 prompts.
let encodeTextRaw = null;
const textCache = new Map();
async function ensureTextEncoder(onStatus, onProgress) {
  encodeTextRaw ??= await getTextEncoder(onStatus, onProgress);
  return async texts => {
    const missing = texts.filter(t => !textCache.has(t));
    if (missing.length) {
      const embs = await encodeTextRaw(missing);
      missing.forEach((t, i) => textCache.set(t, embs[i]));
    }
    return texts.map(t => textCache.get(t));
  };
}

// Ranked rows with thumbnails; bars are relative to the best hit and the
// exact score is printed beside each bar (never inside it). Rows PERSIST
// across renders keyed by file, so re-ranking plays as motion (FLIP), bars
// glide to their new widths, and scores count toward their new values.
// Entries may carry a {looks, means, active} breakdown — the two component
// similarities every fused score is made of — rendered as mini meters.
function makeResultRow(item) {
  const row = document.createElement('div'); row.className = 'result';
  row.dataset.key = item.file;
  const img = Object.assign(document.createElement('img'), { src: item.file, alt: item.caption });
  const meta = document.createElement('div'); meta.className = 'meta';
  const name = document.createElement('div'); name.className = 'name'; name.textContent = item.caption;
  const track = document.createElement('div'); track.className = 'bar-track';
  track.append(Object.assign(document.createElement('div'), { className: 'bar-fill' }));
  const parts = document.createElement('div'); parts.className = 'parts hidden';
  for (const which of ['looks', 'means']) {
    const p = document.createElement('span'); p.className = `part ${which}`;
    p.append(
      Object.assign(document.createElement('span'), { className: 'plbl', textContent: which }),
      Object.assign(document.createElement('span'), { className: 'pmeter' }),
      Object.assign(document.createElement('span'), { className: 'pval', textContent: '0.000' }));
    p.querySelector('.pmeter').append(
      Object.assign(document.createElement('span'), { className: 'pfill' }));
    parts.append(p);
  }
  meta.append(name, track, parts);
  const s = document.createElement('div'); s.className = 'score'; s.textContent = '0.000';
  row.append(img, meta, s);
  return row;
}

function renderResults(el, entries) {
  const rows = (el._rows ??= new Map());
  flip(el, () => {
    const keep = new Set();
    const top = Math.max(...entries.map(e => e.score), 1e-6);
    for (const e of entries) {
      keep.add(e.item.file);
      let row = rows.get(e.item.file);
      if (!row) rows.set(e.item.file, row = makeResultRow(e.item));
      el.append(row);                                  // appends in rank order
      row.querySelector('.bar-fill').style.width =
        Math.max(2, Math.min(100, 100 * e.score / top)) + '%';
      tweenNumber(row.querySelector('.score'), e.score);
      const parts = row.querySelector('.parts');
      parts.classList.toggle('hidden', e.looks === undefined);
      if (e.looks !== undefined) {
        const span = Math.max(e.looks, e.means, 1e-6);
        for (const [which, val] of [['looks', e.looks], ['means', e.means]]) {
          const p = parts.querySelector(`.${which}`);
          p.classList.toggle('dim', e.active !== 'both' && e.active !== which);
          p.querySelector('.pfill').style.width =
            Math.max(2, Math.min(100, 100 * Math.max(0, val) / span)) + '%';
          tweenNumber(p.querySelector('.pval'), val);
        }
      }
    }
    for (const [key, row] of rows) if (!keep.has(key)) { rows.delete(key); row.remove(); }
  });
}

// ----------------------------------------------------------------- search --
const qInput = $('q'), searchStatus = $('searchStatus');
const MODE_NOTES = {
  fused: 'average of visual + semantic similarity — one vector, both signals',
  image: 'query vs image_emb — matches what the pictures LOOK like',
  text: 'query vs text_emb — matches what the captions/tags MEAN',
};
let searching = false;

async function runSearch() {
  const query = qInput.value.trim();
  if (!query || searching) return;
  searching = true;
  const mode = document.querySelector('input[name=mode]:checked').value;
  $('results').classList.add('busy');           // hold previous render, dimmed
  try {
    const encode = await ensureTextEncoder(t => searchStatus.textContent = t, searchProgress);
    $('loadTrack').classList.add('hidden');
    searchStatus.textContent = 'Embedding your query…';
    const [q] = await encode([query]);
    // every score decomposes into the two similarities it is made of:
    // looks = q · image_emb, means = q · text_emb (fused = their average)
    const active = { fused: 'both', image: 'looks', text: 'means' }[mode];
    renderResults($('results'), rank(DB.items, q, mode).map(({ item, score }) => ({
      item, score, active,
      looks: dot(item.image_emb, q), means: dot(item.text_emb, q),
    })));
    searchStatus.textContent =
      `“${query}” — ${mode} similarity, top 5 of ${DB.items.length}. Same math as search.py.`;
    syncURL();
  } catch (err) {
    console.error(err);
    $('loadTrack').classList.add('hidden');
    // graceful fallback: substring match over tags + captions
    const words = query.toLowerCase().split(/\s+/);
    const keyword = DB.items.map(item => ({
      item,
      score: words.filter(w => item.tags.some(t => t.includes(w)) || item.caption.includes(w)).length / 4,
    })).sort((a, b) => b.score - a.score).slice(0, 5);
    renderResults($('results'), keyword);
    searchStatus.textContent =
      'Model failed to load here, showing keyword matches instead — run search.py locally for the real thing.';
  }
  $('results').classList.remove('busy');
  searching = false;
  // if the query changed while we were busy (e.g. a second chip click), run it now
  const now = qInput.value.trim();
  if (now && now !== query) runSearch();
}

qInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', () => {
  $('modeNote').textContent = MODE_NOTES[r.value];
  syncURL();
  runSearch();
}));
for (const ex of EXAMPLES) {
  const b = Object.assign(document.createElement('button'), { className: 'chip', textContent: ex });
  b.addEventListener('click', () => { qInput.value = ex; runSearch(); });
  $('examples').append(b);
}

// ------------------------------------------------------------------- lab --
// One "subject" at a time: a gallery image (stored vectors, instant) or an
// upload (embedded in-browser). Everything below the pickers reacts to it.
const labStatus = $('labStatus');
let subject = null;        // { imageEmb, src, caption, tags, isUpload }
let subjProbs = null;      // per-vocab-tag probabilities for the slider
let agentBusy = false;

const galSubject = item => ({
  imageEmb: item.image_emb, src: item.file,
  caption: item.caption, tags: item.tags, isUpload: false,
});

// pickers: one thumb per gallery image + the upload drop (already in HTML)
for (const item of DB.items) {
  const t = Object.assign(document.createElement('img'),
    { src: item.file, alt: item.caption, loading: 'lazy', tabIndex: 0 });
  t.addEventListener('click', () => setSubject(galSubject(item)));
  t.addEventListener('keydown', e => { if (e.key === 'Enter') setSubject(galSubject(item)); });
  $('labPickers').insertBefore(t, $('drop'));
}

const drop = $('drop'), fileInput = $('file');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('armed'); });
drop.addEventListener('dragleave', () => drop.classList.remove('armed'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('armed');
  if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
  try {
    labStatus.textContent = '';
    const encodeImage = await getImageEncoder(t => labStatus.textContent = t, labProgress);
    $('labTrack').classList.add('hidden');
    labStatus.textContent = 'Embedding your image (in your browser — it never leaves your machine)…';
    const imageEmb = await encodeImage(file);
    labStatus.textContent = '';
    setSubject({ imageEmb, src: URL.createObjectURL(file), caption: file.name, tags: null, isUpload: true });
    if (DB.pca) markUpload($('map'), project(imageEmb, DB.pca));   // drop it on the map too
  } catch (err) {
    console.error(err);
    $('labTrack').classList.add('hidden');
    labStatus.textContent =
      'Could not run the vision model in this browser — try Chrome/Edge/Safari 17+, or run features.py locally.';
  }
}

function setSubject(s) {
  subject = s; subjProbs = null;
  document.querySelectorAll('#labPickers img').forEach(img =>
    img.classList.toggle('selected', img.src.endsWith(s.src)));
  document.querySelectorAll('#gallery figure').forEach((fig, i) =>
    fig.classList.toggle('selected', DB.items[i].file === s.src));

  $('labSubject').classList.remove('hidden');
  $('subjectImg').src = s.src;
  const capEl = $('subjectCaption');
  capEl.replaceChildren();
  if (s.tags) {
    capEl.append(`stored caption: “${s.caption}” · meta tags: `,
      ...s.tags.map(t => Object.assign(document.createElement('span'), { className: 'pill', textContent: t })));
  } else {
    capEl.textContent = `“${s.caption}” — embedded just now, in your browser`;
  }
  $('subjectStrips').replaceChildren(stripRow('image_emb', s.imageEmb));

  $('labCols').classList.remove('hidden');
  $('agentPanel').classList.remove('hidden');
  $('rounds').replaceChildren(); $('verdict').textContent = ''; $('verdict').className = 'verdict';
  $('recordOut').replaceChildren();
  if (motionOK())   // crossfade the panel to the new subject
    for (const id of ['labSubject', 'labCols'])
      $(id).animate([{ opacity: 0.2 }, { opacity: 1 }], { duration: 260, easing: 'ease' });
  computeTagsAndLabels();
  syncURL();
}

// ------------------------------------------------------------ deep links --
// The page's state fits in a URL: likes, query, mode, threshold, Lab pick.
// replaceState keeps the address bar current without touching history.
function syncURL() {
  const p = new URLSearchParams();
  if (likes.size) p.set('likes', [...likes].map(it => DB.items.indexOf(it)).join(','));
  if (qInput.value.trim()) p.set('q', qInput.value.trim());
  const mode = document.querySelector('input[name=mode]:checked').value;
  if (mode !== 'fused') p.set('mode', mode);
  if (+$('thresh').value !== 0.5) p.set('t', $('thresh').value);
  if (subject && !subject.isUpload) {
    p.set('pick', DB.items.findIndex(it => it.file === subject.src));
  }
  history.replaceState(null, '', p.size ? '?' + p : location.pathname);
}

function restoreFromURL() {
  const p = new URLSearchParams(location.search);
  if (p.get('q')) qInput.value = p.get('q');           // filled, not auto-run
  if (p.get('mode')) {
    const r = document.querySelector(`input[name=mode][value="${p.get('mode')}"]`);
    if (r) { r.checked = true; $('modeNote').textContent = MODE_NOTES[r.value]; }
  }
  if (p.get('t')) { $('thresh').value = p.get('t'); $('threshVal').textContent = (+p.get('t')).toFixed(2); }
  if (p.get('likes')) {
    for (const idx of p.get('likes').split(',')) {
      const item = DB.items[+idx], btn = $('likePickers').children[+idx];
      if (!item || !btn) continue;
      likes.add(item);
      btn.classList.add('on'); btn.setAttribute('aria-pressed', 'true');
    }
    renderRecs();
  }
  const pick = p.has('pick') ? DB.items[+p.get('pick')] : null;
  if (pick) setSubject(galSubject(pick));
}

// ------------------------------------------------------- the guided tour --
$('tourBtn').addEventListener('click', () => startTour([
  { sel: '#matrix', title: 'The idea', text: 'Every cell is one dot product between an image vector and a caption vector. The dark diagonal is CLIP’s training objective, visible in real data.' },
  { sel: '#gallery', title: 'The database', text: 'Fourteen images, embedded and auto-tagged offline. Click any of them to open it in the Lab.' },
  { sel: '#map', title: 'The embedding space', text: '512 dimensions squashed to two with PCA. Hover an image to see its true nearest neighbours; similar things live close together.' },
  { sel: '#q', title: 'Search', text: 'Type anything. Your sentence becomes a vector and every image is one dot product away. (First search downloads the model — ~65 MB, once.)' },
  { sel: '#labPickers', title: 'The Lab', text: 'Pick an image to compare multi-class vs dynamic multi-label tags, run the self-critiquing embedding agent, and inspect the raw vectors.' },
  { sel: '#likePickers', title: 'Recommend', text: 'Tap hearts and a two-tower recommender ranks everything instantly — no model needed, because the item embeddings were computed offline.',
    prep: () => { if (!likes.size) for (const i of [4, 5]) $('likePickers').children[i].click(); } },
  { sel: '#optimize .opt-grid', title: 'Optimize it', text: 'Four levers — ensembling, batching, quantization, precomputation — each implemented in the Python repo, each measurable with eval.py.' },
]));

// -------------------------------------------------------- show the math --
// Details on demand: formulas, component meters and raw vectors stay hidden
// until asked for — the default view is the clean overview.
for (const b of document.querySelectorAll('.math-toggle')) {
  b.addEventListener('click', () => {
    const on = b.closest('.card').classList.toggle('show-math');
    b.textContent = on ? 'hide the math' : 'show the math';
    b.setAttribute('aria-pressed', String(on));
  });
}

// ------------------------------------------------- "you are here" in nav --
// Purely additive polish: worst case is simply no highlight.
const navFor = new Map([...document.querySelectorAll('.nav-links a')]
  .map(a => [a.getAttribute('href').slice(1), a]));
const spy = new IntersectionObserver(entries => {
  for (const e of entries) {
    const a = navFor.get(e.target.id === 'gallery-sec' ? 'idea'
      : e.target.id === 'map-sec' ? 'map' : e.target.id);
    if (a && e.isIntersecting) {
      document.querySelectorAll('.nav-links a.here').forEach(x => x.classList.remove('here'));
      a.classList.add('here');
    }
  }
}, { rootMargin: '-20% 0px -60% 0px' });
document.querySelectorAll('section[id]').forEach(s => spy.observe(s));

// ---- multi-class vs multi-label columns -----------------------------------
async function computeTagsAndLabels() {
  const s = subject;
  try {
    const encode = await ensureTextEncoder(t => labStatus.textContent = t, labProgress);
    $('labTrack').classList.add('hidden');
    labStatus.textContent = textCache.size ? '' : 'Embedding the tag vocabulary (once)…';
    const tagEmbs = await encode(tagPrompts());
    const [neutralEmb] = await encode([NEUTRAL_PROMPT]);
    if (subject !== s) return;                       // subject changed mid-flight
    labStatus.textContent = '';

    // multi-class: exactly 5, always (tagger.py)
    const top5 = topTags(s.imageEmb, tagEmbs, VOCAB);
    const maxScore = top5[0].score || 1e-6;
    $('topkList').replaceChildren(...top5.map(({ tag, score }) =>
      meterRow(tag, score / maxScore, score.toFixed(3), true, null)));

    // multi-label: an independent probability per tag (labels.js)
    subjProbs = labelProbs(s.imageEmb, tagEmbs, neutralEmb)
      .map((prob, i) => ({ tag: VOCAB[i], prob }))
      .sort((a, b) => b.prob - a.prob);
    renderLabels();
  } catch (err) {
    console.error(err);
    $('labTrack').classList.add('hidden');
    labStatus.textContent = 'Could not load the text encoder here — run labels.py locally for this part.';
  }
}

function meterRow(tag, frac, valText, on, threshold) {
  const row = document.createElement('div'); row.className = 'meter-row';
  const t = document.createElement('span'); t.className = 'tag' + (on ? ' on' : ''); t.textContent = tag;
  const meter = document.createElement('div'); meter.className = 'meter';
  const fill = document.createElement('div'); fill.className = 'fill' + (frac > 0.97 ? ' round' : '');
  fill.style.width = (100 * frac).toFixed(1) + '%';
  meter.append(fill);
  if (threshold !== null && threshold !== undefined) {
    const th = document.createElement('div'); th.className = 'thresh';
    th.style.left = (100 * threshold).toFixed(1) + '%';
    meter.append(th);
  }
  const v = document.createElement('span'); v.className = 'val'; v.textContent = valText;
  row.append(t, meter, v);
  return row;
}

function renderLabels() {
  if (!subjProbs) return;
  const threshold = parseFloat($('thresh').value);
  $('threshVal').textContent = threshold.toFixed(2);
  const shown = subjProbs.slice(0, 10);            // top 10 of the vocabulary
  $('labelList').replaceChildren(...shown.map(({ tag, prob }) =>
    meterRow(tag, prob, prob.toFixed(2), prob >= threshold, threshold)));
  const n = subjProbs.filter(x => x.prob >= threshold).length;
  $('labelSummary').textContent =
    `${n} label${n === 1 ? '' : 's'} above ${threshold.toFixed(2)} — drag the threshold and watch ` +
    'the set resize. Multi-class always answers 5; multi-label answers what the image holds.';
}
$('thresh').addEventListener('input', () => { renderLabels(); syncURL(); });

// -------------------------------------------------------------- recommend --
// The user tower, served live: likes → mean vector → one dot product per
// item. No model, no download — this is what two-tower serving looks like.
const likes = new Set();
for (const item of DB.items) {
  const b = Object.assign(document.createElement('button'),
    { className: 'like', type: 'button', title: item.caption });
  b.setAttribute('aria-pressed', 'false');
  const img = Object.assign(document.createElement('img'),
    { src: item.file, alt: item.caption, loading: 'lazy' });
  const heart = Object.assign(document.createElement('span'),
    { className: 'heart', textContent: '♥', ariaHidden: 'true' });
  b.append(img, heart);
  b.addEventListener('click', () => {
    likes.has(item) ? likes.delete(item) : likes.add(item);
    b.classList.toggle('on', likes.has(item));
    b.setAttribute('aria-pressed', String(likes.has(item)));
    renderRecs();
    syncURL();
  });
  $('likePickers').append(b);
}

function renderRecs() {
  if (!likes.size) {
    ($('recResults')._rows ??= new Map()).clear();
    $('recResults').replaceChildren();
    $('userVec').replaceChildren();
    $('recStatus').textContent = 'Tap two or three images you like — recommendations appear instantly.';
    $('recExplain').textContent = '';
    return;
  }
  // THE user: one 1024-d vector, shown as it changes with every tap.
  const u = userVector([...likes].map(it => fuse(it.image_emb, it.text_emb)));
  $('userVec').replaceChildren(stripRow('user_vec', u));
  // each recommendation's score splits into the two halves of the fused
  // vector: looks (image side) + means (text side) — they literally add up.
  const uImg = u.slice(0, 512), uTxt = u.slice(512);
  renderResults($('recResults'), recommend(DB.items, [...likes], 5).map(({ item, score }) => ({
    item, score, active: 'both',
    looks: dot(item.image_emb, uImg) / Math.SQRT2,
    means: dot(item.text_emb, uTxt) / Math.SQRT2,
  })));
  $('recStatus').textContent =
    `user_vec = unit(mean of ${likes.size} liked item embedding${likes.size === 1 ? '' : 's'}) — ` +
    'every score is one dot product, and looks + means sum to it exactly.';
  $('recExplain').textContent =
    'Add or remove likes and watch the ranking reshuffle. Liked items are never recommended back. ' +
    'Replace the mean with a trained user model later — nothing else changes.';
}

// ---- the agent -------------------------------------------------------------
$('runAgent').addEventListener('click', async () => {
  if (!subject || agentBusy) return;
  agentBusy = true;
  const s = subject;
  $('rounds').replaceChildren(); $('verdict').textContent = ''; $('recordOut').replaceChildren();
  try {
    const encode = await ensureTextEncoder(t => labStatus.textContent = t, labProgress);
    $('labTrack').classList.add('hidden');
    labStatus.textContent = '';
    let i = 0;
    const best = await runAgent(s.imageEmb, encode, (record, v) => {
      if (subject !== s) return;
      i += 1;
      const div = document.createElement('div');
      div.className = 'round ' + (v.satisfied ? 'pass' : 'fail');
      const no = document.createElement('span'); no.className = 'no'; no.textContent = v.satisfied ? '✓' : i;
      const what = document.createElement('div'); what.className = 'what';
      const tpl = document.createElement('div'); tpl.className = 'tpl';
      tpl.textContent = `round ${i} · propose with “${v.template}”`;
      const checks = document.createElement('div'); checks.className = 'checks';
      const chk = (name, val, bar) => Object.assign(document.createElement('span'), {
        className: val >= bar ? 'ok' : 'no-ok',
        textContent: `${name} ${val.toFixed(2)} ${val >= bar ? '≥' : '<'} ${bar} ${val >= bar ? '✓' : '✗'}`,
      });
      checks.append(chk('aligned', v.aligned, MIN_ALIGNED), chk('confident', v.confident, MIN_CONFIDENT));
      const lbls = document.createElement('div');
      lbls.append(...record.labels.slice(0, 6).map(l => Object.assign(document.createElement('span'),
        { className: 'pill on', textContent: `${l.tag} ${l.prob.toFixed(2)}` })));
      what.append(tpl, checks, lbls);
      div.append(no, what);
      $('rounds').append(div);
    });
    if (subject !== s) { agentBusy = false; return; }
    const v = $('verdict');
    if (best.verdict.satisfied) {
      v.className = 'verdict published';
      v.textContent = `Critic satisfied — record published. In Python this row now enters items.sqlite via item_tower.py.`;
      const dims = document.createElement('pre'); dims.className = 'dims';
      dims.textContent = `published record\n  caption   “${best.record.caption}”\n  labels    ` +
        JSON.stringify(Object.fromEntries(best.record.labels.map(l => [l.tag, +l.prob.toFixed(3)]))) +
        `\n  item_emb  (${best.record.fusedEmb.length},)  unit-length  ← the item-tower vector`;
      $('recordOut').append(dims, stripRow('fused_emb', best.record.fusedEmb));
    } else {
      v.className = 'verdict rejected';
      v.textContent = 'No proposal satisfied the critic — the best draft is returned UNPUBLISHED. ' +
        'item_tower.py refuses this record; a human (or a better template pool) takes over.';
    }
  } catch (err) {
    console.error(err);
    labStatus.textContent = 'Could not run the agent here — try agent.py locally.';
  }
  agentBusy = false;
});

restoreFromURL();
