// The page itself: wires the mirrored pipeline modules to the DOM.
//   templates.js (prompts) · clip.js (encoders) · rank.js (dot products)
//   labels.js (multi-label) · agent.js (propose ⇄ critique) · viz.js (pictures)
// Sections below match the page: theme, matrix, gallery, map, search, lab.
import { VOCAB, tagPrompts, NEUTRAL_PROMPT } from './templates.js';
import { dot, rank, topTags } from './rank.js';
import { labelProbs } from './labels.js';
import { runAgent, MIN_ALIGNED, MIN_CONFIDENT } from './agent.js';
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
recolorMatrix = drawMatrix($('matrix'), DB.items, (imgItem, capItem, score, isDiag) => {
  const readout = $('matrixReadout');
  readout.replaceChildren();
  const val = document.createElement('div');
  val.className = 'val'; val.textContent = score.toFixed(3);
  const pair = document.createElement('div');
  pair.className = 'pair';
  pair.textContent = `image “${imgItem.tags[0]}” × caption “${capItem.caption}”`;
  const note = document.createElement('div');
  note.className = 'pair';
  note.textContent = isDiag
    ? 'This image scored against its own caption — the diagonal the training objective brightens.'
    : 'An off-diagonal pair — contrastive training pushes these apart.';
  readout.append(val, pair, note);
});

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

// Ranked rows with thumbnails; bars are relative to the best hit,
// the exact score is printed beside each bar (never inside it).
function renderResults(el, ranked) {
  el.replaceChildren();
  const top = Math.max(...ranked.map(r => r.score), 1e-6);
  for (const { item, score } of ranked) {
    const row = document.createElement('div'); row.className = 'result';
    const img = Object.assign(document.createElement('img'), { src: item.file, alt: item.caption });
    const meta = document.createElement('div'); meta.className = 'meta';
    const name = document.createElement('div'); name.className = 'name'; name.textContent = item.caption;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill';
    fill.style.width = Math.max(2, Math.min(100, 100 * score / top)) + '%';
    track.append(fill); meta.append(name, track);
    const s = document.createElement('div'); s.className = 'score'; s.textContent = score.toFixed(3);
    row.append(img, meta, s);
    el.append(row);
  }
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
    renderResults($('results'), rank(DB.items, q, mode));
    searchStatus.textContent =
      `“${query}” — ${mode} similarity, top 5 of ${DB.items.length}. Same math as search.py.`;
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
  computeTagsAndLabels();
}

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
$('thresh').addEventListener('input', renderLabels);

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
