// The page itself: wires the mirrored pipeline modules to the DOM.
//   templates.js (prompts) + clip.js (encoders) + rank.js (dot products)
// Sections below match the page: gallery, map, search, upload.
import { VOCAB, tagPrompts, captionFor } from './templates.js';
import { dot, rank, topTags, fuse, softmax } from './rank.js';
import { getTextEncoder, getImageEncoder } from './clip.js';
import { drawMap, drawHeatmap, markUpload, project, stripRow } from './viz.js';

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];

let DB = { items: [], pca: null };
try {
  DB = await (await fetch('db.json')).json();
} catch (err) { // without this, a failed fetch rejects the whole module and the page goes inert
  console.error(err);
  $('searchStatus').textContent = $('uploadStatus').textContent =
    'Could not load db.json — check your connection and reload the page.';
}
let tagEmbs = null; // vocabulary prompt embeddings, computed once on demand

// ---------------------------------------------------------------- gallery --
for (const item of DB.items) {
  const fig = document.createElement('figure');
  const img = Object.assign(document.createElement('img'),
    { src: item.file, alt: item.caption, loading: 'lazy' });
  const cap = document.createElement('figcaption');
  cap.append(...item.tags.map(t =>
    Object.assign(document.createElement('span'), { className: 'pill', textContent: t })));
  fig.append(img, cap);
  $('gallery').append(fig);
}

// -------------------------------------------------------- embedding map --
// Gallery embeddings at their PCA spots; click one to see its raw numbers.
if (DB.pca) drawMap($('map'), DB.items, item => {
  const who = document.createElement('p');
  who.className = 'who';
  who.textContent = `${item.caption} — every bar below is one stored number (blue < 0 < red)`;
  $('mapDetail').replaceChildren(who,
    stripRow('image_emb', item.image_emb),
    stripRow('text_emb', item.text_emb));
});

// ------------------------------------------------------ all-pairs heatmap --
// One dot product per cell (similarity.py's twin); click shows the pair.
if (DB.items.length) drawHeatmap($('heat'), DB.items, (a, b, s) => {
  $('heatDetail').textContent = a === b
    ? `${a.caption} · itself = ${s.toFixed(3)} — a unit vector times itself is 1`
    : `${a.caption}  ·  ${b.caption}  =  ${s.toFixed(3)}`;
});

// ------------------------------------------------------------- shared UI --
// Progress bar for model downloads (transformers.js progress events).
function onProgress(p) {
  if (p.status === 'progress' && p.total) {
    $('loadTrack').classList.remove('hidden');
    $('loadFill').style.width = Math.round(100 * p.loaded / p.total) + '%';
  }
}
const hideProgress = () => $('loadTrack').classList.add('hidden');

// Ranked rows with thumbnails; bars are relative to the best hit,
// the exact score is printed beside each bar.
function renderResults(el, ranked) {
  el.replaceChildren();
  const top = Math.max(...ranked.map(r => r.score), 1e-6);
  for (const { item, score, parts } of ranked) {
    const row = document.createElement('div'); row.className = 'result';
    const img = Object.assign(document.createElement('img'), { src: item.file, alt: item.caption });
    const meta = document.createElement('div'); meta.className = 'meta';
    const name = document.createElement('div'); name.className = 'name'; name.textContent = item.caption;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill';
    fill.style.width = Math.max(2, Math.min(100, 100 * score / top)) + '%';
    track.append(fill); meta.append(name, track);
    if (parts) { // fused mode: show which signal carried this hit
      const p = document.createElement('div');
      p.className = 'parts';
      p.textContent = `= (image ${parts.image.toFixed(3)} + text ${parts.text.toFixed(3)}) / 2`;
      meta.append(p);
    }
    const s = document.createElement('div'); s.className = 'score'; s.textContent = score.toFixed(3);
    row.append(img, meta, s);
    el.append(row);
  }
}

// ----------------------------------------------------------------- search --
const qInput = $('q'), searchStatus = $('searchStatus');
let searching = false;

async function runSearch() {
  const query = qInput.value.trim();
  if (!query || searching) return;
  searching = true;
  const mode = document.querySelector('input[name=mode]:checked').value;
  try {
    const encode = await getTextEncoder(t => searchStatus.textContent = t, onProgress);
    hideProgress();
    searchStatus.textContent = 'Embedding your query…';
    const [q] = await encode([query]);
    const ranked = rank(DB.items, q, mode);
    if (mode === 'fused') for (const r of ranked) // same decomposition search.py prints
      r.parts = { image: dot(r.item.image_emb, q), text: dot(r.item.text_emb, q) };
    renderResults($('results'), ranked);
    searchStatus.textContent =
      `“${query}” — ${mode} similarity, top 5 of ${DB.items.length}. Same math as search.py.`;
  } catch (err) {
    console.error(err);
    hideProgress();
    // graceful fallback: keyword match over tags + captions
    // (skip stopword-length words — "a" would substring-match half the vocab)
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length >= 3);
    const keyword = DB.items.map(item => ({
      item,
      score: words.filter(w => item.tags.some(t => t.includes(w)) || item.caption.toLowerCase().includes(w)).length / 4,
    })).filter(r => r.score > 0).sort((a, b) => b.score - a.score).slice(0, 5);
    renderResults($('results'), keyword);
    searchStatus.textContent = keyword.length
      ? 'Model failed to load here, showing keyword matches instead — run search.py locally for the real thing.'
      : 'Model failed to load here and no keyword matches either — run search.py locally for the real thing.';
  } finally {
    searching = false;
  }
  // if the query or mode changed while we were busy (a chip click, a radio
  // toggled mid-download), run again so the results match the controls
  const now = qInput.value.trim();
  const nowMode = document.querySelector('input[name=mode]:checked').value;
  if (now && (now !== query || nowMode !== mode)) runSearch();
}

qInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
qInput.addEventListener('input', () => { // clearing the box clears the results too
  if (!qInput.value.trim()) {
    $('results').replaceChildren();
    searchStatus.textContent = 'The CLIP text encoder loads on your first search (~65 MB, cached after that).';
  }
});
document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', runSearch));
for (const ex of EXAMPLES) {
  const b = Object.assign(document.createElement('button'), { className: 'chip', textContent: ex });
  b.addEventListener('click', () => { qInput.value = ex; runSearch(); });
  $('examples').append(b);
}

// ----------------------------------------------------------------- upload --
const drop = $('drop'), fileInput = $('file'), uploadStatus = $('uploadStatus');
let uploading = false;

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('armed'); });
drop.addEventListener('dragleave', () => drop.classList.remove('armed'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('armed');
  if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleUpload(fileInput.files[0]);
  fileInput.value = ''; // so picking the SAME file again re-fires change (the retry path)
});

async function handleUpload(file) {
  if (uploading) return;
  uploading = true;
  const preview = $('uploadPreview');
  if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.classList.remove('hidden');
  $('uploadTags').replaceChildren(); $('uploadResults').replaceChildren();
  try {
    const setStatus = t => uploadStatus.textContent = t;
    const encodeImage = await getImageEncoder(setStatus, onProgress);
    hideProgress();
    setStatus('Embedding your image…');
    const imgEmb = await encodeImage(file);

    // Zero-shot meta tags: same prompt template as ingest.py / tagger.py.
    const encodeText = await getTextEncoder(setStatus, onProgress);
    hideProgress();
    if (!tagEmbs) {
      setStatus('Embedding the tag vocabulary (once)…');
      tagEmbs = await encodeText(tagPrompts());
    }
    const scored = topTags(imgEmb, tagEmbs, VOCAB);
    const caption = captionFor(scored.map(s => s.tag));
    const [txtEmb] = await encodeText([caption]);
    const fusedEmb = fuse(imgEmb, txtEmb);
    // this is the complete database-ready record — the mirror of features.extract()

    // softmax over ALL vocabulary scores (temperature.py's lesson): the raw
    // cosines huddle together; probabilities show the real confidence gap
    const tagProbs = softmax(tagEmbs.map(e => dot(e, imgEmb)));
    const tagLine = document.createElement('p');
    tagLine.append('meta tags: ', ...scored.map(s =>
      Object.assign(document.createElement('span'), {
        className: 'pill',
        textContent: `${s.tag} ${s.score.toFixed(3)} → ${(100 * tagProbs[VOCAB.indexOf(s.tag)]).toFixed(0)}%`,
      })));
    const capLine = document.createElement('p');
    capLine.className = 'sub'; capLine.textContent = `templated caption: “${caption}”`;
    const dims = document.createElement('pre');
    dims.className = 'dims';
    const fmt = (name, v) => `${name.padEnd(10)} (${String(v.length).padStart(4)},)  unit-length  [${v.slice(0, 5).map(x => x.toFixed(4).padStart(7)).join(' ')} …]`;
    dims.textContent = 'your database-ready record (same shapes features.py stores):\n'
      + fmt('image_emb', imgEmb) + '\n' + fmt('text_emb', txtEmb) + '\n' + fmt('fused_emb', fusedEmb);
    $('uploadTags').append(tagLine, capLine, dims,
      stripRow('image_emb', imgEmb), stripRow('text_emb', txtEmb), stripRow('fused_emb', fusedEmb));
    if (DB.pca) markUpload($('map'), project(imgEmb, DB.pca));  // drop it on the map

    // Most similar gallery images (image-to-image, like search.py --image).
    const similar = DB.items.map(item => ({ item, score: dot(item.image_emb, imgEmb) }))
      .sort((a, b) => b.score - a.score).slice(0, 3);
    renderResults($('uploadResults'), similar);
    setStatus('Done — tagged with the prompt template, then matched image-to-image. All local.');
  } catch (err) {
    console.error(err);
    hideProgress();
    uploadStatus.textContent = 'Could not load or run the vision model (often just a network hiccup) — '
      + 'drop the image again to retry, or run ingest.py locally.';
  } finally {
    uploading = false;
  }
}
