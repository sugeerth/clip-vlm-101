// The page itself: wires the mirrored pipeline modules to the DOM.
//   templates.js (prompts) + clip.js (encoders) + rank.js (dot products)
// Sections below match the page: gallery, search, upload.
import { VOCAB, tagPrompts, captionFor } from './templates.js';
import { dot, rank, topTags, fuse } from './rank.js';
import { getTextEncoder, getImageEncoder } from './clip.js';

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];

const DB = await (await fetch('db.json')).json();
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
    renderResults($('results'), rank(DB.items, q, mode));
    searchStatus.textContent =
      `“${query}” — ${mode} similarity, top 5 of ${DB.items.length}. Same math as search.py.`;
  } catch (err) {
    console.error(err);
    hideProgress();
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
  searching = false;
  // if the query changed while we were busy (e.g. a second chip click), run it now
  const now = qInput.value.trim();
  if (now && now !== query) runSearch();
}

qInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
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
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
  if (uploading) return;
  uploading = true;
  const preview = $('uploadPreview');
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

    const tagLine = document.createElement('p');
    tagLine.append('meta tags: ', ...scored.map(s =>
      Object.assign(document.createElement('span'),
        { className: 'pill', textContent: `${s.tag} ${s.score.toFixed(3)}` })));
    const capLine = document.createElement('p');
    capLine.className = 'sub'; capLine.textContent = `templated caption: “${caption}”`;
    const dims = document.createElement('pre');
    dims.className = 'dims';
    const fmt = (name, v) => `${name.padEnd(10)} (${String(v.length).padStart(4)},)  unit-length  [${v.slice(0, 5).map(x => x.toFixed(4).padStart(7)).join(' ')} …]`;
    dims.textContent = 'your database-ready record (same shapes features.py stores):\n'
      + fmt('image_emb', imgEmb) + '\n' + fmt('text_emb', txtEmb) + '\n' + fmt('fused_emb', fusedEmb);
    $('uploadTags').append(tagLine, capLine, dims);

    // Most similar gallery images (image-to-image, like search.py --image).
    const similar = DB.items.map(item => ({ item, score: dot(item.image_emb, imgEmb) }))
      .sort((a, b) => b.score - a.score).slice(0, 3);
    renderResults($('uploadResults'), similar);
    setStatus('Done — tagged with the prompt template, then matched image-to-image. All local.');
  } catch (err) {
    console.error(err);
    hideProgress();
    uploadStatus.textContent =
      'Could not run the vision model in this browser — try Chrome/Edge/Safari 17+, or run ingest.py locally.';
  }
  uploading = false;
}
