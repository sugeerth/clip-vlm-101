// CLIP·search — the simple page. One box, real inference, results.
//
// All the complexity this repo teaches is deliberately invisible here:
// the model loads once on first use (with a quiet progress bar), queries
// are embedded live as you type, and ranking is the same fused dot
// product as search.py. The explorable at explore.html shows the insides.
import { dot, rank } from './rank.js';
import { getTextEncoder, getImageEncoder } from './clip.js';

const $ = id => document.getElementById(id);
const EXAMPLES = ['a fluffy animal', 'famous landmark in europe',
  'something delicious to eat', 'outer space', 'water in nature'];
const TOP_K = 8;

const DB = await (await fetch('db.json')).json();

let encodeText = null;      // resolves once, then queries are ~instant
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

// ----------------------------------------------------------------- search --
async function search() {
  const query = $('q').value.trim();
  if (!query && !imageQuery) return;
  if (searching) { rerun = true; return; }
  searching = true;
  try {
    if (imageQuery) {                         // image → image, like search.py --image
      show(DB.items
        .map(item => ({ item, score: dot(item.image_emb, imageQuery.emb) }))
        .sort((a, b) => b.score - a.score).slice(0, TOP_K));
    } else {
      if (!encodeText) status('loading the model — one time, cached after…');
      encodeText ??= await getTextEncoder(status, progress);
      hideBar();
      const [q] = await encodeText([query]);
      show(rank(DB.items, q, 'fused', TOP_K));
    }
  } catch (err) {
    console.error(err);
    hideBar();
    const hits = keywordResults(query);
    show(hits, hits.length ? 'model unavailable here — showing keyword matches'
                           : 'model unavailable and no keyword matches');
  }
  searching = false;
  if (rerun) { rerun = false; search(); }
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

// ---------------------------------------------------- search by image --
// Camera button, drop anywhere, or paste a copied image — same result:
// the photo is embedded locally and ranked image-to-image.
async function imageSearch(file) {
  if (!file || !file.type.startsWith('image/')) return;
  try {
    status('loading the vision model — one time, cached after…');
    const encodeImage = await getImageEncoder(status, progress);
    hideBar();
    status('');
    const emb = await encodeImage(file);
    imageQuery = { emb };
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
