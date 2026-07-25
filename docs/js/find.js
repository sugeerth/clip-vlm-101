// find.js — fast, agentic image search. Type and the grid re-ranks INSTANTLY
// (lexical, zero-debounce, <100ms — Nielsen's "instant" threshold); the on-device
// CLIP text encoder blends in semantic similarity when it loads (progressive
// enhancement, results never wait on a download). The most relevant images render
// LARGEST, top-aligned, edge-to-edge with no gaps. And a ranking-quality agent
// (rankquality.js) watches whether you have to SCROLL — because ~51% of all clicks
// land in the top 3, so a good ranker puts the answer there, and scrolling is
// prima-facie evidence the top-k missed.
import { dot } from './rank.js';
import { getTextEncoder } from './clip.js';
import { rankingQuality, FREE } from './rankquality.js';

const $ = id => document.getElementById(id);
const grid = $('grid'), qInput = $('q'), qmeta = $('qmeta');
const rqFill = $('rqfill'), rqVerdict = $('rqverdict'), rqMetrics = $('rqmetrics'), rqNudge = $('rqnudge');

let DB = { items: [] };
let encodeText = null;          // resolves once CLIP loads; null → lexical only
let qEmb = null;                // the query's CLIP text embedding, when available
let results = [];               // current ranking: [{item, score, gain, relevant}]
let engagedRank = null;         // 1-based rank the user clicked (their answer)
let maxSeenRank = 0;            // deepest rank scrolled into view (the scroll signal)
let debounceId = 0;

const REL = 0.5;                // gain >= REL → "relevant" (a real match, not noise)
const CHIPS = ['cat', 'dog', 'dessert', 'landmark', 'animal', 'mountain', 'something red'];

// --- scoring: instant lexical, + semantic when the encoder is ready ---
function lexical(item, words) {
  if (!words.length) return 0;
  const hay = (item.caption + ' ' + item.tags.join(' ')).toLowerCase();
  let s = 0;
  for (const w of words) {
    if (item.tags.some(t => t === w)) s += 1.0;          // exact tag: strongest
    else if (item.tags.some(t => t.includes(w))) s += 0.7;
    else if (hay.includes(w)) s += 0.4;                  // appears in the caption
  }
  return s / words.length;                               // normalize by query length
}

function rankNow() {
  const query = qInput.value.trim();
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = DB.items.map(item => {
    let s = lexical(item, words);
    if (qEmb && words.length) {
      const cos = dot(qEmb, item.image_emb);             // text→image cosine (~0.15–0.30)
      s += 1.4 * Math.max(0, (cos - 0.15) / 0.15);       // normalize the modality band, blend in
    }
    return { item, score: s };
  });
  scored.sort((a, b) => b.score - a.score || a.item.file.localeCompare(b.item.file));
  const max = scored[0]?.score || 1;
  results = scored.map(r => {
    const gain = words.length ? Math.min(1, r.score / (max || 1)) : 0;
    return { ...r, gain, relevant: r.score > 0 && gain >= REL };
  });

  engagedRank = null; maxSeenRank = 0;                   // a new query is a fresh ranking to judge
  render(words.length > 0);
  updateQuality(words.length > 0);
  qmeta.textContent = words.length
    ? `${results.filter(r => r.score > 0).length} match${results.filter(r => r.score > 0).length === 1 ? '' : 'es'}`
    : `${DB.items.length} images`;
}

// --- the relevance-scaled, gap-free grid (quantized tiers, top-aligned) ---
// the scroll signal is the ACT of scrolling, not what happens to fit on screen:
// only count a tile as "scrolled to" once the user has actually scrolled down, so
// a grid that fits in one viewport is never falsely penalized.
const io = new IntersectionObserver(entries => {
  if (scrollY() < 40) return;                            // hasn't scrolled — no signal
  let changed = false;
  for (const e of entries) if (e.isIntersecting) {
    const r = +e.target.dataset.rank;
    if (r > maxSeenRank) { maxSeenRank = r; changed = true; }
  }
  if (changed && engagedRank === null) updateQuality(true);
}, { threshold: 0.5 });
const scrollY = () => (typeof window !== 'undefined' ? window.scrollY : 0);
addEventListener('scroll', () => {                        // re-judge as the user scrolls the results
  if (scrollY() >= 40 && engagedRank === null && results.length) updateQuality(qInput.value.trim().length > 0);
}, { passive: true });

function render(hasQuery) {
  io.disconnect();
  grid.replaceChildren();
  results.forEach((r, i) => {
    const rank = i + 1;
    const tile = document.createElement('a');
    // top-aligned size tiers: rank 1 = hero, 2–3 = big, rest = normal (biggest = first,
    // never mid-grid — the only way relevance-size encoding helps rather than clutters)
    const size = hasQuery && r.score > 0 ? (rank === 1 ? 'hero' : rank <= FREE ? 'big' : '') : '';
    const pending = hasQuery && r.score <= 0 ? ' pending' : '';
    const top3 = hasQuery && rank <= FREE && r.relevant ? ' top3' : '';
    tile.className = `rtile ${size}${pending}${top3}`;
    tile.href = r.item.file; tile.tabIndex = 0; tile.dataset.rank = rank;
    const desc = (r.item.caption || '').replace(/^a photo of /, '');
    tile.innerHTML =
      `<img src="${r.item.file}" alt="${desc}" loading="lazy">`
      + (hasQuery ? `<span class="rrank">#${rank}</span>` : '')
      + (hasQuery && r.score > 0 ? `<span class="rscore">${r.gain.toFixed(2)}</span>` : '')
      + `<span class="rcap">${desc}</span>`;
    const engage = e => { e.preventDefault(); engagedRank = rank; updateQuality(true); };
    tile.addEventListener('click', engage);
    tile.addEventListener('keydown', e => { if (e.key === 'Enter') engage(e); });
    grid.append(tile);
    io.observe(tile);
  });
}

// --- the ranking-quality agent: P@3 · MRR · nDCG · scroll penalty → one 0-1 score ---
function updateQuality(hasQuery) {
  if (!hasQuery) {
    rqFill.style.width = '0%'; rqVerdict.className = 'rqverdict warn';
    rqVerdict.textContent = 'type to search'; rqMetrics.textContent = ''; rqNudge.classList.remove('on');
    return;
  }
  const gains = results.map(r => r.gain);
  const relevant = results.map(r => r.relevant);
  const scrolling = engagedRank === null && maxSeenRank > FREE;
  // before engagement and before scrolling past the top, show the ranker's own
  // front-loading (no scroll penalty); once they engage or scroll, judge for real.
  const rankForMetric = engagedRank ?? (scrolling ? maxSeenRank : 1);
  const rq = rankingQuality(gains, relevant, rankForMetric);

  rqFill.style.width = `${Math.round(rq.quality * 100)}%`;
  rqFill.style.background = rq.quality >= 0.66 ? 'var(--good)' : rq.quality >= 0.33 ? '#b8860b' : 'var(--bad)';

  let cls, text;
  if (engagedRank !== null) {
    if (engagedRank <= FREE) { cls = 'good'; text = `found it at #${engagedRank} · top-3 ✓`; }
    else { cls = 'bad'; text = `you scrolled to #${engagedRank} · ranking missed`; }
  } else if (scrolling) {
    cls = 'bad'; text = `scrolling past #${FREE} · ranking missed`;
  } else {
    const rel3 = relevant.slice(0, FREE).filter(Boolean).length;
    cls = rel3 >= 1 ? 'good' : 'warn';
    text = rel3 >= 1 ? `${rel3} relevant in the top 3` : 'nothing strong up top';
  }
  rqVerdict.className = `rqverdict ${cls}`;
  rqVerdict.textContent = text;
  rqMetrics.innerHTML =
    `P@3 <b>${rq.precision_at_3.toFixed(2)}</b>`
    + `&nbsp;&nbsp;MRR <b>${rq.mrr.toFixed(2)}</b>`
    + `&nbsp;&nbsp;nDCG@5 <b>${rq.ndcg.toFixed(2)}</b>`
    + `&nbsp;&nbsp;scroll penalty <b>${rq.scroll_penalty.toFixed(2)}</b>`
    + `&nbsp;&nbsp;quality <b>${rq.quality.toFixed(2)}</b>`;
  rqNudge.classList.toggle('on', scrolling && engagedRank === null);
}

// --- wiring: instant lexical on every keystroke, debounce the CLIP encode ---
function onInput() {
  rankNow();                                             // instant, every keystroke
  clearTimeout(debounceId);
  const query = qInput.value.trim();
  if (encodeText && query) {
    debounceId = setTimeout(async () => {
      try { [qEmb] = await encodeText([query]); if (qInput.value.trim() === query) rankNow(); }
      catch { /* stay lexical */ }
    }, 200);
  }
}

async function main() {
  DB = await fetch('db.json').then(r => r.json());
  $('chips').replaceChildren(...CHIPS.map(c => {
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = c;
    b.addEventListener('click', () => { qInput.value = c; onInput(); qInput.focus(); });
    return b;
  }));
  qInput.addEventListener('input', onInput);
  rankNow();                                             // browse mode: show the gallery

  // progressive enhancement: load the CLIP text encoder in the background; when it's
  // ready, re-rank the current query semantically. Blocked/offline → stays lexical.
  getTextEncoder(() => {}, () => {})
    .then(enc => { encodeText = enc; qmeta.title = 'semantic search ready'; if (qInput.value.trim()) onInput(); })
    .catch(() => { /* no network / no model — instant lexical still works */ });
}

main();
