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
import { OnlineRanker } from './learn.js';

const $ = id => document.getElementById(id);
const grid = $('grid'), qInput = $('q'), qmeta = $('qmeta');
const rqFill = $('rqfill'), rqVerdict = $('rqverdict'), rqMetrics = $('rqmetrics'), rqNudge = $('rqnudge');
const rqLearn = $('rqlearn');

// the on-device personal ranker: four floats in localStorage, trained by your
// 👍/👎 AND by the ranking-quality agent's failure case — clicking a result you
// had to scroll to teaches the ranker to put it higher next time. The quality
// agent grades the ranker; your behavior teaches it. (Own key: find's features
// are its own, so weights are not shared with the simple-search ranker.)
const RANK_KEY = 'clipvlm.find.ranker';
const ranker = new OnlineRanker();
try { const st = JSON.parse(localStorage.getItem(RANK_KEY)); if (st) ranker.loadState(st); } catch { /* fresh */ }
const persistRanker = () => { try { localStorage.setItem(RANK_KEY, JSON.stringify(ranker.toState())); } catch { /* private mode */ } };
let taughtNote = '';            // one-shot "learned from that" note for the status line

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
    let cosI = 0, cosT = 0;
    if (qEmb && words.length) {
      cosI = Math.max(0, dot(qEmb, item.image_emb));     // text→image cosine (~0.15–0.30)
      cosT = Math.max(0, dot(qEmb, item.text_emb));
      s += 1.4 * Math.max(0, (cosI - 0.15) / 0.15);      // normalize the modality band, blend in
    }
    return { item, score: s, cosI, cosT };
  });
  scored.sort((a, b) => b.score - a.score || a.item.file.localeCompare(b.item.file));

  // the learner's four features — [cos_image, cos_text, tag_overlap, rank_prior],
  // the same shape learn.js was trained for. rank_prior comes from the BASE order,
  // so the learned preference argues with the base ranking, not with itself.
  let ordered = scored.map((r, i) => ({
    ...r,
    base_score: r.score,
    features: [r.cosI || Math.min(1, r.score), r.cosT,
               words.filter(w => r.item.tags.some(t => t.includes(w))).length,
               1 / (1 + i)],
  }));
  // your personal ranker re-orders: final = (1−β)·base + β·learned, β capped at 50%
  if (words.length && ranker.n) ordered = ranker.rank(ordered, { baseKey: 'base_score' });

  const max = ordered[0]?.score || 1;
  results = ordered.map(r => {
    const gain = words.length ? Math.min(1, (r.score || 0) / (max || 1)) : 0;
    return { ...r, gain, relevant: r.base_score > 0 && gain >= REL };
  });

  engagedRank = null; maxSeenRank = 0;                   // a new query is a fresh ranking to judge
  render(words.length > 0);
  updateQuality(words.length > 0);
  qmeta.textContent = words.length
    ? `${results.filter(r => r.base_score > 0).length} match${results.filter(r => r.base_score > 0).length === 1 ? '' : 'es'}`
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
    const size = hasQuery && r.base_score > 0 ? (rank === 1 ? 'hero' : rank <= FREE ? 'big' : '') : '';
    const pending = hasQuery && r.base_score <= 0 ? ' pending' : '';
    const top3 = hasQuery && rank <= FREE && r.relevant ? ' top3' : '';
    tile.className = `rtile ${size}${pending}${top3}`;
    tile.href = r.item.file; tile.tabIndex = 0; tile.dataset.rank = rank;
    const desc = (r.item.caption || '').replace(/^a photo of /, '');
    tile.innerHTML =
      `<img src="${r.item.file}" alt="${desc}" loading="lazy">`
      + (hasQuery ? `<span class="rrank">#${rank}</span>` : '')
      + (hasQuery && r.base_score > 0 ? `<span class="rscore">${r.gain.toFixed(2)}</span>` : '')
      + `<span class="rcap">${desc}</span>`;
    // the closed loop, explicit: 👍/👎 trains the on-device ranker and re-ranks live
    if (hasQuery) {
      const votes = document.createElement('span');
      votes.className = 'rvotes';
      for (const [glyph, label] of [['👍', 1], ['👎', 0]]) {
        const b = document.createElement('button');
        b.textContent = glyph;
        b.title = label ? 'more like this — teach the ranker' : 'less like this — teach the ranker';
        b.addEventListener('click', e => {
          e.preventDefault(); e.stopPropagation();
          ranker.feedback(r.features, label);
          persistRanker();
          rankNow();                                     // re-rank with the new weights, live
          taughtNote = `learned ${glyph} “${desc.split(',')[0]}” — re-ranked`;
          renderLearn();
        });
        votes.append(b);
      }
      tile.append(votes);
    }
    // the closed loop, implicit: clicking a result you had to SCROLL to is the
    // quality agent's failure case — so it becomes a training example, automatically.
    const engage = e => {
      e.preventDefault();
      engagedRank = rank;
      if (hasQuery && rank > FREE && r.base_score > 0) {
        ranker.feedback(r.features, 1);
        persistRanker();
        taughtNote = `learned from the miss — “${desc.split(',')[0]}” will rank higher next time`;
      }
      updateQuality(true);
    };
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
    renderLearn();
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
  renderLearn();
}

// --- the loop made visible: what your feedback has taught the on-device ranker ---
function renderLearn() {
  if (!ranker.n) {
    rqLearn.innerHTML = 'personal ranker: <b>untrained</b> — 👍/👎 a tile, or click a result you '
      + 'had to scroll to, and it learns (four floats, your browser only)';
    taughtNote = '';
    return;
  }
  const beta = Math.round(100 * 0.5 * ranker.n / (ranker.n + 3));   // learn.js's blend cap
  rqLearn.innerHTML =
    `personal ranker: <b>${ranker.n}</b> signal${ranker.n === 1 ? '' : 's'} (${ranker.nPairs()} pairs)`
    + ` · personal weight <b>${beta}%</b>`
    + (taughtNote ? ` · <span class="taught">${taughtNote}</span>` : '')
    + ` <button id="rqreset" type="button">reset</button>`;
  $('rqreset').addEventListener('click', () => {
    ranker.buffer.length = 0; ranker._refit();          // back to untrained w=[1,0,0,0]
    persistRanker(); taughtNote = '';
    rankNow();
  });
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
  // prefer the grown gallery (written by the automated grow pipeline; hundreds of
  // crawled images with remote thumbnails) and fall back to the frozen 14-image
  // teaching gallery — the pinned lessons always reference db.json, search grows.
  DB = await fetch('db_grown.json').then(r => r.ok ? r.json() : null).catch(() => null)
    || await fetch('db.json').then(r => r.json());
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
