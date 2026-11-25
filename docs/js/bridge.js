// bridge.js — where CLIP·search and ask·200 meet: see ⇄ read.
//
//   gallery photo ─► its CLIP neighbours' tags ─► a web question ─► ask·200 answer
//   web question  ─► the CLIP gallery (words now, pixels with CLIP) ─► "from your gallery"
//   any search on the home page ─► the gallery by meaning  +  the web's one-line answer
//
// Two pure functions are twins of multimodal.py (pinned by test_ask.mjs):
//   knnTags      an image embedding → words, by voting the tags of its nearest gallery
//                images (similarity × 0.5^tag-position). No text encoder, no vocabulary pass:
//                the gallery already knows how to describe what it holds.
//   gallerySearch  words → gallery items by BM25 over caption + tags (keyless, instant);
//                with a CLIP query vector, by the real dot product instead.
// askCard renders ask·200's answer inside any page.
import { bm25, gather, retrieve, extractive, rankImages, claudeAnswer } from './ask.js';

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const argsortDesc = s => s.map((_, i) => i).sort((a, b) => s[b] - s[a] || a - b);

export function knnTags(emb, items, k = 3, n = 3) {
  const sims = items.map(it => dot(it.image_emb, emb));
  const votes = new Map();
  for (const i of argsortDesc(sims).slice(0, k)) {
    items[i].tags.forEach((t, pos) => votes.set(t, (votes.get(t) ?? 0) + sims[i] * 0.5 ** pos));
  }
  const tags = [...votes.keys()];
  return argsortDesc(tags.map(t => votes.get(t))).slice(0, n).map(i => tags[i]);
}

export function gallerySearch(query, items, k = 6, qvec = null) {
  const scores = qvec ? items.map(it => dot(it.image_emb, qvec))
    : bm25(query, items.map(it => `${it.caption} ${it.tags.join(' ')}`));
  return argsortDesc(scores).filter(i => qvec || scores[i] > 0).slice(0, k)
    .map(i => ({ index: i, file: items[i].file, tags: items[i].tags, score: Math.round(scores[i] * 1e4) / 1e4 }));
}

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const host = u => { try { return new URL(u).hostname.replace(/^www\./, ''); } catch { return u; } };
const key = () => { try { return localStorage.getItem('ask_key') || ''; } catch { return ''; } };
const fcKey = () => { try { return localStorage.getItem('ask_fc_key') || ''; } catch { return ''; } };
const model = () => { try { return localStorage.getItem('ask_model') || 'claude-opus-5-5'; } catch { return 'claude-opus-5-5'; } };

// ask·200 inside another page: one cited line + 3 web images + "open in ask·200".
// `isCurrent` lets the caller drop stale answers when the user has typed on.
export async function askCard(el, query, { label = '', isCurrent = () => true, words = 30, fetchFn } = {}) {
  el.classList.remove('hidden');
  el.innerHTML = `<div class="ac-head">⚡ ask·200 — reading the web${label ? ` for <b>${esc(label)}</b>` : ''}…</div>`;
  // with a Firecrawl key saved on the ask page, the home page reads the live web too (5 results)
  const { docs, apiImages } = await gather(query, { limit: 120, firecrawlKey: fcKey(), firecrawlLimit: 5,
    ...(fetchFn ? { fetchFn } : {}) });
  if (!isCurrent()) return;
  const { top, sources, expansion } = retrieve(query, docs);
  if (!top.length) { el.innerHTML = '<div class="ac-head">⚡ ask·200 — the web had nothing on this</div>'; return; }
  const images = rankImages(query, docs, apiImages, new Map(sources.map(s => [s.url, s.n])), 3);
  const by = new Map(sources.map(s => [s.n, s]));
  const paint = (text, mode) => {
    if (!isCurrent()) return;
    const body = esc(text).replace(/\[(\d+)\]/g, (m, n) => by.has(+n)
      ? `<a class="cite" href="${esc(by.get(+n).url)}" target="_blank" rel="noopener" title="${esc(by.get(+n).title)}">${n}</a>` : m);
    el.innerHTML =
      `<div class="ac-head">⚡ ask·200${label ? ` · <b>${esc(label)}</b>` : ''} · ${docs.length} documents · ${esc(mode)}</div>` +
      `<div class="ac-body">${body}</div>` +
      (images.length ? `<div class="ac-imgs">${images.map(im =>
        `<a href="${esc(im.page || im.url)}" target="_blank" rel="noopener" title="${esc(im.alt)}"><img src="${esc(im.url)}" ` +
        `alt="${esc(im.alt)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.parentElement.remove()"></a>`).join('')}</div>` : '') +
      `<div class="ac-foot">${sources.slice(0, 3).map(s => `<a href="${esc(s.url)}" target="_blank" rel="noopener">[${s.n}] ${esc(host(s.url))}</a>`).join(' ')}` +
      ` <a class="ac-more" href="ask.html?q=${encodeURIComponent(query)}">full answer ↗</a></div>`;
  };
  paint(extractive(query, top, words, expansion), 'extractive');
  if (key()) {
    let acc = '';
    try {
      const out = await claudeAnswer({ key: key(), model: model(), question: query, top, words, images,
        onText: t => paint(acc += t, 'writing…') });
      paint(out.text, model() + (out.vision ? ' + vision' : ''));
    } catch { /* keep the extractive line */ }
  }
}
