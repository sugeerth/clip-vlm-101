// The instant layer — why the front page answers on the first keystroke.
// Pure math, no DOM, no model (Node-testable: test_instant.mjs).
//
//   imageVectors() every gallery image_emb packed into ONE Float32Array.
//   scoreAll()     the whole gallery against a query vector, one tight loop,
//                  minus each image's HUB BIAS (below).
//   topK()         best k indices.
//   preview()      ranking BEFORE the model exists: average the precomputed
//                  per-word score rows from lexicon.json (build_lexicon.mjs).
//                  Valid because scores are linear in the query vector, so
//                  mean-of-word-scores ranks exactly like the mean of the
//                  word vectors would.
//
// Why the IMAGE tower, centred? Measured on 38 labelled text queries against
// the 14-image gallery (exact CLIP encodes, top-1 correct):
//     text tower (templated tag captions)   16 / 38
//     fused  (image + text) / 2              25 / 38
//     image tower                            35 / 38
//     image tower − hub bias                 37 / 38
// The caption sentences make a few items "hubs" (pizza, dog) that score high
// for EVERY query. bias[i] = image i's mean score over the 6,000 lexicon
// words — its similarity to "a typical query" — and subtracting it removes
// the hub effect (the same idea as similarity.py --centered).

// words that carry no visual meaning — skipped by both builder and preview
export const STOP = new Set(`a an the of and or to in on at by for with from
into onto over under is are was were be been being it its this that these
those there here some something any anything thing things very really just
me my you your our we us he she they them his her their what which who whom
whose show find photo photos picture pictures image images pic pics of like
as but if then than so not no yes do does did can could would should will
shall may might must about above below near far up down out off again more
most less least much many few all each every both either neither other such
own same too also only www com http https html org net`.split(/\s+/).filter(Boolean));

// ------------------------------------------------------------ gallery --
export function imageVectors(items) {
  const n = items.length, dim = n ? items[0].image_emb.length : 0;
  const mat = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) mat.set(items[i].image_emb, i * dim);
  return { mat, n, dim };
}

// bias: per-item hub bias (lexicon.json) for TEXT queries; omit it for
// photo → photo search, where the text-query prior does not apply.
export function scoreAll({ mat, n, dim }, q, bias = null) {
  const out = new Float32Array(n);
  for (let i = 0, off = 0; i < n; i++, off += dim) {
    let s = 0;
    for (let k = 0; k < dim; k++) s += mat[off + k] * q[k];
    out[i] = bias ? s - bias[i] : s;
  }
  return out;
}

export function topK(scores, k, skip = -1) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (i !== skip) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}

// A lexicon built for a different gallery would rank garbage — so it
// carries a fingerprint of the gallery it was scored against (FNV-1a over
// the model, every file name and a few embedding numbers).
export function gallerySignature(items) {
  let h = 0x811c9dc5;
  const feed = s => {
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  };
  feed(String(items.length));
  for (const it of items) feed(`${it.file}|${it.image_emb.slice(0, 4).join(',')}|${it.text_emb.slice(0, 4).join(',')}`);
  return h.toString(16).padStart(8, '0');
}

// ------------------------------------------------------------ lexicon --
// float scores -> one byte each over a shared [lo, hi] range (the preview
// only needs ordering; 1/255 of the range is far finer than CLIP's noise)
export function quantize(scores) {
  let lo = Infinity, hi = -Infinity;
  for (const s of scores) { if (s < lo) lo = s; if (s > hi) hi = s; }
  const span = hi - lo || 1, bytes = new Uint8Array(scores.length);
  for (let i = 0; i < scores.length; i++) bytes[i] = Math.round(255 * (scores[i] - lo) / span);
  return { lo, hi, bytes };
}

const b64bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

// json = parsed lexicon.json. Returns null when it does not belong to this
// gallery (stale after a re-export) — the page then simply waits for CLIP.
export function loadLexicon(json, items) {
  if (!json || json.sig !== gallerySignature(items) || json.n_items !== items.length) return null;
  const words = json.words.split(' ');
  const scores = b64bytes(json.scores);
  if (scores.length !== words.length * json.n_items) return null;
  return {
    words, scores, n: json.n_items, lo: json.lo, step: (json.hi - json.lo) / 255,
    bias: Float32Array.from(json.bias), row: new Map(words.map((w, i) => [w, i])),
  };
}

export const tokenize = query => query.toLowerCase().split(/[^a-z]+/).filter(Boolean);

// the earliest (most visual / most common) word starting with `prefix`
export function complete(lex, prefix) {
  if (prefix.length < 2) return null;
  for (const w of lex.words) if (w.startsWith(prefix)) return w;
  return null;
}

// a finished word -> its row: exact, else a plain plural ("dogs" → "dog")
function lookup(lex, w) {
  if (lex.row.has(w)) return w;
  if (w.endsWith('es') && lex.row.has(w.slice(0, -2))) return w.slice(0, -2);
  if (w.endsWith('s') && lex.row.has(w.slice(0, -1))) return w.slice(0, -1);
  return null;
}

// -> { scores: Float32Array(n), words: [the words it used] } | null
// (scores are hub-centred, on the same scale as scoreAll(.., bias))
// The last word is still being typed unless the query ends in a separator,
// so it is completed by prefix ("a fluf" previews as "fluffy").
export function preview(lex, query) {
  if (!lex) return null;
  const toks = tokenize(query).filter(t => !STOP.has(t));
  const typing = /[a-z]$/i.test(query);
  const used = [];
  toks.forEach((t, i) => {
    const w = lookup(lex, t) ?? (typing && i === toks.length - 1 ? complete(lex, t) : null);
    if (w && !STOP.has(w) && !used.includes(w)) used.push(w);
  });
  if (!used.length) return null;
  const n = lex.n, out = new Float32Array(n);
  for (const w of used) {
    const r = lex.row.get(w) * n;
    for (let i = 0; i < n; i++) out[i] += lex.lo + lex.step * lex.scores[r + i];
  }
  for (let i = 0; i < n; i++) out[i] = out[i] / used.length - lex.bias[i];
  return { scores: out, words: used };
}

// ------------------------------------------------------------- cache --
// query -> vector, least-recently-used out. Backspacing is free.
export class LRU {
  constructor(cap = 256) { this.cap = cap; this.m = new Map(); }
  get(k) {
    if (!this.m.has(k)) return undefined;
    const v = this.m.get(k); this.m.delete(k); this.m.set(k, v); return v;
  }
  set(k, v) {
    this.m.delete(k); this.m.set(k, v);
    if (this.m.size > this.cap) this.m.delete(this.m.keys().next().value);
  }
}
