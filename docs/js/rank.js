// Mirror of search.py + tagger.py + fusion.py — pure math, no DOM, no model.
// Everything is dot products between unit-length vectors.

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function unit(v) {
  const n = Math.hypot(...v) || 1;
  return v.map(x => x / n);
}

// Similarity of one db row to a 512-d query, under one mode.
// fused = the average of visual and semantic similarity (see fusion.py).
export const scoreItem = (item, q, mode) =>
  mode === 'image' ? dot(item.image_emb, q)
  : mode === 'text' ? dot(item.text_emb, q)
  : (dot(item.image_emb, q) + dot(item.text_emb, q)) / 2;

// Rank all db rows against a query, keep the top k (search.py).
export const rank = (items, q, mode, k = 5) =>
  items.map(item => ({ item, score: scoreItem(item, q, mode) }))
    .sort((a, b) => b.score - a.score).slice(0, k);

// The k tags whose prompt sentences best match the image (tagger.top_tags).
export const topTags = (imageEmb, tagEmbs, vocab, k = 5) =>
  vocab.map((tag, i) => ({ tag, score: dot(tagEmbs[i], imageEmb) }))
    .sort((a, b) => b.score - a.score).slice(0, k);
