// Visualize the embeddings — no libraries.
//
// map:   every image placed by PCA (computed offline by export_web.py) so
//        that images with similar embeddings land close together. project()
//        drops a NEW embedding onto the same map with two dot products.
// strip: the raw values of one embedding — one thin bar per number
//        (blue = negative, red = positive). This is literally what the
//        database stores.
import { dot } from './rank.js';

// Mirror of export_web.py's transform: (v - mean) @ components.T, normalized.
export function project(emb, pca) {
  const centered = emb.map((x, i) => x - pca.mean[i]);
  return pca.components.map((c, k) =>
    Math.min(1, Math.max(0, (dot(centered, c) - pca.lo[k]) / pca.span[k])));
}

const SVG = 'http://www.w3.org/2000/svg';
const el = (tag, attrs) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};
// map [0..1, 0..1] -> padded viewBox coordinates (0 0 100 62)
const pos = xy => [7 + xy[0] * 86, 55 - xy[1] * 48];

export function drawMap(svg, items, onPick) {
  for (const item of items) {
    const [x, y] = pos(item.map);
    const img = el('image', {
      href: item.file, x: x - 3.5, y: y - 3.5, width: 7, height: 7,
      preserveAspectRatio: 'xMidYMid slice',
    });
    img.style.cursor = 'pointer';
    const title = el('title', {});
    title.textContent = item.tags.join(', ');
    img.append(title);
    img.addEventListener('click', () => onPick(item));
    svg.append(img);
  }
}

export function markUpload(svg, xy) {
  svg.querySelector('.upload-marker')?.remove();
  const [x, y] = pos(xy);
  const g = el('g', { class: 'upload-marker' });
  const ring = el('circle', { cx: x, cy: y, r: 4.6 });
  // near the top edge the label flips below the ring so it never clips
  const label = el('text', { x: Math.min(92, Math.max(8, x)), y: y < 13 ? y + 8 : y - 6 });
  label.textContent = 'your image';
  g.append(ring, label);
  svg.append(g);
}

// All-pairs image·image similarity as a clickable grid — similarity.py's
// twin: one dot product per cell, darker = more similar.
export function drawHeatmap(canvas, items, onPick) {
  const embs = items.map(it => it.image_emb);
  const n = items.length, cell = 24;
  const M = embs.map(a => embs.map(b => dot(a, b)));
  let lo = Infinity, hi = -Infinity;
  for (const row of M) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const MID = dark ? [26, 26, 25] : [252, 252, 251];
  const HOT = dark ? [57, 135, 229] : [42, 120, 214];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  canvas.width = canvas.height = n * cell;
  const ctx = canvas.getContext('2d');
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const t = (M[i][j] - lo) / (hi - lo || 1);
    const c = mix(MID, HOT, t);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
  }
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left) / r.width * n);
    const i = Math.floor((e.clientY - r.top) / r.height * n);
    if (i >= 0 && i < n && j >= 0 && j < n) onPick(items[i], items[j], M[i][j]);
  });
}

// A labeled fingerprint strip: label + <canvas>, one pixel column per value.
export function stripRow(label, vec) {
  const row = document.createElement('div');
  row.className = 'strip-row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = `${label} (${vec.length},)`;
  const canvas = document.createElement('canvas');
  drawStrip(canvas, vec);
  row.append(lbl, canvas);
  return row;
}

function drawStrip(canvas, vec) {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const NEG = dark ? [57, 135, 229] : [42, 120, 214];   // blue
  const POS = dark ? [230, 103, 103] : [227, 73, 72];   // red
  const MID = dark ? [56, 56, 53] : [240, 239, 236];    // neutral zero
  canvas.width = vec.length; canvas.height = 24;
  const ctx = canvas.getContext('2d');
  // unit vectors have rms 1/sqrt(n); color saturates at ~3 sigma
  const scale = 3 / Math.sqrt(vec.length);
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  for (let i = 0; i < vec.length; i++) {
    const t = Math.max(-1, Math.min(1, vec[i] / scale));
    const c = t < 0 ? mix(MID, NEG, -t) : mix(MID, POS, t);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(i, 0, 1, 24);
  }
}
