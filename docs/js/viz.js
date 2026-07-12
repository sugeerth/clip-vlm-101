// Visualize the embeddings — no libraries.
//
// matrix: the contrastive-training idea, live — image × caption dot products
//         on a one-hue sequential ramp; the diagonal is the objective.
// map:    every image placed by PCA (computed offline by export_web.py) so
//         that images with similar embeddings land close together. project()
//         drops a NEW embedding onto the same map with two dot products.
// strip:  the raw values of one embedding — one thin bar per number on the
//         diverging blue↔red pair. This is literally what the database stores.
import { dot } from './rank.js';

// ---- theme-aware colors: read the palette off CSS custom properties, so a
// ---- theme flip only needs a re-colorize, never new markup.
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const hexToRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const mixHex = (a, b, t) => {
  const [ra, ga, ba] = hexToRgb(a), [rb, gb, bb] = hexToRgb(b);
  return `rgb(${Math.round(ra + (rb - ra) * t)},${Math.round(ga + (gb - ga) * t)},${Math.round(ba + (bb - ba) * t)})`;
};

// Mirror of export_web.py's transform: (v - mean) @ components.T, normalized.
export function project(emb, pca) {
  const centered = emb.map((x, i) => x - pca.mean[i]);
  return pca.components.map((c, k) =>
    Math.min(1, Math.max(0, (dot(centered, c) - pca.lo[k]) / pca.span[k])));
}

const SVG = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};
// map [0..1, 0..1] -> padded viewBox coordinates (0 0 100 62)
const pos = xy => [7 + xy[0] * 86, 55 - xy[1] * 48];

// ------------------------------------------------------ contrastive matrix --
// items: db rows with image_emb + text_emb. Row i = image i, column j =
// caption j, cell = image_emb[i] · text_emb[j]. Returns colorize() so the
// theme toggle can re-paint without rebuilding the DOM.
export function drawMatrix(grid, items, onHover) {
  const n = items.length;
  grid.style.gridTemplateColumns = `repeat(${n + 1}, auto)`;
  const scores = items.map(a => items.map(b => dot(a.image_emb, b.text_emb)));
  const flat = scores.flat();
  const lo = Math.min(...flat), span = Math.max(...flat) - lo || 1;

  grid.replaceChildren();
  grid.append(Object.assign(document.createElement('span'), { className: 'colhead' }));
  for (let j = 0; j < n; j++) {
    const h = Object.assign(document.createElement('span'), { className: 'colhead', textContent: j + 1 });
    grid.append(h);
  }
  const cells = [];
  for (let i = 0; i < n; i++) {
    const thumb = Object.assign(document.createElement('img'), {
      className: 'rowhead', src: items[i].file, alt: items[i].caption, loading: 'lazy',
    });
    grid.append(thumb);
    for (let j = 0; j < n; j++) {
      const cell = Object.assign(document.createElement('button'), {
        className: 'cell' + (i === j ? ' diag' : ''), type: 'button',
      });
      cell.setAttribute('aria-label',
        `image ${i + 1} × caption ${j + 1}: similarity ${scores[i][j].toFixed(3)}`);
      const enter = () => onHover(items[i], items[j], scores[i][j], i === j);
      cell.addEventListener('pointerenter', enter);
      cell.addEventListener('focus', enter);
      cells.push({ cell, t: (scores[i][j] - lo) / span });
      grid.append(cell);
    }
  }
  const colorize = () => {
    const a = cssVar('--seq-lo'), b = cssVar('--seq-hi');
    for (const { cell, t } of cells) cell.style.background = mixHex(a, b, t);
  };
  colorize();
  return colorize;
}

// ------------------------------------------------------------------- map --
// nnOf(item) -> the item's nearest gallery neighbours (for the hover lines).
// onHover(item | null, clientX, clientY) lets the page own the tooltip.
export function drawMap(svg, items, { onPick, onHover, nnOf }) {
  svg.append(
    el('line', { class: 'gridline', x1: 7, y1: 55, x2: 93, y2: 55 }),
    el('line', { class: 'gridline', x1: 7, y1: 7, x2: 7, y2: 55 }),
  );
  const ax1 = el('text', { class: 'axis-label', x: 50, y: 59.5, 'text-anchor': 'middle' });
  ax1.textContent = 'principal component 1 →';
  const ax2 = el('text', { class: 'axis-label', x: 3.4, y: 31, 'text-anchor': 'middle',
                           transform: 'rotate(-90 3.4 31)' });
  ax2.textContent = 'principal component 2 →';
  svg.append(ax1, ax2);

  const hoverLayer = el('g', {});   // rings + neighbour lines live above images
  const at = new Map();             // item -> [x, y] on the svg
  for (const item of items) at.set(item, pos(item.map));

  for (const item of items) {
    const [x, y] = at.get(item);
    const img = el('image', {
      href: item.file, x: x - 3.5, y: y - 3.5, width: 7, height: 7,
      preserveAspectRatio: 'xMidYMid slice', tabindex: 0, role: 'button',
    });
    img.addEventListener('click', () => onPick(item));
    img.addEventListener('keydown', e => { if (e.key === 'Enter') onPick(item); });
    img.addEventListener('pointerenter', e => {
      hoverLayer.replaceChildren(el('circle', { class: 'hover-ring', cx: x, cy: y, r: 5.4 }));
      for (const nb of nnOf(item)) {
        const [nx, ny] = at.get(nb);
        hoverLayer.append(el('line', { class: 'nn-line', x1: x, y1: y, x2: nx, y2: ny }));
        hoverLayer.append(el('circle', { class: 'hover-ring', cx: nx, cy: ny, r: 4.2, 'stroke-dasharray': '1 0.7' }));
      }
      onHover(item, e.clientX, e.clientY);
    });
    img.addEventListener('pointermove', e => onHover(item, e.clientX, e.clientY));
    img.addEventListener('pointerleave', () => { hoverLayer.replaceChildren(); onHover(null); });
    svg.append(img);
  }
  svg.append(hoverLayer);
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

// ---------------------------------------------------------- fingerprints --
// A labeled fingerprint strip: label + <canvas>, one pixel column per value.
// Returns the row; redrawStrips() re-paints every strip on a theme flip.
const liveStrips = new Set();

export function stripRow(label, vec) {
  const row = document.createElement('div');
  row.className = 'strip-row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = `${label} (${vec.length},)`;
  const canvas = document.createElement('canvas');
  drawStrip(canvas, vec);
  liveStrips.add(canvas);
  canvas.dataset.strip = '1';
  canvas.vec = vec;
  row.append(lbl, canvas);
  return row;
}

export function redrawStrips() {
  for (const canvas of [...liveStrips]) {
    if (!canvas.isConnected) { liveStrips.delete(canvas); continue; }
    drawStrip(canvas, canvas.vec);
  }
}

function drawStrip(canvas, vec) {
  const NEG = hexToRgb(cssVar('--neg')), POS = hexToRgb(cssVar('--pos'));
  const MID = hexToRgb(cssVar('--hairline'));   // neutral zero, one step off surface
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
