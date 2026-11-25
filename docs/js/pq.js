// pq.py's browser twin — an optimised-product-quantized search over a slice of
// the million, entirely in this tab. DOM-free: load a pack, hand it a query
// vector, get row ids back. Two tiers, exactly mirroring pq.py:
//
//   base    pq_books.bin     (m × ks × sub float32 — the OPQ codebooks)
//           pq_codes.bin     (n × m uint8 — 64 bytes per dish, in rotated space)
//           opq_rotation.bin (d × d float32 — the learned rotation)
//           meta.json        ({ manifest, items: [name, cafe, urlSuffix] })
//
//   refine  refine_i8.bin    (n × d int8 — the same dishes, one byte per dim)
//           refine_scale.bin (d float32 — the per-dimension dequant scale)
//
//   search  stage 1: rotate the query by R, then ADC — one table of dot
//           products per subspace, a row's score = m table LOOKUPS summed
//           (no multiplies per row). stage 2 (only if the refine tier is
//           loaded): re-score the top few hundred exactly on the int8 vectors.

export async function loadPack(base = 'million', onProgress = () => {}) {
  const get = async (name, as) => {
    const r = await fetch(`${base}/${name}`);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    onProgress(name);
    return as === 'json' ? r.json() : r.arrayBuffer();
  };
  const [books, codes, rot, meta] = await Promise.all([
    get('pq_books.bin'), get('pq_codes.bin'), get('opq_rotation.bin'), get('meta.json', 'json'),
  ]);
  const m = meta.manifest;
  const d = m.d ?? (m.m * m.sub);
  if (books.byteLength !== m.m * m.ks * m.sub * 4)
    throw new Error('codebook size mismatch — pack is stale or truncated');
  if (codes.byteLength !== m.n * m.m)
    throw new Error('codes size mismatch — pack is stale or truncated');
  if (rot.byteLength !== d * d * 4)
    throw new Error('rotation size mismatch — pack is stale or truncated');
  return { books: new Float32Array(books), codes: new Uint8Array(codes),
           rotation: new Float32Array(rot), d,
           manifest: m, items: meta.items, refine: null };
}

// The refine tier: ~51 MB, so it's fetched only when the user asks for
// precision. loadRefine attaches it to the pack in place.
export async function loadRefine(pack, base = 'million', onProgress = () => {}) {
  const m = pack.manifest, d = pack.d;
  const get = async name => {
    const r = await fetch(`${base}/${name}`);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    onProgress(name);
    return r.arrayBuffer();
  };
  const [i8, scale] = await Promise.all([
    get(m.refine.codes), get(m.refine.scale),
  ]);
  if (i8.byteLength !== m.n * d) throw new Error('refine codes size mismatch');
  if (scale.byteLength !== d * 4) throw new Error('refine scale size mismatch');
  pack.refine = { i8: new Int8Array(i8), scale: new Float32Array(scale) };
  return pack;
}

// R · q — the learned rotation, applied query-side. R x · R q = x · q, so this
// changes nothing about the ranking except which basis the codes were built in.
function rotate(q, pack) {
  const { rotation, d } = pack;
  const out = new Float32Array(d);
  for (let i = 0, o = 0; i < d; i++, o += d) {
    let s = 0;
    for (let j = 0; j < d; j++) s += rotation[o + j] * q[j];
    out[i] = s;
  }
  return out;
}

// (1) the query-side tables: T[j][c] = (R·q)_subvector_j · centroid_c
export function adcTables(q, pack) {
  const { m, ks, sub } = pack.manifest;
  const { books } = pack;
  const qr = rotate(q, pack);
  const T = new Float32Array(m * ks);
  for (let j = 0; j < m; j++) {
    const qOff = j * sub, tOff = j * ks, bOff = j * ks * sub;
    for (let c = 0; c < ks; c++) {
      let s = 0;
      const o = bOff + c * sub;
      for (let dd = 0; dd < sub; dd++) s += books[o + dd] * qr[qOff + dd];
      T[tOff + c] = s;
    }
  }
  return T;
}

// (2) the coarse scan: score every row by lookups, keep the best k (one pass,
// insertion into a tiny sorted list — n·m adds total).
export function pqSearch(pack, T, k = 24) {
  const { n, m, ks } = pack.manifest;
  const { codes } = pack;
  const topIds = new Int32Array(k).fill(-1);
  const topScores = new Float32Array(k).fill(-Infinity);
  for (let i = 0, off = 0; i < n; i++, off += m) {
    let s = 0;
    for (let j = 0; j < m; j++) s += T[j * ks + codes[off + j]];
    if (s <= topScores[k - 1]) continue;
    let at = k - 1;                    // insert, keeping the list sorted
    while (at > 0 && topScores[at - 1] < s) {
      topScores[at] = topScores[at - 1]; topIds[at] = topIds[at - 1]; at--;
    }
    topScores[at] = s; topIds[at] = i;
  }
  const out = [];
  for (let r = 0; r < k && topIds[r] >= 0; r++)
    out.push({ id: topIds[r], score: topScores[r] });
  return out;
}

// stage two: re-score coarse candidates EXACTLY on the int8 vectors. The
// per-dim dequant folds into the query (score = i8_row · (scale ⊙ q)), so the
// hot loop is one integer×float multiply-add per dimension — the same maths
// pq.py's recall_rerank measures, lookup for lookup.
export function rerank(pack, candidates, q, k = 24) {
  const { i8, scale } = pack.refine;
  const d = pack.d;
  const qs = new Float32Array(d);
  for (let j = 0; j < d; j++) qs[j] = scale[j] * q[j];
  const scored = candidates.map(({ id }) => {
    let s = 0;
    const off = id * d;
    for (let j = 0; j < d; j++) s += i8[off + j] * qs[j];
    return { id, score: s };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// convenience: the whole two-stage search in one call. With no refine tier
// loaded it's just the coarse scan; with it, coarse nominates `cand`, the int8
// tier re-ranks to the exact top-k.
export function search(pack, q, k = 24, cand = 400) {
  const coarse = pqSearch(pack, adcTables(q, pack), pack.refine ? cand : k);
  return pack.refine ? rerank(pack, coarse, q, k) : coarse;
}
