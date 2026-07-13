// pq.py's browser twin — product-quantized search over a slice of the
// million, entirely in this tab. DOM-free: load a pack, hand it a query
// vector, get row ids back.
//
//   pack     pq_books.bin  (m × ks × sub float32 — the codebooks)
//            pq_codes.bin  (n × m uint8 — 64 bytes per dish)
//            meta.json     ({ manifest, items: [name, cafe, urlSuffix] })
//
//   search   ADC, same two steps as pq.py: (1) one table of dot products
//            per subspace — m·ks tiny dots, microseconds; (2) a row's
//            score = m table LOOKUPS summed. No multiplies per row.

export async function loadPack(base = 'million', onProgress = () => {}) {
  const get = async (name, as) => {
    const r = await fetch(`${base}/${name}`);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    onProgress(name);
    return as === 'json' ? r.json() : r.arrayBuffer();
  };
  const [books, codes, meta] = await Promise.all([
    get('pq_books.bin'), get('pq_codes.bin'), get('meta.json', 'json'),
  ]);
  const m = meta.manifest;
  if (books.byteLength !== m.m * m.ks * m.sub * 4)
    throw new Error('codebook size mismatch — pack is stale or truncated');
  if (codes.byteLength !== m.n * m.m)
    throw new Error('codes size mismatch — pack is stale or truncated');
  return { books: new Float32Array(books), codes: new Uint8Array(codes),
           manifest: m, items: meta.items };
}

// (1) the query-side tables: T[j][c] = q_subvector_j · centroid_c
export function adcTables(q, pack) {
  const { m, ks, sub } = pack.manifest;
  const { books } = pack;
  const T = new Float32Array(m * ks);
  for (let j = 0; j < m; j++) {
    const qOff = j * sub, tOff = j * ks, bOff = j * ks * sub;
    for (let c = 0; c < ks; c++) {
      let s = 0;
      const o = bOff + c * sub;
      for (let d = 0; d < sub; d++) s += books[o + d] * q[qOff + d];
      T[tOff + c] = s;
    }
  }
  return T;
}

// (2) the scan: score every row by lookups, keep the best k (one pass,
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
