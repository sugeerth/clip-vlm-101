// Mirror of web.py — search the whole internet's images, free, in CLIP space.
//
// The gallery is 14 images because WE didn't index more. The internet's images are
// ALREADY indexed — free, at billion scale, in this repo's CLIP space: LAION crawled
// ~5-6B image-text pairs from Common Crawl and released them CLIP-embedded (the
// safety-cleaned Re-LAION-5B), and rom1504's clip-retrieval serves KNN over them. So
// "search the internet" isn't a crawl you run — it's a query you send, and the index
// cost is $0 because someone already paid it and gave it away.
//
// This is the browser client. It POSTs to any clip-retrieval-compatible KNN backend
// (public or your own) and returns { url, caption, similarity } — the SAME record
// shape the local gallery uses, so the same grid + ranking-quality agent work over
// 14 images or 5 billion. The backend encodes the query, so the client is space-
// agnostic; point it at a ViT-B/32 index to match this repo. fetchFn is injectable
// so tests exercise the whole path with a mock — no network.
// The public LAION backend has been intermittent-to-down since the Dec-2023
// takedown — best-effort only; self-host clip-retrieval for reliability. The
// server encodes your text, so the client is embedding-space-agnostic (the 5B
// index is ViT-L/14; a LAION-400M B/32 index matches this repo's space exactly).
export const DEFAULT_BACKEND = 'https://knn.laion.ai/knn-service';
export const DEFAULT_INDEX = 'laion5B-L-14';     // clip-retrieval's public 5B index (ViT-L/14)
const RETRIES = 3;

async function jpost(url, payload, fetchFn) {
  let last = null;
  for (let attempt = 0; attempt < RETRIES; attempt++) {
    try {
      const resp = await fetchFn(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout?.(30000),
      });
      if (!resp.ok) throw new Error(`backend ${resp.status}`);
      return await resp.json();
    } catch (e) { last = e; }                      // transient → retry
  }
  throw last;
}

const record = h => ({
  url: h.url || h.image || '',
  caption: (h.caption || '').trim(),
  similarity: Number(h.similarity ?? h.score ?? 0) || 0,
});

// drop repeated URLs (the web has many copies of one image), keep order
export function dedup(records) {
  const seen = new Set(), out = [];
  for (const r of records) if (r.url && !seen.has(r.url)) { seen.add(r.url); out.push(r); }
  return out;
}

// query the internet-scale index. text OR imageUrl; up to n deduped, sorted records.
export async function search(query, {
  n = 20, backend = DEFAULT_BACKEND, index = DEFAULT_INDEX, imageUrl = null, safe = true,
} = {}, fetchFn = fetch) {
  const payload = {
    modality: 'image', num_images: n, num_result_ids: n, indice_name: index,
    deduplicate: true, use_safety_model: safe, use_violence_detector: safe,
  };
  if (imageUrl) payload.image_url = imageUrl; else payload.text = query;
  let hits = await jpost(backend, payload, fetchFn);
  if (!Array.isArray(hits)) hits = hits?.results || hits?.images || [];
  const records = dedup(hits.filter(h => h.url || h.image).map(record));
  records.sort((a, b) => b.similarity - a.similarity);
  return records.slice(0, n);
}

// friendly one-liner for the UI when the backend can't be reached
export function reachError(err) {
  const s = String(err?.message || err);
  if (/Failed to fetch|load failed|network|CORS/i.test(s))
    return 'the KNN backend is unreachable — it may be down, or blocking cross-origin requests. ' +
           'Point --backend at your own clip-retrieval server.';
  if (/\b(4\d\d|5\d\d)\b/.test(s)) return `the backend returned an error (${s}).`;
  return s;
}
