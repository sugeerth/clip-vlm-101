// The live web phase's discovery client — a PARALLEL FAN-OUT across five
// independent, keyless, CORS-open image APIs. One provider down (or slow,
// or rate-limited) costs nothing but its own results, and discover()
// returns a per-provider ledger so the page can say exactly what happened.
// A silent web search is indistinguishable from a broken one.
//
//   openverse   api.openverse.org — WordPress/Creative-Commons search;
//               anonymous browser use documented (~1 req/s, ≤20/page)
//   commons     Wikimedia Action API with origin=* — the documented
//               anonymous-CORS mechanism
//   artic       Art Institute of Chicago — keyless, CORS * on the API
//               AND the IIIF image host, CC0/public-domain artworks
//   met         The Met Open Access — keyless, no registration, public
//               domain; two-step (search ids → object records)
//   inat        iNaturalist taxa search — open API, licensed nature photos
//
// All returned records are normalized: { name, thumb_url, source, license,
// provider } — thumb_url fetchable cross-origin for local embedding.
const TIMEOUT_MS = 8000;

async function jfetch(url, fetchFn) {
  const resp = await fetchFn(url, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
const norm = r => (r.thumb_url ? r : null);

export async function searchOpenverse(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({ q: term, page_size: n });
  const data = await jfetch(`https://api.openverse.org/v1/images/?${params}`, fetchFn);
  const license = x => !x.license ? ''
    : x.license === 'cc0' ? 'CC0'
    : x.license === 'pdm' ? 'Public domain'
    : `CC ${x.license.toUpperCase()} ${x.license_version ?? ''}`.trim();
  return (data.results ?? []).map(x => norm({
    name: x.title || x.id, thumb_url: x.thumbnail || x.url,
    source: x.foreign_landing_url || x.url, license: license(x),
    provider: 'openverse',
  })).filter(Boolean);
}

export async function searchCommons(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({
    action: 'query', format: 'json', origin: '*',
    generator: 'search', gsrsearch: `filetype:bitmap ${term}`,
    gsrnamespace: 6, gsrlimit: n,
    prop: 'imageinfo', iiprop: 'url|extmetadata', iiurlwidth: 384,
  });
  const data = await jfetch(`https://commons.wikimedia.org/w/api.php?${params}`, fetchFn);
  return Object.values(data?.query?.pages ?? {}).map(page => {
    const info = page.imageinfo?.[0] ?? {};
    return norm({
      name: (page.title ?? '').replace(/^File:/, ''),
      thumb_url: info.thumburl || info.url || '',
      source: info.descriptionurl || '',
      license: info.extmetadata?.LicenseShortName?.value ?? '',
      provider: 'commons',
    });
  }).filter(Boolean);
}

export async function searchArtic(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({ q: term, fields: 'id,title,image_id', limit: n });
  const data = await jfetch(`https://api.artic.edu/api/v1/artworks/search?${params}`, fetchFn);
  return (data.data ?? []).map(x => norm({
    name: x.title || String(x.id),
    thumb_url: x.image_id
      ? `https://www.artic.edu/iiif/2/${x.image_id}/full/400,/0/default.jpg` : '',
    source: `https://www.artic.edu/artworks/${x.id}`,
    license: 'CC0 / public domain',
    provider: 'artic',
  })).filter(Boolean);
}

export async function searchMet(term, n, fetchFn = fetch) {
  const base = 'https://collectionapi.metmuseum.org/public/collection/v1';
  const found = await jfetch(`${base}/search?q=${encodeURIComponent(term)}&hasImages=true`, fetchFn);
  const ids = (found.objectIDs ?? []).slice(0, n);
  const objects = await Promise.allSettled(
    ids.map(id => jfetch(`${base}/objects/${id}`, fetchFn)));
  return objects.filter(o => o.status === 'fulfilled').map(o => o.value)
    .filter(x => x.isPublicDomain && (x.primaryImageSmall || x.primaryImage))
    .map(x => norm({
      name: x.title || String(x.objectID),
      thumb_url: x.primaryImageSmall || x.primaryImage,
      source: x.objectURL || `https://www.metmuseum.org/art/collection/search/${x.objectID}`,
      license: 'Open Access (CC0)',
      provider: 'met',
    })).filter(Boolean);
}

export async function searchINat(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({ q: term, per_page: n });
  const data = await jfetch(`https://api.inaturalist.org/v1/taxa?${params}`, fetchFn);
  return (data.results ?? []).filter(x => x.default_photo).map(x => norm({
    name: x.preferred_common_name || x.name,
    thumb_url: x.default_photo.medium_url || x.default_photo.square_url || '',
    source: `https://www.inaturalist.org/taxa/${x.id}`,
    license: x.default_photo.license_code
      ? x.default_photo.license_code.toUpperCase().replace('CC-', 'CC ') : '',
    provider: 'inaturalist',
  })).filter(Boolean);
}

export const PROVIDERS = [
  ['openverse', searchOpenverse],
  ['commons', searchCommons],
  ['artic', searchArtic],
  ['met', searchMet],
  ['inaturalist', searchINat],
];

export const MAX_WEB_RESULTS = 12;

// Fan out to every provider AT ONCE; merge what comes back (deduped by
// thumbnail), and report per-provider outcomes in `tried`.
export async function discover(term, nPer = 4, fetchFn = fetch, providers = PROVIDERS) {
  const settled = await Promise.allSettled(
    providers.map(([, fn]) => fn(term, nPer, fetchFn)));
  const tried = [], records = [], seen = new Set();
  settled.forEach((s, i) => {
    const provider = providers[i][0];
    if (s.status === 'fulfilled') {
      tried.push({ provider, ok: true, count: s.value.length });
      for (const r of s.value) {
        if (!seen.has(r.thumb_url)) { seen.add(r.thumb_url); records.push(r); }
      }
    } else {
      tried.push({ provider, ok: false, error: String(s.reason?.message ?? s.reason) });
    }
  });
  return { records: records.slice(0, MAX_WEB_RESULTS), tried };
}
