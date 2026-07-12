// The live web phase's discovery client — now with two providers and
// receipts for FAILURE too. A silent web search that quietly shows nothing
// is indistinguishable from a broken one, so discover() reports exactly
// what it tried and what each provider said.
//
// Providers (in order):
//   openverse  api.openverse.org — the WordPress/Creative-Commons image
//              search. Anonymous browser use is supported (~1 req/s,
//              ≤20 results/page); results carry license + landing page.
//   commons    the MediaWiki Action API with origin=* — the documented
//              anonymous-CORS mechanism (mediawiki.org/wiki/API:Cross-site_requests).
//
// Both hosts also serve their thumbnails cross-origin, so the page can
// fetch the bytes and embed them locally with the vision tower.
const TIMEOUT_MS = 8000;

const norm = r => (r.thumb_url ? r : null);

export async function searchOpenverse(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({ q: term, page_size: n });
  const resp = await fetchFn(`https://api.openverse.org/v1/images/?${params}`,
    { signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  const license = x => !x.license ? ''
    : x.license === 'cc0' ? 'CC0'
    : x.license === 'pdm' ? 'Public domain'
    : `CC ${x.license.toUpperCase()} ${x.license_version ?? ''}`.trim();
  return (data.results ?? []).map(x => norm({
    name: x.title || x.id,
    thumb_url: x.thumbnail || x.url,
    source: x.foreign_landing_url || x.url,
    license: license(x),
    provider: 'openverse',
  })).filter(Boolean);
}

export async function searchCommons(term, n, fetchFn = fetch) {
  const params = new URLSearchParams({
    action: 'query', format: 'json', origin: '*',      // origin=* → anonymous CORS
    generator: 'search', gsrsearch: `filetype:bitmap ${term}`,
    gsrnamespace: 6, gsrlimit: n,
    prop: 'imageinfo', iiprop: 'url|extmetadata', iiurlwidth: 384,
  });
  const resp = await fetchFn(`https://commons.wikimedia.org/w/api.php?${params}`,
    { signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
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

export const PROVIDERS = [
  ['openverse', searchOpenverse],
  ['commons', searchCommons],
];

// Try providers in order; return the first non-empty result set PLUS a
// `tried` ledger so the page can say exactly why the web came up empty.
export async function discover(term, n = 6, fetchFn = fetch) {
  const tried = [];
  for (const [provider, fn] of PROVIDERS) {
    try {
      const records = await fn(term, n, fetchFn);
      tried.push({ provider, ok: true, count: records.length });
      if (records.length) return { records, tried };
    } catch (e) {
      tried.push({ provider, ok: false, error: String(e?.message ?? e) });
    }
  }
  return { records: [], tried };
}
