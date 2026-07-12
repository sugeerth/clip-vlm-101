// Mirror of crawler.py's discover() — the live half of "every search crawls
// the web". Wikimedia Commons is the one great image source a BROWSER can
// crawl: its API and its image host both answer cross-origin requests, and
// every result arrives with attribution and a license. The page asks for a
// handful of freely-licensed matches, embeds the thumbnails locally with
// the vision tower, and ranks them — live discovery, receipts included.
const API = 'https://commons.wikimedia.org/w/api.php';

export async function discover(term, n = 6, width = 384, fetchFn = fetch) {
  const params = new URLSearchParams({
    action: 'query', format: 'json', origin: '*',      // origin=* → CORS
    generator: 'search', gsrsearch: `filetype:bitmap ${term}`,
    gsrnamespace: 6, gsrlimit: n,
    prop: 'imageinfo', iiprop: 'url|extmetadata', iiurlwidth: width,
  });
  const data = await (await fetchFn(`${API}?${params}`)).json();
  return Object.values(data?.query?.pages ?? {}).map(page => {
    const info = page.imageinfo?.[0] ?? {};
    const meta = info.extmetadata ?? {};
    return {
      name: (page.title ?? '').replace(/^File:/, ''),
      thumb_url: info.thumburl || info.url || '',
      source: info.descriptionurl || '',
      license: meta.LicenseShortName?.value ?? '',
    };
  }).filter(r => r.thumb_url);
}
