// The browser answer engine must rank and answer exactly like answer.py.
// ask_fixture.json is written by `python3 answer.py --write-fixture` and
// re-checked by test_smoke.py; here the JS twin reproduces it, and the
// source fan-out is exercised against mocked APIs (no network).
// Run: node docs/js/test_ask.mjs
import { readFileSync } from 'node:fs';
import { tokens, chunk, bm25, rank, numberSources, extractive, plain, gather, retrieve, rankImages } from './ask.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const fx = JSON.parse(readFileSync(new URL('./ask_fixture.json', import.meta.url)));

check(JSON.stringify(tokens('Why is the Sky blue? It\'s Rayleigh-scattering, 4th power!')) === JSON.stringify(fx.tokens),
  'tokens match Python');
check(JSON.stringify(chunk(fx.chunk_text, 20)) === JSON.stringify(fx.chunk), 'sentence chunking matches Python');
for (const c of fx.cases) {
  const s = bm25(c.q, fx.passages.map(p => p.text));
  check(s.every((x, i) => Math.abs(x - c.bm25[i]) < 1e-8), `bm25 matches Python — "${c.q}"`);
  const top = rank(c.q, fx.passages.map(p => ({ ...p })), 4);
  numberSources(top);
  check(JSON.stringify(top.map(p => [p.src, p.n, p.score])) === JSON.stringify(c.rank), `rank + citation numbers match — "${c.q}"`);
  check(extractive(c.q, top, 30) === c.extractive, `extractive answer is byte-identical — "${c.q}"`);
}

check(plain('<p>a &amp; b&#x27;s</p><p>c &#x2F; d</p>') === "a & b's. c / d.", 'HTML → plain text decodes entities');

// fan-out: one source down costs only its own docs; dedup across sources; limit holds
{
  const ok = body => ({ ok: true, json: async () => body });
  const fetchFn = async url => {
    if (url.includes('stackexchange')) throw new TypeError('Failed to fetch');
    if (url.includes('pageids=')) return ok({ query: { pages: { 7: { extract: 'The sky is blue because of Rayleigh scattering. Full article text.' } } } });
    if (url.includes('wikipedia')) return ok({ query: { pages: { 7: { pageid: 7, index: 1, title: 'Sky', fullurl: 'https://w/Sky', extract: 'intro',
      thumbnail: { source: 'https://upload/sky.jpg' } } } } });
    if (url.includes('algolia')) return ok({ hits: [{ objectID: '1', title: 'Blue sky', story_text: '<p>Blue light scatters most.</p>' }] });
    if (url.includes('openalex')) return ok({ results: [{ id: 'https://o/1', display_name: 'Optics', publication_year: 2020,
      abstract_inverted_index: { Rayleigh: [0], scattering: [1], 'dominates.': [2] } }] });
    if (url.includes('api.github.com')) return ok({ items: [{ html_url: 'https://github.com/a/sky', full_name: 'a/sky',
      description: 'sky renderer', topics: ['optics'], stargazers_count: 9, language: 'Rust' }] });
    throw new Error('unexpected ' + url);
  };
  const imageFn = async () => ({ records: [{ thumb_url: 'https://i/1.jpg', source: 'https://p/1', name: 'blue sky photo',
    license: 'CC0', provider: 'openverse' }], tried: [{ provider: 'openverse', ok: true, count: 1 }, { provider: 'met', ok: false, error: 'HTTP 503' }] });
  const { docs, ledger, queries, apiImages } = await gather('why is the sky blue', { fetchFn, imageFn });
  check(docs[0].images[0].url === 'https://upload/sky.jpg', 'Wikipedia thumbnails ride along as page images');
  check(apiImages[0].alt === 'blue sky photo' && ledger.openverse.docs === 1 && ledger.met.error === 'HTTP 503',
    'image providers join the fan-out and the ledger');
  const imgs = rankImages('why is the sky blue', docs, apiImages, retrieve('why is the sky blue', docs).sources.reduce((m, s) => m.set(s.url, s.n), new Map()));
  check(imgs[0].url === 'https://upload/sky.jpg' && imgs[0].n === 1, 'the cited page\'s image ranks first');
  check(queries.length === 2, 'question + its keyword form are both searched');
  check(ledger.stackexchange.error && ledger.wikipedia.docs === 2 && ledger.hackernews.docs === 1, 'the ledger reports each source honestly');
  check(docs.length === 4 && docs[0].text.startsWith('The sky is blue'), 'dedup by URL; top Wikipedia article read in full');
  check(docs.find(d => d.source === 'github').text.includes('9 stars, written in Rust'), 'GitHub repositories become documents');
  check(docs.find(d => d.source === 'openalex').text === 'Rayleigh scattering dominates.', 'OpenAlex abstract rebuilt from its inverted index');
  const { sources, top } = retrieve('why is the sky blue', docs);
  check(sources[0].url === 'https://w/Sky' && top[0].n === 1, 'retrieve numbers the best source [1]');
  const capped = await gather('sky', { fetchFn, limit: 1, imageFn });
  check(capped.docs.length === 1, 'the document limit is enforced');
}

{
  const fi = fx.images;
  const got = rankImages('why is the sky blue', fi.docs, fi.api, fi.cited, 3);
  check(JSON.stringify(got.map(im => [im.url, im.n, im.score])) === JSON.stringify(fi.ranked),
    'image ranking (words + cited bonus, dedup) matches multimodal.rank_images');
}

// the bridge to CLIP·search: same tags and same gallery ranking as multimodal.py, on the real gallery
{
  const { knnTags, gallerySearch } = await import('./bridge.js');
  const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
  const knn = items.map(it => knnTags(it.text_emb, items));
  check(JSON.stringify(knn) === JSON.stringify(fx.bridge.knn), 'knnTags: a caption\'s CLIP neighbours vote the same tags as Python (14/14)');
  for (const [q, want] of Object.entries(fx.bridge.search)) {
    const got = gallerySearch(q, items).map(g => [g.file, g.score]);
    check(JSON.stringify(got) === JSON.stringify(want), `gallerySearch matches multimodal.gallery_search — "${q}"`);
  }
  const qv = items[4].text_emb;                    // the cat's caption, as a CLIP query vector
  check(gallerySearch('', items, 1, qv)[0].file === 'images/004_cat.jpg', 'with a CLIP vector, the gallery ranks by pixels');
}

if (failed) { console.log('ask twin: FAILED'); process.exit(1); }
console.log('ask twin: all checks passed');
