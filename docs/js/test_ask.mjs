// The browser answer engine must rank and answer exactly like answer.py.
// ask_fixture.json is written by `python3 answer.py --write-fixture` and
// re-checked by test_smoke.py; here the JS twin reproduces it, and the
// source fan-out is exercised against mocked APIs (no network).
// Run: node docs/js/test_ask.mjs
import { readFileSync } from 'node:fs';
import { tokens, chunk, bm25, rank, numberSources, extractive, plain, gather, retrieve } from './ask.js';

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
    if (url.includes('wikipedia')) return ok({ query: { pages: { 7: { pageid: 7, index: 1, title: 'Sky', fullurl: 'https://w/Sky', extract: 'intro' } } } });
    if (url.includes('algolia')) return ok({ hits: [{ objectID: '1', title: 'Blue sky', story_text: '<p>Blue light scatters most.</p>' }] });
    if (url.includes('openalex')) return ok({ results: [{ id: 'https://o/1', display_name: 'Optics', publication_year: 2020,
      abstract_inverted_index: { Rayleigh: [0], scattering: [1], 'dominates.': [2] } }] });
    throw new Error('unexpected ' + url);
  };
  const { docs, ledger, queries } = await gather('why is the sky blue', { fetchFn });
  check(queries.length === 2, 'question + its keyword form are both searched');
  check(ledger.stackexchange.error && ledger.wikipedia.docs === 2 && ledger.hackernews.docs === 1, 'the ledger reports each source honestly');
  check(docs.length === 3 && docs[0].text.startsWith('The sky is blue'), 'dedup by URL; top Wikipedia article read in full');
  check(docs.find(d => d.source === 'openalex').text === 'Rayleigh scattering dominates.', 'OpenAlex abstract rebuilt from its inverted index');
  const { sources, top } = retrieve('why is the sky blue', docs);
  check(sources[0].url === 'https://w/Sky' && top[0].n === 1, 'retrieve numbers the best source [1]');
  const capped = await gather('sky', { fetchFn, limit: 1 });
  check(capped.docs.length === 1, 'the document limit is enforced');
}

if (failed) { console.log('ask twin: FAILED'); process.exit(1); }
console.log('ask twin: all checks passed');
