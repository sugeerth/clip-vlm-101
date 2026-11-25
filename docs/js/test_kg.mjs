// Network-free checks that kg.js mirrors kg.py — the knowledge-graph grounding
// (resolve + alias→tag expansions + query expansion + cached live lookup), against
// canned Wikidata API responses. Run: node docs/js/test_kg.mjs  (CI runs it too).
import { resolve, expansions, expandWords, liveResolve, API } from './kg.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const J = x => JSON.stringify(x);

// the same canned backend kg.py's selftest uses
const canned = url => {
  const q = new URL(url).searchParams;
  const body =
    q.get('action') === 'wbsearchentities'
      ? (q.get('search') === 'unknownium' ? { search: [] }
        : { search: [{ id: 'Q146', label: 'house cat' }] })
      : q.get('ids') === 'Q146'
        ? { entities: { Q146: {
            labels: { en: { value: 'house cat' } },
            descriptions: { en: { value: 'domesticated feline' } },
            aliases: { en: [{ value: 'feline' }, { value: 'housecat' }, { value: 'Felis catus' }] },
            claims: { P279: [{ mainsnak: { datavalue: { value: { id: 'Q39201' } } } }] },
          } } }
        : { entities: { Q39201: { labels: { en: { value: 'pet' } } } } };
  return { ok: true, json: async () => body };
};
const fakeFetch = async url => canned(url);

check(API.includes('wikidata.org'), 'the external graph is Wikidata');

const ent = await resolve('cat', fakeFetch);
check(ent.id === 'Q146' && ent.description === 'domesticated feline', 'resolve: entity + description (matches kg.py)');
check(ent.aliases.includes('feline') && J(ent.related) === J(['pet']), 'resolve: aliases + is-a labels');
check(await resolve('unknownium', fakeFetch) === null, 'resolve: unknown terms stay null');

// the committed-slice hot path
const kg = { entities: { cat: ent } };
const exp = expansions(kg);
check(exp['feline'] === 'cat' && exp['house cat'] === 'cat' && !('cat' in exp),
  'expansions: alias→tag map, never self (matches kg.expansions)');

const { words, mapped } = expandWords(['feline', 'snow'], exp);
check(J(words) === J(['feline', 'snow', 'cat']) && J(mapped) === J([{ from: 'feline', to: 'cat' }]),
  'expandWords: "feline" pulls in cat, and says why');
const noop = expandWords(['cat'], exp);
check(J(noop.words) === J(['cat']) && noop.mapped.length === 0, 'expandWords: already-canonical words add nothing');

// the live path caches forever — a term is fetched once, ever (localStorage-free env: still resolves)
let calls = 0;
const counting = async url => { calls++; return canned(url); };
globalThis.localStorage = (() => { let store = {}; return {
  getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; }, } })();
const live1 = await liveResolve('cat', counting);
const callsAfterFirst = calls;
const live2 = await liveResolve('cat', counting);
check(live1.id === 'Q146' && live2.id === 'Q146' && calls === callsAfterFirst,
  'liveResolve: second lookup is served from cache, zero fetches');
const missing = await liveResolve('unknownium', counting);
const callsAfterMiss = calls;
await liveResolve('unknownium', counting);
check(missing === null && calls === callsAfterMiss, 'liveResolve: "KG does not know it" is cached too');
// offline → null, and NOT cached (so it retries when the network returns)
const offline = await liveResolve('newterm', async () => { throw new Error('offline'); });
let retried = 0;
await liveResolve('newterm', async url => { retried++; return canned(url); });
check(offline === null && retried > 0, 'liveResolve: an offline failure is not cache-poisoned');

if (failed) { console.error('some kg.js checks FAILED'); process.exit(1); }
console.log('all kg.js checks passed');
