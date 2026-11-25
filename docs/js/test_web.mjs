// Model-free, network-free checks that web.js mirrors web.py — the internet-scale
// clip-retrieval client (parse + dedup + sort + retry), with a mocked backend.
// Run: node docs/js/test_web.mjs  (CI runs it too).
import { search, dedup, reachError, DEFAULT_BACKEND, DEFAULT_INDEX } from './web.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const J = x => JSON.stringify(x);

// a canned clip-retrieval backend: a dup URL and a blank-url hit, out of order.
const ok = body => ({ ok: true, json: async () => {
  const q = JSON.parse(body).text || '';
  return [
    { url: 'https://ex.com/a.jpg', caption: `${q} one`, similarity: 0.42 },
    { url: 'https://ex.com/b.jpg', caption: `${q} two`, similarity: 0.55 },
    { url: 'https://ex.com/a.jpg', caption: `${q} dup`, similarity: 0.30 },
    { url: '', caption: 'no url', similarity: 0.9 },
    { url: 'https://ex.com/c.jpg', caption: `${q} three`, similarity: 0.48 },
  ];
} });

check(DEFAULT_BACKEND.includes('knn') && DEFAULT_INDEX.includes('laion'), 'defaults point at a clip-retrieval LAION backend');

const fakeFetch = async (url, opts) => ok(opts.body);
const got = await search('cat', { n: 10 }, fakeFetch);
check(J(got.map(r => r.url)) === J(['https://ex.com/b.jpg', 'https://ex.com/c.jpg', 'https://ex.com/a.jpg']),
  'dedup + sort by similarity (matches web.py)');
check(got.every(r => r.url), 'blank-url hits are dropped');
check(got[0].similarity >= got[got.length - 1].similarity, 'sorted descending by similarity');
check(got[0].caption === 'cat two', 'the query is sent to the backend and captions come back');

// dedup() directly
check(dedup([{ url: 'x' }, { url: 'x' }, { url: 'y' }]).length === 2, 'dedup keeps first of each url');

// retries: fail twice, then succeed
let calls = 0;
const flaky = async (url, opts) => { if (++calls < 3) throw new Error('network'); return ok(opts.body); };
const r2 = await search('dog', {}, flaky);
check(r2.length === 3 && calls === 3, 'retries transient failures, then succeeds');

// a permanently-dead backend surfaces a friendly reach error
let threw = null;
try { await search('x', {}, async () => { throw new Error('Failed to fetch'); }); } catch (e) { threw = e; }
check(threw !== null && /unreachable/.test(reachError(threw)), 'a dead backend throws → reachError explains it');

// image→image: an imageUrl is sent instead of text
let sawImageUrl = false;
await search('', { imageUrl: 'https://ex.com/q.jpg' }, async (url, opts) => {
  sawImageUrl = JSON.parse(opts.body).image_url === 'https://ex.com/q.jpg'; return ok(opts.body);
});
check(sawImageUrl, 'image→image sends image_url, not text');

if (failed) { console.error('some web.js checks FAILED'); process.exit(1); }
console.log('all web.js checks passed');
