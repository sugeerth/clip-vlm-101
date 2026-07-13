// Model-free, network-free checks on the web-search fan-out — every fetch
// here is a mock. What we're pinning down: transient failures heal (one
// retry), permanent ones don't burn retries (4xx), browser error prose is
// translated for the ledger, and discover() merges + dedups + reports.
// Run: node docs/js/test_crawler.mjs
import { discover, describeError, searchArtic, MAX_WEB_RESULTS } from './crawler.js';

let failed = false;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`);
  failed ||= !cond;
};

const jsonResp = body => ({ ok: true, json: async () => body });
const httpResp = status => ({ ok: false, status });
const articBody = n => ({
  data: Array.from({ length: n }, (_, i) => ({ id: i, title: `art ${i}`, image_id: `img${i}` })),
});

// retry heals a one-off network reset (openverse's TLS cut, wi-fi wake…)
{
  let calls = 0;
  const flaky = async () => {
    if (++calls === 1) throw new TypeError('Load failed');
    return jsonResp(articBody(2));
  };
  const recs = await searchArtic('cat', 2, flaky);
  check(calls === 2 && recs.length === 2, 'one retry heals a transient network error');
}

// HTTP 4xx is the caller's fault — no retry, fail fast
{
  let calls = 0;
  const notFound = async () => { calls++; return httpResp(404); };
  const err = await searchArtic('cat', 2, notFound).catch(e => e);
  check(calls === 1 && err.message === 'HTTP 404', '4xx fails once, without retrying');
}

// HTTP 5xx might heal — retried, then surfaced if it doesn't
{
  let calls = 0;
  const flaky5xx = async () => (++calls === 1 ? httpResp(503) : jsonResp(articBody(1)));
  const recs = await searchArtic('cat', 1, flaky5xx);
  check(calls === 2 && recs.length === 1, '5xx gets one retry');
}

// describeError — the ledger speaks human, not WebKit
check(describeError(new DOMException('Fetch is aborted', 'AbortError')).includes('no answer'),
  'Safari "Fetch is aborted" → timeout prose');
check(describeError(new DOMException('signal timed out', 'TimeoutError')).includes('no answer'),
  'Chrome TimeoutError → timeout prose');
check(describeError(new TypeError('Failed to fetch')) === 'unreachable — blocked or offline',
  'network TypeError → unreachable prose');
check(describeError(new Error('HTTP 503')) === 'HTTP 503', 'HTTP errors pass through as-is');

// discover — merges providers, dedups by thumb_url, ledger says what happened
{
  const rec = (name, thumb) => ({ name, thumb_url: thumb, source: 's', license: 'CC0', provider: 'x' });
  const providers = [
    ['alpha', async () => [rec('a', 'u1'), rec('b', 'u2')]],
    ['beta', async () => [rec('c', 'u2'), rec('d', 'u3')]],   // u2 duplicates alpha's
    ['gamma', async () => { throw new DOMException('Fetch is aborted', 'AbortError'); }],
  ];
  const { records, tried } = await discover('q', 4, fetch, providers);
  check(records.length === 3 && new Set(records.map(r => r.thumb_url)).size === 3,
    'discover dedups shared thumbnails across providers');
  const ledger = Object.fromEntries(tried.map(t => [t.provider, t]));
  check(ledger.alpha.ok && ledger.alpha.count === 2, 'ledger counts a healthy provider');
  check(!ledger.gamma.ok && ledger.gamma.error.includes('no answer'),
    'ledger reports a dead provider in humanized prose');
}

// discover — the merged list is capped so the embed loop stays bounded
{
  const many = Array.from({ length: 30 }, (_, i) =>
    ({ name: `n${i}`, thumb_url: `u${i}`, source: 's', license: '', provider: 'x' }));
  const { records } = await discover('q', 30, fetch, [['alpha', async () => many]]);
  check(records.length === MAX_WEB_RESULTS, `merged results cap at ${MAX_WEB_RESULTS}`);
}

if (failed) { console.error('some crawler.js checks FAILED'); process.exit(1); }
console.log('all crawler.js checks passed');
