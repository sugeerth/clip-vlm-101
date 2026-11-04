// ask.js — the browser twin of answer.py: read ~200 documents, say one cited line.
//
//   question ─► fan out to 5 keyless, CORS-open sources at once ─► ≤200 docs
//            ─► sentence-aligned passages ─► BM25 top-k (≤2 per source)
//            ─► Claude streams ≤N words with [n] cites   (no key: extractive)
//
// A browser cannot scrape arbitrary sites (CORS forbids reading another
// origin's HTML), so the page reads the sources that DO allow it: Wikipedia
// (intro extracts + full text of the top articles), Hacker News (stories and
// comments), Stack Exchange (questions and answers) and OpenAlex (paper
// abstracts). answer.py, with Scrapling, reads any page on the web.
//
// tokens / chunk / bm25 / rank / numberSources / extractive mirror answer.py
// line for line; test_ask.mjs pins them to Python-computed fixtures.

const STOP = new Set(('a an and are as at be by for from has have how i in is it its of on or ' +
  'that the this to was were what when where which who why will with does do ' +
  'can you your vs').split(' '));
const SENT = /(?<=[.!?])\s+/;

export const tokens = text => (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(t => !STOP.has(t));

export function chunk(text, words = 80) {
  const out = []; let cur = [], n = 0;
  for (let s of text.split(SENT)) {
    s = s.trim();
    if (!s) continue;
    cur.push(s); n += s.split(/\s+/).length;
    if (n >= words) { out.push(cur.join(' ')); cur = []; n = 0; }
  }
  if (cur.length) out.push(cur.join(' '));
  return out;
}

export function bm25(query, docs, k1 = 1.2, b = 0.75) {
  const q = [...new Set(tokens(query))];
  const toks = docs.map(tokens);
  const N = docs.length || 1;
  const avg = toks.reduce((s, d) => s + d.length, 0) / N || 1;
  const df = Object.fromEntries(q.map(t => [t, toks.filter(d => d.includes(t)).length]));
  return toks.map(d => {
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    let s = 0;
    for (const t of q) {
      const f = tf.get(t) ?? 0;
      if (f) {
        const idf = Math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5));
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * d.length / avg));
      }
    }
    return s;
  });
}

// stable sort by descending score, ties keep input order — same as Python's sorted()
const argsortDesc = s => s.map((_, i) => i).sort((a, b) => s[b] - s[a] || a - b);

export function rank(query, passages, k = 10, perSource = 2) {
  const scores = bm25(query, passages.map(p => p.text));
  const out = [], seen = new Map();
  for (const i of argsortDesc(scores)) {
    if (scores[i] <= 0 || out.length >= k) break;
    const src = passages[i].src;
    if ((seen.get(src) ?? 0) < perSource) {
      seen.set(src, (seen.get(src) ?? 0) + 1);
      out.push({ ...passages[i], score: Math.round(scores[i] * 1e4) / 1e4 });
    }
  }
  return out;
}

export function numberSources(top) {
  const num = new Map();
  for (const p of top) { if (!num.has(p.src)) num.set(p.src, num.size + 1); p.n = num.get(p.src); }
  return num;
}

const words = s => s.split(/\s+/).filter(Boolean).length;

export function extractive(query, top, budget = 45) {
  const cands = [];
  for (const p of top) for (const s of p.text.split(SENT)) if (words(s) >= 5 && !s.trimEnd().endsWith('?')) cands.push([s, p.n]);
  if (!cands.length) return 'Not in sources.';
  const scores = bm25(query, cands.map(c => c[0]));
  const picked = [], used = [];
  for (const i of argsortDesc(scores)) {
    const [s, n] = cands[i];
    if (scores[i] <= 0) break;
    const ts = new Set(tokens(s));
    const dup = used.some(u => {
      const inter = [...ts].filter(t => u.has(t)).length;
      return inter / (new Set([...ts, ...u]).size || 1) > 0.5;
    });
    if (dup) continue;
    if (picked.length && words(s) > budget) continue;
    picked.push(`${s.trimEnd()} [${n}]`); used.push(ts);
    budget -= words(s);
    if (budget <= 0) break;
  }
  return picked.length ? picked.join(' ') : 'Not in sources.';
}

export function subqueries(question) {
  const bare = tokens(question).join(' ');
  return [...new Set([question, ...(bare && bare !== question.toLowerCase() ? [bare] : [])])];
}

// ---- the five sources: each returns [{url, title, text, source}] ----

const TIMEOUT_MS = 9000;
async function jget(url, fetchFn) {
  const r = await fetchFn(url, { signal: AbortSignal.timeout?.(TIMEOUT_MS) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', '#x27': "'", '#39': "'", '#x2F': '/', nbsp: ' ' };
export const plain = html => String(html ?? '')
  .replace(/<\/(p|div|li|pre|h\d)>|<br\s*\/?>/gi, '. ').replace(/<[^>]+>/g, ' ')
  .replace(/&(#x?[0-9a-f]+|\w+);/gi, (m, e) => ENT[e] ?? (e[0] === '#'
    ? String.fromCodePoint(parseInt(e[1] === 'x' ? e.slice(2) : e.slice(1), e[1] === 'x' ? 16 : 10)) : m))
  .replace(/\s+/g, ' ').replace(/(\.\s*){2,}/g, '. ').trim();
const WIKI = 'https://en.wikipedia.org/w/api.php?format=json&origin=*&action=query';

export const SOURCES = {
  async wikipedia(q, fetchFn) {
    const d = await jget(`${WIKI}&generator=search&gsrsearch=${encodeURIComponent(q)}&gsrlimit=20` +
      '&prop=extracts|info&inprop=url&exintro=1&explaintext=1&exlimit=20', fetchFn);
    const pages = Object.values(d.query?.pages ?? {}).sort((a, b) => a.index - b.index);
    // the top 4 articles are read in full, not just their intros
    const full = await Promise.all(pages.slice(0, 4).map(p =>
      jget(`${WIKI}&prop=extracts&explaintext=1&exsectionformat=plain&pageids=${p.pageid}`, fetchFn)
        .then(x => x.query.pages[p.pageid].extract).catch(() => null)));
    return pages.map((p, i) => ({ url: p.fullurl, title: p.title, text: full[i] || p.extract || '', source: 'wikipedia' }));
  },
  async hackernews(q, fetchFn) {
    const d = await jget(`https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(q)}` +
      '&tags=(story,comment)&hitsPerPage=60', fetchFn);
    return (d.hits ?? []).map(h => ({
      url: `https://news.ycombinator.com/item?id=${h.objectID}`,
      title: h.title || h.story_title || 'Hacker News comment',
      text: plain(h.story_text || h.comment_text || h.title), source: 'hackernews',
    }));
  },
  async stackexchange(q, fetchFn) {
    const d = await jget('https://api.stackexchange.com/2.3/search/excerpts?order=desc&sort=relevance' +
      `&site=stackoverflow&pagesize=50&q=${encodeURIComponent(q)}`, fetchFn);
    return (d.items ?? []).map(x => ({
      url: x.item_type === 'answer' ? `https://stackoverflow.com/a/${x.answer_id}`
        : `https://stackoverflow.com/q/${x.question_id}`,
      title: plain(x.title), text: plain(x.body || x.excerpt), source: 'stackexchange',
    }));
  },
  async openalex(q, fetchFn) {
    const d = await jget(`https://api.openalex.org/works?per_page=40&search=${encodeURIComponent(q)}` +
      '&select=id,doi,display_name,abstract_inverted_index,publication_year', fetchFn);
    return (d.results ?? []).filter(w => w.abstract_inverted_index).map(w => {
      const words = [];
      for (const [t, pos] of Object.entries(w.abstract_inverted_index)) for (const i of pos) words[i] = t;
      return { url: w.doi || w.id, title: `${w.display_name} (${w.publication_year})`,
        text: words.filter(Boolean).join(' '), source: 'openalex' };
    });
  },
};

// Fan every query to every source at once; one source failing costs only its own docs.
export async function gather(question, { limit = 200, fetchFn = fetch, onProgress } = {}) {
  const queries = subqueries(question);
  const ledger = {};
  const jobs = [];
  for (const [name, fn] of Object.entries(SOURCES)) {
    ledger[name] = { docs: 0, error: null };
    // Wikipedia is cheap and deep: every sub-query; the rest: the question itself
    for (const q of name === 'wikipedia' ? queries : queries.slice(0, 1)) {
      jobs.push(fn(q, fetchFn).then(docs => {
        ledger[name].docs += docs.length; onProgress?.(ledger); return docs;
      }).catch(err => { ledger[name].error = String(err?.message ?? err); onProgress?.(ledger); return []; }));
    }
  }
  const seen = new Set(), docs = [];
  for (const d of (await Promise.all(jobs)).flat()) {
    if (d.text && d.url && !seen.has(d.url) && docs.length < limit) { seen.add(d.url); docs.push(d); }
  }
  return { docs, ledger, queries };
}

export function retrieve(question, docs, k = 10) {
  const passages = docs.flatMap(d => chunk(d.text).map(text => ({ text, src: d.url, title: d.title, source: d.source })));
  const top = rank(question, passages, k);
  const num = numberSources(top);
  const byUrl = new Map(top.map(p => [p.src, p]));
  const sources = [...num].map(([url, n]) => ({ n, url, title: byUrl.get(url).title, source: byUrl.get(url).source }));
  return { top, sources, passages: passages.length };
}

// ---- Claude: the official SDK, straight from the browser with the viewer's own key ----

export const SYSTEM = w => `You are a search engine's answer line. Answer the question in at most ${w} words, ` +
  'using ONLY the numbered sources. Lead with the answer itself: no preamble, no ' +
  'restating the question, no hedging filler. Put a citation like [2] after every ' +
  'claim. If the sources do not contain the answer, reply exactly: Not in sources.';

export async function claudeAnswer({ key, model, question, top, words, onText }) {
  const { default: Anthropic } = await import('https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.137.0/+esm');
  const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
  const ctx = top.map(p => `[${p.n}] ${p.title}\n${p.text}`).join('\n\n');
  const stream = client.beta.messages.stream({
    model, max_tokens: 2000, system: SYSTEM(words),
    output_config: { effort: 'low' },
    betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default',
    messages: [{ role: 'user', content: `Sources:\n${ctx}\n\nQuestion: ${question}` }],
  });
  stream.on('text', t => onText?.(t));
  const msg = await stream.finalMessage();
  if (msg.stop_reason === 'refusal') throw new Error('the model declined');
  return msg.content.filter(b => b.type === 'text').map(b => b.text).join('').trim();
}
