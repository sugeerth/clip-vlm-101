// ask.js — the browser twin of answer.py: read ~200 documents, say one cited line.
//
//   question ─► fan out to 5 keyless, CORS-open sources at once ─► ≤200 docs
//            ─► sentence-aligned passages ─► BM25 top-k (≤2 per source)
//            ─► Claude streams ≤N words with [n] cites   (no key: extractive)
//   images:  Wikipedia thumbnails + 5 image APIs ─► rankImages (words; CLIP pixels opt-in)
//   photo:   drop one in ─► Claude vision (with a key) or CLIP tags (keyless) ─► the query
//   firecrawl (viewer's key): the WHOLE web live — search + scraped pages + news + images;
//            paste a URL and Firecrawl's browser reads that page
//
// A browser cannot scrape arbitrary sites (CORS forbids reading another
// origin's HTML), so the page reads the sources that DO allow it: Wikipedia
// (intro extracts + full text of the top articles), Hacker News (stories and
// comments), Stack Exchange (questions and answers) and OpenAlex (paper
// abstracts) and GitHub (repositories). answer.py, with Scrapling, reads any
// page on the web; sources.py adds arXiv and Reddit there too.
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

// Twin of answer.EXP_STOP (generated from it) — words too common to say what a page is ABOUT.
const EXP_STOP = new Set('about above after again against all also although always amazing among another any anyone anything around awesome back beautiful because been before being below best between both breathtaking but came can cannot come could day days did does doing done down during each either else even ever every few first for from further get gets getting give given goes going gone good got great guide had has have having her here hers him his how however incredible into its just keep know last least less let like likely made magical make makes making many may maybe more most much must near need needs never new next not nothing now often once one only onto other others our ours out over own part per perhaps place put rather ready really right said same say see seen several shall she should show since some something sometimes soon spectacular still stunning such sure take than that the their them then there these they thing things think this those though through thus time times tips today together tonight too toward ultimate under until upon use used uses using very want was way ways well went were what whatever when where whether which while who whom whose why will with within without would year years yet you your yours'.split(' '));

// Pseudo-relevance feedback (RM3-lite), twin of answer.expand_query: words that dominate the
// top-m hits, appear in ≥ minDocs of them, and aren't in the question.
export function expandQuery(query, docs, m = 8, n = 6, minDocs = 2) {
  const scores = bm25(query, docs);
  const top = argsortDesc(scores).filter(i => scores[i] > 0).slice(0, m);
  const q = new Set(tokens(query)), toks = docs.map(tokens), sets = toks.map(t => new Set(t));
  const N = docs.length, w = new Map(), hits = new Map(), df = new Map();
  for (const i of top) {
    const tf = new Map();
    for (const t of toks[i]) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [t, c] of tf) {
      if (q.has(t) || t.length < 4 || /^\d+$/.test(t) || EXP_STOP.has(t)) continue;
      if (!df.has(t)) df.set(t, sets.filter(st => st.has(t)).length);
      w.set(t, (w.get(t) ?? 0) + c / toks[i].length * Math.log(1 + (N - df.get(t) + 0.5) / (df.get(t) + 0.5)));
      hits.set(t, (hits.get(t) ?? 0) + 1);
    }
  }
  return [...w.keys()].filter(t => hits.get(t) >= minDocs)
    .sort((a, b) => w.get(b) - w.get(a) || (a < b ? -1 : a > b ? 1 : 0)).slice(0, n);
}

export function rank(query, passages, k = 10, perSource = 2, expansion = []) {
  const texts = passages.map(p => p.text);
  let scores = bm25(query, texts);
  if (expansion.length) { const e = bm25(expansion.join(' '), texts); scores = scores.map((a, i) => a + 0.5 * e[i]); }
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

const JUNK = /^(watch|click|subscribe|sign up|read more|learn more|share|follow|see also|related|advertisement|cookie|skip to|jump to|fun fact|learn how|discover|book|sign in|log in)\b/i;
const CUE = /\b(because|caused by|due to|occurs? when|happens? when|results? from|is called|are called|which means)\b/i;
// Title Case lines are headings, not answers (twin of answer.headline)
export function headline(s) {
  const long = s.split(/\s+/).filter(w => w.length >= 4 && /^\p{L}/u.test(w));
  return long.length >= 3 && long.filter(w => w[0] !== w[0].toLowerCase()).length / long.length >= 0.7;
}

export function extractive(query, top, budget = 45, expansion = []) {
  const cands = [];
  for (const p of top) for (const s of p.text.split(SENT)) {
    if (words(s) >= 8 && !s.trimEnd().endsWith('?') && !JUNK.test(s.trim()) && !headline(s)) cands.push([s, p.n]);
  }
  if (!cands.length) return 'Not in sources.';
  const sents = cands.map(c => c[0]);
  let base = bm25(query, sents);
  if (expansion.length) { const e = bm25(expansion.join(' '), sents); base = base.map((a, i) => a + 0.5 * e[i]); }
  const scores = base.map((x, i) => x * (CUE.test(sents[i]) ? 1.3 : 1.0));
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
      '&prop=extracts|info|pageimages&inprop=url&exintro=1&explaintext=1&exlimit=20' +
      '&piprop=thumbnail&pithumbsize=480&pilimit=20', fetchFn);
    const pages = Object.values(d.query?.pages ?? {}).sort((a, b) => a.index - b.index);
    // the top 4 articles are read in full, not just their intros
    const full = await Promise.all(pages.slice(0, 4).map(p =>
      jget(`${WIKI}&prop=extracts&explaintext=1&exsectionformat=plain&pageids=${p.pageid}`, fetchFn)
        .then(x => x.query.pages[p.pageid].extract).catch(() => null)));
    return pages.map((p, i) => ({ url: p.fullurl, title: p.title, text: full[i] || p.extract || '', source: 'wikipedia',
      images: p.thumbnail ? [{ url: p.thumbnail.source, alt: p.title, caption: '' }] : [] }));
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
  async github(q, fetchFn) {
    const d = await jget(`https://api.github.com/search/repositories?per_page=20&q=${encodeURIComponent(q)}`, fetchFn);
    return (d.items ?? []).map(r => ({
      url: r.html_url, title: r.full_name, source: 'github',
      text: `${r.full_name}: ${r.description || 'no description'}. Topics: ${(r.topics ?? []).join(', ') || 'none'}. ` +
        `${r.stargazers_count ?? 0} stars, written in ${r.language || 'unknown'}.`,
    }));
  },
};

// ---- Firecrawl: the whole web, live, from the browser (viewer's own key; CORS-open API) ----
// Twins of firecrawl.md_text / md_images, pinned by test_ask.mjs.
const MD_IMAGE = /!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
const MD_LINK = /\[([^\]]*)\]\([^)]*\)/g;
const ICON = /(^data:|\.svg(\?|$)|logo|icon|sprite|avatar|badge|pixel|spacer|blank|button|emoji|\/ads?\/|banner|tracking|gravatar)/i;

const MD_MARK = /^\s{0,3}([-*+>]|\d+\.)\s+/;

// Twin of firecrawl.md_text: paragraph lines JOINED (a line break isn't a sentence end),
// headings dropped, list items / quotes / table rows standalone, short blocks discarded.
export function mdText(md, minChars = 40) {
  const blocks = [], cur = [];
  const flush = () => { if (cur.length) { blocks.push(cur.join(' ')); cur.length = 0; } };
  for (let line of String(md ?? '').replace(MD_IMAGE, '').split(/\r\n|\r|\n/)) {
    line = line.replace(MD_LINK, '$1');
    if (!line.trim()) flush();
    else if (/^\s{0,3}#{1,6}\s/.test(line)) flush();
    else if (MD_MARK.test(line) || line.trimStart().startsWith('|')) { flush(); blocks.push(line.replace(MD_MARK, '')); }
    else cur.push(line);
  }
  flush();
  const keep = [];
  for (let b of blocks) {
    b = b.replace(/[*_`|]+/g, ' ').replace(/\s+/g, ' ').trim();
    if (b.length >= minChars && !/^[-: ]+$/.test(b)) keep.push(/[.!?]$/.test(b) ? b : b + '.');
  }
  return [...new Set(keep)].join(' ');
}

export function mdImages(md) {
  const out = [], seen = new Set();
  for (const [, rawAlt, url] of String(md ?? '').matchAll(MD_IMAGE)) {
    const alt = rawAlt.split(/\s+/).filter(Boolean).join(' ');
    if (/^https?:/.test(url) && alt && !seen.has(url) && !ICON.test(url)) { seen.add(url); out.push({ url, alt, caption: '' }); }
  }
  return out.slice(0, 8);
}

const FIRECRAWL = 'https://api.firecrawl.dev/v2';
async function fcPost(path, body, key, fetchFn) {
  const r = await fetchFn(FIRECRAWL + path, { method: 'POST', signal: AbortSignal.timeout?.(60000),
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || !d.success) throw new Error(r.status === 402 ? 'out of Firecrawl credits' : `HTTP ${r.status}`);
  return d.data ?? {};
}
const fcDoc = (it, source = 'firecrawl') => {
  const meta = it.metadata ?? {}, url = it.url || meta.sourceURL || meta.url || '';
  return { url, title: it.title || meta.title || url, text: mdText(it.markdown) || it.description || '',
    images: mdImages(it.markdown), source };
};

// One call, three result types: web pages scraped to markdown, dated news, images (twin of search_all).
export async function firecrawlSearch(q, key, limit = 8, fetchFn = fetch) {
  const d = await fcPost('/search', { query: q, limit, sources: ['web', 'news', 'images'],
    scrapeOptions: { formats: ['markdown'], onlyMainContent: true } }, key, fetchFn);
  const docs = [...(d.web ?? []).map(it => fcDoc(it)), ...(d.news ?? []).map(it => ({
    url: it.url, title: `${it.title ?? ''} (${it.date ?? 'news'})`, text: mdText(it.snippet, 1) || it.title || '',
    images: [], source: 'firecrawl-news' }))].filter(x => x.url && x.text);
  const images = (d.images ?? []).filter(i => /^https?:/.test(i.imageUrl ?? '') && i.title)
    .map(i => ({ url: i.imageUrl, page: i.url, alt: i.title, license: '', source: 'firecrawl' }));
  return { docs, images };
}

// Any URL, read by Firecrawl's browser (JS rendered) — paste a link, ask about the page.
export async function firecrawlScrape(url, key, fetchFn = fetch) {
  const d = await fcPost('/scrape', { url, formats: ['markdown'], onlyMainContent: true }, key, fetchFn);
  return fcDoc({ ...d, url: d.metadata?.sourceURL || url }, 'firecrawl');
}

// Fan every query to every source at once; one source failing costs only its own docs.
export async function gather(question, { limit = 200, fetchFn = fetch, onProgress, imageFn = discoverImages,
  firecrawlKey = '', firecrawlLimit = 8 } = {}) {
  const queries = subqueries(question);
  const ledger = {};
  const jobs = [];
  const images = imageFn(question, fetchFn).catch(() => ({ records: [], tried: [] }));
  let fcImages = [];
  if (firecrawlKey) {
    ledger.firecrawl = { docs: 0, error: null };
    // a pasted URL is read directly; anything else is searched (web + news + images)
    const isUrl = /^https?:\/\/\S+$/.test(question.trim());
    jobs.push((isUrl ? firecrawlScrape(question.trim(), firecrawlKey, fetchFn).then(d => ({ docs: [d], images: [] }))
      : firecrawlSearch(question, firecrawlKey, firecrawlLimit, fetchFn)).then(({ docs, images }) => {
      ledger.firecrawl.docs = docs.length; fcImages = images; onProgress?.(ledger); return docs;
    }).catch(err => { ledger.firecrawl.error = String(err?.message ?? err); onProgress?.(ledger); return []; }));
  }
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
  const { records, tried } = await images;
  for (const t of tried) ledger[t.provider] = { docs: t.count ?? 0, error: t.ok ? null : t.error, images: true };
  const apiImages = [...fcImages, ...records.map(r => ({ url: r.thumb_url, page: r.source, alt: r.name, license: r.license, source: r.provider }))];
  return { docs, ledger, queries, apiImages };
}

// Image providers (Openverse, Commons, Art Institute, Met, iNaturalist) from crawler.js —
// imported lazily so the Node tests never load the browser-only module graph.
async function discoverImages(term, fetchFn) {
  const { discover } = await import('./crawler.js');
  return discover(term, 6, fetchFn);
}

// Twin of multimodal.rank_images: BM25 on an image's words + a bonus when its page is cited.
export function rankImages(question, docs, apiImages, cited, k = 6, bonus = 1.0) {
  const get = u => (cited instanceof Map ? cited.get(u) : cited[u]) ?? null;
  const cands = [];
  for (const d of docs) for (const im of d.images ?? []) cands.push({ ...im, page: d.url, title: d.title, source: 'page' });
  for (const im of apiImages) cands.push({ ...im, title: '' });
  const seen = new Set(), uniq = cands.filter(c => !seen.has(c.url) && seen.add(c.url));
  if (!uniq.length) return [];
  const scores = bm25(question, uniq.map(c => `${c.alt} ${c.caption ?? ''} ${c.title}`));
  uniq.forEach((c, i) => {
    c.n = get(c.page);
    c.score = Math.round((scores[i] + (c.n ? bonus : 0)) * 1e4) / 1e4;
  });
  return argsortDesc(uniq.map(c => c.score)).map(i => uniq[i]).filter(c => c.score > 0).slice(0, k);
}

// CLIP on pixels, in the tab: blend like multimodal.clip_rerank (half words, half CLIP).
export async function clipRerank(question, images, onStatus = () => {}, weight = 0.5) {
  const { getTextEncoder, getImageEncoder } = await import('./clip.js');
  const [encT, encI] = [await getTextEncoder(onStatus, () => {}), await getImageEncoder(onStatus, () => {})];
  const [q] = await encT([`a photo of ${question}`]);
  const top = Math.max(...images.map(im => im.score)) || 1;
  await Promise.all(images.map(async im => {
    try {
      const v = await encI(await (await fetch(im.url)).blob());
      im.clip = Math.round(v.reduce((s, x, i) => s + x * q[i], 0) * 1e4) / 1e4;
      im.score = Math.round((weight * im.score / top + (1 - weight) * Math.min(1, Math.max(0, im.clip * 3))) * 1e4) / 1e4;
    } catch { /* an image that won't load keeps its word score */ }
  }));
  return [...images].sort((a, b) => b.score - a.score);
}

// A photo becomes a query: CLIP zero-shot over the repo's tag vocabulary (keyless).
export async function clipTags(blob, k = 3, onStatus = () => {}) {
  const [{ getTextEncoder, getImageEncoder }, { VOCAB, tagPrompts }] = await Promise.all([import('./clip.js'), import('./templates.js')]);
  const encT = await getTextEncoder(onStatus, () => {}), encI = await getImageEncoder(onStatus, () => {});
  const [img, tags] = [await encI(blob), await encT(tagPrompts())];
  const s = tags.map(t => t.reduce((acc, x, i) => acc + x * img[i], 0));
  return argsortDesc(s).slice(0, k).map(i => VOCAB[i]);
}

export function retrieve(question, docs, k = 10) {
  const passages = docs.flatMap(d => chunk(d.text).map(text => ({ text, src: d.url, title: d.title, source: d.source })));
  const expansion = expandQuery(question, passages.map(p => p.text));
  const top = rank(question, passages, k, 2, expansion);
  const num = numberSources(top);
  const byUrl = new Map(top.map(p => [p.src, p]));
  const sources = [...num].map(([url, n]) => ({ n, url, title: byUrl.get(url).title, source: byUrl.get(url).source }));
  return { top, sources, passages: passages.length, expansion };
}

// ---- Claude: the official SDK, straight from the browser with the viewer's own key ----

export const SYSTEM = w => `You are a search engine's answer line. Answer the question in at most ${w} words, ` +
  'using ONLY the numbered sources. Lead with the answer itself: no preamble, no ' +
  'restating the question, no hedging filler. Put a citation like [2] after every ' +
  'claim. If the sources do not contain the answer, reply exactly: Not in sources.';

const sdk = async key => {
  const { default: Anthropic } = await import('https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.137.0/+esm');
  return new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
};
const RASTER = /\.(jpe?g|png|gif|webp)(\?|$)/i;

// Twin of answer.py's vision answer: the user's photo first, then up to 3 images from cited
// sources, then the passages. A dead image URL must not cost the answer: retry without them.
export async function claudeAnswer({ key, model, question, top, words, onText, photo = null, images = [] }) {
  const client = await sdk(key);
  const ctx = top.map(p => `[${p.n}] ${p.title}\n${p.text}`).join('\n\n');
  const text = { type: 'text', text: `Sources:\n${ctx}\n\nQuestion: ${question}` };
  const mine = photo ? [{ type: 'text', text: "The user's photo:" }, photo] : [];
  const web = images.filter(im => im.n && RASTER.test(im.url)).slice(0, 3).flatMap(im => [
    { type: 'text', text: `Image from source [${im.n}]:` }, { type: 'image', source: { type: 'url', url: im.url } }]);
  const attempts = web.length ? [[...mine, ...web, text], [...mine, text]] : [[...mine, text]];
  let lastErr;
  for (const content of attempts) {
    try {
      const stream = client.beta.messages.stream({
        model, max_tokens: 2000, system: SYSTEM(words),
        output_config: { effort: 'low' },
        betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default',
        messages: [{ role: 'user', content: content.length === 1 ? content[0].text : content }],
      });
      stream.on('text', t => onText?.(t));
      const msg = await stream.finalMessage();
      if (msg.stop_reason === 'refusal') throw new Error('the model declined');
      return { text: msg.content.filter(b => b.type === 'text').map(b => b.text).join('').trim(), vision: content.length > 1 };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// A File → a base64 image block (what Claude vision takes from a browser).
export const photoBlock = file => new Promise((res, rej) => {
  const r = new FileReader();
  r.onload = () => res({ type: 'image', source: { type: 'base64', media_type: file.type || 'image/jpeg',
    data: String(r.result).split(',')[1] } });
  r.onerror = rej;
  r.readAsDataURL(file);
});

// Photo → a short search query with Claude vision (twin of multimodal.image_to_query).
export async function visionQuery({ key, model, photo, question }) {
  const client = await sdk(key);
  const msg = await client.beta.messages.create({
    model, max_tokens: 300, output_config: { effort: 'low' },
    system: 'Write ONE short web-search query (max 8 words, no quotes) that would find pages explaining ' +
      'what this image shows, focused on the user\'s question. Name the specific thing (species, ' +
      'landmark, product, artwork) if you can.',
    messages: [{ role: 'user', content: [photo, { type: 'text', text: question || 'What is this?' }] }],
  });
  return msg.content.filter(b => b.type === 'text').map(b => b.text).join('').trim().split('\n')[0];
}
