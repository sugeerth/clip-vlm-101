"""answer.py — ask the web a question, get one cited sentence back.

pipeline: question ─► [fan out: Q sub-queries × 4 engines] ─► ≤200 URLs
          ─► [Scrapling: fetch all, concurrently] ─► passages ─► BM25 top-k
          ─► Claude writes ≤N words with [n] cites  (no key: extractive answer)

A Perplexity-shaped search engine whose two rules are: read a LOT, say a
LITTLE. Every page is fetched with Scrapling (github.com/D4Vinci/Scrapling)
— curl_cffi under the hood, impersonating a real Chrome TLS fingerprint, so
ordinary sites answer it like a browser — and parsed with Scrapling's
selectors. The model never browses: it sees only the passages that won the
BM25 ranking, numbered, and must cite them.

    pip install "scrapling[fetchers]" anthropic
    python3 answer.py "why is the sky blue"
    python3 answer.py "what is OPQ in vector search" --pages 200 --words 30
    python3 answer.py "..." --json            # machine-readable, with passages
    python3 answer.py --selftest              # scrapes a local fixture site

Without ANTHROPIC_API_KEY (or an `ant auth login` profile) the answer is
EXTRACTIVE: the highest-scoring novel sentences, verbatim, with their
citations. Scraping is honest work: a truthful User-Agent is not sent (the
point of impersonation is to be served the page a reader sees), so keep
--pages modest on sites you do not own and respect their terms.
"""
import argparse
import asyncio
import json
import math
import pathlib
import re
import sys
import time
import urllib.parse

# --- the text math (pure Python; docs/js/ask.js is its twin, pinned by test) ---

STOP = set("a an and are as at be by for from has have how i in is it its of on or "
           "that the this to was were what when where which who why will with does do "
           "can you your vs".split())
TOKEN = re.compile(r"[a-z0-9]+")
SENT = re.compile(r"(?<=[.!?])\s+")


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk(text, words=80):
    """Sentence-aligned passages of ~`words` words."""
    out, cur, n = [], [], 0
    for s in SENT.split(text):
        s = s.strip()
        if not s:
            continue
        cur.append(s)
        n += len(s.split())
        if n >= words:
            out.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        out.append(" ".join(cur))
    return out


def bm25(query, docs, k1=1.2, b=0.75):
    """Okapi BM25 score for each doc (a string) against the query."""
    q = list(dict.fromkeys(tokens(query)))
    toks = [tokens(d) for d in docs]
    N = len(docs) or 1
    avg = sum(map(len, toks)) / N or 1
    df = {t: sum(1 for d in toks if t in d) for t in q}
    scores = []
    for d in toks:
        tf, s = {}, 0.0
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        for t in q:
            f = tf.get(t, 0)
            if f:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


# words too common to say what a page is ABOUT — never used as feedback terms
EXP_STOP = set("""about above after again against all also although always among another any anyone anything around back because been before being below best between both but came can cannot come could did does doing done down during each either else even ever every few first for from further get gets getting give given goes going gone good got great had has have having her here hers him his how however into its just keep know last least less let like likely made make makes making many may maybe more most much must near need needs never new next not nothing now often once one only onto other others our ours out over own part per perhaps place put rather really right said same say see seen several shall she should show since some something sometimes soon still such sure take than that the their them then there these they thing things think this those though through thus time times today together too toward under until upon use used uses using very want was way ways well went were what whatever when where whether which while who whom whose why will with within without would yet you your yours year years day days tonight amazing beautiful ready tips guide breathtaking stunning incredible magical spectacular ultimate awesome""".split())


def expand_query(query, docs, m=8, n=6, min_docs=2):
    """Pseudo-relevance feedback (RM3-lite): words that dominate the top-m BM25 hits, appear
    in at least `min_docs` of them, and are not in the question — "why do the northern lights
    happen" → appear, wind, solar, southern, australis — so pages that explain the thing in
    other words rank too. docs: list of strings. Twin: ask.js expandQuery."""
    scores = bm25(query, docs)
    top = [i for i in sorted(range(len(docs)), key=lambda i: -scores[i]) if scores[i] > 0][:m]
    q = set(tokens(query))
    toks = [tokens(d) for d in docs]
    sets = [set(d) for d in toks]
    N, w, hits, df = len(docs), {}, {}, {}
    for i in top:
        tf = {}
        for t in toks[i]:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            if t in q or len(t) < 4 or t.isdigit() or t in EXP_STOP:
                continue
            if t not in df:
                df[t] = sum(1 for s_ in sets if t in s_)
            w[t] = w.get(t, 0.0) + c / len(toks[i]) * math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            hits[t] = hits.get(t, 0) + 1
    return sorted((t for t in w if hits[t] >= min_docs), key=lambda t: (-w[t], t))[:n]


def rank(query, passages, k=10, per_source=2, expansion=()):
    """passages: [{text, src}] → top-k by BM25 (+ 0.5 × BM25 of the feedback terms),
    at most `per_source` per source."""
    texts = [p["text"] for p in passages]
    scores = bm25(query, texts)
    if expansion:
        scores = [a + 0.5 * b for a, b in zip(scores, bm25(" ".join(expansion), texts))]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    out, seen = [], {}
    for i in order:
        if scores[i] <= 0 or len(out) >= k:
            break
        src = passages[i]["src"]
        if seen.get(src, 0) < per_source:
            seen[src] = seen.get(src, 0) + 1
            out.append({**passages[i], "score": round(scores[i], 4)})
    return out


def number_sources(top):
    """Give each distinct source a citation number, in rank order."""
    num = {}
    for p in top:
        num.setdefault(p["src"], len(num) + 1)
        p["n"] = num[p["src"]]
    return num


JUNK = re.compile(r"^(watch|click|subscribe|sign up|read more|learn more|share|follow|see also|related|"
                  r"advertisement|cookie|skip to|jump to|fun fact|learn how|discover|book|sign in|log in)\b", re.I)
CUE = re.compile(r"\b(because|caused by|due to|occurs? when|happens? when|results? from|is called|"
                 r"are called|which means)\b", re.I)


def headline(s):
    """Title Case lines ('When to See the Northern Lights in Norway') are headings, not answers."""
    long = [w for w in s.split() if len(w) >= 4 and w[0].isalpha()]
    return len(long) >= 3 and sum(w[0].isupper() for w in long) / len(long) >= 0.7


def extractive(query, top, words=45, expansion=()):
    """No LLM: the best novel sentences, verbatim, each with its [n]. Candidates must read like
    statements (≥ 8 words, not a question, not page chrome); explanations (because, caused by,
    occurs when…) get a 1.3× lift — a search answer should say WHY, not repeat the title."""
    cands = [(s, p["n"]) for p in top for s in SENT.split(p["text"])
             if len(s.split()) >= 8 and not s.rstrip().endswith("?") and not JUNK.match(s.strip())
             and not headline(s)]
    if not cands:
        return "Not in sources."
    sents = [s for s, _ in cands]
    base = bm25(query, sents)
    if expansion:
        base = [a + 0.5 * b for a, b in zip(base, bm25(" ".join(expansion), sents))]
    scores = [x * (1.3 if CUE.search(s) else 1.0) for x, s in zip(base, sents)]
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    picked, used, budget = [], [], words
    for i in order:
        s, n = cands[i]
        if scores[i] <= 0:
            break
        ts = set(tokens(s))
        if any(len(ts & u) / (len(ts | u) or 1) > 0.5 for u in used):
            continue                      # near-duplicate of a sentence we already have
        if picked and len(s.split()) > budget:
            continue
        picked.append(f"{s.rstrip()} [{n}]")
        used.append(ts)
        budget -= len(s.split())
        if budget <= 0:
            break
    return " ".join(picked) if picked else "Not in sources."


# --- search: four engines, scraped with Scrapling ---

def _ddg(href):
    q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    return q.get("uddg", [href])[0]


ENGINES = {   # name: (url template, css selector for result links, href fixer)
    "duckduckgo": ("https://html.duckduckgo.com/html/?q={q}", "a.result__a", _ddg),
    "bing":       ("https://www.bing.com/search?q={q}&count=50", "li.b_algo h2 a", None),
    "wikipedia":  ("https://en.wikipedia.org/w/index.php?search={q}&fulltext=1&ns0=1&limit=20",
                   ".mw-search-result-heading a", None),
    "mojeek":     ("https://www.mojeek.com/search?q={q}", "a.ob", None),
}
SKIP = re.compile(r"\.(pdf|jpe?g|png|gif|zip|mp4|mp3)(\?|$)|youtube\.com/watch", re.I)


def subqueries(question, client=None, model=None):
    """The question, plus up to 3 reformulations (Claude writes them if it can)."""
    qs = [question]
    if client:
        try:
            r = _claude(client, model, 300,
                        "Write 3 short web-search queries (one per line, no numbering) that "
                        "together cover this question from different angles.", question)
            qs += [l.strip(' -•"') for l in r.splitlines() if l.strip()][:3]
        except Exception as e:                     # the search still runs without them
            print(f"  (sub-query expansion skipped: {e})", file=sys.stderr)
    else:
        bare = " ".join(tokens(question))
        if bare and bare != question.lower():
            qs.append(bare)
    return list(dict.fromkeys(qs))


async def _get(url, timeout):
    import logging
    from scrapling.fetchers import AsyncFetcher
    logging.getLogger("scrapling").setLevel(logging.WARNING)
    return await AsyncFetcher.get(url, timeout=timeout, retries=1, impersonate="chrome",
                                  stealthy_headers=True, follow_redirects=True)


async def search(queries, engines, per_engine, timeout):
    """Fan every query out to every engine at once; return deduped URLs + a ledger."""
    jobs = [(e, q) for q in queries for e in engines]

    async def one(e, q):
        tpl, css, fix = engines[e]
        try:
            page = await _get(tpl.format(q=urllib.parse.quote_plus(q)), timeout)
            hrefs = [a.attrib.get("href", "") for a in page.css(css)][:per_engine]
            return e, [urllib.parse.urljoin(page.url, fix(h) if fix else h) for h in hrefs], None
        except Exception as err:
            return e, [], str(err).split("\n")[0][:80]

    urls, ledger = [], {}
    for e, found, err in await asyncio.gather(*(one(*j) for j in jobs)):
        led = ledger.setdefault(e, {"urls": 0, "errors": []})
        led["urls"] += len(found)
        if err:
            led["errors"].append(err)
        urls += [u for u in found if u.startswith("http") and not SKIP.search(u)]
    return list(dict.fromkeys(urls)), ledger


# --- scrape: every page at once, Scrapling's parser pulls the readable text ---

BLOCKS = ("//*[self::p or self::li or self::h2 or self::h3 or self::pre or self::blockquote "
          "or self::dd or self::td][not(ancestor::nav or ancestor::footer or ancestor::header "
          "or ancestor::aside or ancestor::form)]")


def readable(page, min_chars=40):
    """Title + de-duplicated text blocks of a Scrapling page, chrome stripped."""
    title = (page.css("title::text").get() or "").strip()
    seen, out = set(), []
    for el in page.xpath(BLOCKS):
        t = re.sub(r"\s+", " ", el.get_all_text(separator=" ", strip=True)).strip()
        if len(t) >= min_chars and t not in seen:
            seen.add(t)
            out.append(t if t[-1] in ".!?" else t + ".")
    return title, " ".join(out)


async def scrape(urls, concurrency, timeout):
    import multimodal
    sem = asyncio.Semaphore(concurrency)

    async def one(u):
        async with sem:
            try:
                page = await _get(u, timeout)
                if page.status >= 400 or "html" not in str(page.headers.get("content-type", "html")):
                    return None
                title, text = readable(page)
                return {"url": u, "title": title or urllib.parse.urlparse(u).netloc, "text": text,
                        "images": multimodal.page_images(page, u)}
            except Exception:
                return None

    return [d for d in await asyncio.gather(*map(one, urls)) if d and d["text"]]


# --- answer: Claude, succinct and cited — or extractive ---

SYSTEM = ("You are a search engine's answer line. Answer the question in at most {w} words, "
          "using ONLY the numbered sources. Lead with the answer itself: no preamble, no "
          "restating the question, no hedging filler. Put a citation like [2] after every "
          "claim. Images may be attached: the user's own photo, and images labelled with the "
          "source they came from; use what you see, and cite an image's source like any other. "
          "If the sources do not contain the answer, reply exactly: Not in sources.")


def _client():
    try:
        import anthropic
        c = anthropic.Anthropic()
        c.models.list(limit=1)            # fails fast when no credential resolves
        return c
    except Exception:
        return None


def _claude(client, model, max_tokens, system, user, on_text=None):
    with client.beta.messages.stream(
            model=model, max_tokens=max_tokens, system=system,
            output_config={"effort": "low"},        # short answers don't need deep thought
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": user}]) as stream:   # user: str or content blocks
        for t in stream.text_stream:
            if on_text:
                on_text(t)
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("model declined")
    return "".join(b.text for b in msg.content if b.type == "text").strip()


async def ask_async(question, pages=200, k=10, words=45, engines=None, model="claude-opus-5-5",
                    concurrency=40, timeout=10, use_llm=True, stream=None, apis=None,
                    image=None, use_clip=False, n_images=6, gallery=None, save_images=None,
                    firecrawl=None, firecrawl_limit=10, firecrawl_rescue=5):
    """The whole pipeline. `apis`: source names from sources.py (None = all, () = none).
    `image`: a local photo to ask about; `use_clip`: re-rank result images by CLIP pixels.
    `gallery`: a CLIP·search db.json path (or its items) to search alongside the web;
    `save_images`: a directory to download the top web images into, ready for ingest.py.
    `firecrawl`: add Firecrawl search (+ rescue of unreadable pages); None = on iff
    FIRECRAWL_API_KEY is set. Capped at `firecrawl_limit` results + `firecrawl_rescue` pages."""
    import firecrawl as fc
    use_fc = fc.enabled() if firecrawl is None else firecrawl
    fc_ledger = {"docs": 0, "rescued": 0, "errors": []}
    import multimodal
    import sources as api_sources
    t0 = time.time()
    engines = ENGINES if engines is None else engines
    client = _client() if use_llm else None
    items = multimodal.load_gallery(gallery) if isinstance(gallery, str) else gallery
    image_query = multimodal.image_to_query(image, question, client, model, items) if image else None
    question = " ".join(x for x in (question, image_query) if x) if image_query else question
    queries = subqueries(question, client, model)
    per_engine = max(10, math.ceil(pages / max(1, len(queries) * max(1, len(engines)))) * 2)

    async def scraped():
        urls, ledger = await search(queries, engines, per_engine, timeout) if engines else ([], {})
        docs = await scrape(urls[:pages], concurrency, timeout)
        if use_fc and firecrawl_rescue:
            # pages Scrapling fetched but couldn't read (blocked, JS-only, near-empty) → Firecrawl
            got = {d["url"]: d for d in docs}
            weak = [u for u in urls[:pages] if u not in got or len(got[u]["text"]) < fc.WEAK][:firecrawl_rescue]

            async def rescue(u):
                try:
                    return await fc.scrape(u)
                except Exception as e:
                    fc_ledger["errors"].append(str(e)[:80])
                    return None
            for d in await asyncio.gather(*map(rescue, weak)):
                if d and d["text"] and len(d["text"]) > len(got.get(d["url"], {}).get("text", "")):
                    docs = [x for x in docs if x["url"] != d["url"]] + [d]
                    fc_ledger["rescued"] += 1
        return urls[:pages], ledger, docs

    async def from_firecrawl():
        if not use_fc or not firecrawl_limit:
            return [], []
        try:
            return await fc.search_all(question, firecrawl_limit)   # web + news + images, one call
        except Exception as e:
            fc_ledger["errors"].append(str(e)[:80])
            return [], []

    async def from_apis():
        if apis is not None and not apis:
            return {}, {}
        return await api_sources.search_sources(question, apis, timeout)

    async def from_image_apis():
        if apis is not None and not apis:
            return [], {}
        return await multimodal.image_search(question, timeout=timeout)

    (urls, ledger, docs), (api_docs, api_errors), (api_images, img_errors), (fc_docs, fc_images) = \
        await asyncio.gather(scraped(), from_apis(), from_image_apis(), from_firecrawl())
    api_images = api_images + fc_images
    fc_ledger["images"] = len(fc_images)
    have = {d["url"]: i for i, d in enumerate(docs)}
    for d in fc_docs:                                     # same page from both engines: keep the fuller text
        if d["url"] not in have:
            have[d["url"]] = len(docs)
            docs.append(d)
            fc_ledger["docs"] += 1
        elif len(d["text"]) > len(docs[have[d["url"]]]["text"]):
            docs[have[d["url"]]] = d
            fc_ledger["docs"] += 1
    if use_fc:
        ledger["firecrawl"] = fc_ledger
    for name in multimodal.IMAGE_SOURCES:
        if apis is None or apis:
            ledger[name] = {"images": sum(im["source"] == name for im in api_images),
                            "errors": [img_errors[name]] if name in img_errors else []}
    for name, ds in api_docs.items():                     # one ledger for engines and APIs alike
        ledger[name] = {"docs": len(ds), "errors": [api_errors[name]] if name in api_errors else []}
    seen = {d["url"] for d in docs}
    docs += [d for ds in api_docs.values() for d in ds if d["url"] not in seen and not seen.add(d["url"])]
    passages = [{"text": c, "src": d["url"], "title": d["title"]}
                for d in docs for c in chunk(d["text"])]
    expansion = expand_query(question, [p["text"] for p in passages])
    top = rank(question, passages, k=k, expansion=expansion)
    num = number_sources(top)
    titles = {p["src"]: p["title"] for p in top}
    sources = [{"n": n, "url": u, "title": titles[u]} for u, n in num.items()]
    images = multimodal.rank_images(question, docs, api_images, num, k=n_images)
    if use_clip and images:
        images = await multimodal.clip_rerank(question, images)
    from_gallery = []
    if items:
        from_gallery = (multimodal.clip_gallery_search(question, items) if use_clip
                        else multimodal.gallery_search(question, items))
    saved = await multimodal.save_images(images, save_images, question) if save_images and images else []
    mode = "extractive"
    answer = extractive(question, top, words, expansion)
    if client and top:
        ctx = "\n\n".join(f"[{p['n']}] {p['title']}\n{p['text']}" for p in top)
        text = {"type": "text", "text": f"Sources:\n{ctx}\n\nQuestion: {question}"}
        mine = ([{"type": "text", "text": "The user's photo:"}, multimodal.image_block(image)]
                if image else [])
        shown = [im for im in images if im["n"] and multimodal.RASTER.search(im["url"])][:3]
        web = [b for im in shown for b in ({"type": "text", "text": f"Image from source [{im['n']}]:"},
                                           multimodal.image_block(im["url"]))]
        attempts = [[*mine, *web, text]] + ([[*mine, text]] if web else [])
        for content in attempts:              # a dead image URL must not cost the answer
            try:
                answer = _claude(client, model, 2000, SYSTEM.format(w=words), content, stream)
                mode = model + (" + vision" if len(content) > 1 else "")
                break
            except Exception as e:
                print(f"  (LLM answer failed: {e})", file=sys.stderr)
    n_api = sum(map(len, api_docs.values()))
    return {"question": question, "answer": answer, "mode": mode, "sources": sources,
            "images": images, "image_query": image_query, "gallery": from_gallery, "saved": saved,
            "expansion": expansion,
            "passages": top, "queries": queries, "ledger": ledger,
            "stats": {"urls": len(urls), "scraped": len(docs) - n_api - fc_ledger["docs"] - fc_ledger["rescued"], "api_docs": n_api,
                      "firecrawl": fc_ledger["docs"] + fc_ledger["rescued"],
                      "passages": len(passages), "seconds": round(time.time() - t0, 2)}}


def ask(question, **kw):
    return asyncio.run(ask_async(question, **kw))


def show(r, streamed=False):
    if not streamed:
        print(r["answer"])
    print()
    for s in r["sources"]:
        print(f"  [{s['n']}] {urllib.parse.urlparse(s['url']).netloc} — {s['title'][:70]}")
    for im in r.get("images", [])[:3]:
        cite = f"[{im['n']}]" if im.get("n") else f"({im['source']})"
        print(f"  img {cite} {im['url'][:90]}  — {(im['alt'] or im.get('caption', ''))[:50]}")
    for g in r.get("gallery", [])[:3]:
        print(f"  gallery {g['file']}  — {', '.join(g['tags'][:3])}  ({g['score']})")
    if r.get("saved"):
        print(f"  saved {len(r['saved'])} images → ingest them: python3 ingest.py "
              f"{pathlib.Path(r['saved'][0]['file']).parent}/web_*.jpg")
    if r.get("image_query"):
        print(f"  your photo was read as: {r['image_query']!r}")
    st = r["stats"]
    fc_note = f" (+{st['firecrawl']} via Firecrawl)" if st.get("firecrawl") else ""
    print(f"\n  read {st['scraped']}/{st['urls']} pages{fc_note} + {st.get('api_docs', 0)} API docs · {st['passages']:,} passages · "
          f"{len(r['queries'])} queries · {r['mode']} · {st['seconds']}s")


# --- the twin fixture: docs/js/test_ask.mjs must reproduce these numbers ---

TWIN = [
    ("https://a", "The sky is blue because air molecules scatter short blue wavelengths of sunlight "
     "far more than red ones, an effect called Rayleigh scattering. Scattering strength rises "
     "with the inverse fourth power of wavelength."),
    ("https://b", "At sunset light crosses more air, so the blue is scattered away and the horizon "
     "glows red. The sky can look orange too."),
    ("https://c", "Blue sky on Mars is not blue: dust makes the daytime sky butterscotch while "
     "sunsets there look blue."),
    ("https://d", "Neapolitan pizza is baked in a wood fired oven at very high temperature for "
     "about ninety seconds."),
    ("https://a", "Rayleigh scattering also explains why the sky is blue at noon; blue light is "
     "scattered in every direction by tiny molecules of nitrogen and oxygen."),
    ("https://a", "Why is the sky blue? Short wavelengths scatter more. Why is the sky blue? "
     "Short wavelengths scatter more."),
]


def twin_fixture():
    passages = [{"text": t, "src": u, "title": u[-1].upper()} for u, t in TWIN]
    cases = []
    for q in ["why is the sky blue", "rayleigh scattering wavelength", "pizza oven temperature",
              "mars sunset"]:
        top = rank(q, [dict(p) for p in passages], k=4)
        number_sources(top)
        cases.append({"q": q, "bm25": [round(x, 9) for x in bm25(q, [p["text"] for p in passages])],
                      "rank": [[p["src"], p["n"], p["score"]] for p in top],
                      "extractive": extractive(q, top, 30)})
    text = " ".join(t for _, t in TWIN)
    import multimodal
    img_docs = [{"url": "https://a", "title": "Rayleigh scattering", "images": [
                    {"url": "https://a/sky.jpg", "alt": "Blue sky over a field", "caption": "Why the sky is blue"},
                    {"url": "https://a/chart.png", "alt": "Scattering versus wavelength", "caption": ""}]},
                {"url": "https://d", "title": "Pizza", "images": [
                    {"url": "https://d/pizza.jpg", "alt": "Neapolitan pizza", "caption": ""}]}]
    img_api = [{"url": "https://c/sky.jpg", "page": "https://c", "alt": "Clear blue sky at noon",
                "license": "CC0", "source": "commons"},
               {"url": "https://a/sky.jpg", "page": "https://dup", "alt": "duplicate", "license": "", "source": "openverse"}]
    images = multimodal.rank_images("why is the sky blue", img_docs, img_api, {"https://a": 1}, k=3)
    gal = multimodal.load_gallery()
    bridge = {"knn": [multimodal.knn_tags(it["text_emb"], gal) for it in gal],
              "search": {q: [[g["file"], g["score"]] for g in multimodal.gallery_search(q, gal)]
                         for q in ["a fluffy pet", "famous tower in paris", "food", "outer space planet"]}}
    import firecrawl
    mds = ["[Jump to content](#x)\n# Rayleigh scattering\n![Blue sky over a field](https://u/sky.jpg)\n"
           "![](https://u/noalt.jpg) ![site logo](https://u/logo.png)\n"
           "**Rayleigh scattering** makes the [sky](https://w/sky) blue because short wavelengths scatter more.\n"
           "| col a | col b |\n|---|---|\n- A list item long enough to be kept as a sentence of the page\n"
           "1. Numbered steps also keep their words but lose the number marker entirely\n"
           "> Quoted text that is long enough to survive the forty character threshold!\n"
           "Short line\nA repeated line that appears twice in the markdown body here.\n"
           "A repeated line that appears twice in the markdown body here.",
           "", "Just one sentence without terminal punctuation that is long enough"]
    fcfx = [{"md": m, "text": firecrawl.md_text(m), "images": firecrawl.md_images(m)} for m in mds]
    # pseudo-relevance feedback: a question in one vocabulary, answers in another
    aur = [("https://t1", "The Northern Lights Are Visible Tonight Across Canada. Northern lights tours sell out "
            "fast, so book your northern lights trip early. The aurora appears when solar wind particles "
            "reach the magnetic field."),
           ("https://t2", "Northern lights photography tips: use a tripod. The aurora borealis glows green when "
            "solar particles strike oxygen high in the atmosphere."),
           ("https://s1", "Auroras happen because charged particles from the solar wind are caught by the "
            "magnetic field and collide with atmospheric gases, which then glow."),
           ("https://s2", "The magnetic field funnels solar particles toward the poles, which is why auroras "
            "are seen at high latitudes."),
           ("https://x", "Pizza ovens reach very high temperatures in about ninety seconds of baking time.")]
    aps = [{"text": t, "src": u, "title": u} for u, t in aur]
    exq = "why do the northern lights happen"
    exp = expand_query(exq, [x["text"] for x in aps], m=8, n=6, min_docs=2)
    etop = rank(exq, [dict(x) for x in aps], k=4, expansion=exp)
    number_sources(etop)
    feedback = {"q": exq, "passages": aps, "expansion": exp,
                "rank": [[x["src"], x["n"], x["score"]] for x in etop],
                "extractive": extractive(exq, etop, 40, exp),
                "headline": [[h, headline(h)] for h in [
                    "When to See the Northern Lights in Norway", "The aurora appears when solar wind hits.",
                    "Why Is the Sky Blue? A Short Answer", "Rayleigh scattering is strongest for short waves"]]}
    return {"passages": passages, "cases": cases, "chunk_text": text, "chunk": chunk(text, 20),
            "bridge": bridge, "firecrawl": fcfx, "feedback": feedback,
            "images": {"docs": img_docs, "api": img_api, "cited": {"https://a": 1},
                       "ranked": [[im["url"], im["n"], im["score"]] for im in images]},
            "tokens": tokens("Why is the Sky blue? It's Rayleigh-scattering, 4th power!")}


# --- selftest: a real Scrapling crawl of a local fixture web ---

FIXTURE = {
    "/a": "<html><head><title>Rayleigh scattering</title></head><body><nav>Home About "
          "Contact Login and other navigation links here <img src=\"/logo.png\" alt=\"Site logo\"></nav>"
          "<article><figure><img src=\"/sky.jpg\" alt=\"Blue sky over a wheat field\" width=\"800\">"
          "<figcaption>Rayleigh scattering turns the daytime sky blue.</figcaption></figure>"
          "<img src=\"/icons/share.png\" alt=\"share\"><img src=\"/tiny.jpg\" alt=\"sky dot\" width=\"16\">"
          "<p>The sky is blue "
          "because air molecules scatter short blue wavelengths of sunlight far more than red "
          "ones, an effect called Rayleigh scattering.</p><p>Scattering strength rises with the "
          "inverse fourth power of wavelength, so blue light scatters about five times more "
          "than red.</p></article><footer>Copyright footer text that must be ignored by the "
          "reader.</footer></body></html>",
    "/b": "<html><head><title>Sunsets</title></head><body><p>At sunset light crosses more air, "
          "so the blue is scattered away before it arrives and the horizon glows red and orange."
          "</p></body></html>",
    "/c": "<html><head><title>Pizza</title></head><body><p>Neapolitan pizza is baked in a wood "
          "fired oven at very high temperature for about ninety seconds.</p></body></html>",
}


def _serve(pages):
    """A throwaway local web: {path: html} on 127.0.0.1, plus /search listing every path."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            body = ("".join(f'<div class="r"><a class="hit" href="{p}">x</a></div>' for p in pages)
                    if path == "/search" else pages.get(path, ""))
            self.send_response(200 if body else 404)
            binary = isinstance(body, bytes)              # images are served as raw PNG bytes
            self.send_header("content-type", "image/png" if binary else "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body if binary else body.encode())

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def bench(n=200, runs=3):
    """Time the whole pipeline over n local pages: the engine's own cost, minus the internet."""
    filler = " ".join(f"Paragraph {i} about topic {i % 17} and its many details." for i in range(40))
    pages = {f"/p{i}": f"<html><head><title>Page {i}</title></head><body><nav>menu</nav>"
             f"<p>{filler}</p><p>Fact {i}: the answer to question {i} is {i * 7}.</p></body></html>"
             for i in range(n)}
    srv, base = _serve(pages)
    times = []
    for _ in range(runs):
        r = ask("what is the answer to question 42", pages=n, words=20, use_llm=False, apis=(), firecrawl=False,
                engines={"local": (base + "/search?q={q}", "a.hit", None)})
        times.append(r["stats"]["seconds"])
    srv.shutdown()
    st = r["stats"]
    print(f"{st['scraped']}/{st['urls']} pages · {st['passages']:,} passages · "
          f"median {sorted(times)[len(times) // 2]}s over {runs} runs (localhost, no network)")
    print("answer:", r["answer"])
    return r, times


def selftest():
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/search":
                body = "".join(f'<div class="r"><a class="hit" href="{p}">x</a></div>'
                               for p in FIXTURE)
            else:
                body = FIXTURE.get(path, "")
            self.send_response(200 if body else 404)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    r = ask("why is the sky blue", pages=10, words=30, use_llm=False, apis=(), firecrawl=False,
            engines={"local": (base + "/search?q={q}", "a.hit", None)})
    srv.shutdown()
    show(r)
    assert r["stats"]["scraped"] == 3, r["stats"]
    assert r["sources"][0]["title"] == "Rayleigh scattering"
    assert "Rayleigh scattering" in r["answer"] and "[1]" in r["answer"]
    assert "pizza" not in r["answer"].lower() and "footer" not in r["answer"].lower()
    assert [im["url"].rsplit("/", 1)[1] for im in r["images"]] == ["sky.jpg"], r["images"]  # logo/icon/tiny dropped
    assert r["images"][0]["n"] == 1 and "Rayleigh" in r["images"][0]["caption"]
    print("selftest ok — Scrapling fetched + parsed the fixture web, answer is cited")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="*")
    ap.add_argument("--pages", type=int, default=200, help="max pages to scrape (default 200)")
    ap.add_argument("--k", type=int, default=10, help="passages the answer may cite")
    ap.add_argument("--words", type=int, default=45, help="answer word budget")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--no-llm", action="store_true", help="extractive answer only")
    ap.add_argument("--apis", help="comma-separated API sources from sources.py (default: all)")
    ap.add_argument("--image", help="a photo to ask about (Claude vision, or CLIP tags with torch)")
    ap.add_argument("--clip", action="store_true", help="re-rank result images with CLIP (needs torch)")
    ap.add_argument("--gallery", nargs="?", const="docs/db.json",
                    help="also search a CLIP·search gallery (default docs/db.json)")
    ap.add_argument("--save-images", metavar="DIR", help="download top web images for ingest.py")
    ap.add_argument("--firecrawl", action=argparse.BooleanOptionalAction, default=None,
                    help="Firecrawl search + rescue of unreadable pages (default: on iff FIRECRAWL_API_KEY)")
    ap.add_argument("--firecrawl-limit", type=int, default=10, help="Firecrawl search results (credits)")
    ap.add_argument("--firecrawl-rescue", type=int, default=5, help="max pages re-read via Firecrawl")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time 200 local pages end to end")
    ap.add_argument("--write-fixture", action="store_true", help="regenerate docs/js/ask_fixture.json")
    a = ap.parse_args()
    if a.write_fixture:
        open("docs/js/ask_fixture.json", "w").write(json.dumps(twin_fixture(), indent=1) + "\n")
        sys.exit()
    if a.selftest:
        selftest()
        sys.exit()
    if a.bench:
        bench(a.pages)
        sys.exit()
    if not a.question and not a.image:
        ap.error("ask a question (or pass --image)")
    live = not a.json and sys.stdout.isatty()
    out = ask(" ".join(a.question), pages=a.pages, k=a.k, words=a.words, model=a.model,
              use_llm=not a.no_llm, apis=a.apis.split(",") if a.apis else None,
              image=a.image, use_clip=a.clip, gallery=a.gallery, save_images=a.save_images,
              firecrawl=a.firecrawl, firecrawl_limit=a.firecrawl_limit, firecrawl_rescue=a.firecrawl_rescue,
              stream=(lambda t: print(t, end="", flush=True)) if live else None)
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        show(out, streamed=live and out["mode"] != "extractive")
