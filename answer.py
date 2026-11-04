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


def rank(query, passages, k=10, per_source=2):
    """passages: [{text, src}] → top-k by BM25, at most `per_source` per source."""
    scores = bm25(query, [p["text"] for p in passages])
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


def extractive(query, top, words=45):
    """No LLM: the best novel sentences, verbatim, each with its [n]."""
    cands = [(s, p["n"]) for p in top for s in SENT.split(p["text"])
             if len(s.split()) >= 5 and not s.rstrip().endswith("?")]   # questions aren't answers
    if not cands:
        return "Not in sources."
    scores = bm25(query, [s for s, _ in cands])
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
    sem = asyncio.Semaphore(concurrency)

    async def one(u):
        async with sem:
            try:
                page = await _get(u, timeout)
                if page.status >= 400 or "html" not in str(page.headers.get("content-type", "html")):
                    return None
                title, text = readable(page)
                return {"url": u, "title": title or urllib.parse.urlparse(u).netloc, "text": text}
            except Exception:
                return None

    return [d for d in await asyncio.gather(*map(one, urls)) if d and d["text"]]


# --- answer: Claude, succinct and cited — or extractive ---

SYSTEM = ("You are a search engine's answer line. Answer the question in at most {w} words, "
          "using ONLY the numbered sources. Lead with the answer itself: no preamble, no "
          "restating the question, no hedging filler. Put a citation like [2] after every "
          "claim. If the sources do not contain the answer, reply exactly: Not in sources.")


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
            messages=[{"role": "user", "content": user}]) as stream:
        for t in stream.text_stream:
            if on_text:
                on_text(t)
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("model declined")
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def ask(question, pages=200, k=10, words=45, engines=None, model="claude-opus-5-5",
        concurrency=40, timeout=10, use_llm=True, stream=None):
    t0 = time.time()
    engines = engines or ENGINES
    client = _client() if use_llm else None
    queries = subqueries(question, client, model)
    per_engine = max(10, math.ceil(pages / max(1, len(queries) * len(engines))) * 2)
    urls, ledger = asyncio.run(search(queries, engines, per_engine, timeout))
    urls = urls[:pages]
    docs = asyncio.run(scrape(urls, concurrency, timeout))
    passages = [{"text": c, "src": d["url"], "title": d["title"]}
                for d in docs for c in chunk(d["text"])]
    top = rank(question, passages, k=k)
    num = number_sources(top)
    titles = {p["src"]: p["title"] for p in top}
    sources = [{"n": n, "url": u, "title": titles[u]} for u, n in num.items()]
    mode = "extractive"
    answer = extractive(question, top, words)
    if client and top:
        ctx = "\n\n".join(f"[{p['n']}] {p['title']}\n{p['text']}" for p in top)
        try:
            answer = _claude(client, model, 2000, SYSTEM.format(w=words),
                             f"Sources:\n{ctx}\n\nQuestion: {question}", stream)
            mode = model
        except Exception as e:
            print(f"  (LLM answer failed, extractive instead: {e})", file=sys.stderr)
    return {"question": question, "answer": answer, "mode": mode, "sources": sources,
            "passages": top, "queries": queries, "ledger": ledger,
            "stats": {"urls": len(urls), "scraped": len(docs), "passages": len(passages),
                      "seconds": round(time.time() - t0, 2)}}


def show(r, streamed=False):
    if not streamed:
        print(r["answer"])
    print()
    for s in r["sources"]:
        print(f"  [{s['n']}] {urllib.parse.urlparse(s['url']).netloc} — {s['title'][:70]}")
    st = r["stats"]
    print(f"\n  read {st['scraped']}/{st['urls']} pages · {st['passages']:,} passages · "
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
    return {"passages": passages, "cases": cases, "chunk_text": text, "chunk": chunk(text, 20),
            "tokens": tokens("Why is the Sky blue? It's Rayleigh-scattering, 4th power!")}


# --- selftest: a real Scrapling crawl of a local fixture web ---

FIXTURE = {
    "/a": "<html><head><title>Rayleigh scattering</title></head><body><nav>Home About "
          "Contact Login and other navigation links here</nav><article><p>The sky is blue "
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
    r = ask("why is the sky blue", pages=10, words=30, use_llm=False,
            engines={"local": (base + "/search?q={q}", "a.hit", None)})
    srv.shutdown()
    show(r)
    assert r["stats"]["scraped"] == 3, r["stats"]
    assert r["sources"][0]["title"] == "Rayleigh scattering"
    assert "Rayleigh scattering" in r["answer"] and "[1]" in r["answer"]
    assert "pizza" not in r["answer"].lower() and "footer" not in r["answer"].lower()
    print("selftest ok — Scrapling fetched + parsed the fixture web, answer is cited")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="*")
    ap.add_argument("--pages", type=int, default=200, help="max pages to scrape (default 200)")
    ap.add_argument("--k", type=int, default=10, help="passages the answer may cite")
    ap.add_argument("--words", type=int, default=45, help="answer word budget")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--no-llm", action="store_true", help="extractive answer only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write-fixture", action="store_true", help="regenerate docs/js/ask_fixture.json")
    a = ap.parse_args()
    if a.write_fixture:
        open("docs/js/ask_fixture.json", "w").write(json.dumps(twin_fixture(), indent=1) + "\n")
        sys.exit()
    if a.selftest:
        selftest()
        sys.exit()
    if not a.question:
        ap.error("ask a question")
    live = not a.json and sys.stdout.isatty()
    out = ask(" ".join(a.question), a.pages, a.k, a.words, model=a.model, use_llm=not a.no_llm,
              stream=(lambda t: print(t, end="", flush=True)) if live else None)
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        show(out, streamed=live and out["mode"] != "extractive")
