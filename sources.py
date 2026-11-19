"""sources.py — the web as a toolbelt: 7 keyless APIs + search engines + page reading.

pipeline: question ─► [web_search: 4 engines] ─► URLs ─► [read_page: Scrapling]
                  └─► [search_sources: 7 APIs] ─► documents ─┘─► answer.py ranks

answer.py scrapes search-engine results. This file adds the sources that
answer through an API instead of a results page — each returns whole
documents (abstracts, answers, READMEs' descriptions, comments), not links:

    wikipedia      intro extracts of the top articles
    hackernews     stories and comments (Algolia)
    stackexchange  Stack Overflow questions + answers (excerpts)
    openalex       paper abstracts, rebuilt from the inverted index
    arxiv          preprint abstracts (Atom feed, parsed with Scrapling)
    github         repositories: name, description, topics, stars
    reddit         post titles + self-text

Every request goes through Scrapling's AsyncFetcher (answer._get), so one
HTTP stack — Chrome-fingerprinted curl_cffi — serves pages and APIs alike.
The three plain async functions at the bottom (web_search, read_page,
search_sources) are the tool surface: mcp_server.py exposes them over MCP,
research.py hands them to Claude's tool runner.

    python3 sources.py "vector quantization"            # every API at once
    python3 sources.py "rust async" --only github,hackernews
"""
import argparse
import asyncio
import re
import urllib.parse

import answer


def _plain(html):
    """HTML fragment → one line of text, via Scrapling's parser."""
    if not html:
        return ""
    from scrapling.parser import Selector
    text = Selector(f"<div>{html}</div>").get_all_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _wikipedia(page):
    pages = sorted((page.json().get("query") or {}).get("pages", {}).values(),
                   key=lambda p: p.get("index", 0))
    return [{"url": p["fullurl"], "title": p["title"], "text": p.get("extract", "")} for p in pages]


def _hackernews(page):
    return [{"url": f"https://news.ycombinator.com/item?id={h['objectID']}",
             "title": h.get("title") or h.get("story_title") or "Hacker News comment",
             "text": _plain(h.get("story_text") or h.get("comment_text") or h.get("title"))}
            for h in page.json().get("hits", [])]


def _stackexchange(page):
    out = []
    for x in page.json().get("items", []):
        url = (f"https://stackoverflow.com/a/{x['answer_id']}" if x.get("item_type") == "answer"
               else f"https://stackoverflow.com/q/{x['question_id']}")
        out.append({"url": url, "title": _plain(x.get("title")),
                    "text": _plain(x.get("body") or x.get("excerpt"))})
    return out


def _openalex(page):
    out = []
    for w in page.json().get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = {i: t for t, idx in inv.items() for i in idx}
        if pos:
            out.append({"url": w.get("doi") or w["id"],
                        "title": f"{w['display_name']} ({w.get('publication_year')})",
                        "text": " ".join(pos[i] for i in sorted(pos))})
    return out


def _arxiv(page):
    out = []
    for e in page.css("entry"):
        title = re.sub(r"\s+", " ", e.css("title::text").get() or "").strip()
        summary = re.sub(r"\s+", " ", e.css("summary::text").get() or "").strip()
        url = (e.css("id::text").get() or "").strip()
        if url and summary:
            out.append({"url": url, "title": title, "text": summary})
    return out


def _github(page):
    return [{"url": r["html_url"], "title": r["full_name"],
             "text": f"{r['full_name']}: {r.get('description') or 'no description'}. "
                     f"Topics: {', '.join(r.get('topics') or []) or 'none'}. "
                     f"{r.get('stargazers_count', 0)} stars, written in {r.get('language') or 'unknown'}."}
            for r in page.json().get("items", [])]


def _reddit(page):
    out = []
    for c in (page.json().get("data") or {}).get("children", []):
        d = c.get("data", {})
        out.append({"url": "https://www.reddit.com" + d.get("permalink", ""), "title": d.get("title", ""),
                    "text": f"{d.get('title', '')}. {d.get('selftext', '')}".strip()})
    return out


# name: (url template, parser) — templates are module state so tests can point them at localhost
SOURCES = {
    "wikipedia": ("https://en.wikipedia.org/w/api.php?format=json&action=query&generator=search"
                  "&gsrsearch={q}&gsrlimit=20&prop=extracts|info&inprop=url&exintro=1&explaintext=1"
                  "&exlimit=20", _wikipedia),
    "hackernews": ("https://hn.algolia.com/api/v1/search?query={q}&tags=(story,comment)&hitsPerPage=50",
                   _hackernews),
    "stackexchange": ("https://api.stackexchange.com/2.3/search/excerpts?order=desc&sort=relevance"
                      "&site=stackoverflow&pagesize=40&q={q}", _stackexchange),
    "openalex": ("https://api.openalex.org/works?per_page=30&search={q}"
                 "&select=id,doi,display_name,abstract_inverted_index,publication_year", _openalex),
    "arxiv": ("https://export.arxiv.org/api/query?search_query=all:{q}&max_results=25", _arxiv),
    "github": ("https://api.github.com/search/repositories?q={q}&per_page=20", _github),
    "reddit": ("https://www.reddit.com/search.json?q={q}&limit=25&sort=relevance", _reddit),
}


async def search_sources(query, only=None, timeout=10):
    """Ask every API (or `only` these) at once → ({source: [docs]}, {source: error})."""
    names = [n for n in SOURCES if not only or n in only]

    async def one(name):
        tpl, parse = SOURCES[name]
        try:
            page = await answer._get(tpl.format(q=urllib.parse.quote_plus(query)), timeout)
            if page.status >= 400:
                raise RuntimeError(f"HTTP {page.status}")
            docs = [{**d, "source": name} for d in parse(page) if d.get("text")]
            return name, docs, None
        except Exception as err:
            return name, [], str(err).split("\n")[0][:80]

    got, errors = {}, {}
    for name, docs, err in await asyncio.gather(*map(one, names)):
        got[name] = docs
        if err:
            errors[name] = err
    return got, errors


async def web_search(query, limit=20, timeout=10):
    """Search-engine fan-out (answer.ENGINES) → deduplicated result URLs."""
    urls, ledger = await answer.search([query], answer.ENGINES, limit, timeout)
    return urls[:limit], ledger


async def read_page(url, max_chars=8000, timeout=15):
    """Fetch one page with Scrapling → {url, title, text} with nav/footer stripped."""
    page = await answer._get(url, timeout)
    if page.status >= 400:
        raise RuntimeError(f"HTTP {page.status} for {url}")
    title, text = answer.readable(page)
    return {"url": url, "title": title, "text": text[:max_chars], "truncated": len(text) > max_chars}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="+")
    ap.add_argument("--only", help="comma-separated source names")
    a = ap.parse_args()
    got, errors = asyncio.run(search_sources(" ".join(a.query), a.only and a.only.split(",")))
    for name, docs in got.items():
        print(f"{name:14} {len(docs):3} docs  {errors.get(name, '')}")
        for d in docs[:2]:
            print(f"    {d['title'][:70]}  —  {d['url']}")
