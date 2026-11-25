"""firecrawl.py — Firecrawl as ask·200's second scraping engine (firecrawl.dev, v2 API).

pipeline: question ─► [Firecrawl /v2/search: search + scrape in ONE call] ─► docs (markdown)
          URL Scrapling can't read (blocked, JS-only, empty) ─► [Firecrawl /v2/scrape] ─► doc

Scrapling stays the default engine: free, local, fast. Firecrawl is added where it is
strongest, and only when FIRECRAWL_API_KEY is set (each call spends Firecrawl credits):

    search    one request returns web results ALREADY scraped to clean markdown — a whole
              extra engine plus its pages, rendered by a real browser — plus dated NEWS
              and IMAGES (search_all)
    rescue    pages Scrapling fetched but could not read (4xx/5xx, anti-bot walls,
              JavaScript-only apps, near-empty text) are retried through Firecrawl,
              capped by --firecrawl-rescue so a bad query can't drain credits

The HTTP goes through Scrapling's AsyncFetcher like everything else in ask·200.
Response shapes were checked against the live API (search → data.web[] with url,
title, description, markdown, metadata; scrape → data.markdown, metadata, images).

    FIRECRAWL_API_KEY=fc-... python3 answer.py "why is the sky blue"     # auto-on with a key
    python3 answer.py "..." --no-firecrawl                              # Scrapling only
    FIRECRAWL_API_KEY=fc-... python3 firecrawl.py "query"                # search, printed

Its MCP server is separate and complements this: in Claude Code,
    claude mcp add --transport http firecrawl https://mcp.firecrawl.dev/v2/mcp-oauth
    claude mcp login firecrawl
and research.py --firecrawl-mcp hands the hosted server (https://mcp.firecrawl.dev/v2/mcp)
to Claude through the MCP connector.
"""
import os
import re
import urllib.parse


API = "https://api.firecrawl.dev"
MCP_URL = "https://mcp.firecrawl.dev/v2/mcp"
WEAK = 80   # chars of readable text below which a fetched page counts as unreadable (JS shell, wall)


def api_url():
    return os.environ.get("FIRECRAWL_API_URL", API).rstrip("/")


def key():
    return os.environ.get("FIRECRAWL_API_KEY", "")


def enabled():
    return bool(key())


async def _post(path, body, timeout=60):
    from scrapling.fetchers import AsyncFetcher
    page = await AsyncFetcher.post(api_url() + path, json=body, timeout=timeout, retries=1,
                                   headers={"Authorization": f"Bearer {key()}",
                                            "Content-Type": "application/json"})
    data = page.json() if page.body else {}
    if page.status != 200 or not data.get("success"):
        raise RuntimeError(f"firecrawl {path}: HTTP {page.status} {str(data.get('error', ''))[:120]}")
    return data


# --- markdown → the plain text ask·200 ranks, and the images it shows ---

MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def md_images(md, base=""):
    """![alt](url) → [{url, alt, caption}], icons and alt-less images dropped (as page_images)."""
    import multimodal
    out, seen = [], set()
    for alt, url in MD_IMAGE.findall(md or ""):
        url = urllib.parse.urljoin(base, url)
        alt = " ".join(alt.split())
        if url.startswith("http") and alt and url not in seen and not multimodal.ICON.search(url):
            seen.add(url)
            out.append({"url": url, "alt": alt, "caption": ""})
    return out[:8]


MD_MARK = re.compile(r"^\s{0,3}([-*+>]|\d+\.)\s+")


def md_text(md, min_chars=40):
    """Markdown → sentences. Paragraph lines are JOINED (a line break is not a sentence end —
    transcripts and wrapped prose stay whole); headings are dropped (titles, not content);
    list items, quotes and table rows stand alone; images and link targets go; short blocks
    (menus, breadcrumbs, 'Jump to content') are discarded."""
    blocks, cur = [], []

    def flush():
        if cur:
            blocks.append(" ".join(cur))
            cur.clear()
    for line in MD_IMAGE.sub("", md or "").splitlines():
        line = MD_LINK.sub(r"\1", line)
        if not line.strip():
            flush()
        elif re.match(r"^\s{0,3}#{1,6}\s", line):
            flush()                                    # a heading: structure, not prose
        elif MD_MARK.match(line) or line.lstrip().startswith("|"):
            flush()
            blocks.append(MD_MARK.sub("", line))       # a list item / quote / table row
        else:
            cur.append(line)
    flush()
    keep = []
    for b in blocks:
        b = re.sub(r"\s+", " ", re.sub(r"[*_`|]+", " ", b)).strip()
        if len(b) >= min_chars and not set(b) <= set("-: "):
            keep.append(b if b[-1] in ".!?" else b + ".")
    return " ".join(dict.fromkeys(keep))


def _doc(item, source="firecrawl"):
    meta = item.get("metadata") or {}
    url = item.get("url") or meta.get("sourceURL") or meta.get("url") or ""
    md = item.get("markdown") or ""
    text = md_text(md) or (item.get("description") or "")
    return {"url": url, "title": item.get("title") or meta.get("title") or urllib.parse.urlparse(url).netloc,
            "text": text, "images": md_images(md, url), "source": source}


def _news(item):
    """A news hit → a doc: headline + snippet, dated (news is what search APIs miss)."""
    text = md_text(item.get("snippet") or "", min_chars=1) or item.get("title", "")
    return {"url": item.get("url", ""), "title": f"{item.get('title', '')} ({item.get('date', 'news')})".strip(),
            "text": text, "images": [], "source": "firecrawl-news"}


def _image(item):
    return {"url": item.get("imageUrl", ""), "page": item.get("url", ""), "alt": item.get("title", ""),
            "license": "", "source": "firecrawl"}


async def search_all(query, limit=10, timeout=60):
    """One call, three result types: web pages scraped to markdown, news, and images.
    → (docs, images) shaped for answer.py's ranking and multimodal.rank_images."""
    data = (await _post("/v2/search", {"query": query, "limit": limit, "sources": ["web", "news", "images"],
                                       "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}},
                        timeout)).get("data") or {}
    docs = [_doc(it) for it in data.get("web", [])] + [_news(it) for it in data.get("news", [])]
    images = [im for im in map(_image, data.get("images", [])) if im["url"].startswith("http") and im["alt"]]
    return [d for d in docs if d["url"] and d["text"]], images


async def search(query, limit=10, timeout=60):
    """Web results only, each already scraped to markdown → ask·200 docs."""
    data = await _post("/v2/search", {"query": query, "limit": limit,
                                      "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}},
                       timeout)
    return [d for d in map(_doc, (data.get("data") or {}).get("web", [])) if d["url"] and d["text"]]


async def scrape(url, timeout=60):
    """One page through Firecrawl's browser (JS rendered, anti-bot handled) → an ask·200 doc."""
    data = await _post("/v2/scrape", {"url": url, "formats": ["markdown"], "onlyMainContent": True}, timeout)
    doc = data.get("data") or {}
    return _doc({**doc, "url": (doc.get("metadata") or {}).get("sourceURL") or url}, "firecrawl-rescue")


if __name__ == "__main__":
    import argparse
    import asyncio
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="+")
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args()
    if not enabled():
        raise SystemExit("set FIRECRAWL_API_KEY (get one at firecrawl.dev)")
    for d in asyncio.run(search(" ".join(a.query), a.limit)):
        print(f"{d['title'][:70]}\n  {d['url']}\n  {d['text'][:160]}…  ({len(d['images'])} images)\n")
