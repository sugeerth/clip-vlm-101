"""mcp_server.py — ask·200 as an MCP server: any MCP client gets the whole toolbelt.

pipeline: MCP client (Claude Code, Claude Desktop, research.py, ...) ─► [this server]
          ─► ask · web_search · read_page · search_sources ─► Scrapling + 7 APIs

Six tools, all thin wrappers over answer.py, sources.py and multimodal.py:

    ask             the full pipeline: fan out, scrape ≤N pages + APIs, BM25,
                    return a cited extractive answer, the numbered passages and
                    the ranked images, so the calling model writes its own answer
    web_search      4 search engines at once → result URLs
    read_page       one URL → readable text (Scrapling; Firecrawl when blocked/JS-only)
    search_sources  7 keyless APIs at once → whole documents
    find_images     Commons + Openverse → licensed images, ranked
    gallery_search  the CLIP·search gallery (docs/db.json) by meaning — the bridge

    pip install "scrapling[fetchers]" mcp
    python3 mcp_server.py                     # stdio (what desktop clients launch)
    python3 mcp_server.py --http --port 8765  # streamable HTTP on 127.0.0.1

    claude mcp add ask200 -- python3 /path/to/clip-vlm-101/mcp_server.py

The server never calls an LLM itself (`ask` runs extractive): the client
already is one, and this keeps the server keyless.
"""
import argparse
import json
import pathlib

from mcp.server.mcpserver import MCPServer

import answer
import multimodal
import sources

server = MCPServer(
    "ask200",
    instructions="Web research tools. Prefer `ask` for a question: it reads up to `pages` web "
                 "pages plus 7 APIs and returns numbered passages to cite as [n]. Use web_search "
                 "+ read_page to dig into one site, search_sources for papers/code/forums.")


@server.tool()
async def ask(question: str, pages: int = 60, words: int = 45, k: int = 10) -> str:
    """Answer a question from the web: search 4 engines + 7 APIs, scrape up to `pages`
    pages with Scrapling, rank passages with BM25. Returns a cited extractive answer and
    the top-k numbered passages (cite them as [n])."""
    r = await answer.ask_async(question, pages=min(pages, 200), words=words, k=k, use_llm=False)
    return json.dumps({k_: r[k_] for k_ in ("answer", "sources", "images", "stats", "ledger")} |
                      {"passages": [{"n": p["n"], "url": p["src"], "text": p["text"]}
                                    for p in r["passages"]]}, indent=1)


@server.tool()
async def web_search(query: str, limit: int = 20) -> str:
    """Search DuckDuckGo, Bing, Wikipedia and Mojeek at once; return deduplicated result URLs."""
    urls, ledger = await sources.web_search(query, limit)
    return json.dumps({"urls": urls, "ledger": ledger}, indent=1)


@server.tool()
async def read_page(url: str, max_chars: int = 8000, engine: str = "auto") -> str:
    """Fetch one web page and return its readable text — navigation, headers, footers and forms
    stripped. engine: "scrapling" (local, Chrome TLS fingerprint), "firecrawl" (JS-rendering
    browser; needs FIRECRAWL_API_KEY), or "auto" (Scrapling, then Firecrawl if blocked/empty)."""
    return json.dumps(await sources.read_page(url, max_chars, engine=engine), indent=1)


@server.tool()
async def find_images(query: str, limit: int = 8) -> str:
    """Find licensed images for a query on Wikimedia Commons and Openverse, ranked by how well
    their titles/descriptions match. Returns image URL, landing page, description, license."""
    imgs, errors = await multimodal.image_search(query)
    return json.dumps({"images": multimodal.rank_images(query, [], imgs, {}, k=limit), "errors": errors}, indent=1)


GALLERY = str(pathlib.Path(__file__).with_name("docs") / "db.json")


@server.tool()
async def gallery_search(query: str, k: int = 6, clip: bool = False) -> str:
    """Search the CLIP·search image gallery (docs/db.json, stored CLIP vectors) by meaning:
    keyword BM25 over captions and tags, or with clip=true CLIP's text tower against the
    images' pixels (needs torch). Returns files, tags and scores."""
    items = multimodal.load_gallery(GALLERY)
    hits = multimodal.clip_gallery_search(query, items, k) if clip else multimodal.gallery_search(query, items, k)
    return json.dumps({"gallery": GALLERY, "hits": hits}, indent=1)


@server.tool()
async def search_sources(query: str, only: list[str] | None = None, per_source: int = 5) -> str:
    """Query keyless APIs at once — wikipedia, hackernews, stackexchange, openalex, arxiv,
    github, reddit (or just `only` these) — and return their documents."""
    got, errors = await sources.search_sources(query, only)
    return json.dumps({"docs": {n: [{**d, "text": d["text"][:1200]} for d in ds[:per_source]]
                                for n, ds in got.items()}, "errors": errors}, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.http:
        server.run("streamable-http", host="127.0.0.1", port=a.port)
    else:
        server.run()
