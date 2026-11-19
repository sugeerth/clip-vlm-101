"""research.py — agent mode: Claude drives the toolbelt, plus any MCP servers you plug in.

pipeline: question ─► Claude (tool runner loop) ─┬─► local tools: scrape_search · read_page · search_sources · find_images
                                                  ├─► Anthropic server tools: web_search · web_fetch  (--server-tools)
                                                  ├─► remote MCP servers via the MCP connector   (--mcp NAME=URL)
                                                  └─► local MCP servers over stdio               (--mcp-stdio "CMD")
                  ◄── one answer, ≤N words, every claim [n]-cited, then its sources

answer.py is a fixed pipeline: one fan-out, one ranking, one answer. That is
fast and predictable, but it cannot follow a lead. Here Claude decides what
to search next, which page to open, and when it has read enough — with the
same Scrapling + API tools as answer.py, and whatever other tools MCP brings:

    python3 research.py "what changed in Scrapling 0.5"
    python3 research.py "..." --server-tools                     # + Claude's own web search/fetch
    python3 research.py "..." --mcp-stdio "scrapling mcp"        # + Scrapling's browser MCP tools
    python3 research.py "..." --mcp-stdio "python3 mcp_server.py" --no-local
    python3 research.py "..." --mcp docs=https://example.com/mcp # + any remote MCP server
    python3 research.py "where is this and when was it built" --image photo.jpg   # vision

Needs an Anthropic credential (ANTHROPIC_API_KEY or `ant auth login`) and
`pip install "anthropic[mcp]" "scrapling[fetchers]"`. Every tool call is
printed to stderr as it happens, so the trail of evidence is visible.
"""
import argparse
import asyncio
import contextlib
import json
import shlex
import sys

import sources

MODEL = "claude-opus-5-5"
SYSTEM = (
    "You are a research agent with web tools. Search widely, open the most promising pages, "
    "cross-check claims across independent sources, and stop as soon as the answer is well "
    "supported — usually within 3-8 tool calls. Then reply with ONLY: the answer in at most {w} "
    "words, leading with the answer itself, with a citation like [1] after every claim; then a "
    "blank line; then one line per cited source: [n] URL. If the sources disagree, say so in "
    "one clause. If nothing you read answers the question, reply: Not found in sources.")


def local_tools():
    """The ask·200 toolbelt as tool-runner tools (Scrapling engines, pages, 7 APIs)."""
    from anthropic import beta_async_tool

    @beta_async_tool
    async def scrape_search(query: str, limit: int = 15) -> str:
        """Search DuckDuckGo, Bing, Wikipedia and Mojeek at once (results pages scraped with
        Scrapling) and return deduplicated result URLs.

        Args:
            query: The web search query.
            limit: Maximum number of URLs to return.
        """
        urls, ledger = await sources.web_search(query, limit)
        return json.dumps({"urls": urls, "errors": {k: v["errors"] for k, v in ledger.items() if v["errors"]}})

    @beta_async_tool
    async def read_page(url: str, max_chars: int = 6000) -> str:
        """Fetch a web page with Scrapling and return its readable text (navigation, headers,
        footers and forms stripped).

        Args:
            url: The absolute URL to read.
            max_chars: Truncate the text to this many characters.
        """
        try:
            return json.dumps(await sources.read_page(url, max_chars))
        except Exception as e:
            return json.dumps({"url": url, "error": str(e)[:200]})

    @beta_async_tool
    async def search_sources(query: str, only: list[str] | None = None) -> str:
        """Query keyless APIs at once and return whole documents: wikipedia (article intros),
        hackernews (stories, comments), stackexchange (Q&A), openalex and arxiv (paper
        abstracts), github (repositories), reddit (posts).

        Args:
            query: The search query.
            only: Optional subset of source names to query.
        """
        got, errors = await sources.search_sources(query, only)
        return json.dumps({"docs": {n: [{**d, "text": d["text"][:800]} for d in ds[:5]]
                                    for n, ds in got.items()}, "errors": errors})

    @beta_async_tool
    async def find_images(query: str, limit: int = 6) -> str:
        """Find licensed images (Wikimedia Commons, Openverse) for a query, ranked by how well
        their descriptions match. Use when the user asks to see something or a figure helps.

        Args:
            query: What the image should show.
            limit: Maximum number of images.
        """
        import multimodal
        imgs, errors = await multimodal.image_search(query)
        return json.dumps({"images": multimodal.rank_images(query, [], imgs, {}, k=limit), "errors": errors})

    return [scrape_search, read_page, search_sources, find_images]


SERVER_TOOLS = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5}]


async def _stdio_tools(stack, commands, taken):
    """Launch each stdio MCP server and convert its tools for the tool runner."""
    from anthropic.lib.tools.mcp import async_mcp_tool
    from mcp import Client, StdioServerParameters
    tools = []
    for cmd in commands:
        argv = shlex.split(cmd)
        client = await stack.enter_async_context(Client(StdioServerParameters(command=argv[0], args=argv[1:])))
        for t in (await client.list_tools()).tools:
            if t.name in taken:                      # two servers offering `read_page`: first wins
                print(f"  (skipping duplicate tool {t.name!r} from {cmd!r})", file=sys.stderr)
                continue
            taken.add(t.name)
            tools.append(async_mcp_tool(t, client))
    return tools


def _trace(msg):
    for b in msg.content:
        if b.type in ("tool_use", "server_tool_use", "mcp_tool_use"):
            args = json.dumps(b.input)[:110]
            print(f"  → {b.name} {args}", file=sys.stderr)


async def research(question, words=45, model=MODEL, local=True, server_tools=False,
                   mcp=(), mcp_stdio=(), max_iterations=14, client=None, image=None):
    from anthropic import AsyncAnthropic
    client = client or AsyncAnthropic()
    async with contextlib.AsyncExitStack() as stack:
        tools = local_tools() if local else []
        taken = {t.name for t in tools} | ({t["name"] for t in SERVER_TOOLS} if server_tools else set())
        tools += await _stdio_tools(stack, mcp_stdio, taken)
        if server_tools:
            tools += SERVER_TOOLS
        betas = ["server-side-fallback-2026-07-01"]
        extra = {}
        if mcp:
            betas.append("mcp-client-2025-11-20")
            extra["mcp_servers"] = [{"type": "url", "name": n, "url": u} for n, u in mcp]
            tools += [{"type": "mcp_toolset", "mcp_server_name": n} for n, _ in mcp]
        if not tools:
            raise SystemExit("no tools: drop --no-local or add --server-tools / --mcp / --mcp-stdio")
        runner = client.beta.messages.tool_runner(
            model=model, max_tokens=16000, system=SYSTEM.format(w=words), tools=tools,
            messages=[{"role": "user", "content": _content(question, image)}], max_iterations=max_iterations,
            output_config={"effort": "medium"}, betas=betas, fallbacks="default", **extra)
        last = None
        async for msg in runner:
            _trace(msg)
            last = msg
    if last is None:
        raise RuntimeError("the model returned nothing")
    if last.stop_reason == "refusal":
        raise RuntimeError("the model declined this question")
    text = "".join(b.text for b in last.content if b.type == "text").strip()
    if last.stop_reason in ("tool_use", "pause_turn") or not text:
        text += "\n\n(stopped at the tool-call limit before a final answer; raise --max-iterations)"
    return text


def _content(question, image):
    """The user turn: text alone, or the photo (Claude vision) followed by the question."""
    if not image:
        return question
    import multimodal
    return [multimodal.image_block(image), {"type": "text", "text": question or "What is this? Find out."}]


def _pair(s):
    name, _, url = s.partition("=")
    if not url.startswith("http"):
        raise argparse.ArgumentTypeError("use NAME=https://host/path")
    return name, url


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="*")
    ap.add_argument("--words", type=int, default=45)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--server-tools", action="store_true", help="add Anthropic's web_search + web_fetch")
    ap.add_argument("--mcp", type=_pair, action="append", default=[], metavar="NAME=URL",
                    help="remote MCP server via the MCP connector (repeatable)")
    ap.add_argument("--mcp-stdio", action="append", default=[], metavar="CMD",
                    help='local stdio MCP server, e.g. "scrapling mcp" (repeatable)')
    ap.add_argument("--no-local", action="store_true", help="drop the built-in Scrapling/API tools")
    ap.add_argument("--max-iterations", type=int, default=14)
    ap.add_argument("--image", help="a photo (path or URL) Claude sees with the question")
    a = ap.parse_args()
    print(asyncio.run(research(" ".join(a.question), a.words, a.model, not a.no_local,
                               a.server_tools, a.mcp, a.mcp_stdio, a.max_iterations, image=a.image)))
