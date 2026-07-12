"""The spider: a real web crawler — point it at a page, harvest its images.

pipeline: seed URLs ──► [spider: BFS pages ─► <img> tags] ──► images/ ─► ingest

crawler.py ASKS an API (Commons search) and gets curated, licensed results.
This file CRAWLS: it fetches pages you point it at, follows same-domain
links breadth-first, and downloads the images it finds — the classic way
image-search corpora are actually built.

Politeness is enforced in code, not left to good intentions:

    robots.txt   checked per host before every fetch, cached
    pacing       one request per second (no bursts, pages and images alike)
    scope        same-domain as the seeds unless --any-domain
    caps         hard limits on pages visited and images kept
    identity     a truthful User-Agent naming this repo

Quality gate: files under 20 KB or 160 px on their shorter side are
skipped (icons, spacers), and duplicates are dropped by content hash.
Every kept image gets a manifest receipt recording the page it was found
on — but unlike Commons, an arbitrary page carries NO license metadata:
what you may do with crawled files is your responsibility, and the
manifest says so.

    python3 spider.py https://example.com/gallery --max-images 12
    python3 ingest.py images/spider_*.jpg     # then embed + tag + store

The --delay flag exists so tests can crawl a local fixture site quickly;
the politeness default stands for the real web.
"""
import argparse
import hashlib
import io
import json
import pathlib
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser

UA = "clip-vlm-101-spider/1.0 (https://github.com/sugeerth/clip-vlm-101)"
MANIFEST = "crawl_manifest.json"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
MIN_BYTES = 20_000      # below this it's an icon, not a photo
MIN_SIDE = 160          # shorter side, in pixels


class PageParser(HTMLParser):
    """Collects the two things a crawler cares about: links and images."""
    def __init__(self, base):
        super().__init__()
        self.base, self.links, self.images = base, [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(urllib.parse.urljoin(self.base, a["href"]))
        if tag == "img" and a.get("src"):
            self.images.append(urllib.parse.urljoin(self.base, a["src"]))


def _get(url) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=30).read()


def _allowed(robots_cache, url) -> bool:
    """robots.txt per host, cached; unreachable robots means allow."""
    host = urllib.parse.urlparse(url)
    base = f"{host.scheme}://{host.netloc}"
    if base not in robots_cache:
        rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
        try:
            rp.read()
        except OSError:
            rp = None
        robots_cache[base] = rp
    rp = robots_cache[base]
    return rp is None or rp.can_fetch(UA, url)


def _keepable(data: bytes) -> bool:
    """The quality gate: big enough in bytes AND pixels, and a real image."""
    if len(data) < MIN_BYTES:
        return False
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            return min(im.size) >= MIN_SIDE
    except Exception:
        return False


def crawl(seeds, max_pages=10, max_images=12, out="images",
          same_domain=True, delay=1.0):
    """BFS from the seeds; download every keepable image. Returns new paths."""
    out = pathlib.Path(out)
    out.mkdir(exist_ok=True)
    manifest_path = out / MANIFEST
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else [])
    hashes = {m.get("sha1") for m in manifest}
    domains = {urllib.parse.urlparse(s).netloc for s in seeds}

    robots, queue, seen, saved = {}, list(seeds), set(seeds), []
    pages = 0
    while queue and pages < max_pages and len(saved) < max_images:
        page_url = queue.pop(0)
        if not _allowed(robots, page_url):
            print(f"  robots.txt forbids {page_url}")
            continue
        time.sleep(delay)
        try:
            parser = PageParser(page_url)
            parser.feed(_get(page_url).decode("utf-8", "replace"))
        except Exception as e:  # a dead page never stops the crawl
            print(f"  SKIP page {page_url}: {e}")
            continue
        pages += 1

        for img_url in parser.images:
            if len(saved) >= max_images:
                break
            path_part = urllib.parse.urlparse(img_url).path.lower()
            if not path_part.endswith(IMG_EXT) or img_url in seen:
                continue
            seen.add(img_url)
            if not _allowed(robots, img_url):
                continue
            time.sleep(delay)
            try:
                data = _get(img_url)
            except Exception as e:
                print(f"  SKIP {img_url}: {e}")
                continue
            if not _keepable(data):
                print(f"  thin {img_url.rsplit('/', 1)[-1]} (icon-sized — skipped)")
                continue
            sha1 = hashlib.sha1(data).hexdigest()[:12]
            if sha1 in hashes:
                print(f"  have {img_url.rsplit('/', 1)[-1]} (same bytes)")
                continue
            hashes.add(sha1)
            path = out / f"spider_{sha1}{pathlib.Path(path_part).suffix}"
            path.write_bytes(data)
            manifest.append({
                "name": img_url.rsplit("/", 1)[-1], "file": str(path),
                "source": page_url, "sha1": sha1,
                "author": "", "license": "",   # an arbitrary page carries none —
                "term": "spider",              # licensing is YOUR responsibility
            })
            saved.append(path)
            print(f"  +    {path.name}  (from {page_url})")

        for link in parser.links:
            host_ok = (not same_domain
                       or urllib.parse.urlparse(link).netloc in domains)
            if link.startswith(("http://", "https://")) and host_ok and link not in seen:
                seen.add(link)
                queue.append(link)

    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"{len(saved)} new file(s) from {pages} page(s); "
          f"manifest: {manifest_path} ({len(manifest)} records)")
    if saved:
        print("licensing note: crawled pages carry no license metadata — check "
              "before you reuse.\nnext: python3 ingest.py "
              f"{out}/spider_*.jpg && python3 export_web.py")
    return saved


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", nargs="+", help="page URLs to start crawling from")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--out", default="images")
    ap.add_argument("--any-domain", action="store_true",
                    help="follow links off the seed domains too")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests (tests use 0)")
    args = ap.parse_args()
    crawl(args.seeds, args.max_pages, args.max_images, args.out,
          same_domain=not args.any_domain, delay=args.delay)


if __name__ == "__main__":
    main()
