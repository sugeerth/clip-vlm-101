"""The crawler agent: grow the gallery from Wikimedia Commons, with receipts.

pipeline: search term ──► [crawler] ──► images/ + a manifest of attributions

download_samples.py fetches a FIXED list. This agent DISCOVERS: it asks the
Commons search API for freely-licensed images matching a term, downloads
small thumbnails politely (one request at a time, a real User-Agent, a
pause between fetches), and writes a standardized manifest recording where
every file came from — source page, author, license — because a crawler
without receipts is a liability, not a tool.

The output feeds the normal pipeline: crawl, then ingest.

    python3 crawler.py "red panda" -n 6
    python3 ingest.py images/*.jpg          # embed + tag + store the new files

Long-running mode — the corpus-growing agent:

    python3 crawler.py "red panda" --every 60    # re-crawl hourly, forever
    (each round only downloads files it does not already have)

The --api flag exists so tests can point the crawler at a local stub server
— the crawl logic is verified without touching the real Commons.
"""
import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = "clip-vlm-101-crawler/1.0 (https://github.com/sugeerth/clip-vlm-101)"
MANIFEST = "crawl_manifest.json"


def _get(url) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=30).read()


def discover(term, n=6, api=API, width=384):
    """Ask Commons for freely-licensed images matching `term`.

    Returns standardized records: one dict per image with name, a thumb
    URL sized for this pipeline, the source page, author and license.
    """
    params = urllib.parse.urlencode({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": 6, "gsrlimit": n,
        "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": width,
    })
    data = json.loads(_get(f"{api}?{params}"))
    records = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {}) or {}
        field = lambda k: (meta.get(k, {}) or {}).get("value", "")
        records.append({
            "name": page.get("title", "").removeprefix("File:"),
            "thumb_url": info.get("thumburl") or info.get("url", ""),
            "source": info.get("descriptionurl", ""),
            "author": field("Artist"),
            "license": field("LicenseShortName"),
            "term": term,
        })
    return [r for r in records if r["thumb_url"]]


def crawl(term, n=6, out="images", api=API, pause=2.0):
    """Discover, download what's new, extend the manifest. Returns new paths."""
    out = pathlib.Path(out)
    out.mkdir(exist_ok=True)
    manifest_path = out / MANIFEST
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else [])
    have = {m["name"] for m in manifest}

    new_paths = []
    for rec in discover(term, n, api):
        if rec["name"] in have:
            print(f"  have {rec['name']}")
            continue
        time.sleep(pause)  # politeness: no bursts
        safe = "crawl_" + "".join(
            c if c.isalnum() or c in "._-" else "_" for c in rec["name"])
        path = out / safe
        try:
            path.write_bytes(_get(rec["thumb_url"]))
        except Exception as e:  # a dead thumb never stops the crawl
            print(f"  SKIP {rec['name']}: {e}")
            continue
        rec["file"] = str(path)
        manifest.append(rec)
        new_paths.append(path)
        print(f"  +    {safe}  ({rec['license'] or 'license unknown'})")

    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"{len(new_paths)} new file(s); manifest: {manifest_path} "
          f"({len(manifest)} records)")
    return new_paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("term", help="what to search Commons for")
    ap.add_argument("-n", type=int, default=6, help="how many images to ask for")
    ap.add_argument("--out", default="images")
    ap.add_argument("--api", default=API, help="override for tests / mirrors")
    ap.add_argument("--every", type=float, metavar="MIN",
                    help="long-running mode: re-crawl every MIN minutes")
    args = ap.parse_args()

    while True:
        crawl(args.term, args.n, args.out, args.api)
        if not args.every:
            break
        print(f"sleeping {args.every} min — Ctrl-C to stop")
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
