"""multimodal.py — ask·200 sees: images in, images out, images read by the model.

pipeline: scraped pages ─► <img> + alt + <figcaption> ─┐
          Commons + Openverse image APIs ──────────────┼─► rank_images (BM25 on words,
                                                       │    + CLIP on pixels with --clip)
          your photo (--image) ─► Claude vision or CLIP tags ─► a text query
          top images + your photo ─► Claude, as image blocks, when it writes the answer

Three directions, each optional and each falling back to something that works:

    images OUT   every scraped page yields its content images (icons, logos, spacers
                 and tracking pixels filtered); Wikimedia Commons and Openverse add
                 licensed images. They are ranked against the question — by their
                 words (alt text, caption, page title, plus a bonus when the page is
                 cited in the answer) and, with --clip, by CLIP on the pixels: the
                 same two towers the rest of this repo is built on.
    images IN    `answer.py --image photo.jpg "what is this?"` turns the photo into
                 a search query — Claude vision when a key resolves, CLIP zero-shot
                 tags (templates.TAG_VOCABULARY) when torch is installed.
    images READ  when Claude writes the answer it also SEES your photo and up to 3
                 top-ranked images, each labelled with the source it came from.
"""
import base64
import mimetypes
import re
import urllib.parse

import answer

ICON = re.compile(r"(^data:|\.svg(\?|$)|logo|icon|sprite|avatar|badge|pixel|spacer|blank|button|"
                  r"emoji|/ads?/|banner|tracking|gravatar)", re.I)
RASTER = re.compile(r"\.(jpe?g|png|gif|webp)(\?|$)", re.I)


def _src(img, base):
    """The best URL for an <img>: largest srcset candidate, then lazy-load attrs, then src."""
    a = img.attrib
    srcset = a.get("srcset") or a.get("data-srcset") or ""
    if srcset:
        cands = [c.strip().split() for c in srcset.split(",") if c.strip()]
        best = max(cands, key=lambda c: float(re.sub(r"[^\d.]", "", c[1]) or 0) if len(c) > 1 else 0)
        return urllib.parse.urljoin(base, best[0])
    for k in ("data-src", "data-original", "src"):
        if a.get(k):
            return urllib.parse.urljoin(base, a[k])
    return ""


def page_images(page, base, limit=8, min_side=120):
    """Content images of a Scrapling page → [{url, alt, caption}], chrome and icons dropped."""
    out, seen = [], set()
    for img in page.xpath("//img[not(ancestor::nav or ancestor::header or ancestor::footer "
                          "or ancestor::aside)]"):
        url = _src(img, base)
        if not url.startswith("http") or url in seen or ICON.search(url):
            continue
        dims = [img.attrib.get(k, "") for k in ("width", "height")]
        if any(d.isdigit() and int(d) < min_side for d in dims):
            continue                                   # declared too small to be content
        caption = " ".join(" ".join(c.get_all_text(separator=" ", strip=True)
                                    for f in img.xpath("ancestor::figure[1]")
                                    for c in f.css("figcaption")).split())
        alt = " ".join((img.attrib.get("alt") or img.attrib.get("title") or "").split())
        if not (alt or caption):
            continue                                   # an image nobody described can't be ranked by words
        seen.add(url)
        out.append({"url": url, "alt": alt, "caption": caption})
        if len(out) >= limit:
            break
    return out


# --- image APIs: licensed images with descriptions ---

def _commons(page):
    pages = sorted((page.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        desc = re.sub(r"<[^>]+>", " ", (meta.get("ImageDescription") or {}).get("value", ""))
        title = re.sub(r"^File:|\.\w+$", "", p.get("title", ""))
        if info.get("thumburl"):
            out.append({"url": info["thumburl"], "page": info.get("descriptionurl", ""),
                        "alt": " ".join(f"{title}. {desc}".split())[:300],
                        "license": (meta.get("LicenseShortName") or {}).get("value", "")})
    return out


def _openverse(page):
    return [{"url": r.get("thumbnail") or r["url"], "page": r.get("foreign_landing_url") or r["url"],
             "alt": " ".join(f"{r.get('title', '')}. {' '.join(t['name'] for t in r.get('tags') or [])}".split())[:300],
             "license": f"CC {r.get('license', '').upper()} {r.get('license_version') or ''}".strip()}
            for r in page.json().get("results", []) if r.get("url")]


IMAGE_SOURCES = {
    "commons": ("https://commons.wikimedia.org/w/api.php?format=json&action=query&generator=search"
                "&gsrnamespace=6&gsrsearch={q}&gsrlimit=20&prop=imageinfo&iiprop=url|extmetadata"
                "&iiurlwidth=480", _commons),
    "openverse": ("https://api.openverse.org/v1/images/?q={q}&page_size=20", _openverse),
}


async def image_search(query, only=None, timeout=10):
    """Both image APIs at once → ([image], {source: error})."""
    import asyncio

    async def one(name):
        tpl, parse = IMAGE_SOURCES[name]
        try:
            page = await answer._get(tpl.format(q=urllib.parse.quote_plus(query)), timeout)
            if page.status >= 400:
                raise RuntimeError(f"HTTP {page.status}")
            return name, [{**im, "source": name} for im in parse(page)], None
        except Exception as e:
            return name, [], str(e).split("\n")[0][:80]

    imgs, errors = [], {}
    for name, got, err in await asyncio.gather(*(one(n) for n in IMAGE_SOURCES if not only or n in only)):
        imgs += got
        if err:
            errors[name] = err
    return imgs, errors


# --- ranking: words first, pixels optional ---

def rank_images(question, docs, api_images, cited, k=6, cited_bonus=1.0):
    """All candidate images → top-k by BM25 on their words, + a bonus if their page is cited.

    docs: scraped/API docs, each may carry "images"; cited: {page url: citation n}.
    """
    cands = []
    for d in docs:
        for im in d.get("images", []):
            cands.append({**im, "page": d["url"], "title": d["title"], "source": "page"})
    cands += [{**im, "title": ""} for im in api_images]
    seen, uniq = set(), []
    for c in cands:
        if c["url"] not in seen:
            seen.add(c["url"])
            uniq.append(c)
    if not uniq:
        return []
    scores = answer.bm25(question, [f"{c['alt']} {c.get('caption', '')} {c['title']}" for c in uniq])
    for c, s in zip(uniq, scores):
        c["n"] = cited.get(c["page"])
        c["score"] = round(s + (cited_bonus if c["n"] else 0.0), 4)
    order = sorted(range(len(uniq)), key=lambda i: -uniq[i]["score"])
    return [uniq[i] for i in order if uniq[i]["score"] > 0][:k]


async def _download(urls, timeout=10):
    import asyncio

    async def one(u):
        try:
            page = await answer._get(u, timeout)
            return page.body if page.status < 400 else None
        except Exception:
            return None
    return await asyncio.gather(*map(one, urls))


async def clip_rerank(question, images, embedder=None, weight=0.5):
    """Re-score images by CLIP: cosine(question text, image pixels), blended with the word score.

    Needs torch + transformers (embedder.py). Images that fail to download keep their word score.
    """
    import io
    import tempfile
    import numpy as np
    from PIL import Image
    if embedder is None:
        import embedder as emb_mod
        embedder = emb_mod.ClipEmbedder()
    blobs = await _download([im["url"] for im in images])
    paths, idx = [], []
    tmp = tempfile.mkdtemp()
    for i, b in enumerate(blobs):
        try:
            p = f"{tmp}/{i}.png"
            Image.open(io.BytesIO(b)).convert("RGB").save(p)
            paths.append(p)
            idx.append(i)
        except Exception:
            continue
    if not paths:
        return images
    q = embedder.embed_texts([f"a photo of {question}"])[0]
    sims = embedder.embed_images(paths) @ q
    top = max(im["score"] for im in images) or 1.0
    for i, s in zip(idx, sims):
        images[i]["clip"] = round(float(s), 4)
        images[i]["score"] = round(weight * images[i]["score"] / top + (1 - weight) * float(np.clip(s * 3, 0, 1)), 4)
    return sorted(images, key=lambda im: -im["score"])


# --- images in: a photo becomes a query; images become content blocks ---

def image_block(path_or_url):
    """A Claude image content block from a local path (base64) or an http(s) URL."""
    if str(path_or_url).startswith("http"):
        return {"type": "image", "source": {"type": "url", "url": path_or_url}}
    mime = mimetypes.guess_type(str(path_or_url))[0] or "image/jpeg"
    data = base64.standard_b64encode(open(path_or_url, "rb").read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def clip_tags(path, k=3, embedder=None):
    """CLIP zero-shot: the k vocabulary words that best describe a photo (needs torch)."""
    import templates
    import tagger
    if embedder is None:
        import embedder as emb_mod
        embedder = emb_mod.ClipEmbedder()
    tag_embs = embedder.embed_texts(templates.tag_prompts())
    return tagger.top_tags(embedder.embed_images([path])[0], tag_embs, templates.TAG_VOCABULARY, k)


def image_to_query(path, question="", client=None, model="claude-opus-5-5", gallery=None):
    """Photo → a short web-search query. A gallery photo is described by its stored CLIP tags
    (no model, no key); otherwise Claude vision, else CLIP zero-shot tags."""
    import pathlib
    for it in gallery or []:
        if pathlib.Path(it["file"]).name == pathlib.Path(path).name:   # the photo IS in the gallery
            return " ".join(dict.fromkeys(it["tags"][:3] + answer.tokens(question)))
    if client:
        try:
            return answer._claude(
                client, model, 300,
                "Write ONE short web-search query (max 8 words, no quotes) that would find "
                "pages explaining what this image shows, focused on the user's question. "
                "Name the specific thing (species, landmark, product, artwork) if you can.",
                [image_block(path), {"type": "text", "text": question or "What is this?"}]).splitlines()[0]
        except Exception as e:
            print(f"  (vision query failed, trying CLIP: {e})")
    try:
        tags = clip_tags(path)
    except ImportError as e:
        raise SystemExit(f"--image needs an Anthropic key (Claude vision) or torch for CLIP: {e}")
    return " ".join(dict.fromkeys(tags + answer.tokens(question)))


# --- the bridge to CLIP·search: the gallery and the web, feeding each other ---
# Twins of docs/js/bridge.js (knnTags, gallerySearch), pinned by test_ask.mjs.

def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def knn_tags(emb, items, k=3, n=3):
    """An image embedding → words: vote the tags of its k nearest gallery images, each vote
    weighted by similarity × 0.5^position (a tag's rank in its image's list — the first is the
    most specific). The gallery already knows how to describe what it holds."""
    sims = [_dot(it["image_emb"], emb) for it in items]
    votes = {}
    for i in sorted(range(len(items)), key=lambda i: -sims[i])[:k]:
        for pos, t in enumerate(items[i]["tags"]):
            votes[t] = votes.get(t, 0.0) + sims[i] * 0.5 ** pos
    tags = list(votes)
    return [tags[i] for i in sorted(range(len(tags)), key=lambda i: -votes[tags[i]])][:n]


def gallery_search(query, items, k=6, qvec=None):
    """Words → gallery items: BM25 over caption + tags (keyless), or CLIP when given a query vector."""
    scores = ([_dot(it["image_emb"], qvec) for it in items] if qvec is not None else
              answer.bm25(query, [f"{it['caption']} {' '.join(it['tags'])}" for it in items]))
    order = [i for i in sorted(range(len(items)), key=lambda i: -scores[i]) if qvec is not None or scores[i] > 0]
    return [{"index": i, "file": items[i]["file"], "tags": items[i]["tags"], "score": round(scores[i], 4)}
            for i in order[:k]]


def load_gallery(path="docs/db.json"):
    import json
    return json.load(open(path))["items"]


def clip_gallery_search(query, items, k=6, embedder=None):
    """The real thing: embed the question with CLIP's text tower, rank gallery pixels (needs torch)."""
    if embedder is None:
        import embedder as emb_mod
        embedder = emb_mod.ClipEmbedder()
    return gallery_search(query, items, k, qvec=list(embedder.embed_texts([f"a photo of {query}"])[0]))


async def save_images(images, outdir, question=""):
    """Download ranked web images into `outdir` with a manifest, ready for `ingest.py outdir/*`:
    the web's answer grows the CLIP gallery. Returns the manifest entries written."""
    import io
    import json
    import pathlib
    import hashlib
    from PIL import Image
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "web_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    have = {m["sha1"] for m in manifest}
    written = []
    for im, blob in zip(images, await _download([im["url"] for im in images])):
        try:
            sha = hashlib.sha1(blob).hexdigest()
            if sha in have:
                continue                                  # same pixels already saved
            pic = Image.open(io.BytesIO(blob)).convert("RGB")
            if min(pic.size) < 160:
                continue                                  # too small to be worth embedding
            f = out / f"web_{sha[:10]}.jpg"
            pic.save(f, quality=90)
            entry = {"file": str(f), "sha1": sha, "url": im["url"], "page": im.get("page", ""),
                     "alt": im.get("alt", ""), "license": im.get("license") or "unknown — check the page",
                     "question": question}
            manifest.append(entry)
            written.append(entry)
            have.add(sha)
        except Exception:
            continue
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    return written
