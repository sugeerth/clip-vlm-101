"""grow.py — turn the 14-image demo into a 1,000s-image one, for free.

pipeline: [topics] ─► crawler (Commons, licensed) ─► features (CLIP) ─► dedup ─► docs/db.json

The gallery is small because nobody grew it, not because it can't. This is one
command that does: for a big list of diverse topics it crawls freely-licensed
images from Wikimedia Commons (with attribution — crawler.py keeps the receipts),
embeds each with the SAME clip-ViT-B/32 the demo runs, drops near-duplicates, and
writes a bigger docs/db.json in the exact format the browser already searches.

Scales without bloating the repo: each item stores its embedding plus the REMOTE
Commons thumbnail URL (not a committed image file), so 100× the images is a few
MB of JSON and zero binary blobs — the browser loads thumbnails straight from the
free host, ranks with the same dot products, and nothing else changes.

    python3 grow.py --per 12               # ~120 topics × 12 ≈ 1,400 images (100×)
    python3 grow.py --per 30 --merge       # keep the current gallery, add ~3,600 more
    python3 grow.py --topics cat dog sushi --per 20   # your own topics
    python3 grow.py --selftest             # model-free: exercise dedup + export offline

Needs the model + network (runs on your machine, first run downloads CLIP ~600MB).
--selftest needs neither — it rebuilds the payload from the committed gallery.
"""
import argparse
import json
import pathlib

import numpy as np

from export_web import pca_2d

DOCS = pathlib.Path("docs")

# a deliberately broad vocabulary so the grown gallery is DIVERSE, not lopsided —
# animals, food, places, nature, objects, vehicles, plants, art, sport, weather.
DEFAULT_TOPICS = [
    "cat", "dog", "horse", "elephant", "tiger", "panda", "owl", "parrot", "dolphin",
    "butterfly", "bee", "frog", "penguin", "fox", "rabbit", "deer", "lion", "koala",
    "pizza", "sushi", "burger", "ramen", "salad", "pancakes", "ice cream", "coffee",
    "bread", "cheese", "strawberry", "avocado", "curry", "taco", "cupcake", "steak",
    "eiffel tower", "colosseum", "taj mahal", "golden gate bridge", "big ben",
    "great wall of china", "pyramids of giza", "sydney opera house", "mount fuji",
    "mountain", "waterfall", "beach", "desert", "forest", "glacier", "volcano",
    "aurora", "canyon", "coral reef", "rainforest", "lake", "island", "cave",
    "bicycle", "motorcycle", "sailboat", "train", "airplane", "hot air balloon",
    "vintage car", "tractor", "helicopter", "skateboard", "kayak", "rocket",
    "sunflower", "rose", "tulip", "orchid", "cactus", "bonsai", "lavender field",
    "guitar", "piano", "violin", "drum kit", "telescope", "typewriter", "camera",
    "lighthouse", "windmill", "castle", "temple", "cathedral", "skyscraper",
    "soccer", "basketball", "surfing", "rock climbing", "chess", "ballet", "skiing",
    "galaxy", "nebula", "planet", "moon", "lightning", "rainbow", "snowflake",
    "origami", "stained glass", "mosaic", "graffiti", "sculpture", "pottery",
    "mushroom", "seashell", "feather", "autumn leaves", "cherry blossom",
]


def dedup(records, thresh=0.985, key="image_emb"):
    """Greedy near-duplicate removal: keep a record only if it isn't ≥ thresh
    cosine to one already kept. Commons returns the same landmark shot many
    times; this keeps the gallery varied. O(n·kept) — fine for thousands."""
    kept, mats = [], []
    for r in records:
        v = np.asarray(r[key], dtype=np.float64)
        if mats and float(np.max(np.asarray(mats) @ v)) >= thresh:
            continue
        kept.append(r)
        mats.append(v)
    return kept


def build_payload(records):
    """Format embedded records into the exact docs/db.json shape the browser
    reads (items + a 2-D PCA map + the projection basis). `file` is whatever the
    record carries — a local path for the seed gallery, a remote Commons URL for
    crawled images — and the browser renders either as an <img src>."""
    coords, mean, components = pca_2d([r["image_emb"] for r in records])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    items = []
    for i, r in enumerate(records):
        xy = (coords[i] - lo) / span
        item = {
            "file": r["file"],
            "caption": r["caption"],
            "tags": r["tags"],
            "map": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
            "image_emb": [round(float(x), 5) for x in r["image_emb"]],
            "text_emb": [round(float(x), 5) for x in r["text_emb"]],
        }
        if r.get("attr"):                       # attribution travels with the image
            item["attr"] = r["attr"]
        items.append(item)
    return {
        "model": "clip-vit-base-patch32", "dim": 512, "items": items,
        "pca": {
            "mean": [round(float(x), 5) for x in mean],
            "components": [[round(float(x), 5) for x in c] for c in components],
            "lo": [float(lo[0]), float(lo[1])],
            "span": [float(span[0]), float(span[1])],
        },
    }


def _seed_records():
    """The committed gallery as records (for --merge and --selftest), model-free."""
    db = json.loads((DOCS / "db.json").read_text())
    return [{"file": it["file"], "caption": it["caption"], "tags": it["tags"],
             "image_emb": it["image_emb"], "text_emb": it["text_emb"],
             "attr": it.get("attr")} for it in db["items"]]


def grow(topics, per, fx, images_dir="images"):
    """Crawl + embed each topic. Returns records with the REMOTE thumb URL as
    `file` and attribution attached, so nothing binary lands in the repo."""
    import crawler

    for t in topics:
        print(f"crawling {t!r}…")
        crawler.crawl(t, per, out=images_dir)
    manifest = json.loads((pathlib.Path(images_dir) / crawler.MANIFEST).read_text())
    crawled = [m for m in manifest if m.get("file") and m.get("thumb_url")]
    recs = fx.extract_batch([m["file"] for m in crawled])
    out = []
    for r, m in zip(recs, crawled):
        out.append({**r, "file": m["thumb_url"],
                    "attr": {"author": m.get("author", ""), "license": m.get("license", ""),
                             "source": m.get("source", "")}})
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=12, help="images to fetch per topic")
    ap.add_argument("--topics", nargs="*", help="override the built-in topic list")
    ap.add_argument("--merge", action="store_true", help="keep the current gallery and add to it")
    ap.add_argument("--out", default=str(DOCS / "db.json"))
    ap.add_argument("--dedup", type=float, default=0.985, help="near-duplicate cosine cutoff")
    ap.add_argument("--selftest", action="store_true",
                    help="model-free: rebuild the payload from the committed gallery, check dedup + export")
    args = ap.parse_args()

    if args.selftest:
        seed = _seed_records()
        # dedup is a no-op on the curated gallery, and drops an injected copy
        assert len(dedup(seed)) == len(seed), "curated gallery has no near-dupes"
        assert len(dedup(seed + [dict(seed[0])])) == len(seed), "an exact copy must be dropped"
        payload = build_payload(seed)
        assert len(payload["items"]) == len(seed)
        assert len(payload["pca"]["mean"]) == 512 and len(payload["pca"]["components"]) == 2
        assert all(len(it["image_emb"]) == 512 and 0 <= it["map"][0] <= 1 for it in payload["items"])
        assert json.loads(json.dumps(payload))  # round-trips as JSON
        print(f"grow selftest passed  (rebuilt {len(seed)} items + 2-D map, dedup verified)")
        raise SystemExit(0)

    from features import FeatureExtractor

    topics = args.topics or DEFAULT_TOPICS
    fx = FeatureExtractor()
    records = grow(topics, args.per, fx)
    if args.merge:
        records = _seed_records() + records
    before = len(records)
    records = dedup(records, args.dedup)
    payload = build_payload(records)
    pathlib.Path(args.out).write_text(json.dumps(payload))
    print(f"\nwrote {args.out}: {len(payload['items'])} images "
          f"({before - len(records)} near-dupes dropped) — "
          f"{'merged with' if args.merge else 'replaced'} the seed gallery.")
    print("remote thumbnails + embeddings only; no image files committed. "
          "commit db.json and the demo now searches them all.")


if __name__ == "__main__":
    main()
