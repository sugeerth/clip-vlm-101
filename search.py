"""Search the gallery with plain text (or another image).

pipeline: query ─► embedder ─► dot products against every db row ─► top-k

The query goes through the SAME encoder as the stored items, so retrieval
is one dot product per row — no index needed at this scale.

Modes:
  image  compare against what images LOOK like        (visual match)
  text   compare against what captions/tags MEAN      (semantic match)
  fused  average of both, via the concatenated vector (default)

Usage:
    python3 search.py "a fluffy animal"
    python3 search.py "famous landmark at night" --mode image -k 3
    python3 search.py --image my_upload.png            # image-to-image search
"""
import argparse
import os

import numpy as np

import db
import fusion
import temperature


def score(item: dict, query: np.ndarray, mode: str) -> float:
    """Similarity of one db row to a 512-d query vector, under one mode."""
    if mode == "image":
        return float(item["image_emb"] @ query)
    if mode == "text":
        return float(item["text_emb"] @ query)
    # fused: equals the mean of the image and text scores (see fusion.py)
    return float(item["fused_emb"] @ fusion.fused_query(query))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="what you are looking for, in words")
    ap.add_argument("--image", help="search with an image instead of words")
    ap.add_argument("--mode", choices=["image", "text", "fused"], default="fused")
    ap.add_argument("-k", type=int, default=5, help="how many results")
    ap.add_argument("--probs", action="store_true",
                    help="also show softmax probabilities (see temperature.py)")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if not args.query and not args.image:
        ap.error("give a text query or --image")
    if args.k < 1:
        ap.error("-k must be at least 1")
    if args.image and not os.path.exists(args.image):
        ap.error(f"query image not found: {args.image}")
    if not os.path.exists(args.db):  # check BEFORE the 600 MB model load
        raise SystemExit(f"no database at {args.db} — run ingest.py first")

    from embedder import ClipEmbedder  # deferred: score() above needs only numpy

    clip = ClipEmbedder()
    items = db.all_images(db.connect(args.db))
    if not items:
        raise SystemExit(f"{args.db} is empty — run ingest.py first")

    if args.image:  # image query: always compare image-to-image
        query, mode = clip.embed_images([args.image])[0], "image"
        print(f"query image: {args.image}  (mode: image)\n")
    else:
        query, mode = clip.embed_texts([args.query])[0], args.mode
        print(f"query: {args.query!r}  (mode: {mode})\n")

    ranked = sorted(items, key=lambda it: score(it, query, mode), reverse=True)
    # probabilities are softmax over ALL rows, not just the k shown
    probs = (temperature.softmax([score(it, query, mode) for it in ranked])
             if args.probs else None)
    for i, item in enumerate(ranked[: args.k]):
        s = score(item, query, mode)
        bar = "#" * max(1, round(s * 40))
        p = f"  p={probs[i]:.1%}" if probs is not None else ""
        print(f"  {s:+.3f} {bar:<14} {item['path']}{p}")
        if mode == "fused":  # decompose: which signal carried this hit?
            si, st = score(item, query, "image"), score(item, query, "text")
            print(f"          = (image {si:+.3f} + text {st:+.3f}) / 2")
        print(f"          tags: {', '.join(item['tags'])}")


if __name__ == "__main__":
    main()
