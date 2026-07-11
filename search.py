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

import numpy as np

import db
import fusion
from embedder import ClipEmbedder


def score(item: dict, query: np.ndarray, mode: str) -> float:
    """Similarity of one db row to a 512-d query vector, under one mode."""
    if mode == "image":
        return float(item["image_emb"] @ query)
    if mode == "text":
        return float(item["text_emb"] @ query)
    # fused: equals the mean of the image and text scores (see fusion.py)
    return float(item["fused_emb"] @ fusion.fused_query(query))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?", help="what you are looking for, in words")
    ap.add_argument("--image", help="search with an image instead of words")
    ap.add_argument("--mode", choices=["image", "text", "fused"], default="fused")
    ap.add_argument("-k", type=int, default=5, help="how many results")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if not args.query and not args.image:
        ap.error("give a text query or --image")

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
    for item in ranked[: args.k]:
        s = score(item, query, mode)
        bar = "#" * max(1, round(s * 40))
        print(f"  {s:+.3f} {bar:<14} {item['path']}")
        print(f"          tags: {', '.join(item['tags'])}")


if __name__ == "__main__":
    main()
