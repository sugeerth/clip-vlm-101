"""Vector arithmetic: embeddings form a space you can do algebra in.

pipeline: stored embeddings ──► [arithmetic] ──► a combined query ──► top-k

Add and subtract gallery images IN EMBEDDING SPACE, renormalize, and rank
the gallery against the result — the word2vec king−man+woman idea, on CLIP
vectors. The one rule: the sum of unit vectors is NOT unit length, so
renormalize before comparing (combine() does).

Honesty note: CLIP arithmetic is blunter than word2vec's — treat results
as exploration, sometimes surprising, not magic.

Run me:  python3 arithmetic.py cat + dog --json docs/db.json
         python3 arithmetic.py --centroid animal --json docs/db.json
                 (--centroid averages every image carrying that tag)
"""
import argparse
import os

import numpy as np

import db


def combine(vectors, coeffs) -> np.ndarray:
    """Σ coeff·vector, renormalized back onto the unit sphere."""
    v = sum(c * np.asarray(x, dtype=np.float64) for c, x in zip(coeffs, vectors))
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise SystemExit("the combination cancelled itself out to (nearly) zero")
    return (v / n).astype(np.float32)


def find(items, word):
    hits = [it for it in items if word in it["path"]]
    if not hits:
        raise SystemExit(f"no gallery image matches {word!r}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("expr", nargs="*", help="e.g.  cat + dog - apple")
    ap.add_argument("--centroid", help="average every image carrying this tag")
    ap.add_argument("-k", type=int, default=5, help="how many results")
    ap.add_argument("--json", help="read the web export instead of the sqlite db")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if bool(args.expr) == bool(args.centroid):
        ap.error("give an expression like 'cat + dog - apple', or --centroid TAG")

    if args.json:
        items = db.load_json_gallery(args.json)
    else:
        if not os.path.exists(args.db):
            raise SystemExit(f"no database at {args.db} — "
                             "run ingest.py first, or try --json docs/db.json")
        items = db.all_images(db.connect(args.db))
        if not items:
            raise SystemExit(f"{args.db} is empty — run ingest.py first")

    if args.centroid:
        members = [it for it in items if args.centroid in it["tags"]]
        if not members:
            raise SystemExit(f"no image is tagged {args.centroid!r}")
        q = combine([it["image_emb"] for it in members], [1.0] * len(members))
        print(f"centroid of {len(members)} images tagged "
              f"{args.centroid!r}, renormalized:\n")
    else:
        names, coeffs = [args.expr[0]], [1.0]
        for op, name in zip(args.expr[1::2], args.expr[2::2]):
            if op not in "+-":
                ap.error(f"expected + or - between operands, got {op!r}")
            names.append(name)
            coeffs.append(1.0 if op == "+" else -1.0)
        q = combine([find(items, n)["image_emb"] for n in names], coeffs)
        print("query = " + " ".join(
            f"{'+' if c > 0 else '-'} {n}" for c, n in zip(coeffs, names)).lstrip("+ ")
            + ", renormalized:\n")

    ranked = sorted(items, key=lambda it: float(it["image_emb"] @ q), reverse=True)
    for item in ranked[: args.k]:
        s = float(item["image_emb"] @ q)
        print(f"  {s:+.3f} {'#' * max(1, round(s * 40)):<14} {item['path']}")


if __name__ == "__main__":
    main()
