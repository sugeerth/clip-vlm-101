"""All-pairs similarity — the geometry of the embedding space, made visible.

pipeline: stored embeddings ──► [similarity] ──► N×N matrix + the modality gap

Two lessons in one matrix:

1. STRUCTURE — animals score high with animals, landmarks with landmarks:
   the space has neighborhoods (the same thing the 2-D PCA map plots).
2. THE MODALITY GAP (Liang et al., 2022) — an image is far MORE similar to
   other images than to the text embedding of its OWN caption. On the
   sample gallery: image·own-caption ≈ 0.29, image·other-images ≈ 0.57.
   Cross-modal and within-modal scores live on different scales, so never
   mix the two kinds in one ranking.

Run me:  python3 similarity.py                     # uses gallery.sqlite
         python3 similarity.py --json docs/db.json # committed data, no model
"""
import argparse
import os
import pathlib

import numpy as np

import db

SHADES = " .:-=+*#%@"  # heatmap palette: darker character = more similar


def matrix(vecs) -> np.ndarray:
    """Every pairwise dot product at once: (n,d) @ (d,n) → (n,n)."""
    X = np.asarray(vecs)
    return X @ X.T


def modality_gap(items) -> dict:
    """Mean similarity within each tower and across them."""
    I = np.asarray([it["image_emb"] for it in items])
    T = np.asarray([it["text_emb"] for it in items])
    cross = I @ T.T
    off = ~np.eye(len(items), dtype=bool)  # everything but self-pairs
    return {
        "image · other images": float(matrix(I)[off].mean()),
        "text · other texts": float(matrix(T)[off].mean()),
        "image · OWN caption": float(np.diag(cross).mean()),
        "image · other captions": float(cross[off].mean()),
    }


def ascii_heatmap(M, names) -> str:
    lo, hi = float(M.min()), float(M.max())
    span = (hi - lo) or 1.0
    return "\n".join(
        f"{name:>12} |" + "".join(
            SHADES[int((v - lo) / span * (len(SHADES) - 1))] for v in row) + "|"
        for name, row in zip(names, M))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="read the web export instead of the sqlite db")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if args.json:
        items = db.load_json_gallery(args.json)
    else:
        if not os.path.exists(args.db):
            raise SystemExit(f"no database at {args.db} — "
                             "run ingest.py first, or try --json docs/db.json")
        items = db.all_images(db.connect(args.db))
        if not items:
            raise SystemExit(f"{args.db} is empty — run ingest.py first")

    # "images/004_cat.jpg" -> "cat"
    names = [pathlib.Path(it["path"]).stem.split("_", 1)[-1] for it in items]
    M = matrix([it["image_emb"] for it in items])
    print("image · image similarity, one dot product per cell "
          f"(shades '{SHADES}', {M.min():+.2f} → {M.max():+.2f}):\n")
    print(ascii_heatmap(M, names))

    print("\nthe modality gap — mean similarities:")
    for k, v in modality_gap(items).items():
        print(f"  {k:<24} {v:+.3f}  {'#' * round(v * 40)}")
    print("\nan image is ~2× more similar to OTHER IMAGES than to its own"
          "\ncaption — that gap is why search.py never mixes the two scales.")


if __name__ == "__main__":
    main()
