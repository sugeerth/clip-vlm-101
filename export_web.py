"""Export the SQLite gallery to docs/db.json (+ copy images) for the web demo.

pipeline: gallery.sqlite ─► [export_web] ─► docs/db.json + docs/images/

The GitHub Pages demo is 100% static: it loads this JSON, computes the query
embedding in the browser with transformers.js, and ranks with dot products —
the exact same math as search.py.

This also squashes every 512-d image embedding down to 2-D with PCA so the
page can draw the "embedding map" (nearby dots = similar images). The PCA
mean + components ship in db.json, so the browser can project YOUR uploaded
image onto the same map with two dot products.
"""
import argparse
import json
import pathlib
import shutil

import numpy as np

import db

DOCS = pathlib.Path("docs")


def pca_2d(vectors):
    """Project (n, d) vectors to their 2 main directions of variation.

    Returns coords (n, 2), the mean (d,), and the components (2, d) —
    everything needed to project a NEW vector: (v - mean) @ components.T
    """
    X = np.asarray(vectors, dtype=np.float64)
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    components = vt[:2]
    if components.shape[0] < 2:  # a 1-image gallery has only 1 direction
        components = np.vstack([components, np.zeros((2 - components.shape[0], X.shape[1]))])
    return (X - mean) @ components.T, mean, components


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if not pathlib.Path(args.db).exists():
        raise SystemExit(f"no database at {args.db} — run ingest.py first")

    items = db.all_images(db.connect(args.db))
    if not items:
        raise SystemExit(f"{args.db} is empty — run ingest.py first")

    # 2-D map coordinates for every image embedding, normalized to 0..1.
    coords, mean, components = pca_2d([it["image_emb"] for it in items])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)

    (DOCS / "images").mkdir(parents=True, exist_ok=True)
    out = []
    for i, it in enumerate(items):
        src = pathlib.Path(it["path"])
        name = f"{i:03d}_{src.name}"  # index prefix: same-named files from different folders must not clobber
        shutil.copy(src, DOCS / "images" / name)
        xy = (coords[i] - lo) / span
        out.append({
            "file": f"images/{name}",
            "caption": it["caption"],
            "tags": it["tags"],
            "map": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
            # rounding keeps db.json small; retrieval quality is unaffected
            "image_emb": [round(float(x), 5) for x in it["image_emb"]],
            "text_emb": [round(float(x), 5) for x in it["text_emb"]],
        })

    payload = {
        "model": "clip-vit-base-patch32", "dim": 512, "items": out,
        "pca": {  # lets the browser project an uploaded image onto the map
            "mean": [round(float(x), 5) for x in mean],
            "components": [[round(float(x), 5) for x in c] for c in components],
            "lo": [float(lo[0]), float(lo[1])],
            "span": [float(span[0]), float(span[1])],
        },
    }
    (DOCS / "db.json").write_text(json.dumps(payload))
    print(f"wrote docs/db.json ({len(out)} items + 2-D map) and copied images to docs/images/")


if __name__ == "__main__":
    main()
