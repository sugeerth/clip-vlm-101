"""Does retrieval actually work? Measure it: precision@k and MRR.

pipeline: stored embeddings ──► [evaluate] ──► P@1 / P@3 / P@5 / MRR per mode

The protocol is leave-one-out: each image takes a turn as the QUERY (its
image embedding), the other rows are ranked, and a hit counts as RELEVANT
if it shares at least one meta tag with the query. Precision@k = how many
of the top k are relevant; MRR = 1/rank of the first relevant hit, averaged.

Two honest caveats, both lessons in themselves:
- the ground truth is circular-ish: the tags were produced by the same
  model being evaluated. Real evals need labels the model never made.
- 14 images is a toy benchmark; the POINT is the protocol, not the digits.

Run me:  python3 evaluate.py --json docs/db.json   # committed data, no model
"""
import argparse
import os

import numpy as np

import db
from search import score

KS = (1, 3, 5)


def evaluate(items, mode) -> dict:
    """Leave-one-out P@k and MRR for one scoring mode."""
    sums = {f"P@{k}": 0.0 for k in KS} | {"MRR": 0.0}
    for q in items:
        ranked = sorted((it for it in items if it is not q),
                        key=lambda it: score(it, q["image_emb"], mode),
                        reverse=True)
        rel = [bool(set(q["tags"]) & set(it["tags"])) for it in ranked]
        for k in KS:
            sums[f"P@{k}"] += sum(rel[:k]) / k
        sums["MRR"] += 1 / (rel.index(True) + 1) if True in rel else 0.0
    return {name: v / len(items) for name, v in sums.items()}


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

    print(f"leave-one-out over {len(items)} images "
          "(relevant = shares ≥1 tag with the query):\n")
    print(f"  {'mode':<7}" + "".join(f"{f'P@{k}':>8}" for k in KS) + f"{'MRR':>8}")
    for mode in ("image", "text", "fused"):
        m = evaluate(items, mode)
        print(f"  {mode:<7}" + "".join(f"{m[f'P@{k}']:>8.3f}" for k in KS)
              + f"{m['MRR']:>8.3f}")
    print("\nprecision falls as k grows — the classic shape: the easy"
          "\nneighbors come first, then the ranking runs out of them.")


if __name__ == "__main__":
    main()
