"""validate_db.py — the gate a grown gallery must pass before it can ship.

pipeline: docs/db.json ──► [schema · norms · dups · map · receipts] ──► pass / FAIL

grow.py can add hundreds of crawled images to docs/db.json, and an automated
pipeline (.github/workflows/grow.yml) does it on a schedule with no human in the
loop. Automation without a gate is how a corrupt export ships itself — so this is
the gate. Model-free (numpy only), it checks everything about the database that
can be checked without re-running CLIP:

  SCHEMA    every item has file/caption/tags/map/image_emb/text_emb, right types,
            right dimensions (512), and the pca block is well-formed.
  NORMS     embeddings are unit vectors (within rounding: they ship at 5dp), so a
            truncated or double-scaled export is caught, not searched.
  DUPS      no repeated file, and no near-duplicate image pair above the same
            cosine cutoff grow.py dedups at — sampled at scale so it stays fast.
  MAP       every 2-D map coordinate is in [0,1]: the scatter never draws offscreen.
  RECEIPTS  every REMOTE image (http…) carries attribution — a crawler without
            receipts is a liability, and this makes receipts load-bearing.

Exit 0 with a summary, or exit 1 naming exactly what broke. CI runs it on every
push; grow.yml runs it before it is allowed to open a PR.

    python3 validate_db.py                 # gate docs/db.json
    python3 validate_db.py --json path     # gate any export
"""
import argparse
import json
import sys

import numpy as np

DIM = 512
NORM_TOL = 0.02          # embeddings ship rounded to 5dp; unit norm within 2%
DUP_COS = 0.985          # grow.py's near-duplicate cutoff — the DB must honor it
DUP_SAMPLE = 400         # O(n²) cap: check all pairs up to this many items, sample beyond
MIN_ITEMS = 14           # never regress below the seed gallery


def _fail(problems):
    print("db validation FAILED:")
    for p in problems:
        print(f"  ✗ {p}")
    sys.exit(1)


def validate(db, verbose=True):
    """Return a list of problems (empty = valid)."""
    problems = []
    items = db.get("items")
    if not isinstance(items, list) or len(items) < MIN_ITEMS:
        return [f"items missing or fewer than the {MIN_ITEMS}-image seed gallery"]

    # ---- schema + norms + map, item by item ----
    seen_files = set()
    for i, it in enumerate(items):
        where = f"item {i} ({it.get('file', '?')})"
        for key, typ in (("file", str), ("caption", str), ("tags", list), ("map", list)):
            if not isinstance(it.get(key), typ):
                problems.append(f"{where}: bad or missing {key!r}")
        for key in ("image_emb", "text_emb"):
            v = it.get(key)
            if not isinstance(v, list) or len(v) != DIM:
                problems.append(f"{where}: {key} is not a {DIM}-d vector")
                continue
            norm = float(np.linalg.norm(np.asarray(v, dtype=np.float64)))
            if abs(norm - 1.0) > NORM_TOL:
                problems.append(f"{where}: {key} norm {norm:.3f} is not ~1 — truncated or double-scaled?")
        m = it.get("map")
        if isinstance(m, list) and (len(m) != 2 or not all(
                isinstance(x, (int, float)) and -1e-9 <= x <= 1 + 1e-9 for x in m)):
            problems.append(f"{where}: map coordinate outside [0,1]")
        f = it.get("file")
        if isinstance(f, str):
            if f in seen_files:
                problems.append(f"{where}: duplicate file entry")
            seen_files.add(f)
            if f.startswith("http") and not (it.get("attr") or {}).get("source"):
                problems.append(f"{where}: remote image without attribution receipts")
        if isinstance(it.get("tags"), list) and not it["tags"]:
            problems.append(f"{where}: empty tags")

    # ---- pca block ----
    pca = db.get("pca") or {}
    if len(pca.get("mean", [])) != DIM:
        problems.append("pca.mean is not 512-d")
    comps = pca.get("components", [])
    if len(comps) != 2 or any(len(c) != DIM for c in comps):
        problems.append("pca.components is not 2×512")

    # ---- near-duplicates (the same cutoff grow.py enforces) ----
    ok = [it for it in items if isinstance(it.get("image_emb"), list) and len(it["image_emb"]) == DIM]
    if len(ok) >= 2:
        rng = np.random.default_rng(0)
        idx = (np.arange(len(ok)) if len(ok) <= DUP_SAMPLE
               else rng.choice(len(ok), DUP_SAMPLE, replace=False))
        M = np.asarray([ok[j]["image_emb"] for j in idx], dtype=np.float64)
        cos = M @ M.T
        np.fill_diagonal(cos, 0.0)
        worst = float(cos.max())
        if worst >= DUP_COS:
            a, b = np.unravel_index(int(cos.argmax()), cos.shape)
            problems.append(f"near-duplicate pair at cosine {worst:.3f} "
                            f"({ok[int(idx[a])]['file']} ~ {ok[int(idx[b])]['file']}) — dedup didn't run?")

    if verbose and not problems:
        remote = sum(1 for it in items if str(it.get("file", "")).startswith("http"))
        print(f"db valid ✓  {len(items)} items ({remote} remote, {len(items) - remote} local), "
              f"all unit-norm {DIM}-d, no dups above {DUP_COS}, receipts present")
    return problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    args = ap.parse_args()
    db = json.loads(open(args.json).read())
    problems = validate(db)
    if problems:
        _fail(problems)


if __name__ == "__main__":
    main()
