"""The user tower: from liked items to recommendations, in one matmul.

pipeline: liked items ──► [user_tower] ──► user_vec ──► every item ranked

item_tower.py finished the expensive half offline: one verified vector per
item. This file is the serving half. The USER tower has exactly one job —
produce ONE vector in the same space as the item embeddings — and the
simplest honest user tower needs no training at all:

    user_vec = unit( mean( item_emb of everything the user liked ) )

Mean-pooling a user's history is the same trick production systems use to
bootstrap before a learned tower exists — and it drops in transparently
later: replace user_vector() with a trained model, nothing else changes.

Serving is then two lines, with no image model anywhere:

    scores = item_matrix @ user_vec     # every item, one matrix multiply
    top-k  = argsort, minus what they already liked

Usage (after `python3 item_tower.py images/*.jpg`):
    python3 user_tower.py images/cat.jpg images/dog.jpg      # likes → recs
    python3 user_tower.py images/pizza.jpg --k 3
"""
import numpy as np

import item_tower


def user_vector(liked_embs) -> np.ndarray:
    """One unit vector for the user: the renormalized mean of their likes."""
    v = np.asarray(liked_embs, dtype=np.float32).mean(axis=0)
    return v / np.linalg.norm(v)


def recommend(paths, matrix, liked_paths, k: int = 5):
    """Rank every item against the user vector; skip what they already like."""
    liked = {p for p in liked_paths}
    idx = [i for i, p in enumerate(paths) if p in liked]
    if not idx:
        raise ValueError("none of the liked paths are in the item tower")
    scores = matrix @ user_vector(matrix[idx])
    order = np.argsort(scores)[::-1]
    return [(paths[i], float(scores[i])) for i in order
            if paths[i] not in liked][:k]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("likes", nargs="+", help="item images the user liked")
    ap.add_argument("--db", default=item_tower.ITEM_DB_PATH)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    paths, matrix = item_tower.item_matrix(item_tower.connect(args.db))
    if not paths:
        raise SystemExit(f"no items in {args.db} — run item_tower.py first")
    print(f"user vector = mean of {len(args.likes)} liked item embedding(s)")
    for path, score in recommend(paths, matrix, args.likes, args.k):
        print(f"  {score:+.3f}  {path}")
