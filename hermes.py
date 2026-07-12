"""Hermes: the agentic searcher — the query is a draft, not a command.

pipeline: query ──► [hermes: propose ⇄ critique ⇄ refine] ──► ranked results

agent.py audits the WRITE path (an image's features must satisfy a critic
before they are stored). Hermes is its twin on the READ path: your query
gets the same treatment before results are shown.

    PROPOSE   several phrasings of the query. The prompt is the classifier,
              so "cat", "a photo of cat" and "a close-up photo of cat" are
              genuinely different questions to the model.
    CRITIQUE  each phrasing by its retrieval MARGIN — how cleanly the top
              hit separates from the rest of the pack:
                  margin = top1_score - mean(scores of the other top-k)
              A decisive phrasing found something; an indecisive one is
              guessing.
    REFINE    if no phrasing is decisive, ensemble them: average the unit
              vectors and renormalize (ensemble.py's trick) — phrasing
              noise cancels, meaning stays.
    PUBLISH   only then answer, with the whole trace attached.

Usage (after ingest.py has filled the gallery):
    python3 hermes.py "a fluffy animal"
    python3 hermes.py "something delicious" --k 3
"""
import numpy as np

import templates
from search import score

QUERY_TEMPLATES = [
    "{q}",
    "a photo of {q}",
    "a close-up photo of {q}",
    "an image showing {q}",
]

# a decisive retrieval separates top-1 from the pack by at least this
MIN_MARGIN = 0.03


def margin(scores) -> float:
    """Top-1 score minus the mean of the rest — the critic's one number."""
    if len(scores) < 2:
        return float(scores[0]) if len(scores) else 0.0
    return float(scores[0] - np.mean(scores[1:]))


def _rank(items, q, k):
    ranked = sorted(((it, score(it, q, "fused")) for it in items),
                    key=lambda pair: pair[1], reverse=True)[:k]
    return ranked, margin([s for _, s in ranked])


def search(query, encode_texts, items, k: int = 5) -> dict:
    """The loop. encode_texts: list[str] -> (n, 512) unit vectors."""
    phrasings = [templates.fill(t, q=query) for t in QUERY_TEMPLATES]
    embs = encode_texts(phrasings)                 # ONE batch, four drafts
    rounds = []
    for phrasing, emb in zip(phrasings, embs):
        ranked, m = _rank(items, emb, k)
        rounds.append({"phrasing": phrasing, "ranked": ranked, "margin": m})
    best = max(rounds, key=lambda r: r["margin"])
    if best["margin"] >= MIN_MARGIN:
        return {"ranked": best["ranked"], "satisfied": True,
                "chose": best["phrasing"], "rounds": rounds}
    # refine: no phrasing was decisive — ensemble all drafts
    mean = np.asarray(embs).mean(axis=0)
    ranked, _ = _rank(items, mean / np.linalg.norm(mean), k)
    return {"ranked": ranked, "satisfied": False,
            "chose": "an ensemble of all phrasings", "rounds": rounds}


if __name__ == "__main__":
    import argparse

    import db
    from embedder import ClipEmbedder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", help="what you are looking for")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    items = db.all_images(db.connect(args.db))
    if not items:
        raise SystemExit(f"no images in {args.db} — run ingest.py first")
    clip = ClipEmbedder()
    out = search(args.query, clip.embed_texts, items, args.k)

    print("hermes trace:")
    for r in out["rounds"]:
        mark = "✓" if r["margin"] >= MIN_MARGIN else "·"
        print(f"  {mark} {r['phrasing']!r:<42} margin {r['margin']:+.3f}")
    verdict = "decisive" if out["satisfied"] else "no phrasing decisive → ensembled"
    print(f"chose {out['chose']!r} ({verdict})\n")
    for item, s in out["ranked"]:
        print(f"  {s:+.3f}  {item['path']}")
