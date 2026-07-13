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
    CONVERGE  optional feedback passes (--passes): blend the query toward
              its own top hits and re-rank — but every pass must convince
              the EVALUATOR, which scores each ranking by fidelity to the
              ORIGINAL query. A pass that scores worse is rejected and the
              loop stops. Measured honestly: unguarded feedback DRIFTS
              (recall 0.348 → 0.333 on a 5,000-vector corpus); the guard
              never loses ground and settles in ~2 passes. Self-correcting
              means knowing when to stop.
    PUBLISH   only then answer, with the whole trace attached.

Usage (after ingest.py has filled the gallery):
    python3 hermes.py "a fluffy animal"
    python3 hermes.py "something delicious" --k 3 --passes 4
    python3 hermes.py --image images/004_cat.jpg --json docs/db.json
              (agentic image search — runs model-free on the web export)
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


def _rank(items, q, k, exclude=None):
    ranked = sorted(((it, score(it, q, "fused")) for it in items if it is not exclude),
                    key=lambda pair: pair[1], reverse=True)[:k]
    return ranked, margin([s for _, s in ranked])


def _rank_fused(items, q1024, k, exclude=None):
    """Rank by a query already IN fused space (1024-d unit vector)."""
    ranked = sorted(((it, float(it["fused_emb"] @ q1024))
                     for it in items if it is not exclude),
                    key=lambda pair: pair[1], reverse=True)[:k]
    return ranked, margin([s for _, s in ranked])


def evaluate(ranked, q0) -> float:
    """The evaluator's one number: how faithful a ranking is to the
    ORIGINAL query — mean similarity of the hits to q0, not to whatever
    the query has drifted into. Anchoring here is what stops drift."""
    return float(np.mean([it["fused_emb"] @ q0 for it, _ in ranked])) if ranked else 0.0


def refine(q0, items, k=5, passes=4, alpha=0.5, exclude=None):
    """Evaluator-guarded feedback loop in fused space: blend the query
    toward its top hits, re-rank, and keep the pass ONLY if the evaluator
    scores it higher. q0 is a 1024-d fused query (fusion.fused_query lifts
    a 512-d one). Returns (best ranking, ledger of passes)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q = q0.copy()
    best, m = _rank_fused(items, q, k, exclude)
    best_eval = evaluate(best, q0)
    ledger = [{"pass": 1, "eval": best_eval, "margin": m, "verdict": "initial"}]
    for p in range(2, passes + 1):
        fb = np.mean([np.asarray(it["fused_emb"], dtype=np.float64)
                      for it, _ in best[:3]], axis=0)
        q = (1 - alpha) * q + alpha * fb
        q /= np.linalg.norm(q)
        ranked, m = _rank_fused(items, q, k, exclude)
        ev = evaluate(ranked, q0)
        if [it["path"] for it, _ in ranked] == [it["path"] for it, _ in best]:
            ledger.append({"pass": p, "eval": ev, "margin": m, "verdict": "converged"})
            break
        if ev > best_eval:
            best, best_eval = ranked, ev
            ledger.append({"pass": p, "eval": ev, "margin": m, "verdict": "accepted"})
        else:
            ledger.append({"pass": p, "eval": ev, "margin": m, "verdict": "rejected — stopping"})
            break
    return best, ledger


def search_image(query_item, items, k: int = 5, passes: int = 4) -> dict:
    """Agentic image search: the query is a gallery image (or any record
    with image_emb). Model-free — every vector is already stored."""
    import fusion
    q0 = fusion.fused_query(np.asarray(query_item["image_emb"], dtype=np.float64))
    ranked, ledger = refine(q0, items, k, passes, exclude=query_item)
    return {"ranked": ranked, "ledger": ledger,
            "satisfied": ledger[-1]["verdict"] != "initial" or passes == 1}


def extend(items, paths, fx):
    """The crawl-then-search bridge: embed freshly crawled files and add
    them to the working set, so THIS query already searches them."""
    return items + fx.extract_batch([str(p) for p in paths]) if paths else items


def search(query, encode_texts, items, k: int = 5, passes: int = 1) -> dict:
    """The loop. encode_texts: list[str] -> (n, 512) unit vectors."""
    phrasings = [templates.fill(t, q=query) for t in QUERY_TEMPLATES]
    embs = encode_texts(phrasings)                 # ONE batch, four drafts
    rounds = []
    for phrasing, emb in zip(phrasings, embs):
        ranked, m = _rank(items, emb, k)
        rounds.append({"phrasing": phrasing, "ranked": ranked, "margin": m})
    best = max(rounds, key=lambda r: r["margin"])
    if best["margin"] >= MIN_MARGIN:
        out = {"ranked": best["ranked"], "satisfied": True,
               "chose": best["phrasing"], "rounds": rounds}
        q0 = embs[rounds.index(best)]
    else:
        # refine: no phrasing was decisive — ensemble all drafts
        mean = np.asarray(embs).mean(axis=0)
        q0 = mean / np.linalg.norm(mean)
        ranked, _ = _rank(items, q0, k)
        out = {"ranked": ranked, "satisfied": False,
               "chose": "an ensemble of all phrasings", "rounds": rounds}
    if passes > 1:  # evaluator-guarded convergence on top of the chosen draft
        import fusion
        out["ranked"], out["ledger"] = refine(
            fusion.fused_query(np.asarray(q0, dtype=np.float64)), items, k, passes)
    return out


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="what you are looking for, in words")
    ap.add_argument("--image", help="agentic image search: a gallery path to query with")
    ap.add_argument("--json", help="read the web export instead of the sqlite db")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--passes", type=int, default=4,
                    help="max evaluator-guarded feedback passes")
    ap.add_argument("--model", default=None,
                    help="a models.py registry key (clip-b32, siglip2-base, …)")
    ap.add_argument("--crawl", type=int, metavar="N",
                    help="every search crawls: fetch N fresh Commons images "
                         "matching the query and include them in this search")
    args = ap.parse_args()
    if bool(args.query) == bool(args.image):
        ap.error("give a text query OR --image")

    items = (db.load_json_gallery(args.json) if args.json
             else db.all_images(db.connect(args.db)))
    if not items:
        raise SystemExit("no images — run ingest.py first, or try --json docs/db.json")

    if args.image:
        match = [it for it in items if args.image in it["path"]]
        if not match:
            raise SystemExit(f"no gallery image matches {args.image!r}")
        out = search_image(match[0], items, args.k, args.passes)
        print(f"hermes image-query trace for {match[0]['path']}:")
        for entry in out["ledger"]:
            print(f"  pass {entry['pass']}: evaluator {entry['eval']:+.3f} "
                  f"margin {entry['margin']:+.3f}  — {entry['verdict']}")
        print()
        for item, s in out["ranked"]:
            print(f"  {s:+.3f}  {item['path']}   tags: {', '.join(item['tags'])}")
        raise SystemExit(0)

    from embedder import ClipEmbedder  # deferred: image queries never need it

    clip = ClipEmbedder(model_id=args.model) if args.model else ClipEmbedder()
    if args.crawl:
        import crawler
        from features import FeatureExtractor
        print(f"crawling the web for {args.query!r} (commons, n={args.crawl})…")
        items = extend(items, crawler.crawl(args.query, args.crawl),
                       FeatureExtractor(clip=clip))
    out = search(args.query, clip.embed_texts, items, args.k, passes=args.passes)

    print("hermes trace:")
    for r in out["rounds"]:
        mark = "✓" if r["margin"] >= MIN_MARGIN else "·"
        print(f"  {mark} {r['phrasing']!r:<42} margin {r['margin']:+.3f}")
    verdict = "decisive" if out["satisfied"] else "no phrasing decisive → ensembled"
    print(f"chose {out['chose']!r} ({verdict})")
    for entry in out.get("ledger", []):
        print(f"  pass {entry['pass']}: evaluator {entry['eval']:+.3f}  — {entry['verdict']}")
    print()
    for item, s in out["ranked"]:
        print(f"  {s:+.3f}  {item['path']}")
