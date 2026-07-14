"""judge.py — a COUNCIL of LLM judges: many verdicts, aggregated honestly.

pipeline: (query, result) ──► [N judges score] ──► [gate parses] ──► [council] ──► verdict / abstain

The read path already retrieves (search.py), re-ranks (dcn.py / learn2rank.py),
bounds itself (conformal.py) and explains-with-a-gate (explain.py). This is the
last honesty layer: instead of trusting one model's one score, convene a PANEL
of independent judges — each a different rubric — and aggregate their votes the
way a good council does: weight the confident ones, MEASURE how much they agree,
and REFUSE TO RULE when they don't.

  JUDGES     each judge scores a (query, result) pair in [0,1] under its own
             rubric — relevance, specificity, faithfulness — plus a confidence.
             Different rubrics are different questions, so a lone model's bias
             doesn't decide the case (this is the "panel of LLM evaluators"
             idea, Verga et al. 2024: several small judges beat one big one and
             cancel each other's quirks).
  GATE       each judge's RAW text is parsed to a number by parse_score(). A
             judge whose output has no parseable score ABSTAINS — it does not
             get to vote garbage. Same discipline as explain.py's gate: an
             unverifiable claim is dropped, not guessed at.
  COUNCIL    aggregate the valid votes into a confidence-weighted mean, and
             measure CONSENSUS = 1 − (max − min) across judges. Then, like
             conformal.py, the council ABSTAINS rather than pretend:
               • too few judges returned a score (< QUORUM)  → "no quorum"
               • the panel is too split (spread > HUNG_SPREAD) → "hung jury"
             A confident average over a coin flip is exactly the failure this
             prevents. Honest disagreement is a result, not a bug.

The LLM is optional. This file ships a model-free HEURISTIC judge so the whole
mechanism runs on the committed gallery with no downloads (like dcn.py's
untrained demo); js/judge.js swaps in a real in-browser LLM per rubric and runs
its output through the identical gate + council. The math is a twin, byte for
byte.

    python3 judge.py --json docs/db.json --image images/004_cat.jpg   # convene, model-free
    python3 judge.py --json docs/db.json --image images/000_apple.jpg --k 3
"""
import re

import numpy as np

QUORUM = 2            # fewer than this many valid votes → the council can't rule
ACCEPT = 0.5         # weighted-mean at/above this → "relevant"
HUNG_SPREAD = 0.5    # valid votes spanning more than this → hung jury, abstain

# CLIP's fused cosines live in a compressed band (matches ~0.7–0.9, unrelated
# ~0.65), so a raw-cosine relevance judge can't tell them apart. Map that band
# to [0,1] — the same "spread the scores before you score them" idea as
# temperature.py. Heuristic-only: the LLM judges emit a 0–1 score directly.
REL_LO, REL_HI = 0.65, 0.90

# Each judge is a rubric: a name, the question it asks the model, and the fixed
# confidence it carries into the weighted mean (relevance is the surest signal,
# faithfulness the most cautious). The prompts drive the LLM path (js/judge.js);
# the heuristic path below scores the same three rubrics without a model.
RUBRICS = (
    {"name": "relevance",
     "confidence": 0.9,
     "prompt": "How well does this image answer the query? Score 0 to 1."},
    {"name": "specificity",
     "confidence": 0.7,
     "prompt": "Is this a precise match, not just loosely related? Score 0 to 1."},
    {"name": "faithfulness",
     "confidence": 0.6,
     "prompt": "Do the image's own tags justify calling it a match? Score 0 to 1."},
)


def parse_score(text):
    """The gate: pull a score in [0,1] out of a judge's raw text, or None.

    Accepts the forms a small model actually emits — "0.7", ".7", "7/10",
    "8 out of 10", "70%", "score: 0.9" — and rejects anything out of range. No
    parseable score → None → that judge abstains (it does not vote a guess."""
    if text is None:
        return None
    t = str(text).lower()
    # re.ASCII pins \d to [0-9] so this matches the JS twin's ASCII \d exactly —
    # a judge that emits non-Latin digits abstains in both, not just in JS.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|\s+out\s+of\s+)\s*(\d+(?:\.\d+)?)", t, re.ASCII)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0 and 0 <= num <= den:
            return num / den
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t, re.ASCII)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0
    m = re.search(r"(?<![\d.])(?:0?\.\d+|[01](?:\.0+)?)(?![\d])", t, re.ASCII)
    if m:
        v = float(m.group(0))
        if 0 <= v <= 1:
            return v
    return None


def aggregate(votes):
    """The council's ruling from a list of votes.

    votes: [{"name", "score": float|None, "confidence": float, "rationale"?}].
    A None score is an abstention. Returns the decision, the confidence-weighted
    mean, the consensus, and who abstained — abstaining (no quorum / hung jury)
    rather than ruling on a split panel."""
    valid = [v for v in votes if v.get("score") is not None]
    abstained = [v["name"] for v in votes if v.get("score") is None]
    per_judge = [{"name": v["name"], "score": v.get("score"),
                  "confidence": v.get("confidence", 1.0),
                  "rationale": v.get("rationale", "")} for v in votes]
    base = {"per_judge": per_judge, "n_valid": len(valid), "n_total": len(votes),
            "abstained": abstained}
    if len(valid) < QUORUM:
        return {**base, "decision": "abstain", "reason": "no quorum",
                "mean": None, "consensus": None}
    scores = np.array([v["score"] for v in valid], dtype=np.float64)
    weights = np.array([max(v.get("confidence", 1.0), 0.0) for v in valid], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(scores)
    # unrounded: identical arithmetic in both twins → identical floats. Rounding
    # (which differs — Python banker's vs JS half-up) is a DISPLAY concern, done
    # by the CLI and the panel, never baked into the returned math.
    mean = float(np.dot(scores, weights) / weights.sum())
    spread = float(scores.max() - scores.min())
    consensus = max(0.0, 1.0 - spread)
    if spread > HUNG_SPREAD:
        return {**base, "decision": "abstain", "reason": "hung jury",
                "mean": mean, "consensus": consensus, "spread": spread}
    decision = "relevant" if mean >= ACCEPT else "not relevant"
    return {**base, "decision": decision, "reason": "ruled",
            "mean": mean, "consensus": consensus, "spread": spread}


def majority(votes, threshold=ACCEPT):
    """The simpler panel rule: each valid judge casts a yes/no vote (score ≥
    threshold), the council follows the majority, and a TIE abstains. This is
    'multiple LLMs as judges'; aggregate() is the full weighted council."""
    valid = [v for v in votes if v.get("score") is not None]
    if len(valid) < QUORUM:
        return {"decision": "abstain", "reason": "no quorum",
                "yes": 0, "no": 0, "n_valid": len(valid)}
    yes = sum(1 for v in valid if v["score"] >= threshold)
    no = len(valid) - yes
    if yes == no:
        return {"decision": "abstain", "reason": "tie", "yes": yes, "no": no,
                "n_valid": len(valid)}
    return {"decision": "relevant" if yes > no else "not relevant",
            "reason": "majority", "yes": yes, "no": no, "n_valid": len(valid)}


def heuristic_votes(query_item, result_item):
    """Model-free judges: score the same three rubrics from stored signals, so
    the council mechanism runs on the committed gallery with no LLM. Each rubric
    reads a DIFFERENT signal, which is exactly why they can disagree.

      relevance    fused cosine(query, result)          — how close, overall
      specificity  shared tags / query's tag count      — how precisely
      faithfulness 1 if it shares the query's TOP tag, 0.5 if any tag, else 0
    """
    def fused(it):
        v = np.asarray(it["image_emb"], dtype=np.float64)
        w = np.asarray(it["text_emb"], dtype=np.float64)
        return np.concatenate([v, w]) / np.sqrt(2.0)

    cos = float(fused(query_item) @ fused(result_item))
    qtags = list(query_item.get("tags", []))
    rtags = set(result_item.get("tags", []))
    shared = [t for t in qtags if t in rtags]
    relevance = min(max((cos - REL_LO) / (REL_HI - REL_LO), 0.0), 1.0)
    specificity = len(shared) / max(len(qtags), 1)
    if qtags and qtags[0] in rtags:
        faithfulness = 1.0
    elif shared:
        faithfulness = 0.5
    else:
        faithfulness = 0.0
    scores = {"relevance": relevance, "specificity": specificity,
              "faithfulness": faithfulness}
    return [{"name": r["name"], "score": scores[r["name"]],
             "confidence": r["confidence"],
             "rationale": f"{r['name']} signal = {scores[r['name']]:.2f}"}
            for r in RUBRICS]


def council(query_item, result_item):
    """Convene the model-free council on one (query, result) pair."""
    return aggregate(heuristic_votes(query_item, result_item))


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", required=True,
                    help="a gallery path to use as the query (model-free)")
    ap.add_argument("--k", type=int, default=5, help="judge the top-k retrieved results")
    args = ap.parse_args()

    items = (db.load_json_gallery(args.json) if args.json
             else db.all_images(db.connect(args.db)))
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = match[0]

    def fused(it):
        v = np.asarray(it["image_emb"], dtype=np.float64)
        w = np.asarray(it["text_emb"], dtype=np.float64)
        return np.concatenate([v, w]) / np.sqrt(2.0)

    qv = fused(q)
    ranked = sorted((it for it in items if it is not q),
                    key=lambda it: float(fused(it) @ qv), reverse=True)[:args.k]

    print(f"convening the council for query {q['path']}  "
          f"(tags: {', '.join(q['tags'])})\n")
    print(f"  {'result':<26} {'relev':>6} {'spec':>6} {'faith':>6} "
          f"{'mean':>6} {'consen':>7}  verdict")
    for r in ranked:
        v = council(q, r)
        by = {j["name"]: j["score"] for j in v["per_judge"]}
        mean = f"{v['mean']:.2f}" if v["mean"] is not None else "  — "
        con = f"{v['consensus']:.2f}" if v["consensus"] is not None else "  — "
        name = r["path"].split("/")[-1]
        print(f"  {name:<26} {by['relevance']:>6.2f} {by['specificity']:>6.2f} "
              f"{by['faithfulness']:>6.2f} {mean:>6} {con:>7}  "
              f"{v['decision']} ({v['reason']})")
    print("\nthe council rules only when a quorum agrees; a split panel abstains "
          "(hung jury) —\nthe same 'rather say less' honesty as the conformal and "
          "hallucination gates.")
