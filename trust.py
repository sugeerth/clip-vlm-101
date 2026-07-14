"""trust.py — compose the honesty layers into ONE verdict, or abstain.

pipeline: [gate · conformal · council · margin] ──► [trust] ──► trust level, or abstain

Every stage in this repo already knows how to say "I'm not sure": the
hallucination gate redacts, conformal returns an empty set, the council hangs.
Reading four separate panels to decide whether to believe a result is the user's
job — so this does it, composing the signals the SAME way the council composes
its judges: a weighted agreement, with an ABSTAIN when they disagree (a split
decision) or too few weigh in. A council of gates.

Four DIFFERENT lenses on the top result, each a trust contribution in [0,1] — or
None, meaning that layer itself abstained and doesn't vote:

  gate       how STRONG the top match is (its calibrated similarity magnitude)
  conformal  does it CLEAR the distribution-free coverage bar τ? (else abstain)
  council    do independent rubric-judges CONCUR it's relevant? (else abstain)
  margin     is #1 decisively AHEAD of the pack? (Hermes' separation signal)

They read related data through unrelated questions, which is the point: when
strength, calibration, consensus and separation all agree, trust is high; when
they split — a strong cosine the council can't confirm, a winner with no
margin — the composer abstains instead of averaging over a contradiction, the
same "rather say less" honesty as every stage it aggregates.

    python3 trust.py --json docs/db.json --image images/004_cat.jpg
    python3 trust.py --json docs/db.json --image images/000_apple.jpg --k 4
"""
import numpy as np

QUORUM = 2           # fewer than this many voting signals → can't compose
SPLIT = 0.5          # signals spanning more than this → split decision → abstain
HIGH, MED = 0.66, 0.40   # composed-score cutoffs for the trust level
MIN_FOR_HIGH = 3     # "high" needs ≥ this many lenses voting — you can't claim
                     # high trust while half the evidence abstained

# how much each lens counts: the council (richest) a bit more, margin (a
# corroborator) a bit less. Equal-ish on purpose — no single stage decides.
WEIGHTS = {"gate": 1.0, "conformal": 1.0, "council": 1.2, "margin": 0.7}


def compose(signals):
    """The composed verdict from a list of signals:
    [{"name", "trust": float|None, "weight": float, "note"?}]. A None trust is
    an abstention. Returns the level, score, consensus and who abstained —
    abstaining (no quorum / split decision) rather than ruling over a
    contradiction."""
    valid = [s for s in signals if s.get("trust") is not None]
    abstained = [s["name"] for s in signals if s.get("trust") is None]
    per_signal = [{"name": s["name"], "trust": s.get("trust"),
                   "weight": s.get("weight", 1.0), "note": s.get("note", "")}
                  for s in signals]
    base = {"per_signal": per_signal, "n_valid": len(valid),
            "n_total": len(signals), "abstained": abstained}
    if len(valid) < QUORUM:
        return {**base, "level": "abstain", "reason": "not enough signals",
                "score": None, "consensus": None}
    trusts = np.array([s["trust"] for s in valid], dtype=np.float64)
    weights = np.array([max(s.get("weight", 1.0), 0.0) for s in valid], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(trusts)
    score = float(np.dot(trusts, weights) / weights.sum())
    spread = float(trusts.max() - trusts.min())
    consensus = max(0.0, 1.0 - spread)
    if spread > SPLIT:
        return {**base, "level": "abstain", "reason": "split decision",
                "score": score, "consensus": consensus, "spread": spread}
    level = "high" if score >= HIGH else "medium" if score >= MED else "low"
    # broad-participation cap: too many silent lenses can't add up to "high"
    reason = "composed"
    if level == "high" and len(valid) < MIN_FOR_HIGH:
        level, reason = "medium", "capped: too few lenses voted for high"
    return {**base, "level": level, "reason": reason,
            "score": score, "consensus": consensus, "spread": spread}


# ── the four lenses: each turns a stage's raw output into a [0,1] trust, or None ─

def gate_trust(cos, strong, moderate, weak):
    """Match strength, bucketed by the same thresholds explain.py's gate uses
    (passed in, since text→image and image→image live in different bands)."""
    if cos >= strong:
        return 1.0
    if cos >= moderate:
        return 0.7
    if cos >= weak:
        return 0.4
    return 0.1


def conformal_trust(cos, tau, eps=1e-9):
    """Inside the calibrated coverage set → a graded trust (just-cleared ≈ 0.5,
    well-above → 1). Below the bar → None: conformal abstains, so does this. The
    set includes its boundary (cos ≥ τ), so a float-noise tie counts as CLEARED
    — eps keeps the two twins on the same side of an exact tie."""
    if tau is None or not np.isfinite(tau) or cos < tau - eps:
        return None
    denom = max(1.0 - tau, 1e-6)
    return float(min(1.0, 0.5 + 0.5 * (cos - tau) / denom))


def council_trust(verdict):
    """The council's ruling → its weighted mean; a hung/quorum abstain → None."""
    # no verdict, no decision, or an explicit abstain → this lens abstains
    if not verdict or verdict.get("decision") in (None, "abstain"):
        return None
    return float(verdict.get("mean", 0.0))


def margin_trust(scores, scale=0.15):
    """Separation of #1 from the pack (Hermes' margin = top1 − mean(rest)),
    normalized. A decisive winner → ~1; a photo-finish → ~0."""
    if len(scores) < 2:
        return None
    margin = float(scores[0] - np.mean(scores[1:]))
    return float(min(1.0, max(0.0, margin / scale)))


if __name__ == "__main__":
    import argparse

    import db
    import judge

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", required=True, help="a gallery path to query with (model-free)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import conformal

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
                    key=lambda it: float(fused(it) @ qv), reverse=True)
    scores = [float(fused(it) @ qv) for it in ranked]
    # image→image calibration for the conformal lens (matches the CLI regime)
    tau = 1.0 - conformal.calibrate(conformal.loo_scores(items), 0.2)
    # gate thresholds in the image→image band (fused cosines ~0.5–0.9)
    STRONG, MODERATE, WEAK = 0.80, 0.72, 0.66

    print(f"composing a trust verdict for query {q['path']}  "
          f"(tags: {', '.join(q['tags'])})\n")
    print(f"  {'result':<24} {'gate':>5} {'confm':>6} {'counc':>6} {'marg':>5} "
          f"{'score':>6} {'consen':>7}  trust")
    for i, r in enumerate(ranked[:args.k]):
        cos = scores[i]
        icos = float(np.asarray(r["image_emb"], dtype=np.float64)
                     @ np.asarray(q["image_emb"], dtype=np.float64))
        rest = scores[:i] + scores[i + 1:]
        sig = [
            {"name": "gate", "trust": gate_trust(cos, STRONG, MODERATE, WEAK),
             "weight": WEIGHTS["gate"]},
            {"name": "conformal", "trust": conformal_trust(icos, tau),
             "weight": WEIGHTS["conformal"]},
            {"name": "council", "trust": council_trust(judge.council(q, r)),
             "weight": WEIGHTS["council"]},
            {"name": "margin", "trust": margin_trust([cos] + rest),
             "weight": WEIGHTS["margin"]},
        ]
        v = compose(sig)
        cell = lambda x: f"{x:.2f}" if x is not None else "  — "
        by = {s["name"]: s["trust"] for s in v["per_signal"]}
        score = f"{v['score']:.2f}" if v["score"] is not None else "  — "
        con = f"{v['consensus']:.2f}" if v["consensus"] is not None else "  — "
        name = r["path"].split("/")[-1]
        print(f"  {name:<24} {cell(by['gate']):>5} {cell(by['conformal']):>6} "
              f"{cell(by['council']):>6} {cell(by['margin']):>5} {score:>6} {con:>7}  "
              f"{v['level']} ({v['reason']})")
    print("\nfour lenses — strength, calibration, consensus, separation — composed "
          "like a\ncouncil of gates: agreement → a trust level; a split → an "
          "honest abstain.")
