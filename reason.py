"""reason.py — the reasoning layer: trace every step, then map the consequence.

pipeline: [retrieve · rank · conformal · council · debate · trust] ──► [reason] ──► a decision, its why, and what follows

Every stage in this repo produces a signal and, crucially, knows how to abstain.
But a pile of signals is not a decision. This is the layer on top that REASONS
over them: it walks the whole pipeline in order, turns each stage's output into a
premise ("retrieval found it at similarity s"), carries the running conclusion
forward, and ends at a CONSEQUENCE — what to actually DO, why, and what follows
if you do it. Not another number: an explicit chain you can read start to finish,
and a consequence map that says SHOW / CAVEAT / WITHHOLD and what each costs.

Every step carries a status — ok · caution · stop — so the chain is legible at a
glance and drives the same "rather say less" decision the honesty stack is built
on. Model-free (it composes the stored signals), so the whole reasoning runs on
the committed gallery; js/reason.js mirrors it and the live page draws the chain.

    python3 reason.py --json docs/db.json --image images/004_cat.jpg   # trace it end to end
    python3 reason.py --json docs/db.json --image images/000_apple.jpg  # a contested one
"""
import numpy as np

import conformal
import debate as debate_mod
import judge
import trust

# image→image gate bands (match trust.py's CLI regime)
STRONG, MODERATE, WEAK = 0.80, 0.72, 0.66


# Display rounding, HALF-UP via floor(x*100+0.5). Python's :.0%/:.2f are
# round-half-to-even and JS toFixed/Math.round are half-up, so they diverge on
# exact halves; both languages compute this identical IEEE expression instead, so
# the twin's premise/conclusion strings match byte-for-byte at every α and value.
def _pct(v):
    return int(np.floor(v * 100 + 0.5))


def _f2(v):
    m = np.floor(abs(v) * 100 + 0.5) / 100.0
    return f"-{m:.2f}" if (v < 0 and m > 0) else f"{m:.2f}"


def _sf2(v):
    m = np.floor(abs(v) * 100 + 0.5) / 100.0
    return f"-{m:.2f}" if (v < 0 and m > 0) else f"+{m:.2f}"


def _fused(it):
    v = np.asarray(it["image_emb"], dtype=np.float64)
    w = np.asarray(it["text_emb"], dtype=np.float64)
    return np.concatenate([v, w]) / np.sqrt(2.0)


def consequence(tr, council, deb):
    """The decision map: given the composed trust (and WHY it landed there), what
    to actually do — show it, show it with a caveat, or withhold — plus the
    reason and the downstream effect. This is the point of the whole stack."""
    lvl = tr["level"]
    if lvl == "high":
        return {"action": "show it as the answer", "status": "ok",
                "because": "every lens agrees and the panel reached consensus",
                "effect": "the user sees a confident, defensible match"}
    if lvl == "abstain":
        if tr["reason"] == "split decision" or (deb and not deb["consensus"]):
            because = ("the panel deliberated and still split into factions"
                       if deb and not deb["consensus"]
                       else "the honesty lenses split — no majority to trust")
            return {"action": "withhold — genuinely contested", "status": "stop",
                    "because": because,
                    "effect": "ask the user or broaden the query rather than guess"}
        return {"action": "withhold — not enough signal", "status": "stop",
                "because": "too few lenses cleared their own bar to compose a verdict",
                "effect": "say 'no confident match' instead of showing a coin flip"}
    if lvl == "medium":
        why = ("the council couldn't confirm it" if council["decision"] == "abstain"
               else "it missed the calibrated confidence bar" if council["decision"] == "relevant"
               else "the judges leaned against it")
        return {"action": "show it with a caveat", "status": "caution",
                "because": why, "effect": "label it a loose match so the user calibrates trust"}
    return {"action": "show it, flagged as weak", "status": "caution",
            "because": "the lenses agree it's a poor match",
            "effect": "keep it, but make the low confidence explicit"}


def trace(q, items, alpha=0.2):
    """Walk the pipeline for query image `q` and its top retrieved result, and
    return the full reasoning object: the ordered steps, the trust verdict, and
    the consequence."""
    others = [it for it in items if it is not q]
    ranked = sorted(others, key=lambda it: float(_fused(q) @ _fused(it)), reverse=True)
    r = ranked[0]
    scores = [float(_fused(q) @ _fused(it)) for it in ranked]
    cos = scores[0]
    icos = float(np.asarray(r["image_emb"], dtype=np.float64)
                 @ np.asarray(q["image_emb"], dtype=np.float64))
    tau = 1.0 - conformal.calibrate(conformal.loo_scores(items), alpha)

    votes = judge.heuristic_votes(q, r)
    council = judge.aggregate(votes)
    names, ops, ws = debate_mod.from_council(votes)
    deb = debate_mod.debate(ops, ws) if len(ops) >= 2 else None

    # the four trust lenses (image regime), then compose
    margin = float(cos - np.mean(scores[1:])) if len(scores) > 1 else 0.0
    signals = [
        {"name": "gate", "trust": trust.gate_trust(cos, STRONG, MODERATE, WEAK), "weight": trust.WEIGHTS["gate"]},
        {"name": "conformal", "trust": trust.conformal_trust(icos, tau), "weight": trust.WEIGHTS["conformal"]},
        {"name": "council", "trust": trust.council_trust(council), "weight": trust.WEIGHTS["council"]},
        {"name": "margin", "trust": trust.margin_trust([cos] + scores[1:]), "weight": trust.WEIGHTS["margin"]},
    ]
    tr = trust.compose(signals)

    def st(cond_ok, cond_stop=False):
        return "stop" if cond_stop else ("ok" if cond_ok else "caution")

    tag = r["path"].split("/")[-1]
    rnd = lambda v: round(v, 3) if v is not None else None
    steps = [
        {"stage": "retrieve", "icon": "🔍",
         "premise": f"embed the query, score all {len(others)} candidates",
         "conclusion": f"top match is {tag} at similarity {_f2(cos)}",
         "signal": round(cos, 3), "status": st(cos >= MODERATE)},
        {"stage": "rank", "icon": "🥇",
         "premise": "how cleanly does #1 separate from the pack?",
         "conclusion": (f"leads by margin {_sf2(margin)}" if margin >= 0.03
                        else f"a near-tie (margin {_sf2(margin)})"),
         "signal": round(margin, 3), "status": st(margin >= 0.03)},
        {"stage": "conformal", "icon": "🎯",
         "premise": f"is it inside the {_pct(1-alpha)}% coverage set (cos ≥ {_f2(tau)})?",
         "conclusion": ("clears the calibrated bar" if icos >= tau - 1e-9
                        else "below the bar — conformal abstains"),
         "signal": round(icos, 3), "status": st(icos >= tau - 1e-9, icos < tau - 1e-9)},
        {"stage": "council", "icon": "⚖️",
         "premise": f"{council['n_valid']} rubric-judges score it",
         "conclusion": (f"{council['decision']}" +
                        (f" (consensus {_pct(council['consensus'])}%)" if council["consensus"] is not None else " — hung jury")),
         "signal": rnd(council["mean"]), "status": st(council["decision"] == "relevant",
                                                      council["decision"] == "abstain")},
    ]
    if deb is not None:
        camps = " vs ".join("{" + ", ".join(names[i] for i in g) + "}" for g in deb["factions"])
        steps.append({"stage": "debate", "icon": "🗣️",
                      "premise": "the judges argue, updating toward peers they can hear",
                      "conclusion": (f"consensus after {deb['rounds']} rounds → {deb['verdict']}"
                                     if deb["consensus"] else f"contested: {camps}"),
                      "signal": rnd(deb["score"]), "status": st(deb["consensus"], not deb["consensus"])})
    imp = tr["level"]
    steps.append({"stage": "trust", "icon": "🧮",
                  "premise": f"compose the {tr['n_valid']}/{tr['n_total']} lenses that voted",
                  "conclusion": (f"trust: {imp}" + (f" ({_f2(tr['score'])})" if tr["score"] is not None else "")),
                  "signal": rnd(tr["score"]), "status": st(imp == "high", imp == "abstain")})

    return {"result": r, "steps": steps, "trust": tr, "council": council,
            "debate": deb, "consequence": consequence(tr, council, deb)}


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", required=True, help="a gallery path to query with (model-free)")
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    items = (db.load_json_gallery(args.json) if args.json
             else db.all_images(db.connect(args.db)))
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = match[0]
    t = trace(q, items, args.alpha)

    mark = {"ok": "✓", "caution": "⚠", "stop": "✗"}
    print(f"reasoning about {q['path'].split('/')[-1]} → {t['result']['path'].split('/')[-1]}  "
          f"(tags: {', '.join(q['tags'])})\n")
    for i, s in enumerate(t["steps"]):
        pipe = "   │" if i < len(t["steps"]) - 1 else "   ╵"
        print(f"  {mark[s['status']]} {s['icon']} {s['stage']:<9} {s['premise']}")
        print(f"  {pipe}          └─ {s['conclusion']}")
    c = t["consequence"]
    print(f"\n  ⇒ CONSEQUENCE [{mark[c['status']]}]: {c['action']}")
    print(f"       because {c['because']}")
    print(f"       so:     {c['effect']}")
    print("\nnot another score — a chain you can read end to end, ending in what to "
          "actually do\nand what it costs. Every step can say 'stop'.")
