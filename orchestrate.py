"""orchestrate.py — a supervisor that spends compute only where it's needed.

pipeline: (query, result) ──► [glance ▸ panel ▸ debate] ──► verdict / honest abstain

judge.py convenes the whole panel on every case; debate.py always runs the full
deliberation. That's like sending every question to the Supreme Court. Real
systems don't: they answer the easy ones cheaply and ESCALATE only the hard ones.
This is the 2025–2026 production default — an orchestrator-worker supervisor
(Anthropic's research system) fused with an LLM CASCADE (FrugalGPT, RouteLLM)
and confidence-gated escalation (CP-Router, UCCI). It adds no new judge; it
ROUTES the agents this repo already has, up a ladder, stopping the moment it's
sure.

  TIER 1  GLANCE   the single cheapest signal — one judge (relevance). Decisive
                   either way (≥ HI or ≤ LO) → rule now, done in one call.
  ── escalate if the glance lands in the uncertain middle ──
  TIER 2  PANEL    the full council (judge.py): three rubrics, parse-gated,
                   confidence-weighted, consensus measured. Quorum & not split
                   → rule. No quorum / hung jury → escalate (don't abstain yet).
  ── escalate if the panel can't rule ──
  TIER 3  DEBATE   the judges argue (debate.py, Hegselmann–Krause): they either
                   CONVERGE to a consensus verdict, or split into named FACTIONS
                   → the ladder's honest terminal state: ABSTAIN, factions shown.

The dial is the escalation gate, and it is DETERMINISTIC — routing depends only
on the agents' own gated signals, so difficulty D always lands on tier T (the
same reason CP-Router/UCCI calibrate their thresholds with conformal/isotonic
methods, not a coin flip). That makes the payoff measurable on the committed
gallery: easy cases resolve at tier 1 for 1 call instead of 3, hard cases climb,
contested cases abstain — a real cost-vs-accuracy-vs-abstention curve.

Model-free and deterministic like the agents it routes; js/orchestrate.js is a
byte-identical twin, and the live trace (trace.js) shows WHICH tier fired and WHY
it escalated. In the LLM path, tier 1 issues one judge call and only escalation
pays for the other two — llm_calls reports what a real deployment would spend.

    python3 orchestrate.py --json docs/db.json --image images/004_cat.jpg   # route the top-k
    python3 orchestrate.py --json docs/db.json --eval                        # the compute-saved curve
"""
import judge
import debate

# the glance's decisive band: outside [LO, HI] the cheap signal is trusted;
# inside it, the case is uncertain and escalates to the full panel.
GLANCE_HI = 0.75
GLANCE_LO = 0.25


def orchestrate(query_item, result_item):
    """Route one (query, result) up the ladder. Returns the verdict plus the
    full escalation record: which tier ruled, the path taken, the judge calls a
    real deployment would spend, and the tier-specific evidence."""
    votes = judge.heuristic_votes(query_item, result_item)
    by_name = {v["name"]: v for v in votes}
    path = []

    # ---- TIER 1 · GLANCE — the single cheapest judge ----
    glance = float(by_name["relevance"]["score"])
    path.append({"tier": 1, "name": "glance", "signal": glance})
    if glance >= GLANCE_HI:
        return _verdict("relevant", "glance", 1, 1, path, glance=glance)
    if glance <= GLANCE_LO:
        return _verdict("not relevant", "glance", 1, 1, path, glance=glance)

    # ---- TIER 2 · PANEL — the full council ----
    council = judge.aggregate(votes)
    path.append({"tier": 2, "name": "panel", "decision": council["decision"],
                 "reason": council["reason"], "consensus": council["consensus"]})
    if council["decision"] in ("relevant", "not relevant"):
        return _verdict(council["decision"], "panel", 2, 3, path, council=council)

    # ---- TIER 3 · DEBATE — only when the panel is deadlocked ----
    names, ops, ws = debate.from_council(votes)
    if len(ops) < 2:                       # escalated to debate but can't seat it
        path.append({"tier": 3, "name": "debate", "skipped": "too few seats"})
        return _verdict("abstain", "no quorum", 3, 3, path, council=council)
    d = debate.debate(ops, ws)
    camps = [[names[i] for i in g] for g in d["factions"]]
    path.append({"tier": 3, "name": "debate", "consensus": d["consensus"],
                 "rounds": d["rounds"], "factions": camps})
    if d["consensus"]:
        return _verdict(d["verdict"], "debate consensus", 3, 3, path,
                        council=council, debate=d, factions=camps)
    return _verdict("abstain", "contested", 3, 3, path,
                    council=council, debate=d, factions=camps)


def _verdict(decision, reason, tier, llm_calls, path, **evidence):
    return {"decision": decision, "reason": reason, "tier": tier,
            "llm_calls": llm_calls, "path": path, **evidence}


def route_stats(pairs):
    """Run the orchestrator over many (query, result) pairs and measure the
    adaptive-compute payoff: where cases resolved, calls spent vs the naive
    'always convene the panel', and how many honestly abstained."""
    tiers = {1: 0, 2: 0, 3: 0}
    abstains = spent = 0
    for q, r in pairs:
        out = orchestrate(q, r)
        tiers[out["tier"]] += 1
        spent += out["llm_calls"]
        abstains += out["decision"] == "abstain"
    naive = 3 * len(pairs)                 # the panel-on-everything baseline
    return {"n": len(pairs), "tiers": tiers, "spent": spent, "naive": naive,
            "saved": naive - spent, "abstains": abstains}


if __name__ == "__main__":
    import argparse

    import numpy as np

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", help="a gallery path to use as the query (model-free)")
    ap.add_argument("--k", type=int, default=5, help="route the top-k retrieved results")
    ap.add_argument("--eval", action="store_true",
                    help="the compute-saved curve across the whole gallery")
    args = ap.parse_args()

    items = (db.load_json_gallery(args.json) if args.json
             else db.all_images(db.connect(args.db)))

    def fused(it):
        v = np.asarray(it["image_emb"], dtype=np.float64)
        w = np.asarray(it["text_emb"], dtype=np.float64)
        return np.concatenate([v, w]) / np.sqrt(2.0)

    def top_k(q, k):
        qv = fused(q)
        return sorted((it for it in items if it is not q),
                      key=lambda it: float(fused(it) @ qv), reverse=True)[:k]

    if args.eval:
        pairs = [(q, top_k(q, 1)[0]) for q in items]
        s = route_stats(pairs)
        print(f"adaptive escalation over {s['n']} top-hit cases:\n")
        print(f"  resolved at TIER 1 glance : {s['tiers'][1]:>2}/{s['n']}")
        print(f"  resolved at TIER 2 panel  : {s['tiers'][2]:>2}/{s['n']}")
        print(f"  went to  TIER 3 debate    : {s['tiers'][3]:>2}/{s['n']}")
        print(f"  honestly abstained        : {s['abstains']:>2}/{s['n']}\n")
        pct = f"{s['saved'] / s['naive']:.0%}" if s['naive'] else "0%"
        print(f"  judge calls spent : {s['spent']}   vs panel-on-everything: {s['naive']}"
              f"   → saved {s['saved']} ({pct})")
        print("\neasy cases stop at a glance; only the genuinely uncertain pay for the")
        print("panel, only the deadlocked pay for a debate. Compute follows difficulty.")
        raise SystemExit(0)

    if not args.image:
        ap.error("give --image PATH (a query) or --eval")
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = match[0]

    print(f"orchestrating query {q['path'].split('/')[-1]}  (tags: {', '.join(q['tags'])})\n")
    for r in top_k(q, args.k):
        out = orchestrate(q, r)
        ladder = "  ▸  ".join(f"{p['name']}" for p in out["path"])
        name = r["path"].split("/")[-1]
        note = ""
        if out.get("factions") and len(out["factions"]) > 1:
            note = "  factions: " + " | ".join("{" + ", ".join(c) + "}" for c in out["factions"])
        print(f"  {name:<26} tier {out['tier']} · {out['llm_calls']} call(s)   "
              f"{out['decision']:<13} ({out['reason']})")
        print(f"      path: {ladder}{note}")
    print("\nthe supervisor escalates only on doubt and abstains only when the debate "
          "stays split —\ncheap reflex first, deliberation last, honesty at the end.")
