"""debate.py — multiple agents that TALK, not just vote.

pipeline: council stances ──► [debate: argue ⇄ update ⇄ converge] ──► consensus / CONTESTED

judge.py polls its judges INDEPENDENTLY and averages them. Real deliberation is
agents ARGUING: each hears the others and updates its position — but only toward
peers it finds credible. That is bounded-confidence opinion dynamics
(Hegselmann–Krause, 2002), and it does something a vote cannot: it either
CONVERGES the panel to a consensus, or it splits into FACTIONS that will not move
each other — a genuinely contested case, surfaced with the dissenters NAMED
instead of averaged away. (Letting evaluators exchange positions across rounds
also just scores better: multi-agent debate, Du et al. 2023.)

  ROUND     each agent i moves to the confidence-weighted mean of every agent
            within EPS of its current opinion (itself included). Talk only sways
            you if it is already close enough to hear.
  CONVERGE  repeat until nobody moves more than TOL, or MAX_ROUNDS.
  RULE      one faction  → CONSENSUS: verdict = the shared opinion (≥ ½ relevant).
            many factions → CONTESTED: they deliberated and still disagree →
            abstain, and report who is in which camp.

The dynamics are deterministic, so the whole debate runs model-free on the
committed gallery (like dcn.py's untrained demo); js/debate.js mirrors it, and a
live "let them debate" button turns the council's silent judges into a
conversation. Constants match this file.

    python3 debate.py --json docs/db.json --image images/000_apple.jpg   # watch them argue
    python3 debate.py --json docs/db.json --eval                          # debate as evaluation
"""
import numpy as np

EPS = 0.30           # confidence bound: you only update toward peers within this
MAX_ROUNDS = 12
TOL = 1e-4           # converged when no agent moves more than this in a round
RELEVANT = 0.5       # a consensus opinion at/above this rules "relevant"


def factions(opinions, eps=EPS):
    """Single-linkage clusters on the line: agents whose (sorted) opinions stay
    within eps of a neighbour share a faction. One faction = consensus."""
    order = sorted(range(len(opinions)), key=lambda i: opinions[i])
    groups, cur = [], [order[0]]
    for k in order[1:]:
        if opinions[k] - opinions[cur[-1]] <= eps + 1e-12:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


def step(opinions, weights, eps=EPS):
    """One round of deliberation: each agent → confidence-weighted mean of the
    peers within eps of it (itself always included)."""
    x = np.asarray(opinions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(len(x)):
        near = np.abs(x - x[i]) <= eps + 1e-12
        wn = w[near]
        wn = wn if wn.sum() > 0 else np.ones_like(wn)
        out[i] = float(np.dot(x[near], wn) / wn.sum())
    return out


def debate(opinions, weights=None, eps=EPS, max_rounds=MAX_ROUNDS, tol=TOL):
    """Run the panel to convergence. Returns the round-by-round trajectory, the
    final factions, and the verdict — consensus or contested (abstain)."""
    x = np.asarray(opinions, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    traj = [x.tolist()]
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        nxt = step(x, w, eps)
        traj.append(nxt.tolist())
        moved = float(np.max(np.abs(nxt - x)))
        x = nxt
        if moved < tol:
            break
    facs = factions(x, eps)
    consensus = len(facs) == 1
    # who crossed the relevant/not line between their opening and closing stance
    start, final = np.asarray(opinions, dtype=np.float64), x
    flips = [i for i in range(len(final))
             if (start[i] >= RELEVANT) != (final[i] >= RELEVANT)]
    if consensus:
        score = float(np.dot(final, w) / w.sum())
        verdict, reason = ("relevant" if score >= RELEVANT else "not relevant"), "consensus"
    else:
        score, verdict, reason = None, "abstain", "contested"
    return {"trajectory": traj, "final": final.tolist(), "factions": facs,
            "consensus": consensus, "verdict": verdict, "reason": reason,
            "score": score, "rounds": rounds, "flips": flips,
            "n_factions": len(facs)}


def from_council(votes):
    """Turn the council's judges into debating agents: opening opinion = each
    judge's gated score, credibility = its confidence. Judges that abstained
    (no score) don't get a seat — you can't argue a blank."""
    seated = [v for v in votes if v.get("score") is not None]
    names = [v["name"] for v in seated]
    opinions = [float(v["score"]) for v in seated]
    weights = [float(v.get("confidence", 1.0)) for v in seated]
    return names, opinions, weights


if __name__ == "__main__":
    import argparse

    import db
    import judge

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", help="a gallery path to query with (model-free)")
    ap.add_argument("--k", type=int, default=1, help="debate the top-k results")
    ap.add_argument("--eval", action="store_true",
                    help="debate as evaluation: consensus rate across the gallery")
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
        # multi-agent-debate-as-evaluator: run the council then the debate on
        # every gallery query's top hit, and measure the deliberation.
        consensus = contested = flipped = changed = 0
        total = 0
        for q in items:
            r = top_k(q, 1)[0]
            votes = judge.heuristic_votes(q, r)
            council = judge.aggregate(votes)
            names, ops, ws = from_council(votes)
            if len(ops) < 2:
                continue
            d = debate(ops, ws)
            total += 1
            consensus += d["consensus"]
            contested += not d["consensus"]
            flipped += bool(d["flips"])
            # did deliberation change the ruling vs the independent council?
            indep = council["decision"]
            if d["verdict"] != indep and not (d["verdict"] == "abstain" and indep == "abstain"):
                changed += 1
        print(f"debate as evaluation over {total} top-hit panels:\n")
        print(f"  reached consensus : {consensus:>2}/{total}  ({consensus/total:.0%})")
        print(f"  stayed contested  : {contested:>2}/{total}  ({contested/total:.0%})")
        print(f"  ≥1 agent flipped  : {flipped:>2}/{total}")
        print(f"  verdict changed vs independent council : {changed}/{total}")
        print("\ndeliberation converges the easy cases and isolates the contested "
              "ones —\nwhat an independent average hides, the debate names.")
        raise SystemExit(0)

    if not args.image:
        ap.error("give --image PATH (a query) or --eval")
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = match[0]

    for r in top_k(q, args.k):
        votes = judge.heuristic_votes(q, r)
        names, ops, ws = from_council(votes)
        print(f"debate on {q['path'].split('/')[-1]} → {r['path'].split('/')[-1]}  "
              f"(agents: {', '.join(names)})\n")
        if len(ops) < 2:
            print("  fewer than two seated agents — nothing to debate.")
            continue
        d = debate(ops, ws)
        for rnd, pos in enumerate(d["trajectory"]):
            bar = "  ".join(f"{names[i][:5]:>5} {pos[i]:.2f}" for i in range(len(pos)))
            print(f"  round {rnd}:  {bar}")
        camps = " | ".join("{" + ", ".join(names[i] for i in g) + "}" for g in d["factions"])
        print(f"\n  → {d['verdict']} ({d['reason']}) after {d['rounds']} rounds; "
              f"factions: {camps}")
        if d["flips"]:
            print(f"    agents who changed their mind: {', '.join(names[i] for i in d['flips'])}")
        print()
    print("agents update only toward peers within EPS — talk sways you only when "
          "it's\nclose enough to hear. Consensus, or a contested split named out "
          "loud.")
