"""flow.py — structured orchestration: a typed DAG of agents, contracts enforced.

pipeline: nodes + edges ──► [topo-run, validate every handoff] ──► outputs + audit trace

orchestrate.py routes ONE case up a linear ladder. Real systems are GRAPHS: a
lead fans out to parallel sub-agents, their results fan back in, and every
handoff is a CONTRACT — a schema the runtime checks, so an agent that goes
off-spec is QUARANTINED, not silently propagated downstream. That's the
structured-outputs + orchestrator-worker pattern (Anthropic's multi-agent
research system; LangGraph's typed state graph), shrunk to a deterministic,
model-free runtime you can read end to end.

  NODE      a named agent: the upstream outputs it NEEDS, a run(inputs, ctx)
            that returns a dict, and a CONTRACT — the keys its output MUST have.
  FLOW      a DAG of nodes. run() topologically orders them (deterministically),
            executes each once its needs are satisfied, and VALIDATES every
            output against its contract. An off-contract node is quarantined and
            every node downstream of it is skipped — the graph fails closed, the
            way a typed pipeline does, instead of passing bad data along.
  FAN-OUT   one node spawns a LIST of sub-agents (workers) and validates each
            against a per-worker contract, dropping the ones that go off-spec and
            returning the survivors. Orchestrator-worker, made structural.
  TRACE     run() returns a structured record of every hop — order, status
            (ok / off-contract / skipped), and the keys produced — automation you
            can audit. The same discipline as judge.py's parse-gate, applied to
            the WHOLE graph instead of one judge.

Deterministic and model-free like the agents it wires; js/flow.js is a
byte-identical twin, and the trace is the same object in both. Compose your own
agents by handing Flow a list of Nodes; the demo wires this repo's judges (as
parallel sub-agents) → council (fan-in) → escalate, contracts enforced at each
edge — and shows a rogue sub-agent getting dropped while the graph still rules.

    python3 flow.py --json docs/db.json --image images/004_cat.jpg   # run the graph, print the trace
    python3 flow.py --json docs/db.json --demo-contract               # watch a rogue agent get quarantined
"""
import judge
import debate

RUBRICS = ("relevance", "specificity", "faithfulness")


class Node:
    """One agent in the graph: a name, the upstream node outputs it consumes
    (`needs`), a `run(inputs, ctx) -> dict`, and a `contract` — the output keys
    the runtime will insist on before any downstream node may read it."""

    def __init__(self, name, run, needs=(), contract=()):
        self.name = name
        self.run = run
        self.needs = tuple(needs)
        self.contract = tuple(contract)


class Flow:
    """A DAG of Nodes. Deterministic topological execution with a contract check
    at every handoff — the structured-orchestration runtime in ~40 lines."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self._by = {n.name: n for n in self.nodes}
        if len(self._by) != len(self.nodes):
            raise ValueError("duplicate node names")
        for n in self.nodes:
            for d in n.needs:
                if d not in self._by:
                    raise ValueError(f"node {n.name!r} needs unknown node {d!r}")

    def order(self):
        """Kahn's algorithm, tie-broken by insertion order so the schedule is a
        pure function of the graph (not of dict/set iteration). Raises on a cycle."""
        pos = {n.name: i for i, n in enumerate(self.nodes)}
        indeg = {n.name: len(n.needs) for n in self.nodes}
        ready = [n.name for n in self.nodes if indeg[n.name] == 0]
        seq = []
        while ready:
            ready.sort(key=lambda x: pos[x])            # deterministic pick
            name = ready.pop(0)
            seq.append(name)
            for m in self.nodes:                        # release dependents
                if name in m.needs:
                    indeg[m.name] -= 1
                    if indeg[m.name] == 0:
                        ready.append(m.name)
        if len(seq) != len(self.nodes):
            raise ValueError("cycle in flow")
        return seq

    def run(self, context=None):
        """Execute the graph. Returns {outputs, trace, quarantined}. A node whose
        output misses a contract key — or whose upstream was quarantined — is
        skipped, and its own dependents skip in turn. The graph fails closed."""
        ctx = context or {}
        outputs, trace, dead = {}, [], set()
        for name in self.order():
            node = self._by[name]
            blocked = [d for d in node.needs if d in dead or d not in outputs]
            if blocked:
                dead.add(name)
                trace.append({"node": name, "status": "skipped", "blocked_by": sorted(blocked)})
                continue
            inputs = {d: outputs[d] for d in node.needs}
            result = node.run(inputs, ctx)
            missing = [k for k in node.contract if k not in result]
            if missing:
                dead.add(name)
                trace.append({"node": name, "status": "off-contract", "missing": missing})
                continue
            outputs[name] = result
            trace.append({"node": name, "status": "ok", "keys": sorted(result.keys())})
        return {"outputs": outputs, "trace": trace, "quarantined": sorted(dead)}


def fan_out(name, workers, needs=(), worker_contract=()):
    """A node that spawns sub-agents: `workers` is a list of (worker_name, fn).
    Each fn(inputs, ctx) is run in order, validated against `worker_contract`, and
    KEPT only if it satisfies it — off-spec sub-agents are dropped, not trusted.
    Returns {'workers': [survivors…], 'dropped': [names…]}. Orchestrator-worker
    with structured validation at the sub-agent level."""
    def run(inputs, ctx):
        kept, dropped = [], []
        for wname, fn in workers:
            out = fn(inputs, ctx)
            if all(k in out for k in worker_contract):
                kept.append({"worker": wname, **out})
            else:
                dropped.append(wname)
        return {"workers": kept, "dropped": dropped}
    return Node(name, run, needs=needs, contract=("workers",))


# ---------------------------------------------------------------------------
# the demo graph: this repo's agents wired as a structured orchestration DAG
#
#   panel (fan-out: 3 judge sub-agents) ──► council (fan-in) ──► decide (escalate)
#
# every edge carries a contract; a sub-agent or a node that breaks it is
# quarantined and the graph degrades honestly instead of trusting bad data.
# ---------------------------------------------------------------------------

def _judge_worker(rubric):
    """One rubric judge as a sub-agent: reads the (query, result) from context,
    returns its structured vote. Mirrors judge.heuristic_votes, one rubric."""
    def fn(inputs, ctx):
        votes = judge.heuristic_votes(ctx["query"], ctx["result"])
        v = next(x for x in votes if x["name"] == rubric)
        return {"name": v["name"], "score": v["score"], "confidence": v["confidence"]}
    return fn


def _council_node():
    def run(inputs, ctx):
        votes = [{"name": w["name"], "score": w["score"], "confidence": w["confidence"]}
                 for w in inputs["panel"]["workers"]]
        agg = judge.aggregate(votes)
        return {"decision": agg["decision"], "reason": agg["reason"],
                "mean": agg["mean"], "consensus": agg["consensus"], "n_valid": agg["n_valid"]}
    return Node("council", run, needs=("panel",), contract=("decision", "reason"))


def _decide_node():
    """Terminal agent: pass the council's ruling through, or — if the panel hung —
    escalate to a debate over the surviving sub-agents (orchestrate.py's ladder,
    here as one node of a graph). Always emits a contracted final decision."""
    def run(inputs, ctx):
        c = inputs["council"]
        if c["decision"] in ("relevant", "not relevant"):
            return {"decision": c["decision"], "reason": c["reason"], "via": "council"}
        votes = [{"name": w["name"], "score": w["score"], "confidence": w["confidence"]}
                 for w in inputs["panel"]["workers"]]
        names, ops, ws = debate.from_council(votes)
        if len(ops) < 2:
            return {"decision": "abstain", "reason": "no quorum", "via": "debate"}
        d = debate.debate(ops, ws)
        if d["consensus"]:
            return {"decision": d["verdict"], "reason": "consensus", "via": "debate"}
        return {"decision": "abstain", "reason": "contested", "via": "debate",
                "factions": [[names[i] for i in g] for g in d["factions"]]}
    return Node("decide", run, needs=("council", "panel"), contract=("decision", "reason", "via"))


def verdict_flow(extra_workers=()):
    """Build the demo graph. `extra_workers` lets you inject rogue sub-agents to
    watch the contract gate quarantine them (used by --demo-contract)."""
    workers = [(r, _judge_worker(r)) for r in RUBRICS] + list(extra_workers)
    panel = fan_out("panel", workers, worker_contract=("name", "score", "confidence"))
    return Flow([panel, _council_node(), _decide_node()])


def run_verdict(query_item, result_item, extra_workers=()):
    """Convenience: build + run the demo graph on one (query, result) pair."""
    flow = verdict_flow(extra_workers)
    return flow.run({"query": query_item, "result": result_item})


if __name__ == "__main__":
    import argparse

    import numpy as np

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--image", help="a gallery path to use as the query (model-free)")
    ap.add_argument("--k", type=int, default=3, help="run the graph on the top-k results")
    ap.add_argument("--demo-contract", action="store_true",
                    help="inject a rogue sub-agent and watch the runtime quarantine it")
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

    def show(res):
        for step in res["trace"]:
            mark = {"ok": "✓", "off-contract": "✗", "skipped": "·"}[step["status"]]
            extra = ""
            if step["status"] == "off-contract":
                extra = f"  missing {step['missing']}"
            elif step["status"] == "skipped":
                extra = f"  (blocked by {step['blocked_by']})"
            print(f"    {mark} {step['node']:<9} {step['status']}{extra}")

    if args.demo_contract:
        q = items[0]
        r = top_k(q, 1)[0]
        # a rogue sub-agent that returns the WRONG shape (no score/confidence)
        rogue = ("rogue", lambda inputs, ctx: {"name": "rogue", "opinion": "trust me"})
        res = run_verdict(q, r, extra_workers=[rogue])
        panel = res["outputs"].get("panel", {})
        print(f"structured contract enforcement — query {q['path'].split('/')[-1]}:\n")
        show(res)
        print(f"\n  the panel spawned 4 sub-agents; the rogue emitted no `score` and was "
              f"DROPPED\n  (dropped: {panel.get('dropped')}). The council still ruled from the "
              f"{len(panel.get('workers', []))} valid votes:\n  → {res['outputs'].get('decide')}")
        print("\n  a bad agent can't poison the graph: off-contract output is quarantined, "
              "not\n  propagated. The runtime fails closed, and the honest verdict survives.")
        raise SystemExit(0)

    if not args.image:
        ap.error("give --image PATH (a query) or --demo-contract")
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = match[0]

    print(f"structured orchestration graph for query {q['path'].split('/')[-1]}  "
          f"(tags: {', '.join(q['tags'])})\n")
    for r in top_k(q, args.k):
        res = run_verdict(q, r)
        out = res["outputs"].get("decide", {})
        print(f"  {r['path'].split('/')[-1]}:")
        show(res)
        note = ""
        if out.get("factions") and len(out["factions"]) > 1:
            note = "  factions: " + " | ".join("{" + ", ".join(c) + "}" for c in out["factions"])
        print(f"    ⇒ {out.get('decision')} ({out.get('reason')}, via {out.get('via')}){note}\n")
    print("panel (3 judge sub-agents) → council (fan-in) → decide (escalate) — every "
          "edge\ncontract-checked. Multiple sub-agents, multiple agents, one auditable graph.")
