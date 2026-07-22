// Mirror of flow.py — structured orchestration: a typed DAG of agents.
//
// orchestrate.js routes ONE case up a linear ladder. Real systems are GRAPHS: a
// lead fans out to parallel sub-agents, results fan back in, and every handoff is
// a CONTRACT — a schema the runtime checks, so an agent that goes off-spec is
// QUARANTINED, not silently propagated. That's the structured-outputs +
// orchestrator-worker pattern (Anthropic's research system; LangGraph's typed
// state graph), as a deterministic model-free runtime. Byte-identical to flow.py.
//
//   NODE     { name, needs, run(inputs, ctx)->object, contract: [keys…] }
//   FLOW     topo-orders the nodes (deterministic), runs each once its needs are
//            met, validates the output against its contract; an off-contract node
//            (or one downstream of one) is skipped — the graph fails closed.
//   FAN-OUT  one node spawns sub-agents, drops the off-spec ones, returns survivors.
import { heuristicVotes, aggregate } from './judge.js';
import { debate, fromCouncil } from './debate.js';

export const RUBRICS = ['relevance', 'specificity', 'faithfulness'];

export class Flow {
  // nodes: [{ name, run, needs?: string[], contract?: string[] }]
  constructor(nodes) {
    this.nodes = nodes.map(n => ({ needs: [], contract: [], ...n }));
    this.by = {};
    for (const n of this.nodes) {
      if (this.by[n.name]) throw new Error('duplicate node names');
      this.by[n.name] = n;
    }
    for (const n of this.nodes)
      for (const d of n.needs)
        if (!this.by[d]) throw new Error(`node '${n.name}' needs unknown node '${d}'`);
  }

  // Kahn's algorithm, tie-broken by insertion order → schedule is a pure function
  // of the graph (not of object key order). Throws on a cycle.
  order() {
    const pos = Object.fromEntries(this.nodes.map((n, i) => [n.name, i]));
    const indeg = Object.fromEntries(this.nodes.map(n => [n.name, n.needs.length]));
    let ready = this.nodes.filter(n => indeg[n.name] === 0).map(n => n.name);
    const seq = [];
    while (ready.length) {
      ready.sort((a, b) => pos[a] - pos[b]);
      const name = ready.shift();
      seq.push(name);
      for (const m of this.nodes)
        if (m.needs.includes(name) && --indeg[m.name] === 0) ready.push(m.name);
    }
    if (seq.length !== this.nodes.length) throw new Error('cycle in flow');
    return seq;
  }

  // Execute. Returns { outputs, trace, quarantined }. A node whose output misses a
  // contract key — or whose upstream was quarantined — is skipped, and its own
  // dependents skip in turn. The graph fails closed.
  run(context = {}) {
    const outputs = {}, trace = [], dead = new Set();
    for (const name of this.order()) {
      const node = this.by[name];
      const blocked = node.needs.filter(d => dead.has(d) || !(d in outputs));
      if (blocked.length) {
        dead.add(name);
        trace.push({ node: name, status: 'skipped', blocked_by: blocked.sort() });
        continue;
      }
      const inputs = Object.fromEntries(node.needs.map(d => [d, outputs[d]]));
      const result = node.run(inputs, context);
      const missing = node.contract.filter(k => !(k in result));
      if (missing.length) {
        dead.add(name);
        trace.push({ node: name, status: 'off-contract', missing });
        continue;
      }
      outputs[name] = result;
      trace.push({ node: name, status: 'ok', keys: Object.keys(result).sort() });
    }
    return { outputs, trace, quarantined: [...dead].sort() };
  }
}

// A node that spawns sub-agents: workers is [[name, fn], …]. Each fn(inputs, ctx)
// runs in order, is validated against workerContract, and KEPT only if it passes —
// off-spec sub-agents are dropped. Returns { workers: [survivors…], dropped: […] }.
export function fanOut(name, workers, { needs = [], workerContract = [] } = {}) {
  const run = (inputs, ctx) => {
    const kept = [], dropped = [];
    for (const [wname, fn] of workers) {
      const out = fn(inputs, ctx);
      if (workerContract.every(k => k in out)) kept.push({ worker: wname, ...out });
      else dropped.push(wname);
    }
    return { workers: kept, dropped };
  };
  return { name, run, needs, contract: ['workers'] };
}

// --- the demo graph: this repo's agents as a structured orchestration DAG ---
//   panel (fan-out: 3 judge sub-agents) → council (fan-in) → decide (escalate)

const judgeWorker = rubric => (inputs, ctx) => {
  const v = heuristicVotes(ctx.query, ctx.result).find(x => x.name === rubric);
  return { name: v.name, score: v.score, confidence: v.confidence };
};

const councilNode = () => ({
  name: 'council', needs: ['panel'], contract: ['decision', 'reason'],
  run: (inputs) => {
    const votes = inputs.panel.workers.map(w => ({ name: w.name, score: w.score, confidence: w.confidence }));
    const agg = aggregate(votes);
    return { decision: agg.decision, reason: agg.reason, mean: agg.mean,
      consensus: agg.consensus, n_valid: agg.nValid };
  },
});

const decideNode = () => ({
  name: 'decide', needs: ['council', 'panel'], contract: ['decision', 'reason', 'via'],
  run: (inputs) => {
    const c = inputs.council;
    if (c.decision === 'relevant' || c.decision === 'not relevant')
      return { decision: c.decision, reason: c.reason, via: 'council' };
    const votes = inputs.panel.workers.map(w => ({ name: w.name, score: w.score, confidence: w.confidence }));
    const { names, opinions, weights } = fromCouncil(votes);
    if (opinions.length < 2) return { decision: 'abstain', reason: 'no quorum', via: 'debate' };
    const d = debate(opinions, weights);
    if (d.consensus) return { decision: d.verdict, reason: 'consensus', via: 'debate' };
    return { decision: 'abstain', reason: 'contested', via: 'debate',
      factions: d.factions.map(g => g.map(i => names[i])) };
  },
});

export function verdictFlow(extraWorkers = []) {
  const workers = [...RUBRICS.map(r => [r, judgeWorker(r)]), ...extraWorkers];
  const panel = fanOut('panel', workers, { workerContract: ['name', 'score', 'confidence'] });
  return new Flow([panel, councilNode(), decideNode()]);
}

export const runVerdict = (queryItem, resultItem, extraWorkers = []) =>
  verdictFlow(extraWorkers).run({ query: queryItem, result: resultItem });
