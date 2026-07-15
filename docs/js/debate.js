// Mirror of debate.py — multiple agents that TALK, not just vote.
//
// judge.js polls its judges independently and averages them. Real deliberation
// is agents ARGUING: each hears the others and updates — but only toward peers
// within EPS of its own view (bounded-confidence dynamics, Hegselmann–Krause
// 2002). That either CONVERGES the panel to consensus or splits it into FACTIONS
// that won't move each other — a contested case, dissenters named, not averaged
// away. (Multi-agent debate also just scores better — Du et al. 2023.) The
// dynamics are deterministic; constants match debate.py.
export const EPS = 0.30;
export const MAX_ROUNDS = 12;
export const TOL = 1e-4;
export const RELEVANT = 0.5;

// single-linkage clusters on the line: one faction ⇒ consensus.
export function factions(opinions, eps = EPS) {
  const order = [...opinions.keys()].sort((a, b) => opinions[a] - opinions[b]);
  const groups = [];
  let cur = [order[0]];
  for (const k of order.slice(1)) {
    if (opinions[k] - opinions[cur[cur.length - 1]] <= eps + 1e-12) cur.push(k);
    else { groups.push(cur); cur = [k]; }
  }
  groups.push(cur);
  return groups;
}

// one round: each agent → confidence-weighted mean of peers within eps (self incl).
export function step(opinions, weights, eps = EPS) {
  return opinions.map((xi, i) => {
    let sw = 0, sx = 0;
    opinions.forEach((xj, j) => {
      if (Math.abs(xj - xi) <= eps + 1e-12) { sw += weights[j]; sx += xj * weights[j]; }
    });
    if (sw <= 0) {                       // no positive weight nearby → unweighted mean of peers (matches debate.py)
      sw = 0; sx = 0;
      opinions.forEach(xj => { if (Math.abs(xj - xi) <= eps + 1e-12) { sw += 1; sx += xj; } });
    }
    return sx / sw;
  });
}

export function debate(opinions, weights = null, eps = EPS, maxRounds = MAX_ROUNDS, tol = TOL) {
  const w = weights || opinions.map(() => 1);
  let x = [...opinions];
  const traj = [[...x]];
  let rounds = 0;
  for (let r = 1; r <= maxRounds; r++) {
    rounds = r;                         // set first, so a full loop leaves rounds=maxRounds (matches Python's for-else)
    const nxt = step(x, w, eps);
    traj.push([...nxt]);
    const moved = Math.max(...nxt.map((v, i) => Math.abs(v - x[i])));
    x = nxt;
    if (moved < tol) break;
  }
  const facs = factions(x, eps);
  const consensus = facs.length === 1;
  const flips = opinions.map((s, i) => ((s >= RELEVANT) !== (x[i] >= RELEVANT) ? i : -1)).filter(i => i >= 0);
  const W = w.reduce((a, b) => a + b, 0);
  let score = null, verdict = 'abstain', reason = 'contested';
  if (consensus) {
    score = x.reduce((a, v, i) => a + v * w[i], 0) / W;
    verdict = score >= RELEVANT ? 'relevant' : 'not relevant';
    reason = 'consensus';
  }
  return { trajectory: traj, final: x, factions: facs, consensus, verdict, reason,
    score, rounds, flips, nFactions: facs.length };
}

// turn the council's judges into debating agents: opinion = gated score,
// credibility = confidence; abstained judges get no seat (can't argue a blank).
export function fromCouncil(votes) {
  const seated = votes.filter(v => v.score !== null && v.score !== undefined);
  return {
    names: seated.map(v => v.name),
    opinions: seated.map(v => v.score),
    weights: seated.map(v => v.confidence ?? 1.0),
  };
}
