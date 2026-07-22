// Mirror of orchestrate.py — a supervisor that spends compute only where needed.
//
// judge.js convenes the whole panel on every case; debate.js always runs the full
// deliberation. Real systems escalate instead: answer the easy ones cheaply, pay
// for deliberation only on the hard ones. This is the 2025–2026 production default
// — an orchestrator-worker supervisor (Anthropic's research system) fused with an
// LLM CASCADE (FrugalGPT, RouteLLM) and confidence-gated escalation (CP-Router,
// UCCI). It adds no new judge; it ROUTES the agents this repo already has up a
// ladder, stopping the moment it's sure.
//
//   TIER 1  GLANCE  one judge (relevance). Decisive (≥ HI or ≤ LO) → rule, 1 call.
//   TIER 2  PANEL   the full council (judge.js). Quorum & not split → rule.
//   TIER 3  DEBATE  the judges argue (debate.js, Hegselmann–Krause): converge →
//                   consensus, or split into named FACTIONS → honest ABSTAIN.
//
// The escalation gate is DETERMINISTIC — routing depends only on the agents' own
// gated signals — so this is a byte-identical twin of orchestrate.py. In the LLM
// path, tier 1 issues one judge call and only escalation pays for the other two;
// llmCalls reports what a real deployment would spend.
import { heuristicVotes, aggregate } from './judge.js';
import { debate, fromCouncil } from './debate.js';

export const GLANCE_HI = 0.75;
export const GLANCE_LO = 0.25;

const verdict = (decision, reason, tier, llmCalls, path, evidence = {}) =>
  ({ decision, reason, tier, llmCalls, path, ...evidence });

// Route one (query, result) up the ladder. Returns the verdict plus the full
// escalation record: which tier ruled, the path, the judge calls a real
// deployment would spend, and the tier-specific evidence.
export function orchestrate(queryItem, resultItem) {
  const votes = heuristicVotes(queryItem, resultItem);
  const byName = Object.fromEntries(votes.map(v => [v.name, v]));
  const path = [];

  // ---- TIER 1 · GLANCE — the single cheapest judge ----
  const glance = byName.relevance.score;
  path.push({ tier: 1, name: 'glance', signal: glance });
  if (glance >= GLANCE_HI) return verdict('relevant', 'glance', 1, 1, path, { glance });
  if (glance <= GLANCE_LO) return verdict('not relevant', 'glance', 1, 1, path, { glance });

  // ---- TIER 2 · PANEL — the full council ----
  const council = aggregate(votes);
  path.push({ tier: 2, name: 'panel', decision: council.decision,
    reason: council.reason, consensus: council.consensus });
  if (council.decision === 'relevant' || council.decision === 'not relevant')
    return verdict(council.decision, 'panel', 2, 3, path, { council });

  // ---- TIER 3 · DEBATE — only when the panel is deadlocked ----
  const { names, opinions, weights } = fromCouncil(votes);
  if (opinions.length < 2) {                 // can't argue a blank — stay abstained
    path.push({ tier: 3, name: 'debate', skipped: 'too few seats' });
    return verdict('abstain', 'no quorum', 2, 3, path, { council });
  }
  const d = debate(opinions, weights);
  const camps = d.factions.map(g => g.map(i => names[i]));
  path.push({ tier: 3, name: 'debate', consensus: d.consensus, rounds: d.rounds, factions: camps });
  if (d.consensus)
    return verdict(d.verdict, 'debate consensus', 3, 3, path, { council, debate: d, factions: camps });
  return verdict('abstain', 'contested', 3, 3, path, { council, debate: d, factions: camps });
}

// Run the orchestrator over many (query, result) pairs and measure the
// adaptive-compute payoff — mirrors orchestrate.route_stats.
export function routeStats(pairs) {
  const tiers = { 1: 0, 2: 0, 3: 0 };
  let abstains = 0, spent = 0;
  for (const [q, r] of pairs) {
    const out = orchestrate(q, r);
    tiers[out.tier] += 1;
    spent += out.llmCalls;
    if (out.decision === 'abstain') abstains += 1;
  }
  const naive = 3 * pairs.length;
  return { n: pairs.length, tiers, spent, naive, saved: naive - spent, abstains };
}
