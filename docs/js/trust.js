// Mirror of trust.py — compose the honesty layers into ONE verdict, or abstain.
//
// gate, conformal, council and margin each already know how to say "not sure".
// Reading four panels to decide whether to believe a result is the user's job,
// so this does it: it composes the signals the SAME way judge.js composes its
// judges — a weighted agreement, with an ABSTAIN when they disagree (a split
// decision) or too few weigh in. A council of gates. Constants match trust.py.
//
//   gate       how STRONG the top match is (calibrated magnitude)
//   conformal  does it CLEAR the coverage bar τ? (else abstain)
//   council    do independent rubric-judges CONCUR? (else abstain)
//   margin     is #1 decisively AHEAD of the pack? (Hermes' separation)
export const QUORUM = 2;
export const SPLIT = 0.5;
export const HIGH = 0.66, MED = 0.40;
export const MIN_FOR_HIGH = 3;
export const WEIGHTS = { gate: 1.0, conformal: 1.0, council: 1.2, margin: 0.7 };

// The composed verdict from [{ name, trust: number|null, weight, note? }].
export function compose(signals) {
  const valid = signals.filter(s => s.trust !== null && s.trust !== undefined);
  const abstained = signals.filter(s => s.trust === null || s.trust === undefined).map(s => s.name);
  const perSignal = signals.map(s => ({ name: s.name, trust: s.trust ?? null,
    weight: s.weight ?? 1.0, note: s.note ?? '' }));
  const base = { perSignal, nValid: valid.length, nTotal: signals.length, abstained };
  if (valid.length < QUORUM)
    return { ...base, level: 'abstain', reason: 'not enough signals', score: null, consensus: null };
  const trusts = valid.map(s => s.trust);
  let weights = valid.map(s => Math.max(s.weight ?? 1.0, 0));
  if (weights.reduce((a, b) => a + b, 0) <= 0) weights = valid.map(() => 1);
  const W = weights.reduce((a, b) => a + b, 0);
  const score = trusts.reduce((a, t, i) => a + t * weights[i], 0) / W;
  const spread = Math.max(...trusts) - Math.min(...trusts);
  const consensus = Math.max(0, 1 - spread);
  if (spread > SPLIT)
    return { ...base, level: 'abstain', reason: 'split decision', score, consensus, spread };
  let level = score >= HIGH ? 'high' : score >= MED ? 'medium' : 'low';
  let reason = 'composed';
  if (level === 'high' && valid.length < MIN_FOR_HIGH) {   // broad-participation cap
    level = 'medium'; reason = 'capped: too few lenses voted for high';
  }
  return { ...base, level, reason, score, consensus, spread };
}

// ── the four lenses: a stage's raw output → a [0,1] trust, or null ──
export function gateTrust(cos, strong, moderate, weak) {
  if (cos >= strong) return 1.0;
  if (cos >= moderate) return 0.7;
  if (cos >= weak) return 0.4;
  return 0.1;
}

export function conformalTrust(cos, tau, eps = 1e-9) {
  // the set includes its boundary (cos ≥ τ), so a float-noise tie counts as
  // CLEARED — eps keeps the twins on the same side of an exact tie.
  if (tau === null || tau === undefined || !isFinite(tau) || cos < tau - eps) return null;
  const denom = Math.max(1 - tau, 1e-6);
  return Math.min(1, 0.5 + 0.5 * (cos - tau) / denom);
}

export function councilTrust(verdict) {
  // no verdict, no decision, or an explicit abstain → this lens abstains
  if (!verdict || !verdict.decision || verdict.decision === 'abstain') return null;
  return verdict.mean ?? 0.0;
}

export function marginTrust(scores, scale = 0.15) {
  if (scores.length < 2) return null;
  const margin = scores[0] - scores.slice(1).reduce((a, b) => a + b, 0) / (scores.length - 1);
  return Math.min(1, Math.max(0, margin / scale));
}
