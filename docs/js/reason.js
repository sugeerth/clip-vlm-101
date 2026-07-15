// Mirror of reason.py — the reasoning layer: trace every step, map the consequence.
//
// Every stage produces a signal and knows how to abstain, but a pile of signals
// isn't a decision. This walks the whole pipeline in order, turns each stage's
// output into a premise → conclusion with a status (ok · caution · stop), and
// ends at a CONSEQUENCE: show / caveat / withhold, why, and what it costs. The
// consequence map is the shared, twin-tested core; the live page reuses it to
// reason over a text query while trace() below mirrors the image-query CLI.
import { calibrate, looScores } from './conformal.js';
import { heuristicVotes, aggregate } from './judge.js';
import { debate, fromCouncil } from './debate.js';
import { compose, gateTrust, conformalTrust, councilTrust, marginTrust, WEIGHTS } from './trust.js';

export const STRONG = 0.80, MODERATE = 0.72, WEAK = 0.66;
// display rounding, HALF-UP via floor(x*100+0.5) — the identical IEEE expression
// Python uses, so the twin strings match byte-for-byte (toFixed/:.2f and
// Math.round/:.0% otherwise disagree on exact halves). See reason.py.
const pct = v => Math.floor(v * 100 + 0.5);
const f2 = v => { const m = Math.floor(Math.abs(v) * 100 + 0.5) / 100; return (v < 0 && m > 0) ? '-' + m.toFixed(2) : m.toFixed(2); };
const sf2 = v => { const m = Math.floor(Math.abs(v) * 100 + 0.5) / 100; return (v < 0 && m > 0) ? '-' + m.toFixed(2) : '+' + m.toFixed(2); };
const rnd = v => (v === null || v === undefined) ? null : Math.round(v * 1000) / 1000;
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const fused = it => { const v = [...it.image_emb, ...it.text_emb]; return v.map(x => x / Math.SQRT2); };
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

// The decision map — the point of the whole stack. Pure; matches reason.py.
export function consequence(tr, council, deb) {
  const lvl = tr.level;
  if (lvl === 'high')
    return { action: 'show it as the answer', status: 'ok',
      because: 'every lens agrees and the panel reached consensus',
      effect: 'the user sees a confident, defensible match' };
  if (lvl === 'abstain') {
    if (tr.reason === 'split decision' || (deb && !deb.consensus))
      return { action: 'withhold — genuinely contested', status: 'stop',
        because: (deb && !deb.consensus) ? 'the panel deliberated and still split into factions'
          : 'the honesty lenses split — no majority to trust',
        effect: 'ask the user or broaden the query rather than guess' };
    return { action: 'withhold — not enough signal', status: 'stop',
      because: 'too few lenses cleared their own bar to compose a verdict',
      effect: "say 'no confident match' instead of showing a coin flip" };
  }
  if (lvl === 'medium') {
    const why = council.decision === 'abstain' ? "the council couldn't confirm it"
      : council.decision === 'relevant' ? 'it missed the calibrated confidence bar'
      : 'the judges leaned against it';
    return { action: 'show it with a caveat', status: 'caution', because: why,
      effect: 'label it a loose match so the user calibrates trust' };
  }
  return { action: 'show it, flagged as weak', status: 'caution',
    because: "the lenses agree it's a poor match",
    effect: 'keep it, but make the low confidence explicit' };
}

const st = (ok, stop = false) => stop ? 'stop' : (ok ? 'ok' : 'caution');

// Mirror of reason.trace — image-query, model-free, for the twin test.
export function trace(q, items, alpha = 0.2) {
  const others = items.filter(it => it !== q);
  const ranked = [...others].sort((a, b) => dot(fused(q), fused(b)) - dot(fused(q), fused(a)));
  const r = ranked[0];
  const scores = ranked.map(it => dot(fused(q), fused(it)));
  const cos = scores[0];
  const icos = dot(r.image_emb, q.image_emb);
  const tau = 1 - calibrate(looScores(items), alpha);

  const votes = heuristicVotes(q, r);
  const council = aggregate(votes);
  const { names, opinions, weights } = fromCouncil(votes);
  const deb = opinions.length >= 2 ? debate(opinions, weights) : null;

  const margin = scores.length > 1 ? cos - mean(scores.slice(1)) : 0;
  const signals = [
    { name: 'gate', trust: gateTrust(cos, STRONG, MODERATE, WEAK), weight: WEIGHTS.gate },
    { name: 'conformal', trust: conformalTrust(icos, tau), weight: WEIGHTS.conformal },
    { name: 'council', trust: councilTrust(council), weight: WEIGHTS.council },
    { name: 'margin', trust: marginTrust([cos, ...scores.slice(1)]), weight: WEIGHTS.margin },
  ];
  const tr = compose(signals);
  const tag = (r.file || r.path || '').split('/').pop();

  const steps = [
    { stage: 'retrieve', icon: '🔍', premise: `embed the query, score all ${others.length} candidates`,
      conclusion: `top match is ${tag} at similarity ${f2(cos)}`, signal: rnd(cos), status: st(cos >= MODERATE) },
    { stage: 'rank', icon: '🥇', premise: 'how cleanly does #1 separate from the pack?',
      conclusion: margin >= 0.03 ? `leads by margin ${sf2(margin)}` : `a near-tie (margin ${sf2(margin)})`,
      signal: rnd(margin), status: st(margin >= 0.03) },
    { stage: 'conformal', icon: '🎯', premise: `is it inside the ${pct(1 - alpha)}% coverage set (cos ≥ ${f2(tau)})?`,
      conclusion: icos >= tau - 1e-9 ? 'clears the calibrated bar' : 'below the bar — conformal abstains',
      signal: rnd(icos), status: st(icos >= tau - 1e-9, icos < tau - 1e-9) },
    { stage: 'council', icon: '⚖️', premise: `${council.nValid} rubric-judges score it`,
      conclusion: council.decision + (council.consensus !== null ? ` (consensus ${pct(council.consensus)}%)` : ' — hung jury'),
      signal: rnd(council.mean), status: st(council.decision === 'relevant', council.decision === 'abstain') },
  ];
  if (deb) {
    const camps = deb.factions.map(g => '{' + g.map(i => names[i]).join(', ') + '}').join(' vs ');
    steps.push({ stage: 'debate', icon: '🗣️', premise: 'the judges argue, updating toward peers they can hear',
      conclusion: deb.consensus ? `consensus after ${deb.rounds} rounds → ${deb.verdict}` : `contested: ${camps}`,
      signal: rnd(deb.score), status: st(deb.consensus, !deb.consensus) });
  }
  steps.push({ stage: 'trust', icon: '🧮', premise: `compose the ${tr.nValid}/${tr.nTotal} lenses that voted`,
    conclusion: `trust: ${tr.level}` + (tr.score !== null ? ` (${f2(tr.score)})` : ''),
    signal: rnd(tr.score), status: st(tr.level === 'high', tr.level === 'abstain') });

  return { result: r, steps, trust: tr, council, debate: deb, consequence: consequence(tr, council, deb) };
}
