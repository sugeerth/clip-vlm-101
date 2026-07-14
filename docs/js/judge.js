// Mirror of judge.py — a COUNCIL of LLM judges, aggregated honestly.
//
// After retrieval → re-rank → conformal → explain+gate, this is the last
// honesty layer: don't trust one model's one score. Convene a PANEL of judges,
// each a different rubric (relevance / specificity / faithfulness). A GATE
// (parseScore) turns each judge's raw text into a number in [0,1]; a judge with
// no parseable score ABSTAINS instead of voting garbage. The COUNCIL takes a
// confidence-weighted mean, measures CONSENSUS = 1 − (max − min), and — like
// conformal.js — ABSTAINS rather than rule when there's no quorum or the panel
// is too split (a "hung jury"). Panel-of-evaluators idea: Verga et al. 2024.
//
// The math is a byte-for-byte twin of judge.py. judge.py ships a model-free
// heuristic judge (runs on the committed gallery); councilWithLLM() below swaps
// in a real in-browser LLM per rubric and runs its output through the SAME gate
// and the SAME aggregate().
const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

export const QUORUM = 2;
export const ACCEPT = 0.5;
export const HUNG_SPREAD = 0.5;
export const REL_LO = 0.65, REL_HI = 0.90;

export const RUBRICS = [
  { name: 'relevance', confidence: 0.9, prompt: 'How well does this image answer the query? Score 0 to 1.' },
  { name: 'specificity', confidence: 0.7, prompt: 'Is this a precise match, not just loosely related? Score 0 to 1.' },
  { name: 'faithfulness', confidence: 0.6, prompt: "Do the image's own tags justify calling it a match? Score 0 to 1." },
];

// The gate: pull a score in [0,1] out of a judge's raw text, or null. Accepts
// "0.7", ".7", "7/10", "8 out of 10", "70%", "score: 0.9"; rejects out-of-range.
export function parseScore(text) {
  if (text === null || text === undefined) return null;
  const t = String(text).toLowerCase();
  let m = t.match(/(\d+(?:\.\d+)?)\s*(?:\/|\s+out\s+of\s+)\s*(\d+(?:\.\d+)?)/);
  if (m) {
    const num = parseFloat(m[1]), den = parseFloat(m[2]);
    if (den > 0 && num >= 0 && num <= den) return num / den;
  }
  m = t.match(/(\d+(?:\.\d+)?)\s*%/);
  if (m) {
    const v = parseFloat(m[1]);
    if (v >= 0 && v <= 100) return v / 100;
  }
  m = t.match(/(?<![\d.])(?:0?\.\d+|[01](?:\.0+)?)(?![\d])/);
  if (m) {
    const v = parseFloat(m[0]);
    if (v >= 0 && v <= 1) return v;
  }
  return null;
}

// The council's ruling from a list of votes:
// [{ name, score: number|null, confidence, rationale? }]. A null score abstains.
export function aggregate(votes) {
  const valid = votes.filter(v => v.score !== null && v.score !== undefined);
  const abstained = votes.filter(v => v.score === null || v.score === undefined).map(v => v.name);
  const perJudge = votes.map(v => ({ name: v.name, score: v.score ?? null,
    confidence: v.confidence ?? 1.0, rationale: v.rationale ?? '' }));
  const base = { perJudge, nValid: valid.length, nTotal: votes.length, abstained };
  if (valid.length < QUORUM)
    return { ...base, decision: 'abstain', reason: 'no quorum', mean: null, consensus: null };
  const scores = valid.map(v => v.score);
  let weights = valid.map(v => Math.max(v.confidence ?? 1.0, 0));
  if (weights.reduce((s, w) => s + w, 0) <= 0) weights = valid.map(() => 1);
  const W = weights.reduce((s, w) => s + w, 0);
  // unrounded: identical arithmetic in both twins → identical floats. Rounding
  // (Python banker's vs JS half-up) is a DISPLAY concern, done by the panel.
  const mean = scores.reduce((s, v, i) => s + v * weights[i], 0) / W;
  const spread = Math.max(...scores) - Math.min(...scores);
  const consensus = Math.max(0, 1 - spread);
  if (spread > HUNG_SPREAD)
    return { ...base, decision: 'abstain', reason: 'hung jury', mean, consensus, spread };
  return { ...base, decision: mean >= ACCEPT ? 'relevant' : 'not relevant', reason: 'ruled',
    mean, consensus, spread };
}

// The simpler panel rule: each valid judge casts a yes/no vote; majority wins,
// a tie abstains. 'multiple LLMs as judges'; aggregate() is the full council.
export function majority(votes, threshold = ACCEPT) {
  const valid = votes.filter(v => v.score !== null && v.score !== undefined);
  if (valid.length < QUORUM) return { decision: 'abstain', reason: 'no quorum', yes: 0, no: 0, nValid: valid.length };
  const yes = valid.filter(v => v.score >= threshold).length;
  const no = valid.length - yes;
  if (yes === no) return { decision: 'abstain', reason: 'tie', yes, no, nValid: valid.length };
  return { decision: yes > no ? 'relevant' : 'not relevant', reason: 'majority', yes, no, nValid: valid.length };
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const fused = it => { const v = [...it.image_emb, ...it.text_emb]; const r = Math.SQRT2; return v.map(x => x / r); };

// Model-free judges: score the three rubrics from stored signals, so the
// council mechanism runs with no LLM. Each rubric reads a different signal —
// which is exactly why they can disagree. Mirrors judge.heuristic_votes.
export function heuristicVotes(queryItem, resultItem) {
  const cos = dot(fused(queryItem), fused(resultItem));
  const qtags = queryItem.tags || [];
  const rtags = new Set(resultItem.tags || []);
  const shared = qtags.filter(t => rtags.has(t));
  const relevance = Math.min(Math.max((cos - REL_LO) / (REL_HI - REL_LO), 0), 1);
  const specificity = shared.length / Math.max(qtags.length, 1);
  const faithfulness = (qtags.length && rtags.has(qtags[0])) ? 1.0 : (shared.length ? 0.5 : 0.0);
  const scores = { relevance, specificity, faithfulness };
  return RUBRICS.map(r => ({ name: r.name, score: scores[r.name], confidence: r.confidence,
    rationale: `${r.name} signal = ${scores[r.name].toFixed(2)}` }));
}

export const council = (queryItem, resultItem) => aggregate(heuristicVotes(queryItem, resultItem));

// OPTIONAL: convene a council of REAL in-browser LLM judges. Each rubric is one
// call to a small instruct model (SmolLM2, like explain.js); the raw reply runs
// through parseScore (a judge that emits no number abstains) and the SAME
// aggregate(). Lazy-loads transformers.js; WebGPU if present, else WASM. On any
// failure a judge abstains, so the council degrades gracefully to a quorum.
let genP = null;
export async function councilWithLLM(query, ev, onStatus = () => {}) {
  const facts = `query: "${query}"\nresult tags: ${(ev.tags || []).join(', ') || 'none'}\n`
    + `shared with query: ${(ev.shared || []).join(', ') || 'none'}\n`
    + `retrieval similarity: ${(ev.topScore ?? 0).toFixed(2)}`;
  let gen;
  try {
    const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
    genP ??= (async () => {
      onStatus('loading a small language model (one time)…');
      const { pipeline } = await import(CDN);
      return pipeline('text-generation', 'HuggingFaceTB/SmolLM2-135M-Instruct',
        webgpu ? { dtype: 'q4f16', device: 'webgpu' } : { dtype: 'q4' });
    })().catch(e => { genP = null; throw e; });
    gen = await genP;
  } catch (err) {
    console.error('councilWithLLM (model load):', err);
    return { ...aggregate(RUBRICS.map(r => ({ name: r.name, score: null, confidence: r.confidence }))),
      source: 'llm unavailable' };
  }
  const votes = [];
  for (const r of RUBRICS) {
    onStatus(`judge ${votes.length + 1}/${RUBRICS.length}: ${r.name}…`);
    let raw = null;
    try {
      const out = await gen([
        { role: 'system', content: 'You are a strict judge. Reply with ONLY a number from 0 to 1 (e.g. 0.7). No words.' },
        { role: 'user', content: `${r.prompt}\n${facts}` },
      ], { max_new_tokens: 8, do_sample: false });
      raw = out[0].generated_text.at(-1).content;
    } catch (err) { console.error(`judge ${r.name}:`, err); }
    votes.push({ name: r.name, score: parseScore(raw), confidence: r.confidence,
      rationale: raw ? `said "${String(raw).trim().slice(0, 24)}"` : 'no answer' });
  }
  return { ...aggregate(votes), source: 'llm (gated)' };
}
