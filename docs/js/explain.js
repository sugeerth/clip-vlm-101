// Mirror of explain.py — say WHY results matched, and gate the hallucinations.
//
// Two halves, and the second is the point:
//   describe()  a grounded explanation built ONLY from evidence (shared tags,
//               the top score, a calibrated strength word) — passes the gate
//               by construction.
//   verify()    the hallucination gate: whatever wrote the text — this
//               template or an untrusted in-browser LLM — a sentence survives
//               only if every content word is a real tag/query word/safe glue,
//               every number matches a real score, every strength word is the
//               true one. Else the whole sentence is redacted, with a reason.
//
// Closed-world exact-match attribution (AIS / RAGAS-faithfulness): the
// evidence is a finite tag+score set, so entailment IS set membership — no
// NLI model, no LLM judge. The optional explainWithLLM() lets a real model
// phrase things, then runs its output through the SAME verify() and falls
// back to the template if the model invents anything.
import { VOCAB } from './templates.js';

export const STRONG = 0.30, MODERATE = 0.25, WEAK = 0.20;
const NUM_TOL = 0.02;
const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

export const bucket = s =>
  s >= STRONG ? 'strong' : s >= MODERATE ? 'moderate' : s >= WEAK ? 'weak' : 'very weak';

function norm(w) {
  w = w.toLowerCase().replace(/[^\w.%-]/g, '');
  return (w.length > 3 && w.endsWith('s')) ? w.slice(0, -1) : w;
}

// NORMALIZED through norm() so text and vocab agree on plurals; must cover
// every glue word the template emits in BOTH the strong- and weak-match tails.
const SAFE_VOCAB = new Set(['the', 'a', 'an', 'these', 'this', 'they', 'all',
  'both', 'most', 'and', 'or', 'of', 'to', 'with', 'in', 'on', 'no', 'not',
  'dont', 'single', 'image', 'images', 'result', 'results', 'match', 'matches',
  'matched', 'similar', 'similarity', 'score', 'scores', 'query', 'search',
  'top', 'share', 'shares', 'shared', 'common', 'tag', 'tags', 'show', 'shows',
  'way', 'ways', 'different', 'strongest', 'confident', 'loose', 'treat',
  'them', 'as', 'isnt', 'is', 'are', 'that', 'it', 'model'].map(norm));
const STRENGTH_WORDS = new Set(['strong', 'moderate', 'weak', 'very weak', 'perfect', 'exact']);
const VOCAB_SET = new Set(VOCAB.map(norm));
function numbers(text) {
  const out = [];
  for (const m of text.matchAll(/(\d+(?:\.\d+)?)\s*%/g)) out.push(+m[1] / 100);
  for (const m of text.matchAll(/(?<![\d%])0?\.\d+(?![\d%])/g)) out.push(+m[0]);
  return out;
}
const join = ws => ws.length <= 1 ? (ws[0] ?? '')
  : ws.slice(0, -1).join(', ') + ' and ' + ws.at(-1);

// ranked: [{ item: { tags }, score }, ...] — the verifiable world.
export function buildEvidence(query, ranked, k = 5) {
  const top = ranked.slice(0, k);
  if (!top.length) return { query, queryToks: new Set(), tags: new Set(), shared: [],
    scores: [], topScore: 0, strength: 'very weak' };
  const tagSets = top.map(r => new Set((r.item.tags || []).map(norm)));
  const tags = new Set(tagSets.flatMap(s => [...s]));
  const shared = (top[0].item.tags || []).filter(t => tagSets.every(s => s.has(norm(t))));
  const scores = top.map(r => Math.round(r.score * 100) / 100);
  const topScore = Math.max(...scores);
  return { query, queryToks: new Set((query || '').split(/\s+/).map(norm).filter(Boolean)),
    tags, shared, scores, topScore, strength: bucket(topScore) };
}

export function describe(ev) {
  if (!ev.scores.length) return 'No results to explain.';
  const n = ev.scores.length;
  const head = ev.shared.length
    ? `The top ${n} results all show ${join(ev.shared.slice(0, 3))}.`
    : `The top ${n} results share no single tag — they match the query in different ways.`;
  let tail = ` The strongest match scores ${ev.topScore.toFixed(2)} (${ev.strength}).`;
  if (ev.strength === 'weak' || ev.strength === 'very weak')
    tail += ' Treat them as loose matches — the model isnt confident.';
  return head + tail;
}

// The gate. Returns { verified, stripped:[{text,reasons}], clean }.
export function verify(text, ev, numTol = NUM_TOL) {
  const kept = [], stripped = [];
  for (const sentence of text.trim().split(/(?<=[.!?])\s+/)) {
    if (!sentence.trim()) continue;
    const reasons = [], vague = [];
    const low = sentence.toLowerCase();
    for (const phrase of STRENGTH_WORDS)
      if (phrase !== ev.strength && new RegExp(`\\b${phrase}\\b`).test(low)
          && !(phrase === 'weak' && ev.strength === 'very weak'))
        reasons.push(`says '${phrase}' — the match is '${ev.strength}'`);
    for (const num of numbers(sentence))
      if (!ev.scores.some(s => Math.abs(num - s) <= numTol))
        reasons.push(`cites ${num.toFixed(2)} — matches no result score`);
    for (const raw of sentence.match(/[A-Za-z][\w-]*/g) || []) {
      const w = norm(raw);
      if (!w || SAFE_VOCAB.has(w) || STRENGTH_WORDS.has(w) || ev.tags.has(w) || ev.queryToks.has(w))
        continue;
      if (VOCAB_SET.has(w)) reasons.push(`claims '${raw}' — not in the results`);
      else vague.push(raw);
    }
    if (vague.length && !reasons.length) reasons.push(`unverifiable wording: ${vague.slice(0, 4).join(', ')}`);
    else if (vague.length) reasons.push(`and unverifiable wording: ${vague.slice(0, 4).join(', ')}`);
    (reasons.length ? stripped : kept).push({ text: sentence, reasons });
  }
  return { verified: kept.map(s => s.text).join(' '), stripped, clean: !stripped.length };
}

// One call: grounded explanation, gated. draft (LLM prose) is verified instead
// of the template when given; the template is the floor if nothing survives.
export function explain(query, ranked, { draft = null, k = 5 } = {}) {
  const ev = buildEvidence(query, ranked, k);
  const res = verify(draft ?? describe(ev), ev);
  res.evidence = ev;
  res.explanation = res.verified || describe(ev);
  return res;
}

// OPTIONAL: let a small in-browser LLM phrase the explanation, then gate it.
// Lazy-loads transformers.js only on demand; WebGPU if present, else WASM;
// falls back to the template if the model or the gate leaves nothing.
let genP = null;
export async function explainWithLLM(query, ranked, onStatus = () => {}, k = 5) {
  const ev = buildEvidence(query, ranked, k);
  try {
    const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
    genP ??= (async () => {
      onStatus('loading a small language model (one time)…');
      const { pipeline } = await import(CDN);
      return pipeline('text-generation', 'HuggingFaceTB/SmolLM2-135M-Instruct',
        webgpu ? { dtype: 'q4f16', device: 'webgpu' } : { dtype: 'q4' });
    })().catch(e => { genP = null; throw e; });
    const gen = await genP;
    onStatus('writing an explanation…');
    const facts = `query: "${query}"\nshared tags: ${ev.shared.join(', ') || 'none'}\n`
      + `top similarity: ${ev.topScore.toFixed(2)} (${ev.strength})\nresult count: ${ev.scores.length}`;
    const out = await gen([
      { role: 'system', content: 'You explain image-search results in one or two short sentences. Use ONLY the given tags, numbers and words. Never invent objects or scores.' },
      { role: 'user', content: facts },
    ], { max_new_tokens: 80, do_sample: false });
    const draft = out[0].generated_text.at(-1).content;
    const res = explain(query, ranked, { draft, k });
    res.source = res.verified ? 'llm (gated)' : 'template (llm output failed the gate)';
    return res;
  } catch (err) {
    console.error('explainWithLLM:', err);
    const res = explain(query, ranked, { k });
    res.source = 'template (llm unavailable)';
    return res;
  }
}
