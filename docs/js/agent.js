// Mirror of agent.py — the embedding agent: propose ⇄ critique, publish only
// when satisfied.
//
// Each round, a PROPOSER drafts labels + a caption through one prompt template;
// a CRITIC then checks the draft with two dot products a human reviewer would
// approve of: does the caption's embedding point back at the image (aligned)?
// and are the accepted labels sure of themselves (confident)? Satisfied → stop.
// Not → next round, next template. Same conditional-edge loop as agent.py.
import { dot, fuse } from './rank.js';
import { multiLabel } from './labels.js';
import { VOCAB, TEMPLATE_POOL, NEUTRAL_PROMPT, tagPrompts, captionFor } from './templates.js';

export const MIN_ALIGNED = 0.20;    // matching CLIP pairs score ~0.2–0.35
export const MIN_CONFIDENT = 0.60;  // accepted labels must average this

const critique = (record, template) => {
  const aligned = dot(record.imageEmb, record.textEmb);
  const confident = record.labels.length
    ? record.labels.reduce((s, l) => s + l.prob, 0) / record.labels.length : 0;
  return {
    aligned, confident, template,
    satisfied: aligned >= MIN_ALIGNED && confident >= MIN_CONFIDENT,
    score: aligned / MIN_ALIGNED + confident / MIN_CONFIDENT,
  };
};

// The loop. encodeText: async texts => unit vectors. onRound fires after each
// round so the page can render the agent thinking. Returns the best
// {record, verdict} — publish only if verdict.satisfied.
export async function runAgent(imageEmb, encodeText, onRound = () => {}) {
  const neutralEmb = (await encodeText([NEUTRAL_PROMPT]))[0];
  let best = null;
  for (const template of TEMPLATE_POOL) {
    const tagEmbs = await encodeText(tagPrompts(template));
    const labels = multiLabel(imageEmb, tagEmbs, neutralEmb, VOCAB);
    const caption = labels.length ? captionFor(labels.map(l => l.tag)) : NEUTRAL_PROMPT;
    const [textEmb] = await encodeText([caption]);
    const record = { labels, caption, imageEmb, textEmb, fusedEmb: fuse(imageEmb, textEmb) };
    const verdict = critique(record, template);
    onRound(record, verdict);
    if (!best || verdict.score > best.verdict.score) best = { record, verdict };
    if (verdict.satisfied) break;          // the conditional edge: done early
  }
  return best;
}
