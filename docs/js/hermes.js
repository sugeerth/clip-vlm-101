// Hermes — the agentic searcher. Mirror of hermes.py.
//
// A plain search embeds your words once and hopes. Hermes treats the query
// as a DRAFT and works it, the same propose ⇄ critique ⇄ refine loop the
// ingest-side embedding agent (agent.js) runs on images:
//
//   PROPOSE   several phrasings of the query — the prompt is the classifier,
//             so "cat", "a photo of cat" and "a close-up photo of cat" are
//             genuinely different questions to the model
//   CRITIQUE  each phrasing by its retrieval MARGIN: how cleanly the best
//             hit separates from the rest of the pack. A decisive phrasing
//             found something; an indecisive one is guessing
//   REFINE    if no phrasing is decisive, ensemble them — average the unit
//             vectors so phrasing noise cancels (ensemble.py's trick)
//   PUBLISH   only then answer, with the whole trace attached
import { rank, unit } from './rank.js';

export const QUERY_TEMPLATES = [
  '{q}',
  'a photo of {q}',
  'a close-up photo of {q}',
  'an image showing {q}',
];

// a decisive retrieval separates top-1 from the pack by at least this
export const MIN_MARGIN = 0.03;

// top-1 score minus the mean of the rest of the top-k
export const margin = ranked => ranked.length > 1
  ? ranked[0].score - ranked.slice(1).reduce((s, r) => s + r.score, 0) / (ranked.length - 1)
  : (ranked[0]?.score ?? 0);

export async function hermesSearch(query, encodeText, items, k = 8) {
  const phrasings = QUERY_TEMPLATES.map(t => t.replace('{q}', query));
  const embs = await encodeText(phrasings);          // ONE batch, four drafts
  const rounds = phrasings.map((phrasing, i) => {
    const ranked = rank(items, embs[i], 'fused', k);
    return { phrasing, ranked, margin: margin(ranked) };
  });
  let bestI = 0;
  for (let i = 1; i < rounds.length; i++) if (rounds[i].margin > rounds[bestI].margin) bestI = i;
  const best = rounds[bestI];
  if (best.margin >= MIN_MARGIN) {
    // `qvec` is the embedding that produced these results — so a downstream
    // re-ranker scores the SAME phrasing the trace says Hermes chose.
    return { ranked: best.ranked, satisfied: true, chose: best.phrasing, rounds, qvec: embs[bestI] };
  }
  // no phrasing is decisive → refine: ensemble all drafts and answer with that
  const d = embs[0].length;
  const mean = new Array(d).fill(0);
  for (const e of embs) for (let i = 0; i < d; i++) mean[i] += e[i] / embs.length;
  const ens = unit(mean);
  return {
    ranked: rank(items, ens, 'fused', k),
    satisfied: false, chose: 'an ensemble of all phrasings', rounds, qvec: ens,
  };
}
