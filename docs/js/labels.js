// Mirror of labels.py — multi-LABEL classification: every tag decides for itself.
//
// For each tag, a two-way softmax between "a photo of a cat" (tag prompt) and
// "a photo" (neutral prompt) collapses to a sigmoid on the score gap — an
// independent probability per tag, so the label set is DYNAMIC: sized by the
// image, not by top-k.
import { dot } from './rank.js';

// CLIP's learned softmax temperature (the model ships with exp(4.6) ≈ 100).
export const LOGIT_SCALE = 100.0;
export const DEFAULT_THRESHOLD = 0.5;

// One independent probability per tag: sigmoid of the (tag − neutral) gap.
export const labelProbs = (imageEmb, tagEmbs, neutralEmb, scale = LOGIT_SCALE) =>
  tagEmbs.map(t =>
    1 / (1 + Math.exp(-scale * (dot(t, imageEmb) - dot(neutralEmb, imageEmb)))));

// The dynamic label set: [{tag, prob}], best first, above threshold.
export function multiLabel(imageEmb, tagEmbs, neutralEmb, vocab,
                           threshold = DEFAULT_THRESHOLD) {
  const probs = labelProbs(imageEmb, tagEmbs, neutralEmb);
  return vocab.map((tag, i) => ({ tag, prob: probs[i] }))
    .filter(x => x.prob >= threshold)
    .sort((a, b) => b.prob - a.prob);
}
