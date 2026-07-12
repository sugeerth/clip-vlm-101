// Mirror of user_tower.py — the user half of a two-tower recommender,
// faked with math you can read.
//
// The item tower is already done: every gallery image carries precomputed
// embeddings (that is the point — the expensive half happens offline). The
// user tower's one job is to produce ONE vector in the same space, and the
// simplest honest version is mean-pooling the user's liked items:
//
//     user_vec = unit( mean( item_emb of the likes ) )
//     scores   = every item · user_vec         ← serving, no model anywhere
import { dot, unit, fuse } from './rank.js';

// One unit vector for the user: the renormalized mean of their likes.
export function userVector(likedEmbs) {
  const d = likedEmbs[0].length;
  const mean = new Array(d).fill(0);
  for (const e of likedEmbs) for (let i = 0; i < d; i++) mean[i] += e[i] / likedEmbs.length;
  return unit(mean);
}

// Rank every item against the user vector; never recommend what they liked.
// Items use the same fused [image ; text] vector the Python item tower stores.
export function recommend(items, likedItems, k = 5) {
  const itemEmb = it => fuse(it.image_emb, it.text_emb);
  const u = userVector(likedItems.map(itemEmb));
  const liked = new Set(likedItems);
  return items.filter(it => !liked.has(it))
    .map(item => ({ item, score: dot(itemEmb(item), u) }))
    .sort((a, b) => b.score - a.score).slice(0, k);
}
