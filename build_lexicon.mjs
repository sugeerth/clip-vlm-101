// build_lexicon.mjs — the instant layer's dictionary (docs/lexicon.json).
//
// pipeline: db.json (export_web.py) ──► THIS ──► docs/lexicon.json ──► docs/js/instant.js
//
// The front page answers on the FIRST keystroke, before the ~65 MB text
// encoder has even downloaded. It can, because ranking is linear in the
// query vector:
//
//     score(image, q) = image_emb · q − bias[image]
//     q ≈ mean(word vectors)          ⇒   score ≈ mean(image_emb · word) − bias
//
// So instead of shipping word VECTORS (512 floats each) we ship each word's
// precomputed SCORE against every gallery item (14 bytes each, uint8) — the
// bag-of-words ranking falls out of averaging rows. It is a preview: the
// real encoder replaces it the moment it is ready.
//
// It also ships bias[i] = image i's mean score over all lexicon words: the
// hub correction instant.js subtracts from every text query (37/38 top-1 on
// a labelled query set vs 35/38 without it — see the table in instant.js).
//
// This runs in Node with transformers.js — the SAME q8 checkpoint the
// browser runs — so a one-word preview is bit-for-bit the browser's answer.
//
//   npm i --no-save @huggingface/transformers@3.5.2
//   node build_lexicon.mjs [--words words.txt] [--max 6000]
//
// --words: optional frequency-ordered word list (one per line), e.g.
// google-10000-english-usa-no-swears.txt. Re-run after re-exporting db.json:
// the page checks the signature and ignores a stale lexicon.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VOCAB } from './docs/js/templates.js';
import { MODELS, DEFAULT_MODEL } from './docs/js/models.js';
import { STOP, gallerySignature, imageVectors, quantize } from './docs/js/instant.js';

const ROOT = dirname(fileURLToPath(import.meta.url));
const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const MAX = Number(arg('--max', 6000));
const WORDS_FILE = arg('--words', null);

// Visual words first: prefix completion picks the EARLIEST word with the
// typed prefix, so "ca" → "cat", "flu" → "fluffy", not "california".
const VISUAL = `
cat kitten dog puppy bear parrot bird eagle owl penguin duck chicken horse cow
pig sheep goat lion tiger elephant giraffe zebra monkey gorilla panda fox wolf
deer rabbit squirrel mouse frog turtle snake lizard fish shark whale dolphin
octopus crab butterfly bee spider insect ant ladybug dinosaur dragon pet pets
animal animals wildlife
pizza burger sandwich pasta noodles ramen sushi rice bread cake cookie
chocolate cream coffee tea wine beer juice milk cheese egg salad soup steak
meat apple banana orange lemon strawberry grape cherry watermelon pineapple
mango peach pear berries vegetables tomato carrot potato corn mushroom pepper
spicy sweet delicious tasty dessert breakfast lunch dinner snack candy donut
pie fruit food drink eat
mountain mountains hill valley volcano canyon desert forest jungle beach ocean
sea lake river waterfall island cave glacier snow ice rain storm clouds sky
sunset sunrise night stars moon sun planet earth space galaxy universe nebula
pluto mars saturn rocket astronaut field meadow garden park flower flowers rose
tulip sunflower daisy tree trees leaves grass autumn winter spring summer water
nature landscape scenery outdoors
city town village street road highway bridge tower skyscraper building
buildings house home castle palace church cathedral temple mosque ruins
monument statue landmark museum market shop restaurant cafe kitchen bedroom
office school station airport harbor paris london rome tokyo europe asia
africa america france italy england japan china india eiffel architecture
car truck bus train plane airplane boat ship sailboat bicycle bike motorcycle
scooter helicopter tractor taxi vehicle
book books computer laptop phone camera clock watch chair table bed sofa lamp
door window toy toys ball doll teddy guitar piano violin drum painting art
sculpture glass bottle cup plate bowl knife spoon hat shoes dress shirt bag
umbrella balloon candle gift kite
person people man woman child baby boy girl family crowd friends portrait face
smile selfie athlete soldier king queen
fluffy furry cute adorable fuzzy soft wild tiny small big huge giant tall old
ancient historic famous modern colorful bright dark red yellow green blue
purple pink black white golden silver wooden metal stone frozen wet sunny
cloudy foggy snowy rainy peaceful calm beautiful scenic majestic romantic
scary spooky funny happy sad angry fresh hot cold warm cozy vintage retro
futuristic outer orbit cold
sport sports music dance travel vacation holiday wedding party christmas
halloween birthday fire smoke light reflection
`.split(/\s+/).filter(Boolean);

// 1. the word list — gallery vocabulary, visual words, then plain English
const freq = WORDS_FILE && existsSync(WORDS_FILE)
  ? readFileSync(WORDS_FILE, 'utf8').split(/\r?\n/) : [];
const seen = new Set(), words = [];
for (const w of [...VOCAB, ...VISUAL, ...freq]) {
  const t = w.trim().toLowerCase();
  if (!/^[a-z]{2,}$/.test(t) || STOP.has(t) || seen.has(t)) continue;
  seen.add(t); words.push(t);
  if (words.length >= MAX) break;
}

// 2. the gallery — the SAME image vectors the page ranks with
const db = JSON.parse(readFileSync(join(ROOT, 'docs/db.json'), 'utf8'));
const { mat, n, dim } = imageVectors(db.items);

// 3. embed every word with the browser's own checkpoint (q8, like clip.js)
const m = MODELS[DEFAULT_MODEL];
const T = await import('@huggingface/transformers');
const tok = await T.AutoTokenizer.from_pretrained(m.id);
const model = await T.CLIPTextModelWithProjection.from_pretrained(m.id, { dtype: 'q8' });

const scores = new Float32Array(words.length * n);
const t0 = Date.now(), B = 64;
for (let s = 0; s < words.length; s += B) {
  const batch = words.slice(s, s + B);
  const { text_embeds } = await model(tok(batch, { padding: m.padding, truncation: true }));
  const d = text_embeds.dims[1], data = text_embeds.data;
  if (d !== dim) throw new Error(`model dim ${d} != gallery dim ${dim}`);
  for (let b = 0; b < batch.length; b++) {
    const row = data.subarray(b * d, (b + 1) * d);
    let norm = 0;
    for (let k = 0; k < d; k++) norm += row[k] * row[k];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let k = 0; k < d; k++) dot += mat[i * d + k] * row[k];
      scores[(s + b) * n + i] = dot / norm;
    }
  }
  process.stdout.write(`\r  embedded ${Math.min(s + B, words.length)}/${words.length} words`);
}
console.log(`  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);

// 4. the hub bias: each image's mean raw score over every word
const bias = new Float64Array(n);
for (let w = 0; w < words.length; w++) for (let i = 0; i < n; i++) bias[i] += scores[w * n + i];
for (let i = 0; i < n; i++) bias[i] /= words.length;

// 5. quantize the RAW scores to one byte per (word, item) and write
const q = quantize(scores);
const out = {
  about: 'precomputed CLIP scores: word × gallery image (image tower), uint8, row-major, + per-image hub bias — see build_lexicon.mjs',
  model: m.id, dtype: 'q8', sig: gallerySignature(db.items),
  n_items: n, n_words: words.length, lo: q.lo, hi: q.hi,
  bias: Array.from(bias, b => +b.toFixed(5)),
  words: words.join(' '),
  scores: Buffer.from(q.bytes).toString('base64'),
};
const path = join(ROOT, 'docs/lexicon.json');
writeFileSync(path, JSON.stringify(out));
console.log(`wrote ${path}: ${words.length} words × ${n} items, ` +
  `${(JSON.stringify(out).length / 1024).toFixed(0)} KB, range [${q.lo.toFixed(3)}, ${q.hi.toFixed(3)}]`);
