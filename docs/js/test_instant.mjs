// The instant layer (instant.js + lexicon.json), model-free.
//   node docs/js/test_instant.mjs
// Pins: the committed lexicon belongs to the committed gallery (a re-export
// of db.json without re-running build_lexicon.mjs fails here, not silently
// on the live page), the preview math is the documented linear average,
// and the gallery's own names preview to the right photo.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  STOP, imageVectors, scoreAll, topK, gallerySignature, quantize,
  loadLexicon, tokenize, complete, preview, LRU,
} from './instant.js';

const DOCS = join(dirname(fileURLToPath(import.meta.url)), '..');
const db = JSON.parse(readFileSync(join(DOCS, 'db.json'), 'utf8'));
const lexJson = JSON.parse(readFileSync(join(DOCS, 'lexicon.json'), 'utf8'));
let passed = 0;
const ok = (cond, msg) => {
  if (!cond) { console.error(`FAIL  ${msg}`); process.exit(1); }
  passed++; console.log(`ok    ${msg}`);
};
const name = i => db.items[i].file.replace(/images\/\d+_|\.jpg/g, '');

// 1. the committed lexicon matches the committed gallery
const lex = loadLexicon(lexJson, db.items);
ok(lex !== null, `lexicon.json signature ${lexJson.sig} matches docs/db.json`);
ok(lex.words.length === lexJson.n_words && lex.scores.length === lex.words.length * db.items.length,
  `${lex.words.length} words × ${db.items.length} items, one byte each`);
ok(lex.bias.length === db.items.length && [...lex.bias].every(Number.isFinite), 'one finite hub bias per image');

// 2. a stale lexicon is refused, never used
const tampered = { ...db.items[0], image_emb: [9, ...db.items[0].image_emb.slice(1)] };
ok(loadLexicon(lexJson, [tampered, ...db.items.slice(1)]) === null, 'changed gallery → lexicon refused');
ok(loadLexicon(lexJson, db.items.slice(1)) === null, 'fewer items → lexicon refused');
ok(gallerySignature(db.items) === gallerySignature(JSON.parse(JSON.stringify(db.items))), 'signature is deterministic');

// 3. the gallery's own names preview to their own photo
const NAMES = { apple: 'apple', bear: 'bear', bicycle: 'bicycle', castle: 'castle', cat: 'cat',
  dog: 'dog', eiffel: 'eiffel_tower', mountain: 'mountains', parrot: 'parrot', pizza: 'pizza',
  planet: 'pluto', sunflower: 'sunflower', waterfall: 'waterfall' };
for (const [word, want] of Object.entries(NAMES)) {
  const p = preview(lex, word + ' ');
  ok(p && name(topK(p.scores, 1)[0]) === want, `preview "${word}" → ${want}`);
}

// 4. preview = mean of the words' dequantized rows − bias (the linearity claim)
const row = w => {
  const r = lex.row.get(w) * lex.n;
  return Array.from({ length: lex.n }, (_, i) => lex.lo + lex.step * lex.scores[r + i]);
};
const p2 = preview(lex, 'fluffy animal ');
const f = row('fluffy'), a = row('animal');
ok(p2.words.join() === 'fluffy,animal'
  && p2.scores.every((s, i) => Math.abs(s - ((f[i] + a[i]) / 2 - lex.bias[i])) < 1e-6),
  'two-word preview = average of the two rows − bias');

// 5. tokenizing, stopwords, prefix completion, plurals
ok(tokenize("A Bird's NEST, 2 eggs").join() === 'a,bird,s,nest,eggs', 'tokenize: lowercase letters only');
ok(preview(lex, 'a fluf').words.join() === 'fluffy', 'last word still typing → completed by prefix ("fluf" → fluffy)');
ok(preview(lex, 'fluf ') === null, 'a FINISHED unknown word is not guessed');
ok(complete(lex, 'ca') === 'cat', 'visual words win prefix completion ("ca" → cat)');
ok(!lex.row.has('parrots') && preview(lex, 'parrots ').words.join() === 'parrot',
  'a plural not in the lexicon falls back to its singular ("parrots" → parrot)');
ok(preview(lex, 'the of a ') === null && preview(lex, 'xyzzy ') === null, 'stopwords / gibberish → no preview');
ok([...STOP].every(w => !lex.row.has(w)), 'no stopword is in the lexicon');

// 6. exact scoring: one packed loop == plain dot products − bias
const IMG = imageVectors(db.items);
const q = db.items[4].text_emb;                     // any unit vector will do
const s = scoreAll(IMG, q, lex.bias);
const manual = db.items.map((it, i) => it.image_emb.reduce((t, x, k) => t + x * q[k], 0) - lex.bias[i]);
ok(s.every((v, i) => Math.abs(v - manual[i]) < 1e-5), 'scoreAll = image_emb · q − bias, every item');
ok(topK(scoreAll(IMG, db.items[4].image_emb), 1)[0] === 4, 'photo → photo: an image finds itself');
ok(!topK(scoreAll(IMG, db.items[4].image_emb), 13, 4).includes(4), 'topK skip: "more like this" omits the photo itself');

// 7. quantization error ≤ half a step; LRU keeps the recent, drops the old
const raw = Float32Array.from({ length: 999 }, (_, i) => Math.sin(i) * 0.2 + 0.25);
const qz = quantize(raw), step = (qz.hi - qz.lo) / 255;
ok(raw.every((v, i) => Math.abs(qz.lo + step * qz.bytes[i] - v) <= step / 2 + 1e-9), 'uint8 quantization within ±½ step');
const c = new LRU(2); c.set('a', 1); c.set('b', 2); c.get('a'); c.set('c', 3);
ok(c.get('a') === 1 && c.get('b') === undefined && c.get('c') === 3, 'LRU evicts the least recently used');

console.log(`\n${passed} passed`);
