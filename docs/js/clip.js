// Mirror of embedder.py — CLIP-family checkpoints running in your browser
// via transformers.js. Models load lazily on first use; the browser caches
// them after that. No DOM here: callers pass onStatus/onProgress callbacks.
//
// Multiple "brains" are supported through the registry in models.js —
// setActiveModel() switches which checkpoint the getters resolve to, and
// encoders are cached PER MODEL so switching back is instant. The two
// model families differ in exactly two ways, both handled here:
//   clip   (CLIP, MobileCLIP)  *WithProjection classes, .text_embeds /
//                              .image_embeds, per-checkpoint padding rule
//   siglip (SigLIP, SigLIP 2)  SiglipTextModel / SiglipVisionModel, both
//                              towers emit .pooler_output, 64-token pads
import { unit } from './rank.js';
import { MODELS, DEFAULT_MODEL } from './models.js';

export const MODEL_ID = MODELS[DEFAULT_MODEL].id;   // kept for back-compat
const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

let hf = null;
const lib = async () => (hf ??= await import(CDN));

let active = DEFAULT_MODEL;
const textCache = new Map();    // model key -> in-flight/resolved encoder
const imageCache = new Map();

export const getActiveModel = () => active;
export function setActiveModel(key) {
  if (!MODELS[key]) throw new Error(`unknown model: ${key}`);
  active = key;
}

// (n, d) tensor -> array of n unit-length plain arrays (embedder._unit).
function sliceRows(tensor) {
  const [n, d] = tensor.dims, data = tensor.data, rows = [];
  for (let i = 0; i < n; i++) rows.push(unit(Array.from(data.slice(i * d, (i + 1) * d))));
  return rows;
}

// Both getters memoize the IN-FLIGHT PROMISE per model, so two concurrent
// first uses share one download, and clear it on failure for clean retries.

// resolves to: async (texts) => array of unit vectors (dim per model)
export function getTextEncoder(onStatus, onProgress) {
  const key = active, m = MODELS[key];
  if (!textCache.has(key)) {
    textCache.set(key, (async () => {
      onStatus(`Loading ${m.label.split(' · ')[0]} text encoder…`);
      const T = await lib();
      const tokenizer = await T.AutoTokenizer.from_pretrained(m.id, { progress_callback: onProgress });
      const tokOpts = { padding: m.padding, truncation: true };
      if (m.maxLength) tokOpts.max_length = m.maxLength;
      if (m.kind === 'siglip') {
        const model = await T.SiglipTextModel.from_pretrained(m.id, { dtype: 'q8', progress_callback: onProgress });
        return async texts => sliceRows((await model(tokenizer(texts, tokOpts))).pooler_output);
      }
      const model = await T.CLIPTextModelWithProjection.from_pretrained(m.id, { dtype: 'q8', progress_callback: onProgress });
      return async texts => sliceRows((await model(tokenizer(texts, tokOpts))).text_embeds);
    })().catch(err => { textCache.delete(key); throw err; }));
  }
  return textCache.get(key);
}

// resolves to: async (blob) => one unit vector (dim per model)
export function getImageEncoder(onStatus, onProgress) {
  const key = active, m = MODELS[key];
  if (!imageCache.has(key)) {
    imageCache.set(key, (async () => {
      onStatus(`Loading ${m.label.split(' · ')[0]} vision encoder…`);
      const T = await lib();
      const processor = await T.AutoProcessor.from_pretrained(m.id, { progress_callback: onProgress });
      const Vision = m.kind === 'siglip' ? T.SiglipVisionModel : T.CLIPVisionModelWithProjection;
      const model = await Vision.from_pretrained(m.id, { dtype: 'q8', progress_callback: onProgress });
      return async blob => {
        const image = await T.RawImage.fromBlob(blob);
        const out = await model(await processor(image));
        return sliceRows(m.kind === 'siglip' ? out.pooler_output : out.image_embeds)[0];
      };
    })().catch(err => { imageCache.delete(key); throw err; }));
  }
  return imageCache.get(key);
}
