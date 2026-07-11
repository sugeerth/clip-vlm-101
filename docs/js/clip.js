// Mirror of embedder.py — the SAME CLIP checkpoint, running in your browser
// via transformers.js. Models load lazily on first use; the browser caches
// them after that. No DOM here: callers pass onStatus/onProgress callbacks.
import { unit } from './rank.js';

export const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

let hf = null, textP = null, imageP = null;
const lib = async () => (hf ??= await import(CDN));

// (n, d) tensor -> array of n unit-length plain arrays (embedder._unit).
function sliceRows(tensor) {
  const [n, d] = tensor.dims, data = tensor.data, rows = [];
  for (let i = 0; i < n; i++) rows.push(unit(Array.from(data.slice(i * d, (i + 1) * d))));
  return rows;
}

// Both getters memoize the IN-FLIGHT PROMISE so two concurrent first uses
// share one download, and clear it on failure so a retry can succeed.

// resolves to: async (texts) => array of 512-d unit vectors
export function getTextEncoder(onStatus, onProgress) {
  textP ??= (async () => {
    onStatus('Loading CLIP text encoder…');
    const { AutoTokenizer, CLIPTextModelWithProjection } = await lib();
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
    const model = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8', progress_callback: onProgress });
    return async texts => {
      const inputs = tokenizer(texts, { padding: true, truncation: true });
      const { text_embeds } = await model(inputs);
      return sliceRows(text_embeds);
    };
  })().catch(err => { textP = null; throw err; });
  return textP;
}

// resolves to: async (blob) => one 512-d unit vector
export function getImageEncoder(onStatus, onProgress) {
  imageP ??= (async () => {
    onStatus('Loading CLIP vision encoder…');
    const { AutoProcessor, CLIPVisionModelWithProjection, RawImage } = await lib();
    const processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: onProgress });
    const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8', progress_callback: onProgress });
    return async blob => {
      const image = await RawImage.fromBlob(blob);
      const inputs = await processor(image);
      const { image_embeds } = await model(inputs);
      return sliceRows(image_embeds)[0];
    };
  })().catch(err => { imageP = null; throw err; });
  return imageP;
}
