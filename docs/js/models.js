// The model registry — the page's selectable "brains".
//
// Everything in this repo is model-agnostic math (cosines between unit
// vectors), so upgrading the model is just swapping the encoder — BUT
// embeddings from different models NEVER mix (the repo's own lesson).
// The committed gallery vectors were computed with CLIP B/32; choosing
// another brain re-embeds the gallery live, in the browser, before use.
//
// Registry facts verified against the transformers.js source and each
// checkpoint's model card (July 2026):
//   - MobileCLIP (Apple '24) ships as model_type "clip": the same
//     CLIPTextModelWithProjection / CLIPVisionModelWithProjection classes
//     and .text_embeds/.image_embeds fields — but its text tower REQUIRES
//     padding:'max_length' (77 tokens), not padding:true.
//   - SigLIP 2 (Google '25) has no projection heads: SiglipTextModel /
//     SiglipVisionModel, embeddings in .pooler_output (768-d), tokenizer
//     padded to exactly 64 tokens. Sigmoid-trained: fine for cosine
//     ranking, different calibration for probabilities.
export const MODELS = {
  'clip-b32': {
    id: 'Xenova/clip-vit-base-patch32', dim: 512, kind: 'clip',
    padding: true,
    label: 'CLIP B/32 · 2021 · baseline', size: '~150 MB', accuracy: '63%',
  },
  'mobileclip-s0': {
    id: 'Xenova/mobileclip_s0', dim: 512, kind: 'clip',
    padding: 'max_length',
    label: 'MobileCLIP S0 · 2024 · small + fast', size: '~55 MB', accuracy: '68%',
  },
  'mobileclip-blt': {
    id: 'Xenova/mobileclip_blt', dim: 512, kind: 'clip',
    padding: 'max_length',
    label: 'MobileCLIP B-LT · 2024 · best value', size: '~150 MB', accuracy: '77%',
  },
  'siglip2-b16': {
    id: 'onnx-community/siglip2-base-patch16-224-ONNX', dim: 768, kind: 'siglip',
    padding: 'max_length', maxLength: 64,
    label: 'SigLIP 2 · 2025 · strongest + multilingual', size: '~380 MB', accuracy: '78%',
  },
};

// The default matches the PRECOMPUTED gallery vectors shipped in db.json.
export const DEFAULT_MODEL = 'clip-b32';
