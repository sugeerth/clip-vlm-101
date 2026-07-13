"""The model registry: newer, stronger CLIP-family brains, one key away.

pipeline: --model KEY ──► [models] ──► the right checkpoint, encoded RIGHT

Every file in this repo is model-agnostic math — cosines between unit
vectors — so upgrading the model is just swapping the encoder. But two
things genuinely differ per family, and both are silent-quality-loss traps
if ignored:

  padding    CLIP pads text to the longest in the batch (padding=True).
             SigLIP/SigLIP 2 were TRAINED with padding="max_length"
             (64 tokens) — pad-to-longest quietly wrecks their embeddings.
  scoring    CLIP ships one learned temperature (exp(logit_scale) ~ 100).
             SigLIP is sigmoid-trained and ships logit_scale AND
             logit_bias — its per-tag probabilities are calibrated
             directly: sigmoid(scale * cosine + bias), no neutral prompt.

And the repo's own iron law still holds: embeddings from different models
NEVER mix. One database, one model — item_tower.py stamps every row.
"""

_CLIP_TEXT = {"padding": True, "truncation": True}
_SIGLIP_TEXT = {"padding": "max_length", "truncation": True, "max_length": 64}

MODELS = {
    "clip-b32": {
        "hf_id": "openai/clip-vit-base-patch32", "dim": 512, "kind": "clip",
        "text_kwargs": _CLIP_TEXT,
        "note": "2021 baseline — matches the committed gallery (~63% IN-1k 0-shot)",
    },
    "clip-l14": {
        "hf_id": "openai/clip-vit-large-patch14", "dim": 768, "kind": "clip",
        "text_kwargs": _CLIP_TEXT,
        "note": "bigger CLIP (~75%), ~1.7 GB",
    },
    "siglip2-base": {
        "hf_id": "google/siglip2-base-patch16-224", "dim": 768, "kind": "siglip",
        "text_kwargs": _SIGLIP_TEXT,
        "note": "2025, sigmoid-trained, multilingual (~78%)",
    },
    "siglip2-384": {
        "hf_id": "google/siglip2-base-patch16-384", "dim": 768, "kind": "siglip",
        "text_kwargs": _SIGLIP_TEXT,
        "note": "same brain, 384px eyes — better on small detail",
    },
}

DEFAULT = "clip-b32"


def resolve(key_or_hf_id: str) -> dict:
    """Registry key, full HF id, or any unknown id (assumed CLIP-style)."""
    if key_or_hf_id in MODELS:
        return dict(MODELS[key_or_hf_id])
    for spec in MODELS.values():
        if spec["hf_id"] == key_or_hf_id:
            return dict(spec)
    return {"hf_id": key_or_hf_id, "dim": None, "kind": "clip",
            "text_kwargs": _CLIP_TEXT, "note": "unregistered — treated as CLIP"}
