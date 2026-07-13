"""CLIP wrapper: images and sentences go in, unit-length 512-d vectors come out.

pipeline: image / sentence ──► [embedder] ──► 512-d unit vector

Because both encoders project into the SAME vector space, an image of a cat
and the sentence "a photo of a cat" land close together. Every feature in
this repo (tagging, captioning, search) is just cosine similarity on these
vectors — and since they are unit-length, cosine similarity is a dot product.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import models

MODEL_ID = "openai/clip-vit-base-patch32"  # the default; see models.MODELS


def best_device() -> str:
    """CUDA on NVIDIA, MPS on Apple Silicon, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbedder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        spec = models.resolve(model_id)   # registry key or any HF id
        self.device = device or best_device()
        self.model = AutoModel.from_pretrained(spec["hf_id"]).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(spec["hf_id"])
        self.kind, self.dim = spec["kind"], spec["dim"]
        self._text_kwargs = spec["text_kwargs"]
        # scoring constants belong to the CHECKPOINT — read, never hardcode.
        # CLIP: exp(logit_scale) ~ 100, no bias. SigLIP: scale AND bias, and
        # sigmoid(scale * cosine + bias) is a calibrated per-tag probability.
        self.logit_scale = float(self.model.logit_scale.exp())
        self.logit_bias = float(getattr(self.model, "logit_bias", torch.zeros(1)))

    @torch.no_grad()
    def embed_images(self, paths) -> np.ndarray:
        """List of image paths -> (n, dim) array of unit vectors."""
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return _unit(self.model.get_image_features(**inputs))

    @torch.no_grad()
    def embed_texts(self, texts) -> np.ndarray:
        """List of sentences -> (n, dim) array of unit vectors.

        Padding rule comes from the registry: SigLIP-family models were
        trained with padding="max_length" and silently degrade without it.
        """
        inputs = self.processor(
            text=list(texts), return_tensors="pt", **self._text_kwargs
        ).to(self.device)
        return _unit(self.model.get_text_features(**inputs))


def _unit(t) -> np.ndarray:
    if not isinstance(t, torch.Tensor):
        # transformers v5 returns an output object; pooler_output is the
        # projected 512-d embedding (v4 returned the tensor directly)
        t = t.pooler_output
    return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()
