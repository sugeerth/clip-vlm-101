"""Fusing two embeddings into one vector — the concatenation trick.

pipeline: image_emb + text_emb ──► [fusion] ──► fused_emb (1024-d)

Every image row has two unit vectors: image_emb (what it LOOKS like) and
text_emb (what its caption/tags MEAN). Gluing them end-to-end gives one
vector that carries both signals:

    fused_emb   = [ image_emb ; text_emb ] / √2
    fused query = [ q ; q ] / √2              (the 512-d query, duplicated)

Dividing by √2 keeps both unit-length, so a single dot product does it all:

    fused_emb · fused_query(q) = (image_emb·q + text_emb·q) / 2
                               = the AVERAGE of visual and semantic similarity
"""
import numpy as np


def fuse(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Concatenate an image vector and a text vector into one unit vector."""
    fused = np.concatenate([image_emb, text_emb]) / np.sqrt(2)
    return fused.astype(np.float32)  # keep the whole pipeline in float32


def fused_query(query_emb: np.ndarray) -> np.ndarray:
    """Lift a 512-d query into fused space by pairing it with itself."""
    return fuse(query_emb, query_emb)
