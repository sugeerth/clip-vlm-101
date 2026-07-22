"""int8, per dimension — quantize.py's lesson, sized for the million.

pipeline: (n, 512) f32 ──► [intq] ──► int8 + per-dim scale ──► same top-k, ½ bytes

quantize.py used ONE shared scale for the whole matrix. That wastes the byte on
CLIP: a few outlier dimensions are ~10× wider than the rest, so one global scale
sizes the byte for the outliers and quantizes everything else to a pinhead. A
PER-DIMENSION scale gives every column its own ±127 range — same one byte per
value, measurably better recall (0.68 → 0.86 at scale). Dequant folds into the
query, so the scan stays integer on the matrix:

    x·q  ≈  Σ_d int8[d]·scale[d]·q[d]  =  int8 · (scale ⊙ q)

On its own int8 reshuffles a tight top-10; its industrial role is stage one of a
two-stage search — nominate a cheap top-100, then re-score those exactly.
"""
import os

import numpy as np

from scan import scan
from store import CHUNK, DATA

I8 = os.path.join(DATA, "img_emb_i8.npy")
I8S = os.path.join(DATA, "img_emb_i8_scale.npy")   # (d+1,): per-dim scale then row count


def build_i8(img, rows, log=print):
    """Write the int8 image tower + its per-dim scales, keyed to the row count
    so a grown corpus rebuilds it. scale[d] = max|col d| / 127."""
    amax = np.zeros(img.shape[1], dtype=np.float32)
    for lo in range(0, rows, CHUNK):
        amax = np.maximum(amax, np.abs(np.asarray(img[lo:lo + CHUNK], np.float32)).max(0))
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    i8 = np.lib.format.open_memmap(I8, mode="w+", dtype=np.int8, shape=(rows, img.shape[1]))
    for lo in range(0, rows, CHUNK):
        hi = min(lo + CHUNK, rows)
        i8[lo:hi] = np.round(np.asarray(img[lo:hi], np.float32) / scale).astype(np.int8)
    i8.flush()
    np.save(I8S, np.concatenate([scale, [float(rows)]]).astype(np.float64))
    log(f"  int8 per-dim: {I8} ({os.path.getsize(I8) / 1e6:,.0f} MB)")


def i8_stale(rows, dim=512):
    if not (os.path.exists(I8) and os.path.exists(I8S)):
        return True
    meta = np.atleast_1d(np.load(I8S))
    return meta.size != dim + 1 or int(meta[-1]) != rows


def load_i8():
    """Returns (i8 memmap, per-dim scale (d,)). Dequant folds into the query."""
    meta = np.load(I8S)
    return np.load(I8, mmap_mode="r"), meta[:-1].astype(np.float32)


def i8_search(i8, scale, q, k=10, rows=None):
    return scan([(i8, 1.0)], (scale * q).astype(np.float32), k=k, rows=rows)


def i8_rerank(i8, scale, img, q, k=10, cand=100, rows=None):
    """int8 nominates a cheap top-`cand`; f16 scores exactly. The workhorse of
    real systems: quantization reshuffles a tight top-k but rarely evicts a true
    neighbour from a top-100, so the exact re-rank recovers the truth."""
    pool = i8_search(i8, scale, q, k=cand, rows=rows)[0]
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top]
