"""The exact answer, at three rungs of the storage hierarchy: memmap → RAM → GPU.

pipeline: query ──► [scan] ──► exact top-k (the baseline everything is judged by)

Every approximate method in the engine (ivf.py, intq.py, opq.py, binary.py) is
measured against this. The scan itself is one dot product per stored vector; what
changes with scale is only WHERE the bytes live:

  scan       f16 off disk, chunked — bounded memory, but paging + f16→f32 casting
  scan_ram   the same top-k once the tower is promoted to f32 RAM — one matmul
  scan_gpu   the same again on the GPU (unified memory on Apple silicon)

Fused search is just two towers at weight 1/2, summed — fusion.py's identity,
never materialising a 1024-d matrix.
"""
import numpy as np

from store import CHUNK


def topk_merge(ids, scores, new_ids, new_scores, k):
    """Keep the best k of (old ∪ new) — how a chunked scan stays O(k) memory."""
    ids = np.concatenate([ids, new_ids])
    scores = np.concatenate([scores, new_scores])
    keep = np.argpartition(scores, -min(k, len(scores)))[-k:]
    order = keep[np.argsort(scores[keep])[::-1]]
    return ids[order], scores[order]


def scan(mats, q, k=10, rows=None, chunk=CHUNK):
    """Exact top-k over one pass of the memmapped matrices.

    mats = [(matrix, weight), ...] — fused search is two towers at weight 1/2
    each, summed per chunk: fusion.py's identity, never a 1024-d matrix."""
    rows = rows if rows is not None else len(mats[0][0])
    ids = np.empty(0, dtype=np.int64)
    scores = np.empty(0, dtype=np.float32)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        s = np.zeros(hi - lo, dtype=np.float32)
        for M, w in mats:
            s += w * (np.asarray(M[lo:hi], dtype=np.float32) @ q)
        ids, scores = topk_merge(ids, scores, np.arange(lo, hi), s, k)
    return ids, scores


def scan_ram(mats, q, k=10):
    """The same exact top-k as scan(), for matrices already promoted to f32
    RAM: one matmul per tower, one argpartition. No chunks — RAM needs none."""
    s = mats[0][1] * (mats[0][0] @ q)
    for M, w in mats[1:]:
        s += w * (M @ q)
    keep = np.argpartition(s, -min(k, len(s)))[-k:]
    order = keep[np.argsort(s[keep])[::-1]]
    return order.astype(np.int64), s[order]


def gpu_promote(mats):
    """One more rung: RAM -> the GPU. On Apple silicon this is nearly free
    (unified memory), and the exact scan becomes a device matmul + topk.
    Returns None when there's no usable device."""
    import torch
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        return None
    return dev, [(torch.from_numpy(np.ascontiguousarray(M)).to(dev), w)
                 for M, w in mats]


def scan_gpu(gpu, q, k=10):
    """scan_ram's device twin — identical answers, another order of magnitude."""
    import torch
    dev, gmats = gpu
    qt = torch.from_numpy(np.ascontiguousarray(q)).to(dev)
    s = gmats[0][1] * (gmats[0][0] @ qt)
    for M, w in gmats[1:]:
        s = s + w * (M @ qt)
    scores, ids = torch.topk(s, min(k, s.shape[0]))
    return ids.cpu().numpy().astype(np.int64), scores.cpu().numpy()


def ensemble_query(embed, text):
    """ensemble.py's lesson on the QUERY side: average two phrasings of the
    same intent, renormalise. Worth a few recall points for free."""
    E = embed([text, f"a photo of {text}"])
    v = E.mean(axis=0)
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)
