"""Binary quantization: 1 bit per dimension, the cheapest distance a CPU has.

pipeline: (n, 512) f32 ──► [binary] ──► 64 bytes of sign bits ──► popcount filter

quantize.py's int8 was ½ the bytes; OPQ's codes were 1/32. Binary is the same
64 bytes as OPQ but a DIFFERENT tradeoff: keep only sign(x) for each dimension,
packed 8 to a byte. No codebooks, no multiplies — the distance is the number of
sign bits two vectors DISAGREE on, which is popcount(a XOR b), a handful of CPU
instructions. On unit vectors more agreeing signs ⇒ higher cosine, closely
enough to make binary a superb COARSE filter: the widest, cheapest net in the
approximation cascade (millions → a few thousand in one integer pass), handed to
OPQ / int8 / exact to actually order. This is the one rung the million-scale
engine was missing (cascade.py teaches it on the 14-image gallery).

Keyed to the row count it was built for, like every other index here.
"""
import os

import numpy as np

from store import CHUNK, DATA

BIN = os.path.join(DATA, "img_binary.npy")          # (rows, d//8) uint8 — packed sign bits
BINR = os.path.join(DATA, "img_binary_rows.npy")    # the row count it was built for

# popcount lookup: bits set in each byte value 0..255 — the whole distance metric
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def binarize(x):
    """Pack sign bits: (..., d) float -> (..., d//8) uint8. sign>=0 is a 1 bit."""
    return np.packbits(np.asarray(x) >= 0, axis=-1)


def build_binary(img, rows, log=print):
    """Pack every row's sign bits, keyed to the row count. d must be a multiple
    of 8 (512 is); one uint8 holds 8 dimensions, so 512-d -> 64 bytes."""
    d8 = img.shape[1] // 8
    bits = np.lib.format.open_memmap(BIN, mode="w+", dtype=np.uint8, shape=(rows, d8))
    for lo in range(0, rows, CHUNK):
        hi = min(lo + CHUNK, rows)
        bits[lo:hi] = binarize(np.asarray(img[lo:hi], np.float32))
    bits.flush()
    np.save(BINR, np.array([rows], dtype=np.int64))
    log(f"  binary 1-bit: {BIN} ({os.path.getsize(BIN) / 1e6:,.0f} MB)")


def bin_stale(rows):
    if not (os.path.exists(BIN) and os.path.exists(BINR)):
        return True
    return int(np.atleast_1d(np.load(BINR))[0]) != rows


def load_binary():
    return np.load(BIN, mmap_mode="r")


def binary_search(bits, q, k=10, rows=None):
    """Rank by agreeing sign bits (= d − Hamming). One XOR + table-lookup +
    sum per row — no multiplies. Returns (top-k ids, agreeing-bit counts)."""
    rows = min(rows if rows is not None else len(bits), len(bits))   # never index past the packed rows
    qb = binarize(q)                                # the query's own sign bits
    if qb.shape[-1] != bits.shape[1]:               # d must match the packed width
        raise ValueError(f"query packs to {qb.shape[-1]} bytes, index has {bits.shape[1]}")
    d = bits.shape[1] * 8
    ids = np.empty(0, dtype=np.int64)
    scores = np.empty(0, dtype=np.int32)
    for lo in range(0, rows, CHUNK):                # chunked: bounded memory at scale
        hi = min(lo + CHUNK, rows)
        ham = POPCOUNT[np.asarray(bits[lo:hi]) ^ qb].sum(1)   # disagreeing bits
        agree = (d - ham).astype(np.int32)          # more agreement = closer
        cid = np.arange(lo, hi)
        ids = np.concatenate([ids, cid]); scores = np.concatenate([scores, agree])
        keep = np.argpartition(scores, -min(k, len(scores)))[-k:]
        ids, scores = ids[keep], scores[keep]
    order = np.argsort(scores)[::-1]
    return ids[order], scores[order]


def binary_rerank(bits, img, q, k=10, cand=1000, rows=None):
    """Binary nominates a cheap top-`cand`; f16 scores those exactly. The
    coarsest stage of the cascade, standalone."""
    pool = binary_search(bits, q, k=cand, rows=rows)[0]
    e = np.asarray(img[pool], np.float32) @ q
    top = np.argsort(e)[::-1][:k]
    return pool[top], e[top]
