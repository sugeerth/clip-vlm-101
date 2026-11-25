# A million rows — where the database's shape starts to matter

*Real measured recall and latency for exact scan, IVF, int8 and OPQ at 690k vectors.*

[← README](../README.md) · [all chapters](README.md)

Every other chapter runs on the committed 14-image gallery. This one runs the
same design on a real corpus and measures what it costs — recall and
milliseconds, per method, in two tables.

Read those tables with two caveats in mind. The corpus is 690,000 rows, not a
million. And every recall number is scored on *self-queries* — a vector
searching for itself — which is the hardest case a compressed index faces.
Both are restated below where they matter.

## The corpus: 690,000 real dishes

`scale.py` grows the SAME
design to **690,000 real dishes** (of the dataset's 1M+, the slice this laptop
has ingested so far) — [Qdrant's public Wolt food
dataset](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings),
whose image embeddings were precomputed with the *same* clip-ViT-B-32
checkpoint this repo uses — and computes every text embedding itself,
one dish name at a time through `embedder.py` (~1,900 names/s on MPS).
Same 512-d space; the demo's text tower queries it directly. The shape
lessons below are identical at any scale.

## The layout: records in SQLite, vectors in a memmap

What changes at scale is the *layout*, not the math. `db.py`'s
vector-BLOBs-in-SQLite is perfect at 14 rows and wrong at 690k (a scan would
decode 690k BLOBs per query), so `scale.py` splits the two jobs the
way real vector stores do:

```
data/scale.sqlite      the RECORDS - name, caption, cafe, url; row id i = matrix row i
data/img_emb_f16.npy   the SCAN    - packed (690k x 512) float16 per tower, memory-
data/txt_emb_f16.npy               mapped: chunked matrix @ vector, no BLOB decodes
```

## Four searches, one flag apart

Four searches, each a concept the repo teaches: **brute** (`search.py`'s scan,
chunked), **ivf** (`ann.py`'s index trained on a 100k sample), **int8**
(`quantize.py`'s lesson from [chapter 2](02-understanding-clip.md), now with a
*per-dimension* scale), and **OPQ**
(`opq.py` — a learned rotation + product quantization, **64 bytes/vector**).
The last two are compression: coarse on their own, but a short *exact re-rank*
of their top few hundred candidates recovers the truth. Fused search uses
`fusion.py`'s identity — *(image·q + text·q) / 2*, the one
[chapter 1](01-the-core-pipeline.md) derives — so no 1024-d matrix exists.

```bash
python3 scale.py ingest                                  # data/part-*.parquet -> the DB
python3 scale.py bench                                    # builds int8+OPQ+ivf, honest table
python3 scale.py search "quattro formaggi pizza" --method opq   # 64B coarse -> exact re-rank
python3 scale.py search "ramen" --method ivf --probes 16 # scan ~2%, keep ~all of it
python3 scale.py serve                                    # live at localhost:8071
```

`serve` is the storage-hierarchy lesson: the memmap scan was paging + casting a
gigabyte, not math. Promote both towers to f32 RAM once and the same exact scan
is one matmul — **18.6 ms for the truth, 12.3 ms on the GPU, 2 ms with ivf** —
behind a one-file UI (fused ranking, queries ensembled over two phrasings like
`ensemble.py`, every answer printing what it cost and scanned).

## Compression vs. accuracy

recall@{1,10,100} against the exact top-k,
self-queries (the hardest case), measured by `scale.py bench` on 690k:

| method | bytes/vec | recall@1 | recall@10 | recall@100 |
| --- | --- | --- | --- | --- |
| brute force, f32 (the truth) | 2048 | 1.00 | 1.00 | 1.00 |
| int8, per-dimension | 512 | 0.60 | 0.86 | 0.93 |
| int8 -> exact re-rank | 512 | 0.66 | **0.91** | 0.93 |
| OPQ, coarse | **64** | 0.62 | 0.59 | 0.62 |
| **OPQ -> exact re-rank** | **64** | 0.68 | **0.92** | 0.92 |

The bottom row is the headline: **64 bytes a vector — 32x smaller than float32 —
and recall@10 of 0.92**. Two lessons the 14-image gallery could never teach: a
*per-dimension* int8 scale beats one global scale on CLIP's outlier dimensions
(0.86 vs 0.68), and **OPQ's learned rotation earns ~10 recall points over plain
product quantization for free** (0.59 coarse vs 0.44) — same 64 bytes, a better
split. Coarse codes are a filter; the exact re-rank of their shortlist is the
answer.

Those five rows are a subset. `scale.py bench` prints fourteen, including a
**binary 1-bit (popcount)** tier and the full **bin→OPQ→int8→exact cascade**, and
ivf at six probe counts (1, 2, 4, 8, 16, 32) — plus a brute-force row on the GPU
when one is available. The GPU and two of the ivf rows are in the next table; the binary and cascade rows
are not reproduced on this page, and no numbers are guessed for them — run
`python3 scale.py bench` to see them, or query them directly with
`python3 scale.py search "..." --method binary` and `--method cascade`.
[Chapter 0](00-every-file.md) lists which files they live in.

## Speed — where the bytes live

The other axis: the storage hierarchy, exact throughout, and ivf
as the recall *dial*:

| where the bytes live | ms / query | recall@10 |
| --- | --- | --- |
| f16 memmap (paged off disk) | ~110 | 1.00 — the truth |
| f32 in RAM (`serve`) | **18.6** | 1.00 — the truth |
| f32 on the GPU / MPS (`serve`) | **12.3** | 1.00 — the truth |
| ivf, 8 probes | 2.0 | 0.86 (0.9% scanned) |
| ivf, 16 probes | 4.8 | 0.90 (1.7% scanned) |

ivf recall *plateaus* — a few true neighbours live in cells no nearby probe
visits — so 16 probes keep 0.90 while touching under 2% of the rows.

## The same two stages, in a browser tab

The part you can touch: **[the scale page](https://sugeerth.github.io/clip-vlm-101/scale.html)
searches a 100,000-dish slice live in your browser** — `pq.py` OPQ-packs every
vector to **64 bytes** (32x smaller than float32), so the base index is an ~8 MB
pack on GitHub Pages — with an 18 MB metadata file fetched alongside it, ~26 MB
lands before the first result — and `js/pq.js` scores it with pure table lookups (ADC, the
query rotated by the shipped rotation) in a few ms per keystroke. Coarse recall
is ~0.62; tap **+ precision** and it streams a 51 MB int8 tier and re-ranks on
top for ~0.91 — the same two-stage architecture as the local engine, in a tab.
Every number is measured and printed on the page, not promised.

## What CI can check without the corpus

The 4 GB of parquet and the built database stay out of git; CI runs
`scale.py selftest`, `pq.py selftest` and `opq.py selftest` — the same
scan/ivf/int8/OPQ machinery on synthetic clustered vectors, no model, no
downloads — plus `test_pq.mjs`, which pins the JS twin (rotation + two-stage
re-rank) to the Python math. The full story with the measured numbers:
**[the scale report](https://sugeerth.github.io/clip-vlm-101/scale.html)**.

---
← [README](../README.md) · [all chapters](README.md) · prev: [Recommendation — two towers and a ranker that learns you](05-recommendation.md) · next: [Grow the gallery — and swap in a newer model](07-grow-your-gallery.md)
