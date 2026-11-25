# Understanding CLIP — and squeezing more out of it

*The seven model-free lessons: temperature, the modality gap, vector arithmetic, quantization, IVF.*

[← README](../README.md) · [all chapters](README.md)

`docs/db.json` ships the real embeddings of the 14 sample images. Most
lessons here read that file; `ann.py` runs on synthetic vectors unless you
pass `--json`, and `temperature.py` reads the committed gallery directly, so
it needs no flag. Either way, every lesson runs with no model download and no
torch install — and CI re-runs all of them on every push. So: three facts
about CLIP first, then the seven standalone lessons that prove each one, with
the exact number you should see on your screen.

## Three facts that fall out of the training

CLIP was trained contrastively on 400M (image, caption) pairs: pull each
image toward its own caption, push it from everyone else's. Three facts
fall out of that training, and every optimization here exploits one of them:

1. **Both towers share one space** → similarity is a dot product, so
   ranking a whole database is a single matrix multiply (`search.py`,
   `item_tower.item_matrix`). No index needed until ~1M items — past that,
   see [A million rows](06-a-million-rows.md).
2. **The prompt is the classifier** → better phrasing = better accuracy,
   for free. `ensemble.py` embeds each tag through many templates and
   averages the unit vectors: phrasing noise cancels, meaning stays. The
   CLIP paper's 80-template ensemble gains **+3.5%** ImageNet zero-shot
   accuracy (~5% with prompt engineering); it costs one extra text batch,
   once. Try it: `python3 features.py images/*.jpg --ensemble`
3. **The model ships a learned temperature** (logit scale ≈ 100) → raw
   cosines live in a narrow band (~0.2–0.35 for matches); multiply by the
   scale before a softmax/sigmoid or every probability collapses toward
   0.5. `labels.py` does exactly this.

Speed levers, in the order worth pulling: **batch** your inputs — one
forward pass per tower per chunk, not two round-trips per image
(`features.extract_batch`, used by `ingest.py`); **cache** anything text —
the vocabulary matrix never changes between runs; **quantize** for
deployment — the [browser demo](08-the-web-demo.md) runs the same
checkpoint in int8 (`q8`), 4× smaller with near-identical rankings; and
**precompute the item side** entirely (`item_tower.py`) so serving never
loads the model at all.

## Reproduce these numbers (no model, no downloads)

Every lesson below runs with no model download and no torch install — `ann.py`
on synthetic vectors, the rest on the committed `docs/db.json` embeddings — and
CI re-runs all of them on every push. The seven standalone lessons, one concept
each:

| command | the number you should see |
|---|---|
| `python3 temperature.py` | top hit 7.9% at scale 1 → **99.7%** at CLIP's learned scale 100 |
| `python3 similarity.py --json docs/db.json` | the modality gap: image·images **+0.57** vs image·own-caption **+0.29** |
| `python3 retrieval_eval.py --json docs/db.json` | image mode **P@1 = 0.857**, MRR ≈ 0.88 |
| `python3 arithmetic.py --centroid animal --json docs/db.json` | top 4 = exactly the 4 animal images |
| `python3 quantize.py --json docs/db.json` | 4× smaller, **39/42** top-3 neighbor slots unchanged |
| `python3 similarity.py --json docs/db.json --centered` | own-caption margin **+0.120 → +0.388** after centering |
| `python3 ann.py` | probes 1: recall **0.75** scanning **1.8%**; probes 8: **0.94** at 12.8% |

The rest of the repo runs the same way, on the same committed data. Later
chapters explain what these verdicts mean — [Agents](03-agents.md) for
`hermes.py`, `debate.py` and `reason.py`, [Trust](04-trust-and-evaluation.md)
for `conformal.py`, `judge.py`, `trust.py` and `drift.py`,
[Recommendation](05-recommendation.md) for `learn2rank.py`, and
[A million rows](06-a-million-rows.md) for `scale.py` and `opq.py`:

| command | the number you should see |
|---|---|
| `python3 hermes.py --image pizza --json docs/db.json` | the evaluator **rejects** the drifting pass and keeps the honest ranking |
| `python3 learn2rank.py` | after 👍 the tag-sharers / 👎 the rest, `tag_overlap` importance dominates and the parrot lifts above the sunflower |
| `python3 conformal.py --json docs/db.json` | coverage sits **on or above** every target (80% → 84.6%, 90% → 92.3%); the set grows as you demand more confidence |
| `python3 judge.py --json docs/db.json --image images/004_cat.jpg` | the council rules **relevant** for cat→dog and **not relevant** for cat→bicycle; swap in `--image images/000_apple.jpg` for the **hung jury**, where a shared tag and the vision signal disagree |
| `python3 trust.py --json docs/db.json --image images/004_cat.jpg` | cat→dog composes to **high** trust (all four lenses agree); with `--image images/000_apple.jpg` the lenses split and the verdict **abstains** instead of averaging |
| `python3 drift.py --json docs/db.json --selftest` | the detectors escalate **stable → shift → drift → drift** as a growing fraction of the stream goes off-distribution; PSI is monotone in the contamination |
| `python3 debate.py --json docs/db.json --eval` | the panel reaches **consensus on 12/14** top hits and stays **contested on 2** (the tag-fluke cases); ≥1 agent changes its mind on 8/14 |
| `python3 reason.py --json docs/db.json --image images/004_cat.jpg` | every step passes (retrieve→…→trust) → **high trust → "show it as the answer"**; `--image .../000_apple.jpg` breaks at the council → **"show with a caveat"** |
| `python3 scale.py selftest` | chunked scan == naive argsort; ivf keeps most of the truth scanning <½ the rows; int8/OPQ → exact re-rank recovers it |
| `python3 opq.py selftest` | OPQ's learned rotation beats plain PQ on anisotropic data (the whole reason it exists) |

(The numbers are pinned to the committed sample gallery; re-exporting your
own gallery changes them — that's the point.)

---
← [README](../README.md) · [all chapters](README.md) · prev: [The core pipeline — image in, searchable vectors out](01-the-core-pipeline.md) · next: [Agents — propose, critique, argue, explain](03-agents.md)
