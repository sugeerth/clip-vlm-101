# CLIP VLM 101 — vision-language embeddings, from zero to search

**Live demo (runs entirely in your browser):**
https://sugeerth.github.io/clip-vlm-101/ — CLIP·search: one box, real
inference, results. The full explorable walkthrough lives at
[/explore.html](https://sugeerth.github.io/clip-vlm-101/explore.html).

A deliberately tiny, readable pipeline that shows how a CLIP-style
vision-language model turns **images + prompt templates** into **embeddings**,
stores them in a **database**, and answers **searches** — then takes it one
step further: **dynamic multi-label meta tags**, a **self-critiquing embedding
agent** that only publishes features it can defend, and an **item tower**
ready for two-tower recommendation, plus the serving half that turns likes
into recommendations. Fourteen tiny pipeline files plus seven standalone math
lessons — one concept each — a sample downloader, and a smoke test.
Standard-library SQLite, no frameworks. Read it top to bottom in 20 minutes,
then swap in your own images.

## The whole idea in one picture

```
 your image ──► vision encoder ──► image_emb (512 floats: what it LOOKS like)
                                        │
 "a photo of a {tag}"  ×  vocabulary    │  zero-shot tagging
        └──► text encoder ──► tag scores┘──► meta tags: [cat, pet, portrait…]
                                                  │
 "a photo of {tags}" caption ──► text encoder ──► text_emb (512: what it MEANS)
                                                  │
             fused_emb (1024) = [ image_emb ; text_emb ] / √2   ◄── concatenation
                                                  │
                                     SQLite row: path · caption · tags · 3 vectors
                                                  │
 search: "a fluffy animal" ──► text encoder ──► dot products ──► top-k results
```

And the recommendation-ready extension on top of the same vectors:

```
 image_emb  ×  "a photo of a {tag}" vs "a photo"   per-tag sigmoid (labels.py)
      └──► DYNAMIC multi-label set {cat: 0.99, pet: 0.87}  — sized by the image
                          │
        agent loop (agent.py): PROPOSE draft ──► CRITIC checks it
                          │        ▲    aligned? confident?│
                          │        └──── no: next template ┘ yes ──► publish
                          │
        item_tower.py: items.sqlite — verified item embeddings, OFFLINE
                          │
        user_tower.py: likes ─► unit(mean) ─► user_vec · every item = recs
```

Both encoders project into the **same** vector space, so an image of a cat and
the sentence "a photo of a cat" land close together. Everything here — tagging,
captioning, search — is just cosine similarity between unit vectors, which is a
single dot product.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download_samples.py        # or drop your own images in images/
.venv/bin/python features.py images/cat.jpg # one image → meta tags + embeddings
.venv/bin/python ingest.py images/*.jpg     # embed + auto-tag + store in SQLite
.venv/bin/python search.py "a fluffy animal"
.venv/bin/python hermes.py "a fluffy animal"  # the agentic version, with a trace
.venv/bin/python search.py --image images/cat.jpg   # image-to-image search
.venv/bin/python item_tower.py images/*.jpg  # agent-verified item embeddings
.venv/bin/python user_tower.py images/cat.jpg images/dog.jpg  # likes → recs
.venv/bin/python eval.py images/*.jpg        # measure: template vs ensemble
```

First run downloads the CLIP model (~600 MB). Device is auto-detected:
CUDA → Apple Silicon MPS → CPU (all work; this model is small).

## Given an image → meta tags + embeddings

`features.py` is the one-call API if you just want database-ready features
for your own project:

```python
from features import FeatureExtractor

fx = FeatureExtractor()            # loads CLIP once
fx.embed("photo.jpg")              # IMAGE-ONLY: one (512,) unit vector — the
                                   #   text tower is never touched
record = fx.extract("photo.jpg")   # full record: meta tags + all three vectors
record["tags"]                     # ['cat', 'pet', 'animal', 'portrait', 'wildlife']
record["caption"]                  # 'a photo of cat, pet, animal, portrait, wildlife'
```

Batch it from the shell — one prompt template, many images:

```bash
.venv/bin/python features.py images/*.jpg                 # meta tags per image
.venv/bin/python features.py images/*.jpg --image-only    # embeddings only
.venv/bin/python features.py images/*.jpg --tag-template "a drawing of a {tag}"
```

**Exact dimensions** (model `openai/clip-vit-base-patch32`; these are what
`db.py` stores as BLOBs — `export_web.py` ships `image_emb` and `text_emb`
to the web demo, which recomputes the fused score as their average):

| vector | shape | dtype | unit-length | bytes as BLOB | meaning |
|---|---|---|---|---|---|
| `image_emb` | `(512,)` | float32 | yes | 2 048 | what the image *looks* like (vision tower) |
| `text_emb` | `(512,)` | float32 | yes | 2 048 | what its caption/tags *mean* (text tower) |
| `fused_emb` | `(1024,)` | float32 | yes | 4 096 | `[image_emb ; text_emb] / √2` — both signals |

Both towers project into the same 512-d space, which is why one image and
one sentence can be compared with a plain dot product (cosine similarity,
range −1…1). Or from the shell: `python3 features.py photo.jpg` prints the
tags and all three vectors' shapes; `--json` dumps the full record.

## The files — one concept each

Each file's docstring starts with a `pipeline:` line showing where it sits.
Suggested reading order:

| file | lines | the one concept it teaches |
|---|---|---|
| `templates.py` | ~60 | prompt templates: sentences with holes, filled per tag |
| `embedder.py` | ~55 | CLIP → unit-length 512-d vectors for images AND text |
| `tagger.py` | ~25 | zero-shot meta tags = dot products + argsort, no training |
| `labels.py` | ~55 | **multi-label**: per-tag sigmoid vs a neutral prompt → dynamic label sets |
| `ensemble.py` | ~60 | **prompt ensembling**: average many templates per tag, +3.5% for free |
| `fusion.py` | ~30 | the concatenation: `[image ; text] / √2`, and why it works |
| `features.py` | ~130 | **the one-call API**: image-only `embed()`, batch tags, full records |
| `agent.py` | ~135 | **the agent**: propose ⇄ critique loop, publishes only when satisfied |
| `db.py` | ~100 | vectors as float32 BLOBs in plain SQLite (+ the JSON gallery loader) |
| `ingest.py` | ~65 | *composition*: batch `features.extract_batch` over files → store |
| `item_tower.py` | ~120 | **two-tower recsys**: agent-verified item embeddings, offline |
| `user_tower.py` | ~65 | **the serving half**: mean-pooled likes → recommendations, one matmul |
| `eval.py` | ~100 | **the benchmark**: top-1/top-5 hit rates — prove an optimization helps |
| `search.py` | ~90 | text / image / fused retrieval with dot products |
| `hermes.py` | ~110 | **the agentic searcher**: propose phrasings ⇄ critique margins ⇄ refine |
| `export_web.py` | ~90 | dump the DB to `docs/db.json` + the 2-D PCA map coords |

Five standalone lessons build on the stored vectors — every one runs
**without the model** via `--json docs/db.json` (real committed embeddings),
and the lessons also run **live on the [explorable page](https://sugeerth.github.io/clip-vlm-101/explore.html#lessons)**,
where `docs/js/lessons.js` reproduces the same numbers in JavaScript (CI
holds both languages to it):

| lesson | lines | the one concept it teaches |
|---|---|---|
| `temperature.py` | ~50 | softmax + CLIP's learned logit scale: scores → probabilities |
| `similarity.py` | ~90 | the N×N image matrix + the modality gap (why scales don't mix) |
| `retrieval_eval.py` | ~80 | retrieval evaluation: leave-one-out precision@k and MRR |
| `arithmetic.py` | ~90 | vector algebra: `cat + dog − apple`, centroids, renormalize |
| `quantize.py` | ~75 | int8 scalar quantization: 4× smaller, measure the damage |
| `similarity.py --centered` | +25 | closing the gap: center each modality, margin widens ~3× |
| `ann.py` | ~110 | IVF (what vector DBs do): scan ~2%, keep ~75% of the truth |

The browser demo mirrors the same pipeline in `docs/js/` with **matching
module names**: `templates.js` ↔ `templates.py`, `clip.js` ↔ `embedder.py`,
`rank.js` ↔ `tagger.py`+`fusion.py`+`search.py`, `labels.js` ↔ `labels.py`,
`agent.js` ↔ `agent.py`, `recsys.js` ↔ `user_tower.py`; `viz.js`, `motion.js`,
and `tour.js` are page-only (the matrix, map and strips, the animation
helpers, the guided tour), and `app.js` wires everything together. Read a
Python file, then its twin — same pipeline, two languages.

## Why concatenate embeddings?

`fusion.py` is the whole answer in 30 lines:
`fused_emb = [image_emb ; text_emb] / √2` keeps the fused vector unit-length,
so a dot product against a duplicated query `[q ; q] / √2` equals the
**average of the visual similarity and the semantic (tag/caption) similarity**.
One vector, one dot product, both signals — no extra model, no re-ranking step.
`search.py --mode image|text|fused` lets you compare all three behaviors.

## Multi-class vs multi-label: dynamic meta tags

`tagger.py` is multi-**class**: argsort the scores, keep the top k — every
image gets exactly 5 tags whether it contains one thing or twelve.
`labels.py` is multi-**label**: each tag answers its own yes/no question,
so the label set is **dynamic**, sized by the image.

The whole trick is one comparison per tag. For "cat", CLIP scores which
sentence describes the image better — `"a photo of a cat"` (tag prompt) or
`"a photo"` (the same sentence with no tag). A two-way softmax over the two
scores collapses to a sigmoid on their gap: an independent probability per
tag. Keep everything above 0.5:

```python
import labels
labels.multi_label(image_emb, tag_embs, neutral_emb, vocabulary)
# {'cat': 0.99, 'pet': 0.87}          ← 2 labels for a simple image
# {'pizza': 0.98, 'food': 0.91, ...}  ← more labels for a busier one
```

New labels are new words in `templates.TAG_VOCABULARY` — still no training.

## The embedding agent: propose ⇄ critique, publish only when satisfied

`agent.py` doesn't take the model's first answer on faith. It drafts
features, **checks its own work**, and refuses to publish a record it
can't defend:

```
round 1..n   PROPOSER  one prompt template → labels + caption + vectors
each round   CRITIC    aligned?   caption embedding · image embedding ≥ 0.20
                       confident? accepted labels average ≥ 0.60 probability
the edge     satisfied → stop, publish   |   not → next round, next template
```

Each round is a different proposer — a different phrasing from
`templates.TEMPLATE_POOL`. The critic's checks are two dot products a human
reviewer would approve of: *does the caption actually point back at the
image?* and *how sure are the labels?* If no proposal passes, the best
draft is returned **unpublished** and the caller decides.

This is the same shape as a LangGraph state graph — two nodes (propose,
critique) and one conditional edge (satisfied?) — hand-rolled in ~40 lines
of plain Python so every decision stays readable. Swap in LangGraph (or any
agent framework) when you need checkpointing, retries, or parallel fan-out
across many workers; the loop's logic transfers unchanged.

## Two-tower recommendation: the item side, done offline

A two-tower recommender scores `(user, item)` pairs with a single dot
product between two encoders that share a vector space. `item_tower.py` is
the **item tower**: it runs the agent over your catalog images **offline**,
and stores only critic-approved records in `items.sqlite` — labels with
probabilities, caption, model version, timestamp, and the 1024-d fused
embedding as the item vector.

```bash
.venv/bin/python item_tower.py images/*.jpg   # build the tower, offline
```

```python
import item_tower
con = item_tower.connect()
paths, matrix = item_tower.item_matrix(con)   # (n, 1024) — the whole tower
scores = matrix @ user_vec                    # rank every item: one matmul
```

`user_tower.py` is the serving half: the simplest honest user tower is the
renormalized **mean of the liked items' vectors** (the same mean-pooling
production systems bootstrap with), and recommending is `item_matrix @
user_vec` — one matrix multiply, no image model at serving time. Swap the
mean for a trained model later; nothing else changes. That's the point of
two towers: the expensive half is finished ahead of time. Try it live in
the demo's **Recommend** section — it runs on the stored vectors with no
model download at all.

## Hermes: the agent on the read path

`agent.py` audits the WRITE path — an image's features must satisfy a critic
before they are stored. **Hermes** (`hermes.py` / `js/hermes.js`) is its twin
on the READ path: your query is treated as a draft, not a command.

```
 query ──► PROPOSE 4 phrasings ("cat", "a photo of cat", …)
              │            each one is a different question to the model
           CRITIQUE by retrieval margin: top1 − mean(rest of top-k)
              │            decisive phrasing? ──yes──► publish that ranking
              └──────────── no ──► REFINE: ensemble the phrasings, publish
```

The live search box at https://sugeerth.github.io/clip-vlm-101/ runs Hermes
on every text query — the muted "🪽 hermes chose …" line under the results
expands into the full trace of what it tried and why.

## Understanding CLIP — and squeezing more out of it

CLIP was trained contrastively on 400M (image, caption) pairs: pull each
image toward its own caption, push it from everyone else's. Three facts
fall out of that training, and every optimization here exploits one of them:

1. **Both towers share one space** → similarity is a dot product, so
   ranking a whole database is a single matrix multiply (`search.py`,
   `item_tower.item_matrix`). No index needed until ~1M items.
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
deployment — the browser demo runs the same checkpoint in int8 (`q8`),
4× smaller with near-identical rankings; and **precompute the item side**
entirely (`item_tower.py`) so serving never loads the model at all.

## Custom prompt templates

The template is the interface to the model. Change it and re-ingest:

```bash
.venv/bin/python ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
.venv/bin/python ingest.py holiday.png --caption "me hiking in Yosemite"
```

Add your own candidate tags in `templates.py` (`TAG_VOCABULARY`) — zero-shot
tagging means new tags need **no training**, just new words.

## The web demo

`docs/` is a static GitHub Pages site — an explorable explanation in the
[distill.pub](https://distill.pub) tradition. `export_web.py` dumps the SQLite
gallery to `docs/db.json`; the page then runs the *same* CLIP model in your
browser via [transformers.js](https://huggingface.co/docs/transformers.js).
Nothing you type or upload leaves your machine. Highlights:

- **The contrastive matrix** — all 14 images × all 14 captions as one live
  similarity heatmap, computed from the stored vectors (no model download
  needed). The bright diagonal *is* the training objective.
- **The embedding map** — 2-D PCA of every image embedding; hover for each
  image's true nearest neighbours in 512-d, upload to see yours land on it.
- **The Lab** — pick or upload an image, then compare multi-class top-5
  against the dynamic multi-label set with a draggable threshold, and run
  the embedding agent round by round, critic verdicts and all.
- **Search** — text / image / fused retrieval with the same dot products as
  `search.py`, plus a keyword fallback when the model can't load.

Light and dark themes are both hand-tuned (toggle in the header), and every
JS module mirrors its Python twin by name.

## Reproduce these numbers (no model, no downloads)

`docs/db.json` ships the real embeddings of the 14 sample images, so every
lesson runs on committed data — and CI re-runs all of them on every push:

| command | the number you should see |
|---|---|
| `python3 temperature.py` | top hit 7.9% at scale 1 → **99.7%** at CLIP's learned scale 100 |
| `python3 similarity.py --json docs/db.json` | the modality gap: image·images **+0.57** vs image·own-caption **+0.29** |
| `python3 retrieval_eval.py --json docs/db.json` | image mode **P@1 = 0.857**, MRR ≈ 0.88 |
| `python3 arithmetic.py --centroid animal --json docs/db.json` | top 4 = exactly the 4 animal images |
| `python3 quantize.py --json docs/db.json` | 4× smaller, **39/42** top-3 neighbor slots unchanged |
| `python3 similarity.py --json docs/db.json --centered` | own-caption margin **+0.120 → +0.388** after centering |
| `python3 ann.py` | probes 1: recall **0.75** scanning **1.8%**; probes 8: **0.94** at 12.8% |

(The numbers are pinned to the committed sample gallery; re-exporting your
own gallery changes them — that's the point.)

## Tests

```bash
.venv/bin/python test_smoke.py   # templates, tagging, fusion math, PCA, DB — needs only numpy
```

The smoke test stubs the CLIP encoder, so it runs without torch/transformers
installed (`pip install numpy` is enough) — that's also exactly what CI does
on every push (`.github/workflows/test.yml`).

## Sample image credits

Sample photos are downloaded from [Wikimedia Commons](https://commons.wikimedia.org)
(freely licensed; see each file's Commons page, e.g.
`commons.wikimedia.org/wiki/File:Cat03.jpg`). The model is
[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

## License

MIT — take it, fork it, rebuild it with your own images.
