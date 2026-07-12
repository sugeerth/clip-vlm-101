# CLIP VLM 101 — vision-language embeddings, from zero to search

**Live demo (runs entirely in your browser):** https://sugeerth.github.io/clip-vlm-101/

A deliberately tiny, readable pipeline that shows how a CLIP-style
vision-language model turns **images + prompt templates** into **embeddings**,
stores them in a **database**, and answers **searches** — then takes it one
step further: **dynamic multi-label meta tags**, a **self-critiquing embedding
agent** that only publishes features it can defend, and an **item tower**
ready for two-tower recommendation. Twelve tiny pipeline files — one concept
each — plus a sample downloader and a smoke test. Standard-library SQLite,
no frameworks. Read it top to bottom in 20 minutes, then swap in your own
images.

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
        item_tower.py: items.sqlite — verified item embeddings, OFFLINE,
        the item half of a two-tower recommender (user tower comes later)
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
.venv/bin/python search.py --image images/cat.jpg   # image-to-image search
.venv/bin/python item_tower.py images/*.jpg  # agent-verified item embeddings
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
| `fusion.py` | ~30 | the concatenation: `[image ; text] / √2`, and why it works |
| `features.py` | ~115 | **the one-call API**: image-only `embed()`, batch tags, full records |
| `agent.py` | ~130 | **the agent**: propose ⇄ critique loop, publishes only when satisfied |
| `db.py` | ~70 | vectors as float32 BLOBs in plain SQLite |
| `ingest.py` | ~45 | *composition*: loop `features.extract` over files → store |
| `item_tower.py` | ~120 | **two-tower recsys**: agent-verified item embeddings, offline |
| `search.py` | ~70 | text / image / fused retrieval with dot products |
| `export_web.py` | ~80 | dump the DB to `docs/db.json` + the 2-D PCA map coords |

The browser demo mirrors the same pipeline in `docs/js/` with **matching
module names**: `templates.js` ↔ `templates.py`, `clip.js` ↔ `embedder.py`,
`rank.js` ↔ `tagger.py`+`fusion.py`+`search.py`, and `app.js` wires them to
the page. Read a Python file, then its twin — same pipeline, two languages.

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

The user tower comes later (train it to map user history into the same
space); serving then never touches an image model — just this table and one
matrix multiply. That's the point of two towers: the expensive half is
finished ahead of time.

## Custom prompt templates

The template is the interface to the model. Change it and re-ingest:

```bash
.venv/bin/python ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
.venv/bin/python ingest.py holiday.png --caption "me hiking in Yosemite"
```

Add your own candidate tags in `templates.py` (`TAG_VOCABULARY`) — zero-shot
tagging means new tags need **no training**, just new words.

## The web demo

`docs/` is a static GitHub Pages site. `export_web.py` dumps the SQLite gallery
to `docs/db.json`; the page then runs the *same* CLIP model in your browser via
[transformers.js](https://huggingface.co/docs/transformers.js) to embed your
query (text or an uploaded image) and ranks with the same dot products as
`search.py`. Nothing you upload leaves your machine.

The page also **visualizes the embeddings**: a 2-D PCA map of every image
embedding (coordinates computed by `export_web.py`; similar images land close
together — click any image for its raw-value fingerprint strip), and uploads
are projected onto the same map live with two dot products.

## Tests

```bash
.venv/bin/python test_smoke.py   # templates, fusion math, DB round-trip — no model needed
```

## Sample image credits

Sample photos are downloaded from [Wikimedia Commons](https://commons.wikimedia.org)
(freely licensed; see each file's Commons page, e.g.
`commons.wikimedia.org/wiki/File:Cat03.jpg`). The model is
[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

## License

MIT — take it, fork it, rebuild it with your own images.
