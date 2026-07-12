# CLIP VLM 101 — vision-language embeddings, from zero to search

**Live demo (runs entirely in your browser):** https://sugeerth.github.io/clip-vlm-101/

A deliberately tiny, readable pipeline that shows how a CLIP-style
vision-language model turns **images + prompt templates** into **embeddings**,
stores them in a **database**, and answers **searches**. Ten tiny pipeline
files — one concept each — plus a sample downloader and a smoke test.
Standard-library SQLite, no frameworks. Read it top to bottom in 15 minutes,
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
| `templates.py` | ~55 | prompt templates: sentences with holes, filled per tag |
| `embedder.py` | ~55 | CLIP → unit-length 512-d vectors for images AND text |
| `tagger.py` | ~25 | zero-shot meta tags = dot products + argsort, no training |
| `fusion.py` | ~30 | the concatenation: `[image ; text] / √2`, and why it works |
| `features.py` | ~160 | **the one-call API**: `embed()`, ensembled tags, batched records |
| `db.py` | ~75 | vectors as float32 BLOBs in plain SQLite |
| `ingest.py` | ~60 | *composition*: batch `features.extract_batch` over files → store |
| `search.py` | ~85 | text / image / fused retrieval with dot products |
| `temperature.py` | ~50 | softmax + CLIP's learned logit scale: scores → probabilities |
| `export_web.py` | ~90 | dump the DB to `docs/db.json` + the 2-D PCA map coords |

The browser demo mirrors the same pipeline in `docs/js/` with **matching
module names**: `templates.js` ↔ `templates.py`, `clip.js` ↔ `embedder.py`,
`rank.js` ↔ `tagger.py`+`fusion.py`+`search.py`, `viz.js` ↔ `export_web.py`
(the 2-D embedding map), and `app.js` wires them to the page. Read a Python
file, then its twin — same pipeline, two languages.

## Why concatenate embeddings?

`fusion.py` is the whole answer in 30 lines:
`fused_emb = [image_emb ; text_emb] / √2` keeps the fused vector unit-length,
so a dot product against a duplicated query `[q ; q] / √2` equals the
**average of the visual similarity and the semantic (tag/caption) similarity**.
One vector, one dot product, both signals — no extra model, no re-ranking step.
`search.py --mode image|text|fused` lets you compare all three behaviors.

## Custom prompt templates

The template is the interface to the model. Change it and re-ingest:

```bash
.venv/bin/python ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
.venv/bin/python ingest.py holiday.png --caption "me hiking in Yosemite"
```

Add your own candidate tags in `templates.py` (`TAG_VOCABULARY`) — zero-shot
tagging means new tags need **no training**, just new words.

## Prompt ensembling — the CLIP paper's free accuracy boost

One template is one *phrasing* of the question, and CLIP is sensitive to
phrasing. The fix from the CLIP paper: embed each tag with **several**
phrasings and average the vectors (then re-normalize — the math lives in
`features.tag_embs`, the phrasings in `templates.ENSEMBLE_TAG_TEMPLATES`):

```bash
.venv/bin/python ingest.py images/*.jpg --ensemble        # all built-in phrasings
.venv/bin/python ingest.py images/*.jpg \
    --tag-template "a photo of a {tag}" \
    --tag-template "a drawing of a {tag}"                  # or roll your own
```

The average cancels each phrasing's quirks; what survives is the tag's
meaning. Zero training, one extra flag.

Ingest is also **batched**: `features.extract_batch` runs each encoder once
per chunk of 16 images instead of twice per file — the same trick that keeps
real pipelines' GPUs busy.

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
