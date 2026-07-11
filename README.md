# CLIP VLM 101 — vision-language embeddings, from zero to search

**Live demo (runs entirely in your browser):** https://sugeerth.github.io/clip-vlm-101/

A deliberately tiny, readable pipeline that shows how a CLIP-style
vision-language model turns **images + prompt templates** into **embeddings**,
stores them in a **database**, and answers **searches**. Eight tiny pipeline
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
.venv/bin/python ingest.py images/*.jpg     # embed + auto-tag + store in SQLite
.venv/bin/python search.py "a fluffy animal"
.venv/bin/python search.py --image images/cat.jpg   # image-to-image search
```

First run downloads the CLIP model (~600 MB). Device is auto-detected:
CUDA → Apple Silicon MPS → CPU (all work; this model is small).

## The files — one concept each

Each file's docstring starts with a `pipeline:` line showing where it sits.
Suggested reading order:

| file | lines | the one concept it teaches |
|---|---|---|
| `templates.py` | ~45 | prompt templates: sentences with holes, filled per tag |
| `embedder.py` | ~55 | CLIP → unit-length 512-d vectors for images AND text |
| `tagger.py` | ~25 | zero-shot meta tags = dot products + argsort, no training |
| `fusion.py` | ~30 | the concatenation: `[image ; text] / √2`, and why it works |
| `db.py` | ~70 | vectors as float32 BLOBs in plain SQLite |
| `ingest.py` | ~55 | *composition*: embed → tag → caption → fuse → store |
| `search.py` | ~70 | text / image / fused retrieval with dot products |
| `export_web.py` | ~45 | dump the DB to `docs/db.json` for the static web demo |

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
