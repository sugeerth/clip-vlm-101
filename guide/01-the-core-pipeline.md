# The core pipeline — image in, searchable vectors out

*features.py as a one-call API, why the two embeddings are concatenated, and how dynamic multi-label tags work.*

[← README](../README.md) · [all chapters](README.md)

This is the path every other chapter stands on: one image goes in, one
database-ready row comes out — a caption, a few tags, and three vectors.
In order below: the single call that produces that row, what each of the
three vectors is, why the third one is a concatenation of the first two,
how the tag set can size itself to the image, and the prompt template that
decides all of it. The shell commands assume the `.venv` and the sample
images from the [README](../README.md) quickstart.

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

`python3 features.py photo.jpg` prints the tags and all three vectors'
shapes; `--json` dumps the full record.

### The three vectors

**Exact dimensions** (model `openai/clip-vit-base-patch32`; these are what
`db.py` stores as BLOBs — `export_web.py` ships `image_emb` and `text_emb`
to [the web demo](08-the-web-demo.md), which recomputes the fused score as
their average):

| vector | shape | dtype | unit-length | bytes as BLOB | meaning |
|---|---|---|---|---|---|
| `image_emb` | `(512,)` | float32 | yes | 2 048 | what the image *looks* like (vision tower) |
| `text_emb` | `(512,)` | float32 | yes | 2 048 | what its caption/tags *mean* (text tower) |
| `fused_emb` | `(1024,)` | float32 | yes | 4 096 | `[image_emb ; text_emb] / √2` — both signals |

Both towers project into the same 512-d space, which is why one image and
one sentence can be compared with a plain dot product (cosine similarity,
range −1…1).

## Why concatenate embeddings?

The first two vectors come out of the model. The third one you build.

`fusion.py` is the whole answer in 28 lines:
`fused_emb = [image_emb ; text_emb] / √2` keeps the fused vector unit-length,
so a dot product against a duplicated query `[q ; q] / √2` equals the
**average of the visual similarity and the semantic (tag/caption) similarity**.
One vector, one dot product, both signals — no extra model, no re-ranking step.
`search.py --mode image|text|fused` lets you compare all three behaviors.

## Multi-class vs multi-label: dynamic meta tags

Half of that fused vector is the text side, and the text side is only ever as
good as the tags — the caption is the tags glued into a sentence. So how the
tags get picked matters, and there are two ways to pick them.

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

Both paths read the same candidate list out of `templates.TAG_VOCABULARY`,
and both score it by filling those candidates into the same prompt sentence.
That sentence is the one thing left to tune.

## Custom prompt templates

The template is the interface to the model. Change it and re-ingest:

```bash
.venv/bin/python ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
.venv/bin/python ingest.py holiday.png --caption "me hiking in Yosemite"
```

Add your own candidate tags in `templates.py` (`TAG_VOCABULARY`) — zero-shot
tagging means new tags need **no training**, just new words.

That is the core: one model, one dot product, and two knobs — the vocabulary
and the template. [Chapter 2](02-understanding-clip.md) takes the same
vectors apart: temperature, the modality gap, vector arithmetic,
quantization, IVF.

---
← [README](../README.md) · [all chapters](README.md) · prev: [Every file in the repo, tiered](00-every-file.md) · next: [Understanding CLIP — and squeezing more out of it](02-understanding-clip.md)
