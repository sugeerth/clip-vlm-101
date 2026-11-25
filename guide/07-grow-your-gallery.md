# Grow the gallery — and swap in a newer model

*One command to crawl ~114 topics into a bigger gallery, plus the model registry.*

[← README](../README.md) · [all chapters](README.md)

The gallery is 14 images and the model is CLIP ViT-B/32. Both are defaults, not
limits. This chapter is how you make the repo yours: one command that bulk-crawls
a much bigger gallery from free image hosts, and a registry that lets you pick a
newer model by key.

## Grow the gallery 100× with one command — for free

The demo ships with 14 curated images, but that's a *seed*, not a ceiling.
`grow.py` turns it into a **thousands-of-images** gallery in one command, free:

```bash
python3 grow.py --per 12            # ~114 diverse topics × 12 ≈ 1,400 images (100×)
python3 grow.py --per 30 --merge   # keep the current 14, add ~3,400 more
python3 grow.py --topics cat sushi "eiffel tower" --per 20   # your own topics
```

It crawls freely-licensed images from Wikimedia Commons (with attribution —
`crawler.py` keeps the receipts), embeds each with the *same* clip-ViT-B/32 the
demo runs, drops near-duplicates by embedding cosine, and writes a bigger
`docs/db.json` in the exact format the browser already searches — **no code
change, no committed image blobs**: each item stores its embedding plus the
**remote Commons thumbnail URL**, so 100× the images is a few MB of JSON and the
browser loads thumbnails straight from the free host. The search stays instant —
it's the same dot products, and the page's one O(n²) step (the conformal
calibration, [chapter 4](04-trust-and-evaluation.md)) is capped to a
representative sample, so **load and search are fast
at any gallery size** (measured: 1,400 images → interactive in ~1.6 s, a query
answered in ~130 ms, all in the browser).

`python3 grow.py --selftest` exercises the dedup + export offline (no model, no
network) — that's what CI runs.

### The two lower-level ways underneath it

`grow.py` is a wrapper. If you want the pieces:

- **Ask** (`crawler.py`): the Commons search API returns curated, freely
  licensed images with attribution — every download gets a manifest receipt.
- **Crawl** (`spider.py`): a real BFS web crawler for any site you point it
  at. Politeness is enforced in code — robots.txt per host, one request per
  second, same-domain scope, hard caps on pages and images — plus a quality
  gate (icon-sized files skipped) and content-hash dedupe. Crawled pages
  carry no license metadata, and the manifest says so: reuse is on you.

```bash
.venv/bin/python crawler.py "red panda" -n 6              # grow the gallery, with receipts
.venv/bin/python spider.py https://example.com/gallery    # or crawl any site
```

Either way the output feeds the same pipeline
([chapter 1](01-the-core-pipeline.md)), and the live search box now
searches YOUR corpus.

**Already too many for raw JSON?** The repo's own
`scale.html` searches a **100,000-image** slice in the browser via
`pq.py`/`js/pq.js` — product-quantized to 64 bytes each, an ~8 MB pack plus 18 MB of metadata — the path
to 100k–1M images at a tiny download. The measured recall and latency behind
that path are in [chapter 6](06-a-million-rows.md).

### Every search crawls, live

After each text query the demo fans out
to **five independent, keyless, CORS-open image APIs in parallel** —
[Openverse](https://api.openverse.org), Wikimedia Commons (`origin=*`),
the [Art Institute of Chicago](https://api.artic.edu/docs/), the
[Met Open Access collection](https://metmuseum.github.io/), and
[iNaturalist](https://api.inaturalist.org/v1/docs/) — merges what comes
back, embeds every thumbnail *in your browser* with the vision tower, and
ranks them under the gallery results with a license receipt on every card
(`js/crawler.js`; toggle with the 🌐 chip). One provider down costs only
its own results, and the section header reports the per-provider ledger —
a silent web search is indistinguishable from a broken one. The Python
twin: `hermes.py "red panda" --crawl 6`.

## Pick your brain: newer models, one key away

CLIP ViT-B/32 is the 2021 baseline. `models.py` (and its browser mirror
`js/models.js`) registers stronger drop-ins — and handles the two silent
traps that break naive swaps: SigLIP-family text encoders REQUIRE
`padding="max_length"` (pad-to-longest quietly wrecks their embeddings),
and SigLIP's sigmoid training means its per-tag probabilities are
calibrated directly (`sigmoid(scale·cos + bias)`, constants read from the
checkpoint — `labels.siglip_label_probs`), no neutral prompt needed.

| key | model | IN-1k 0-shot | notes |
|---|---|---|---|
| `clip-b32` | CLIP ViT-B/32 ('21) | ~63% | the default; matches the committed gallery |
| `clip-l14` | CLIP ViT-L/14 ('21) | ~75% | big and slow |
| `siglip2-base` | SigLIP 2 B/16 ('25) | ~78% | sigmoid-trained, multilingual |
| `siglip2-384` | SigLIP 2 B/16-384 | ~79% | same brain, higher-res eyes |

Pick a key, measure the difference yourself, then rebuild the index:

```bash
python3 eval.py images/*.jpg --model clip-b32      # benchmark the baseline…
python3 eval.py images/*.jpg --model siglip2-base  # …then prove the upgrade
python3 ingest.py images/*.jpg  # (one db = one model — re-ingest after switching)
```

On the live demo, the 🧠 dropdown swaps brains (MobileCLIP S0/B-LT and
SigLIP 2 via transformers.js) — the gallery re-embeds right in your
browser, because embeddings from different models never mix.

---
← [README](../README.md) · [all chapters](README.md) · prev: [A million rows — where the database's shape starts to matter](06-a-million-rows.md) · next: [The web demo — the same pipeline, in your browser](08-the-web-demo.md)
