# CLIP VLM 101 — vision-language embeddings, from zero to search

**Live demo (runs entirely in your browser):**
https://sugeerth.github.io/clip-vlm-101/ — CLIP·search: one box, real
inference, results. The full explorable walkthrough lives at
[/explore.html](https://sugeerth.github.io/clip-vlm-101/explore.html).

**The whole system in one page:** [ARCHITECTURE.md](ARCHITECTURE.md) — the
four stages a real billion-scale visual search runs (encode → **retrieve**
cheap → **rank** rich → **explain** + hallucination-gate), and which file
demonstrates each.

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
.venv/bin/python learn2rank.py                # 👍/👎 → a ranker that learns your taste
.venv/bin/python conformal.py --json docs/db.json  # a coverage guarantee, or abstain
.venv/bin/python judge.py --json docs/db.json --image images/004_cat.jpg  # a council of judges rules
.venv/bin/python trust.py --json docs/db.json --image images/004_cat.jpg  # compose every lens → one trust verdict
.venv/bin/python drift.py --json docs/db.json   # watch a stream drift: stable → shift → DRIFT
.venv/bin/python debate.py --json docs/db.json --image images/000_apple.jpg  # watch the agents argue
.venv/bin/python debate.py --json docs/db.json --eval   # multi-agent debate as evaluation
.venv/bin/python reason.py --json docs/db.json --image images/004_cat.jpg  # trace the whole stack → a decision
.venv/bin/python crawler.py "red panda" -n 6 # grow the gallery, with receipts
.venv/bin/python spider.py https://example.com/gallery  # or crawl any site
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
| `embedder.py` | ~70 | any registered model → unit vectors for images AND text |
| `models.py` | ~60 | **the registry**: newer brains + the per-family padding/scoring rules |
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
| `dcn.py` | ~130 | **the ranking stage**: a Deep&Cross Network v2 — the query×item interaction a dot product can't express |
| `learn2rank.py` | ~130 | **the ranker that learns YOU**: pairwise RankNet over your 👍/👎, on-device, blended-and-capped so a few clicks can't wreck retrieval |
| `conformal.py` | ~140 | **a coverage guarantee, or an honest abstain**: split conformal prediction → the smallest result set that contains the truth ≥ 1−α of the time |
| `explain.py` | ~150 | **explain + hallucination gate**: say why results matched, and redact any claim the results don't support |
| `judge.py` | ~200 | **a council of LLM judges**: several rubrics score each result, a gate parses every score, and the council rules — or abstains on a hung jury |
| `trust.py` | ~200 | **the capstone**: composes all four honesty lenses (gate · conformal · council · margin) into ONE trust verdict — or abstains when they disagree |
| `drift.py` | ~230 | **the monitor**: PSI + KS + conformal-coverage drift detection on a stream — stable / shift / DRIFT, with the failure cases to inspect and a scheduled CI gate |
| `debate.py` | ~180 | **multiple agents that talk**: the council's judges DEBATE via bounded-confidence dynamics — converge to consensus or split into named factions (contested) |
| `reason.py` | ~200 | **the reasoning layer**: traces the whole pipeline into one legible chain (each step premise→conclusion→status) and maps it to a CONSEQUENCE — show / caveat / withhold |
| `hermes.py` | ~180 | **the agentic searcher**: propose ⇄ evaluate ⇄ refine, to convergence |
| `scaling.py` | ~180 | **two-billion, on an envelope**: memory · O(√N) · shards · latency · cascade |
| `cascade.py` | ~170 | **approximate at every level**: binary → PQ → int8 → exact, recall kept |
| `crawler.py` | ~120 | **the crawler agent**: grow the gallery from Commons, with receipts |
| `grow.py` | ~150 | **grow the gallery 100× in one command**: bulk-crawl ~114 diverse topics → embed → dedup → a bigger `db.json` (remote thumbnails, no committed blobs) |
| `spider.py` | ~170 | **the web crawler**: BFS any site for images — robots.txt, pacing, caps |
| `scale.py` | ~620 | **one million rows**: records in SQLite, scans in packed f16 memmaps — ivf + int8 + RAM serving at industrial size |
| `pq.py` | ~200 | **product quantization**: 64 bytes/vector + table-lookup search — small enough that a 100k slice ships to the BROWSER (`js/pq.js` is its twin) |
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
`agent.js` ↔ `agent.py`, `recsys.js` ↔ `user_tower.py`, `learn.js` ↔
`learn2rank.py`, `conformal.js` ↔ `conformal.py`, `judge.js` ↔ `judge.py`,
`trust.js` ↔ `trust.py`, `drift.js` ↔ `drift.py`, `debate.js` ↔ `debate.py`,
`reason.js` ↔ `reason.py`; `viz.js`, `motion.js`,
`tour.js`, and `trace.js` are page-only (the matrix, map and strips, the
animation helpers, the guided tour, and the live agent trace), and `app.js`
wires everything together. Read a Python file, then its twin — same pipeline,
two languages.

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

## A ranker that learns YOU — on your device, no server

`dcn.py` shows the ranking *mechanism* with hand-set weights and admits the one
honest caveat: it's untrained. `learn2rank.py` (`js/learn.js`) closes it — it
*learns* the weights live from your 👍/👎, the click-relevance signal supplied
by you and sent nowhere. A linear scorer `s = w·x` over
`[cos_image, cos_text, tag_overlap, rank_prior]`, trained by **pairwise RankNet**
(Burges et al., ICML 2005): for every (👍, 👎) pair, push the liked one above the
disliked one.

```
o = w·(xᵢ − xⱼ)          # margin between a liked i and a disliked j
λ = −σ / (1 + e^{σo})     # RankNet's per-pair gradient
w ← w − lr·(λ·(xᵢ−xⱼ) + l2·w)
```

Three safeguards keep a handful of clicks from wrecking retrieval: `w` starts
`[1,0,0,0]`, so an **untrained ranker is exactly the base order**; one-sided
feedback (only 👍 or only 👎) can't form a pair, so it falls back to a Rocchio
nudge instead of a degenerate gradient; and the learned score is *blended and
capped at 50%* — `final = (1−β)·base + β·learned`, `β = 0.5·n/(n+3)` — so
retrieval always keeps at least half the vote. The entire model is four floats
in `localStorage`: your personal ranker, private by construction, wiped by a
reset. In the demo the learned-weight bars and 👍/👎 buttons live under the
results; `python3 learn2rank.py` runs the same math with a printed trace.

## A coverage guarantee, or an honest abstain

Every other file returns a top-k and hopes. `conformal.py` (`js/conformal.js`)
makes a promise you can check: given a confidence level (say 90%), it returns the
**smallest set of results that contains the true match at least 90% of the
time** — or, when nothing clears the bar, it *abstains* ("no confident match")
rather than guess. This is **split conformal prediction** (Vovk et al. 2005;
[Angelopoulos & Bates, arXiv:2107.07511](https://arxiv.org/abs/2107.07511)), and
for retrieval it collapses to one honestly-calibrated cosine threshold:

```
score      1 − cos(query, relevant)                 # nonconformity of a match
calibrate  k = ⌈(n+1)(1−α)⌉;  q̂ = k-th smallest score   # rank-corrected quantile
predict    return every item with  cos ≥ 1 − q̂       # covers the truth ≥ 1−α
```

The set is **adaptive for free**: a clear winner gives a set of one, a pile of
near-ties a big set — so set *size* is the per-query confidence signal. The
guarantee is distribution-free (nothing assumed about CLIP, only that queries are
exchangeable) and finite-sample: `1−α ≤ coverage ≤ 1−α + 1/(n+1)`. On the
14-image gallery `n` is tiny, so coverage moves in ~7% steps — 80% lands exactly,
90% rounds up to ~93% — and we say so rather than truncate the set to look tidy
(truncating would break the promise). `python3 conformal.py --json docs/db.json`
prints the coverage/size table; the demo shows the 80% set or the abstain badge
under every search.

## A council of LLM judges — many verdicts, aggregated honestly

One model's one score is a single point of failure. `judge.py` (`js/judge.js`)
convenes a **panel of judges**, each a different rubric — *relevance*,
*specificity*, *faithfulness* — and aggregates them the way a good council does
(the panel-of-evaluators idea, [Verga et al. 2024](https://arxiv.org/abs/2404.18796):
several small judges beat one big one and cancel each other's quirks):

- **the gate** — each judge's raw text is parsed to a number in `[0,1]` by
  `parse_score()` (it accepts `0.7`, `7/10`, `70%`, `score: 0.9`). A judge whose
  output has **no parseable score abstains** — it doesn't get to vote garbage,
  the same discipline as the hallucination gate.
- **the council** — a **confidence-weighted mean** of the valid votes, plus
  `consensus = 1 − (max − min)`. Then, like conformal, it **abstains rather than
  pretend**: fewer than a quorum of scores → *"no quorum"*; a panel that's too
  split → *"hung jury"*. A confident average over a coin flip is exactly the
  failure this prevents.

```bash
python3 judge.py --json docs/db.json --image images/004_cat.jpg
#   005_dog.jpg     relev 0.84  spec 0.80  faith 0.50  mean 0.73  → relevant (ruled)
#   002_bicycle.jpg relev 0.13  spec 0.00  faith 0.00  mean 0.05  → not relevant (ruled)
# and apple → pluto, which share the 'apple' tag by a fluke: faithfulness says
#   1.00, vision says 0.22 → the judges are split → the council abstains (hung jury)
```

The CLI runs a **model-free** heuristic judge (three rubrics read three stored
signals — cosine, tag-overlap, top-tag — which is exactly why they can disagree),
so the whole council mechanism runs on the committed gallery with no downloads,
like `dcn.py`'s untrained demo. On the live demo the **⚖️ convene a council of
LLM judges** button under the explanation runs three *real* in-browser
`SmolLM2-135M` judges (via transformers.js, nothing leaves your machine), pipes
each reply through the identical gate, and shows every judge's score, the
weighted verdict, and the consensus — or an honest abstain. The aggregation math
is a Python/JS twin, pinned byte-for-byte by CI.

## One trust verdict — composed from every honesty lens, or an abstain

The gate, conformal, the council — each already knows how to say *"I'm not
sure."* Reading four panels to decide whether to believe a result is the user's
job, so `trust.py` (`js/trust.js`) does it: it composes the signals the **same
way the council composes its judges** — a weighted agreement, with an **abstain
when they disagree**. A council of gates.

Four *different* lenses on the top result, each a trust contribution in `[0,1]`,
or **None** if that layer itself abstained:

| lens | question | abstains when |
|---|---|---|
| **gate** | how *strong* is the top match? (calibrated magnitude) | — |
| **conformal** | does it *clear* the distribution-free coverage bar τ? | below the bar |
| **council** | do independent rubric-judges *concur*? | hung / no quorum |
| **margin** | is #1 decisively *ahead* of the pack? (Hermes' separation) | a lone result |

`compose()` takes a confidence-weighted mean, measures `consensus = 1 − (max −
min)`, and refuses to rule when the lenses **split** (spread too wide) or too few
weigh in — and it **can't call "high" while half the evidence abstained** (a
participation cap). When strength, calibration, consensus and separation all
agree, trust is *high*; when a strong cosine meets a hung council, it lands at
*medium* or abstains — honestly.

```bash
python3 trust.py --json docs/db.json --image images/004_cat.jpg
#   005_dog.jpg  gate 1.00  confm 0.71  counc 0.73  marg 1.00  →  high     (all four agree)
#   …bicycle     gate 0.13  confm  —    counc 0.05  marg 0.07  →  low
#   apple→…      gate 0.40  confm 0.53  counc 0.03  marg 0.13  →  abstain  (split decision)
```

On the live demo a **trust headline** sits atop the explanation, composing the
three always-available lenses (gate + conformal + margin) immediately and folding
in the council's verdict the moment you convene it — one honest answer to *"how
much should I believe this?"*, with the per-lens pills shown so you can see which
lenses agreed and which abstained.

## Drift detection — is the live stream still the world we calibrated for?

Every guarantee above holds only while live queries stay **exchangeable** with
the gallery they were tuned on — conformal's coverage, the council's thresholds,
the trust composer. Production breaks that quietly: the data shifts and the
honest-looking numbers keep printing. `drift.py` (`js/drift.js`) is the monitor
that watches for it, on a stream of results, with three **distribution-free**
detectors on a quality signal (the same-tag match similarity):

- **PSI** (population stability index): `Σ (l−r)·ln(l/r)` over reference-quantile
  bins — the industry default (`<0.10` stable · `0.10–0.25` shift · `>0.25` drift).
- **KS** (Kolmogorov–Smirnov): the largest gap between the two CDFs, assuming
  nothing about the distribution — the same spirit as conformal.
- **coverage**: the repo-native one — calibrate a conformal bar on the reference,
  then measure coverage on the live window. Coverage falling below target **is**
  exchangeability breaking; conformal detects its own drift for free.

It also sorts the live window into **positive cases** (cleared the bar) and
**failure cases** (fell short — the ones to look at), and reports the failure
rate.

```bash
python3 drift.py --json docs/db.json
#   window          PSI    KS   cov  fail  status
#   t0 · baseline  0.00  0.00  83%   17%  · stable
#   t1 · 15% off   0.13  0.13  73%   27%  ~ shift
#   t2 · 35% off   0.57  0.30  57%   43%  ⚠ DRIFT   (PSI, KS and coverage all trip)
#   t3 · 60% off   1.07  0.55  42%   58%  ⚠ DRIFT
```

And it runs **periodically, in CI/CD**: `.github/workflows/drift.yml` is a
scheduled (daily cron) + on-demand job that freezes a reference at calibration
time (`drift_reference.json`), compares the current gallery against it, uploads a
**self-contained HTML dashboard** (`python3 drift.py --html`), and **fails the
run if the live data has drifted** (`python3 drift.py --gate`) — so re-exporting
`docs/db.json` to a corpus that breaks the guarantees trips a red ✗ instead of
silently shipping. The detector math is a Python/JS twin, pinned byte-for-byte by
CI.

## Multiple agents that talk — deliberation, not just a vote

The council polls its judges **independently** and averages them. Real
deliberation is agents **arguing**: each hears the others and updates its
position — but only toward peers it finds credible. `debate.py` (`js/debate.js`)
runs that as **bounded-confidence opinion dynamics**
([Hegselmann–Krause, 2002](https://www.jasss.org/5/3/2.html); the
multi-agent-debate-as-evaluator idea is [Du et al. 2023](https://arxiv.org/abs/2305.14325)),
and it does something a vote can't:

- **round** — each agent moves to the confidence-weighted mean of every agent
  within `EPS` of its current opinion (itself included). Talk sways you only when
  it's already close enough to hear.
- **converge** — repeat until nobody moves, or `MAX_ROUNDS`.
- **rule** — **one faction → consensus** (verdict = the shared opinion); **many
  factions → contested** — they deliberated and still disagree, so it abstains
  and **names who's in which camp**.

```bash
python3 debate.py --json docs/db.json --image images/004_cat.jpg
#   round 0:  relev 0.84  speci 0.80  faith 0.50
#   round 1:  relev 0.82  speci 0.73  faith 0.66     ← faithfulness gets talked upward
#   round 2:  relev 0.75  speci 0.75  faith 0.75
#   → relevant (consensus) after 3 rounds; factions: {relevance, specificity, faithfulness}
```

For `apple → pizza`, the faithfulness agent (which scored 1.0 on a shared-`apple`
tag fluke) sits outside everyone's confidence bound — it **won't move and can't
be moved**, so the panel splits `{relevance, specificity} | {faithfulness}` and
the debate reports **contested**, dissenter named. And it can do the opposite of
a hung jury: give it `[0.2, 0.5, 0.8]` — which the council would hang on (spread
0.6) — and the moderate agent **bridges** the extremes into a single consensus.
On the live demo, **🗣️ let them debate** under the council verdict animates the
agents' opinions converging or splitting across rounds. Deterministic dynamics, a
Python/JS twin pinned by CI; as evaluation (`--eval`) it converges the easy cases
and isolates the contested ones — what an independent average hides, the debate
names.

## The reasoning layer — trace every step, then map the consequence

Every stage produces a signal and knows how to abstain, but a pile of signals
isn't a decision. `reason.py` (`js/reason.js`) is the layer on top that **reasons
over the whole pipeline**: it walks it in order, turns each stage's output into a
**premise → conclusion** with a status (**ok · caution · stop**), and ends at a
**consequence** — what to actually *do*, why, and what it costs.

```
python3 reason.py --json docs/db.json --image images/004_cat.jpg
  ✓ 🔍 retrieve   embed the query, score all 13 candidates
     │              └─ top match is dog at similarity 0.86
  ✓ 🥇 rank       leads by margin +0.20
  ✓ 🎯 conformal  clears the calibrated bar
  ✓ ⚖️ council    relevant (consensus 66%)
  ✓ 🗣️ debate     consensus after 3 rounds → relevant
  ✓ 🧮 trust      trust: high (0.84)
  ⇒ CONSEQUENCE: show it as the answer
       because every lens agrees and the panel reached consensus
```

For `apple → pizza` the chain is legible about *where* it breaks: retrieve and
conformal pass, but the **council abstains** (hung) and the **debate is
contested**, so trust lands at *medium* and the consequence becomes **"show it
with a caveat, because the council couldn't confirm it."** The **consequence
map** is a small decision tree — *high → show · medium → caveat · low → flag ·
abstain → withhold (and whether to ask the user or say "no confident match")* —
so the same "rather say less" honesty that every stage carries resolves into one
action you can act on. On the live demo, **🧠 trace the reasoning** draws the
whole chain as a flow — every step's in and out, colored by status, ending at the
consequence — and it **redraws live** as you convene the council and let it
debate. The step assembly and the consequence map are a Python/JS twin pinned by
CI; it composes the stored signals, so the entire reasoning runs model-free.

And **▶ watch the agents work** *plays* it in real time: `js/trace.js` runs the
pipeline top to bottom, each node **pending → running (pulsing) → resolved**, so
you watch the agents get called — the council's three `SmolLM2` judges **stream
in one at a time** as their LLM calls actually return (via an `onVote` callback
on `councilWithLLM`). Page-only, like `viz.js`/`motion.js`/`tour.js` — there's no
"live" in a batch script — it's the one place you see the whole stack *execute*,
not just its result.

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
calibration) is capped to a representative sample, so **load and search are fast
at any gallery size** (measured: 1,400 images → interactive in ~1.6 s, a query
answered in ~130 ms, all in the browser).

`python3 grow.py --selftest` exercises the dedup + export offline (no model, no
network) — that's what CI runs.

### The two lower-level ways underneath it

- **Ask** (`crawler.py`): the Commons search API returns curated, freely
  licensed images with attribution — every download gets a manifest receipt.
- **Crawl** (`spider.py`): a real BFS web crawler for any site you point it
  at. Politeness is enforced in code — robots.txt per host, one request per
  second, same-domain scope, hard caps on pages and images — plus a quality
  gate (icon-sized files skipped) and content-hash dedupe. Crawled pages
  carry no license metadata, and the manifest says so: reuse is on you.

Either way the output feeds the same pipeline, and the live search box now
searches YOUR corpus. **Already too many for raw JSON?** The repo's own
`scale.html` searches a **100,000-image** slice in the browser via
`pq.py`/`js/pq.js` — product-quantized to 64 bytes each, a ~7 MB pack — the path
to 100k–1M images at a tiny download.

And **every search crawls, live**: after each text query the demo fans out
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

```bash
python3 eval.py images/*.jpg --model clip-b32      # benchmark the baseline…
python3 eval.py images/*.jpg --model siglip2-base  # …then prove the upgrade
python3 ingest.py images/*.jpg  # (one db = one model — re-ingest after switching)
```

On the live demo, the 🧠 dropdown swaps brains (MobileCLIP S0/B-LT and
SigLIP 2 via transformers.js) — the gallery re-embeds right in your
browser, because embeddings from different models never mix.

## One million rows — where the database's shape starts to matter

Everything above runs on a 14-image gallery. `scale.py` grows the SAME
design to **1,000,000 real dishes** — [Qdrant's public Wolt food
dataset](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings),
whose image embeddings were precomputed with the *same* clip-ViT-B-32
checkpoint this repo uses — and computes a million text embeddings itself,
one dish name at a time through `embedder.py` (~1,900 names/s on MPS).
Same 512-d space; the demo's text tower queries it directly.

What changes at a million is the *layout*, not the math. `db.py`'s
vector-BLOBs-in-SQLite is perfect at 14 rows and wrong at 1M (a scan would
decode a million BLOBs per query), so `scale.py` splits the two jobs the
way real vector stores do:

```
data/scale.sqlite      the RECORDS - name, caption, cafe, url; row id i = matrix row i
data/img_emb_f16.npy   the SCAN    - packed (1M x 512) float16 per tower, memory-
data/txt_emb_f16.npy                mapped: chunked matrix @ vector, no BLOB decodes
```

Three searches, all concepts the repo already taught: **brute** (`search.py`'s
scan, chunked), **ivf** (`ann.py`'s index trained on a 100k sample), **int8**
(`quantize.py` applied to the whole matrix). Fused search uses `fusion.py`'s
identity — *(image·q + text·q) / 2* — so no 1024-d matrix ever exists.

```bash
python3 scale.py ingest                                  # data/part-*.parquet -> the DB
python3 scale.py search "quattro formaggi pizza" --mode fused
python3 scale.py search "ramen" --ann --probes 8         # scan ~1%, keep ~all of it
python3 scale.py bench --ram --queries "sushi,burger,ramen"   # the honest table
python3 scale.py serve                                   # live at localhost:8071
```

`serve` is the storage-hierarchy lesson: the 868 ms scan was paging + casting a
memory-mapped gigabyte, not math. Promote both towers to f32 RAM once and the
same exact scan is one matmul — **27.7 ms for the truth, 4 ms with ivf** — behind
a one-file UI (fused ranking, queries ensembled over two phrasings à la
`ensemble.py`, and every answer prints what it cost and what it scanned).

Measured on the full 1,000,000 rows (Apple silicon, warm cache, median over
8 real text queries — records 288 MB, each f16 tower 1.02 GB, int8 512 MB,
ivf index 10 MB):

| search | ms / query | recall@10 | rows scanned |
| --- | --- | --- | --- |
| brute force, f16 → f32 | 868 | 1.00 — the truth | 100% |
| brute force, int8 | 235 | 0.68 | 100% |
| int8 top-100 → exact re-rank | 235 | **0.91** | 100% |
| ivf, 8 probes | 11.0 | 0.90 | **1.0%** |
| ivf, 16 probes | 19.7 | 0.95 | 1.9% |
| brute force, f32 in RAM (`serve`) | **27.7** | 1.00 — the truth | 100% |
| ivf-RAM, 8 probes (`serve`) | **4.0** | 0.90 | 1.0% |

Two lessons the 14-image gallery could never teach: one shared int8 scale is
too coarse for CLIP's outlier dimensions on a *packed* top-10 (0.68!), but the
truth rarely leaves the top-100 — fetch cheap, score exactly, 0.91 at the same
speed. And ivf recall *plateaus*: past 16 probes you pay ~2× the latency per
+0.00 recall, because a few true neighbours live in cells no nearby probe visits.

And the part you can touch: **[the scale page](https://sugeerth.github.io/clip-vlm-101/scale.html)
searches a 100,000-dish slice live in your browser** — `pq.py` product-quantizes
every vector to **64 bytes** (32× smaller than float32), so the whole index is a
~7 MB pack on GitHub Pages, and `js/pq.js` scores it with pure table lookups
(ADC) in a few milliseconds per keystroke, against a query embedded by the same
in-browser text tower as the demo. Recall is measured and printed on the page,
not promised.

The 4 GB of parquet and the built database stay out of git; CI runs
`scale.py selftest` and `pq.py selftest` — the same scan/ivf/int8/PQ machinery
on synthetic clustered vectors, no model, no downloads — plus `test_pq.mjs`,
which pins the JS twin to the Python math. The full story with the measured
numbers: **[the scale report](https://sugeerth.github.io/clip-vlm-101/scale.html)**.

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
| `python3 hermes.py --image pizza --json docs/db.json` | the evaluator **rejects** the drifting pass and keeps the honest ranking |
| `python3 learn2rank.py` | after 👍 the tag-sharers / 👎 the rest, `tag_overlap` importance dominates and the parrot lifts above the sunflower |
| `python3 conformal.py --json docs/db.json` | coverage sits **on or above** every target (80% → 84.6%, 90% → 92.3%); the set grows as you demand more confidence |
| `python3 judge.py --json docs/db.json --image images/004_cat.jpg` | the council rules **relevant** for cat→dog, **not relevant** for cat→bicycle, and **hung jury** where a shared tag and the vision signal disagree |
| `python3 trust.py --json docs/db.json --image images/004_cat.jpg` | cat→dog composes to **high** trust (all four lenses agree); where the lenses split, the verdict **abstains** instead of averaging |
| `python3 drift.py --json docs/db.json --selftest` | the detectors escalate **stable → shift → drift → drift** as a growing fraction of the stream goes off-distribution; PSI is monotone in the contamination |
| `python3 debate.py --json docs/db.json --eval` | the panel reaches **consensus on 12/14** top hits and stays **contested on 2** (the tag-fluke cases); ≥1 agent changes its mind on 8/14 |
| `python3 reason.py --json docs/db.json --image images/004_cat.jpg` | every step passes (retrieve→…→trust) → **high trust → "show it as the answer"**; `--image .../000_apple.jpg` breaks at the council → **"show with a caveat"** |
| `python3 scale.py selftest` | chunked scan == naive argsort; ivf probes=8 keeps ≥7/10 of the truth scanning <½ the rows |

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
