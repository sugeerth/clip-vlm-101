# Every file in the repo, tiered

*The full inventory — start with the eight core files, then pick a chapter.*

[← README](../README.md) · [all chapters](README.md)

The layout is deliberately flat: every file sits at the repo root and imports
its neighbours directly — no packaging, no `src/` tree, no import graph to
unpick before you can execute one concept. Most files are runnable on their own
as `python3 <file>.py`. Nine are not: `templates.py`, `embedder.py`,
`tagger.py`, `fusion.py`, `labels.py`, `models.py`, `ensemble.py`, `db.py`
and `agent.py` are pure libraries — they have no `__main__` block and do
nothing when run directly; you import them, or reach them through a CLI
that does. Most files' docstrings start with a `pipeline:` line showing
where they sit (`download_samples.py` and `test_smoke.py` are the
exceptions).

Flat is easy to read one file at a time and hard to start on, so the inventory
is tiered below. Read tier 1 in order. After that, jump to the tier that
matches the chapter you care about and ignore the rest — the tiers are
independent. The `lines` column is `wc -l`.

## Tier 1 — the eight files to read first

The whole pipeline: a tag and an image go in, a searchable gallery comes out.
Read them in this order.
→ [The core pipeline — image in, searchable vectors out](01-the-core-pipeline.md);
tier 6 replaces the last three (`db.py`, `ingest.py`, `search.py`) at 690k rows

| file | lines | the one concept it teaches |
|---|---|---|
| `templates.py` | 58 | prompt templates: sentences with holes, filled per tag |
| `embedder.py` | 68 | any registered model → unit vectors for images AND text |
| `tagger.py` | 21 | zero-shot meta tags = dot products + argsort, no training |
| `fusion.py` | 28 | the concatenation: `[image ; text] / √2`, and why it works |
| `features.py` | 157 | **the one-call API**: image-only `embed()`, batch tags, full records |
| `db.py` | 98 | vectors as float32 BLOBs in plain SQLite (+ the JSON gallery loader) |
| `ingest.py` | 74 | *composition*: batch `features.extract_batch` over files → store |
| `search.py` | 88 | text / image / fused retrieval with dot products |

## Tier 2 — better tags, better prompts, a newer brain

Three files that upgrade the tagging half without touching anything above.
→ [The core pipeline](01-the-core-pipeline.md) for `labels.py`,
[Understanding CLIP](02-understanding-clip.md) for `ensemble.py`,
[Grow the gallery](07-grow-your-gallery.md) for `models.py`

| file | lines | the one concept it teaches |
|---|---|---|
| `labels.py` | 65 | **multi-label**: per-tag sigmoid vs a neutral prompt → dynamic label sets |
| `ensemble.py` | 58 | **prompt ensembling**: average many templates per tag, +3.5% for free |
| `models.py` | 59 | **the registry**: newer brains + the per-family padding/scoring rules |

## Tier 3 — agents

Loops that propose, criticise their own output, and argue with each other.
→ [Agents — propose, critique, argue, explain](03-agents.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `agent.py` | 123 | **the agent**: propose ⇄ critique loop, publishes only when satisfied |
| `hermes.py` | 230 | **the agentic searcher**: propose ⇄ evaluate ⇄ refine, to convergence |
| `debate.py` | 195 | **multiple agents that talk**: the council's judges DEBATE via bounded-confidence dynamics — converge to consensus or split into named factions (contested) |
| `reason.py` | 192 | **the reasoning layer**: traces the whole pipeline into one legible chain (each step premise→conclusion→status) and maps it to a CONSEQUENCE — show / caveat / withhold |

## Tier 4 — trust and evaluation

The honesty layer: redact what the results don't support, abstain instead of
guessing, and watch whether the guarantees still hold — plus the plain
hit-rate benchmark you use to prove an optimization actually helped.
→ [Trust — coverage guarantees, judges, drift](04-trust-and-evaluation.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `eval.py` | 100 | **the benchmark**: top-1/top-5 hit rates — prove an optimization helps |
| `explain.py` | 212 | **explain + hallucination gate**: say why results matched, and redact any claim the results don't support — taught in [ARCHITECTURE.md](../ARCHITECTURE.md) § 3, not in a guide chapter |
| `conformal.py` | 150 | **a coverage guarantee, or an honest abstain**: split conformal prediction → the smallest result set that contains the truth ≥ 1−α of the time |
| `judge.py` | 235 | **a council of LLM judges**: several rubrics score each result, a gate parses every score, and the council rules — or abstains on a hung jury |
| `trust.py` | 186 | **the capstone**: composes all four honesty lenses (gate · conformal · council · margin) into ONE trust verdict — or abstains when they disagree |
| `drift.py` | 334 | **the monitor**: PSI + KS + conformal-coverage drift detection on a stream — stable / shift / DRIFT, with the failure cases to inspect and a scheduled CI gate |

## Tier 5 — recommendation and ranking

The same vectors, used to rank items for a person instead of for a query.
→ [Recommendation — two towers and a ranker that learns you](05-recommendation.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `item_tower.py` | 117 | **two-tower recsys**: agent-verified item embeddings, offline |
| `user_tower.py` | 62 | **the serving half**: mean-pooled likes → recommendations, one matmul |
| `dcn.py` | 150 | **the ranking stage**: a Deep&Cross Network v2 — the query×item interaction a dot product can't express |
| `learn2rank.py` | 164 | **the ranker that learns YOU**: pairwise RankNet over your 👍/👎, on-device, blended-and-capped so a few clicks can't wreck retrieval |

## Tier 6 — scale and quantization

`scaling.py` and `cascade.py` are the paper-and-envelope versions. `scale.py`
is the real engine — storage, scan, IVF, int8 and OPQ all in one file, under
one CLI. `pq.py` is the same quantization, shipped to a browser.
→ [A million rows — where the database's shape starts to matter](06-a-million-rows.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `scaling.py` | 185 | **two-billion, on an envelope**: memory · O(√N) · shards · latency · cascade |
| `cascade.py` | 192 | **approximate at every level**: binary → PQ → int8 → exact, recall kept |
| `scale.py` | 903 | **at scale** — records in SQLite + packed f16 memmaps, the exact scan at three rungs of the storage hierarchy (memmap → RAM → GPU), the inverted-file index, int8 per dimension, and the OPQ re-rank; one CLI over all of it (`ingest`, `search`, `bench`, `serve`, `selftest`) |
| `opq.py` | 168 | **optimised PQ**: a learned rotation before the split — same 64 bytes, ~10 recall points more |
| `pq.py` | 256 | **product quantization for the BROWSER**: 64 bytes/vector + ADC lookups (`js/pq.js` twin), lazy int8 refine tier |

## Tier 7 — growing the gallery

Four ways to get more images: fetch the fixed sample list, ask an API,
bulk-crawl topics, or crawl a site. The model registry that swaps the brain is
`models.py`, in tier 2.
→ [Grow the gallery — and swap in a newer model](07-grow-your-gallery.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `download_samples.py` | 61 | the seed: fetch the 14 freely-licensed sample images into `images/` |
| `crawler.py` | 126 | **the crawler agent**: grow the gallery from Commons, with receipts |
| `grow.py` | 175 | **grow the gallery 100× in one command**: bulk-crawl ~114 diverse topics → embed → dedup → a bigger `db.json` (remote thumbnails, no committed blobs) |
| `spider.py` | 190 | **the web crawler**: BFS any site for images — robots.txt, pacing, caps |

## Tier 8 — the standalone lessons

→ [Understanding CLIP — and squeezing more out of it](02-understanding-clip.md)

Seven standalone lessons build on the stored vectors, and every one runs
**without the model**: `ann.py` on synthetic random vectors (its `--json` is
optional — it also indexes the real gallery), `temperature.py` on the
committed `docs/db.json` it loads with no flags at all, and the rest on that
same gallery via `--json docs/db.json`. The lessons also run **live on the
[explorable page](https://sugeerth.github.io/clip-vlm-101/explore.html#lessons)**,
where `docs/js/lessons.js` reproduces the same numbers in JavaScript (CI
holds both languages to it):

| lesson | lines | the one concept it teaches |
|---|---|---|
| `temperature.py` | 48 | softmax + CLIP's learned logit scale: scores → probabilities |
| `similarity.py` | 114 | the N×N image matrix + the modality gap (why scales don't mix) |
| `retrieval_eval.py` | 70 | retrieval evaluation: leave-one-out precision@k and MRR |
| `arithmetic.py` | 89 | vector algebra: `cat + dog − apple`, centroids, renormalize |
| `quantize.py` | 76 | int8 scalar quantization: 4× smaller, measure the damage |
| `similarity.py --centered` | same file | closing the gap: center each modality, margin widens ~3× |
| `ann.py` | 110 | IVF (what vector DBs do): scan ~2%, keep ~75% of the truth |

## Tier 9 — the web demo

One Python file exports the gallery; the rest of the demo lives in `docs/js/`.
→ [The web demo — the same pipeline, in your browser](08-the-web-demo.md)

| file | lines | the one concept it teaches |
|---|---|---|
| `export_web.py` | 89 | dump the DB to `docs/db.json` + the 2-D PCA map coords |

The browser demo mirrors the same pipeline in `docs/js/` with **matching
module names**. Twenty have a Python twin:

| JS module | Python twin |
|---|---|
| `templates.js` | `templates.py` |
| `clip.js` | `embedder.py` |
| `models.js` | `models.py` |
| `rank.js` | `tagger.py`+`fusion.py`+`search.py` |
| `labels.js` | `labels.py` |
| `agent.js` | `agent.py` |
| `hermes.js` | `hermes.py` |
| `explain.js` | `explain.py` |
| `recsys.js` | `user_tower.py` |
| `learn.js` | `learn2rank.py` |
| `conformal.js` | `conformal.py` |
| `judge.js` | `judge.py` |
| `trust.js` | `trust.py` |
| `drift.js` | `drift.py` |
| `debate.js` | `debate.py` |
| `reason.js` | `reason.py` |
| `lessons.js` | `temperature.py`+`similarity.py`+`retrieval_eval.py`+`arithmetic.py`+`quantize.py`+`ann.py` |
| `pq.js` | `pq.py` |
| `crawler.js` | `crawler.py` (reached from Python as `hermes.py "…" --crawl N`) |
| `simple.js` | *page-only* — the front page's one search box, composed from the modules above |

`viz.js`, `motion.js`, `tour.js`, and `trace.js` are page-only (the matrix, map
and strips, the animation helpers, the guided tour, and the live agent trace),
and `app.js` wires everything together. Read a Python file, then its twin —
same pipeline, two languages.

## Tier 10 — the test

| file | lines | what it covers |
|---|---|---|
| `test_smoke.py` | 1115 | the model-free smoke test: templates, tagging, fusion math, PCA, the DB, and the math of every layer above — stubs the CLIP encoder so it needs only numpy, and CI runs it on every push |

---
← [README](../README.md) · [all chapters](README.md) · next: [The core pipeline — image in, searchable vectors out](01-the-core-pipeline.md)
