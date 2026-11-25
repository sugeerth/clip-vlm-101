# Architecture — the whole system, and where each file sits

This repo teaches one search on 14 images. But every file is a scaled-down
version of a piece of a **real, billion-scale visual search system**. This
page is the map: the four stages a production system actually has, why each
exists, and which file in here demonstrates it. The design is the same at 14
images and at 1,000,000,000 — only the *layout* changes.

```
        query ("a fluffy animal")  or  an image
                        │
        ┌───────────────▼────────────────┐
        │ 0 · ENCODE   two towers         │  embedder.py · models.py
        │   image→512d, text→512d,        │  fusion.py · templates.py
        │   same space, one dot product   │  (clip.js in the browser)
        └───────────────┬────────────────┘
                        │  a query vector
        ┌───────────────▼────────────────┐
        │ 1 · RETRIEVE   billions→hundreds│  search.py · ann.py (IVF)
        │   cheap ANN over compressed     │  pq.py (64 B/vec) · scale.py (1 M)
        │   vectors. recall, not order.   │  quantize.py (int8)
        └───────────────┬────────────────┘
                        │  ~hundreds of candidates
        ┌───────────────▼────────────────┐
        │ 2 · RANK   hundreds→ordered top │  dcn.py  ← the piece a dot
        │   rich query×item interaction   │  product structurally cannot do
        │   the towers cannot express     │  (hermes.py refines the query)
        │   + LEARN the weights from 👍/👎 │  learn2rank.py (learn.js twin) —
        │   on-device, no server          │  closes dcn's "untrained" caveat
        └───────────────┬────────────────┘
                        │  the final ordered results
        ┌───────────────▼────────────────┐
        │ 3 · EXPLAIN + GATE   say WHY,   │  explain.py (explain.js twin)
        │   never lie. grounded template  │  agent.py (the same verify-before-
        │   or LLM → hallucination gate   │  publish idea, on the write path)
        │   + a coverage GUARANTEE / abstain │ conformal.py (conformal.js twin)
        │   + a COUNCIL of LLM judges / abstain │ judge.py (judge.js twin)
        │     └ that DEBATE: consensus / factions │ debate.py (debate.js twin)
        │   ⇒ ONE composed TRUST verdict / abstain │ trust.py (trust.js twin)
        │   ⇒ a REASONING chain → a CONSEQUENCE │ reason.py (reason.js twin):
        │      show / caveat / withhold          │ every step in & out, to a decision
        └───────────────┬─────────────────┘
                        │  and, watching the whole thing over time:
        ┌───────────────▼─────────────────┐
        │ MONITOR   is the live stream still │ drift.py (drift.js twin)
        │   what we calibrated for? PSI/KS/  │ — a scheduled CI/CD gate, not a
        │   coverage → stable / shift / DRIFT │ one-time check
        └─────────────────────────────────┘
```

## 0 · Encode — two towers, one space

CLIP projects an image and a sentence into the **same** 512-d space, so
similarity is a single dot product. That independence — the two towers never
see each other until the dot product — is the load-bearing choice: item
vectors can be computed **offline, once**, and indexed, so a query is one
cheap probe. `embedder.py` runs any registered model (`models.py`: CLIP,
MobileCLIP, SigLIP 2); `fusion.py` concatenates image+text into one vector.

## 1 · Retrieve — billions → hundreds, cheaply

At 14 rows you scan everything (`search.py`). At a billion you cannot, so
retrieval trades a little accuracy for enormous speed, exactly as production
vector stores do:

- **ANN / IVF** (`ann.py`): cluster the corpus, scan only the nearest few
  clusters — ~2% of vectors, ~75% of the true neighbours. This is what FAISS
  IVF and ScaNN do ([FAISS](https://arxiv.org/abs/2401.08281),
  [ScaNN](https://arxiv.org/abs/1908.10396)).
- **ANN / HNSW** (`hnsw.py`): the graph index modern vector databases (Qdrant,
  Weaviate, pgvector, Vespa, FAISS-HNSW) actually default to — a layered
  navigable-small-world graph you *walk* toward the query instead of cells you
  scan. Hops grow like log N, not √N, so at matched work it keeps more of the
  truth than IVF (measured, same vectors: same recall for ~20% fewer distance
  computations, and it reaches recall IVF can't touch)
  ([Malkov & Yashunin 2016](https://arxiv.org/abs/1603.09320)).
- **ANN / DiskANN** (`diskann.py`): a billion vectors on ONE 64 GB box. HNSW
  needs the whole graph and every full vector in RAM; DiskANN keeps a ~64-byte
  PQ sketch + the graph in RAM and the full-precision vectors on SSD. A Vamana
  graph — built by *robust-prune* with a slack α>1 that spares long edges to
  keep the diameter short — is navigated on the RAM sketch, then only the ~L
  finalists' true vectors are read from disk and reranked (measured, same
  vectors: reranking recovers ~all the recall while reading ~L of N)
  ([Subramanya et al. 2019 / DiskANN](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)).
- **Compression** (`quantize.py`, `pq.py`): int8 is 4× smaller;
  **product quantization** is ~32× smaller (64 bytes/vector) and searches by
  table lookup with no multiplies — small enough that `pq.js` ships a
  100k-vector index *to the browser*.
- **A million, laid out right** (`scale.py`): records in SQLite, vectors in a
  packed float16 memmap — the split every real vector store makes.

Retrieval is **recall-oriented**: get the right candidates into the shortlist.
It is deliberately *not* trying to get the final order right.

## 2 · Rank — the interaction a dot product can't express

This is the stage this session added, and the honest answer to "two towers
work, but what's the *more*?" A two-tower score is one fixed bilinear form
`q·v`. It cannot learn "when the query fires on *these* dimensions, weight
tag-overlap high and raw cosine low" — there is **no query×item feature
interaction**, by construction. That independence is what made retrieval
cheap; it is also a modeling ceiling.

The industry answer is a **two-stage funnel** (YouTube RecSys'16, Google Play
Wide&Deep, Pinterest): cheap retrieval narrows billions→hundreds, then a
**rich ranking model** re-scores only those hundreds with full feature
interaction. It can afford the cost because it runs on a tiny set, never the
corpus.

`dcn.py` is that ranker — a **Deep & Cross Network v2**
([Wang et al., WWW 2021](https://arxiv.org/abs/2008.13535)). Its whole idea is
one explicit multiplicative cross per layer:

```
x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
```

Its features are the per-candidate signals a dot product throws away:
`[cos_image, cos_text, tag_overlap, rank_prior]`. At `W=0` the residual makes
it a passthrough that **reproduces the retrieval order exactly**; switch on one
cross weight and it expresses `cos_image · tag_overlap` — "prefer candidates
that both *look* similar **and** *share* a tag" — which the towers cannot
represent. (Untrained, `dcn.py` *demonstrates the mechanism*; production learns
`W` from click/relevance labels. That's the one honest caveat.)

**`learn2rank.py` closes that caveat, on your own device.** `dcn.py` shows the
ranking *mechanism* with hand-set weights; `learn2rank.py` (twin `js/learn.js`)
*learns* them live from your 👍/👎 — the click/relevance signal, supplied by you,
never sent anywhere. It is a linear scorer `s = w·x` over the same features,
trained by **pairwise RankNet** ([Burges et al., ICML 2005](https://icml.cc/Conferences/2005/proceedings/papers/012_Learning_BurgesEtAl.pdf)):
for every (👍 i, 👎 j) pair, nudge `i` above `j` —
`o = w·(xᵢ−xⱼ); λ = −σ/(1+e^{σo}); w ← w − lr·(λ·(xᵢ−xⱼ) + l2·w)`. Three safeguards
keep a handful of clicks from wrecking retrieval: `w` starts `[1,0,0,0]` so an
untrained ranker **is** the base order; one-sided feedback (only 👍, or only 👎)
falls back to a Rocchio nudge instead of a degenerate pairwise gradient; and the
learned score is *blended*, capped at 50% — `final = (1−β)·base + β·learned`,
`β = 0.5·n/(n+3)` — so retrieval always keeps at least half the vote. The whole
model is four floats in `localStorage`: your personal ranker, private by
construction. In the live demo the learned-weight bars and the 👍/👎 buttons sit
under the results; a reset wipes it.

`hermes.py` is the agentic query-side complement: it proposes phrasings,
critiques each by retrieval margin, and refines — improving stage 1's input
before stage 2 ever runs.

## 3 · Explain + Gate — say why, and never lie

A friendly system says *why* it matched. But generated prose can hallucinate —
a "sunset" no result shows, a score that never happened. So `explain.py` is two
halves, and the second is the point:

- **Explain**: a grounded one-liner built *only* from verifiable facts — the
  tags shared across the top results, the top score, a calibrated strength word
  (strong/moderate/weak, from CLIP's real cosine bands). Grounded *by
  construction*: it can only emit evidence.
- **The hallucination gate** `verify(text, evidence)`: the trust boundary.
  Whatever wrote the text — the template, or an optional in-browser LLM
  (`explain.js` can load SmolLM2) — every sentence is kept only if every content
  word is a real tag/query word, every number matches a real score, and every
  strength word is the true one. Anything else is redacted, with a reason.

Because the evidence is a **closed, finite set** of tags and numbers,
"is this claim supported?" collapses from an NLI/LLM-judge problem to **set
membership**. This is the strict special case of attribution/faithfulness —
[AIS](https://arxiv.org/abs/2112.12870), RAGAS-faithfulness, [FEVER] — done as
an exact lexical gate, no model required. The LLM is untrusted; the template is
the floor it always falls back to. It is the same discipline as `agent.py` on
the *write* path (verify features before publishing) — this repo verifies
before it *speaks*, too.

**The gate says the words are honest; `conformal.py` says the *results* are —
with a number.** Every other file returns a top-k and hopes; `conformal.py`
(twin `js/conformal.js`) returns the smallest set of results that contains the
true match **at least 1−α of the time**, or, when nothing clears the bar, it
*abstains* — "no confident match" — instead of guessing. This is **split
conformal prediction** ([Vovk et al. 2005](https://link.springer.com/book/10.1007/b106715);
[Angelopoulos & Bates, arXiv:2107.07511](https://arxiv.org/abs/2107.07511)), and
for retrieval it collapses to one honestly-calibrated cosine threshold: the
nonconformity of a (query, relevant) pair is `1 − cos`; calibrate on `n` labeled
pairs with the rank-corrected quantile `q̂ = ⌈(n+1)(1−α)⌉`-th smallest score; then
return every item with `cos ≥ 1 − q̂`. The set is **adaptive for free** — a clear
winner gives a set of one, a pile of near-ties a big set, so set *size* is the
per-query confidence signal — and the guarantee is distribution-free (it assumes
nothing about CLIP, only that queries are exchangeable). It is finite-sample and
*marginal*: `1−α ≤ coverage ≤ 1−α+1/(n+1)`, so on the 14-image gallery coverage
moves in ~7% steps and we say so rather than truncate the set to look tidy —
truncating would break the promise. Where the gate is the trust boundary on the
*explanation*, conformal is the trust boundary on the *retrieval itself*: both
would rather say less than say something they can't stand behind.

**`judge.py` adds a third trust boundary — the verdict — by refusing to trust
one model's one score.** It convenes a **council of LLM judges** (`js/judge.js`
twin), each a different rubric (relevance, specificity, faithfulness), and
aggregates them the way the [panel-of-LLM-evaluators](https://arxiv.org/abs/2404.18796)
literature (Verga et al. 2024) recommends: several small judges cancel one big
judge's bias. Every judge's raw reply passes the same *gate* discipline —
`parse_score()` extracts a number in `[0,1]` or the judge **abstains** (a
scoreless vote is dropped, never guessed) — and the council takes a
confidence-weighted mean plus a **consensus** = `1 − (max − min)`. Then, exactly
like conformal, it **abstains rather than pretend**: no quorum of valid scores,
or a panel too split (a *hung jury*), yields no ruling instead of a confident
average over a coin flip. `judge.py` ships a model-free heuristic judge (the CLI
runs the mechanism on the committed gallery, like `dcn.py`'s untrained demo);
the live page's **⚖️ council** button runs three real in-browser `SmolLM2`
judges through the identical gate and aggregation.

**`debate.py` makes those judges *talk*.** A council votes independently; real
deliberation is agents arguing — each updating toward peers within its confidence
bound (**bounded-confidence dynamics**, Hegselmann–Krause 2002; multi-agent
debate as evaluator, Du et al. 2023). `debate.py` (`debate.js` twin) runs the
council's judges as agents that either **converge to a consensus** or **split
into factions that won't move each other** — a contested case surfaced with the
dissenter *named*, not averaged away. It resolves the ambiguity a vote leaves on
the table in both directions: a lone tag-fluke agent that can't be moved becomes
a named faction, while a moderate agent can *bridge* two extremes the council
would have hung on. The live **🗣️ let them debate** button animates the opinions
converging or splitting across rounds; as evaluation it converges the easy cases
and isolates the contested ones. Deterministic, model-free, a twin pinned by CI.

**`trust.py` is the capstone: it composes those boundaries into ONE verdict.**
Reading four panels to decide whether to believe a result is the user's job, so
`trust.py` (`trust.js` twin) does it — composing the signals the SAME way the
council composes its judges, one level up: *a council of gates.* Four different
lenses on the top result — **gate** (match strength), **conformal** (clears the
coverage bar?), **council** (judges concur?), **margin** (decisively ahead? —
Hermes' separation signal) — each contributing a trust in `[0,1]` or abstaining.
`compose()` takes a confidence-weighted mean and a consensus, and — like every
stage it aggregates — **abstains rather than average over a contradiction**: a
split panel (spread too wide) or too few voting lenses yields no ruling, and it
can't claim *high* trust while half the evidence abstained. When strength,
calibration, consensus and separation all agree, trust is high; when a strong
cosine meets a hung council, it lands at medium or abstains. On the live page a
trust headline sits atop the explanation and folds in the council the moment it's
convened. The honesty boundaries — gate, conformal, council — now resolve to a
single answer to *"how much should I believe this?"*, and that answer, too, would
rather abstain than bluff.

**`reason.py` is the layer that turns all of it into a decision.** A trust score
still isn't an action. `reason.py` (`reason.js` twin) walks the whole pipeline in
order, turns each stage's output into a **premise → conclusion** with a status
(ok · caution · stop), and ends at a **consequence** — *show it · show with a
caveat · withhold* — with the reason and the downstream effect. It is the one
place the entire stack becomes legible end to end: for a clean query every step
is ✓ and the consequence is "show it as the answer"; for a tag-fluke query the
chain is explicit about *where* it breaks (council hung, debate contested) and
resolves to "show with a caveat" or "withhold and ask." The live **🧠 trace the
reasoning** button draws the chain as a flow and redraws it as the council and
debate run. Model-free, deterministic, a twin pinned by CI — the reasoning on top
of everything, ending in what to actually do. And **▶ watch the agents work**
(`js/trace.js`, page-only) *plays* the pipeline live — each node pending → running
→ resolved, the council's LLM judges streaming in one at a time as their calls
return — the one place you watch the whole stack execute in real time.

**`drift.py` closes the loop: it watches whether any of this still holds.** Every
guarantee here assumes live queries stay *exchangeable* with the calibration
gallery; drift is when that quietly breaks. `drift.py` (`drift.js` twin) monitors
a stream with three distribution-free detectors — **PSI**, **KS**, and the
repo-native **conformal coverage** (calibrate a bar on the reference, watch it
fail on live data — conformal detects its own drift for free) — and rules
*stable / shift / DRIFT*, sorting each window into positive and failure cases.
Unlike everything above, it isn't a per-query check: `.github/workflows/drift.yml`
runs it **periodically** (a daily cron) and on every corpus change, freezes a
reference at calibration time, uploads an HTML dashboard, and **fails the run if
the live data drifts** — the monitoring/observability layer a real deployment
lives or dies on, in the same distribution-free spirit as the rest.

## The one-sentence version

**Retrieve cheap over billions (two towers + ANN + PQ), rank rich over the
surviving hundreds (DCN's query×item cross, learned live from your 👍/👎 on-device),
then explain the result, gate the explanation so it can't lie, quote a
coverage-guaranteed set, let a council of LLM judges rule and debate, compose it
all into one trust verdict, and reason from there to a decision — or abstain** —
the same four stages
Google/YouTube/Pinterest run, shrunk to 14 images you can read end to end in an
afternoon.

## Reading order by stage

| stage | files | run it |
|---|---|---|
| encode | `embedder.py` `models.py` `fusion.py` `templates.py` | `python3 features.py images/cat.jpg` |
| retrieve | `search.py` `ann.py` `hnsw.py` `diskann.py` `pq.py` `quantize.py` `scale.py` `cascade.py` | `python3 ann.py` · `python3 hnsw.py` · `python3 diskann.py` · `python3 pq.py` |
| rank | `dcn.py` `learn2rank.py` `hermes.py` | `python3 dcn.py --image images/004_cat.jpg` · `python3 learn2rank.py` |
| explain+gate | `explain.py` `conformal.py` `judge.py` `trust.py` `agent.py` `debate.py` `orchestrate.py` `flow.py` | `python3 explain.py --image images/004_cat.jpg` · `python3 conformal.py --json docs/db.json` · `python3 judge.py … --image images/004_cat.jpg` · `python3 trust.py … --image images/004_cat.jpg` · `python3 orchestrate.py --json docs/db.json --eval` · `python3 flow.py --json docs/db.json --demo-contract` |

Sources: DCN v2 [arXiv:2008.13535], DCN v1 [arXiv:1708.05123], RankNet (Burges et
al., ICML 2005), YouTube two-stage (Covington et al., RecSys 2016), Wide&Deep
[arXiv:1606.07792], FAISS [arXiv:2401.08281], ScaNN [arXiv:1908.10396], Matryoshka
[arXiv:2205.13147], AIS [arXiv:2112.12870], "Why do These Match?"
[arXiv:1905.10797], conformal prediction (Vovk et al. 2005; Angelopoulos & Bates
[arXiv:2107.07511]), LLM-as-a-judge (Zheng et al. [arXiv:2306.05685]), panel of
LLM evaluators (Verga et al. [arXiv:2404.18796]).
