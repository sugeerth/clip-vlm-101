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
        └───────────────┬────────────────┘
                        │  the final ordered results
        ┌───────────────▼────────────────┐
        │ 3 · EXPLAIN + GATE   say WHY,   │  explain.py (explain.js twin)
        │   never lie. grounded template  │  agent.py (the same verify-before-
        │   or LLM → hallucination gate   │  publish idea, on the write path)
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
  [ScaNN](https://arxiv.org/abs/1908.10396); HNSW/DiskANN for graph and
  on-SSD variants).
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

## The one-sentence version

**Retrieve cheap over billions (two towers + ANN + PQ), rank rich over the
surviving hundreds (DCN's query×item cross), then explain the result and gate
the explanation so it can't lie** — the same four stages Google/YouTube/Pinterest
run, shrunk to 14 images you can read end to end in an afternoon.

## Reading order by stage

| stage | files | run it |
|---|---|---|
| encode | `embedder.py` `models.py` `fusion.py` `templates.py` | `python3 features.py images/cat.jpg` |
| retrieve | `search.py` `ann.py` `pq.py` `quantize.py` `scale.py` | `python3 ann.py` · `python3 pq.py` |
| rank | `dcn.py` `hermes.py` | `python3 dcn.py --image images/004_cat.jpg` |
| explain+gate | `explain.py` `agent.py` | `python3 explain.py --image images/004_cat.jpg` |

Sources: DCN v2 [arXiv:2008.13535], DCN v1 [arXiv:1708.05123], YouTube two-stage
(Covington et al., RecSys 2016), Wide&Deep [arXiv:1606.07792], FAISS
[arXiv:2401.08281], ScaNN [arXiv:1908.10396], Matryoshka [arXiv:2205.13147],
AIS [arXiv:2112.12870], "Why do These Match?" [arXiv:1905.10797].
