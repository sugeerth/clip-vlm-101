# Recommendation — two towers and a ranker that learns you

*Offline item embeddings, mean-pooled likes as a user vector, and an on-device pairwise ranker.*

[← README](../README.md) · [all chapters](README.md)

Search and recommendation want the same thing: one vector per item, one vector
for the query, both in the same space. So this chapter trains no new encoder.
It reuses the 1024-d fused embeddings from
[the core pipeline](01-the-core-pipeline.md), keeps only the records the critic
in [Agents](03-agents.md) approved, and calls them items. Your likes become the
query vector. On top of that, a four-weight ranker learns your taste from
👍/👎 without leaving your browser.

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

The multi-label rows at the top are [chapter 1](01-the-core-pipeline.md)'s and
the agent loop is [chapter 3](03-agents.md)'s; this chapter is the last two rows.

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

`user_tower.py` is the serving half. The simplest honest user tower is the
renormalized **mean of the liked items' vectors** (the same mean-pooling
production systems bootstrap with), and recommending is `item_matrix @
user_vec` — one matrix multiply, no image model at serving time. Swap the
mean for a trained model later; nothing else changes. That's the point of
two towers: the expensive half is finished ahead of time.

```bash
.venv/bin/python user_tower.py images/cat.jpg images/dog.jpg  # likes → recs
```

Try it live in the **Recommend** section of
[the web demo](08-the-web-demo.md) — it runs on the stored vectors with no
model download at all.

## A ranker that learns YOU — on your device, no server

Retrieval gets you an order; a ranker decides the final one.
`dcn.py` shows that ranking *mechanism* with hand-set weights and admits the one
honest caveat: it's untrained — [ARCHITECTURE.md](../ARCHITECTURE.md) covers the
stage it sits in. `learn2rank.py` (`js/learn.js`) closes the caveat — it
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

---
← [README](../README.md) · [all chapters](README.md) · prev: [Trust — coverage guarantees, judges, drift](04-trust-and-evaluation.md) · next: [A million rows — where the database's shape starts to matter](06-a-million-rows.md)
