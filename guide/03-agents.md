# Agents — propose, critique, argue, explain

*The self-critiquing embedding agent, the agentic searcher, multi-agent debate, and the reasoning trace.*

[← README](../README.md) · [all chapters](README.md)

Four agents, two paths through the pipeline:

- **write path** — `agent.py` makes an image's features pass a critic before they are stored.
- **read path** — `hermes.py` treats your query as a draft, `debate.py` makes the judges argue instead of vote, and `reason.py` walks the finished chain and says what to do about it.
- **same shape both ways** — propose, critique, publish only what survives; abstain when nothing does.

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

```bash
.venv/bin/python hermes.py "a fluffy animal"          # the agentic version, with a trace
python3 hermes.py --image pizza --json docs/db.json   # image query; no model needed
```

A text query embeds the phrasings, so it needs the model; `--image` reuses the
stored vectors and runs model-free.

The live search box at https://sugeerth.github.io/clip-vlm-101/ runs Hermes
on every text query — the muted "🪽 hermes chose …" line under the results
expands into the full trace of what it tried and why.

## Multiple agents that talk — deliberation, not just a vote

The council of LLM judges in [Trust](04-trust-and-evaluation.md) polls its judges
**independently** and averages them. Real
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

**The honest footnote.** That same `--eval` run prints one more line:
`verdict changed vs independent council : 0/14`. Three rounds of agent-to-agent
argument did not overturn a single ruling the cheap independent council had
already reached — it bought a better *account* of where the panel disagreed, not
different answers. That is the coordination tax, measured on this gallery, and
it is worth knowing before you put a debate on a read path.

## The reasoning layer — trace every step, then map the consequence

Every stage produces a signal and knows how to abstain — the calibrated coverage
bar, the council and the composed trust verdict all live in
[Trust](04-trust-and-evaluation.md) — but a pile of signals isn't a decision.
`reason.py` (`js/reason.js`) is the layer on top that **reasons
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
on `councilWithLLM`). Page-only, like the other browser-only files
`viz.js`/`motion.js`/`tour.js` ([chapter 0](00-every-file.md) lists them) — there's no
"live" in a batch script — it's the one place you see the whole stack *execute*,
not just its result.

---
← [README](../README.md) · [all chapters](README.md) · prev: [Understanding CLIP — and squeezing more out of it](02-understanding-clip.md) · next: [Trust — coverage guarantees, judges, drift](04-trust-and-evaluation.md)
