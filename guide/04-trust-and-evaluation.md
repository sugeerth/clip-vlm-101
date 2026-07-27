# Trust — coverage guarantees, judges, drift

*Conformal prediction, a council of LLM judges, one composed trust verdict, and drift detection on a stream.*

[← README](../README.md) · [all chapters](README.md)

A search always returns something. The harder skill is knowing when **not** to
answer. Four files add that skill, one lens at a time, and every one of them can
abstain instead of guessing:

- **`conformal.py`** — a coverage bar you can check, or *"no confident match"*.
- **`judge.py`** — a panel of rubric judges, or *no quorum* / *hung jury*.
- **`trust.py`** — the composer: one verdict out of every lens, or an abstain
  when the lenses disagree.
- **`drift.py`** — the monitor: whether live data still matches the data those
  lenses were calibrated on.

Read them in that order — each lens, then the composer, then the monitor. Those
four commands run from the repo root against the committed gallery, so none of
them download a model.

## A coverage guarantee, or an honest abstain

`conformal.py` (`js/conformal.js`) makes a promise you can check: given a
confidence level (say 90%), it returns the **smallest set of results that
contains the true match at least 90% of the time** — or, when nothing clears the
bar, it *abstains* ("no confident match") rather than guess. This is **split
conformal prediction** (Vovk et al. 2005;
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
  the same discipline as the hallucination gate in [`explain.py`](../explain.py).
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
like the untrained `dcn.py` demo in [chapter 5](05-recommendation.md). On the
live demo the **⚖️ convene a council of LLM judges** button under the explanation
runs three *real* in-browser `SmolLM2-135M` judges (via transformers.js, nothing
leaves your machine), pipes each reply through the identical gate, and shows
every judge's score, the weighted verdict, and the consensus — or an honest
abstain. The aggregation math is a Python/JS twin, pinned byte-for-byte by CI.

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

The margin lens is the retrieval-margin critique Hermes runs on the read path —
see [chapter 3](03-agents.md).

`compose()` takes a confidence-weighted mean, measures `consensus = 1 − (max −
min)`, and refuses to rule when the lenses **split** (spread too wide) or too few
weigh in — and it **can't call "high" while half the evidence abstained** (a
participation cap). When strength, calibration, consensus and separation all
agree, trust is *high*; when a strong cosine meets a hung council, it lands at
*medium* or abstains — honestly.

```bash
python3 trust.py --json docs/db.json --image images/004_cat.jpg
#   005_dog.jpg  gate 1.00  confm 0.71  counc 0.73  marg 1.00  →  high     (all four agree)
#   …bicycle     gate 0.40  confm  —    counc 0.05  marg 0.07  →  low
#   apple→…      gate 0.40  confm 0.53  counc 0.03  marg 0.13  →  abstain  (split decision)
```

On the live demo a **trust headline** sits atop the explanation, composing the
three always-available lenses (gate + conformal + margin) immediately and folding
in the council's verdict the moment you convene it — one honest answer to *"how
much should I believe this?"*, with the per-lens pills shown so you can see which
lenses agreed and which abstained.

## Drift detection — is the live stream still the world we calibrated for?

Every guarantee in this chapter holds only while live queries stay
**exchangeable** with the gallery they were tuned on — conformal's coverage, the
council's thresholds, the trust composer. Production breaks that quietly: the
data shifts and the honest-looking numbers keep printing. `drift.py`
(`js/drift.js`) is the monitor that watches for it, on a stream of results, with
three **distribution-free** detectors on a quality signal (the same-tag match
similarity):

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

## The benchmark you run before and after a change — `eval.py`

Those lenses judge one answer. `eval.py` judges the *tagger*: it scores zero-shot
tagging against a hand-labeled truth table for the 14 sample images and prints
top-1/top-5 hit rates for the plain single template and the prompt ensemble side
by side. Run it before an optimization and again after — that before/after pair
is what turns "this helps" into a number.

```bash
python3 eval.py images/*.jpg        # measure: template vs ensemble
#   tagger              top-1    top-5
#   single template     11/14    14/14
#   ensemble (8 tmpl)   12/14    14/14
```

Unlike the four lenses above it embeds images, so it downloads the model (~600 MB
on first run). Pass `--model` to score two brains on the same images —
[chapter 7](07-grow-your-gallery.md).

---
← [README](../README.md) · [all chapters](README.md) · prev: [Agents — propose, critique, argue, explain](03-agents.md) · next: [Recommendation — two towers and a ranker that learns you](05-recommendation.md)
