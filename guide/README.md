# The guide

The README is the front door: what this is, and how to run it in three
commands. This is the rest — one chapter per idea, each runnable.

| # | chapter | what it teaches |
|---|---|---|
| 0 | [Every file in the repo, tiered](00-every-file.md) | The full inventory — start with the eight core files, then pick a chapter. |
| 1 | [The core pipeline — image in, searchable vectors out](01-the-core-pipeline.md) | features.py as a one-call API, why the two embeddings are concatenated, and how dynamic multi-label tags work. |
| 2 | [Understanding CLIP — and squeezing more out of it](02-understanding-clip.md) | The seven model-free lessons: temperature, the modality gap, vector arithmetic, quantization, IVF. |
| 3 | [Agents — propose, critique, argue, explain](03-agents.md) | The self-critiquing embedding agent, the agentic searcher, multi-agent debate, and the reasoning trace. |
| 4 | [Trust — coverage guarantees, judges, drift](04-trust-and-evaluation.md) | Conformal prediction, a council of LLM judges, one composed trust verdict, and drift detection on a stream. |
| 5 | [Recommendation — two towers and a ranker that learns you](05-recommendation.md) | Offline item embeddings, mean-pooled likes as a user vector, and an on-device pairwise ranker. |
| 6 | [A million rows — where the database's shape starts to matter](06-a-million-rows.md) | Real measured recall and latency for exact scan, IVF, int8 and OPQ at 690k vectors. |
| 7 | [Grow the gallery — and swap in a newer model](07-grow-your-gallery.md) | One command to crawl ~114 topics into a bigger gallery, plus the model registry. |
| 8 | [The web demo — the same pipeline, in your browser](08-the-web-demo.md) | How transformers.js runs the identical model client-side, and how the JS mirrors the Python. |

Also worth reading: [ARCHITECTURE.md](../ARCHITECTURE.md) — the four
stages a real billion-scale visual search runs, and which file demonstrates each.

← [back to the README](../README.md)
