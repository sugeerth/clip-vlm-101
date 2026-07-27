# The web demo — the same pipeline, in your browser

*How transformers.js runs the identical model client-side, and how the JS mirrors the Python.*

[← README](../README.md) · [all chapters](README.md)

Every chapter so far ran Python on your machine. The same pipeline also runs in
a browser tab, on the same model.

`docs/` is a static GitHub Pages site — an explorable explanation in the
[distill.pub](https://distill.pub) tradition. `export_web.py` dumps the SQLite
gallery to `docs/db.json`; the page then runs the *same* CLIP model in your
browser via [transformers.js](https://huggingface.co/docs/transformers.js).
Nothing you type or upload leaves your machine.

Live: the search box at <https://sugeerth.github.io/clip-vlm-101/>, and the full
walkthrough at
[/explore.html](https://sugeerth.github.io/clip-vlm-101/explore.html).
Highlights:

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

Light and dark themes are both hand-tuned (toggle in the header), and most
JS modules in `docs/js/` mirror a Python file of the same name — the main pairs
are mapped in [Every file in the repo, tiered](00-every-file.md).

---
← [README](../README.md) · [all chapters](README.md) · prev: [Grow the gallery — and swap in a newer model](07-grow-your-gallery.md)
