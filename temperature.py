"""Softmax with a logit scale: turning similarity scores into probabilities.

pipeline: similarity scores ──► [temperature] ──► probabilities, sum = 1

CLIP's raw cosine scores huddle in a narrow band — on the sample gallery,
almost everything scores within ~0.15 of everything else — so a sorted list
hides how CONFIDENT the model is. Softmax spreads scores into probabilities,
and one number decides how much:

    scale →   0    every candidate equally likely (infinite temperature)
    scale →   ∞    winner takes all (temperature zero)

scale is 1/temperature. The remarkable part: CLIP LEARNS this number during
training (its `logit_scale`, ≈100 for this checkpoint) — ×100 stretches that
narrow band wide enough for softmax to separate.

Run me:  python3 temperature.py     (uses the committed docs/db.json — no model)
"""
import numpy as np

LOGIT_SCALE = 100.0  # what openai/clip-vit-base-patch32 learned in training


def softmax(scores, scale: float = LOGIT_SCALE) -> np.ndarray:
    """exp(scale·s) / Σ exp(scale·s), with the max subtracted for stability.

    Subtracting the max changes nothing (softmax is shift-invariant) but
    keeps exp() from overflowing — the standard trick, worth knowing.
    """
    z = np.asarray(scores, dtype=np.float64) * scale
    e = np.exp(z - z.max())
    return e / e.sum()


if __name__ == "__main__":
    import db

    items = db.load_json_gallery()
    cat = next(it for it in items if "cat" in it["path"])
    scores = [float(it["image_emb"] @ cat["text_emb"]) for it in items]
    print(f"query: the stored caption embedding of {cat['path']!r}")
    print(f"raw scores: min {min(scores):+.3f}  max {max(scores):+.3f}"
          "  — a narrow band; now watch the scale spread it\n")
    for scale in (1, 20, LOGIT_SCALE):
        p = softmax(scores, scale)
        top = ", ".join(f"{items[i]['path'].split('/')[-1]} {p[i]:.1%}"
                        for i in np.argsort(p)[::-1][:3])
        print(f"  scale {scale:>5g}: {top}")
