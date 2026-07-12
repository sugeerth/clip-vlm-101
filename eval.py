"""Measure the tagger: is that optimization actually helping? Prove it.

pipeline: images + ground truth ──► [eval] ──► top-1 / top-5 hit rates

Every optimization claim in this repo should survive contact with a number.
This harness scores zero-shot tagging against a tiny ground-truth table
(the sample gallery, labeled by hand below) and prints hit rates for the
plain single-template tagger and the prompt ENSEMBLE side by side:

    python3 eval.py images/*.jpg

    tagger              top-1    top-5
    single template     11/14    14/14
    ensemble (8 tmpl)   12/14    14/14

A hit means one of the image's acceptable tags appears in the top-1 / top-5
predictions. Fourteen images is a smoke-test-sized benchmark — enough to
catch regressions and show the mechanics; swap in your own images and
truth table for anything serious.
"""
from pathlib import Path

import numpy as np

import ensemble
import tagger
import templates

# Acceptable vocabulary tags per sample image (keyed by filename stem).
# More than one tag can be right — "cat" is also honestly a "pet".
GROUND_TRUTH = {
    "apple": {"apple", "fruit"},
    "bear": {"bear", "animal", "wildlife"},
    "bicycle": {"bicycle", "vehicle"},
    "castle": {"castle", "palace", "landmark"},
    "cat": {"cat", "pet", "animal"},
    "dog": {"dog", "pet", "animal"},
    "eiffel_tower": {"tower", "landmark", "architecture"},
    "london": {"city", "tower", "landmark", "architecture", "bridge"},
    "mountains": {"mountain", "landscape"},
    "parrot": {"parrot", "bird"},
    "pizza": {"pizza", "food"},
    "pluto": {"planet", "moon", "space"},
    "sunflower": {"sunflower", "flower"},
    "waterfall": {"waterfall", "river"},
}


def hit_rates(image_embs, truths, tag_embs, vocabulary, k: int = 5):
    """(top-1 hits, top-k hits, n) for one prompt matrix over many images."""
    top1 = topk = 0
    for emb, truth in zip(image_embs, truths):
        tags = tagger.top_tags(emb, tag_embs, vocabulary, k)
        top1 += tags[0] in truth
        topk += bool(truth.intersection(tags))
    return top1, topk, len(truths)


def evaluate(clip, paths, truth_table=GROUND_TRUTH, vocabulary=templates.TAG_VOCABULARY):
    """Compare single-template vs ensemble tagging on labeled images."""
    known = [(p, truth_table[Path(p).stem.lower()]) for p in paths
             if Path(p).stem.lower() in truth_table]
    if not known:
        raise ValueError("no images matched the ground-truth table")
    image_embs = clip.embed_images([p for p, _ in known])
    truths = [t for _, t in known]
    single = clip.embed_texts(templates.tag_prompts(vocabulary=vocabulary))
    ens = ensemble.ensemble_tag_embs(clip, vocabulary)
    return {
        "n": len(known),
        "skipped": len(paths) - len(known),
        "single template": hit_rates(image_embs, truths, single, vocabulary),
        f"ensemble ({len(ensemble.ENSEMBLE_TEMPLATES)} tmpl)":
            hit_rates(image_embs, truths, ens, vocabulary),
    }


if __name__ == "__main__":
    import argparse

    from embedder import ClipEmbedder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", help="labeled sample images")
    args = ap.parse_args()

    results = evaluate(ClipEmbedder(), args.images)
    if results["skipped"]:
        print(f"({results['skipped']} image(s) not in GROUND_TRUTH — skipped)")
    print(f"\n{'tagger':<22} {'top-1':>7} {'top-5':>7}")
    for name, val in results.items():
        if name in ("n", "skipped"):
            continue
        t1, tk, n = val
        print(f"{name:<22} {t1:>4}/{n:<3} {tk:>4}/{n:<3}")
