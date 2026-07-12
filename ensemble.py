"""Prompt ensembling: the free accuracy upgrade from the CLIP paper.

pipeline: vocabulary × MANY templates ──► [ensemble] ──► one better vector per tag

One phrasing is a noisy estimate of what a tag means. "a photo of a cat"
leans photographic; a drawing of a cat scores lower than it should. So ask
the same question many ways and AVERAGE the answers:

    "a photo of a cat"          ┐
    "a close-up photo of a cat" ├─► text encoder ─► mean ─► renormalize
    "a drawing of a cat"        │        = ONE sturdier "cat" vector
    "a photo of many cats"      ┘

Averaging unit vectors cancels the phrasing noise and keeps the shared
meaning; renormalizing makes the result a unit vector again, so every
downstream dot product works unchanged. The CLIP paper's 80-template
ensemble lifts ImageNet zero-shot accuracy by ~3.5% — with prompt
engineering on top, ~5%. Same model, same weights, better answers.

Cost: you embed vocabulary × templates sentences ONCE (text only, cached
by the caller), then tagging is exactly as fast as before — the ensemble
collapses into the same (n_tags, 512) matrix.

    from ensemble import ensemble_tag_embs
    tag_embs = ensemble_tag_embs(clip, templates.TAG_VOCABULARY)
    # drop-in replacement anywhere a (n_tags, 512) prompt matrix is used

Or from the shell:  python3 features.py images/*.jpg --ensemble
"""
import numpy as np

import templates

# A curated slice of the CLIP paper's 80 ImageNet templates — enough
# phrasing diversity to average out the noise, few enough to read.
ENSEMBLE_TEMPLATES = [
    "a photo of a {tag}",
    "a close-up photo of a {tag}",
    "a photo of many {tag}",
    "a photo of the large {tag}",
    "a photo of the small {tag}",
    "a bright photo of a {tag}",
    "a drawing of a {tag}",
    "a {tag} in the wild",
]


def ensemble_tag_embs(clip, vocabulary=templates.TAG_VOCABULARY,
                      ensemble=ENSEMBLE_TEMPLATES):
    """One averaged, renormalized unit vector per tag: (len(vocabulary), 512).

    Embeds every template × tag sentence in one batch, then averages
    across templates. Drop-in replacement for single-template tag_embs.
    """
    prompts = [templates.fill(t, tag=tag) for tag in vocabulary for t in ensemble]
    embs = clip.embed_texts(prompts)                      # (tags*templates, 512)
    mean = embs.reshape(len(vocabulary), len(ensemble), -1).mean(axis=1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)
