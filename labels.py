"""Multi-label classification: every tag decides for itself.

pipeline: image_emb ──► [labels] ──► {"cat": 0.99, "pet": 0.87, ...}

tagger.py is multi-CLASS: argsort forces exactly k winners, even when the
image only contains two things — or twelve. This file is multi-LABEL: each
tag is scored independently, so the label set is DYNAMIC — sized by what is
actually in the image, not by k.

The whole trick is one comparison per tag. For "cat" we ask CLIP which
sentence describes the image better:

    "a photo of a cat"   (the tag prompt)
    "a photo"            (the neutral prompt — same sentence, no tag)

A two-way softmax over those two scores collapses to a sigmoid on their
gap, so every tag gets its own independent probability in 0..1. Gap > 0
means adding the word "cat" made the sentence match the image BETTER —
the cat is probably in there. Keep every tag above a threshold: that is
the entire multi-label classifier. No training, and new labels are new
words in the vocabulary, nothing more.
"""
import numpy as np

# CLIP's learned softmax temperature (the model ships with exp(4.6) ~= 100).
# Cosine gaps are small numbers; this scales them into sigmoid range.
LOGIT_SCALE = 100.0

# probability a tag must reach to become a label (0.5 = "beats neutral")
DEFAULT_THRESHOLD = 0.5


def label_probs(image_emb, tag_embs, neutral_emb, scale=LOGIT_SCALE):
    """One independent probability per tag: sigmoid of the (tag - neutral) gap.

    image_emb:   (512,) unit vector of the image.
    tag_embs:    (len(vocabulary), 512) unit vectors of the tag prompts.
    neutral_emb: (512,) unit vector of the neutral prompt ("a photo").

    softmax over two scores IS the sigmoid of their difference — so this
    is a per-tag binary question, not a competition between tags.
    """
    gap = tag_embs @ image_emb - neutral_emb @ image_emb
    return 1.0 / (1.0 + np.exp(-scale * gap))


def multi_label(image_emb, tag_embs, neutral_emb, vocabulary,
                threshold=DEFAULT_THRESHOLD):
    """The dynamic label set: {tag: probability}, best first, above threshold."""
    probs = label_probs(image_emb, tag_embs, neutral_emb)
    order = np.argsort(probs)[::-1]
    return {vocabulary[i]: round(float(probs[i]), 4)
            for i in order if probs[i] >= threshold}


def siglip_label_probs(image_emb, tag_embs, scale, bias):
    """SigLIP's NATIVE calibrated per-tag probability — no neutral prompt.

    SigLIP was trained with a per-pair sigmoid loss, so unlike CLIP its
    probabilities are absolute:  sigmoid(scale * cosine + bias),  with
    scale/bias read from the checkpoint (embedder exposes .logit_scale /
    .logit_bias — typically ~118 and ~-12.9). Correct labels commonly
    score 0.1-0.4, so threshold around 0.1-0.3, not 0.5.
    """
    return 1.0 / (1.0 + np.exp(-(scale * (tag_embs @ image_emb) + bias)))
