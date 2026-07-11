"""Zero-shot meta tags: classification with no training, just dot products.

pipeline: image_emb ──► [tagger] ──► meta tags ["cat", "pet", ...]

CLIP scores how well each candidate sentence ("a photo of a cat",
"a photo of a dog", ...) describes an image. Sort the scores, keep the
winners — that is the entire tagger. New tags need new words, not training.
"""
import numpy as np


def top_tags(image_emb, tag_embs, vocabulary, k: int = 5):
    """The k vocabulary tags whose prompt sentences best match the image.

    image_emb: (512,) unit vector of the image.
    tag_embs:  (len(vocabulary), 512) unit vectors of the templated prompts,
               in the same order as vocabulary.
    """
    scores = tag_embs @ image_emb            # one dot product per candidate
    best = np.argsort(scores)[::-1][:k]
    return [vocabulary[i] for i in best]
