"""Ingest images into the gallery: embed -> auto-tag -> caption -> store.

For every image (your own upload or the bundled examples):
  1. The vision encoder turns it into a 512-d image embedding.
  2. Each vocabulary tag is phrased through the prompt template
     ("a photo of a {tag}"), embedded, and scored against the image.
     The top-scoring tags become the image's meta tags. Zero-shot, no training.
  3. The tags are folded into a caption, which the text encoder embeds too.
  4. image and text embeddings are CONCATENATED into one fused vector and
     all three are written to SQLite.

Usage:
    python3 ingest.py images/*.jpg
    python3 ingest.py my_upload.png --caption "me hiking in Yosemite"
    python3 ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
"""
import argparse

import numpy as np

import db
import templates
from embedder import ClipEmbedder

TOP_TAGS = 5


def fuse(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Concatenate the two unit vectors into one 1024-d vector.

    Dividing by sqrt(2) makes the result unit-length again, so a dot product
    with a fused query equals the AVERAGE of the image and text similarities.
    """
    return np.concatenate([image_emb, text_emb]) / np.sqrt(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="image files to ingest")
    ap.add_argument("--tag-template", default=templates.DEFAULT_TAG_TEMPLATE)
    ap.add_argument("--caption-template", default=templates.DEFAULT_CAPTION_TEMPLATE)
    ap.add_argument("--caption", help="your own caption (skips the template)")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()

    clip = ClipEmbedder()
    print(f"model {clip.model.name_or_path} on {clip.device}")
    con = db.connect(args.db)

    # Embed the whole tag vocabulary once, phrased through the prompt template.
    tag_embs = clip.embed_texts(templates.tag_prompts(args.tag_template))

    for path in args.paths:
        image_emb = clip.embed_images([path])[0]

        # Zero-shot tagging: which tag sentences sit closest to this image?
        scores = tag_embs @ image_emb
        best = np.argsort(scores)[::-1][:TOP_TAGS]
        tags = [templates.TAG_VOCABULARY[i] for i in best]

        caption = args.caption or templates.caption_for(tags, args.caption_template)
        text_emb = clip.embed_texts([caption])[0]

        db.add_image(con, path, caption, tags, image_emb, text_emb,
                     fuse(image_emb, text_emb))
        print(f"  + {path}\n      tags    {tags}\n      caption {caption!r}")

    print(f"done — {len(db.all_images(con))} images in {args.db}")


if __name__ == "__main__":
    main()
