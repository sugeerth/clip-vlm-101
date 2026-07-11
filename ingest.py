"""Ingest images into the gallery — the whole pipeline, composed.

pipeline: image ─► embedder ─► tagger ─► templates.caption_for ─► fusion ─► db

Each step lives in the module named after it; this file just wires them up.

Usage:
    python3 ingest.py images/*.jpg
    python3 ingest.py my_upload.png --caption "me hiking in Yosemite"
    python3 ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
"""
import argparse

import db
import fusion
import tagger
import templates
from embedder import ClipEmbedder


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
        tags = tagger.top_tags(image_emb, tag_embs, templates.TAG_VOCABULARY)
        caption = args.caption or templates.caption_for(tags, args.caption_template)
        text_emb = clip.embed_texts([caption])[0]
        db.add_image(con, path, caption, tags, image_emb, text_emb,
                     fusion.fuse(image_emb, text_emb))
        print(f"  + {path}\n      tags    {tags}\n      caption {caption!r}")

    print(f"done — {len(db.all_images(con))} images in {args.db}")


if __name__ == "__main__":
    main()
