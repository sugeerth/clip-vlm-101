"""Ingest images into the gallery — the whole pipeline, composed.

pipeline: image ─► features (embed + tag + caption + fuse) ─► db

features.py does the extraction; this file just loops and stores.

Usage:
    python3 ingest.py images/*.jpg
    python3 ingest.py my_upload.png --caption "me hiking in Yosemite"
    python3 ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
    python3 ingest.py images/*.jpg --ensemble   # average all built-in templates
"""
import argparse

from PIL import Image

import db
import templates
from features import FeatureExtractor


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="image files to ingest")
    ap.add_argument("--tag-template", action="append",
                    help="prompt template for tags; repeat the flag to ensemble")
    ap.add_argument("--ensemble", action="store_true",
                    help="tag with all built-in templates, averaged (CLIP-paper trick)")
    ap.add_argument("--caption-template", default=templates.DEFAULT_CAPTION_TEMPLATE)
    ap.add_argument("--caption", help="your own caption (skips the template)")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if args.caption and len(args.paths) > 1:
        ap.error("--caption is one caption for one image — ingest that file by itself")
    tag_template = (templates.ENSEMBLE_TAG_TEMPLATES if args.ensemble
                    else args.tag_template or templates.DEFAULT_TAG_TEMPLATE)

    # weed out unreadable files first, so one bad path can't spoil a batch
    paths = []
    for p in args.paths:
        try:
            Image.open(p).verify()
            paths.append(p)
        except OSError as e:  # missing / truncated / not-an-image: skip, keep going
            print(f"  SKIP {p}: {e}")

    fx = FeatureExtractor(tag_template, args.caption_template)
    print(f"model {fx.clip.model.name_or_path} on {fx.clip.device}")
    con = db.connect(args.db)

    for r in fx.extract_batch(paths, caption=args.caption):
        db.add_image(con, r["path"], r["caption"], r["tags"],
                     r["image_emb"], r["text_emb"], r["fused_emb"])
        print(f"  + {r['path']}\n      tags    {r['tags']}\n      caption {r['caption']!r}")

    print(f"done — {db.count_images(con)} images in {args.db}")


if __name__ == "__main__":
    main()
