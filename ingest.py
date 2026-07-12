"""Ingest images into the gallery — the whole pipeline, composed.

pipeline: image ─► features (embed + tag + caption + fuse) ─► db

features.py does the extraction; this file just loops and stores.

Usage:
    python3 ingest.py images/*.jpg
    python3 ingest.py my_upload.png --caption "me hiking in Yosemite"
    python3 ingest.py images/*.jpg --tag-template "a blurry picture of a {tag}"
"""
import argparse

import db
import templates
from features import FeatureExtractor


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="image files to ingest")
    ap.add_argument("--tag-template", default=templates.DEFAULT_TAG_TEMPLATE)
    ap.add_argument("--caption-template", default=templates.DEFAULT_CAPTION_TEMPLATE)
    ap.add_argument("--caption", help="your own caption (skips the template)")
    ap.add_argument("--db", default=db.DB_PATH)
    args = ap.parse_args()
    if args.caption and len(args.paths) > 1:
        ap.error("--caption is one caption for one image — ingest that file by itself")

    fx = FeatureExtractor(args.tag_template, args.caption_template)
    print(f"model {fx.clip.model.name_or_path} on {fx.clip.device}")
    con = db.connect(args.db)

    for path in args.paths:
        try:
            r = fx.extract(path, caption=args.caption)
        except OSError as e:  # missing / truncated / not-an-image: skip, keep going
            print(f"  SKIP {path}: {e}")
            continue
        db.add_image(con, r["path"], r["caption"], r["tags"],
                     r["image_emb"], r["text_emb"], r["fused_emb"])
        print(f"  + {path}\n      tags    {r['tags']}\n      caption {r['caption']!r}")

    print(f"done — {len(db.all_images(con))} images in {args.db}")


if __name__ == "__main__":
    main()
