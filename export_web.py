"""Export the SQLite gallery to docs/db.json (+ copy images) for the web demo.

The GitHub Pages demo is 100% static: it loads this JSON, computes the query
embedding in the browser with transformers.js, and ranks with dot products —
the exact same math as search.py.
"""
import json
import pathlib
import shutil

import db

DOCS = pathlib.Path("docs")


def main():
    items = db.all_images(db.connect())
    if not items:
        raise SystemExit("gallery.sqlite is empty — run ingest.py first")

    (DOCS / "images").mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        src = pathlib.Path(it["path"])
        shutil.copy(src, DOCS / "images" / src.name)
        out.append({
            "file": f"images/{src.name}",
            "caption": it["caption"],
            "tags": it["tags"],
            # rounding keeps db.json small; retrieval quality is unaffected
            "image_emb": [round(float(x), 5) for x in it["image_emb"]],
            "text_emb": [round(float(x), 5) for x in it["text_emb"]],
        })

    payload = {"model": "clip-vit-base-patch32", "dim": 512, "items": out}
    (DOCS / "db.json").write_text(json.dumps(payload))
    print(f"wrote docs/db.json ({len(out)} items) and copied images to docs/images/")


if __name__ == "__main__":
    main()
