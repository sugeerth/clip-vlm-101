"""Fast sanity checks — no model download needed. Run: python3 test_smoke.py"""
import os
import tempfile

import numpy as np

import db
import templates
from ingest import fuse
from search import score


def test_templates():
    assert templates.fill("a photo of a {tag}", tag="cat") == "a photo of a cat"
    prompts = templates.tag_prompts()
    assert len(prompts) == len(templates.TAG_VOCABULARY)
    assert prompts[0] == f"a photo of a {templates.TAG_VOCABULARY[0]}"
    assert templates.caption_for(["cat", "pet"]) == "a photo of cat, pet"


def test_fuse_math():
    rng = np.random.default_rng(0)
    a = rng.normal(size=512); a /= np.linalg.norm(a)
    b = rng.normal(size=512); b /= np.linalg.norm(b)
    fused = fuse(a, b)
    assert fused.shape == (1024,)
    assert abs(np.linalg.norm(fused) - 1.0) < 1e-6  # still unit-length
    # fused dot fused-query == mean of the two per-mode similarities
    q = rng.normal(size=512); q /= np.linalg.norm(q)
    item = {"image_emb": a, "text_emb": b, "fused_emb": fused}
    expected = (score(item, q, "image") + score(item, q, "text")) / 2
    assert abs(score(item, q, "fused") - expected) < 1e-6


def test_db_roundtrip():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=512), rng.normal(size=512)
    path = os.path.join(tempfile.mkdtemp(), "t.sqlite")
    con = db.connect(path)
    db.add_image(con, "x.jpg", "a photo of cat", ["cat"], a, b, fuse(a, b))
    (item,) = db.all_images(con)
    assert item["tags"] == ["cat"] and item["caption"] == "a photo of cat"
    assert np.allclose(item["image_emb"], a.astype(np.float32))
    assert item["fused_emb"].shape == (1024,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  pass {name}")
    print("all smoke tests passed")
