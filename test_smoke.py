"""Fast sanity checks — no model download needed. Run: python3 test_smoke.py"""
import os
import tempfile

import numpy as np

import db
import fusion
import tagger
import templates
from search import score


class HashClip:
    """Content-sensitive fake encoder: same text -> same unit vector, no model.
    (The stub inside test_feature_extractor ignores its input; this one
    doesn't, which is what ensembling and batching tests need.)"""
    def embed_texts(self, texts):
        import zlib
        rows = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(str(t).encode()))
            v = rng.normal(size=512)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float32)
    embed_images = embed_texts  # a path is just a string to hash


def test_templates():
    assert templates.fill("a photo of a {tag}", tag="cat") == "a photo of a cat"
    prompts = templates.tag_prompts()
    assert len(prompts) == len(templates.TAG_VOCABULARY)
    assert prompts[0] == f"a photo of a {templates.TAG_VOCABULARY[0]}"
    assert templates.caption_for(["cat", "pet"]) == "a photo of cat, pet"
    assert templates.ENSEMBLE_TAG_TEMPLATES[0] == templates.DEFAULT_TAG_TEMPLATE
    for t in templates.ENSEMBLE_TAG_TEMPLATES:
        assert templates.fill(t, tag="cat").count("cat") == 1


def test_prompt_ensembling():
    from features import FeatureExtractor
    vocab = ["cat", "dog", "car"]
    tpls = [templates.DEFAULT_TAG_TEMPLATE, "a drawing of a {tag}"]
    # a 1-template ensemble is exactly that template
    one = FeatureExtractor(vocabulary=vocab, clip=HashClip()).tag_embs
    same = FeatureExtractor(tag_template=[tpls[0]], vocabulary=vocab, clip=HashClip()).tag_embs
    assert np.allclose(one, same)
    # a 2-template ensemble is the re-normalized mean of the two singles
    two = FeatureExtractor(tag_template=tpls, vocabulary=vocab, clip=HashClip()).tag_embs
    a = FeatureExtractor(tag_template=tpls[0], vocabulary=vocab, clip=HashClip()).tag_embs
    b = FeatureExtractor(tag_template=tpls[1], vocabulary=vocab, clip=HashClip()).tag_embs
    mean = (a + b) / 2
    assert two.shape == (3, 512)
    assert np.allclose(np.linalg.norm(two, axis=1), 1.0, atol=1e-6)
    assert np.allclose(two, mean / np.linalg.norm(mean, axis=1, keepdims=True), atol=1e-6)


def test_extract_batch():
    from features import FeatureExtractor
    fx = FeatureExtractor(clip=HashClip())
    paths = ["./a.jpg", "b.jpg", "c.jpg"]
    batch = fx.extract_batch(paths, batch_size=2)  # 3 paths -> a chunk boundary
    assert [r["path"] for r in batch] == ["a.jpg", "b.jpg", "c.jpg"]  # normpathed
    for r, single in zip(batch, (fx.extract(p) for p in paths)):
        assert r["tags"] == single["tags"] and r["caption"] == single["caption"]
        assert np.allclose(r["image_emb"], single["image_emb"])
        assert np.allclose(r["fused_emb"], single["fused_emb"])


def test_top_tags():
    vocab = ["cat", "dog", "car"]
    tag_embs = np.eye(3, 4)                       # 3 fake unit prompt vectors in 4-d
    image_emb = np.array([0.1, 0.9, 0.0, 0.0])    # closest to "dog", then "cat"
    assert tagger.top_tags(image_emb, tag_embs, vocab, k=2) == ["dog", "cat"]


def test_fusion_math():
    rng = np.random.default_rng(0)
    a = rng.normal(size=512); a /= np.linalg.norm(a)
    b = rng.normal(size=512); b /= np.linalg.norm(b)
    fused = fusion.fuse(a, b)
    assert fused.shape == (1024,)
    assert abs(np.linalg.norm(fused) - 1.0) < 1e-6  # still unit-length
    # fused dot fused-query == mean of the two per-mode similarities
    q = rng.normal(size=512); q /= np.linalg.norm(q)
    item = {"image_emb": a, "text_emb": b, "fused_emb": fused}
    expected = (score(item, q, "image") + score(item, q, "text")) / 2
    assert abs(score(item, q, "fused") - expected) < 1e-6


def test_feature_extractor():
    """FeatureExtractor with a stub CLIP — checks shapes, tags, and caption."""
    from features import FeatureExtractor

    class StubClip:  # deterministic fake encoder, no model download
        def embed_texts(self, texts):
            rng = np.random.default_rng(42)
            v = rng.normal(size=(len(texts), 512))
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
        def embed_images(self, paths):
            return self.embed_texts(paths)

    fx = FeatureExtractor(clip=StubClip())
    r = fx.extract("fake.jpg")
    assert r["image_emb"].shape == (512,) and r["text_emb"].shape == (512,)
    assert r["fused_emb"].shape == (1024,)
    assert abs(np.linalg.norm(r["fused_emb"]) - 1.0) < 1e-5
    assert len(r["tags"]) == 5 and all(t in templates.TAG_VOCABULARY for t in r["tags"])
    assert r["caption"] == templates.caption_for(r["tags"])


def test_pca_2d():
    from export_web import pca_2d
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 16))
    coords, mean, components = pca_2d(X)
    assert coords.shape == (10, 2) and mean.shape == (16,) and components.shape == (2, 16)
    # projecting a vector with mean+components reproduces its coordinate
    assert np.allclose((X[0] - mean) @ components.T, coords[0])
    # a 1-image gallery still yields 2 components (second is zero-padded)
    coords, mean, components = pca_2d(rng.normal(size=(1, 16)))
    assert coords.shape == (1, 2) and components.shape == (2, 16)
    assert np.allclose(coords, 0)


def test_db_roundtrip():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=512), rng.normal(size=512)
    path = os.path.join(tempfile.mkdtemp(), "t.sqlite")
    con = db.connect(path)
    db.add_image(con, "x.jpg", "a photo of cat", ["cat"], a, b, fusion.fuse(a, b))
    (item,) = db.all_images(con)
    assert item["tags"] == ["cat"] and item["caption"] == "a photo of cat"
    assert np.allclose(item["image_emb"], a.astype(np.float32))
    assert item["fused_emb"].shape == (1024,)
    # same path again = replace, not duplicate (path is UNIQUE)
    db.add_image(con, "x.jpg", "a photo of dog", ["dog"], a, b, fusion.fuse(a, b))
    (item,) = db.all_images(con)
    assert item["caption"] == "a photo of dog"
    assert db.count_images(con) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  pass {name}")
    print("all smoke tests passed")
