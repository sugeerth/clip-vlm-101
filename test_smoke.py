"""Fast sanity checks — no model download needed. Run: python3 test_smoke.py"""
import os
import tempfile

import numpy as np

import db
import fusion
import tagger
import templates
from search import score


def test_templates():
    assert templates.fill("a photo of a {tag}", tag="cat") == "a photo of a cat"
    prompts = templates.tag_prompts()
    assert len(prompts) == len(templates.TAG_VOCABULARY)
    assert prompts[0] == f"a photo of a {templates.TAG_VOCABULARY[0]}"
    assert templates.caption_for(["cat", "pet"]) == "a photo of cat, pet"


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


def test_ensemble():
    """ensemble.py: template-averaged tag vectors are unit-length and ordered."""
    import ensemble

    class CountingClip:
        calls = 0
        def embed_texts(self, texts):
            CountingClip.calls += 1
            rng = np.random.default_rng(7)
            v = rng.normal(size=(len(texts), 32))
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    vocab = ["cat", "dog", "car"]
    embs = ensemble.ensemble_tag_embs(CountingClip(), vocab)
    assert embs.shape == (3, 32)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)
    assert CountingClip.calls == 1   # ONE batch for all templates x tags


def test_extract_batch():
    """features.extract_batch: same records as extract(), one pass per tower."""
    from features import FeatureExtractor

    class StubClip:
        def embed_texts(self, texts):
            rng = np.random.default_rng(hash(tuple(texts)) % 2**32)
            v = rng.normal(size=(len(texts), 512))
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
        def embed_images(self, paths):
            rng = np.random.default_rng(0)
            v = rng.normal(size=(len(paths), 512))
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    fx = FeatureExtractor(clip=StubClip())
    records = fx.extract_batch(["a.jpg", "b.jpg"])
    assert [r["path"] for r in records] == ["a.jpg", "b.jpg"]
    for r in records:
        assert r["image_emb"].shape == (512,) and r["fused_emb"].shape == (1024,)
        assert r["caption"] == templates.caption_for(r["tags"])


def test_multi_label():
    """labels.py: per-tag sigmoid vs neutral prompt — the label set is dynamic."""
    import labels

    vocab = ["cat", "dog", "car"]
    tag_embs = np.array([[1, 0, 0, 0],     # "a photo of a cat"
                         [0, 1, 0, 0],     # "a photo of a dog"
                         [0, 0, 1, 0.0]])  # "a photo of a car"
    neutral = np.array([0, 0, 0, 1.0])     # "a photo"

    one_thing = np.array([0.9, 0, 0, 0.1])           # clearly a cat
    got = labels.multi_label(one_thing, tag_embs, neutral, vocab)
    assert list(got) == ["cat"] and got["cat"] > 0.99

    two_things = np.array([0.6, 0.6, 0, 0.1])        # a cat AND a dog
    got = labels.multi_label(two_things, tag_embs, neutral, vocab)
    assert set(got) == {"cat", "dog"}                # dynamic: 2 labels, not k

    probs = labels.label_probs(one_thing, tag_embs, neutral)
    assert probs.shape == (3,) and probs[0] > 0.99 and probs[1] < 0.01


class ScriptedClip:
    """A fake encoder that returns fixed vectors per exact sentence."""
    def __init__(self, image_vec, by_text, default):
        self.image_vec, self.by_text, self.default = image_vec, by_text, default

    def embed_texts(self, texts):
        return np.stack([self.by_text.get(t, self.default) for t in texts])

    def embed_images(self, paths):
        return np.stack([self.image_vec for _ in paths])


def test_agent_satisfied():
    """agent.py: a good proposal passes the critic on round 1 and stops."""
    import templates
    from agent import EmbeddingAgent

    image = np.array([1, 0, 0, 0.0])
    clip = ScriptedClip(
        image_vec=image,
        by_text={
            "a photo of a cat": np.array([0.995, 0, 0.1, 0]),  # matches image
            "a photo": np.array([0.1, 0, 0.995, 0]),           # neutral, weak
            "a photo of cat": image,                           # caption aligns
        },
        default=np.array([0, 1, 0, 0.0]),  # every other tag: orthogonal
    )
    record, verdict = EmbeddingAgent(clip=clip).run("cat.jpg")
    assert verdict.satisfied
    assert verdict.template == templates.TEMPLATE_POOL[0]  # stopped on round 1
    assert list(record["labels"]) == ["cat"] and record["labels"]["cat"] > 0.99
    assert record["caption"] == "a photo of cat"
    assert record["fused_emb"].shape == (8,)  # [image ; text] in this 4-d fake


def test_agent_unsatisfied():
    """agent.py: when no template works, every round runs and nothing passes."""
    from agent import EmbeddingAgent

    # every sentence embeds orthogonal to the image: alignment 0, all gaps 0
    clip = ScriptedClip(image_vec=np.array([1, 0, 0, 0.0]),
                        by_text={}, default=np.array([0, 1, 0, 0.0]))
    bot = EmbeddingAgent(clip=clip)
    record, verdict = bot.run("mystery.jpg")
    assert not verdict.satisfied          # caller must not publish this
    assert verdict.aligned < 0.2
    assert len(bot._prompt_embs) == len(bot.template_pool)  # all rounds tried


def test_item_tower():
    import item_tower

    rng = np.random.default_rng(2)
    emb = rng.normal(size=1024).astype(np.float32)
    record = {"path": "x.jpg", "caption": "a photo of cat",
              "labels": {"cat": 0.99, "pet": 0.7}, "fused_emb": emb}
    path = os.path.join(tempfile.mkdtemp(), "items.sqlite")
    con = item_tower.connect(path)
    item_tower.add_item(con, record)
    (item,) = item_tower.all_items(con)
    assert item["labels"] == {"cat": 0.99, "pet": 0.7}
    assert item["model"] and item["created_at"]
    assert np.allclose(item["item_emb"], emb)
    paths, matrix = item_tower.item_matrix(con)
    assert paths == ["x.jpg"] and matrix.shape == (1, 1024)


def test_pca_2d():
    from export_web import pca_2d
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 16))
    coords, mean, components = pca_2d(X)
    assert coords.shape == (10, 2) and mean.shape == (16,) and components.shape == (2, 16)
    # projecting a vector with mean+components reproduces its coordinate
    assert np.allclose((X[0] - mean) @ components.T, coords[0])


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  pass {name}")
    print("all smoke tests passed")
