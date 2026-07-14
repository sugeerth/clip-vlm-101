"""Fast sanity checks — no model download needed. Run: python3 test_smoke.py"""
import os
import pathlib
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
    for t in templates.TEMPLATE_POOL:
        assert templates.fill(t, tag="cat").count("cat") == 1


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


def test_similarity():
    from similarity import matrix, modality_gap
    rng = np.random.default_rng(7)
    V = rng.normal(size=(5, 8))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    M = matrix(V)
    assert M.shape == (5, 5)
    assert np.allclose(M, M.T) and np.allclose(np.diag(M), 1.0)
    # the modality-gap ordering holds on the committed real gallery
    gap = modality_gap(db.load_json_gallery())
    assert gap["image · OWN caption"] + 0.1 < gap["image · other images"]
    assert gap["image · other captions"] < gap["image · OWN caption"]


def test_retrieval_eval():
    from retrieval_eval import evaluate

    def item(tag, v):
        v = np.asarray(v, float); v /= np.linalg.norm(v)
        return {"tags": [tag], "image_emb": v, "text_emb": v,
                "fused_emb": np.concatenate([v, v]) / np.sqrt(2)}

    items = [item("a", [1, 0, 0, 0]), item("a", [0.9, 0.1, 0, 0]),
             item("b", [0, 0, 1, 0]), item("b", [0, 0, 0.9, 0.1])]
    m = evaluate(items, "image")  # each query's one groupmate must rank first
    assert m["P@1"] == 1.0 and m["MRR"] == 1.0
    assert abs(m["P@3"] - 1 / 3) < 1e-9  # only 1 of any 3 can be relevant
    # the committed gallery reproduces the README number
    real = evaluate(db.load_json_gallery(), "image")
    assert abs(real["P@1"] - 0.857) < 0.01 and real["MRR"] > 0.8


def test_arithmetic_combine():
    from arithmetic import combine
    a, b, c = np.eye(3)
    v = combine([a, b, c], [1, 1, -1])
    assert abs(np.linalg.norm(v) - 1) < 1e-6  # renormalized onto the sphere
    assert v[0] > 0 and v[2] < 0
    try:  # a combination that cancels out is an explicit error
        combine([a, a], [1, -1])
        assert False, "should have raised"
    except SystemExit:
        pass
    # the 'animal' centroid retrieves exactly the 4 animal images
    items = db.load_json_gallery()
    animals = [it for it in items if "animal" in it["tags"]]
    q = combine([it["image_emb"] for it in animals], [1] * len(animals))
    ranked = sorted(items, key=lambda it: float(it["image_emb"] @ q), reverse=True)
    assert {it["path"] for it in ranked[:4]} == {it["path"] for it in animals}


def test_quantize():
    from quantize import dequantize, quantize, top_neighbors
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q, scale = quantize(X)
    assert q.dtype == np.int8 and q.nbytes * 4 == X.nbytes  # 4x smaller
    assert np.abs(dequantize(q, scale) - X).max() <= scale / 2 + 1e-6
    # int8 keeps almost every top-3 neighbor on the committed gallery
    I = np.asarray([it["image_emb"] for it in db.load_json_gallery()])
    qi, _ = quantize(I)
    exact = top_neighbors(I @ I.T)
    approx = top_neighbors(qi.astype(np.int32) @ qi.astype(np.int32).T)
    assert sum(x == y for x, y in zip(exact, approx)) >= 12  # 13/14 committed


def test_centering():
    from similarity import center, modality_gap
    items = db.load_json_gallery()
    I = center([it["image_emb"] for it in items])
    T = center([it["text_emb"] for it in items])
    assert np.allclose(np.linalg.norm(I, axis=1), 1.0, atol=1e-9)  # re-unit
    centered = [dict(it, image_emb=I[i], text_emb=T[i]) for i, it in enumerate(items)]
    g0, g1 = modality_gap(items), modality_gap(centered)
    margin = lambda g: g["image · OWN caption"] - g["image · other captions"]
    assert margin(g1) > 2.5 * margin(g0)   # the fix widens the margin ~3x
    assert abs(g1["image · other images"]) < 0.1  # noise floor lands near zero


def test_ann():
    from ann import build, recall_at_k, search, synthetic
    X, Q = synthetic()
    C, lists = build(X)
    assert sum(len(l) for l in lists) == len(X)  # every vector in exactly one list

    exact = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]

    def run(probes):
        rec = scanned = 0
        for qi, q in enumerate(Q):
            found, n = search(q, X, C, lists, probes=probes)
            rec += recall_at_k(found, exact[qi])
            scanned += n
        return rec / len(Q), scanned / len(Q) / len(X)

    r1, s1 = run(1)
    r8, s8 = run(8)
    assert r1 > 0.7 and s1 < 0.03      # scan under 3%%, keep over 70%% of truth
    assert r8 > 0.9 and r8 > r1 and s8 > s1  # probes: more truth, more work


def test_softmax():
    from temperature import softmax
    scores = [0.3, 0.2, 0.1]
    p = softmax(scores)
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[1] > p[2]        # order preserved
    assert np.allclose(softmax(scores, scale=0), 1 / 3)          # scale 0: uniform
    assert softmax(scores, scale=1000)[0] > 0.999                # huge scale: one-hot
    assert np.allclose(softmax([s + 7 for s in scores]), p)      # shift-invariant


def test_dcn():
    """dcn.py: W=0 reproduces retrieval order; one cross lifts tag-sharers."""
    import dcn

    # 3 candidates: B has lower cosine than A but shares a tag with the query
    cand = [
        {"item": "A", "cos_image": 0.80, "cos_text": 0.80, "tag_overlap": 0, "rank_prior": 1.0},
        {"item": "B", "cos_image": 0.70, "cos_text": 0.70, "tag_overlap": 3, "rank_prior": 0.5},
        {"item": "C", "cos_image": 0.60, "cos_text": 0.60, "tag_overlap": 0, "rank_prior": 0.33},
    ]
    passthrough = dcn.CrossNetwork(dim=len(dcn.FEATURES), num_layers=1)  # W=0
    order0 = [c["item"] for c in dcn.rerank(cand, passthrough)]
    assert order0 == ["A", "B", "C"]                # pure fused-cosine order

    crossed = dcn.CrossNetwork(dim=len(dcn.FEATURES), num_layers=1)
    crossed.set_cross(0, dcn.FEATURES.index("cos_image"),
                      dcn.FEATURES.index("tag_overlap"), 6.0)
    order1 = [c["item"] for c in dcn.rerank(cand, crossed)]
    assert order1[0] == "B"                          # tag-sharer lifted above A
    # the cross is a real interaction: B's score now exceeds A's
    scored = {c["item"]: c["dcn_score"] for c in dcn.rerank(cand, crossed)}
    assert scored["B"] > scored["A"] > scored["C"]

    # forward() is the exact DCN-v2 formula x0 ⊙ (W x + b) + x
    x0 = dcn.make_features(0.5, 0.5, 2, 1.0)
    net = dcn.CrossNetwork(dim=4, num_layers=1)
    net.bs[0][:] = 0.1
    assert np.allclose(net.forward(x0), x0 * (net.Ws[0] @ x0 + net.bs[0]) + x0)


def test_explain():
    """explain.py: the template passes its own gate; the gate strips lies."""
    import explain

    ranked = [
        ({"tags": ["cat", "pet", "animal"]}, 0.31),
        ({"tags": ["cat", "pet", "portrait"]}, 0.27),
        ({"tags": ["cat", "dog", "pet"]}, 0.22),
    ]
    ev = explain.evidence("a fluffy cat", ranked, k=3)
    assert ev["shared"] == ["cat", "pet"] and ev["top_score"] == 0.31
    assert ev["strength"] == "strong"                # 0.31 ≥ 0.30

    # the grounded template must ALWAYS pass its own gate (regression guard)
    template = explain.describe(ev, k=3)
    assert explain.verify(template, ev)["clean"], f"template tripped its gate: {template}"

    # a hallucinated draft: a real tag not present ('dog' isn't in the top set),
    # a fabricated number, and a wrong strength word — all caught
    lie = "These all show a car. The match is weak at 0.99."
    checked = explain.verify(lie, ev)          # 'car' is a real tag, absent here
    assert not checked["clean"] and checked["verified"] == ""
    joined = " ".join(r for s in checked["stripped"] for r in s["reasons"])
    assert "car" in joined and "0.99" in joined and "weak" in joined

    # THE regression that matters: the template must pass its gate for a WEAK
    # match too, where it emits the "the model isnt confident" honesty tail
    weak = [({"tags": ["cat", "pet"]}, 0.21), ({"tags": ["cat", "dog"]}, 0.20)]
    wev = explain.evidence("a cat", weak, k=2)
    assert wev["strength"] == "weak"
    wtpl = explain.describe(wev, k=2)
    assert "isnt confident" in wtpl                     # the weak tail is present
    assert explain.verify(wtpl, wev)["clean"], f"weak template tripped its gate: {wtpl}"

    # very-weak (the lone token 'very') and empty results ('explain') must pass too
    vwev = explain.evidence("a cat", [({"tags": ["cat"]}, 0.12)], k=1)
    assert vwev["strength"] == "very weak"
    assert explain.verify(explain.describe(vwev, k=1), vwev)["clean"], "very-weak template tripped its gate"
    empty = explain.evidence("a cat", [], k=1)
    assert explain.verify(explain.describe(empty, k=1), empty)["clean"], "empty template tripped its gate"

    # buckets
    assert explain.bucket(0.31) == "strong" and explain.bucket(0.26) == "moderate"
    assert explain.bucket(0.21) == "weak" and explain.bucket(0.10) == "very weak"

    # explain() falls back to the template when the LLM draft is fully stripped
    out = explain.explain("a fluffy cat", ranked, k=3, draft="Clearly a helicopter at 0.99.")
    assert out["explanation"] == explain.describe(ev, k=3)


def test_model_registry():
    """models.py: keys, ids and unknown ids all resolve; padding rules hold."""
    import models
    from labels import siglip_label_probs

    by_key = models.resolve("siglip2-base")
    assert by_key["kind"] == "siglip" and by_key["dim"] == 768
    assert by_key["text_kwargs"]["padding"] == "max_length"    # the big trap
    assert by_key["text_kwargs"]["max_length"] == 64
    by_id = models.resolve("openai/clip-vit-base-patch32")
    assert by_id["kind"] == "clip" and by_id["dim"] == 512
    assert by_id["text_kwargs"]["padding"] is True
    unknown = models.resolve("someone/some-clip")
    assert unknown["kind"] == "clip" and unknown["hf_id"] == "someone/some-clip"
    assert models.resolve("clip-b32")["hf_id"] == models.MODELS[models.DEFAULT]["hf_id"]

    # SigLIP native calibration: sigmoid(scale * cos + bias), checkpoint values
    tag_embs = np.array([[1, 0, 0], [0, 1, 0.0]])
    img = np.array([1, 0, 0.0])
    p = siglip_label_probs(img, tag_embs, scale=118.0, bias=-12.9)
    assert 0.9 < p[0] <= 1.0          # perfect match: confident
    assert p[1] < 1e-4                # unrelated: bias drives it to ~0


def test_hermes_extend():
    """hermes.extend: crawled files join the working set for THIS search."""
    import hermes

    class StubFx:
        def extract_batch(self, paths):
            return [{"path": p, "tags": ["x"], "caption": "a photo of x",
                     "image_emb": np.ones(4), "text_emb": np.ones(4),
                     "fused_emb": np.ones(8) / np.sqrt(8)} for p in paths]

    items = [{"path": "old.jpg"}]
    grown = hermes.extend(items, [pathlib.Path("new1.jpg"), "new2.jpg"], StubFx())
    assert [it["path"] for it in grown] == ["old.jpg", "new1.jpg", "new2.jpg"]
    assert hermes.extend(items, [], StubFx()) == items   # no crawl, no change


def test_spider():
    """spider.py: BFS a local fixture site — caps, robots, quality gate."""
    import http.server
    import json
    import shutil
    import threading

    import spider

    site = pathlib.Path(tempfile.mkdtemp())
    out = pathlib.Path(tempfile.mkdtemp())
    # two real photos + one icon-sized decoy + a robots-forbidden area
    shutil.copy("images/cat.jpg", site / "cat.jpg")
    shutil.copy("images/dog.jpg", site / "dog.jpg")
    (site / "tiny.jpg").write_bytes(b"\xff\xd8" + b"0" * 100)     # icon-sized
    (site / "robots.txt").write_text("User-agent: *\nDisallow: /private/\n")
    private = site / "private"; private.mkdir()
    shutil.copy("images/bear.jpg", private / "secret.jpg")
    (site / "index.html").write_text(
        '<img src="cat.jpg"><img src="tiny.jpg">'
        '<a href="page2.html">next</a><a href="http://off-domain.example/x">off</a>'
        '<a href="private/page3.html">private</a>')
    (site / "page2.html").write_text('<img src="dog.jpg"><img src="cat.jpg">')
    (private / "page3.html").write_text('<img src="secret.jpg">')

    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(site), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        saved = spider.crawl([f"{base}/index.html"], max_pages=10,
                             max_images=10, out=str(out), delay=0)
        names = {p.name.split("_")[0] for p in saved}
        assert len(saved) == 2 and names == {"spider"}   # cat + dog, deduped
        manifest = json.loads((out / spider.MANIFEST).read_text())
        assert len(manifest) == 2
        assert all(m["source"].startswith(base) and m["sha1"] for m in manifest)
        assert not any("secret" in m["name"] for m in manifest)  # robots held
        assert not any("tiny" in m["name"] for m in manifest)    # gate held
        # re-crawl finds nothing new (same bytes -> same sha1)
        again = spider.crawl([f"{base}/index.html"], max_pages=10,
                             max_images=10, out=str(out), delay=0)
        assert again == []
        # caps are hard limits
        capped = spider.crawl([f"{base}/index.html"], max_pages=10,
                              max_images=1, out=str(tempfile.mkdtemp()), delay=0)
        assert len(capped) == 1
    finally:
        srv.shutdown()


def test_hermes():
    """hermes.py: decisive phrasing wins; indecisive queries get ensembled."""
    import hermes

    def item(v):
        v = np.asarray(v, float)
        return {"image_emb": v, "text_emb": v,
                "fused_emb": np.concatenate([v, v]) / np.sqrt(2)}

    items = [item([1, 0, 0, 0]), item([0, 1, 0, 0]), item([0, 0, 1, 0])]
    decisive = np.array([0.9, 0.1, 0.0, 0.0])
    uniform = np.array([1, 1, 1, 0]) / np.sqrt(3)

    def encode_decisive(texts):
        return np.stack([decisive if t.startswith("a photo of") else uniform
                         for t in texts])

    out = hermes.search("cat", encode_decisive, items, k=3)
    assert out["satisfied"] and out["chose"] == "a photo of cat"
    assert out["ranked"][0][0] is items[0]          # the decisive hit
    assert len(out["rounds"]) == len(hermes.QUERY_TEMPLATES)
    ms = [r["margin"] for r in out["rounds"]]
    assert max(ms) >= hermes.MIN_MARGIN and ms[1] == max(ms)

    out2 = hermes.search("cat", lambda ts: np.stack([uniform] * len(ts)), items, k=3)
    assert not out2["satisfied"] and "ensemble" in out2["chose"]
    assert len(out2["ranked"]) == 3                 # still answers

    assert hermes.margin([0.9, 0.1, 0.0]) == 0.85   # the critic's number


def test_hermes_refine():
    """The evaluator guard: accepts gains, rejects drift, detects convergence."""
    import fusion
    import hermes

    items = db.load_json_gallery()
    # cat: feedback converges (same top-k) — the ledger says so
    cat = next(it for it in items if "cat" in it["path"])
    out = hermes.search_image(cat, items, k=5, passes=4)
    assert out["ledger"][0]["verdict"] == "initial"
    assert out["ledger"][-1]["verdict"] in ("converged", "rejected — stopping", "accepted")
    assert len(out["ranked"]) == 5
    assert all(it is not cat for it, _ in out["ranked"])  # self excluded

    # pizza: the feedback pass drifts; the evaluator must REJECT it and
    # keep the initial ranking (this is the whole point of the guard)
    pizza = next(it for it in items if "pizza" in it["path"])
    out2 = hermes.search_image(pizza, items, k=5, passes=4)
    ledger = out2["ledger"]
    assert ledger[-1]["verdict"].startswith("rejected")
    assert ledger[-1]["eval"] < ledger[0]["eval"]  # it drifted, and was caught
    # the published ranking is the PASS-1 ranking, not the drifted one
    q0 = fusion.fused_query(np.asarray(pizza["image_emb"], dtype=np.float64))
    initial, _ = hermes._rank_fused(items, q0, 5, exclude=pizza)
    assert [it["path"] for it, _ in out2["ranked"]] == [it["path"] for it, _ in initial]

    # guarded refine can never publish a ranking the evaluator scores below
    # the initial one — on every gallery image
    for q_item in items:
        r = hermes.search_image(q_item, items, k=5, passes=4)
        assert hermes.evaluate(r["ranked"],
            fusion.fused_query(np.asarray(q_item["image_emb"], dtype=np.float64))) \
            >= r["ledger"][0]["eval"] - 1e-6


def test_crawler():
    """crawler.py against a local stub of the Commons API — no real network."""
    import http.server
    import json as jsonlib
    import threading

    import crawler

    JPEG = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api"):
                port = self.server.server_address[1]
                body = jsonlib.dumps({"query": {"pages": {
                    "1": {"title": "File:Red panda.jpg",
                          "imageinfo": [{"thumburl": f"http://127.0.0.1:{port}/img/a.jpg",
                                         "descriptionurl": "https://commons.example/A",
                                         "extmetadata": {"Artist": {"value": "Ann"},
                                                         "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]},
                    "2": {"title": "File:Red panda 2.jpg",
                          "imageinfo": [{"thumburl": f"http://127.0.0.1:{port}/img/b.jpg",
                                         "descriptionurl": "https://commons.example/B",
                                         "extmetadata": {}}]},
                }}}).encode()
            else:
                body = JPEG
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep the test output quiet
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api = f"http://127.0.0.1:{srv.server_address[1]}/api"
    out = tempfile.mkdtemp()

    new = crawler.crawl("red panda", n=2, out=out, api=api, pause=0)
    assert len(new) == 2 and all(p.read_bytes() == JPEG for p in new)
    manifest = jsonlib.loads((pathlib.Path(out) / crawler.MANIFEST).read_text())
    assert len(manifest) == 2
    assert manifest[0]["license"] == "CC BY-SA 4.0" and manifest[0]["author"] == "Ann"
    assert manifest[0]["source"] == "https://commons.example/A"
    # idempotent: a second crawl downloads nothing new
    assert crawler.crawl("red panda", n=2, out=out, api=api, pause=0) == []
    srv.shutdown()


def test_load_json_gallery():
    items = db.load_json_gallery()
    assert len(items) == 14
    for it in items:
        assert it["image_emb"].shape == (512,) and it["fused_emb"].shape == (1024,)
        assert abs(np.linalg.norm(it["image_emb"]) - 1) < 0.01   # 5-dp rounded units
    # the MODALITY GAP (Liang et al. 2022), visible right in the committed
    # data: an image is far more similar to OTHER IMAGES than to the text
    # embedding of its OWN caption — cross-modal scores live on their own scale
    own_caption = np.mean([it["image_emb"] @ it["text_emb"] for it in items])
    other_images = np.mean([a["image_emb"] @ b["image_emb"]
                            for a in items for b in items if a is not b])
    assert own_caption + 0.1 < other_images

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


def test_user_tower():
    """user_tower.py: mean-pooled likes rank the catalog, likes excluded."""
    import user_tower

    # 4 unit item vectors in 4-d: two "animals" near each other, two far away
    matrix = np.array([[1, 0, 0, 0],
                       [0.9, 0.1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1.0]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    paths = ["cat.jpg", "dog.jpg", "pizza.jpg", "pluto.jpg"]

    u = user_tower.user_vector(matrix[:2])
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6

    recs = user_tower.recommend(paths, matrix, ["cat.jpg"], k=2)
    assert recs[0][0] == "dog.jpg"                  # nearest non-liked item
    assert all(p != "cat.jpg" for p, _ in recs)     # likes never recommended

    try:
        user_tower.recommend(paths, matrix, ["unknown.jpg"])
        assert False, "should reject likes missing from the tower"
    except ValueError:
        pass


def test_eval_harness():
    """eval.py: hit rates count correctly against the ground-truth table."""
    import eval as ev

    vocab = ["cat", "dog", "car"]
    tag_embs = np.eye(3, 4)
    cat_img = np.array([0.9, 0.1, 0, 0])       # top-1 = cat
    car_img = np.array([0.2, 0.5, 0.9, 0])     # top-1 = car, top-2 has dog
    t1, tk, n = ev.hit_rates([cat_img, car_img], [{"cat"}, {"dog"}],
                             tag_embs, vocab, k=2)
    assert (t1, tk, n) == (1, 2, 2)            # dog only hits within top-2

    class StubClip:
        def embed_images(self, paths):
            rng = np.random.default_rng(5)
            v = rng.normal(size=(len(paths), 64))
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        embed_texts = embed_images

    results = ev.evaluate(StubClip(), ["cat.jpg", "dog.jpg", "mystery.jpg"])
    assert results["n"] == 2 and results["skipped"] == 1
    assert len([k for k in results if k not in ("n", "skipped")]) == 2


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
