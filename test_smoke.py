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


def test_scaling():
    """scaling.py: the billion-scale arithmetic is internally consistent."""
    import scaling

    # memory scales linearly and PQ-64 is 32x smaller than float32
    assert scaling.memory(1_000, "float32") == 1_000 * 512 * 4
    assert scaling.memory(1_000, "float32") == 32 * scaling.memory(1_000, "PQ-64")
    assert scaling.memory(2_000, "int8") == 2 * scaling.memory(1_000, "int8")
    # a billion in PQ-64 is ~64 GB (fits a handful of boxes), not ~2 TB
    assert scaling.memory(int(1e9), "PQ-64") == int(1e9) * 64

    # IVF is sublinear: candidates scanned grow like sqrt(N), not N
    small = scaling.ivf_plan(1_000_000, nprobe=32)
    big = scaling.ivf_plan(1_000_000_000, nprobe=32)   # 1000x the corpus
    assert big["nlist"] == round(1e9 ** 0.5)
    # 1000x more data scans only ~sqrt(1000) ≈ 32x more candidates
    ratio = big["candidates_scanned"] / small["candidates_scanned"]
    assert 25 < ratio < 40
    assert big["fraction_scanned"] < 0.005          # under half a percent of a billion

    # sharding: a billion is exactly 1000 of the measured million
    sh = scaling.shard_plan(int(1e9), int(1e6))
    assert sh["shards"] == 1000 and sh["per_shard"] == 1_000_000

    # latency budget: parallel shards, interactive total, fan-out only when sharded
    budget, total = scaling.latency_budget(sharded=True)
    assert total == sum(ms for _, ms in budget)
    assert 10 < total < 60                            # tens of ms, at a billion
    _, solo = scaling.latency_budget(sharded=False)
    assert solo < total                               # no scatter/gather when not sharded

    # the approximation cascade: only the final shortlist ever touches float32
    cas = scaling.cascade_plan(int(2e9), nprobe=32)
    assert cas["float32_per_query"] == scaling.CASCADE_KEEP[-1]   # 50, not 2 billion
    assert cas["float32_fraction"] < 1e-7                         # a rounding error
    # each level scores fewer than the last (a funnel), binary the widest
    scored = cas["scored_per_level"]
    assert scored == sorted(scored, reverse=True) and scored[0] > 1e6
    # binary and PQ are both 64 bytes at 512-d; resident index is their sum
    assert scaling.memory(1000, "binary") == 1000 * (512 // 8)
    assert cas["resident_bytes"] == int(2e9) * (scaling.ENCODINGS["binary"]
                                                + scaling.ENCODINGS["PQ-64"])
    # two billion is 2,000 shards of the measured million
    assert scaling.shard_plan(int(2e9), int(1e6))["shards"] == 2000


def test_cascade():
    """cascade.py: approximate at every level, keep ~all the recall."""
    import numpy as np
    import cascade
    from ann import synthetic

    X, Q = synthetic(n=3000, n_queries=60, dim=64, blobs=48, noise=0.2)
    truth = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]
    cas = cascade.Cascade(X)

    # the cascade's top-k comes back the right length and from the corpus
    top, steps = cas.search(Q[0], trace=True)
    assert len(top) == 10 and set(top).issubset(range(len(X)))
    # the funnel narrows monotonically: L0 ≥ L1 ≥ L2 ≥ L3 ≥ L4
    widths = [w for _, w in steps]
    assert widths == sorted(widths, reverse=True)   # a funnel: never widens
    assert widths[-1] == 10 and widths[0] > widths[-1]  # and narrows overall

    # the ceiling: an EXACT scan of the same IVF cells
    def ceiling(qi):
        near = np.argsort(Q[qi] @ cas.C.T)[::-1][:8]
        cand = np.concatenate([cas.lists[j] for j in near])
        return cand[np.argsort(X[cand] @ Q[qi])[::-1][:10]]

    rec_cascade = np.mean([len(set(cas.search(Q[i])) & set(truth[i])) / 10
                           for i in range(len(Q))])
    rec_ceiling = np.mean([len(set(ceiling(i)) & set(truth[i])) / 10
                           for i in range(len(Q))])
    # every level is approximate, yet the cascade keeps essentially all the
    # recall the exact scan of those cells could — that is the whole claim
    assert rec_cascade >= 0.98 * rec_ceiling
    assert rec_cascade > 0.85            # and it is genuinely high, not just close

    # binary encoding is 1 bit/dim and Hamming ranks a vector nearest itself
    bits = cascade.binary_encode(X)
    assert bits.shape == X.shape and bits.dtype == bool
    h = cascade.hamming(bits[0], bits)
    assert h[0] == 0 and h.argmin() == 0


def test_hnsw():
    """hnsw.py: the navigable-small-world graph is well-formed and, at matched
    work, keeps more of the truth than IVF — the whole reason the graph exists."""
    from ann import build as ivf_build, recall_at_k, search as ivf_search, synthetic
    from hnsw import HNSW

    X, Q = synthetic(n=1500, n_queries=60)
    exact = np.argsort(Q @ X.T, axis=1)[:, ::-1][:, :10]
    idx = HNSW(X, M=16, ef_construction=48)

    # a well-formed HNSW: every node lives on layer 0, layers shrink going up,
    # and the entry point sits on the top layer.
    assert len(idx.layers[0]) == len(X)
    sizes = [len(l) for l in idx.layers]
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))     # a pyramid, not a wall
    assert idx.entry in idx.layers[idx.top]
    assert idx.top == len(idx.layers) - 1

    # every node is pruned to the per-layer budget (M0 on layer 0), no self-loops,
    # and every neighbour id is a real node — the graph stays sparse and valid.
    for node, nbrs in idx.layers[0].items():
        assert len(nbrs) <= idx.M0
        assert node not in nbrs
        assert len(set(nbrs)) == len(nbrs)              # no duplicate edges
        assert all(0 <= nb < len(X) for nb in nbrs)
    # links start bidirectional; pruning may drop the far side, so most (not all)
    # edges survive both ways — the graph is well-connected, not a set of one-way streets.
    both = sum(node in idx.layers[0][nb] for node, nbrs in idx.layers[0].items() for nb in nbrs)
    total = sum(len(nbrs) for nbrs in idx.layers[0].values())
    assert both > 0.5 * total

    # deterministic: same seed, same graph, same answers
    idx2 = HNSW(X, M=16, ef_construction=48)
    assert idx.layers[0] == idx2.layers[0]

    def hnsw_recall(ef):
        idx.dist_calls = 0
        rec = sum(recall_at_k(idx.search(q, ef=ef), exact[qi]) for qi, q in enumerate(Q))
        return rec / len(Q), idx.dist_calls / len(Q)

    sweep = {ef: hnsw_recall(ef) for ef in (8, 16, 32, 64)}
    r_lo, w_lo = sweep[8]
    r_hi, w_hi = sweep[64]
    assert r_hi > r_lo and w_hi > w_lo          # ef is a dial: more truth, more work
    assert r_hi > 0.9                            # and it genuinely finds the truth

    # apples-to-apples: give IVF its best shot here, then show HNSW reaches the
    # SAME recall for FEWER distance computations — the graph's whole reason to exist.
    C, lists = ivf_build(X, n_lists=64)
    ivf_rec = ivf_scan = 0
    for qi, q in enumerate(Q):
        found, n = ivf_search(q, X, C, lists, probes=16)
        ivf_rec += recall_at_k(found, exact[qi])
        ivf_scan += n + len(C)
    ivf_rec /= len(Q); ivf_scan /= len(Q)
    # the cheapest HNSW pass that matches IVF's recall costs strictly less work
    matched = [w for r, w in sweep.values() if r >= ivf_rec]
    assert matched and min(matched) < ivf_scan


def test_softmax():
    from temperature import softmax
    scores = [0.3, 0.2, 0.1]
    p = softmax(scores)
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[1] > p[2]        # order preserved
    assert np.allclose(softmax(scores, scale=0), 1 / 3)          # scale 0: uniform
    assert softmax(scores, scale=1000)[0] > 0.999                # huge scale: one-hot
    assert np.allclose(softmax([s + 7 for s in scores]), p)      # shift-invariant


def test_learn2rank():
    """learn2rank.py: untrained==base; feedback learns tag preference, capped."""
    import learn2rank as l2r

    # A and B tie on base score; only B shares tags. C is a lower non-sharer.
    cand = [
        {"item": "A", "base_score": 0.60, "features": [0.60, 0.60, 0, 0.5]},
        {"item": "B", "base_score": 0.60, "features": [0.60, 0.60, 4, 0.5]},
        {"item": "C", "base_score": 0.50, "features": [0.50, 0.50, 0, 0.3]},
    ]
    r = l2r.OnlineRanker()
    assert [c["item"] for c in r.rank(cand)] == ["A", "B", "C"]   # untrained == base
    assert r.rank(cand)[0]["beta"] == 0.0

    # 👍 the tag-sharer B, 👎 the tagless A and C — feedback breaks the tie
    r.feedback([0.60, 0.60, 4, 0.5], 1)
    r.feedback([0.60, 0.60, 0, 0.5], 0)
    r.feedback([0.50, 0.50, 0, 0.3], 0)
    assert r.n_pairs() == 2                       # 1 pos × 2 neg
    ranked = r.rank(cand)
    assert ranked[0]["item"] == "B"              # the tag-sharer wins the tie
    assert ranked[0]["beta"] <= l2r.W_MAX + 1e-9  # blend never exceeds the cap
    # the safety cap: feedback can't flip a LARGE base gap in a few clicks
    big = [{"item": "X", "base_score": 0.95, "features": [0.95, 0.95, 0, 1.0]},
           {"item": "Y", "base_score": 0.50, "features": [0.50, 0.50, 4, 0.4]}]
    rr = l2r.OnlineRanker(); rr.feedback(big[1]["features"], 1); rr.feedback(big[0]["features"], 0)
    assert rr.rank(big)[0]["item"] == "X"        # retrieval keeps ≥ half the vote
    imp = r.importance()
    assert imp["tag_overlap"]["importance"] > imp["cos_image"]["importance"]

    # one-sided feedback (all 👎) uses Rocchio and does NOT blow up / NaN
    r2 = l2r.OnlineRanker()
    for f in ([0.9, 0.9, 0, 1.0], [0.8, 0.8, 0, 0.5]):
        r2.feedback(f, 0)
    assert r2.n_pairs() == 0                      # no pairs → no RankNet
    out = r2.rank(cand)
    assert all(np.isfinite(c["score"]) for c in out) and len(out) == 3

    # state round-trips (the localStorage "personal model")
    r3 = l2r.OnlineRanker().load_state(r.to_state())
    assert np.allclose(r3.w, r.w) and r3.n == r.n


def test_conformal():
    """conformal.py: valid-or-conservative coverage, monotone size, exact quantile."""
    import conformal

    items = db.load_json_gallery()
    scores = conformal.loo_scores(items)
    n = len(scores)
    assert n >= 10

    # quantile identity: in-sample, at least k/n of scores are <= q̂ (>= 1-alpha)
    for a in (0.4, 0.2, 0.1):
        qhat = conformal.calibrate(scores, a)
        k = int(np.ceil((n + 1) * (1 - a)))
        if k <= n:
            frac = float(np.mean(scores <= qhat))
            assert frac >= k / n - 1e-9 >= (1 - a) - 1e-9

    # headline guarantee at alpha=0.2 (80%): honest jackknife coverage holds
    cov80 = conformal.jackknife_coverage(scores, 0.2)
    assert cov80 >= 0.80 - 1 / (n + 1)

    # coverage and set size are both non-decreasing as we demand more confidence
    rows = conformal.report(items, alphas=(0.4, 0.3, 0.2, 0.1))
    covs = [r["coverage"] for r in rows]
    sizes = [r["avg_set"] for r in rows]
    assert covs == sorted(covs) and sizes == sorted(sizes)
    assert all(r["coverage"] >= r["target"] - 1 / (n + 1) for r in rows)

    # the set is exactly {cos >= 1 - q̂}
    qhat = conformal.calibrate(scores, 0.2)
    idx, tau = conformal.predict(items[0]["image_emb"], items, qhat)
    cos = conformal.cosines(items[0]["image_emb"], items)
    assert all(cos[i] >= tau - 1e-9 for i in idx)


def test_judge():
    """judge.py: the score gate, and the council's quorum / hung-jury / ruling."""
    import judge

    # the gate: accept the forms a small model emits, reject out of range
    assert judge.parse_score("0.7") == 0.7
    assert judge.parse_score(".7") == 0.7
    assert judge.parse_score("7/10") == 0.7
    assert judge.parse_score("8 out of 10") == 0.8
    assert judge.parse_score("70%") == 0.7
    assert judge.parse_score("score: 0.9") == 0.9
    assert judge.parse_score("relevant") is None      # no number → abstain
    assert judge.parse_score("2.5") is None            # out of [0,1]
    assert judge.parse_score("150%") is None

    # a judge with no parseable score ABSTAINS — it doesn't vote garbage
    votes = [{"name": "a", "score": 0.8, "confidence": 0.9},
             {"name": "b", "score": None, "confidence": 0.7},
             {"name": "c", "score": 0.7, "confidence": 0.6}]
    v = judge.aggregate(votes)
    assert v["n_valid"] == 2 and v["abstained"] == ["b"]
    # confidence-weighted mean, not a plain average
    assert abs(v["mean"] - (0.8 * 0.9 + 0.7 * 0.6) / (0.9 + 0.6)) < 1e-6
    assert v["decision"] == "relevant"

    # too few valid votes → the council can't rule
    assert judge.aggregate([{"name": "a", "score": 0.9, "confidence": 1.0},
                             {"name": "b", "score": None, "confidence": 1.0}]
                           )["decision"] == "abstain"

    # a split panel is a HUNG JURY (spread > HUNG_SPREAD) → abstain, not a
    # confident average over a coin flip
    hung = judge.aggregate([{"name": "a", "score": 0.1, "confidence": 1.0},
                            {"name": "b", "score": 0.9, "confidence": 1.0}])
    assert hung["decision"] == "abstain" and hung["reason"] == "hung jury"
    assert abs(hung["consensus"] - 0.2) < 1e-9    # 1 - (0.9 - 0.1)

    # majority: yes/no votes, ties abstain
    assert judge.majority(votes)["decision"] == "relevant"           # 2 yes, 0 no
    tie = judge.majority([{"name": "a", "score": 0.9}, {"name": "b", "score": 0.1}])
    assert tie["decision"] == "abstain" and tie["reason"] == "tie"

    # the model-free heuristic council rules on a clear same-tag match and
    # abstains when the tag signal and the visual signal disagree
    items = db.load_json_gallery()
    by = lambda s: next(it for it in items if s in it["path"])
    strong = judge.council(by("004_cat"), by("005_dog"))            # cat → dog
    assert strong["decision"] == "relevant" and strong["n_valid"] == 3
    split = judge.council(by("000_apple"), by("011_pluto"))         # shares 'apple' by fluke
    assert split["decision"] == "abstain" and split["reason"] == "hung jury"

    assert judge.QUORUM == 2 and judge.ACCEPT == 0.5 and judge.HUNG_SPREAD == 0.5


def test_trust():
    """trust.py: compose the honesty lenses — agreement, split-decision abstain,
    and the participation cap."""
    import trust

    # four agreeing lenses → high; weighted mean, consensus
    hi = trust.compose([{"name": "gate", "trust": 1.0, "weight": 1.0},
                        {"name": "conformal", "trust": 0.71, "weight": 1.0},
                        {"name": "council", "trust": 0.73, "weight": 1.2},
                        {"name": "margin", "trust": 1.0, "weight": 0.7}])
    assert hi["level"] == "high" and abs(hi["score"] - 0.8426) < 1e-3

    # two lenses abstain → "high" is CAPPED to medium (you can't claim high trust
    # while half the evidence declined to vote)
    cap = trust.compose([{"name": "gate", "trust": 0.7, "weight": 1.0},
                         {"name": "conformal", "trust": None, "weight": 1.0},
                         {"name": "council", "trust": None, "weight": 1.2},
                         {"name": "margin", "trust": 0.9, "weight": 0.7}])
    assert cap["level"] == "medium" and cap["reason"].startswith("capped")

    # the lenses split → abstain, not an average over a contradiction
    assert trust.compose([{"name": "a", "trust": 0.2, "weight": 1},
                          {"name": "b", "trust": 0.9, "weight": 1}])["level"] == "abstain"
    # spread exactly at SPLIT is NOT a split (strict >)
    assert trust.compose([{"name": "a", "trust": 0.4, "weight": 1},
                          {"name": "b", "trust": 0.9, "weight": 1}])["reason"] == "composed"
    # too few voting → abstain
    assert trust.compose([{"name": "a", "trust": 0.9, "weight": 1},
                          {"name": "b", "trust": None, "weight": 1}])["level"] == "abstain"

    # the four lenses
    assert trust.gate_trust(0.85, 0.8, 0.72, 0.66) == 1.0
    assert trust.gate_trust(0.68, 0.8, 0.72, 0.66) == 0.4
    assert trust.conformal_trust(0.5, 0.6) is None          # below the bar → abstain
    assert trust.conformal_trust(0.63, 0.63) == 0.5         # boundary is included
    assert trust.council_trust({"decision": "abstain"}) is None
    assert trust.margin_trust([0.8]) is None                # a lone result has no margin

    assert trust.QUORUM == 2 and trust.SPLIT == 0.5 and trust.MIN_FOR_HIGH == 3


def test_drift():
    """drift.py: the detectors fire in order (stable → shift → drift) as a live
    window drifts, and PSI/coverage move the right way."""
    import drift

    # PSI/KS basics
    assert drift.psi([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0     # a window vs itself
    assert drift.psi([1, 1, 1], [2, 2, 2]) == 0.0           # constant reference → no bins
    assert drift.ks_stat([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    assert abs(drift.ks_stat([0, 0, 0, 0], [1, 1, 1, 1]) - 1.0) < 1e-12

    items = db.load_json_gallery()
    ref = drift.quality_signal(items)
    assert len(ref) == 60                                    # same-tag pair similarities

    stream = [(0.00, "stable"), (0.15, "shift"), (0.35, "drift"), (0.60, "drift")]
    levels = [drift.monitor(ref, drift.drift_window(ref, f), 0.2)["level"] for f, _ in stream]
    assert levels == [want for _, want in stream], levels

    # drift_window uses explicit half-up rounding (matches the JS twin) at an
    # exact half: frac*n = 2.5 → k = 3
    assert int((drift.drift_window([1, 1, 1, 1, 1], 0.5) < 1).sum()) == 3
    assert drift.drift_window([0.5], 0.5)[0] == 0.3

    # PSI monotone in the contamination fraction; coverage falls
    psis = [drift.psi(ref, drift.drift_window(ref, f)) for f, _ in stream]
    assert psis == sorted(psis)
    covs = [drift.monitor(ref, drift.drift_window(ref, f), 0.2)["coverage"] for f, _ in stream]
    assert covs == sorted(covs, reverse=True)               # coverage only degrades

    # the worst window trips all three detectors
    assert len(drift.monitor(ref, drift.drift_window(ref, 0.60), 0.2)["reasons"]) == 3

    # positive vs failure cases against the calibrated bar
    _, bar = drift.coverage(ref, drift.drift_window(ref, 0.60), 0.2)
    pos, fail = drift.classify(items, drift.item_quality(items), bar)
    assert len(pos) == 13 and len(fail) == 1
    assert "bicycle" in fail[0]["item"]["tags"]

    assert drift.PSI_DRIFT == 0.25 and drift.COV_SLACK == 0.10


def test_debate():
    """debate.py: bounded-confidence deliberation — converge to consensus, split
    into factions, or bridge a would-be hung jury."""
    import debate
    import judge

    # factions: single-linkage clusters on the line
    assert debate.factions([0.1, 0.2, 0.9]) == [[0, 1], [2]]
    assert len(debate.factions([0.5, 0.5, 0.5])) == 1

    # a step only moves you toward peers within EPS
    assert list(debate.step([0.2, 0.8], [1, 1])) == [0.2, 0.8]           # too far → frozen
    assert [round(x, 4) for x in debate.step([0.4, 0.6], [1, 1])] == [0.5, 0.5]
    # zero-weight fallback: unweighted neighborhood mean (matches the JS twin)
    assert [round(x, 4) for x in debate.step([0.4, 0.5], [0, 0])] == [0.45, 0.45]

    items = db.load_json_gallery()
    by = lambda s: next(it for it in items if s in it["path"])
    seat = lambda v: ([x["score"] for x in v if x["score"] is not None],
                      [x["confidence"] for x in v if x["score"] is not None])

    # the tag-fluke agent won't move → contested, dissenter named
    o, w = seat(judge.heuristic_votes(by("000_apple"), by("010_pizza")))
    d = debate.debate(o, w)
    assert d["verdict"] == "abstain" and d["reason"] == "contested"
    assert d["factions"] == [[0, 1], [2]] and d["flips"] == [0]

    # cat → dog: they talk it out and converge
    o, w = seat(judge.heuristic_votes(by("004_cat"), by("005_dog")))
    d = debate.debate(o, w)
    assert d["verdict"] == "relevant" and d["reason"] == "consensus" and d["rounds"] == 3
    assert [round(x, 4) for x in d["final"]] == [0.75, 0.75, 0.75]

    # the bridge: a would-be hung jury (spread 0.6) that a moderate agent
    # deliberates into consensus — what a vote can't do
    d = debate.debate([0.2, 0.5, 0.8], [1, 1, 1])
    assert d["consensus"] and d["n_factions"] == 1
    assert [round(x, 4) for x in d["final"]] == [0.5, 0.5, 0.5]

    # hitting max_rounds reports rounds == max_rounds (twin off-by-one guard)
    capped = debate.debate([0.0, 0.28, 0.56, 0.84], [1, 1, 1, 1], max_rounds=3)
    assert capped["rounds"] == 3 and len(capped["trajectory"]) == 4

    # abstained judges get no seat
    names, ops, _ = debate.from_council([{"name": "a", "score": 0.8, "confidence": 0.9},
                                         {"name": "b", "score": None, "confidence": 0.7},
                                         {"name": "c", "score": 0.6, "confidence": 0.6}])
    assert names == ["a", "c"] and len(ops) == 2

    assert debate.EPS == 0.30 and debate.RELEVANT == 0.5


def test_grow():
    """grow.py: the offline gallery-growth logic — dedup + the db.json export."""
    import json

    import grow

    seed = grow._seed_records()
    assert len(seed) >= 10
    # the curated gallery has no near-duplicates; an exact copy is dropped
    assert len(grow.dedup(seed)) == len(seed)
    assert len(grow.dedup(seed + [dict(seed[0])])) == len(seed)

    # build_payload reproduces the exact db.json shape (items + 2-D PCA basis)
    payload = grow.build_payload(seed)
    assert payload["dim"] == 512 and len(payload["items"]) == len(seed)
    assert len(payload["pca"]["mean"]) == 512 and len(payload["pca"]["components"]) == 2
    it = payload["items"][0]
    assert len(it["image_emb"]) == 512 and len(it["text_emb"]) == 512
    assert 0.0 <= it["map"][0] <= 1.0 and 0.0 <= it["map"][1] <= 1.0
    assert json.loads(json.dumps(payload))  # serializes cleanly

    # a big diverse vocabulary → a ~100x gallery at a dozen images each
    assert len(grow.DEFAULT_TOPICS) >= 100
    assert len(grow.DEFAULT_TOPICS) * 12 >= 100 * len(seed) // 10


def test_reason():
    """reason.py: the end-to-end reasoning trace and the consequence map."""
    import reason

    # the consequence map — every branch
    assert reason.consequence({"level": "high"}, {}, None)["action"] == "show it as the answer"
    assert reason.consequence({"level": "high"}, {}, None)["status"] == "ok"
    split = reason.consequence({"level": "abstain", "reason": "split decision"}, {}, None)
    assert split["status"] == "stop" and "contested" in split["action"]
    assert "not enough signal" in reason.consequence(
        {"level": "abstain", "reason": "not enough signals"}, {}, None)["action"]
    # a contested debate forces the contested branch regardless of trust's reason
    assert "contested" in reason.consequence(
        {"level": "abstain", "reason": "ruled"}, {}, {"consensus": False})["action"]
    med = reason.consequence({"level": "medium"}, {"decision": "abstain"}, None)
    assert med["status"] == "caution" and "council couldn't" in med["because"]
    assert "weak" in reason.consequence({"level": "low"}, {"decision": "not relevant"}, None)["action"]

    items = db.load_json_gallery()
    by = lambda s: next(it for it in items if s in it["path"])

    # cat → dog: every step passes → high trust → show it
    cat = reason.trace(by("004_cat"), items)
    assert "005_dog" in cat["result"]["path"]
    assert len(cat["steps"]) == 6 and all(s["status"] == "ok" for s in cat["steps"])
    assert cat["trust"]["level"] == "high" and cat["consequence"]["action"] == "show it as the answer"

    # apple → pizza: retrieve/conformal ok, council + debate STOP → caveat
    apple = reason.trace(by("000_apple"), items)
    stat = {s["stage"]: s["status"] for s in apple["steps"]}
    assert stat["retrieve"] == "ok" and stat["conformal"] == "ok"
    assert stat["council"] == "stop" and stat["debate"] == "stop"
    assert apple["trust"]["level"] == "medium"
    assert apple["consequence"]["status"] == "caution"
    assert apple["consequence"]["action"] == "show it with a caveat"
    assert apple["debate"]["consensus"] is False


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
