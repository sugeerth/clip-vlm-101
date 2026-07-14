"""dcn.py — the RANKING stage: the interaction a dot product can't express.

pipeline: retrieved candidates ──► [DCN cross network] ──► re-ordered top-k

Two towers retrieve FAST precisely because they never interact: the query
vector and each item vector are built apart and compared with ONE dot
product. That independence is what makes billion-scale ANN possible — and it
is also a ceiling. A single bilinear score q·v cannot learn "when the query
fires on THESE dimensions, weight tag-overlap high and raw cosine low." There
is no query×item feature interaction, by construction.

Real systems fix this with a two-stage funnel (YouTube RecSys'16, Google
Play Wide&Deep, Pinterest): cheap RETRIEVAL narrows billions → hundreds
(this repo's two-tower + ann.py/pq.py/scale.py), then a rich RANKING model
re-scores only those hundreds with full feature interaction. It can afford
to — it runs on a tiny set, never the corpus.

This file is that ranker, as a Deep & Cross Network v2 (Wang et al., WWW
2021, arXiv:2008.13535). Its whole idea is ONE explicit multiplicative cross
per layer:

    x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l

x_0 is the feature vector; ⊙ is element-wise; W_l is a d×d matrix mixing
every feature before the gate; the residual +x_l keeps lower-degree crosses.
l layers make feature crosses up to degree l+1 — depth IS the interaction
order, explicitly (a plain MLP only ever crosses features implicitly).

The features are the per-candidate signals a dot product throws away:

    x_0 = [ cos_image, cos_text, tag_overlap, rank_prior ]

At W=0, b=0 the residual makes every layer a passthrough and the default head
reads cos_fused — so the DCN reproduces the retrieval order EXACTLY. Switch on
ONE cross weight and it expresses cos_image·tag_overlap — "prefer candidates
that both LOOK similar AND SHARE a tag" — which the two-tower score structurally
cannot represent. Untrained, this DEMONSTRATES the mechanism; production learns
W from click/relevance labels (there is no gradient here, on purpose — the
lesson is the cross, not the fit). Real feature vectors also concatenate the
raw q, v_img, v_txt embeddings, which is why W is a full d×d matrix.
"""
import numpy as np

FEATURES = ("cos_image", "cos_text", "tag_overlap", "rank_prior")
FUSED_HEAD = np.array([0.5, 0.5, 0.0, 0.0])   # reads (cos_image+cos_text)/2


def make_features(cos_image, cos_text, tag_overlap, rank_prior) -> np.ndarray:
    """x_0 — the per-candidate signals the retrieval dot product drops."""
    return np.array([cos_image, cos_text, tag_overlap, rank_prior], dtype=np.float64)


class CrossNetwork:
    """DCN-v2 cross layers: x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l."""

    def __init__(self, dim: int, num_layers: int = 2):
        self.dim, self.num_layers = dim, num_layers
        self.Ws = [np.zeros((dim, dim)) for _ in range(num_layers)]  # W=0 → passthrough
        self.bs = [np.zeros(dim) for _ in range(num_layers)]

    def set_cross(self, layer, out_feat, in_feat, weight):
        """Hand-set ONE cross so a learner can watch a single interaction fire:
        output feature `out_feat` gets mixed with input feature `in_feat`."""
        self.Ws[layer][out_feat, in_feat] = weight
        return self

    def init_random(self, seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        self.Ws = [rng.normal(scale=scale, size=(self.dim, self.dim)) for _ in range(self.num_layers)]
        self.bs = [rng.normal(scale=scale, size=self.dim) for _ in range(self.num_layers)]
        return self

    def forward(self, x0: np.ndarray) -> np.ndarray:
        x = x0.copy()
        for W, b in zip(self.Ws, self.bs):
            x = x0 * (W @ x + b) + x           # the DCN-v2 cross, verbatim
        return x


def rerank(candidates, cross: CrossNetwork, head=FUSED_HEAD, k=None):
    """Re-score retrieved candidates with the DCN and re-order by the logit.

    candidates: dicts carrying at least the FEATURES keys. Returns them sorted
    by DCN logit (`head · cross(x0)`), each annotated with 'dcn_score'.
    """
    scored = []
    for c in candidates:
        x0 = make_features(c["cos_image"], c["cos_text"], c["tag_overlap"], c["rank_prior"])
        scored.append({**c, "dcn_score": float(head @ cross.forward(x0))})
    scored.sort(key=lambda c: c["dcn_score"], reverse=True)
    return scored[:k] if k else scored


def candidates_from_gallery(query_item, items, query_tags=None):
    """Build re-ranker candidates from a query image over the gallery — the
    exact signals stage 1 produced, ready for stage 2. Model-free."""
    q_img = np.asarray(query_item["image_emb"], dtype=np.float64)
    q_txt = np.asarray(query_item.get("text_emb", query_item["image_emb"]), dtype=np.float64)
    qtags = set(query_tags if query_tags is not None else query_item.get("tags", []))
    pool = [it for it in items if it is not query_item]
    by_cos = sorted(pool, key=lambda it: float(np.asarray(it["image_emb"]) @ q_img), reverse=True)
    cand = []
    for rank, it in enumerate(by_cos):
        cand.append({
            "item": it,
            "cos_image": float(np.asarray(it["image_emb"], dtype=np.float64) @ q_img),
            "cos_text": float(np.asarray(it["text_emb"], dtype=np.float64) @ q_txt),
            "tag_overlap": len(qtags & set(it.get("tags", []))),
            "rank_prior": 1.0 / (rank + 1),
        })
    return cand


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="images/004_cat.jpg",
                    help="a gallery image path to query with (model-free)")
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--cross-weight", type=float, default=6.0,
                    help="strength of the cos_image × tag_overlap cross")
    args = ap.parse_args()

    items = db.load_json_gallery(args.json)
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    cand = candidates_from_gallery(match[0], items)

    passthrough = CrossNetwork(dim=len(FEATURES), num_layers=1)          # W=0
    crossed = CrossNetwork(dim=len(FEATURES), num_layers=1)
    crossed.set_cross(0, FEATURES.index("cos_image"),
                      FEATURES.index("tag_overlap"), args.cross_weight)

    print(f"query image: {match[0]['path']}  tags: {', '.join(match[0]['tags'])}\n")
    print("stage 1 — retrieval order (fused cosine, no interaction):")
    for c in rerank(cand, passthrough, k=args.k):
        print(f"  {c['dcn_score']:+.3f}  {c['item']['path']}  "
              f"(cos {c['cos_image']:+.3f}, shared tags {c['tag_overlap']})")
    print(f"\nstage 2 — DCN re-rank with one cross (cos_image × tag_overlap, w={args.cross_weight}):")
    for c in rerank(cand, crossed, k=args.k):
        print(f"  {c['dcn_score']:+.3f}  {c['item']['path']}  "
              f"(cos {c['cos_image']:+.3f}, shared tags {c['tag_overlap']})")
    print("\nthe cross lifts candidates that both look similar AND share tags —"
          "\nan interaction one dot product cannot express.")
