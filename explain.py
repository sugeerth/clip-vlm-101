"""explain.py — say WHY these results matched, and never lie about it.

pipeline: query + ranked results ──► [explain] ──► a grounded sentence + a GATE

A search returns images; a friendly system also says *why*. But the instant
you generate prose you can hallucinate — describe a "sunset" no result shows,
quote a score that never happened. So this file is two halves, and the second
is the important one:

  EXPLAIN   build a short explanation from ONLY verifiable facts — the tags
            SHARED across the top results (the common thread), the strongest
            similarity bucketed into a plain word, and honesty when the model
            isn't confident. Grounded BY CONSTRUCTION: it can only emit
            evidence tokens and fixed glue, so it always passes the gate.

  GATE      verify(text, evidence): the trust boundary. Whatever wrote the
            text — this template, or an untrusted in-browser LLM — every
            SENTENCE is checked. A sentence is kept only if every content word
            is a known tag/query word (or safe glue), every number matches a
            real score, and every strength word is the true one. Anything else
            is a hallucination and the whole sentence is redacted, with a
            reason.

Why exact-match membership is a legitimate hallucination check: the evidence
here is a CLOSED, FINITE set of tags and numbers, so "is this claim supported
by the source?" collapses from an NLI/LLM-judge problem to set membership.
This is the strict special case of *attribution / faithfulness* — AIS
(Attributable to Identified Sources, Rashkin et al. 2023, arXiv:2112.12870)
and RAGAS faithfulness — implemented as an exact lexical gate, no model
needed. The template is the floor the gate can always fall back to.
"""
import re

import numpy as np

import templates
from search import score

# Match-strength buckets for RAW cosine of L2-normalized CLIP ViT-B/32 vectors
# (the unscaled value, not ×100). Heuristics, not law — LAION-400M kept pairs
# at ≥0.30, LAION-5B at ~0.26–0.28; real matches live ~0.15–0.40. Overridable.
STRONG, MODERATE, WEAK = 0.30, 0.25, 0.20
NUM_TOL = 0.02          # a claimed number may differ from a score by this much
TOP_K = 5

STRENGTH_WORDS = {"strong", "moderate", "weak", "very weak", "perfect", "exact"}


def bucket(s: float) -> str:
    return ("strong" if s >= STRONG else "moderate" if s >= MODERATE
            else "weak" if s >= WEAK else "very weak")


def _norm(w: str) -> str:
    """lowercase, strip surrounding punctuation, drop a trailing plural 's'."""
    w = re.sub(r"[^\w.%-]", "", w.lower())
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


# The closed whitelist of glue/meta words the template may use. The gate strips
# any OTHER content word, so this must cover every non-evidence word the
# template emits (both strong- AND weak-match tails) — test_explain locks that
# invariant. NORMALIZED through _norm so text and vocab agree on plurals.
# Expand to reduce false strips; never loosen the tag/number tests.
SAFE_VOCAB = {_norm(w) for w in (
    "the", "a", "an", "these", "this", "they", "all", "both", "most", "and",
    "or", "of", "to", "with", "in", "on", "no", "not", "dont", "single",
    "image", "images", "result", "results", "match", "matches", "matched",
    "similar", "similarity", "score", "scores", "query", "search", "top",
    "share", "shares", "shared", "common", "tag", "tags", "show", "shows",
    "way", "ways", "different", "strongest", "confident", "loose", "treat",
    "them", "as", "isnt", "is", "are", "that", "it", "model", "very", "explain",
)}


def _numbers(text: str):
    """Every number a text claims, as fractions (percent -> /100)."""
    out = [float(m.group(1)) / 100.0 for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text)]
    out += [float(m.group(0)) for m in re.finditer(r"(?<![\d%])0?\.\d+(?![\d%])", text)]
    return out


def evidence(query, ranked, k: int = TOP_K, vocabulary=templates.TAG_VOCABULARY) -> dict:
    """The verifiable world: the tags/scores the results actually support."""
    top = ranked[:k]
    if not top:
        return {"query": query, "query_toks": set(), "tags": set(), "shared": [],
                "scores": [], "top_score": 0.0, "strength": "very weak"}
    tag_sets = [set(_norm(t) for t in it.get("tags", [])) for it, _ in top]
    allowed = set().union(*tag_sets)
    shared = [t for t in top[0][0].get("tags", []) if all(_norm(t) in ts for ts in tag_sets)]
    scores = [round(float(s), 2) for _, s in top]
    return {
        "query": query,
        "query_toks": {_norm(w) for w in re.split(r"\s+", query or "")} - {""},
        "tags": allowed, "shared": shared, "scores": scores,
        "top_score": max(scores), "strength": bucket(max(scores)),
    }


def describe(ev: dict, k: int = TOP_K) -> str:
    """A grounded explanation built ONLY from the evidence (passes its own gate)."""
    if not ev["scores"]:
        return "No results to explain."
    n = len(ev["scores"])
    head = (f"The top {n} results all show {_join(ev['shared'][:3])}."
            if ev["shared"] else
            f"The top {n} results share no single tag — they match the query in different ways.")
    tail = f" The strongest match scores {ev['top_score']:.2f} ({ev['strength']})."
    if ev["strength"] in ("weak", "very weak"):
        tail += " Treat them as loose matches — the model isnt confident."
    return head + tail


def verify(text: str, ev: dict, num_tol: float = NUM_TOL) -> dict:
    """The hallucination gate — closed-world exact-match attribution.

    A sentence survives only if every content word is a real tag, a query
    word, or safe glue; every number matches a real score; every strength
    word is the true one. Otherwise the whole sentence is redacted (a
    half-deleted sentence is still misleading). Returns the verified text and
    the stripped claims with reasons.
    """
    kept, stripped = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence.strip():
            continue
        reasons, vague = [], []
        low = sentence.lower()
        for phrase in STRENGTH_WORDS:                         # wrong confidence word
            if phrase != ev["strength"] and re.search(rf"\b{phrase}\b", low) \
                    and not (phrase == "weak" and ev["strength"] == "very weak"):
                reasons.append(f"says '{phrase}' — the match is '{ev['strength']}'")
        for num in _numbers(sentence):                        # a figure no score supports
            if not any(abs(num - s) <= num_tol for s in ev["scores"]):
                reasons.append(f"cites {num:.2f} — matches no result score")
        for raw in re.findall(r"[A-Za-z][\w-]*", sentence):
            w = _norm(raw)
            if not w or w.isdigit() or w in SAFE_VOCAB or w in STRENGTH_WORDS \
                    or w in ev["tags"] or w in ev["query_toks"]:
                continue
            if w in templates.TAG_VOCABULARY:                 # a KNOWN concept, absent
                reasons.append(f"claims '{raw}' — not in the results")
            else:                                             # unverifiable wording
                vague.append(raw)
        if vague and not reasons:                             # only vague words → one note
            reasons.append(f"unverifiable wording: {', '.join(vague[:4])}")
        elif vague:
            reasons.append(f"and unverifiable wording: {', '.join(vague[:4])}")
        (stripped if reasons else kept).append({"text": sentence, "reasons": reasons})
    return {
        "verified": " ".join(s["text"] for s in kept),
        "stripped": [s for s in stripped],
        "clean": not stripped,
    }


def explain(query, ranked, k: int = TOP_K, draft: str | None = None,
            vocabulary=templates.TAG_VOCABULARY) -> dict:
    """One call: evidence -> grounded explanation, gated. If `draft` is given
    (e.g. an LLM's prose), it is verified INSTEAD of the template and whatever
    it invents is stripped; the template is the floor if nothing survives."""
    ev = evidence(query, ranked, k, vocabulary)
    text = draft if draft is not None else describe(ev, k)
    result = verify(text, ev)
    result["evidence"] = ev
    result["explanation"] = result["verified"] or describe(ev, k)   # never blank
    return result


def _join(words):
    words = list(words)
    if len(words) <= 1:
        return words[0] if words else ""
    return ", ".join(words[:-1]) + " and " + words[-1]


if __name__ == "__main__":
    import argparse

    import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="images/004_cat.jpg",
                    help="a gallery image path to query with (model-free, image-to-image)")
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    items = db.load_json_gallery(args.json)
    match = [it for it in items if args.image in it["path"]]
    if not match:
        raise SystemExit(f"no gallery image matches {args.image!r}")
    q = np.asarray(match[0]["image_emb"], dtype=np.float64)
    ranked = sorted(((it, score(it, q, "image")) for it in items if it is not match[0]),
                    key=lambda pair: pair[1], reverse=True)[: args.k]

    out = explain(match[0].get("caption", args.image), ranked, args.k)
    print(f"query image: {match[0]['path']}\n")
    for it, s in ranked:
        print(f"  {s:+.3f}  {it['path']}   tags: {', '.join(it['tags'])}")
    print(f"\nexplanation: {out['explanation']}")

    # the GATE catching an injected LLM hallucination: a concept the results
    # don't support ("train", a real tag but absent) AND a fabricated score.
    lie = describe(out["evidence"], args.k) + " They also clearly show a train at 0.97."
    checked = verify(lie, out["evidence"])
    print(f"\ngate demo — an untrusted LLM draft with two invented claims:\n  {lie}")
    print(f"  → verified: {checked['verified']}")
    for s in checked["stripped"]:
        print(f"  → STRIPPED: “{s['text']}”  ({'; '.join(s['reasons'])})")
