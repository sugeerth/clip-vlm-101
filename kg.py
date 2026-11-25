"""kg.py — ground the search in an external knowledge graph (Wikidata), fast.

pipeline: term ──► [Wikidata: entity · aliases · is-a] ──► docs/kg.json ──► instant expansion

A search box only knows the words in its captions. A KNOWLEDGE GRAPH knows what
things ARE: that "feline" and "housecat" name the cat (Q146), that a parrot IS-A
bird, that the Eiffel Tower is a lattice tower in Paris. Wikidata is that graph —
~115M entities, collaboratively maintained, free, with an open API — and this
module grounds the gallery in it.

Speed is the design constraint, so the graph is consulted at BUILD time, not
query time: `--build` resolves every gallery tag against Wikidata once (entity id,
canonical label, description, aliases, what it's an instance/subclass of) and
commits the slice as docs/kg.json — a few KB. The browser then expands queries
INSTANTLY from the committed slice ("feline" → cat) with zero network on the hot
path; live Wikidata lookups are a cached progressive enhancement for words the
slice doesn't know. External knowledge, local latency.

Like crawler.py and web.py, the HTTP layer is injectable, so the whole path —
entity resolution, alias harvest, is-a labels, slice build — runs model- and
network-free in tests against canned API responses.

    python3 kg.py cat                  # resolve one term against Wikidata
    python3 kg.py --build              # resolve every gallery tag → docs/kg.json
    python3 kg.py --selftest           # network-free: parse + build, canned responses
"""
import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

API = "https://www.wikidata.org/w/api.php"
UA = "clip-vlm-101-kg/1.0 (https://github.com/sugeerth/clip-vlm-101)"
ISA_PROPS = ("P31", "P279")        # instance-of, subclass-of — the "what is it" edges
MAX_ALIASES = 8
MAX_RELATED = 5


def _get(url, opener=None, timeout=30):
    if opener is None:
        def opener(u, to):
            req = urllib.request.Request(u, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=to).read()
    return json.loads(opener(url, timeout))


def _api(params, opener=None):
    qs = urllib.parse.urlencode({**params, "format": "json", "origin": "*"})
    return _get(f"{API}?{qs}", opener=opener)


def resolve(term, opener=None):
    """One term → its Wikidata grounding: {term, id, label, description, aliases,
    related} or None if the graph doesn't know it. `related` holds the labels of
    what the entity is an instance/subclass of — its place in the graph."""
    found = _api({"action": "wbsearchentities", "search": term,
                  "language": "en", "limit": 1, "type": "item"}, opener)
    hits = found.get("search") or []
    if not hits:
        return None
    qid = hits[0]["id"]
    got = _api({"action": "wbgetentities", "ids": qid, "languages": "en",
                "props": "labels|descriptions|aliases|claims"}, opener)
    ent = (got.get("entities") or {}).get(qid) or {}
    label = ((ent.get("labels") or {}).get("en") or {}).get("value", term)
    desc = ((ent.get("descriptions") or {}).get("en") or {}).get("value", "")
    aliases = [a["value"] for a in (ent.get("aliases") or {}).get("en", [])][:MAX_ALIASES]
    # the is-a edges: target QIDs of P31/P279, then one batched label fetch
    targets = []
    for prop in ISA_PROPS:
        for claim in (ent.get("claims") or {}).get(prop, []):
            v = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
            if isinstance(v, dict) and v.get("id"):
                targets.append(v["id"])
    related = []
    if targets:
        got2 = _api({"action": "wbgetentities", "ids": "|".join(targets[:MAX_RELATED]),
                     "languages": "en", "props": "labels"}, opener)
        for tid in targets[:MAX_RELATED]:
            lab = (((got2.get("entities") or {}).get(tid) or {}).get("labels") or {})
            if (lab.get("en") or {}).get("value"):
                related.append(lab["en"]["value"])
    return {"term": term, "id": qid, "label": label, "description": desc,
            "aliases": aliases, "related": related}


def build(tags, opener=None, pause=0.3):
    """Resolve every tag once (politely) → the committed slice for docs/kg.json."""
    entities = {}
    for i, tag in enumerate(sorted(set(tags))):
        ent = resolve(tag, opener=opener)
        if ent:
            entities[tag] = ent
        if opener is None and i < len(set(tags)) - 1:
            time.sleep(pause)                    # real API: one request at a time
    return {"source": "wikidata", "api": API, "entities": entities}


def expansions(kg):
    """The instant hot path: a lowercase alias/label → canonical tag map, so a
    query word the captions never used still finds its images ("feline" → cat)."""
    out = {}
    for tag, ent in (kg.get("entities") or {}).items():
        for name in [ent.get("label", "")] + list(ent.get("aliases") or []):
            key = name.strip().lower()
            if key and key != tag:
                out.setdefault(key, tag)
    return out


def gallery_tags(db_path="docs/db.json"):
    db = json.loads(pathlib.Path(db_path).read_text())
    return sorted({t for it in db["items"] for t in it.get("tags", [])})


# ---- canned Wikidata responses: the whole path, no network ----

def _fake_opener(url, timeout):
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    if q["action"][0] == "wbsearchentities":
        term = q["search"][0]
        if term == "unknownium":
            return b'{"search": []}'
        return json.dumps({"search": [{"id": "Q146", "label": "house cat"}]}).encode()
    ids = q["ids"][0]
    if ids == "Q146":
        return json.dumps({"entities": {"Q146": {
            "labels": {"en": {"value": "house cat"}},
            "descriptions": {"en": {"value": "domesticated feline"}},
            "aliases": {"en": [{"value": "feline"}, {"value": "housecat"}, {"value": "Felis catus"}]},
            "claims": {"P279": [{"mainsnak": {"datavalue": {"value": {"id": "Q39201"}}}}]},
        }}}).encode()
    return json.dumps({"entities": {"Q39201": {"labels": {"en": {"value": "pet"}}}}}).encode()


def _selftest():
    ent = resolve("cat", opener=_fake_opener)
    assert ent["id"] == "Q146" and ent["description"] == "domesticated feline"
    assert "feline" in ent["aliases"] and ent["related"] == ["pet"]
    assert resolve("unknownium", opener=_fake_opener) is None, "unknown terms stay None"
    kg = build(["cat", "unknownium"], opener=_fake_opener)
    assert set(kg["entities"]) == {"cat"}
    exp = expansions(kg)
    assert exp["feline"] == "cat" and exp["house cat"] == "cat" and "cat" not in exp
    assert json.loads(json.dumps(kg))                    # round-trips as JSON
    print("kg selftest passed  (resolve + build + alias→tag expansion, no network)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("term", nargs="?", help="a term to resolve against Wikidata")
    ap.add_argument("--build", action="store_true",
                    help="resolve every gallery tag and write docs/kg.json")
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--out", default="docs/kg.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        raise SystemExit(0)

    if args.build:
        tags = gallery_tags(args.json)
        print(f"resolving {len(tags)} gallery tags against Wikidata…")
        kg = build(tags)
        pathlib.Path(args.out).write_text(json.dumps(kg, ensure_ascii=False))
        exp = expansions(kg)
        print(f"wrote {args.out}: {len(kg['entities'])}/{len(tags)} tags grounded, "
              f"{len(exp)} instant expansions (e.g. "
              f"{', '.join(f'{k}→{v}' for k, v in list(exp.items())[:4])})")
        raise SystemExit(0)

    if not args.term:
        ap.error("give a term, --build, or --selftest")
    ent = resolve(args.term)
    if not ent:
        print(f"wikidata doesn't know {args.term!r}")
        raise SystemExit(1)
    print(f"{ent['term']} → {ent['id']}  {ent['label']}")
    print(f"  {ent['description']}")
    if ent["aliases"]:
        print(f"  also called: {', '.join(ent['aliases'])}")
    if ent["related"]:
        print(f"  is a: {', '.join(ent['related'])}")


if __name__ == "__main__":
    main()
