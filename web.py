"""web.py — search the whole internet's images, for free, in CLIP space.

pipeline: query ──► [clip-retrieval KNN backend] ──► image URLs + captions + similarity

The gallery is 14 images because WE didn't index more. But the internet's images
are ALREADY indexed — for free, at billion scale, in the very CLIP space this repo
teaches. LAION crawled ~5-6 billion image-text pairs from Common Crawl and released
them CLIP-embedded (the safety-cleaned Re-LAION-5B); rom1504's `clip-retrieval`
builds a memory-mapped FAISS index over those embeddings and serves nearest-neighbour
search. So "search the internet's images" is not a crawl you run — it's a KNN query
you send, and the index cost is $0 because someone already paid it and gave it away.

This is the CLIENT. It POSTs a query to any clip-retrieval-compatible KNN backend
(the public one, or your own) and returns standardized records — url, caption,
similarity — the same shape the local gallery uses, so the SAME ranking, grid and
ranking-quality agent work over 14 images or 5 billion. The backend encodes the
query itself (text or an image URL), so the client is embedding-space-agnostic;
point it at a ViT-B/32 index to match this repo exactly.

Free, at your own scale (all open-source, dataset is URLs not redistributed images):
    img2dataset   download+resize the image-text pairs (or reuse Re-LAION-5B)
    open_clip     embed them (the same CLIP family)
    autofaiss     build a memory-mapped IVF/PQ index that fits one machine
    clip-retrieval serve KNN — the backend this client talks to

    python3 web.py "a red panda in snow" --backend https://knn.laion.ai/knn-service
    python3 web.py --selftest        # model- and network-free: parse + dedup + errors
"""
import argparse
import json
import urllib.request

# A clip-retrieval backend. The public LAION one (knn.laion.ai) has been
# intermittent-to-DOWN since the Dec-2023 takedown, so treat it as best-effort and
# self-host for anything reliable (see the recipe above). The client is written to
# clip-retrieval's documented POST /knn-service contract, so it works against your
# own backend unchanged.  The 5B index is ViT-L/14 (768-d); because the SERVER
# encodes your text, the client is embedding-space-agnostic for text queries — you
# don't need to match dimensions. (To match this repo's ViT-B/32 space exactly, run
# a LAION-400M B/32 index; the 5B set is L/14.)
DEFAULT_BACKEND = "https://knn.laion.ai/knn-service"
DEFAULT_INDEX = "laion5B-L-14"        # clip-retrieval's public 5B index name (ViT-L/14)
UA = "clip-vlm-101-web/1.0 (https://github.com/sugeerth/clip-vlm-101)"
RETRIES = 3


def _post(url, payload, opener=None, timeout=30):
    """POST JSON, return parsed JSON. `opener` is injectable so tests (and the
    --selftest) can stand in a canned backend without touching the network."""
    if opener is None:
        def opener(u, data, headers, to):
            req = urllib.request.Request(u, data=data, headers=headers, method="POST")
            return urllib.request.urlopen(req, timeout=to).read()
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    last = None
    for attempt in range(RETRIES):
        try:
            return json.loads(opener(url, body, headers, timeout))
        except Exception as e:                       # transient: retry with backoff
            last = e
    raise last


def _record(hit):
    """One backend hit → the repo's standard record shape (url, caption, score)."""
    return {
        "url": hit.get("url") or hit.get("image") or "",
        "caption": (hit.get("caption") or "").strip(),
        "similarity": float(hit.get("similarity", hit.get("score", 0.0)) or 0.0),
    }


def dedup(records):
    """Drop repeated URLs (the web has many copies of the same image), keep order."""
    seen, out = set(), []
    for r in records:
        u = r["url"]
        if u and u not in seen:
            seen.add(u)
            out.append(r)
    return out


def search(query, n=20, backend=DEFAULT_BACKEND, index=DEFAULT_INDEX,
           image_url=None, safe=True, opener=None):
    """Query the internet-scale index. Text OR an image URL; returns up to `n`
    deduped records sorted by similarity. `opener` is injectable for tests."""
    payload = {
        "modality": "image", "num_images": n, "num_result_ids": n,
        "indice_name": index, "deduplicate": True,
        "use_safety_model": safe, "use_violence_detector": safe,
    }
    if image_url:
        payload["image_url"] = image_url
    else:
        payload["text"] = query
    hits = _post(backend, payload, opener=opener)
    if not isinstance(hits, list):                   # some deployments wrap in a dict
        hits = hits.get("results", hits.get("images", [])) if isinstance(hits, dict) else []
    records = dedup(_record(h) for h in hits if (h.get("url") or h.get("image")))
    records.sort(key=lambda r: r["similarity"], reverse=True)
    return records[:n]


# ------- a canned backend so the whole path is testable with no network -------

def _fake_backend(url, body, headers, timeout):
    """Stand-in clip-retrieval backend: echoes a small, realistic result set
    (with a duplicate URL and a blank-url hit, to exercise dedup + filtering)."""
    q = json.loads(body).get("text", "")
    base = [
        {"url": "https://ex.com/a.jpg", "caption": f"{q} one", "similarity": 0.42},
        {"url": "https://ex.com/b.jpg", "caption": f"{q} two", "similarity": 0.55},
        {"url": "https://ex.com/a.jpg", "caption": f"{q} dup", "similarity": 0.30},
        {"url": "", "caption": "no url", "similarity": 0.9},
        {"url": "https://ex.com/c.jpg", "caption": f"{q} three", "similarity": 0.48},
    ]
    return json.dumps(base).encode()


def _selftest():
    got = search("cat", n=10, opener=_fake_backend)
    assert [r["url"] for r in got] == ["https://ex.com/b.jpg", "https://ex.com/c.jpg",
                                       "https://ex.com/a.jpg"], "dedup + sort by similarity"
    assert all(r["url"] for r in got), "blank-url hits are dropped"
    assert got[0]["similarity"] >= got[-1]["similarity"], "sorted descending"
    # retries: a backend that fails twice then succeeds still returns
    calls = {"n": 0}
    def flaky(url, body, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        return _fake_backend(url, body, headers, timeout)
    assert len(search("dog", opener=flaky)) == 3 and calls["n"] == 3, "retries then succeeds"
    print("web selftest passed  (parse + dedup + sort + retry, no network)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="text query (or use --image-url)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--image-url", help="image→image: search by a picture's URL")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--selftest", action="store_true",
                    help="model- and network-free: parse + dedup + retry")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        raise SystemExit(0)
    if not (args.query or args.image_url):
        ap.error("give a text query or --image-url")

    hits = search(args.query or "", n=args.n, backend=args.backend, index=args.index,
                  image_url=args.image_url)
    print(f"{len(hits)} results from {args.index} @ {args.backend}\n")
    for i, r in enumerate(hits, 1):
        print(f"  {i:>2}. {r['similarity']:.3f}  {r['url']}")
        if r["caption"]:
            print(f"      {r['caption'][:80]}")
    print("\nthe internet's images, already CLIP-indexed for free — you sent a query, "
          "not a crawler.")


if __name__ == "__main__":
    main()
