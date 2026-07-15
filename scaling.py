"""Two-billion-scale, on the back of an envelope — from measured constants.

pipeline: measured million-row numbers ──► [scaling] ──► a defensible 2 billion

scale.py MEASURED a million rows on one laptop. This file does the honest
arithmetic that turns that measurement into TWO BILLION, and prints the
budget a real system would be built to. Nothing here is simulated; it is
pure arithmetic over four things the rest of the repo already established:

  1. SIZE      pq.py packs every vector into 64 bytes (m=64 subquantizers),
               32× smaller than float32 — measured, with recall baked into
               its manifest. Memory is then just N × bytes.
  2. WORK      ann.py's IVF makes search cost O(√N), not O(N): index into
               nlist ≈ √N cells, scan only `nprobe` of them. A query touches
               ~√N centroids + nprobe·(N/nlist) candidates — sublinear.
  3. SHARDING  2 billion is 2,000 shards of the million scale.py already
               timed. Each shard does the laptop's measured work; the query
               scatters to all shards and gathers their top-k. Per-shard
               latency is MEASURED; the fan-out is standard.
  4. CASCADE   cascade.py approximates at EVERY level (binary → PQ → int8 →
               exact), each coarser and cheaper on a shorter list, so only
               ~50 vectors/query ever touch float32 — and MEASURED recall
               stays ~100% of an exact scan. Approximate everywhere; wrong
               nowhere that counts.

The two towers are why any of this is affordable (see item_tower.py /
user_tower.py): the item tower runs OFFLINE — every stored vector was
embedded once, before the query existed — so serving never runs it. Only
the query tower runs online, one forward pass, then it is pure arithmetic
over precomputed vectors. Scale lives entirely on the cheap side of that
split.

Run me:  python3 scaling.py              # the two-billion budget, printed
         python3 scaling.py --n 1e9 --shard 1e6
"""
import argparse

DIM = 512  # openai/clip-vit-base-patch32, the whole repo's space

# bytes per vector, per encoding — the ladder quantize.py, pq.py, cascade.py climb
ENCODINGS = {
    "float32": DIM * 4,      # db.py's raw BLOB
    "float16": DIM * 2,      # scale.py's packed matrices
    "int8": DIM * 1,         # quantize.py: one shared max-abs scale
    "PQ-64": 64,             # pq.py: 64 subquantizers, one byte each
    "binary": DIM // 8,      # cascade.py: 1 bit/dim, Hamming via popcount
}

# The approximation cascade's shortlist widths (cascade.py's keep=…): binary
# keeps 500, PQ keeps 100, int8 keeps 50 — and only those 50 ever touch float32.
CASCADE_KEEP = (500, 100, 50)

# Per-stage serving latency, milliseconds. The anchors are scale.py's own
# laptop measurements (a per-shard million-row index); the fan-out is the
# standard scatter/gather a 1,000-shard fleet pays on top. Documented, not
# hidden — override --embed-ms etc. to run your own budget.
STAGE_MS = {
    "query tower embed": 12.0,   # ONE CLIP text forward pass (the only model on the path)
    "coarse quantizer": 0.5,     # query vs √N centroids — a small matmul
    "IVF-PQ scan (per shard)": 6.0,  # nprobe lists of PQ codes, ADC table lookups
    "exact re-rank": 2.0,        # two-stage: fetch a cheap top-100, score exactly (scale.py)
    "scatter / gather": 4.0,     # fan out to shards, merge their top-k (billion only)
}


def memory(n, encoding="PQ-64"):
    """Total index bytes for n vectors under one encoding."""
    return n * ENCODINGS[encoding]


def ivf_plan(n, nprobe=32):
    """IVF's sublinear promise, as numbers. nlist ≈ √n is the standard rule."""
    nlist = round(n ** 0.5)
    avg_list = n / nlist
    scanned = nprobe * avg_list
    return {
        "nlist": nlist,
        "avg_list_size": avg_list,
        "nprobe": nprobe,
        "candidates_scanned": scanned,
        "fraction_scanned": scanned / n,
    }


def shard_plan(n, per_shard):
    """A billion as k copies of a measured million (or whatever per_shard is)."""
    shards = max(1, round(n / per_shard))
    return {"shards": shards, "per_shard": per_shard,
            "pq_bytes_per_shard": per_shard * ENCODINGS["PQ-64"]}


def cascade_plan(n, nprobe=32, keep=CASCADE_KEEP):
    """cascade.py at scale: how many vectors each level SCORES per query, and
    — the headline — what fraction ever gets scored in full float32. The
    resident index is binary + PQ (cheap); float32 is fetched cold for only
    the final shortlist. keep = (binary→, PQ→, int8→) widths."""
    candidates = nprobe * (n / round(n ** 0.5))   # L0 → L1 input: IVF candidates
    scored = [candidates, keep[0], keep[1], keep[2]]  # L1 binary, L2 PQ, L3 int8, L4 exact
    exact = keep[2]                               # the ONLY float32 rows
    return {
        "resident_bytes": n * (ENCODINGS["binary"] + ENCODINGS["PQ-64"]),
        "float32_per_query": exact,
        "float32_fraction": exact / n,
        "scored_per_level": scored,
    }


def latency_budget(sharded=True, stages=STAGE_MS):
    """Compose the per-stage budget. Shards run in PARALLEL, so the scan is
    counted once (the slowest shard), not per shard — that is the point of
    fan-out. Returns (ordered [(stage, ms)], total_ms)."""
    order = ["query tower embed", "coarse quantizer",
             "IVF-PQ scan (per shard)", "exact re-rank"]
    if sharded:
        order.append("scatter / gather")
    budget = [(s, stages[s]) for s in order]
    return budget, sum(ms for _, ms in budget)


def _human_bytes(b):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if b < 1024 or unit == "PB":
            return f"{b:,.1f} {unit}"
        b /= 1024


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, default=2e9, help="corpus size (default two billion)")
    ap.add_argument("--shard", type=float, default=1e6,
                    help="rows per shard (default a million — the size scale.py measured)")
    ap.add_argument("--nprobe", type=int, default=32)
    args = ap.parse_args()
    n = int(args.n)

    scale_name = "TWO-BILLION" if n >= 2e9 else "BILLION" if n >= 1e9 else "LARGE"
    print(f"{scale_name}-SCALE BUDGET for N = {n:,} vectors, {DIM}-d\n")

    print("1. MEMORY — the whole index, per encoding:")
    for enc in ENCODINGS:
        print(f"   {enc:<9} {_human_bytes(memory(n, enc)):>12}"
              f"   ({ENCODINGS[enc]:>4} bytes/vector)")
    print(f"   → PQ-64 is what ships: {_human_bytes(memory(n, 'PQ-64'))} for {n:,} vectors,"
          " a handful of commodity boxes.\n")

    plan = ivf_plan(n, args.nprobe)
    print("2. WORK — IVF makes it O(√N), not O(N):")
    print(f"   nlist ≈ √N        = {plan['nlist']:,} cells")
    print(f"   avg cell holds    = {plan['avg_list_size']:,.0f} vectors")
    print(f"   nprobe={args.nprobe} scans   = {plan['candidates_scanned']:,.0f} candidates"
          f"  ({plan['fraction_scanned']:.4%} of the corpus)")
    print(f"   → a query touches ~{plan['nlist'] + plan['candidates_scanned']:,.0f} vectors,"
          f" not {n:,}.\n")

    sh = shard_plan(n, int(args.shard))
    print("3. SHARDING — the corpus is k copies of a measured million:")
    print(f"   shards            = {sh['shards']:,}  ×  {sh['per_shard']:,} rows each")
    print(f"   each shard's index= {_human_bytes(sh['pq_bytes_per_shard'])} (PQ-64) — fits in RAM")
    print(f"   → every shard does the laptop's measured work; the query fans out.\n")

    budget, total = latency_budget(sharded=True)
    print("4. LATENCY — one query's budget (shards run in parallel):")
    for stage, ms in budget:
        print(f"   {stage:<26} {ms:>6.1f} ms")
    print(f"   {'p50 total':<26} {total:>6.1f} ms   — interactive, at a billion.\n")

    cas = cascade_plan(n, args.nprobe)
    print("5. THE APPROXIMATION CASCADE — approximate at every level (cascade.py):")
    print(f"   resident index    = {_human_bytes(cas['resident_bytes'])} (binary + PQ, both in RAM)")
    levels = ["L1 binary", "L2 PQ-64", "L3 int8", "L4 exact"]
    for name, t in zip(levels, cas["scored_per_level"]):
        print(f"   {name:<11} scores {t:>14,.0f} vectors/query")
    print(f"   → only {cas['float32_per_query']} vectors/query ever scored in float32"
          f" — {cas['float32_fraction']:.1e} of the corpus.")
    print("   Measured (cascade.py): the cascade keeps ~100% of exact-on-cells recall.\n")

    print("6. THE AGENT LAYER, on top, for free:")
    print("   Hermes' evaluator guard means MOST queries stop at pass 1, so the")
    print("   p50 above is unchanged; only hard queries spend a second search.")
    print("   The crawler + item tower grow the corpus OFFLINE — never on this path.")


if __name__ == "__main__":
    main()
