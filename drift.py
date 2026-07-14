"""drift.py — is the live stream still the world we calibrated for?

pipeline: reference window ─┐
          live window ──────┴─► [drift] ─► stable / shift / DRIFT, and why

Every guarantee in this repo assumes the future looks like the past. Conformal's
coverage, the council's thresholds, the trust composer — all hold only while
live queries stay EXCHANGEABLE with the gallery they were tuned on. Production
breaks that quietly: the data shifts under you and the honest-looking numbers
keep printing. This is the monitor that watches for it, on a stream of results.

Pick a signal to watch (here: a per-query QUALITY score — the top match's
similarity, or the composed trust score). Freeze a REFERENCE window from
calibration time; compare each LIVE window against it with three
distribution-free detectors:

  PSI   population stability index  Σ (l−r)·ln(l/r) over reference-quantile bins.
        The industry default: <0.10 stable · 0.10–0.25 shift · >0.25 drift.
  KS    Kolmogorov–Smirnov: the largest gap between the two CDFs. Assumes
        nothing about the distribution — the same spirit as conformal.
  COVERAGE  the repo-native detector: calibrate a conformal threshold on the
        reference, then measure coverage on the live window. Coverage falling
        below target IS exchangeability breaking — conformal detects its own
        drift for free.

Then it sorts the live window into POSITIVE cases (cleared the bar — confident)
and FAILURE cases (fell short — the ones to look at), and reports the failure
rate. A rising failure rate is drift you can act on, with examples attached.

    python3 drift.py --json docs/db.json           # simulate a stream, watch it drift
    python3 drift.py --json docs/db.json --selftest # deterministic, model-free (CI)
"""
import numpy as np

PSI_SHIFT, PSI_DRIFT = 0.10, 0.25        # standard PSI bands
KS_ALPHA = 0.05                          # KS significance for the critical value
COV_SLACK = 0.10                         # coverage may dip this far below target before it's drift


def psi(ref, live, bins=8):
    """Population Stability Index over quantile bins fixed from the reference.
    Symmetric, always ≥ 0; bigger = more shift. Bins that collapse on ties are
    merged, so it stays sane on tiny galleries."""
    ref = np.asarray(ref, dtype=np.float64)
    live = np.asarray(live, dtype=np.float64)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                                   # a constant reference → no bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    l = np.histogram(live, edges)[0] / len(live)
    eps = 1e-6
    r = np.clip(r, eps, None)
    l = np.clip(l, eps, None)
    return float(np.sum((l - r) * np.log(l / r)))


def ks_stat(ref, live):
    """Two-sample Kolmogorov–Smirnov statistic: max |CDF_ref − CDF_live|."""
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    live = np.sort(np.asarray(live, dtype=np.float64))
    grid = np.concatenate([ref, live])
    c_ref = np.searchsorted(ref, grid, side="right") / len(ref)
    c_live = np.searchsorted(live, grid, side="right") / len(live)
    return float(np.max(np.abs(c_ref - c_live)))


def ks_critical(n, m, alpha=KS_ALPHA):
    """The KS rejection threshold at `alpha` for sample sizes n, m."""
    c = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}.get(alpha, 1.36)
    return float(c * np.sqrt((n + m) / (n * m)))


def coverage(ref_scores, live_scores, alpha, higher_is_better=True):
    """Calibrate a conformal bar on the reference, then the fraction of the live
    window that clears it. `alpha` is the miscoverage; target coverage is 1−α.
    (score = quality, so nonconformity = 1 − score.)"""
    import conformal
    ref = np.asarray(ref_scores, dtype=np.float64)
    live = np.asarray(live_scores, dtype=np.float64)
    qhat = conformal.calibrate(1.0 - ref if higher_is_better else ref, alpha)
    bar = (1.0 - qhat) if higher_is_better else qhat
    if not np.isfinite(qhat):
        return 1.0, float("-inf") if higher_is_better else float("inf")
    covered = live >= bar if higher_is_better else live <= bar
    return float(np.mean(covered)), float(bar)


def monitor(ref, live, alpha=0.2):
    """Compare a live window to the reference on all three detectors and rule:
    stable / shift / drift, with the reasons that fired and the failure rate."""
    p = psi(ref, live)
    k = ks_stat(ref, live)
    k_crit = ks_critical(len(ref), len(live))
    cov, bar = coverage(ref, live, alpha)
    target = 1.0 - alpha

    reasons = []
    if p > PSI_DRIFT:
        reasons.append(f"PSI {p:.2f} > {PSI_DRIFT} (population shifted)")
    if k > k_crit:
        reasons.append(f"KS {k:.2f} > {k_crit:.2f} (distributions differ)")
    if cov < target - COV_SLACK:
        reasons.append(f"coverage {cov:.0%} < target {target:.0%} (exchangeability broke)")

    if reasons:
        level = "drift"
    elif p > PSI_SHIFT or k > k_crit * 0.75:
        level = "shift"
    else:
        level = "stable"
    return {"level": level, "reasons": reasons, "psi": p, "ks": k,
            "ks_critical": k_crit, "coverage": cov, "target": target,
            "bar": bar, "failure_rate": 1.0 - cov, "n_ref": len(ref), "n_live": len(live)}


def classify(items, scores, bar):
    """Split a live window into positive (cleared the bar) and failure cases —
    the failures are what a human should actually look at."""
    positive, failure = [], []
    for it, s in zip(items, scores):
        (positive if s >= bar else failure).append({"item": it, "score": float(s)})
    return positive, failure


# ── model-free signal: each gallery image is a query, the signal is its best
#    same-tag match similarity (retrieval quality). A stream is windows of these. ─

def quality_signal(items, key="image_emb"):
    """The retrieval-quality population: every same-tag pair's similarity. Each
    (query, relevant) match contributes one sample, so a 14-image gallery yields
    dozens — enough for PSI/KS to mean something. Model-free, from stored vectors."""
    import conformal
    sig = []
    for i, q in enumerate(items):
        cos = conformal.cosines(q[key], items, key)
        for j, it in enumerate(items):
            if j != i and set(it["tags"]) & set(q["tags"]):
                sig.append(float(cos[j]))
    return np.array(sig, dtype=np.float64)


def item_quality(items, key="image_emb"):
    """Per-item health: each image's best same-tag match similarity — one number
    per query, for sorting the live window into positive vs failure CASES (the
    pooled quality_signal is for the detectors; this is for the case list)."""
    import conformal
    out = []
    for i, q in enumerate(items):
        cos = conformal.cosines(q[key], items, key)
        cos[i] = -np.inf
        rel = [j for j, it in enumerate(items)
               if j != i and set(it["tags"]) & set(q["tags"])]
        out.append(max((cos[j] for j in rel), default=0.0))
    return np.array(out, dtype=np.float64)


def drift_window(sig, frac, drop=0.2):
    """Deterministically CONTAMINATE a window: a `frac` fraction of the queries go
    off-distribution and score `drop` lower (clipped). That's how drift really
    arrives — not everything decays at once, a growing slice does — and it gives a
    graduated alarm. The degraded slice is evenly spaced, so no randomness: the
    stream is reproducible in CI and byte-identical across the twin."""
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    k = int(np.floor(frac * n + 0.5))        # explicit half-up (not Python's
                                             # banker's round) so the twin agrees
    live = sig.copy()
    for i in range(k):
        live[(i * n) // k] = max(0.0, live[(i * n) // k] - drop)
    return live


def _stream():
    """The simulated stream: window 0 is the reference, then a growing fraction
    of queries go off-distribution."""
    return [("t0 · baseline", 0.00), ("t1 · 15% off", 0.15),
            ("t2 · 35% off", 0.35), ("t3 · 60% off", 0.60)]


def render_html(items, ref, alpha):
    """A self-contained, theme-aware drift dashboard: the stream verdicts, the
    reference-vs-drifted histograms, and the positive/failure case split."""
    def bars(sig, edges, color):
        counts = np.histogram(sig, edges)[0]
        top = max(counts.max(), 1)
        w = 100 / len(counts)
        rects = "".join(
            f'<rect x="{i*w:.2f}%" y="{100-100*c/top:.1f}%" width="{w*0.86:.2f}%" '
            f'height="{100*c/top:.1f}%" fill="{color}"></rect>'
            for i, c in enumerate(counts))
        return f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="hist">{rects}</svg>'

    edges = np.linspace(min(ref), max(ref), 13)
    worst = drift_window(ref, _stream()[-1][1])
    rows = ""
    for name, frac in _stream():
        m = monitor(ref, drift_window(ref, frac), alpha)
        cls = m["level"]
        rows += (f'<tr class="{cls}"><td>{name}</td><td>{m["psi"]:.2f}</td>'
                 f'<td>{m["ks"]:.2f}</td><td>{m["coverage"]:.0%}</td>'
                 f'<td>{m["failure_rate"]:.0%}</td><td class="lv">{cls.upper()}</td></tr>')
    _, bar = coverage(ref, worst, alpha)
    pos, fail = classify(items, item_quality(items), bar)
    fails = "".join(
        f'<li><b>{f["score"]:.3f}</b> {f["item"]["path"].split("/")[-1]} '
        f'<span>({", ".join(f["item"]["tags"][:3])})</span></li>'
        for f in sorted(fail, key=lambda x: x["score"])[:6])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>drift monitor</title><style>
:root{{--bg:#f9f9f7;--fg:#0b0b0b;--mut:#6b6a66;--line:#e1e0d9;--card:#fff;--ok:#2a8a4a;--warn:#b06a1a;--bad:#c0392b;--ref:#2a78d6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--fg:#fff;--mut:#9a988e;--line:#2c2c2a;--card:#1a1a19;--ref:#3987e5}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,system-ui,sans-serif;padding:32px}}
.wrap{{max-width:760px;margin:0 auto}}h1{{font-size:1.5rem;margin:0 0 4px}}.sub{{color:var(--mut);margin:0 0 24px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--mut);font-weight:600;font-size:.82rem}}
.lv{{font-weight:700;font-size:.8rem}}tr.stable .lv{{color:var(--ok)}}tr.shift .lv{{color:var(--warn)}}tr.drift .lv{{color:var(--bad)}}
.hist{{width:100%;height:70px;background:transparent;border-bottom:1px solid var(--line)}}
.h2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.h2 .lab{{font-size:.8rem;color:var(--mut);margin-top:6px}}
ul{{list-style:none;padding:0;margin:8px 0 0}}li{{padding:4px 0;font-variant-numeric:tabular-nums}}li span{{color:var(--mut)}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600}}
.pill.ok{{color:var(--ok);border:1px solid var(--ok)}}.pill.bad{{color:var(--bad);border:1px solid var(--bad)}}
</style></head><body><div class="wrap">
<h1>🌊 drift monitor</h1>
<p class="sub">is the live stream still the world we calibrated for? {len(items)} images ·
signal: same-tag match similarity · conformal target {1-alpha:.0%}</p>
<div class="card"><table>
<tr><th>window</th><th>PSI</th><th>KS</th><th>coverage</th><th>failure</th><th>status</th></tr>
{rows}</table>
<p class="sub" style="margin:12px 0 0">PSI &lt;0.10 stable · 0.10–0.25 shift · &gt;0.25 drift —
the industry bands; coverage below target means exchangeability broke.</p></div>
<div class="card"><div class="h2">
<div>{bars(ref, edges, 'var(--ref)')}<div class="lab">reference (calibration)</div></div>
<div>{bars(worst, edges, 'var(--bad)')}<div class="lab">live · worst window (drifted)</div></div>
</div></div>
<div class="card"><b>cases at the calibrated bar {bar:.3f}</b> &nbsp;
<span class="pill ok">{len(pos)} positive</span> <span class="pill bad">{len(fail)} failure</span>
<ul>{fails}</ul></div>
</div></body></html>"""


if __name__ == "__main__":
    import argparse
    import json

    import db

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/db.json")
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--selftest", action="store_true",
                    help="assert the detectors fire in the right order, model-free")
    ap.add_argument("--html", metavar="FILE", help="render the drift dashboard to HTML")
    ap.add_argument("--gate", action="store_true",
                    help="compare the CURRENT gallery to the frozen reference; exit 1 on drift")
    ap.add_argument("--save-reference", metavar="FILE",
                    help="freeze the current quality signal as the reference baseline")
    ap.add_argument("--reference", default="drift_reference.json",
                    help="the frozen reference baseline the gate compares against")
    args = ap.parse_args()

    items = (db.load_json_gallery(args.json) if args.json
             else db.all_images(db.connect(args.db)))
    ref = quality_signal(items)

    if args.save_reference:
        with open(args.save_reference, "w") as fh:
            json.dump({"signal": ref.tolist(), "alpha": args.alpha}, fh, indent=1)
        print(f"froze {len(ref)} reference samples → {args.save_reference}")
        raise SystemExit(0)

    if args.html:
        with open(args.html, "w") as fh:
            fh.write(render_html(items, ref, args.alpha))
        print(f"drift dashboard → {args.html}")
        raise SystemExit(0)

    if args.gate:
        # compare the CURRENT gallery (live) to the committed calibration reference.
        try:
            with open(args.reference) as fh:
                base = np.array(json.load(fh)["signal"], dtype=np.float64)
        except FileNotFoundError:
            with open(args.reference, "w") as fh:
                json.dump({"signal": ref.tolist(), "alpha": args.alpha}, fh, indent=1)
            print(f"no reference yet — froze the current gallery → {args.reference} (pass)")
            raise SystemExit(0)
        m = monitor(base, ref, args.alpha)
        print(f"gate: current gallery vs reference — {m['level'].upper()}  "
              f"(PSI {m['psi']:.2f}, KS {m['ks']:.2f}, coverage {m['coverage']:.0%})")
        for r in m["reasons"]:
            print(f"  └─ {r}")
        raise SystemExit(1 if m["level"] == "drift" else 0)

    stream = _stream()

    if args.selftest:
        levels = [monitor(ref, drift_window(ref, f), args.alpha)["level"] for _, f in stream]
        assert levels[0] == "stable", f"baseline should be stable, got {levels[0]}"
        assert levels[-1] == "drift", f"heavy contamination should drift, got {levels[-1]}"
        # PSI is monotone in the contamination fraction — more off-distribution, more PSI
        psis = [psi(ref, drift_window(ref, f)) for _, f in stream]
        assert psis == sorted(psis), f"PSI not monotone: {psis}"
        print(f"drift selftest passed  (levels: {' → '.join(levels)})")
        raise SystemExit(0)

    print(f"watching {len(items)} live queries against the calibration reference "
          f"(signal: best same-tag similarity):\n")
    print(f"  {'window':<16} {'PSI':>6} {'KS':>6} {'cov':>6} {'fail':>6}  status")
    for name, frac in stream:
        live = drift_window(ref, frac)
        m = monitor(ref, live, args.alpha)
        flag = {"stable": "· stable", "shift": "~ shift",
                "drift": "⚠ DRIFT"}[m["level"]]
        print(f"  {name:<16} {m['psi']:>6.2f} {m['ks']:>6.2f} "
              f"{m['coverage']:>6.0%} {m['failure_rate']:>6.0%}  {flag}")
        for r in m["reasons"]:
            print(f"       └─ {r}")

    # sort the images into positive vs failure cases against the calibrated bar —
    # the failures are what a human should actually look at.
    _, bar = coverage(ref, drift_window(ref, stream[-1][1]), args.alpha)
    pos, fail = classify(items, item_quality(items), bar)
    print(f"\ncases at the calibrated bar {bar:.3f}: {len(pos)} positive · "
          f"{len(fail)} failure. failures to inspect:")
    for f in sorted(fail, key=lambda x: x["score"])[:5]:
        tags = ", ".join(f["item"]["tags"][:3])
        print(f"  {f['score']:.3f}  {f['item']['path'].split('/')[-1]:<24} ({tags})")
    print("\nsame distribution-free spirit as conformal: no assumption about the "
          "data,\njust a promise you can check — and an alarm when it stops holding.")
