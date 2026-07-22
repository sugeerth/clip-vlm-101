"""The corpus, live at localhost — a whole vector database in one file.

pipeline: text or photo ──► [serve] ──► embed ──► cascade / ivf / exact ──► grid

The scale story you can click: the compressed indexes (binary + OPQ + int8) are
promoted to RAM — about ⅛ the size of the f32 towers — so the CASCADE engine
searches a million-plus in a fraction of the memory, reading full precision for
only ~100 vectors per query. The exact and ivf engines (which need the towers)
are there too, so you can watch the same query cost 3 different amounts. Every
answer prints its own latency, engine and how much of the corpus it touched.
"""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import engine
import ivf as ivfmod
from scan import ensemble_query, gpu_promote, scan_gpu, scan_ram
from store import IMG, TXT, connect, fetch, n_rows

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__ROWS__ dishes, live</title><style>
:root{--page:#0b0b0d;--surface:#16161a;--card:#1b1b20;--ink:#f4f4f2;--ink2:#b9b8b2;
--muted:#83827c;--hair:#2a2a30;--accent:#5b9dff;--accent2:#8fbcff;--glow:rgba(91,157,255,.15)}
@media(prefers-color-scheme:light){:root{--page:#f7f7f4;--surface:#fff;--card:#fff;
--ink:#111;--ink2:#4b4a47;--muted:#8a8983;--hair:#e6e5df;--accent:#2f6fe0;--accent2:#1f57bd;
--glow:rgba(47,111,224,.10)}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
-webkit-font-smoothing:antialiased}
main{max-width:1040px;margin:0 auto;padding:44px 22px 80px}
.head{display:flex;flex-direction:column;gap:4px;margin-bottom:22px}
h1{font-size:1.9rem;font-weight:700;letter-spacing:-.025em;margin:0}
h1 em{font-style:normal;background:linear-gradient(90deg,var(--accent),var(--accent2));
-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--muted);font-size:.86rem}
.pill{display:flex;gap:6px;align-items:center;background:var(--surface);
border:1px solid var(--hair);border-radius:16px;padding:9px 9px 9px 18px;
box-shadow:0 1px 0 rgba(0,0,0,.04),0 8px 30px var(--glow)}
.pill input[type=search]{flex:1;min-width:0;border:none;background:none;color:var(--ink);
font:inherit;font-size:1.08rem;outline:none}
.pill input::placeholder{color:var(--muted)}
.iconbtn{border:none;background:var(--page);color:var(--ink2);font-size:1.1rem;cursor:pointer;
width:38px;height:38px;border-radius:11px;display:grid;place-items:center;transition:.15s}
.iconbtn:hover{color:var(--accent);background:var(--glow)}
body.dragging .pill{border-color:var(--accent);box-shadow:0 0 0 4px var(--glow)}
.ctl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:14px 3px 0;
font-size:.82rem;color:var(--ink2)}
.seg{display:inline-flex;background:var(--surface);border:1px solid var(--hair);
border-radius:11px;padding:3px;gap:2px}
.seg button{border:none;background:none;color:var(--ink2);font:inherit;font-size:.8rem;
cursor:pointer;padding:5px 12px;border-radius:8px;transition:.15s}
.seg button[aria-pressed=true]{background:var(--accent);color:#fff}
.ctl label{display:flex;gap:7px;align-items:center;cursor:pointer}
.ctl input[type=range]{accent-color:var(--accent);width:120px}
.ctl select{font:inherit;font-size:.8rem;color:var(--ink2);background:var(--surface);
border:1px solid var(--hair);border-radius:9px;padding:4px 8px}
.lat{margin:20px 3px 4px;font-size:.86rem;color:var(--muted);min-height:1.4em}
.lat b{color:var(--accent2);font-variant-numeric:tabular-nums}
.lat .badge{display:inline-block;font-size:.7rem;font-weight:600;letter-spacing:.03em;
text-transform:uppercase;color:var(--accent);background:var(--glow);
border-radius:6px;padding:2px 7px;margin-right:7px;vertical-align:1px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
gap:16px;margin-top:14px}
figure{margin:0;background:var(--card);border:1px solid var(--hair);border-radius:16px;
overflow:hidden;opacity:0;transform:translateY(10px);animation:rise .35s cubic-bezier(.2,.7,.2,1) forwards;
transition:transform .18s,box-shadow .18s;cursor:zoom-in}
figure:hover{transform:translateY(-3px);box-shadow:0 12px 34px rgba(0,0,0,.22)}
@keyframes rise{to{opacity:1;transform:none}}
figure img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:var(--hair)}
figcaption{padding:9px 12px 11px}
figcaption .s{color:var(--accent2);font-variant-numeric:tabular-nums;font-weight:600;font-size:.78rem}
figcaption .nm{font-size:.82rem;color:var(--ink);margin-top:2px;line-height:1.35;
display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
figcaption .cf{font-size:.72rem;color:var(--muted);margin-top:3px}
.hidden{display:none!important}
/* full-resolution lightbox */
#lb{position:fixed;inset:0;background:rgba(0,0,0,.86);display:none;place-items:center;
z-index:50;padding:28px;cursor:zoom-out;backdrop-filter:blur(6px)}
#lb.on{display:grid}
#lb img{max-width:min(94vw,1100px);max-height:82vh;border-radius:14px;
box-shadow:0 30px 90px rgba(0,0,0,.6)}
#lb .cap{position:fixed;bottom:26px;left:0;right:0;text-align:center;color:#eee;
font-size:.9rem;text-shadow:0 1px 6px #000}
#lb .cap b{color:var(--accent2)}
</style></head><body><main>
<div class="head">
  <h1><em>__ROWS__</em> dishes, live</h1>
  <p class="sub">a whole vector database in one file · compressed indexes in RAM ·
     every answer prints what it cost</p>
</div>
<div class="pill"><input id="q" type="search" autofocus
  placeholder="ramen with a soft-boiled egg… — or drop a photo anywhere">
<button id="cam" class="iconbtn" title="reverse image search: pick a photo">◎</button>
<input id="file" type="file" accept="image/*" class="hidden"></div>
<div class="ctl">
  <div class="seg" id="engine" role="group" aria-label="engine">
    <button data-e="cascade" aria-pressed="true" title="binary → OPQ → int8 → exact, low memory">cascade</button>
    <button data-e="ivf" aria-pressed="false" title="inverted-file index: scan a few cells">ivf</button>
    <button data-e="exact" aria-pressed="false" title="scan every vector, the truth">exact</button>
  </div>
  <label id="probesWrap">probes <input type="range" id="probes" min="1" max="32" value="8">
    <b id="pv">8</b></label>
  <label>mode <select id="mode"><option value="fused" selected>fused</option>
    <option value="image">image</option><option value="text">text</option></select></label>
</div>
<div class="lat" id="lat"></div><div class="grid" id="grid"></div></main>
<div id="lb"><img id="lbimg" alt=""><div class="cap" id="lbcap"></div></div>
<script>
const $=id=>document.getElementById(id);let deb=0,seq=0,eng='cascade';
function fmtEngine(e){return e.replace('exact on','GPU')}
function render(r,label){
$('lat').innerHTML=`<span class="badge">${fmtEngine(r.engine)}</span>`+
`<b>${r.ms} ms</b> across ${r.rows.toLocaleString()} rows`+
(r.scanned<r.rows?` — touched ${(100*r.scanned/r.rows).toFixed(1)}%`:'')+
` · ${label} embedded in ${r.embed_ms} ms`;
$('grid').replaceChildren(...r.results.map(x=>{const f=document.createElement('figure');
const i=document.createElement('img');i.src=x.url;i.alt=x.name;i.loading='lazy';
i.onerror=()=>f.classList.add('hidden');
f.onclick=()=>{$('lbimg').src=x.url;$('lbimg').alt=x.name;
$('lbcap').innerHTML=`<b>${x.score>=0?'+':''}${x.score}</b> · ${esc(x.name)}`+(x.cafe?' — '+esc(x.cafe):'');
$('lb').classList.add('on')};
const c=document.createElement('figcaption');
c.innerHTML=`<span class="s">${x.score>=0?'+':''}${x.score}</span>`+
`<div class="nm">${esc(x.name)}</div>`+(x.cafe?`<div class="cf">${esc(x.cafe)}</div>`:'');
f.append(i,c);return f}))}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
async function go(){const q=$('q').value.trim();if(!q)return;const my=++seq;
$('lat').textContent='searching…';
const p=new URLSearchParams({q,mode:$('mode').value,engine:eng,
probes:$('probes').value,k:24});
try{const r=await(await fetch('/api/search?'+p)).json();if(my!==seq)return;
if(r.error){$('lat').textContent=r.error;return}render(r,'query')}
catch(e){$('lat').textContent='search failed: '+e.message}}
async function goImage(file){if(!file||!file.type.startsWith('image/'))return;
const my=++seq;$('lat').textContent='embedding your photo…';
const p=new URLSearchParams({engine:eng,probes:$('probes').value,k:24});
try{const r=await(await fetch('/api/search-image?'+p,{method:'POST',body:file})).json();
if(my!==seq)return;if(r.error){$('lat').textContent=r.error;return}render(r,'photo')}
catch(e){$('lat').textContent='image search failed: '+e.message}}
$('q').addEventListener('input',()=>{clearTimeout(deb);deb=setTimeout(go,250)});
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
$('probes').addEventListener('input',()=>{$('pv').textContent=$('probes').value;go()});
$('engine').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;
[...$('engine').children].forEach(x=>x.setAttribute('aria-pressed',x===b));
eng=b.dataset.e;$('probesWrap').style.opacity=eng==='ivf'?1:.4;go()});
$('mode').addEventListener('change',go);
$('cam').addEventListener('click',()=>$('file').click());
$('file').addEventListener('change',()=>goImage($('file').files[0]));
$('lb').addEventListener('click',()=>$('lb').classList.remove('on'));
addEventListener('keydown',e=>{if(e.key==='Escape')$('lb').classList.remove('on')});
addEventListener('dragover',e=>{e.preventDefault();document.body.classList.add('dragging')});
addEventListener('dragleave',e=>{if(!e.relatedTarget)document.body.classList.remove('dragging')});
addEventListener('drop',e=>{e.preventDefault();document.body.classList.remove('dragging');
goImage(e.dataTransfer.files[0])});
addEventListener('paste',e=>{const f=[...(e.clipboardData?.items||[])]
.find(i=>i.type.startsWith('image/'));if(f)goImage(f.getAsFile())});
$('probesWrap').style.opacity=.4;
</script></body></html>"""


def cmd_serve(args):
    from embedder import ClipEmbedder

    rows = n_rows(connect())
    if rows == 0:
        raise SystemExit("no rows — run `python3 scale.py ingest` first")
    print(f"loading the corpus for {rows:,} rows …")
    t0 = time.time()
    img = np.load(IMG, mmap_mode="r")        # f16 memmap: the cascade's exact tier reads ~100/query
    idx = engine.load_indexes(rows, to_ram=True)   # binary+OPQ+int8 in RAM (≈ ⅛ the f32 towers)
    print(f"  compressed indexes in RAM ({time.time() - t0:,.0f}s) — cascade ready")

    ivf_idx = None                           # ivf/exact need the f32 towers; load them too
    Rimg = np.asarray(img[:rows], dtype=np.float32)
    Rtxt = np.asarray(np.load(TXT, mmap_mode="r")[:rows], dtype=np.float32)
    mats = {"image": [(Rimg, 1.0)], "text": [(Rtxt, 1.0)], "fused": [(Rimg, 0.5), (Rtxt, 0.5)]}
    if not ivfmod.ivf_stale(rows):
        Cv, orderv, startsv, _ = ivfmod.load_ivf()
        ivf_idx = (Cv, orderv, startsv)
    gpu = {}
    try:
        g = gpu_promote([(Rimg, 1.0), (Rtxt, 1.0)])
        if g:
            dev, ((gi, _), (gt, _)) = g
            gpu = {"image": (dev, [(gi, 1.0)]), "text": (dev, [(gt, 1.0)]),
                   "fused": (dev, [(gi, 0.5), (gt, 0.5)])}
            scan_gpu(gpu["fused"], np.zeros(512, np.float32))
            print(f"  towers on {dev} — exact scans run on the GPU")
    except Exception as e:
        print(f"  (no GPU path: {e})")
    print(f"  ready in {time.time() - t0:,.0f}s")

    emb = ClipEmbedder()
    emb.embed_texts(["warm-up"])
    mps = threading.Lock()

    def run_query(qv, mode, eng, probes, k, embed_ms):
        t0 = time.time()
        if eng == "cascade":                 # low-memory, image-tower, accurate
            ids, scores, scanned = engine.cascade_search(qv, img, idx, k=k, rows=rows)
            label = "cascade"
        elif eng == "ivf" and ivf_idx:
            ids, scores, scanned = ivfmod.ivf_search(qv, mats[mode], *ivf_idx, k=k, probes=probes)
            label = "ivf"
        elif gpu:
            ids, scores = scan_gpu(gpu[mode], qv, k=k)
            scanned, label = rows, f"exact on {gpu[mode][0]}"
        else:
            ids, scores = scan_ram(mats[mode], qv, k=k)
            scanned, label = rows, "exact in RAM"
        ms = (time.time() - t0) * 1e3
        con = connect(); recs = fetch(con, ids); con.close()
        return json.dumps({
            "ms": round(ms, 1), "embed_ms": round(embed_ms, 1),
            "scanned": int(scanned), "rows": rows, "engine": label,
            "results": [{"id": int(i), "name": n, "cafe": c, "url": url,
                         "score": round(float(s), 3)}
                        for (i, n, cap, c, url), s in zip(recs, scores)],
        }).encode()

    def knobs(p):
        return (p.get("mode", ["fused"])[0],
                p.get("engine", ["cascade"])[0],
                max(1, min(64, int(p.get("probes", ["8"])[0]))),
                max(1, min(50, int(p.get("k", ["24"])[0]))))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self.send(200, PAGE.replace("__ROWS__", f"{rows:,}").encode(),
                                 "text/html; charset=utf-8")
            if u.path != "/api/search":
                return self.send(404, b"{}", "application/json")
            p = parse_qs(u.query)
            query = p.get("q", [""])[0].strip()
            if not query:
                return self.send(400, b'{"error":"empty query"}', "application/json")
            mode, eng, probes, k = knobs(p)
            if mode not in mats:
                return self.send(400, b'{"error":"bad mode"}', "application/json")
            t0 = time.time()
            with mps:
                qv = ensemble_query(emb.embed_texts, query)
            embed_ms = (time.time() - t0) * 1e3
            return self.send(200, run_query(qv, mode, eng, probes, k, embed_ms),
                             "application/json")

        def do_POST(self):
            from PIL import Image
            u = urlparse(self.path)
            if u.path != "/api/search-image":
                return self.send(404, b"{}", "application/json")
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 25_000_000:
                return self.send(400, b'{"error":"bad image size"}', "application/json")
            _, eng, probes, k = knobs(parse_qs(u.query))
            try:
                pil = Image.open(io.BytesIO(self.rfile.read(size))).convert("RGB")
            except Exception:
                return self.send(400, b'{"error":"not an image"}', "application/json")
            t0 = time.time()
            import torch
            with mps:
                inputs = emb.processor(images=[pil], return_tensors="pt").to(emb.device)
                with torch.no_grad():
                    feats = emb.model.get_image_features(**inputs)
                qv = (feats.pooler_output if hasattr(feats, "pooler_output") else feats)
                qv = qv[0].float().cpu().numpy()
                qv /= np.linalg.norm(qv) or 1.0
            embed_ms = (time.time() - t0) * 1e3
            # reverse image search is image-tower; cascade/ivf both score it
            eng = eng if eng in ("cascade", "ivf") else "cascade"
            return self.send(200, run_query(qv.astype(np.float32), "image", eng,
                                            probes, k, embed_ms), "application/json")

    port = args.port
    while True:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    print(f"live: http://localhost:{port}")
    srv.serve_forever()
