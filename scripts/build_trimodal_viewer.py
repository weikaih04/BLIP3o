"""Build the tri-modal comparison VIEWER (self-contained HTML) for the research hub.

One page, one asset at a time: input views / I1 / IM / T / GT side by side, with the
per-modality SS-IoU and the caption that was actually fed. A single grid image forces
you to squint at 14 rows at once; this lets you look at one asset properly and step
through them.

Stills are resized + JPEG'd + base64-embedded so the page is useful the instant it
loads with no extra requests. The I1/IM/T columns can additionally flip to the *actual
generated mesh*, orbit-able and camera-linked, served as separate compressed .glb files
(see scripts/compress_trimodal_glbs.py) that are fetched only for the asset on screen.
model-viewer is vendored next to them, not pulled from a CDN, and is only injected the
first time somebody asks for 3D.
"""
import argparse, base64, glob, io, json, os, shutil
from PIL import Image

SRC = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal/_parts"
SUMMARY = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal/summary.json"
GLB_SRC = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal/web"
GLB_REPORT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal/web_compression_report.json"
MODEL_VIEWER = ("/fsx/home/weikai.huang/research-proj-report/reports/3d-generation-editing/"
                "assets-glb-viewer/model-viewer.min.js")
OUT = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/runs/cache_logs/trimodal/trimodal-viewer.html"
ASSET_DIR_NAME = "assets-trimodal-3d"      # sibling folder of the html, on the hub
W_MAIN, W_THUMB, Q = 420, 96, 80


def b64(path, width, quality=Q):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


ap = argparse.ArgumentParser()
ap.add_argument("--publish-assets", default="",
                help="directory to copy the .glb files + model-viewer.min.js into")
args = ap.parse_args()

summary = json.load(open(SUMMARY))
glb_stats = {}
if os.path.isfile(GLB_REPORT):
    glb_stats = {r["file"]: r for r in json.load(open(GLB_REPORT))["files"]}

assets = []
for jp in sorted(glob.glob(f"{SRC}/*.json")):
    d = json.load(open(jp))
    s8 = d["sha8"]
    imgs = {k: f"{SRC}/{s8}_{k}.png" for k in ("input", "i1", "im", "t", "gt")}
    if not all(os.path.isfile(p) for p in imgs.values()):
        continue
    modes = d.get("modes", {})
    glb, mesh = {}, {}
    for k in ("i1", "im", "t"):
        f = f"{s8}_{k}.glb"
        if os.path.isfile(f"{GLB_SRC}/{f}"):
            glb[k] = f"{ASSET_DIR_NAME}/{f}"
            r = glb_stats.get(f, {})
            mesh[k] = {"kb": round(os.path.getsize(f"{GLB_SRC}/{f}") / 1024),
                       "tris": r.get("out_faces"), "iou": r.get("silhouette_iou")}
    assets.append({
        "sha8": s8, "sha": d.get("sha", ""), "caption": d.get("caption", ""),
        "view_i1": d.get("view_i1"), "views_im": (modes.get("im") or {}).get("views"),
        "iou": {k: (modes.get(k) or {}).get("iou") for k in ("i1", "im", "t")},
        "img": {k: b64(p, W_MAIN) for k, p in imgs.items()},
        "thumb": b64(imgs["gt"], W_THUMB, 70),
        "glb": glb, "mesh": mesh,
    })
# worst regressions first — the interesting cases are where multi-view LOSES to one image
assets.sort(key=lambda a: (a["iou"]["im"] or 0) - (a["iou"]["i1"] or 0))

MEANS = {k: summary.get(f"mean_iou_{k}") for k in ("i1", "im", "t")}
LEAK = "edade2fc"          # memorised duplicate — excluded from the honest means
clean = [a for a in assets if a["sha8"] != LEAK]
MEANS_CLEAN = {k: sum(a["iou"][k] for a in clean if a["iou"][k] is not None) / len(clean)
               for k in ("i1", "im", "t")}

n_glb = sum(len(a["glb"]) for a in assets)
glb_bytes = sum(os.path.getsize(f"{GLB_SRC}/{os.path.basename(p)}")
                for a in assets for p in a["glb"].values())
MESH_NOTE = (f"{n_glb} meshes, {glb_bytes/1e6:.0f} MB total, "
             f"{glb_bytes/max(n_glb,1)/1e6:.2f} MB each on average")

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tri-modal comparison viewer — one model, three conditionings</title>
<style>
:root{
  --ground:#f6f6f4; --panel:#fff; --edge:#e3e3df; --ink:#15171c; --dim:#666c7a;
  --accent:#0f7d92; --win:#1f8a5b; --loss:#b4442f; --render-bg:#101216;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0d0f13; --panel:#15181e; --edge:#252932; --ink:#e6e8ec; --dim:#8b93a3;
  --accent:#4fc3d9; --win:#48c98a; --loss:#e2765d; --render-bg:#0a0b0e;}}
:root[data-theme="dark"]{
  --ground:#0d0f13; --panel:#15181e; --edge:#252932; --ink:#e6e8ec; --dim:#8b93a3;
  --accent:#4fc3d9; --win:#48c98a; --loss:#e2765d; --render-bg:#0a0b0e;}
:root[data-theme="light"]{
  --ground:#f6f6f4; --panel:#fff; --edge:#e3e3df; --ink:#15171c; --dim:#666c7a;
  --accent:#0f7d92; --win:#1f8a5b; --loss:#b4442f; --render-bg:#101216;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:1.5rem;font-weight:640;letter-spacing:-.02em;margin:0 0 4px;text-wrap:balance}
.sub{color:var(--dim);font-size:.93rem;margin:0 0 22px;max-width:66ch}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
       margin-bottom:22px}
.stat{background:var(--panel);border:1px solid var(--edge);border-radius:8px;padding:11px 13px}
.stat .k{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--dim)}
.stat .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.3rem;
         font-weight:600;margin-top:2px}
.stat .n{font-size:.74rem;color:var(--dim);font-family:var(--mono)}
.main{display:grid;grid-template-columns:210px 1fr;gap:18px;align-items:start}
@media(max-width:820px){.main{grid-template-columns:1fr}}
.rail{background:var(--panel);border:1px solid var(--edge);border-radius:8px;overflow:hidden}
.rail .hd{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
          padding:9px 12px;border-bottom:1px solid var(--edge)}
.item{display:grid;grid-template-columns:38px 1fr;gap:9px;align-items:center;width:100%;
      text-align:left;background:none;border:0;border-bottom:1px solid var(--edge);
      padding:7px 10px;cursor:pointer;color:inherit;font:inherit}
.item:hover{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.item[aria-current="true"]{background:color-mix(in srgb,var(--accent) 15%,transparent);
      box-shadow:inset 3px 0 0 var(--accent)}
.item:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.item img{width:38px;height:38px;object-fit:cover;border-radius:4px;background:var(--render-bg)}
.item .id{font-family:var(--mono);font-size:.76rem}
.item .d{font-family:var(--mono);font-size:.72rem;font-variant-numeric:tabular-nums}
.up{color:var(--win)} .dn{color:var(--loss)}
.stage{background:var(--panel);border:1px solid var(--edge);border-radius:8px;padding:16px}
.bar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:0 0 12px;
     font-size:.78rem;color:var(--dim)}
.seg{display:inline-flex;border:1px solid var(--edge);border-radius:7px;overflow:hidden}
.seg button{font:inherit;font-size:.78rem;padding:4px 11px;background:none;border:0;
     color:var(--dim);cursor:pointer;border-right:1px solid var(--edge)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:color-mix(in srgb,var(--accent) 16%,transparent);
     color:var(--ink);font-weight:600}
.bar label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
.bar input{accent-color:var(--accent);margin:0}
.bar .hint{margin-left:auto;font-family:var(--mono);font-size:.72rem}
.cols{display:grid;grid-template-columns:.85fr 1fr 1fr 1fr .85fr;gap:10px}
@media(max-width:980px){.cols{grid-template-columns:repeat(2,1fr)}}
.cell figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:6px;
     font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
     margin:0 0 5px}
.cell .iou{font-family:var(--mono);font-size:.8rem;font-variant-numeric:tabular-nums;
     letter-spacing:0;text-transform:none;color:var(--ink)}
.best{color:var(--win);font-weight:600}
.cell figure{margin:0}
.view{position:relative;width:100%;aspect-ratio:1;border-radius:6px;overflow:hidden;
     background:var(--render-bg);border:1px solid var(--edge)}
.view img{width:100%;height:100%;object-fit:cover;display:block}
.view model-viewer{width:100%;height:100%;display:block;background-color:var(--render-bg);
     --progress-bar-color:var(--accent);--progress-mask:transparent}
.flip{position:absolute;right:6px;bottom:6px;z-index:2;font:inherit;font-size:.66rem;
     letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:5px;
     cursor:pointer;color:#e9edf2;background:rgba(12,14,18,.72);
     border:1px solid rgba(255,255,255,.22);backdrop-filter:blur(3px)}
.flip:hover{background:rgba(12,14,18,.9)}
.flip:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.flip[aria-pressed="true"]{background:color-mix(in srgb,var(--accent) 78%,#0b0d10);
     border-color:transparent;color:#fff}
.sig{margin-top:4px;font-family:var(--mono);font-size:.68rem;color:var(--dim);
     display:flex;justify-content:space-between;gap:6px;min-height:1.05em}
.sig .bad{color:var(--loss)}
.cap{margin-top:14px;padding:11px 13px;border-radius:7px;border:1px solid var(--edge);
     background:color-mix(in srgb,var(--accent) 5%,transparent);font-size:.88rem}
.cap b{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
     display:block;margin-bottom:3px;font-weight:600}
.meta{margin-top:10px;font-family:var(--mono);font-size:.75rem;color:var(--dim);
     display:flex;gap:16px;flex-wrap:wrap}
.note{margin-top:22px;border-left:3px solid var(--accent);padding:2px 0 2px 13px;
     font-size:.88rem;color:var(--dim);max-width:74ch}
.note b{color:var(--ink)}
kbd{font-family:var(--mono);font-size:.72rem;border:1px solid var(--edge);border-radius:4px;
    padding:1px 5px;background:var(--ground)}
</style>
</head>
<body>
<div class="wrap">
<!-- hub.py rewrites this anchor in place; emitting it here keeps it inside .wrap
     instead of being injected flush against the top-left corner of <body>. -->
<a class="back-link" href="/projects/3d-generation-editing.html" style="display:inline-block;margin:0 0 16px;padding:6px 14px;background:#f4f4f4;border:1px solid #e2e2e2;border-radius:6px;color:#444;text-decoration:none;font-size:14px">← Back to project</a>
<h1>One model, three conditionings</h1>
<p class="sub">The same S3 checkpoint generating the same held-out object from a single image,
from four views, and from a text caption alone — full cascade each time (SS → shape → texture),
no ground-truth structure, identical seed. Sorted by how much multi-view loses to single-image.</p>

<div class="strip" id="strip"></div>

<div class="main">
  <nav class="rail" aria-label="assets">
    <div class="hd">14 held-out assets</div>
    <div id="list"></div>
  </nav>
  <section class="stage">
    <div class="bar">
      <div class="seg" role="group" aria-label="what the I1 / IM / T columns show">
        <button type="button" id="allStill" aria-pressed="true">Renders</button>
        <button type="button" id="all3d" aria-pressed="false">3D meshes</button>
      </div>
      <label><input type="checkbox" id="link" checked> link cameras</label>
      <label><input type="checkbox" id="spin"> auto-rotate</label>
      <span class="hint">drag orbit · scroll zoom · <kbd>3</kbd> toggle</span>
    </div>
    <div class="cols" id="cols"></div>
    <div class="cap"><b>caption fed to the text modality</b><span id="cap"></span></div>
    <div class="meta" id="meta"></div>
  </section>
</div>

<p class="note"><b>Reading the numbers.</b> SS-IoU is occupancy overlap with ground truth at 64³.
Text has no image to anchor scale or pose, so its IoU is low by construction — judge that column
on whether the object is the right thing, not on the number. <b>edade2fc is memorised</b>: IoU is
exactly 1.000 from all three modalities including a 57-token text-only cond, and the render is
pixel-identical to ground truth, so a content-duplicate of it sits in the training corpus under a
different hash. It is excluded from the "clean" means above.
Move with <kbd>↑</kbd><kbd>↓</kbd> or <kbd>J</kbd><kbd>K</kbd>.</p>

<p class="note"><b>About the 3D.</b> The I1/IM/T columns can swap their render for the mesh the
cascade actually produced — drag to orbit, and with <i>link cameras</i> on all three follow each
other so you are comparing the same viewpoint. Meshes load only for the asset on screen
(__MESH_NOTE__), and they are decimated previews: ~25k faces and a 768² baseColor against the
~199k-face, 2048² originals, with the metallic/roughness map folded into scalar factors where it
was constant. Silhouette IoU against the undecimated mesh is printed under each viewer; anything
below 0.95 is flagged. <b>Ground truth has no mesh here</b> — that column is a render of the
reference asset, which is not in this cache, so it stays a render.</p>
</div>
<script>
const A = __ASSETS__, MEANS = __MEANS__, MEANS_CLEAN = __MEANS_CLEAN__;
const MV = "__ASSET_DIR__/model-viewer.min.js";
const f3 = v => v == null ? "—" : v.toFixed(3);
const LABEL = {input:"input views", i1:"I1 · single image", im:"IM · 4 views", t:"T · text", gt:"ground truth"};
const MODES = ["i1", "im", "t"];

document.getElementById("strip").innerHTML = [
  ["I1 single image", MEANS.i1, MEANS_CLEAN.i1],
  ["IM 4 views",      MEANS.im, MEANS_CLEAN.im],
  ["T text caption",  MEANS.t,  MEANS_CLEAN.t],
  ["IM − I1",         MEANS.im - MEANS.i1, MEANS_CLEAN.im - MEANS_CLEAN.i1],
].map(([k, v, c]) => `<div class="stat"><div class="k">${k}</div>
  <div class="v">${v >= 0 ? "" : "−"}${Math.abs(v).toFixed(3)}</div>
  <div class="n">${c >= 0 ? "" : "−"}${Math.abs(c).toFixed(3)} excl. memorised</div></div>`).join("");

const list = document.getElementById("list");
list.innerHTML = A.map((a, i) => {
  const d = (a.iou.im ?? 0) - (a.iou.i1 ?? 0);
  return `<button class="item" data-i="${i}" role="link">
    <img src="${a.thumb}" alt="">
    <span><span class="id">${a.sha8}</span><br>
    <span class="d ${d >= 0 ? "up" : "dn"}">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}</span></span>
  </button>`;
}).join("");

/* ---- model-viewer is ~0.9 MB; don't make the stills-only reader pay for it ---- */
let mvLoading = null;
function ensureModelViewer() {
  if (!mvLoading) {
    mvLoading = new Promise((ok, no) => {
      const s = document.createElement("script");
      s.type = "module"; s.src = MV;
      s.onload = () => customElements.whenDefined("model-viewer").then(ok);
      s.onerror = () => no(new Error("model-viewer failed to load"));
      document.head.appendChild(s);
    });
  }
  return mvLoading;
}

/* ---- per-column state, kept while you step through assets ---- */
const want3d = {i1: false, im: false, t: false};
const linkEl = document.getElementById("link"), spinEl = document.getElementById("spin");
let syncing = false;

function viewers() { return [...document.querySelectorAll(".view model-viewer")]; }

function syncFrom(src) {
  if (!linkEl.checked || syncing) return;
  syncing = true;
  for (const v of viewers()) {
    if (v === src) continue;
    v.cameraOrbit = src.getCameraOrbit().toString();
    v.fieldOfView = src.getFieldOfView() + "deg";
  }
  syncing = false;
}

function mount(slot, key, a) {
  ensureModelViewer().then(() => {
    if (!slot.isConnected || slot.dataset.sha !== a.sha8) return;   // navigated away
    const mv = document.createElement("model-viewer");
    mv.setAttribute("src", a.glb[key]);
    mv.setAttribute("poster", a.img[key]);
    mv.setAttribute("alt", LABEL[key] + " mesh for " + a.sha8);
    mv.setAttribute("camera-controls", "");
    mv.setAttribute("touch-action", "pan-y");
    mv.setAttribute("interaction-prompt", "none");
    mv.setAttribute("environment-image", "neutral");
    mv.setAttribute("exposure", "1.05");
    mv.setAttribute("camera-orbit", "20deg 75deg auto");
    if (spinEl.checked) mv.setAttribute("auto-rotate", "");
    mv.addEventListener("camera-change", e => {
      if (e.detail.source === "user-interaction") syncFrom(mv);
    });
    mv.addEventListener("error", () => {          // degrade to the still, say so
      slot.querySelector("model-viewer")?.remove();
      slot.querySelector("img").hidden = false;
      const sig = slot.parentElement.querySelector(".sig");
      if (sig) sig.innerHTML = '<span class="bad">mesh failed to load — showing render</span>';
      want3d[key] = false;
      slot.querySelector(".flip").setAttribute("aria-pressed", "false");
    });
    slot.querySelector("img").hidden = true;
    slot.insertBefore(mv, slot.querySelector(".flip"));
  }).catch(() => { want3d[key] = false; });
}

function setMode(key, on) {
  want3d[key] = on;
  const btn = document.querySelector(`.flip[data-key="${key}"]`);
  if (btn) btn.setAttribute("aria-pressed", String(on));
  document.getElementById("all3d").setAttribute("aria-pressed",
    String(MODES.some(k => want3d[k])));
  document.getElementById("allStill").setAttribute("aria-pressed",
    String(!MODES.some(k => want3d[k])));
  const slot = document.querySelector(`.view[data-key="${key}"]`);
  if (!slot || !slot.dataset.glb) return;
  const existing = slot.querySelector("model-viewer");
  if (on && !existing) mount(slot, key, A[cur]);
  if (!on && existing) { existing.remove(); slot.querySelector("img").hidden = false; }
}

let cur = 0;
function show(i) {
  cur = (i + A.length) % A.length;
  const a = A[cur];
  const best = ["i1", "im", "t"].reduce((m, k) => (a.iou[k] ?? -1) > (a.iou[m] ?? -1) ? k : m, "i1");
  document.getElementById("cols").innerHTML = ["input", "i1", "im", "t", "gt"].map(k => {
    const has = !!a.glb[k], m = a.mesh[k];
    const flip = has ? `<button class="flip" data-key="${k}" type="button"
        aria-pressed="${want3d[k]}" title="show the generated mesh">3D</button>` : "";
    const sig = has
      ? `<span>${m.tris ? (m.tris / 1000).toFixed(0) + "k tris" : ""}</span>
         <span class="${m.iou != null && m.iou < 0.95 ? "bad" : ""}">${
           m.iou != null ? "sil " + m.iou.toFixed(3) : ""} · ${m.kb} KB</span>`
      : (k === "gt" ? "<span>render only — no mesh in cache</span>" : "");
    return `<div class="cell"><figure>
      <figcaption><span>${LABEL[k]}</span>
        <span class="iou ${k === best ? "best" : ""}">${k in a.iou ? f3(a.iou[k]) : ""}</span></figcaption>
      <div class="view" data-key="${k}" data-sha="${a.sha8}" ${has ? 'data-glb="1"' : ""}>
        <img src="${a.img[k]}" alt="${LABEL[k]} for ${a.sha8}">${flip}</div>
      <div class="sig">${sig}</div></figure></div>`;
  }).join("");
  MODES.forEach(k => { if (want3d[k] && a.glb[k]) setMode(k, true); });
  document.getElementById("cap").textContent = " " + a.caption;
  document.getElementById("meta").innerHTML =
    `<span>sha ${a.sha8}</span><span>I1 view ${a.view_i1}</span>` +
    `<span>IM views [${(a.views_im || []).join(", ")}]</span>`;
  [...list.children].forEach((el, j) => el.setAttribute("aria-current", j === cur));
  list.children[cur].scrollIntoView({block: "nearest"});
}

document.getElementById("cols").addEventListener("click", e => {
  const b = e.target.closest(".flip");
  if (b) setMode(b.dataset.key, b.getAttribute("aria-pressed") !== "true");
});
document.getElementById("all3d").addEventListener("click", () => MODES.forEach(k => setMode(k, true)));
document.getElementById("allStill").addEventListener("click", () => MODES.forEach(k => setMode(k, false)));
spinEl.addEventListener("change", () => viewers().forEach(v =>
  spinEl.checked ? v.setAttribute("auto-rotate", "") : v.removeAttribute("auto-rotate")));

list.addEventListener("click", e => {
  const b = e.target.closest(".item"); if (b) show(+b.dataset.i);
});
addEventListener("keydown", e => {
  // model-viewer binds the arrow keys itself once it has focus — don't fight it
  if (e.target.closest && e.target.closest("model-viewer")) return;
  if (["ArrowDown", "j"].includes(e.key)) { show(cur + 1); e.preventDefault(); }
  if (["ArrowUp", "k"].includes(e.key))   { show(cur - 1); e.preventDefault(); }
  if (e.key === "3") { const on = !MODES.some(k => want3d[k]); MODES.forEach(k => setMode(k, on)); }
});
show(0);
</script>
</body>
</html>"""

html = (HTML.replace("__ASSETS__", json.dumps(assets))
            .replace("__MEANS_CLEAN__", json.dumps(MEANS_CLEAN))
            .replace("__MEANS__", json.dumps(MEANS))
            .replace("__ASSET_DIR__", ASSET_DIR_NAME)
            .replace("__MESH_NOTE__", MESH_NOTE))
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB, {len(assets)} assets, {n_glb} meshes)")

if args.publish_assets:
    os.makedirs(args.publish_assets, exist_ok=True)
    shutil.copy(MODEL_VIEWER, args.publish_assets)
    for a in assets:
        for rel in a["glb"].values():
            shutil.copy(f"{GLB_SRC}/{os.path.basename(rel)}", args.publish_assets)
    tot = sum(os.path.getsize(os.path.join(args.publish_assets, f))
              for f in os.listdir(args.publish_assets))
    print(f"published {n_glb} glb + model-viewer.min.js -> {args.publish_assets} "
          f"({tot/1e6:.1f} MB)")
