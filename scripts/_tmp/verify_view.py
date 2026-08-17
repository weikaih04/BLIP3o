"""Which view did the cached DINO actually come from?

build_dino_cache.py:67 picks the good view ONLY if GOOD_VIEWS=1 is in the env,
and the offset only if VIEW_SET is set; otherwise it falls back to `v` — and the
loop starts at v=0, i.e. the below-ground shot. build_vlm_cache_v22.pick_view has
no such switch. So a missing env var silently gives Qwen and DINO DIFFERENT
views, with no error. merge_vd_cache then drops the `view` key, so the file
cannot answer this. Brute-force it: run DINO on all 16 renders and see which one
reproduces the cached tokens."""
import os, sys, json, glob, numpy as np, torch
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o import _paths
from trellis2_blip3o.data.tasks.threed import ImageTo3DDataset
from trellis2_blip3o.dino_align import (DinoV3FeatureExtractor,
                                        TRELLIS_DINOV3_NAME, DINOV3_IMAGE_SIZE)
C = "/fsx/home/weikai.huang/3dgen/data/vlm_hidden_cache/v22_3dvlm_tok1024_mv1"
M = "/fsx/home/weikai.huang/3dgen/model/BLIP3o/manifests/v5_matclean_cached_train.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
recs = []
for l in open(M):
    r = json.loads(l)
    if os.path.exists(f"{C}/{r['sha256'][:2]}/{r['sha256']}/v000.npz"):
        recs.append(r)
    if len(recs) >= N: break
ds = ImageTo3DDataset.__new__(ImageTo3DDataset)   # only need _load_views
ds.crop_to_object = True
# same construction as build_dino_cache.py:45-48, same default image size
_ext = DinoV3FeatureExtractor(TRELLIS_DINOV3_NAME, image_size=DINOV3_IMAGE_SIZE)
_ext.model.eval().cuda()
dino = lambda ims: _ext(ims)
pick = lambda sha, off: 5 + (int(sha[:8], 16) + off) % 7
print(f"{'sha':14} {'pick_view A/B/C':>16} {'DINO 实际视角':>14}  一致?")
for r in recs:
    sha = r["sha256"]
    cached = torch.from_numpy(np.load(f"{C}/{sha[:2]}/{sha}/v000.npz")["dino_hidden"]).float()
    best, bestd = None, 1e9
    for v in range(16):
        try: imgs = ds._load_views(r["renders_dir"], [v])
        except Exception: continue
        if not imgs: continue
        with torch.no_grad(): f = dino(imgs).reshape(-1, 1024).float().cpu()
        if f.shape != cached.shape: continue
        d = float((f - cached).abs().mean())
        if d < bestd: best, bestd = v, d
    picks = [pick(sha, o) for o in (0, 3, 5)]
    print(f"{sha[:12]:14} {str(picks):>16} {best:>14}  "
          f"{'YES' if best in picks else 'NO'}  (mean|diff|={bestd:.4f})")
