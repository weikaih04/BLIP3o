"""Functional end-to-end load test: StreamingImageTo3D in NODE-SHARD mode reads the built
shard0 (streams path + node-local patch + _build decode) and yields training-shaped samples.
Single process → node-local world=1 → reads the whole shard (no sharding), enough to validate
the decode/keys/shapes match what the collator+model expect."""
import os, sys
sys.path.insert(0, "/fsx/home/weikai.huang/3dgen/model/BLIP3o")
os.chdir("/fsx/home/weikai.huang/3dgen/model/BLIP3o")
from trellis2_blip3o.data.streaming_task import StreamingImageTo3D

SHARD = "/opt/dlami/nvme/weikaih_mds_v3_shards/shard0"
ds = StreamingImageTo3D(shard_dirs=[SHARD], fuse_dino=False, ss_only=False,
                        max_slat_tokens=8192, shuffle=True, batch_size=1)
print(f"len(node shard) = {len(ds.ds)}")
it = iter(ds)
s = next(it)
print("sample keys:", sorted(s.keys()))
print("id              :", s["id"])
print("cond_hidden     :", tuple(s["cond_hidden"].shape), s["cond_hidden"].dtype)
print("cond_keep_mask  :", tuple(s["cond_keep_mask"].shape))
print("target_ss_latent:", tuple(s["target_ss_latent"].shape))
if "target_shape_slat_512_item" in s:
    sh = s["target_shape_slat_512_item"]
    print("shape_slat      : coords", tuple(sh["coords"].shape), "feats", tuple(sh["feats"].shape))
if "target_tex_slat_512_item" in s:
    tx = s["target_tex_slat_512_item"]
    print("tex_slat        : x_0.feats", tuple(tx["x_0"].feats.shape), "cond.feats", tuple(tx["concat_cond"].feats.shape))
else:
    print("tex_slat        : (none — this asset has no pbr@512)")
n = 1
for _ in zip(range(8), it):
    n += 1
print(f"OK — iterated {n} samples from node-local shard0, all decoded.")
