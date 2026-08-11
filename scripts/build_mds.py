"""Convert ready_v2_clean.jsonl → MosaicML Streaming (MDS) shards on local NVMe.
Each sample carries the raw npz bytes of cond (Qwen+DINO merged) / ss / shape / pbr +
a meta json. StreamingDataset (training side) does node/rank-aware sharding + deterministic
resume over these shards. Parallel: N writers → out/part_K/, then merge into one index.

Usage:
  python scripts/build_mds.py --out /opt/dlami/nvme/weikaih_mds [--procs 32]
      [--manifest ...] [--limit N] [--size-limit 256mb]
Resumable-ish: re-running overwrites; for a clean redo `rm -rf` the out dir first.
"""
import os, sys, json, time, argparse, multiprocessing as mp
sys.path.insert(0, "/fsx/sfr/weikaih/3dgen/model/BLIP3o")
os.chdir("/fsx/sfr/weikaih/3dgen/model/BLIP3o")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _preflight import require, assert_wrote
from trellis2_blip3o.vlm_cache import entry_path, view_key, combo_key
from streaming import MDSWriter
from streaming.base.util import merge_index

MANI = "/fsx/sfr/weikaih/3dgen/data/trellis2/manifests/ready_v2/ready_v2_clean.jsonl"
ROOT = os.environ.get("COND_ROOT",
    "/fsx/sfr/weikaih/3dgen/data/vlm_hidden_cache/qwen35-2b_tok1024_crop_mv1")
# COND_MODE=single (default) → single-view v000 cond from ROOT (unchanged legacy behavior).
# COND_MODE=im             → multi-image 4-view combo cond (m00.npz) from IM_COND_ROOT.
COND_MODE = os.environ.get("COND_MODE", "single").lower()
IM_COND_ROOT = os.environ.get("COND_ROOT_IM",
    "/fsx/home/weikai.huang/3dgen/vlm_hidden_cache/v22_im4l")


def _cond_path(sha):
    if COND_MODE == "im":
        return entry_path(IM_COND_ROOT, sha, combo_key(0))   # {root}/{sha[:2]}/{sha}/m00.npz
    return entry_path(ROOT, sha, view_key(0))                # {root}/{sha[:2]}/{sha}/v000.npz
COLUMNS = {"sha": "str", "subset": "str",
           "cond": "bytes", "ss": "bytes", "shape": "bytes", "pbr": "bytes",
           "meta": "json"}

def load_manifest(mani, limit=None):
    recs = []
    with open(mani) as f:
        for line in f:
            r = json.loads(line)
            sha = r.get("sha256")
            ss, shp = r.get("ss_latent_64"), r.get("shape_latent_512")
            if not (sha and ss and shp):
                continue
            recs.append({"sha": sha, "subset": r.get("subset", "?"),
                         "cond": _cond_path(sha),
                         "ss": ss, "shape": shp, "pbr": r.get("pbr_latent_512"),
                         "n_views": r.get("n_views"), "aesthetic": r.get("aesthetic_score"),
                         "captions": r.get("captions") or []})
            if limit and len(recs) >= limit:
                break
    return recs

def _rd(p):
    with open(p, "rb") as fh:
        return fh.read()

def write_part(args):
    pid, recs, outdir, size_limit = args
    part = os.path.join(outdir, f"part_{pid:03d}")
    n = nb = 0
    t0 = time.time()
    with MDSWriter(out=part, columns=COLUMNS, compression=None, size_limit=size_limit, hashes=[]) as w:
        for r in recs:
            try:
                cond = _rd(r["cond"]); ss = _rd(r["ss"]); shape = _rd(r["shape"])
                pbr = _rd(r["pbr"]) if (r.get("pbr") and os.path.exists(r["pbr"])) else b""
            except OSError:
                continue                         # missing/corrupt source → skip
            w.write({"sha": r["sha"], "subset": r["subset"],
                     "cond": cond, "ss": ss, "shape": shape, "pbr": pbr,
                     "meta": {"n_views": r["n_views"], "aesthetic": r["aesthetic"],
                              "captions": r["captions"]}})
            n += 1; nb += len(cond) + len(ss) + len(shape) + len(pbr)
    return pid, n, nb, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=MANI)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--size-limit", default="256mb")
    ap.add_argument("--shard", type=int, default=0, help="this shard's id (0..num_shards-1)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (node-sharding); recs[i] kept iff i%%num_shards==shard")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # A renamed cond root makes every read miss; the per-record `except OSError: continue`
    # below then skips the whole manifest and the writer still closes a valid, empty dataset.
    require(a.manifest, "manifest")
    require(IM_COND_ROOT if COND_MODE == "im" else ROOT,
            f"cond cache root (COND_MODE={COND_MODE})")
    print(f"loading {a.manifest} ... (COND_MODE={COND_MODE}, "
          f"cond_root={IM_COND_ROOT if COND_MODE=='im' else ROOT})", flush=True)
    recs = load_manifest(a.manifest, a.limit)
    if a.num_shards > 1:
        # node-sharding: deterministic stride over the SAME ordered manifest → consistent across nodes
        recs = [r for i, r in enumerate(recs) if i % a.num_shards == a.shard]
        print(f"shard {a.shard}/{a.num_shards}: {len(recs)} samples (stride i%%{a.num_shards}=={a.shard})", flush=True)
    # round-robin assignment → each part gets a balanced subset of all subsets
    parts = [[] for _ in range(a.procs)]
    for i, r in enumerate(recs):
        parts[i % a.procs].append(r)
    print(f"{len(recs)} samples → {a.procs} MDS writer parts (size_limit={a.size_limit})", flush=True)

    t0 = time.time(); tot = 0
    with mp.Pool(a.procs) as pool:
        for pid, n, nb, dt in pool.imap_unordered(
                write_part, [(k, parts[k], a.out, a.size_limit) for k in range(a.procs)]):
            tot += n
            print(f"  part {pid:03d}: {n} samples {nb/1e9:.1f}GB {dt:.0f}s | total {tot}", flush=True)

    # A zero-sample dataset is structurally valid: merging it and exiting 0 is exactly what
    # turns a missing source into something that reads as a successful build.
    assert_wrote(tot, what="MDS samples")

    # merge the per-part indices into one logical MDS dataset (sig 2: root auto-discovers part_*)
    part_dirs = [os.path.join(a.out, f"part_{k:03d}") for k in range(a.procs)]
    merge_index(a.out, keep_local=True)
    print(f"DONE: {tot} samples in {(time.time()-t0)/60:.1f} min → {a.out} "
          f"({sum(os.path.getsize(os.path.join(dp,f)) for dp in part_dirs for f in os.listdir(dp))/1e12:.2f}TB)", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
