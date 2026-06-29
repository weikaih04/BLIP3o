"""Read per-sample arrays from packed WebDataset-style .tar shards by direct
positional read (os.pread) using the per-shard seek index emitted by
scripts/pack_webdataset.py — no tarfile at train time.

Why: the cond/ss/shape/pbr small files live on Lustre (/fsx); 8 ranks opening
~96 small files/step is metadata/latency-bound (~80 samp/s, the multi-GPU step-time
penalty). Packed shards on local NVMe give ~200+ samp/s cold and RAM speed warm,
fully hiding data behind the ~1.5s compute. Random os.pread keeps the existing
map-style mixture (per-rank shuffle + shared-task-RNG straggler-fix) untouched.

Fully lazy: the index and file handles are built on first read in whatever process
(DataLoader worker) calls read() — so nothing fork-unsafe is inherited from the parent.
"""
import os, io, json, glob, threading
import numpy as np


class PackedShardReader:
    def __init__(self, root: str):
        self.root = root
        self._index = None                  # sha -> (shard_id, {ext: [off, sz]})
        self._fh = {}                       # shard_id -> raw file fd (lazy, per-process)
        self._lock = threading.Lock()

    # ---- lazy index: load every shard-*.idx.json once (~48MB for 400k samples) ----
    def _ensure_index(self):
        if self._index is not None:
            return
        with self._lock:
            if self._index is not None:
                return
            idx = {}
            files = sorted(glob.glob(os.path.join(self.root, "shard-*.idx.json")))
            if not files:
                raise FileNotFoundError(
                    f"[packed_reader] no shard-*.idx.json under {self.root} "
                    "(run scripts/pack_webdataset.py first)")
            for f in files:
                sid = int(os.path.basename(f).split("-")[1].split(".")[0])
                with open(f) as fh:
                    d = json.load(fh)
                for sha, ent in d.items():
                    idx[sha] = (sid, ent)
            self._index = idx

    def __contains__(self, sha: str) -> bool:
        self._ensure_index()
        return sha in self._index

    def shas(self) -> set:
        """set of all shas present in the locally-staged shards (for node-local record filtering)."""
        self._ensure_index()
        return set(self._index.keys())

    def _fd(self, sid: int) -> int:
        fd = self._fh.get(sid)
        if fd is None:
            fd = os.open(os.path.join(self.root, f"shard-{sid:06d}.tar"), os.O_RDONLY)
            self._fh[sid] = fd
        return fd

    def read(self, sha: str, ext: str) -> bytes:
        """raw bytes of member {sha}.{ext} (raises KeyError on miss → caller resamples)."""
        self._ensure_index()
        sid, ent = self._index[sha]          # KeyError if sha not packed
        off, sz = ent[ext]                   # KeyError if ext (e.g. pbr) absent
        return os.pread(self._fd(sid), sz, off)   # thread-safe positional read

    def load(self, sha: str, ext: str):
        """np.load of the member (npz/npy) from its packed bytes."""
        return np.load(io.BytesIO(self.read(sha, ext)))

    def has(self, sha: str, ext: str) -> bool:
        self._ensure_index()
        e = self._index.get(sha)
        return bool(e and ext in e[1])
