"""sys.path + env setup for trellis2_blip3o.

Adds TRELLIS.2 (external) and the project root to sys.path. The forked BLIP3o
package lives in this repo so we don't need to add it externally.

Import this once at the top of every entry script; idempotent.
"""
import os
import sys


# project root = trellis2_blip3o/ (the dir containing this package)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE = os.path.dirname(_REPO)  # world_explore/

_TRELLIS2 = os.path.join(_WORKSPACE, "third_party_3d_gen", "TRELLIS.2")
_SAM3D = os.path.join(_WORKSPACE, "third_party_3d_gen", "sam-3d-objects")

# Add the repo root so the forked blip3o/, tok/, trl/ packages are importable.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for p in (_TRELLIS2, _SAM3D):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# sam3d_objects/__init__.py runs heavy init unless this env var is set.
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

# TRELLIS.2's attention layer prefers flash_attn but falls back to pytorch SDPA.
os.environ.setdefault("ATTN_BACKEND", "sdpa")

# nvdiffrast JITs a CUDA extension at first import; point it at our conda env.
_conda = sys.prefix
_cuda_inc = os.path.join(_conda, "targets", "x86_64-linux", "include")
if os.path.isdir(_cuda_inc):
    os.environ.setdefault("CUDA_HOME", _conda)
    os.environ.setdefault("CPATH", _cuda_inc + ":" + os.environ.get("CPATH", ""))

# Path constants for downstream files.
TRELLIS2_ROOT = _TRELLIS2
SAM3D_ROOT = _SAM3D
REPO_ROOT = _REPO
WORLD_EXPLORE_ROOT = _WORKSPACE
CHECKPOINTS_ROOT = os.path.join(_WORKSPACE, "checkpoints")
DATA_ROOT = os.path.join(_REPO, "data")
