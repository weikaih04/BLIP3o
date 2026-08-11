"""Two guards against jobs that fail while looking like they succeeded.

Both failures this file exists for have the same shape: something that was correct when it was
written expires as the tree moves, and nothing announces it.

  * A cond-cache root gets renamed. Every record then misses, `except OSError: continue` swallows
    each one, and the writer closes a structurally valid, ZERO-sample MDS dataset and exits 0.
    Training reads it and sees nothing. Nothing in the log says anything is wrong.
  * An enumeration script points at a cache that no longer exists, walks the whole tree first, and
    only dies at the very end — after the expensive part, with nothing written.

`require` turns the first minute into a crash instead of the third hour into a silent no-op;
`assert_wrote` refuses to report success for an empty artefact.
"""
import os
import sys


def require(path, why):
    """Abort now, naming what the path was for, rather than discovering it downstream."""
    if not path or not os.path.exists(path):
        sys.exit(f"[preflight] missing {why}: {path!r}")
    return path


def assert_wrote(n, skipped=0, what="records"):
    """A zero-row artefact must not exit 0 — that is indistinguishable from success."""
    if n == 0:
        sys.exit(f"[preflight] wrote 0 {what} ({skipped} skipped) — refusing to exit 0")
    if skipped and skipped > n:
        print(f"[preflight] WARNING: skipped {skipped} > wrote {n} {what}", flush=True)
    return n
