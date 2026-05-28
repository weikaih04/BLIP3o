"""Mixture infrastructure: `MixtureIterableDataset` + `MultiTaskCollator` + a
yaml-config loader that wires them up.

Design choices (see thread for full discussion):

* **Batch-granularity sampling** (one task per batch). At each `__iter__` step we
  draw a task by weight, then yield `batch_size` items from that task's cycle.
  HF Trainer's DataLoader pulls items 1-by-1 and the default collator gets a
  homogeneous list → routing by `_task` is trivial. (This is what
  LLaVA / Qwen-VL / InternVL do.)

* **Temperature sampling**: `p ∝ weight ** (1/τ)`. τ=1 → raw weights; τ→0 →
  argmax; τ→∞ → uniform. Use τ<1 to suppress oversized datasets.

* **Per-task infinite cycle**: each finite Dataset is wrapped in an infinite
  iterator. A "step" is the unit; epochs are not meaningful in mixture mode.

* **Per-task seed isolation**: each worker draws task and within-task indices
  from a NumPy generator seeded by `(rank, worker_id, task_idx)` so multi-worker
  DataLoader doesn't double-sample.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from . import registry


@dataclass
class TaskSpec:
    name: str          # registered task name (string id)
    weight: float      # raw mixing weight
    args: Dict[str, Any]  # kwargs passed to the task class

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        if "name" not in d:
            raise ValueError(f"task spec missing 'name': {d}")
        w = float(d.get("weight", 1.0))
        if w < 0:
            raise ValueError(f"task {d['name']!r} has negative weight {w}")
        return cls(name=d["name"], weight=w, args=dict(d.get("args", {})))


def _temperature_normalize(weights: Sequence[float], temperature: float) -> np.ndarray:
    """p ∝ w^(1/τ), then normalize. τ=0 → one-hot on argmax; τ very large → uniform."""
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any():
        raise ValueError(f"weights must be non-negative: {w}")
    if w.sum() <= 0:
        raise ValueError("all task weights are zero — nothing to sample")
    if temperature <= 0:
        out = np.zeros_like(w)
        out[int(np.argmax(w))] = 1.0
        return out
    p = w ** (1.0 / temperature)
    return p / p.sum()


class MixtureIterableDataset(IterableDataset):
    """Infinite iterator over per-task cycles, sampled by weight × temperature.

    Items carry `_task: <name>` so the collator can route. Batch granularity is
    enforced here: we yield `batch_size` items from the SAME task before
    re-rolling, so each batch sees one task.

    Two efficiency fixes baked in (see MULTI_TASK_DATA.md §Efficiency):
      • **Rank-synced task choice**: the per-step task draw uses a seed that
        does NOT include `rank`. All DDP/DeepSpeed ranks therefore pick the SAME
        task on the SAME step → no straggler at the ZeRO reduce-scatter / DDP
        all-reduce barrier. Within a task, the index sampler IS per-rank so
        data parallel still gives each rank distinct items.
      • **Per-batch task params**: if a task's Dataset exposes
        `set_batch_params(rng) -> None`, we call it ONCE per batch before
        yielding the `batch_size` items. This lets a task that has per-item
        randomness (e.g. multi_image_to_3d's n_views ∈ [2, max_views]) fix the
        param at batch level → zero intra-batch padding waste.

    Worker safety: with `num_workers > 1` the DataLoader may interleave items
    from different workers (different tasks) into one batch, breaking
    homogeneity. We emit a warning at iter time and recommend num_workers ≤ 1.
    (A pre-batched yield API + identity collator would lift this; see TODO.)
    """

    def __init__(
        self,
        tasks: List[Dataset],
        weights: Sequence[float],
        batch_size: int,
        temperature: float = 1.0,
        granularity: str = "batch",
        base_seed: int = 0,
    ):
        super().__init__()
        if len(tasks) != len(weights):
            raise ValueError(f"len(tasks)={len(tasks)} != len(weights)={len(weights)}")
        if granularity not in ("batch", "item"):
            raise ValueError(f"granularity must be 'batch' or 'item', got {granularity!r}")
        self.tasks = list(tasks)
        self.probs = _temperature_normalize(weights, temperature)
        self.batch_size = int(batch_size)
        self.granularity = granularity
        self.base_seed = int(base_seed)

    def task_names(self) -> List[str]:
        return [getattr(t, "task_name", t.__class__.__name__) for t in self.tasks]

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        rank = int(os.environ.get("RANK", 0))

        if num_workers > 1:
            import warnings
            warnings.warn(
                "MixtureIterableDataset with num_workers > 1 may interleave items "
                "from different workers (different tasks) into the same batch, "
                "breaking the homogeneity the MultiTaskCollator assumes. "
                "Use --dataloader_num_workers 0 or 1.",
                stacklevel=2,
            )

        # Task-choice RNG: SHARED across ranks (rank NOT in seed). This is the key
        # straggler fix: all ranks pick the same task on the same step → no waiting
        # at the DeepSpeed/DDP reduce barrier when tasks have different step-times.
        task_seed = (self.base_seed * 1_000_003) ^ (worker_id + 1)
        task_rng = np.random.default_rng(task_seed)

        # Within-task index RNG: PER-RANK (rank IS in seed). Each rank sees a
        # different slice of each task's dataset → data parallel.
        within_seed = (self.base_seed * 1_000_003) ^ (rank * 9176) ^ (worker_id + 1)
        per_task_rng = [
            np.random.default_rng(within_seed ^ (i * 7919 + 1))
            for i in range(len(self.tasks))
        ]

        # Per-task cyclic index streams (each shuffles within its own dataset every "epoch").
        def make_cycler(ti: int):
            ds = self.tasks[ti]
            n = len(ds)
            tr = per_task_rng[ti]
            while True:
                order = tr.permutation(n) if n > 1 else np.array([0])
                for j in order:
                    yield ds[int(j)]
        cyclers = [make_cycler(i) for i in range(len(self.tasks))]

        # Worker-id stride: with num_workers workers, each yields every num_workers-th
        # batch. We "burn" the cycler items for skipped batches so that all workers
        # remain consistent with the shared task_rng sequence.
        skip = worker_id
        while True:
            t = int(task_rng.choice(len(self.tasks), p=self.probs))
            n_emit = self.batch_size if self.granularity == "batch" else 1
            if skip > 0:
                for _ in range(n_emit):
                    next(cyclers[t])
                skip = (skip - 1) % max(1, num_workers)
                continue

            # Per-batch param hook: tasks that need batch-level decisions (e.g.
            # multi_image_to_3d picks n_views ONCE for the whole batch) implement
            # set_batch_params(rng); we call it here BEFORE the n_emit loop so all
            # items in this batch share that param.
            ds = self.tasks[t]
            setter = getattr(ds, "set_batch_params", None)
            if callable(setter):
                setter(task_rng)
            try:
                for _ in range(n_emit):
                    yield next(cyclers[t])
            finally:
                # Reset, so the next task draw starts clean.
                clearer = getattr(ds, "clear_batch_params", None)
                if callable(clearer):
                    clearer()
            skip = (num_workers - 1) if num_workers > 1 else 0


@dataclass
class MultiTaskCollator:
    """Routes the batch to the right task's `collate_fn` based on `_task`.

    With batch-granularity mixture sampling, every item in the batch shares the
    same task — we just check `batch[0]["_task"]` and dispatch.
    """
    processor: Any

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        task_name = batch[0].get("_task")
        if task_name is None:
            raise KeyError(
                "Batch item missing '_task' field. Every TaskDataset.__getitem__ "
                "must emit it. Got keys: " + repr(list(batch[0].keys()))
            )
        # Sanity check: all items in the batch must agree on task (batch granularity).
        for i, item in enumerate(batch[1:], 1):
            if item.get("_task") != task_name:
                raise ValueError(
                    f"MultiTaskCollator received a heterogeneous batch: item[0]._task="
                    f"{task_name!r} but item[{i}]._task={item.get('_task')!r}. "
                    "Check MixtureIterableDataset.granularity ('batch' enforces homogeneity)."
                )
        cls = registry.get_task(task_name)
        out = cls.collate_fn(batch, self.processor)
        # Stamp the task on the output so the model's forward can dispatch.
        out.setdefault("_task", task_name)
        return out


# --------------------------------------------------------------------------
# YAML loader
# --------------------------------------------------------------------------
@dataclass
class Mixture:
    dataset: MixtureIterableDataset
    collator: MultiTaskCollator
    specs: List[TaskSpec]

    def summary(self) -> str:
        names = self.dataset.task_names()
        probs = self.dataset.probs
        lines = ["Mixture:"]
        for n, p, s in zip(names, probs, self.specs):
            lines.append(f"  - {n:<24s} weight={s.weight:.3f}  p={p:.3f}")
        return "\n".join(lines)


def build_mixture(
    config_path: str,
    processor,
    batch_size: int,
    base_seed: int = 0,
) -> Mixture:
    """Read a yaml mixture config and instantiate Mixture(dataset, collator).

    Expected config schema (see configs/mix_*.yaml):

        tasks:
          - name: image_to_3d
            weight: 0.4
            args: { manifest: data/manifests/ready_v1.jsonl, ... }
          - name: vqa
            weight: 0.2
            args: { manifest: data/manifests/llava.jsonl, ... }
        sampling:
          granularity: batch        # batch | item
          temperature: 1.0
    """
    # Ensure tasks are imported (decorators populate the registry on import).
    import trellis2_blip3o.data.tasks  # noqa: F401

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "tasks" not in cfg:
        raise ValueError(f"mixture config must have a top-level 'tasks' list: {config_path}")

    specs = [TaskSpec.from_dict(d) for d in cfg["tasks"]]
    if not specs:
        raise ValueError(f"mixture config has empty tasks list: {config_path}")

    sampling = cfg.get("sampling") or {}
    granularity = sampling.get("granularity", "batch")
    temperature = float(sampling.get("temperature", 1.0))

    # Instantiate each task dataset
    datasets = []
    for s in specs:
        cls = registry.get_task(s.name)
        datasets.append(cls(**s.args))

    ds = MixtureIterableDataset(
        tasks=datasets,
        weights=[s.weight for s in specs],
        batch_size=batch_size,
        temperature=temperature,
        granularity=granularity,
        base_seed=base_seed,
    )
    collator = MultiTaskCollator(processor=processor)
    return Mixture(dataset=ds, collator=collator, specs=specs)
