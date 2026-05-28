"""Shared base classes / protocol for task datasets.

A `TaskDataset` is just a `torch.utils.data.Dataset` (or IterableDataset) that:
  1. Sets a class-level `task_name` via `@register_task("…")`.
  2. `__getitem__` returns a dict that includes `"_task": self.task_name`.
  3. Provides `collate_fn(batch, processor) -> dict` (static or class method).

The mixture infra invokes the right `collate_fn` per batch by reading
`batch[0]["_task"]`, so each task can use its own padding / tokenization /
chat-template logic without leaking complexity to neighbors.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    """Convention-only base — subclasses just need `task_name` (set via
    `@register_task`) and a `collate_fn` that knows the task's batch shape."""

    task_name: ClassVar[str] = ""

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Any]], processor) -> Dict[str, Any]:
        """Build the model-ready batch. MUST include `_task` in the returned dict so
        the model's forward can dispatch on it."""
        raise NotImplementedError
