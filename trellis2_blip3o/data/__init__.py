"""Multi-task data infrastructure for TrellisNativeVLM.

Public API:
    from trellis2_blip3o.data import build_mixture, register_task, get_task

A task is a `Dataset` decorated with `@register_task("<name>")` that emits
`{"_task": "<name>", ...}` dicts and provides a `collate_fn(batch, processor)`.
Tasks live in `trellis2_blip3o/data/tasks/`. The mixture is configured via
YAML; see `configs/mix_3d_only.yaml`.
"""
from .registry import register_task, get_task, registered_tasks
from .mixture import (
    MixtureIterableDataset,
    MultiTaskCollator,
    Mixture,
    TaskSpec,
    build_mixture,
)

# Trigger task registration by importing the tasks package.
from . import tasks  # noqa: F401

__all__ = [
    "register_task",
    "get_task",
    "registered_tasks",
    "MixtureIterableDataset",
    "MultiTaskCollator",
    "Mixture",
    "TaskSpec",
    "build_mixture",
]
