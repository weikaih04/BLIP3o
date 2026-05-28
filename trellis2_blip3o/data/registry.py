"""Task registry — `@register_task("name")` makes a Dataset class findable by
short string id, so mixture configs can refer to tasks by name only.

Usage in a task file:
    @register_task("image_to_3d")
    class ImageTo3DDataset(TaskDataset): ...

Usage in mixture loader:
    cls = registry.get_task("image_to_3d")     # → ImageTo3DDataset
    ds  = cls(**args_from_yaml)
"""
from __future__ import annotations

from typing import Dict, Type

_TASKS: Dict[str, Type] = {}


def register_task(name: str):
    """Class decorator: register a TaskDataset subclass under `name`.

    A single class can be registered under multiple names (e.g. one ChatDataset
    serves "vqa" / "grounding" / "text_sft") — just decorate it multiple times.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"register_task requires a non-empty str name, got {name!r}")

    def deco(cls):
        if name in _TASKS and _TASKS[name] is not cls:
            raise ValueError(
                f"task name {name!r} already registered to {_TASKS[name].__name__}; "
                f"cannot re-register {cls.__name__}"
            )
        cls.task_name = name  # the most-recently-applied name wins on the attribute
        _TASKS[name] = cls
        return cls

    return deco


def get_task(name: str):
    """Look up a registered task class by name. Raises with a helpful list if absent."""
    if name not in _TASKS:
        raise KeyError(
            f"unknown task {name!r}; registered: {sorted(_TASKS)}. "
            f"Did you forget to `import trellis2_blip3o.data.tasks` so @register_task runs?"
        )
    return _TASKS[name]


def registered_tasks() -> Dict[str, Type]:
    """All current registrations. Used by error messages and config validation."""
    return dict(_TASKS)
