"""Importing this module is what triggers task registration via
`@register_task`. Every task file MUST be imported here, or `build_mixture`
won't know it exists."""
from . import threed   # noqa: F401  registers text_to_3d / image_to_3d / multi_image_to_3d
from . import chat     # noqa: F401  registers vqa / grounding / text_sft
