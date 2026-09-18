"""Motion sources and sampling for PLCS dataset generation.

Imports are resolved lazily because the ACCAD common-format adapter consumes
the low-level AMASS contract while :mod:`motion_sampler` consumes that adapter.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
        MotionSampler,
        MotionSequence,
    )
    from src.tasks.plcs.generate_dataset.sampling.motion_source import (
        ACCADMotionLibrary,
        MotionCategory,
        PLCSMotionClip,
        infer_accad_category,
        load_amass_motion_clip,
    )

_MOTION_SAMPLER_EXPORTS = frozenset({"MotionSampler", "MotionSequence"})
_MOTION_SOURCE_EXPORTS = frozenset(
    {
        "ACCADMotionLibrary",
        "MotionCategory",
        "PLCSMotionClip",
        "infer_accad_category",
        "load_amass_motion_clip",
    }
)


def __getattr__(name: str) -> Any:
    if name in _MOTION_SAMPLER_EXPORTS:
        module = import_module(
            "src.tasks.plcs.generate_dataset.sampling.motion_sampler"
        )
        return getattr(module, name)
    if name in _MOTION_SOURCE_EXPORTS:
        module = import_module("src.tasks.plcs.generate_dataset.sampling.motion_source")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "ACCADMotionLibrary",
    "MotionCategory",
    "MotionSampler",
    "MotionSequence",
    "PLCSMotionClip",
    "infer_accad_category",
    "load_amass_motion_clip",
]
