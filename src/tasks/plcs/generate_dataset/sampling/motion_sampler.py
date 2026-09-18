"""Weighted sampling across explicitly adapted PLCS motion formats."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    MotionCategory,
    load_amass_motion_clip,
)
from src.tasks.plcs.motion import Coco17MotionClip, load_motion_clip
from src.tasks.plcs.motion.sources import AccadCoco17Adapter

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MotionFormat(StrEnum):
    """Registered source adapters accepted at the generation boundary."""

    AMASS_SMPLH_V1 = "amass_smplh_v1"
    COCO17_MOTION_V1 = "coco17_motion_v1"


@dataclass(frozen=True, slots=True)
class MotionSourceConfig:
    """One weighted category backed by exactly one registered format."""

    format: MotionFormat
    paths: tuple[Path, ...]
    weight: float


class MotionSampler:
    """Sample source files and adapt them to :class:`Coco17MotionClip`.

    Source-specific SMPL logic terminates inside registered adapters. The scene
    generator consumes only the common COCO-17 contract.
    """

    def __init__(
        self,
        config: DictConfig,
        smplh_model_path: Path | str,
        coco17_regressor_path: Path | str,
        device: str | torch.device = "cpu",
        *,
        accad_adapter: AccadCoco17Adapter | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self._motion_sources = self._parse_motion_sources()
        self._motion_files = self._index_motion_files()
        self._native_fps_cache: dict[Path, float] = {}
        needs_accad = any(
            source.format is MotionFormat.AMASS_SMPLH_V1
            for source in self._motion_sources.values()
        )
        self._accad_adapter = accad_adapter
        if needs_accad and self._accad_adapter is None:
            self._accad_adapter = AccadCoco17Adapter(
                smplh_model_path=smplh_model_path,
                coco17_regressor_path=coco17_regressor_path,
                device=self.device,
            )

        total_files = sum(len(files) for files in self._motion_files.values())
        print(f"MotionSampler: indexed {total_files} canonicalizable motion files")
        for category, files in self._motion_files.items():
            source = self._motion_sources[category]
            print(f"  - {category} [{source.format.value}]: {len(files)} files")

    def _parse_motion_sources(self) -> dict[str, MotionSourceConfig]:
        sources: dict[str, MotionSourceConfig] = {}
        for raw_category, raw in self.config.motion_sources.items():
            category = str(raw_category)
            if not category or category != category.strip():
                raise ValueError("Motion category names must be non-empty and trimmed.")
            keys = set(raw.keys())
            if keys != {"format", "paths", "weight"}:
                raise ValueError(
                    f"motion_sources.{category} must contain exactly format, paths, "
                    f"and weight; got {sorted(keys)}."
                )
            try:
                source_format = MotionFormat(str(raw.format))
            except ValueError as error:
                raise ValueError(
                    f"motion_sources.{category}.format is not registered: {raw.format!r}."
                ) from error
            raw_paths = raw.paths
            if isinstance(raw_paths, str):
                raise TypeError(
                    f"motion_sources.{category}.paths must be a path sequence."
                )
            paths = tuple(Path(str(path)) for path in raw_paths)
            if not paths:
                raise ValueError(f"motion_sources.{category}.paths must not be empty.")
            raw_weight = raw.weight
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise TypeError(f"motion_sources.{category}.weight must be numeric.")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    f"motion_sources.{category}.weight must be positive and finite."
                )
            if source_format is MotionFormat.AMASS_SMPLH_V1:
                try:
                    MotionCategory(category)
                except ValueError as error:
                    raise ValueError(
                        "AMASS/SMPL-H categories must be running, walking, or general; "
                        f"got {category!r}."
                    ) from error
            sources[category] = MotionSourceConfig(
                format=source_format,
                paths=paths,
                weight=weight,
            )
        if not sources:
            raise ValueError("At least one motion source must be configured.")
        return sources

    def _index_motion_files(self) -> dict[str, tuple[Path, ...]]:
        indexed: dict[str, tuple[Path, ...]] = {}
        for category, source in self._motion_sources.items():
            files: list[Path] = []
            for configured in source.paths:
                path = configured.resolve()
                if path.is_file():
                    candidates = [path]
                elif path.is_dir():
                    pattern = (
                        "*_poses.npz"
                        if source.format is MotionFormat.AMASS_SMPLH_V1
                        else "*.motion.npz"
                    )
                    candidates = sorted(path.rglob(pattern))
                else:
                    raise FileNotFoundError(
                        f"Configured motion source does not exist: {path}"
                    )
                for candidate in candidates:
                    if source.format is MotionFormat.AMASS_SMPLH_V1:
                        valid_name = candidate.name.endswith("_poses.npz")
                    else:
                        valid_name = candidate.name.endswith(".motion.npz")
                    if not valid_name:
                        raise ValueError(
                            f"{candidate} does not match {source.format.value}."
                        )
                    files.append(candidate.resolve())
            unique = tuple(sorted(set(files)))
            if len(unique) != len(files):
                raise ValueError(
                    f"motion_sources.{category} resolves duplicate motion files."
                )
            if not unique:
                raise FileNotFoundError(
                    f"motion_sources.{category} contains no {source.format.value} files."
                )
            indexed[category] = unique
        return indexed

    def sample_motion(
        self,
        category: str | None = None,
        *,
        required_fps: float | None = None,
    ) -> Coco17MotionClip:
        """Sample one compatible file and return the common contract.

        ``required_fps`` is used by multi-person generation, whose persisted scene
        has one shared time axis. It filters native sources rather than resampling
        them, so no source motion is silently sped up or slowed down.
        """
        if required_fps is not None:
            if (
                isinstance(required_fps, bool)
                or not isinstance(required_fps, (int, float))
                or not math.isfinite(float(required_fps))
                or float(required_fps) <= 0.0
            ):
                raise ValueError("required_fps must be a positive finite number.")
            required_fps = float(required_fps)
        eligible = {
            name: tuple(
                path
                for path in files
                if required_fps is None
                or math.isclose(
                    self._native_fps(path, source=self._motion_sources[name]),
                    required_fps,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )
            for name, files in self._motion_files.items()
        }
        if category is None:
            categories = tuple(name for name, files in eligible.items() if files)
            if not categories:
                raise RuntimeError(
                    f"No configured motion matches required_fps={required_fps}."
                )
            weights = tuple(self._motion_sources[name].weight for name in categories)
            selected_category = random.choices(categories, weights=weights, k=1)[0]
        else:
            if category not in self._motion_sources:
                raise ValueError(f"Unknown motion category: {category}")
            selected_category = category
            if not eligible[category]:
                raise RuntimeError(
                    f"Motion category {category!r} has no file matching "
                    f"required_fps={required_fps}."
                )
        selected_path = random.choice(eligible[selected_category])
        return self.load_motion(selected_path, category=selected_category)

    def _native_fps(self, path: Path, *, source: MotionSourceConfig) -> float:
        cached = self._native_fps_cache.get(path)
        if cached is not None:
            return cached
        if source.format is MotionFormat.COCO17_MOTION_V1:
            fps = load_motion_clip(path).fps
        else:
            with np.load(path, allow_pickle=False) as archive:
                if "mocap_framerate" not in archive.files:
                    raise ValueError(
                        f"{path}: missing required mocap_framerate for FPS matching."
                    )
                raw = np.asarray(archive["mocap_framerate"])
            if raw.size != 1 or not np.issubdtype(raw.dtype, np.number):
                raise ValueError(f"{path}: mocap_framerate must be one numeric scalar.")
            fps = float(raw.reshape(()).item())
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path}: native FPS must be positive and finite.")
        self._native_fps_cache[path] = fps
        return fps

    def load_motion(self, path: Path | str, *, category: str) -> Coco17MotionClip:
        """Load one configured file through its explicitly registered adapter."""
        if category not in self._motion_sources:
            raise ValueError(f"Unknown motion category: {category}")
        source = self._motion_sources[category]
        resolved = Path(path).resolve()
        if source.format is MotionFormat.COCO17_MOTION_V1:
            clip = load_motion_clip(resolved)
            if clip.category != category:
                raise ValueError(
                    f"Configured category {category!r} disagrees with artifact "
                    f"category {clip.category!r}: {resolved}"
                )
            return clip
        if self._accad_adapter is None:
            raise RuntimeError("AMASS/SMPL-H source has no configured adapter.")
        raw = load_amass_motion_clip(resolved, category=MotionCategory(category))
        return self._accad_adapter.convert(raw)

    def get_available_categories(self) -> list[str]:
        """Return configured categories in deterministic config order."""
        return list(self._motion_sources)

    def get_category_file_count(self, category: str) -> int:
        """Return the indexed file count for one configured category."""
        return len(self._motion_files.get(category, ()))


# Deliberate compatibility alias: callers importing MotionSequence now receive
# the source-independent representation instead of the removed AMASS container.
MotionSequence = Coco17MotionClip


__all__ = [
    "MotionFormat",
    "MotionSampler",
    "MotionSequence",
    "MotionSourceConfig",
]
