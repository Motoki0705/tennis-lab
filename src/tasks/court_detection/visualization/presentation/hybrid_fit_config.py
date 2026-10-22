"""Strict, CPU-only configuration boundary for hybrid fit search and rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.visualization.presentation.hybrid_fit_search import (
    HybridFitSearch,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

BOUNDARY = "court_detection.hybrid_fit"


def _exact(mapping: Mapping[str, object], keys: set[str], path: str) -> None:
    if set(mapping) != keys:
        raise ValueError(f"{path} requires exactly {sorted(keys)}")


def _text(mapping: Mapping[str, object], key: str, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise ValueError(f"{path}.{key} must be nonempty and trimmed")
    return value


def _integer(mapping: Mapping[str, object], key: str, path: str) -> int:
    value = cast(int, require_config_value(mapping, key, int, path=path))
    if value < 1:
        raise ValueError(f"{path}.{key} must be positive")
    return value


def _number(mapping: Mapping[str, object], key: str, path: str) -> float:
    value = float(
        cast(float | int, require_config_value(mapping, key, (float, int), path=path))
    )
    if not math.isfinite(value):
        raise ValueError(f"{path}.{key} must be finite")
    return value


@dataclass(frozen=True)
class HybridFitConfig:
    mode: str
    resolver: PathResolver
    checkpoint: Path
    device: str
    source: TennisCourtDetectorSourceConfig
    input_kind: str
    image_path: Path | None
    sample_id: str | None
    output_dir: Path
    search: HybridFitSearch
    width: int
    height: int
    fps: float
    gif_width: int
    gif_fps: float

    @classmethod
    def from_config(cls, config: DictConfig) -> HybridFitConfig:
        mapping = as_config_mapping(config, path="config")
        _exact(
            mapping,
            {
                "paths",
                "run",
                "mode",
                "checkpoint",
                "device",
                "source",
                "input",
                "sample_id",
                "search",
                "output",
            },
            "config",
        )
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                require_config_mapping(mapping, "paths", path="config"),
                repository_root=PROJECT_ROOT,
            )
        )
        mode = _text(mapping, "mode", "config")
        if mode not in {"search", "render"}:
            raise ValueError("mode must be search or render")
        device = _text(mapping, "device", "config")
        if device not in {"cuda", "cpu"} or (mode == "search" and device != "cuda"):
            raise ValueError("device must be cuda or cpu; search requires cuda")
        source = TennisCourtDetectorSourceConfig.from_mapping(
            mapping["source"], resolver=resolver
        )
        raw_input = require_config_mapping(mapping, "input", path="config")
        _exact(raw_input, {"kind", "path"}, "input")
        input_kind = _text(raw_input, "kind", "input")
        if input_kind not in {"tennis_court_detector", "image"}:
            raise ValueError("input.kind must be tennis_court_detector or image")
        raw_path = require_config_value(
            raw_input, "path", (str, type(None)), path="input"
        )
        sample_id = cast(
            str | None,
            require_config_value(
                mapping, "sample_id", (str, type(None)), path="config"
            ),
        )
        if sample_id is not None and (
            not sample_id
            or sample_id != sample_id.strip()
            or sample_id in {".", ".."}
            or "/" in sample_id
            or "\\" in sample_id
        ):
            raise ValueError("sample_id must be a nonempty filename stem")
        image_path = None
        if input_kind == "image":
            if mode != "render" or sample_id is not None:
                raise ValueError(
                    "input.kind=image requires mode=render and sample_id=null"
                )
            image_path = resolver.resolve(
                PathRole.DATA, _text(raw_input, "path", "input")
            )
        elif raw_path is not None:
            raise ValueError("input.path must be null for TennisCourtDetector")
        elif mode == "render" and sample_id is None:
            raise ValueError(
                "mode=render requires sample_id for TennisCourtDetector val"
            )
        if mode == "search" and sample_id is not None:
            raise ValueError("sample_id is only valid for mode=render")
        search = require_config_mapping(mapping, "search", path="config")
        _exact(
            search,
            {"min_kp_only_error_px", "max_hybrid_error_px", "max_samples", "top_k"},
            "search",
        )
        selection = HybridFitSearch(
            _number(search, "min_kp_only_error_px", "search"),
            _number(search, "max_hybrid_error_px", "search"),
            _integer(search, "max_samples", "search"),
            _integer(search, "top_k", "search"),
        )
        output = require_config_mapping(mapping, "output", path="config")
        _exact(output, {"width", "height", "fps", "gif_width", "gif_fps"}, "output")
        width, height = (_integer(output, key, "output") for key in ("width", "height"))
        fps, gif_fps = (_number(output, key, "output") for key in ("fps", "gif_fps"))
        if width % 2 or height % 2 or min(fps, gif_fps) <= 0:
            raise ValueError(
                "Output dimensions must be even; frame rates must be positive"
            )
        run = require_config_mapping(mapping, "run", path="config")
        _exact(run, {"output_dir"}, "run")
        return cls(
            mode=mode,
            resolver=resolver,
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, _text(mapping, "checkpoint", "config")
            ),
            device=device,
            source=source,
            input_kind=input_kind,
            image_path=image_path,
            sample_id=sample_id,
            output_dir=resolver.resolve(
                PathRole.OUTPUT, _text(run, "output_dir", "run")
            )
            / "presentation",
            search=selection,
            width=width,
            height=height,
            fps=fps,
            gif_width=_integer(output, "gif_width", "output"),
            gif_fps=gif_fps,
        )


def validate_hybrid_fit_boundary(config: DictConfig) -> None:
    HybridFitConfig.from_config(config)


register_boundary_validator(BOUNDARY, validate_hybrid_fit_boundary)
