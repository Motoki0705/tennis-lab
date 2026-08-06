"""Strict, portable configuration for the mutable scene pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _command(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a command token sequence")
    result = tuple(value)
    if not result or any(
        type(token) is not str or not token.strip() for token in result
    ):
        raise ValueError(f"{name} must contain non-empty string tokens")
    return result


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _real(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    return float(value)


@dataclass(frozen=True, slots=True)
class NhtBoundaryConfig:
    reconstruct_command: tuple[str, ...]
    render_command: tuple[str, ...]
    config: Path | None
    working_directory: Path | None
    environment: Mapping[str, str]

    @classmethod
    def from_value(cls, value: object, resolver: PathResolver) -> Self:
        raw = _mapping(value, "nht")
        _keys(
            raw,
            {
                "reconstruct_command",
                "render_command",
                "config",
                "working_directory",
                "environment",
            },
            "nht",
        )
        config = (
            resolver.resolve(PathRole.EXTERNAL_ASSET, str(raw["config"]))
            if raw["config"] is not None
            else None
        )
        working = (
            resolver.resolve(PathRole.EXTERNAL_ASSET, str(raw["working_directory"]))
            if raw["working_directory"] is not None
            else None
        )
        environment_raw = _mapping(raw["environment"], "nht.environment")
        environment = {
            str(key): str(item)
            for key, item in environment_raw.items()
            if str(key).strip() and str(item).strip()
        }
        if len(environment) != len(environment_raw):
            raise ValueError("nht.environment keys and values must be non-empty")
        return cls(
            reconstruct_command=_command(
                raw["reconstruct_command"], "nht.reconstruct_command"
            ),
            render_command=_command(raw["render_command"], "nht.render_command"),
            config=config,
            working_directory=working,
            environment=environment,
        )


@dataclass(frozen=True, slots=True)
class AlignmentEvidenceConfig:
    mode: str
    maximum_views: int
    maximum_image_size: int
    maximum_pixels_per_view: int
    minimum_line_brightness: float
    maximum_line_saturation: float
    minimum_local_contrast: float
    minimum_projected_pixels_per_view: int
    raster_size: int
    optimizer_iterations: int
    optimizer_population_size: int
    minimum_fit_template_score: float
    line_inlier_distance_m: float
    minimum_holdout_view_fraction: float
    minimum_holdout_inlier_fraction: float

    @classmethod
    def from_value(cls, value: object) -> Self:
        raw = _mapping(value, "alignment.evidence")
        fields = {
            "mode",
            "maximum_views",
            "maximum_image_size",
            "maximum_pixels_per_view",
            "minimum_line_brightness",
            "maximum_line_saturation",
            "minimum_local_contrast",
            "minimum_projected_pixels_per_view",
            "raster_size",
            "optimizer_iterations",
            "optimizer_population_size",
            "minimum_fit_template_score",
            "line_inlier_distance_m",
            "minimum_holdout_view_fraction",
            "minimum_holdout_inlier_fraction",
        }
        _keys(raw, fields, "alignment.evidence")
        if raw["mode"] not in {"image_achromatic", "sparse_control"}:
            raise ValueError("alignment.evidence.mode is unsupported")
        result = cls(
            mode=str(raw["mode"]),
            maximum_views=_integer(
                raw["maximum_views"], "alignment.evidence.maximum_views"
            ),
            maximum_image_size=_integer(
                raw["maximum_image_size"], "alignment.evidence.maximum_image_size"
            ),
            maximum_pixels_per_view=_integer(
                raw["maximum_pixels_per_view"],
                "alignment.evidence.maximum_pixels_per_view",
            ),
            minimum_line_brightness=_real(
                raw["minimum_line_brightness"],
                "alignment.evidence.minimum_line_brightness",
            ),
            maximum_line_saturation=_real(
                raw["maximum_line_saturation"],
                "alignment.evidence.maximum_line_saturation",
            ),
            minimum_local_contrast=_real(
                raw["minimum_local_contrast"],
                "alignment.evidence.minimum_local_contrast",
            ),
            minimum_projected_pixels_per_view=_integer(
                raw["minimum_projected_pixels_per_view"],
                "alignment.evidence.minimum_projected_pixels_per_view",
            ),
            raster_size=_integer(raw["raster_size"], "alignment.evidence.raster_size"),
            optimizer_iterations=_integer(
                raw["optimizer_iterations"], "alignment.evidence.optimizer_iterations"
            ),
            optimizer_population_size=_integer(
                raw["optimizer_population_size"],
                "alignment.evidence.optimizer_population_size",
            ),
            minimum_fit_template_score=_real(
                raw["minimum_fit_template_score"],
                "alignment.evidence.minimum_fit_template_score",
            ),
            line_inlier_distance_m=_real(
                raw["line_inlier_distance_m"],
                "alignment.evidence.line_inlier_distance_m",
            ),
            minimum_holdout_view_fraction=_real(
                raw["minimum_holdout_view_fraction"],
                "alignment.evidence.minimum_holdout_view_fraction",
            ),
            minimum_holdout_inlier_fraction=_real(
                raw["minimum_holdout_inlier_fraction"],
                "alignment.evidence.minimum_holdout_inlier_fraction",
            ),
        )
        integers = (
            result.maximum_views,
            result.maximum_image_size,
            result.maximum_pixels_per_view,
            result.minimum_projected_pixels_per_view,
            result.raster_size,
            result.optimizer_iterations,
            result.optimizer_population_size,
        )
        if any(number < 2 for number in integers) or result.maximum_views < 4:
            raise ValueError("alignment.evidence integer limits are too small")
        fractions = (
            result.minimum_line_brightness,
            result.maximum_line_saturation,
            result.minimum_local_contrast,
            result.minimum_fit_template_score,
            result.minimum_holdout_view_fraction,
            result.minimum_holdout_inlier_fraction,
        )
        if any(not 0.0 <= number <= 1.0 for number in fractions):
            raise ValueError(
                "alignment.evidence normalized thresholds must lie in [0,1]"
            )
        if result.line_inlier_distance_m <= 0.0:
            raise ValueError(
                "alignment.evidence.line_inlier_distance_m must be positive"
            )
        return result


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    minimum_ground_points: int
    minimum_ground_support_fraction: float
    minimum_positive_camera_fraction: float
    holdout_fraction: float
    evidence: AlignmentEvidenceConfig

    @classmethod
    def from_value(cls, value: object) -> Self:
        raw = _mapping(value, "alignment")
        _keys(
            raw,
            {
                "minimum_ground_points",
                "minimum_ground_support_fraction",
                "minimum_positive_camera_fraction",
                "holdout_fraction",
                "evidence",
            },
            "alignment",
        )
        result = cls(
            minimum_ground_points=_integer(
                raw["minimum_ground_points"], "alignment.minimum_ground_points"
            ),
            minimum_ground_support_fraction=_real(
                raw["minimum_ground_support_fraction"],
                "alignment.minimum_ground_support_fraction",
            ),
            minimum_positive_camera_fraction=_real(
                raw["minimum_positive_camera_fraction"],
                "alignment.minimum_positive_camera_fraction",
            ),
            holdout_fraction=_real(
                raw["holdout_fraction"], "alignment.holdout_fraction"
            ),
            evidence=AlignmentEvidenceConfig.from_value(raw["evidence"]),
        )
        if result.minimum_ground_points < 10:
            raise ValueError("alignment.minimum_ground_points must be at least 10")
        for name, number in (
            ("minimum_ground_support_fraction", result.minimum_ground_support_fraction),
            (
                "minimum_positive_camera_fraction",
                result.minimum_positive_camera_fraction,
            ),
            ("holdout_fraction", result.holdout_fraction),
        ):
            if not 0.0 < number < 1.0:
                raise ValueError(f"alignment.{name} must lie in (0,1)")
        return result


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    samples_per_domain: int

    @classmethod
    def from_value(cls, value: object) -> Self:
        raw = _mapping(value, "datasets")
        _keys(raw, {"samples_per_domain"}, "datasets")
        result = cls(
            samples_per_domain=_integer(
                raw["samples_per_domain"], "datasets.samples_per_domain"
            )
        )
        if result.samples_per_domain < 1:
            raise ValueError("datasets.samples_per_domain must be positive")
        return result


@dataclass(frozen=True, slots=True)
class ScenePipelineConfig:
    schema: str
    seed: int
    roots: RuntimePathRoots
    nht: NhtBoundaryConfig
    alignment: AlignmentConfig
    datasets: DatasetConfig

    @property
    def resolver(self) -> PathResolver:
        return PathResolver(self.roots)

    @classmethod
    def load(cls, path: Path, repository_root: Path) -> Self:
        text = path.read_text()
        try:
            value: Any = json.loads(text)
        except json.JSONDecodeError:
            value = yaml.safe_load(text)
        raw = _mapping(value, "pipeline config")
        _keys(
            raw, {"schema", "seed", "roots", "nht", "alignment", "datasets"}, "config"
        )
        if raw["schema"] != "tennis_scene_pipeline_config_v1":
            raise ValueError("Unsupported scene pipeline config schema")
        seed = _integer(raw["seed"], "seed")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        roots = RuntimePathRoots.from_mapping(
            _mapping(raw["roots"], "roots"), repository_root=repository_root
        )
        resolver = PathResolver(roots)
        return cls(
            schema=str(raw["schema"]),
            seed=seed,
            roots=roots,
            nht=NhtBoundaryConfig.from_value(raw["nht"], resolver),
            alignment=AlignmentConfig.from_value(raw["alignment"]),
            datasets=DatasetConfig.from_value(raw["datasets"]),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "roots": dict(self.roots.as_mapping()),
            "nht": {
                "reconstruct_command": list(self.nht.reconstruct_command),
                "render_command": list(self.nht.render_command),
                "config": str(self.nht.config) if self.nht.config else None,
                "working_directory": (
                    str(self.nht.working_directory)
                    if self.nht.working_directory
                    else None
                ),
                "environment": dict(self.nht.environment),
            },
            "alignment": {
                "minimum_ground_points": self.alignment.minimum_ground_points,
                "minimum_ground_support_fraction": self.alignment.minimum_ground_support_fraction,
                "minimum_positive_camera_fraction": self.alignment.minimum_positive_camera_fraction,
                "holdout_fraction": self.alignment.holdout_fraction,
                "evidence": {
                    "mode": self.alignment.evidence.mode,
                    "maximum_views": self.alignment.evidence.maximum_views,
                    "maximum_image_size": self.alignment.evidence.maximum_image_size,
                    "maximum_pixels_per_view": self.alignment.evidence.maximum_pixels_per_view,
                    "minimum_line_brightness": self.alignment.evidence.minimum_line_brightness,
                    "maximum_line_saturation": self.alignment.evidence.maximum_line_saturation,
                    "minimum_local_contrast": self.alignment.evidence.minimum_local_contrast,
                    "minimum_projected_pixels_per_view": self.alignment.evidence.minimum_projected_pixels_per_view,
                    "raster_size": self.alignment.evidence.raster_size,
                    "optimizer_iterations": self.alignment.evidence.optimizer_iterations,
                    "optimizer_population_size": self.alignment.evidence.optimizer_population_size,
                    "minimum_fit_template_score": self.alignment.evidence.minimum_fit_template_score,
                    "line_inlier_distance_m": self.alignment.evidence.line_inlier_distance_m,
                    "minimum_holdout_view_fraction": self.alignment.evidence.minimum_holdout_view_fraction,
                    "minimum_holdout_inlier_fraction": self.alignment.evidence.minimum_holdout_inlier_fraction,
                },
            },
            "datasets": {"samples_per_domain": self.datasets.samples_per_domain},
        }


RUN_BOUNDARY_SCHEMA = StrictConfigSchema(
    name="synthetic.scene_pipeline",
    fields={
        "scene_id": ConfigField.of(str),
        "input_video": ConfigField.of(str, type(None)),
        "pipeline_config": ConfigField.of(str),
        "from_stage": ConfigField.of(str),
        "targets": ConfigField.sequence(ConfigField.of(str)),
        "nht_from_stage": ConfigField.of(str),
    },
)


def _project_resolver() -> PathResolver:
    roots = RuntimePathRoots.from_mapping(
        {f"{role.value}_root": "." for role in PathRole},
        repository_root=PROJECT_ROOT,
    )
    return PathResolver(roots)


@dataclass(frozen=True, slots=True)
class ScenePipelineRunConfig:
    """Strict Hydra boundary for one scene-pipeline invocation."""

    scene_id: str
    input_video: Path | None
    pipeline_config: Path
    from_stage: str
    targets: tuple[str, ...]
    nht_from_stage: str

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        plain = OmegaConf.to_container(cfg, resolve=True)
        raw = RUN_BOUNDARY_SCHEMA.validate(_mapping(plain, "scene pipeline run"))
        scene_id = cast(str, raw["scene_id"])
        if not scene_id.strip() or scene_id != scene_id.strip():
            raise ValueError("scene_id must be non-empty and trimmed")

        from_stage = cast(str, raw["from_stage"])
        allowed_stages = {
            "ingest",
            "reconstruction",
            "alignment",
            "court_dataset",
            "blcs_dataset",
            "plcs_dataset",
            "report",
        }
        if from_stage not in allowed_stages:
            raise ValueError(f"Unsupported from_stage: {from_stage}")

        targets = tuple(cast(Sequence[str], raw["targets"]))
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(target not in {"court", "blcs", "plcs"} for target in targets)
        ):
            raise ValueError("targets must be a unique non-empty domain list")

        nht_from_stage = cast(str, raw["nht_from_stage"])
        if nht_from_stage not in {
            "frames",
            "preprocess",
            "sfm",
            "sfm_selection",
            "nht_training",
            "scene_export",
            "reconstruction_report",
        }:
            raise ValueError(f"Unsupported nht_from_stage: {nht_from_stage}")

        pipeline_config = _project_resolver().resolve(
            PathRole.PROJECT, cast(str, raw["pipeline_config"])
        )
        if not pipeline_config.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_config}")
        pipeline = ScenePipelineConfig.load(pipeline_config, PROJECT_ROOT)

        input_value = cast(str | None, raw["input_video"])
        input_video = (
            pipeline.resolver.resolve(PathRole.EXTERNAL_ASSET, input_value)
            if input_value is not None
            else None
        )
        if input_video is not None and not input_video.is_file():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        if from_stage == "ingest" and input_video is None:
            raise ValueError("An ingest run requires input_video")
        if from_stage != "ingest" and input_video is not None:
            raise ValueError("input_video is accepted only when from_stage=ingest")

        return cls(
            scene_id=scene_id,
            input_video=input_video,
            pipeline_config=pipeline_config,
            from_stage=from_stage,
            targets=targets,
            nht_from_stage=nht_from_stage,
        )


def validate_run_boundary(cfg: DictConfig) -> None:
    """Validate the complete scene-pipeline Hydra boundary."""
    ScenePipelineRunConfig.from_config(cfg)
