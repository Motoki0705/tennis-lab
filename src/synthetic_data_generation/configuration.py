"""Strict Hydra adapter for the canonical mutable scene pipeline.

The composed configuration is the only source of runtime values.  This module
does not retain the removed generic path-pipeline, artifact identity, executable
digest, or compatibility schemas.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    PartitionThresholds,
)
from src.synthetic_data_generation.alignment.settings import (
    AlignmentEvidenceSettings,
    CorrespondenceSettings,
    CourtCandidateFitSettings,
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitShape,
    OrbitStableField,
    OrbitTargetMode,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.runtime import DatasetPerformanceBudget
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageName,
)
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace
from src.synthetic_data_generation.reconstruction.contracts import (
    NHT_RECONSTRUCT_COMMAND,
    NHTPipelineConfig,
    NHTTrainingRuntime,
)
from src.synthetic_data_generation.rendering.nht.contracts import NHT_RENDER_COMMAND
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    pass

SCENE_PIPELINE_BOUNDARY = "synthetic.scene_pipeline"
SCENE_PIPELINE_SCHEMA = "canonical_scene_pipeline_v1"

_COURT_METADATA_FIELDS = frozenset(
    {
        "camera_parameters",
        "camera_profile",
        "candidate_id",
        "seed",
        "target_court",
        "transform",
    }
)

ConfigMapping = Mapping[str, object]
EnumT = TypeVar("EnumT", bound=StrEnum)


def _mapping(value: object, *, path: str) -> ConfigMapping:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ConfigurationTypeError(
            f"{path}: expected mapping, got {type(value).__name__}."
        )
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationTypeError(f"{path}: all keys must be strings.")
    return cast(ConfigMapping, value)


def _exact(value: object, *, path: str, keys: set[str]) -> ConfigMapping:
    mapping = _mapping(value, path=path)
    missing = sorted(keys - set(mapping))
    if missing:
        raise MissingConfigurationKeyError(
            "Missing required configuration key(s): "
            + ", ".join(f"{path}.{key}" for key in missing)
            + "."
        )
    unknown = sorted(set(mapping) - keys)
    if unknown:
        raise UnknownConfigurationKeyError(
            "Unknown configuration key(s): "
            + ", ".join(f"{path}.{key}" for key in unknown)
            + "."
        )
    return mapping


def _value(
    mapping: ConfigMapping,
    key: str,
    expected: type[object] | tuple[type[object], ...],
    *,
    path: str,
) -> object:
    if key not in mapping:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key: {path}.{key}."
        )
    value = mapping[key]
    accepted = expected if isinstance(expected, tuple) else (expected,)
    if type(value) not in accepted:
        expected_names = " | ".join(candidate.__name__ for candidate in accepted)
        raise ConfigurationTypeError(
            f"{path}.{key}: expected {expected_names}, got {type(value).__name__}."
        )
    return value


def _text(mapping: ConfigMapping, key: str, *, path: str) -> str:
    value = cast(str, _value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise SemanticConfigurationError(
            f"{path}.{key} must be a non-empty trimmed string."
        )
    return value


def _integer(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    minimum: int,
) -> int:
    value = cast(int, _value(mapping, key, int, path=path))
    if value < minimum:
        raise SemanticConfigurationError(
            f"{path}.{key} must be an integer >= {minimum}."
        )
    return value


def _number(mapping: ConfigMapping, key: str, *, path: str) -> float:
    value = float(cast("int | float", _value(mapping, key, (int, float), path=path)))
    if not math.isfinite(value):
        raise SemanticConfigurationError(f"{path}.{key} must be finite.")
    return value


def _flag(mapping: ConfigMapping, key: str, *, path: str) -> bool:
    return cast(bool, _value(mapping, key, bool, path=path))


def _sequence(value: object, *, path: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationTypeError(f"{path}: expected a non-string sequence.")
    return tuple(value)


def _text_sequence(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    minimum_length: int = 1,
) -> tuple[str, ...]:
    values = _sequence(
        _value(mapping, key, (list, tuple), path=path), path=f"{path}.{key}"
    )
    if len(values) < minimum_length or any(
        type(item) is not str or not item or item != item.strip() for item in values
    ):
        raise SemanticConfigurationError(
            f"{path}.{key} must contain at least {minimum_length} non-empty strings."
        )
    result = tuple(cast(str, item) for item in values)
    if len(result) != len(set(result)):
        raise SemanticConfigurationError(f"{path}.{key} must not contain duplicates.")
    return result


def _enum_sequence(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    enum_type: type[EnumT],
) -> tuple[EnumT, ...]:
    """Parse a non-empty unique sequence against one finite vocabulary."""
    values = _text_sequence(mapping, key, path=path)
    try:
        return tuple(enum_type(value) for value in values)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise SemanticConfigurationError(
            f"{path}.{key} contains an unknown value; allowed values are [{allowed}]."
        ) from error


def _enum_value(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    enum_type: type[EnumT],
) -> EnumT:
    """Parse one scalar against a finite typed vocabulary."""
    value = _text(mapping, key, path=path)
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise SemanticConfigurationError(
            f"{path}.{key} contains an unknown value; allowed values are [{allowed}]."
        ) from error


def _number_sequence(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    minimum_length: int = 1,
) -> tuple[float, ...]:
    values = _sequence(
        _value(mapping, key, (list, tuple), path=path), path=f"{path}.{key}"
    )
    if len(values) < minimum_length or any(
        type(item) not in (int, float) for item in values
    ):
        raise ConfigurationTypeError(
            f"{path}.{key} must contain at least {minimum_length} numeric values."
        )
    result = tuple(float(cast("int | float", item)) for item in values)
    if any(not math.isfinite(item) for item in result):
        raise SemanticConfigurationError(f"{path}.{key} values must be finite.")
    return result


def _ordered_range(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    positive: bool,
) -> tuple[float, float]:
    values = _number_sequence(mapping, key, path=path, minimum_length=2)
    if len(values) != 2:
        raise ConfigurationTypeError(f"{path}.{key} must contain exactly two values.")
    low, high = values
    if low > high or (positive and low <= 0.0):
        raise SemanticConfigurationError(
            f"{path}.{key} must be an ordered{' positive' if positive else ''} range."
        )
    return low, high


def _require_true(value: bool, *, path: str) -> None:
    if not value:
        raise SemanticConfigurationError(f"{path} must be true for production.")


@dataclass(frozen=True, slots=True)
class PipelineStageSettings:
    """Config-owned stage execution policy for the canonical runner."""

    config_schema: str
    seed: int
    preflight_before_invalidation: bool
    invalidate_descendants: bool
    atomic_fixed_path_publication: bool
    write_resolved_config: bool

    @classmethod
    def from_mapping(cls, value: object) -> PipelineStageSettings:
        raw = _exact(
            value,
            path="pipeline",
            keys={
                "config_schema",
                "seed",
                "preflight_before_invalidation",
                "invalidate_descendants",
                "atomic_fixed_path_publication",
                "write_resolved_config",
            },
        )
        result = cls(
            config_schema=_text(raw, "config_schema", path="pipeline"),
            seed=_integer(raw, "seed", path="pipeline", minimum=0),
            preflight_before_invalidation=_flag(
                raw, "preflight_before_invalidation", path="pipeline"
            ),
            invalidate_descendants=_flag(
                raw, "invalidate_descendants", path="pipeline"
            ),
            atomic_fixed_path_publication=_flag(
                raw, "atomic_fixed_path_publication", path="pipeline"
            ),
            write_resolved_config=_flag(raw, "write_resolved_config", path="pipeline"),
        )
        if result.config_schema != SCENE_PIPELINE_SCHEMA:
            raise SemanticConfigurationError(
                f"pipeline.config_schema must be {SCENE_PIPELINE_SCHEMA!r}."
            )
        for name in (
            "preflight_before_invalidation",
            "invalidate_descendants",
            "atomic_fixed_path_publication",
            "write_resolved_config",
        ):
            _require_true(cast(bool, getattr(result, name)), path=f"pipeline.{name}")
        return result


@dataclass(frozen=True, slots=True)
class NHTCommandPaths:
    """Installed public NHT commands and explicit subprocess execution policy."""

    reconstruct_executable: str | Path
    render_executable: str | Path
    pipeline_config: NHTPipelineConfig
    training_runtime: NHTTrainingRuntime
    environment: Mapping[str, str]
    reconstruction_timeout_seconds: float
    render_timeout_seconds: float

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> NHTCommandPaths:
        raw = _exact(
            value,
            path="nht",
            keys={
                "reconstruct_executable",
                "render_executable",
                "pipeline_config_path",
                "training_python_path",
                "trainer_path",
                "environment",
                "reconstruction_timeout_seconds",
                "render_timeout_seconds",
            },
        )
        reconstruct = _installed_nht_command(
            raw,
            key="reconstruct_executable",
            expected=NHT_RECONSTRUCT_COMMAND,
        )
        render = _installed_nht_command(
            raw,
            key="render_executable",
            expected=NHT_RENDER_COMMAND,
        )
        pipeline_config = _nht_pipeline_config(raw, resolver=resolver)
        training_runtime = _nht_training_runtime(raw, resolver=resolver)
        environment_raw = _mapping(raw["environment"], path="nht.environment")
        unknown_environment = sorted(
            set(environment_raw)
            - {
                "CUDA_VISIBLE_DEVICES",
                "TENNIS_LAB_NHT_MINIMUM_MEDIAN_TRACK_LENGTH",
                "TENNIS_LAB_NHT_MINIMUM_SPARSE_POINTS",
            }
        )
        if unknown_environment:
            raise UnknownConfigurationKeyError(
                "Unknown NHT public environment key(s): "
                + ", ".join(f"nht.environment.{key}" for key in unknown_environment)
                + "."
            )
        environment: dict[str, str] = {}
        for key in sorted(environment_raw):
            value = environment_raw[key]
            if (
                not key
                or key != key.strip()
                or type(value) is not str
                or not value
                or value != value.strip()
            ):
                raise SemanticConfigurationError(
                    "nht.environment must map trimmed non-empty names to "
                    "trimmed non-empty strings."
                )
            environment[key] = value
        reconstruction_timeout = _number(
            raw, "reconstruction_timeout_seconds", path="nht"
        )
        render_timeout = _number(raw, "render_timeout_seconds", path="nht")
        if min(reconstruction_timeout, render_timeout) <= 0.0:
            raise SemanticConfigurationError(
                "NHT subprocess timeouts must be positive."
            )
        return cls(
            reconstruct_executable=reconstruct,
            render_executable=render,
            pipeline_config=pipeline_config,
            training_runtime=training_runtime,
            environment=environment,
            reconstruction_timeout_seconds=reconstruction_timeout,
            render_timeout_seconds=render_timeout,
        )


def _nht_pipeline_config(
    mapping: ConfigMapping,
    *,
    resolver: PathResolver,
) -> NHTPipelineConfig:
    """Resolve and validate the public NHT config without provider imports."""
    configured = _text(mapping, "pipeline_config_path", path="nht")
    lexical_path = resolver.resolve_symlink_entry(
        PathRole.EXTERNAL_ASSET,
        configured,
    )
    if lexical_path.is_symlink():
        raise PathContractError(
            f"nht.pipeline_config_path must not be a symbolic link: {lexical_path}"
        )
    if not lexical_path.exists():
        raise PathContractError(
            f"nht.pipeline_config_path does not exist: {lexical_path}"
        )
    if not lexical_path.is_file():
        raise PathContractError(
            f"nht.pipeline_config_path is not a file: {lexical_path}"
        )
    resolved = resolver.resolve(PathRole.EXTERNAL_ASSET, configured)
    return NHTPipelineConfig.load(resolved)


def _nht_training_runtime(
    mapping: ConfigMapping,
    *,
    resolver: PathResolver,
) -> NHTTrainingRuntime:
    """Resolve the dedicated trainer environment without dereferencing its Python."""
    configured_python = _text(mapping, "training_python_path", path="nht")
    python = resolver.resolve_symlink_entry(
        PathRole.EXTERNAL_ASSET,
        configured_python,
    )
    configured_trainer = _text(mapping, "trainer_path", path="nht")
    trainer = resolver.resolve(PathRole.EXTERNAL_ASSET, configured_trainer)
    return NHTTrainingRuntime(python=python, trainer=trainer)


def _installed_nht_command(
    mapping: ConfigMapping,
    *,
    key: str,
    expected: str,
) -> str | Path:
    """Accept one public command name or an installed absolute executable."""
    configured = _text(mapping, key, path="nht")
    if configured == expected:
        return configured
    executable = Path(configured)
    if not executable.is_absolute() or executable.name != expected:
        raise SemanticConfigurationError(
            f"nht.{key} must be {expected!r} or an absolute path with that basename."
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PathContractError(f"nht.{key} is not an executable file: {executable}")
    return executable


@dataclass(frozen=True, slots=True)
class AlignmentConfiguration:
    """Complete evidence extraction and independent fit/holdout acceptance policy."""

    evidence: AlignmentEvidenceSettings
    acceptance: AlignmentAcceptancePolicy
    transform_inverse_atol: float
    projection_atol_px: float

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> AlignmentConfiguration:
        raw = _exact(
            value,
            path="alignment",
            keys={
                "evidence",
                "acceptance",
                "transform_inverse_atol",
                "projection_atol_px",
            },
        )
        evidence = cls._evidence(raw["evidence"], resolver=resolver)
        acceptance = cls._acceptance(raw["acceptance"])
        result = cls(
            evidence=evidence,
            acceptance=acceptance,
            transform_inverse_atol=_number(
                raw, "transform_inverse_atol", path="alignment"
            ),
            projection_atol_px=_number(raw, "projection_atol_px", path="alignment"),
        )
        if min(result.transform_inverse_atol, result.projection_atol_px) <= 0.0:
            raise SemanticConfigurationError("alignment tolerances must be positive.")
        return result

    @staticmethod
    def _evidence(
        value: object,
        *,
        resolver: PathResolver,
    ) -> AlignmentEvidenceSettings:
        path = "alignment.evidence"
        raw = _exact(
            value,
            path=path,
            keys={
                "seed",
                "fit_fraction",
                "holdout_fraction",
                "minimum_fit_cameras",
                "minimum_holdout_cameras",
                "camera_prefix_count",
                "line_model",
                "ground_plane",
                "projection",
                "candidate_fit",
                "correspondences",
            },
        )
        line_path = f"{path}.line_model"
        line_raw = _exact(
            raw["line_model"],
            path=line_path,
            keys={
                "checkpoint_path",
                "device",
                "probability_threshold",
                "maximum_selected_pixels_per_camera",
            },
        )
        line_model = CourtLineModelSettings(
            checkpoint_path=resolver.resolve(
                PathRole.CHECKPOINT, _text(line_raw, "checkpoint_path", path=line_path)
            ),
            device=_text(line_raw, "device", path=line_path),
            probability_threshold=_number(
                line_raw, "probability_threshold", path=line_path
            ),
            maximum_selected_pixels_per_camera=_integer(
                line_raw,
                "maximum_selected_pixels_per_camera",
                path=line_path,
                minimum=1,
            ),
        )
        ground_path = f"{path}.ground_plane"
        ground_raw = _exact(
            raw["ground_plane"],
            path=ground_path,
            keys={
                "footprint_quantile",
                "footprint_margin",
                "minimum_camera_height",
                "maximum_camera_height",
                "histogram_bin_width",
                "candidate_half_width",
                "ransac_threshold",
                "refine_threshold",
                "ransac_iterations",
                "ransac_sample_limit",
                "refine_iterations",
                "minimum_candidate_points",
                "minimum_support_points",
                "minimum_normal_up_cosine",
                "minimum_positive_camera_fraction",
                "support_bounds_quantile",
            },
        )
        ground = GroundPlaneSettings(
            footprint_quantile=_number(
                ground_raw, "footprint_quantile", path=ground_path
            ),
            footprint_margin=_number(ground_raw, "footprint_margin", path=ground_path),
            minimum_camera_height=_number(
                ground_raw, "minimum_camera_height", path=ground_path
            ),
            maximum_camera_height=_number(
                ground_raw, "maximum_camera_height", path=ground_path
            ),
            histogram_bin_width=_number(
                ground_raw, "histogram_bin_width", path=ground_path
            ),
            candidate_half_width=_number(
                ground_raw, "candidate_half_width", path=ground_path
            ),
            ransac_threshold=_number(ground_raw, "ransac_threshold", path=ground_path),
            refine_threshold=_number(ground_raw, "refine_threshold", path=ground_path),
            ransac_iterations=_integer(
                ground_raw, "ransac_iterations", path=ground_path, minimum=1
            ),
            ransac_sample_limit=_integer(
                ground_raw, "ransac_sample_limit", path=ground_path, minimum=1
            ),
            refine_iterations=_integer(
                ground_raw, "refine_iterations", path=ground_path, minimum=1
            ),
            minimum_candidate_points=_integer(
                ground_raw, "minimum_candidate_points", path=ground_path, minimum=1
            ),
            minimum_support_points=_integer(
                ground_raw, "minimum_support_points", path=ground_path, minimum=1
            ),
            minimum_normal_up_cosine=_number(
                ground_raw, "minimum_normal_up_cosine", path=ground_path
            ),
            minimum_positive_camera_fraction=_number(
                ground_raw, "minimum_positive_camera_fraction", path=ground_path
            ),
            support_bounds_quantile=_number(
                ground_raw, "support_bounds_quantile", path=ground_path
            ),
        )
        projection_path = f"{path}.projection"
        projection_raw = _exact(
            raw["projection"],
            path=projection_path,
            keys={
                "minimum_ray_plane_cosine",
                "maximum_ray_distance",
                "bounds_margin",
                "proximity_scale",
                "proximity_power",
                "grid_spacing",
                "minimum_projected_points_per_camera",
            },
        )
        projection = LineProjectionSettings(
            minimum_ray_plane_cosine=_number(
                projection_raw, "minimum_ray_plane_cosine", path=projection_path
            ),
            maximum_ray_distance=_number(
                projection_raw, "maximum_ray_distance", path=projection_path
            ),
            bounds_margin=_number(
                projection_raw, "bounds_margin", path=projection_path
            ),
            proximity_scale=_number(
                projection_raw, "proximity_scale", path=projection_path
            ),
            proximity_power=_number(
                projection_raw, "proximity_power", path=projection_path
            ),
            grid_spacing=_number(projection_raw, "grid_spacing", path=projection_path),
            minimum_projected_points_per_camera=_integer(
                projection_raw,
                "minimum_projected_points_per_camera",
                path=projection_path,
                minimum=1,
            ),
        )
        candidate_path = f"{path}.candidate_fit"
        candidate_raw = _exact(
            raw["candidate_fit"],
            path=candidate_path,
            keys={
                "maximum_candidate_count",
                "maximum_retained_state_count",
                "minimum_explained_evidence_fraction",
                "samples_per_metre",
                "minimum_nht_scene_units_per_metre",
                "maximum_nht_scene_units_per_metre",
                "orientation_minimum_radians",
                "orientation_maximum_radians",
                "score_distance_metres",
                "minimum_template_score",
                "family_orientation_tolerance_radians",
                "family_scale_relative_tolerance",
                "minimum_center_separation_metres",
                "optimizer_maximum_iterations",
                "optimizer_population_size",
                "optimizer_tolerance",
                "maximum_fit_points",
                "common_scale_relative_tolerance",
                "scale_bound_margin_relative",
                "evidence_assignment_distance_metres",
                "whole_template_inlier_distance_metres",
                "minimum_whole_template_inlier_fraction",
                "maximum_whole_template_q95_error_metres",
                "minimum_semantic_segment_inlier_fraction",
                "maximum_court_footprint_overlap_fraction",
            },
        )
        candidate = CourtCandidateFitSettings(
            maximum_candidate_count=_integer(
                candidate_raw,
                "maximum_candidate_count",
                path=candidate_path,
                minimum=1,
            ),
            maximum_retained_state_count=_integer(
                candidate_raw,
                "maximum_retained_state_count",
                path=candidate_path,
                minimum=1,
            ),
            minimum_explained_evidence_fraction=_number(
                candidate_raw,
                "minimum_explained_evidence_fraction",
                path=candidate_path,
            ),
            samples_per_metre=_number(
                candidate_raw, "samples_per_metre", path=candidate_path
            ),
            minimum_nht_scene_units_per_metre=_number(
                candidate_raw,
                "minimum_nht_scene_units_per_metre",
                path=candidate_path,
            ),
            maximum_nht_scene_units_per_metre=_number(
                candidate_raw,
                "maximum_nht_scene_units_per_metre",
                path=candidate_path,
            ),
            orientation_minimum_radians=_number(
                candidate_raw, "orientation_minimum_radians", path=candidate_path
            ),
            orientation_maximum_radians=_number(
                candidate_raw, "orientation_maximum_radians", path=candidate_path
            ),
            score_distance_metres=_number(
                candidate_raw, "score_distance_metres", path=candidate_path
            ),
            minimum_template_score=_number(
                candidate_raw, "minimum_template_score", path=candidate_path
            ),
            family_orientation_tolerance_radians=_number(
                candidate_raw,
                "family_orientation_tolerance_radians",
                path=candidate_path,
            ),
            family_scale_relative_tolerance=_number(
                candidate_raw,
                "family_scale_relative_tolerance",
                path=candidate_path,
            ),
            minimum_center_separation_metres=_number(
                candidate_raw,
                "minimum_center_separation_metres",
                path=candidate_path,
            ),
            optimizer_maximum_iterations=_integer(
                candidate_raw,
                "optimizer_maximum_iterations",
                path=candidate_path,
                minimum=1,
            ),
            optimizer_population_size=_integer(
                candidate_raw,
                "optimizer_population_size",
                path=candidate_path,
                minimum=1,
            ),
            optimizer_tolerance=_number(
                candidate_raw, "optimizer_tolerance", path=candidate_path
            ),
            maximum_fit_points=_integer(
                candidate_raw, "maximum_fit_points", path=candidate_path, minimum=1
            ),
            common_scale_relative_tolerance=_number(
                candidate_raw,
                "common_scale_relative_tolerance",
                path=candidate_path,
            ),
            scale_bound_margin_relative=_number(
                candidate_raw,
                "scale_bound_margin_relative",
                path=candidate_path,
            ),
            evidence_assignment_distance_metres=_number(
                candidate_raw,
                "evidence_assignment_distance_metres",
                path=candidate_path,
            ),
            whole_template_inlier_distance_metres=_number(
                candidate_raw,
                "whole_template_inlier_distance_metres",
                path=candidate_path,
            ),
            minimum_whole_template_inlier_fraction=_number(
                candidate_raw,
                "minimum_whole_template_inlier_fraction",
                path=candidate_path,
            ),
            maximum_whole_template_q95_error_metres=_number(
                candidate_raw,
                "maximum_whole_template_q95_error_metres",
                path=candidate_path,
            ),
            minimum_semantic_segment_inlier_fraction=_number(
                candidate_raw,
                "minimum_semantic_segment_inlier_fraction",
                path=candidate_path,
            ),
            maximum_court_footprint_overlap_fraction=_number(
                candidate_raw,
                "maximum_court_footprint_overlap_fraction",
                path=candidate_path,
            ),
        )
        correspondence_path = f"{path}.correspondences"
        correspondence_raw = _exact(
            raw["correspondences"],
            path=correspondence_path,
            keys={
                "maximum_match_distance_metres",
                "maximum_correspondences_per_camera",
                "minimum_correspondences_per_camera",
            },
        )
        correspondences = CorrespondenceSettings(
            maximum_match_distance_metres=_number(
                correspondence_raw,
                "maximum_match_distance_metres",
                path=correspondence_path,
            ),
            maximum_correspondences_per_camera=_integer(
                correspondence_raw,
                "maximum_correspondences_per_camera",
                path=correspondence_path,
                minimum=1,
            ),
            minimum_correspondences_per_camera=_integer(
                correspondence_raw,
                "minimum_correspondences_per_camera",
                path=correspondence_path,
                minimum=1,
            ),
        )
        return AlignmentEvidenceSettings(
            seed=_integer(raw, "seed", path=path, minimum=0),
            fit_fraction=_number(raw, "fit_fraction", path=path),
            holdout_fraction=_number(raw, "holdout_fraction", path=path),
            minimum_fit_cameras=_integer(
                raw, "minimum_fit_cameras", path=path, minimum=1
            ),
            minimum_holdout_cameras=_integer(
                raw, "minimum_holdout_cameras", path=path, minimum=1
            ),
            camera_prefix_count=_integer(
                raw, "camera_prefix_count", path=path, minimum=1
            ),
            line_model=line_model,
            ground_plane=ground,
            projection=projection,
            candidate_fit=candidate,
            correspondences=correspondences,
        )

    @staticmethod
    def _acceptance(value: object) -> AlignmentAcceptancePolicy:
        raw = _exact(
            value,
            path="alignment.acceptance",
            keys={"fit", "holdout"},
        )

        def thresholds(partition: str) -> PartitionThresholds:
            path = f"alignment.acceptance.{partition}"
            partition_raw = _exact(
                raw[partition],
                path=path,
                keys={
                    "minimum_camera_count",
                    "minimum_correspondence_count",
                    "inlier_distance_m",
                    "minimum_inlier_fraction",
                    "maximum_rms_error_m",
                    "maximum_q95_error_m",
                },
            )
            return PartitionThresholds(
                minimum_camera_count=_integer(
                    partition_raw, "minimum_camera_count", path=path, minimum=1
                ),
                minimum_correspondence_count=_integer(
                    partition_raw,
                    "minimum_correspondence_count",
                    path=path,
                    minimum=3,
                ),
                inlier_distance_m=_number(
                    partition_raw, "inlier_distance_m", path=path
                ),
                minimum_inlier_fraction=_number(
                    partition_raw, "minimum_inlier_fraction", path=path
                ),
                maximum_rms_error_m=_number(
                    partition_raw, "maximum_rms_error_m", path=path
                ),
                maximum_q95_error_m=_number(
                    partition_raw, "maximum_q95_error_m", path=path
                ),
            )

        return AlignmentAcceptancePolicy(
            fit=thresholds("fit"),
            holdout=thresholds("holdout"),
        )


@dataclass(frozen=True, slots=True)
class CourtTrajectoryPolicy:
    """Typed trajectory-family policy independent of view and sampling."""

    shapes: tuple[OrbitShape, ...]
    axis_ratios: tuple[float, ...]
    orientations_degrees: tuple[float, ...]
    center_kinds: tuple[OrbitCenterKind, ...]
    captured_offset_scale_range: tuple[float, float]
    base_heights_m: tuple[float, ...]
    vertical_modulations_m: tuple[float, ...]
    curve_modes: tuple[OrbitCurveMode, ...]
    sfm_boundary_margin_m: float | None
    sfm_boundary_expansion_percent: float
    sfm_complex_center_on_hull: bool
    spatial_coverage_cell_m: float | None

    @classmethod
    def from_mapping(cls, value: object) -> CourtTrajectoryPolicy:
        raw = _exact(
            value,
            path="dataset.court.trajectory",
            keys={
                "shapes",
                "axis_ratios",
                "orientations_degrees",
                "center_kinds",
                "captured_offset_scale_range",
                "base_heights_m",
                "vertical_modulations_m",
                "curve_modes",
                "sfm_boundary_margin_m",
                "sfm_boundary_expansion_percent",
                "sfm_complex_center_on_hull",
                "spatial_coverage_cell_m",
            },
        )
        path = "dataset.court.trajectory"
        result = cls(
            shapes=_enum_sequence(
                raw,
                "shapes",
                path=path,
                enum_type=OrbitShape,
            ),
            axis_ratios=_number_sequence(raw, "axis_ratios", path=path),
            orientations_degrees=_number_sequence(
                raw, "orientations_degrees", path=path, minimum_length=3
            ),
            center_kinds=_enum_sequence(
                raw,
                "center_kinds",
                path=path,
                enum_type=OrbitCenterKind,
            ),
            captured_offset_scale_range=_ordered_range(
                raw, "captured_offset_scale_range", path=path, positive=True
            ),
            base_heights_m=_number_sequence(
                raw, "base_heights_m", path=path, minimum_length=3
            ),
            vertical_modulations_m=_number_sequence(
                raw, "vertical_modulations_m", path=path
            ),
            curve_modes=_enum_sequence(
                raw,
                "curve_modes",
                path=path,
                enum_type=OrbitCurveMode,
            ),
            sfm_boundary_margin_m=(
                None
                if raw["sfm_boundary_margin_m"] is None
                else _number(raw, "sfm_boundary_margin_m", path=path)
            ),
            sfm_boundary_expansion_percent=_number(
                raw, "sfm_boundary_expansion_percent", path=path
            ),
            sfm_complex_center_on_hull=_flag(
                raw, "sfm_complex_center_on_hull", path=path
            ),
            spatial_coverage_cell_m=(
                None
                if raw["spatial_coverage_cell_m"] is None
                else _number(raw, "spatial_coverage_cell_m", path=path)
            ),
        )
        if result.sfm_boundary_margin_m is not None:
            margin = result.sfm_boundary_margin_m
            if margin < 0.0 or result.captured_offset_scale_range[1] > 1.0:
                raise SemanticConfigurationError(
                    "SfM bounds require non-negative margin and radius scales <= 1."
                )
        if result.sfm_complex_center_on_hull and result.sfm_boundary_margin_m is None:
            raise SemanticConfigurationError(
                "Captured-hull complex centre requires explicit SfM bounds."
            )
        expansion = result.sfm_boundary_expansion_percent
        if expansion < 0.0 or (
            expansion > 0.0 and result.sfm_boundary_margin_m is None
        ):
            raise SemanticConfigurationError(
                "SfM expansion requires a non-negative percent and explicit SfM bounds."
            )
        if result.spatial_coverage_cell_m is not None:
            cell_m = result.spatial_coverage_cell_m
            if cell_m <= 0.0 or result.sfm_boundary_margin_m is None:
                raise SemanticConfigurationError(
                    "Spatial coverage requires a positive cell size and explicit SfM bounds."
                )
        if not {OrbitShape.CIRCLE, OrbitShape.ELLIPSE}.issubset(result.shapes):
            raise SemanticConfigurationError(
                "Court trajectory shapes must include circle and ellipse."
            )
        if 1.0 not in result.axis_ratios or not any(
            ratio <= 0.8 for ratio in result.axis_ratios
        ):
            raise SemanticConfigurationError(
                "Court trajectory axis ratios require circle=1 and an ellipse <= 0.8."
            )
        if any(not 0.0 < ratio <= 1.0 for ratio in result.axis_ratios):
            raise SemanticConfigurationError(
                "Court trajectory axis ratios must be within (0, 1]."
            )
        if not {0.0, 45.0, 90.0}.issubset(result.orientations_degrees):
            raise SemanticConfigurationError(
                "Court orientations must include 0, 45, and 90 degrees."
            )
        if set(result.center_kinds) != set(OrbitCenterKind):
            raise SemanticConfigurationError(
                "Court center kinds must be complex and court."
            )
        if len(set(result.base_heights_m)) < 3 or min(result.base_heights_m) <= 0.0:
            raise SemanticConfigurationError(
                "Court base heights require three positive levels."
            )
        if min(result.vertical_modulations_m) < 0.0 or not any(
            value > 0.0 for value in result.vertical_modulations_m
        ):
            raise SemanticConfigurationError(
                "Court vertical modulation requires a positive value."
            )
        if set(result.curve_modes) != set(OrbitCurveMode):
            raise SemanticConfigurationError(
                "Court trajectories require a smooth non-planar mode."
            )
        return result


@dataclass(frozen=True, slots=True)
class CourtViewPolicy:
    """Typed camera-target and coverage policy."""

    target_modes: tuple[OrbitTargetMode, ...]
    coverage_modes: tuple[OrbitCoverageMode, ...]
    look_at_height_m: tuple[float, float]
    hfov_degrees: tuple[float, float]
    look_at_jitter_radius_m: float

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        schema_version: CourtDatasetSchemaVersion,
    ) -> CourtViewPolicy:
        raw = _exact(
            value,
            path="dataset.court.view",
            keys={
                "target_modes",
                "coverage_modes",
                "look_at_height_m",
                "hfov_degrees",
                "look_at_jitter_radius_m",
            },
        )
        path = "dataset.court.view"
        result = cls(
            target_modes=_enum_sequence(
                raw,
                "target_modes",
                path=path,
                enum_type=OrbitTargetMode,
            ),
            coverage_modes=_enum_sequence(
                raw,
                "coverage_modes",
                path=path,
                enum_type=OrbitCoverageMode,
            ),
            look_at_height_m=_ordered_range(
                raw, "look_at_height_m", path=path, positive=False
            ),
            hfov_degrees=_ordered_range(raw, "hfov_degrees", path=path, positive=True),
            look_at_jitter_radius_m=_number(raw, "look_at_jitter_radius_m", path=path),
        )
        radius = result.look_at_jitter_radius_m
        if radius < 0.0 or (
            radius > 0.0 and schema_version is not CourtDatasetSchemaVersion.V3
        ):
            raise SemanticConfigurationError(
                "Look-at jitter requires v3 and a non-negative radius."
            )
        if not isinstance(schema_version, CourtDatasetSchemaVersion):
            raise ConfigurationTypeError(
                "dataset.court.schema_version must be a CourtDatasetSchemaVersion."
            )
        if schema_version is CourtDatasetSchemaVersion.V1 and set(
            result.target_modes
        ) != set(OrbitTargetMode):
            raise SemanticConfigurationError(
                "Court v1 target modes must include complex/court center and both baselines."
            )
        if schema_version in (
            CourtDatasetSchemaVersion.V2,
            CourtDatasetSchemaVersion.V3,
        ) and result.target_modes != (OrbitTargetMode.COURT_CENTER,):
            raise SemanticConfigurationError(
                "Court v2/v3 target modes must be exactly [court_center]."
            )
        if set(result.coverage_modes) != set(OrbitCoverageMode):
            raise SemanticConfigurationError(
                "Court coverage modes must be full, near_full, and partial."
            )
        if result.look_at_height_m[0] < 0.0:
            raise SemanticConfigurationError(
                "Court look-at heights must be non-negative."
            )
        if result.hfov_degrees[1] >= 180.0:
            raise SemanticConfigurationError(
                "Court HFOV range must stay below 180 degrees."
            )
        return result


@dataclass(frozen=True, slots=True)
class CourtSamplingPolicy:
    """Deterministic coverage selection, budget, split, and release gates."""

    mode: OrbitSamplingMode
    seed: int
    stable_field_order: tuple[OrbitStableField, ...]
    coverage_objective: tuple[OrbitCoverageObjective, ...]
    proposal_budget: int
    minimum_trajectory_groups: int
    minimum_accepted_frames: int
    maximum_adjacent_step_m: float
    minimum_accepted_fraction: float
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    shard_group_count: int

    @classmethod
    def from_mapping(cls, value: object) -> CourtSamplingPolicy:
        raw = _exact(
            value,
            path="dataset.court.sampling",
            keys={
                "mode",
                "seed",
                "stable_field_order",
                "coverage_objective",
                "proposal_budget",
                "minimum_trajectory_groups",
                "minimum_accepted_frames",
                "maximum_adjacent_step_m",
                "minimum_accepted_fraction",
                "train_fraction",
                "validation_fraction",
                "test_fraction",
                "shard_group_count",
            },
        )
        path = "dataset.court.sampling"
        result = cls(
            mode=_enum_value(
                raw,
                "mode",
                path=path,
                enum_type=OrbitSamplingMode,
            ),
            seed=_integer(raw, "seed", path=path, minimum=0),
            stable_field_order=_enum_sequence(
                raw,
                "stable_field_order",
                path=path,
                enum_type=OrbitStableField,
            ),
            coverage_objective=_enum_sequence(
                raw,
                "coverage_objective",
                path=path,
                enum_type=OrbitCoverageObjective,
            ),
            proposal_budget=_integer(raw, "proposal_budget", path=path, minimum=1),
            minimum_trajectory_groups=_integer(
                raw, "minimum_trajectory_groups", path=path, minimum=1
            ),
            minimum_accepted_frames=_integer(
                raw, "minimum_accepted_frames", path=path, minimum=1
            ),
            maximum_adjacent_step_m=_number(raw, "maximum_adjacent_step_m", path=path),
            minimum_accepted_fraction=_number(
                raw, "minimum_accepted_fraction", path=path
            ),
            train_fraction=_number(raw, "train_fraction", path=path),
            validation_fraction=_number(raw, "validation_fraction", path=path),
            test_fraction=_number(raw, "test_fraction", path=path),
            shard_group_count=_integer(raw, "shard_group_count", path=path, minimum=1),
        )
        if set(result.stable_field_order) != set(OrbitStableField):
            raise SemanticConfigurationError(
                "Court stable_field_order must list every typed trajectory field exactly once."
            )
        if set(result.coverage_objective) != set(OrbitCoverageObjective):
            raise SemanticConfigurationError(
                "Court coverage_objective must list every objective token family exactly once."
            )
        if result.proposal_budget != 4_800:
            raise SemanticConfigurationError(
                "B00 Court proposal_budget must be exactly 4,800."
            )
        if result.minimum_trajectory_groups < 24:
            raise SemanticConfigurationError(
                "Court production requires at least 24 groups."
            )
        if result.minimum_accepted_frames < 2_000:
            raise SemanticConfigurationError(
                "Court production requires at least 2,000 frames."
            )
        if not 0.0 < result.maximum_adjacent_step_m <= 1.05:
            raise SemanticConfigurationError(
                "Court adjacent arc step must be within (0, 1.05]."
            )
        if not 0.9 <= result.minimum_accepted_fraction <= 1.0:
            raise SemanticConfigurationError(
                "Court accepted fraction must be within [0.9, 1]."
            )
        fractions = (
            result.train_fraction,
            result.validation_fraction,
            result.test_fraction,
        )
        if min(fractions) <= 0.0 or not math.isclose(
            sum(fractions), 1.0, abs_tol=1e-12
        ):
            raise SemanticConfigurationError(
                "Court split fractions must be positive and sum to 1."
            )
        if result.shard_group_count > result.minimum_trajectory_groups:
            raise SemanticConfigurationError(
                "Court shard_group_count cannot exceed minimum trajectory groups."
            )
        return result


def _performance_budget(value: object, *, path: str) -> DatasetPerformanceBudget:
    raw = _exact(
        value,
        path=path,
        keys={
            "maximum_wall_seconds",
            "maximum_published_bytes",
            "maximum_published_fraction_of_dense_reference",
            "maximum_nht_invocations",
            "maximum_background_cache_misses",
            "maximum_complete_array_scans_per_sample",
            "maximum_batch_frames",
            "execution_device",
            "require_cuda",
        },
    )
    try:
        return DatasetPerformanceBudget(
            maximum_wall_seconds=_number(
                raw,
                "maximum_wall_seconds",
                path=path,
            ),
            maximum_published_bytes=_integer(
                raw,
                "maximum_published_bytes",
                path=path,
                minimum=1,
            ),
            maximum_published_fraction_of_dense_reference=_number(
                raw,
                "maximum_published_fraction_of_dense_reference",
                path=path,
            ),
            maximum_nht_invocations=_integer(
                raw,
                "maximum_nht_invocations",
                path=path,
                minimum=1,
            ),
            maximum_background_cache_misses=_integer(
                raw,
                "maximum_background_cache_misses",
                path=path,
                minimum=1,
            ),
            maximum_complete_array_scans_per_sample=_integer(
                raw,
                "maximum_complete_array_scans_per_sample",
                path=path,
                minimum=1,
            ),
            maximum_batch_frames=_integer(
                raw,
                "maximum_batch_frames",
                path=path,
                minimum=1,
            ),
            execution_device=_text(raw, "execution_device", path=path),
            require_cuda=_flag(raw, "require_cuda", path=path),
        )
    except (TypeError, ValueError) as error:
        raise SemanticConfigurationError(f"{path}: {error}") from error


@dataclass(frozen=True, slots=True)
class CourtDatasetConfiguration:
    """Complete typed Court dataset policy."""

    schema_version: CourtDatasetSchemaVersion
    trajectory: CourtTrajectoryPolicy
    view: CourtViewPolicy
    sampling: CourtSamplingPolicy
    performance: DatasetPerformanceBudget
    metadata_fields: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> CourtDatasetConfiguration:
        raw = _exact(
            value,
            path="dataset.court",
            keys={
                "schema_version",
                "trajectory",
                "view",
                "sampling",
                "performance",
                "metadata_fields",
            },
        )
        metadata = _text_sequence(raw, "metadata_fields", path="dataset.court")
        if not _COURT_METADATA_FIELDS.issubset(metadata):
            raise SemanticConfigurationError(
                "dataset.court.metadata_fields omits required provenance."
            )
        schema_version = _enum_value(
            raw,
            "schema_version",
            path="dataset.court",
            enum_type=CourtDatasetSchemaVersion,
        )
        trajectory = CourtTrajectoryPolicy.from_mapping(raw["trajectory"])
        view = CourtViewPolicy.from_mapping(
            raw["view"],
            schema_version=schema_version,
        )
        sampling = CourtSamplingPolicy.from_mapping(raw["sampling"])
        performance = _performance_budget(
            raw["performance"],
            path="dataset.court.performance",
        )
        if (
            not performance.require_cuda
            or not performance.execution_device.startswith("cuda")
            or performance.maximum_nht_invocations > sampling.shard_group_count
            or performance.maximum_complete_array_scans_per_sample > 2
        ):
            raise SemanticConfigurationError(
                "Court production performance must require CUDA, fit the resolved "
                "shard count, and permit at most two complete scans per sample."
            )
        return cls(
            schema_version=schema_version,
            trajectory=trajectory,
            view=view,
            sampling=sampling,
            performance=performance,
            metadata_fields=metadata,
        )


@dataclass(frozen=True, slots=True)
class ScenePipelineConfiguration:
    """Resolved canonical request and every config-owned stage/domain policy."""

    profile: str
    resolver: PathResolver
    workspace: SceneWorkspace
    request: ScenePipelineRequest
    stages: PipelineStageSettings
    nht: NHTCommandPaths
    alignment: AlignmentConfiguration
    court: CourtDatasetConfiguration

    @classmethod
    def from_config(cls, value: object) -> ScenePipelineConfiguration:
        root = _exact(
            value,
            path="configuration",
            keys={
                "profile",
                "roots",
                "request",
                "pipeline",
                "nht",
                "alignment",
                "dataset",
            },
        )
        roots = RuntimePathRoots.from_mapping(
            _mapping(root["roots"], path="roots"),
            repository_root=PROJECT_ROOT,
        )
        resolver = PathResolver(roots)
        stages = PipelineStageSettings.from_mapping(root["pipeline"])
        request_raw = _exact(
            root["request"],
            path="request",
            keys={
                "scene_id",
                "source_video",
                "targets",
                "from_stage",
                "through_stage",
            },
        )
        target_values = _text_sequence(request_raw, "targets", path="request")
        try:
            targets = tuple(DatasetTarget(item) for item in target_values)
            from_stage = StageName(_text(request_raw, "from_stage", path="request"))
            through_stage = StageName(
                _text(request_raw, "through_stage", path="request")
            )
        except ValueError as error:
            raise SemanticConfigurationError(
                f"request contains an unknown target or stage: {error}"
            ) from error
        if len(targets) != len(set(targets)):
            raise SemanticConfigurationError(
                "request.targets must not contain duplicates."
            )
        source_video = resolver.resolve(
            PathRole.DATA,
            _text(request_raw, "source_video", path="request"),
        )
        request = ScenePipelineRequest(
            scene_id=_text(request_raw, "scene_id", path="request"),
            source_video=source_video,
            targets=frozenset(targets),
            from_stage=from_stage,
            through_stage=through_stage,
            config_schema=stages.config_schema,
        )
        dataset = _exact(root["dataset"], path="dataset", keys={"court"})
        return cls(
            profile=_text(root, "profile", path="configuration"),
            resolver=resolver,
            workspace=SceneWorkspace.resolve(resolver, request.scene_id),
            request=request,
            stages=stages,
            nht=NHTCommandPaths.from_mapping(root["nht"], resolver=resolver),
            alignment=AlignmentConfiguration.from_mapping(
                root["alignment"],
                resolver=resolver,
            ),
            court=CourtDatasetConfiguration.from_mapping(dataset["court"]),
        )


def validate_scene_pipeline_boundary(config: DictConfig) -> None:
    """Fail closed before the canonical CLI performs any mutation."""
    ScenePipelineConfiguration.from_config(config)


register_boundary_validator(SCENE_PIPELINE_BOUNDARY, validate_scene_pipeline_boundary)

__all__ = [
    "AlignmentConfiguration",
    "CourtDatasetConfiguration",
    "CourtDatasetSchemaVersion",
    "CourtSamplingPolicy",
    "CourtTrajectoryPolicy",
    "CourtViewPolicy",
    "NHTCommandPaths",
    "PipelineStageSettings",
    "SCENE_PIPELINE_BOUNDARY",
    "SCENE_PIPELINE_SCHEMA",
    "ScenePipelineConfiguration",
    "validate_scene_pipeline_boundary",
]
