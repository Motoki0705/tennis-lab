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
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    PartitionThresholds,
)
from src.synthetic_data_generation.alignment.settings import (
    AlignmentEvidenceSettings,
    CorrespondenceSettings,
    CourtCandidateFitSettings,
    CourtLineArchitectureSettings,
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinateFrame,
    GaussianCoordinates,
    GaussianUnit,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.dataset.blcs.source import (
    BLCSTrajectorySourceSettings,
)
from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
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
)
from src.synthetic_data_generation.rendering.nht.contracts import NHT_RENDER_COMMAND
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyConfig
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionCategory
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
from src.utils.projection.camera_projector import CameraConfig
from src.utils.schema.court import CourtConfig

if TYPE_CHECKING:
    import torch

    from src.synthetic_data_generation.dataset.plcs.composition import AvatarAppearance
    from src.synthetic_data_generation.dataset.plcs.handler import (
        PLCSObjectRequest,
        PLCSStageParameters,
    )
    from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip

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
_BLCS_METADATA_FIELDS = _COURT_METADATA_FIELDS | frozenset(
    {"source_frame", "source_trajectory"}
)
_PLCS_METADATA_FIELDS = _COURT_METADATA_FIELDS | frozenset(
    {"motion_source", "source_frame"}
)

ConfigMapping = Mapping[str, object]


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
    values = _sequence(_value(mapping, key, (list, tuple), path=path), path=f"{path}.{key}")
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


def _number_sequence(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    minimum_length: int = 1,
) -> tuple[float, ...]:
    values = _sequence(_value(mapping, key, (list, tuple), path=path), path=f"{path}.{key}")
    if len(values) < minimum_length or any(type(item) not in (int, float) for item in values):
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


def _camera_profile(value: object) -> CameraProfileConfig:
    profile = CameraProfileConfig.from_mapping(_mapping(value, path="camera"))
    for slot in profile.slots:
        if slot.height_m[0] <= 0.0:
            raise SemanticConfigurationError(
                f"camera slot {slot.slot_id!r} height must be positive."
            )
        if not 0.0 < slot.hfov_degrees[0] <= slot.hfov_degrees[1] < 180.0:
            raise SemanticConfigurationError(
                f"camera slot {slot.slot_id!r} HFOV must stay within (0, 180)."
            )
        if slot.look_at_height_m[0] < 0.0:
            raise SemanticConfigurationError(
                f"camera slot {slot.slot_id!r} look-at height must be non-negative."
            )
    return profile


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
            invalidate_descendants=_flag(raw, "invalidate_descendants", path="pipeline"),
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
    environment: Mapping[str, str]
    reconstruction_timeout_seconds: float
    render_timeout_seconds: float

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> NHTCommandPaths:
        raw = _exact(
            value,
            path="nht",
            keys={
                "reconstruct_executable",
                "render_executable",
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
        environment_raw = _mapping(raw["environment"], path="nht.environment")
        unknown_environment = sorted(
            set(environment_raw) - {"CUDA_VISIBLE_DEVICES"}
        )
        if unknown_environment:
            raise UnknownConfigurationKeyError(
                "Unknown NHT public environment key(s): "
                + ", ".join(
                    f"nht.environment.{key}" for key in unknown_environment
                )
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
            raise SemanticConfigurationError("NHT subprocess timeouts must be positive.")
        return cls(
            reconstruct_executable=reconstruct,
            render_executable=render,
            environment=environment,
            reconstruction_timeout_seconds=reconstruction_timeout,
            render_timeout_seconds=render_timeout,
        )


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
            transform_inverse_atol=_number(raw, "transform_inverse_atol", path="alignment"),
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
                "maximum_cameras",
                "line_model",
                "ground_plane",
                "projection",
                "candidate_fit",
                "correspondences",
            },
        )
        line_raw = _exact(
            raw["line_model"],
            path=f"{path}.line_model",
            keys={
                "checkpoint_path",
                "backbone_repository_path",
                "backbone_checkpoint_path",
                "device",
                "expected_short_side",
                "probability_threshold",
                "maximum_selected_pixels_per_camera",
                "architecture",
            },
        )
        architecture_path = f"{path}.line_model.architecture"
        architecture_raw = _exact(
            line_raw["architecture"],
            path=architecture_path,
            keys={
                "backbone_name",
                "backbone_strict",
                "backbone_train_mode",
                "backbone_last_n_blocks",
                "backbone_out_indices",
                "backbone_layer_mode",
                "lora_enabled",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_target_modules",
                "decoder_channels",
                "decoder_reassemble_factors",
                "line_bce_weight",
                "line_dice_weight",
                "line_positive_weight",
            },
        )
        out_indices = _sequence(
            _value(
                architecture_raw,
                "backbone_out_indices",
                (list, tuple),
                path=architecture_path,
            ),
            path=f"{architecture_path}.backbone_out_indices",
        )
        if len(out_indices) != 4 or any(type(item) is not int for item in out_indices):
            raise ConfigurationTypeError(
                f"{architecture_path}.backbone_out_indices must contain four integers."
            )
        architecture = CourtLineArchitectureSettings(
            backbone_name=_text(architecture_raw, "backbone_name", path=architecture_path),
            backbone_strict=_flag(architecture_raw, "backbone_strict", path=architecture_path),
            backbone_train_mode=_text(
                architecture_raw, "backbone_train_mode", path=architecture_path
            ),
            backbone_last_n_blocks=_integer(
                architecture_raw,
                "backbone_last_n_blocks",
                path=architecture_path,
                minimum=0,
            ),
            backbone_out_indices=tuple(cast(int, item) for item in out_indices),
            backbone_layer_mode=_text(
                architecture_raw, "backbone_layer_mode", path=architecture_path
            ),
            lora_enabled=_flag(architecture_raw, "lora_enabled", path=architecture_path),
            lora_rank=_integer(
                architecture_raw, "lora_rank", path=architecture_path, minimum=1
            ),
            lora_alpha=_number(architecture_raw, "lora_alpha", path=architecture_path),
            lora_dropout=_number(
                architecture_raw, "lora_dropout", path=architecture_path
            ),
            lora_target_modules=_text_sequence(
                architecture_raw, "lora_target_modules", path=architecture_path
            ),
            decoder_channels=_integer(
                architecture_raw,
                "decoder_channels",
                path=architecture_path,
                minimum=1,
            ),
            decoder_reassemble_factors=_number_sequence(
                architecture_raw,
                "decoder_reassemble_factors",
                path=architecture_path,
                minimum_length=4,
            ),
            line_bce_weight=_number(
                architecture_raw, "line_bce_weight", path=architecture_path
            ),
            line_dice_weight=_number(
                architecture_raw, "line_dice_weight", path=architecture_path
            ),
            line_positive_weight=_number(
                architecture_raw, "line_positive_weight", path=architecture_path
            ),
        )
        line_path = f"{path}.line_model"
        line_model = CourtLineModelSettings(
            checkpoint_path=resolver.resolve(
                PathRole.CHECKPOINT,
                _text(line_raw, "checkpoint_path", path=line_path),
            ),
            backbone_repository_path=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _text(line_raw, "backbone_repository_path", path=line_path),
            ),
            backbone_checkpoint_path=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _text(line_raw, "backbone_checkpoint_path", path=line_path),
            ),
            device=_text(line_raw, "device", path=line_path),
            expected_short_side=_integer(
                line_raw, "expected_short_side", path=line_path, minimum=1
            ),
            probability_threshold=_number(
                line_raw, "probability_threshold", path=line_path
            ),
            maximum_selected_pixels_per_camera=_integer(
                line_raw,
                "maximum_selected_pixels_per_camera",
                path=line_path,
                minimum=1,
            ),
            architecture=architecture,
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
            footprint_quantile=_number(ground_raw, "footprint_quantile", path=ground_path),
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
            bounds_margin=_number(projection_raw, "bounds_margin", path=projection_path),
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
                "candidate_count",
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
                "separation_penalty",
                "optimizer_maximum_iterations",
                "optimizer_population_size",
                "optimizer_tolerance",
                "maximum_fit_points",
                "common_scale_relative_tolerance",
            },
        )
        candidate = CourtCandidateFitSettings(
            candidate_count=_integer(
                candidate_raw, "candidate_count", path=candidate_path, minimum=1
            ),
            samples_per_metre=_number(candidate_raw, "samples_per_metre", path=candidate_path),
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
            separation_penalty=_number(
                candidate_raw, "separation_penalty", path=candidate_path
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
            maximum_cameras=_integer(raw, "maximum_cameras", path=path, minimum=1),
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
                inlier_distance_m=_number(partition_raw, "inlier_distance_m", path=path),
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

    shapes: tuple[str, ...]
    axis_ratios: tuple[float, ...]
    orientations_degrees: tuple[float, ...]
    center_kinds: tuple[str, ...]
    captured_offset_scale_range: tuple[float, float]
    base_heights_m: tuple[float, ...]
    vertical_modulations_m: tuple[float, ...]
    curve_modes: tuple[str, ...]

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
            },
        )
        path = "dataset.court.trajectory"
        result = cls(
            shapes=_text_sequence(raw, "shapes", path=path),
            axis_ratios=_number_sequence(raw, "axis_ratios", path=path),
            orientations_degrees=_number_sequence(
                raw, "orientations_degrees", path=path, minimum_length=3
            ),
            center_kinds=_text_sequence(raw, "center_kinds", path=path),
            captured_offset_scale_range=_ordered_range(
                raw, "captured_offset_scale_range", path=path, positive=True
            ),
            base_heights_m=_number_sequence(raw, "base_heights_m", path=path, minimum_length=3),
            vertical_modulations_m=_number_sequence(
                raw, "vertical_modulations_m", path=path
            ),
            curve_modes=_text_sequence(raw, "curve_modes", path=path),
        )
        if set(result.shapes) != {"circle", "ellipse"}:
            raise SemanticConfigurationError("Court trajectory shapes must be circle and ellipse.")
        if 1.0 not in result.axis_ratios or not any(ratio <= 0.8 for ratio in result.axis_ratios):
            raise SemanticConfigurationError(
                "Court trajectory axis ratios require circle=1 and an ellipse <= 0.8."
            )
        if any(not 0.0 < ratio <= 1.0 for ratio in result.axis_ratios):
            raise SemanticConfigurationError("Court trajectory axis ratios must be within (0, 1].")
        if not {0.0, 45.0, 90.0}.issubset(result.orientations_degrees):
            raise SemanticConfigurationError("Court orientations must include 0, 45, and 90 degrees.")
        if set(result.center_kinds) != {"complex", "court"}:
            raise SemanticConfigurationError("Court center kinds must be complex and court.")
        if len(set(result.base_heights_m)) < 3 or min(result.base_heights_m) <= 0.0:
            raise SemanticConfigurationError("Court base heights require three positive levels.")
        if min(result.vertical_modulations_m) < 0.0 or not any(
            value > 0.0 for value in result.vertical_modulations_m
        ):
            raise SemanticConfigurationError("Court vertical modulation requires a positive value.")
        if "sinusoidal_height" not in result.curve_modes:
            raise SemanticConfigurationError("Court trajectories require a smooth non-planar mode.")
        return result


@dataclass(frozen=True, slots=True)
class CourtViewPolicy:
    """Typed camera-target and coverage policy."""

    target_modes: tuple[str, ...]
    coverage_modes: tuple[str, ...]
    look_at_height_m: tuple[float, float]
    hfov_degrees: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object) -> CourtViewPolicy:
        raw = _exact(
            value,
            path="dataset.court.view",
            keys={"target_modes", "coverage_modes", "look_at_height_m", "hfov_degrees"},
        )
        path = "dataset.court.view"
        result = cls(
            target_modes=_text_sequence(raw, "target_modes", path=path),
            coverage_modes=_text_sequence(raw, "coverage_modes", path=path),
            look_at_height_m=_ordered_range(raw, "look_at_height_m", path=path, positive=False),
            hfov_degrees=_ordered_range(raw, "hfov_degrees", path=path, positive=True),
        )
        if set(result.coverage_modes) != {"full", "near_full", "partial"}:
            raise SemanticConfigurationError(
                "Court coverage modes must be full, near_full, and partial."
            )
        if result.look_at_height_m[0] < 0.0:
            raise SemanticConfigurationError("Court look-at heights must be non-negative.")
        if result.hfov_degrees[1] >= 180.0:
            raise SemanticConfigurationError("Court HFOV range must stay below 180 degrees.")
        return result


@dataclass(frozen=True, slots=True)
class CourtSamplingPolicy:
    """Deterministic coverage selection, budget, split, and release gates."""

    seed: int
    stable_field_order: tuple[str, ...]
    coverage_objective: tuple[str, ...]
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
            seed=_integer(raw, "seed", path=path, minimum=0),
            stable_field_order=_text_sequence(raw, "stable_field_order", path=path),
            coverage_objective=_text_sequence(raw, "coverage_objective", path=path),
            proposal_budget=_integer(raw, "proposal_budget", path=path, minimum=1),
            minimum_trajectory_groups=_integer(
                raw, "minimum_trajectory_groups", path=path, minimum=1
            ),
            minimum_accepted_frames=_integer(
                raw, "minimum_accepted_frames", path=path, minimum=1
            ),
            maximum_adjacent_step_m=_number(raw, "maximum_adjacent_step_m", path=path),
            minimum_accepted_fraction=_number(raw, "minimum_accepted_fraction", path=path),
            train_fraction=_number(raw, "train_fraction", path=path),
            validation_fraction=_number(raw, "validation_fraction", path=path),
            test_fraction=_number(raw, "test_fraction", path=path),
            shard_group_count=_integer(raw, "shard_group_count", path=path, minimum=1),
        )
        if result.proposal_budget != 4_800:
            raise SemanticConfigurationError("B00 Court proposal_budget must be exactly 4,800.")
        if result.minimum_trajectory_groups < 24:
            raise SemanticConfigurationError("Court production requires at least 24 groups.")
        if result.minimum_accepted_frames < 2_000:
            raise SemanticConfigurationError("Court production requires at least 2,000 frames.")
        if not 0.0 < result.maximum_adjacent_step_m <= 1.05:
            raise SemanticConfigurationError("Court adjacent arc step must be within (0, 1.05].")
        if not 0.9 <= result.minimum_accepted_fraction <= 1.0:
            raise SemanticConfigurationError("Court accepted fraction must be within [0.9, 1].")
        fractions = (result.train_fraction, result.validation_fraction, result.test_fraction)
        if min(fractions) <= 0.0 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
            raise SemanticConfigurationError("Court split fractions must be positive and sum to 1.")
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
        trajectory = CourtTrajectoryPolicy.from_mapping(raw["trajectory"])
        view = CourtViewPolicy.from_mapping(raw["view"])
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
            trajectory=trajectory,
            view=view,
            sampling=sampling,
            performance=performance,
            metadata_fields=metadata,
        )


@dataclass(frozen=True, slots=True)
class FullFrameChunkPolicy:
    """Full source/global timeline and transactional chunk policy."""

    frame_selection: str
    chunk_size_frames: int
    require_contiguous_frame_indices: bool
    require_exact_frame_inventory: bool
    reuse_shards_within_stage_attempt: bool
    discard_shards_on_rerun: bool

    @classmethod
    def from_mapping(cls, value: object, *, path: str) -> FullFrameChunkPolicy:
        raw = _exact(
            value,
            path=path,
            keys={
                "frame_selection",
                "chunk_size_frames",
                "require_contiguous_frame_indices",
                "require_exact_frame_inventory",
                "reuse_shards_within_stage_attempt",
                "discard_shards_on_rerun",
            },
        )
        result = cls(
            frame_selection=_text(raw, "frame_selection", path=path),
            chunk_size_frames=_integer(raw, "chunk_size_frames", path=path, minimum=1),
            require_contiguous_frame_indices=_flag(
                raw, "require_contiguous_frame_indices", path=path
            ),
            require_exact_frame_inventory=_flag(
                raw, "require_exact_frame_inventory", path=path
            ),
            reuse_shards_within_stage_attempt=_flag(
                raw, "reuse_shards_within_stage_attempt", path=path
            ),
            discard_shards_on_rerun=_flag(raw, "discard_shards_on_rerun", path=path),
        )
        if result.frame_selection != "all_source_frames":
            raise SemanticConfigurationError(f"{path}.frame_selection must select all source frames.")
        for name in (
            "require_contiguous_frame_indices",
            "require_exact_frame_inventory",
            "reuse_shards_within_stage_attempt",
            "discard_shards_on_rerun",
        ):
            _require_true(cast(bool, getattr(result, name)), path=f"{path}.{name}")
        return result


def _fixed_number_tuple(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    length: int,
) -> tuple[float, ...]:
    values = _number_sequence(mapping, key, path=path, minimum_length=length)
    if len(values) != length:
        raise ConfigurationTypeError(
            f"{path}.{key} must contain exactly {length} numeric values."
        )
    return values


def _fixed_integer_tuple(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
    length: int,
) -> tuple[int, ...]:
    values = _sequence(
        _value(mapping, key, (list, tuple), path=path),
        path=f"{path}.{key}",
    )
    if len(values) != length or any(type(item) is not int for item in values):
        raise ConfigurationTypeError(
            f"{path}.{key} must contain exactly {length} integer values."
        )
    return tuple(cast(int, item) for item in values)


def _optional_number_pair(
    mapping: ConfigMapping,
    key: str,
    *,
    path: str,
) -> tuple[float, float] | None:
    value = _value(mapping, key, (list, tuple, type(None)), path=path)
    if value is None:
        return None
    values = _sequence(value, path=f"{path}.{key}")
    if len(values) != 2 or any(type(item) not in (int, float) for item in values):
        raise ConfigurationTypeError(
            f"{path}.{key} must be null or contain exactly two numeric values."
        )
    result = cast(
        tuple[float, float],
        tuple(float(cast("int | float", item)) for item in values),
    )
    if any(not math.isfinite(item) for item in result) or result[0] > result[1]:
        raise SemanticConfigurationError(
            f"{path}.{key} must be null or a finite ordered range."
        )
    return result


def _blcs_source_settings(value: object) -> BLCSTrajectorySourceSettings:
    from src.synthetic_data_generation.dataset.blcs.source import (
        BLCSTrajectorySourceSettings,
    )

    path = "dataset.blcs.trajectory_source"
    raw = _exact(
        value,
        path=path,
        keys={
            "scene_count",
            "split_scene_counts",
            "multi_object",
            "maximum_physics_attempts_per_object",
            "timeline",
            "device",
        },
    )
    split_path = f"{path}.split_scene_counts"
    split_raw = _exact(
        raw["split_scene_counts"],
        path=split_path,
        keys={"train", "validation", "test"},
    )
    counts = {
        split: _integer(split_raw, split, path=split_path, minimum=0)
        for split in ("train", "validation", "test")
    }
    timeline_path = f"{path}.timeline"
    timeline_raw = _exact(
        raw["timeline"],
        path=timeline_path,
        keys={
            "num_frames",
            "min_tracks",
            "max_tracks",
            "max_concurrent",
            "min_reuse_gap_frames",
            "start_index_range",
            "min_active_frames",
            "overlap_probability",
            "min_gap_frames",
            "max_gap_frames",
        },
    )
    start_range = _fixed_integer_tuple(
        timeline_raw,
        "start_index_range",
        path=timeline_path,
        length=2,
    )
    timeline = TimelineConfig(
        num_frames=_integer(timeline_raw, "num_frames", path=timeline_path, minimum=1),
        min_tracks=_integer(timeline_raw, "min_tracks", path=timeline_path, minimum=1),
        max_tracks=_integer(timeline_raw, "max_tracks", path=timeline_path, minimum=1),
        max_concurrent=_integer(
            timeline_raw, "max_concurrent", path=timeline_path, minimum=1
        ),
        min_reuse_gap_frames=_integer(
            timeline_raw, "min_reuse_gap_frames", path=timeline_path, minimum=0
        ),
        start_index_range=cast(tuple[int, int], start_range),
        min_active_frames=_integer(
            timeline_raw, "min_active_frames", path=timeline_path, minimum=1
        ),
        overlap_probability=_number(
            timeline_raw, "overlap_probability", path=timeline_path
        ),
        min_gap_frames=_integer(
            timeline_raw, "min_gap_frames", path=timeline_path, minimum=0
        ),
        max_gap_frames=_integer(
            timeline_raw, "max_gap_frames", path=timeline_path, minimum=0
        ),
    )
    return BLCSTrajectorySourceSettings(
        scene_count=_integer(raw, "scene_count", path=path, minimum=1),
        split_scene_counts=counts,
        multi_object=_flag(raw, "multi_object", path=path),
        maximum_physics_attempts_per_object=_integer(
            raw,
            "maximum_physics_attempts_per_object",
            path=path,
            minimum=1,
        ),
        timeline=timeline,
        device=_text(raw, "device", path=path),
    )


def _blcs_generator_config(value: object) -> GeneratorConfig:
    path = "dataset.blcs.generator"
    raw = _exact(
        value,
        path=path,
        keys={"physics", "rally", "camera", "targeted_velocity", "court"},
    )
    physics_path = f"{path}.physics"
    physics_raw = _exact(
        raw["physics"],
        path=physics_path,
        keys={
            "gravity",
            "k_drag",
            "k_magnus",
            "e_z",
            "mu",
            "alpha_net",
            "alpha_net_cord",
            "alpha_fence",
            "net_half_thickness",
            "net_cord_radius",
            "dt",
            "use_drag",
            "use_magnus",
            "wind",
            "gravity_range",
            "k_drag_range",
            "k_magnus_range",
            "e_z_range",
            "mu_range",
            "wind_speed_range",
            "wind_direction_range_deg",
        },
    )
    physics = PhysicsConfig(
        gravity=_number(physics_raw, "gravity", path=physics_path),
        k_drag=_number(physics_raw, "k_drag", path=physics_path),
        k_magnus=_number(physics_raw, "k_magnus", path=physics_path),
        e_z=_number(physics_raw, "e_z", path=physics_path),
        mu=_number(physics_raw, "mu", path=physics_path),
        alpha_net=_number(physics_raw, "alpha_net", path=physics_path),
        alpha_net_cord=_number(physics_raw, "alpha_net_cord", path=physics_path),
        alpha_fence=_number(physics_raw, "alpha_fence", path=physics_path),
        net_half_thickness=_number(
            physics_raw, "net_half_thickness", path=physics_path
        ),
        net_cord_radius=_number(physics_raw, "net_cord_radius", path=physics_path),
        dt=_number(physics_raw, "dt", path=physics_path),
        use_drag=_flag(physics_raw, "use_drag", path=physics_path),
        use_magnus=_flag(physics_raw, "use_magnus", path=physics_path),
        wind=cast(
            tuple[float, float, float],
            _fixed_number_tuple(physics_raw, "wind", path=physics_path, length=3),
        ),
        gravity_range=_optional_number_pair(
            physics_raw, "gravity_range", path=physics_path
        ),
        k_drag_range=_optional_number_pair(
            physics_raw, "k_drag_range", path=physics_path
        ),
        k_magnus_range=_optional_number_pair(
            physics_raw, "k_magnus_range", path=physics_path
        ),
        e_z_range=_optional_number_pair(physics_raw, "e_z_range", path=physics_path),
        mu_range=_optional_number_pair(physics_raw, "mu_range", path=physics_path),
        wind_speed_range=_optional_number_pair(
            physics_raw, "wind_speed_range", path=physics_path
        ),
        wind_direction_range_deg=_optional_number_pair(
            physics_raw, "wind_direction_range_deg", path=physics_path
        ),
    )
    rally_path = f"{path}.rally"
    rally_raw = _exact(
        raw["rally"],
        path=rally_path,
        keys={
            "z_range",
            "spin_x_range",
            "spin_y_range",
            "spin_z_range",
            "max_sim_frames",
            "output_fps",
            "sim_fps",
            "max_rallies",
            "max_total_frames",
            "hit_timing_range",
            "return_z_range",
            "serve_probability",
            "serve_z_range",
            "toss_vz_range",
            "toss_xy_noise_range",
            "toss_max_frames",
            "toss_z0_tolerance",
            "volley_probability",
            "normal_return_probability",
            "late_return_probability",
            "out_court_target_probability",
        },
    )

    def rally_pair(key: str) -> tuple[float, float]:
        return cast(
            tuple[float, float],
            _fixed_number_tuple(rally_raw, key, path=rally_path, length=2),
        )

    rally = RallyConfig(
        z_range=rally_pair("z_range"),
        spin_x_range=rally_pair("spin_x_range"),
        spin_y_range=rally_pair("spin_y_range"),
        spin_z_range=rally_pair("spin_z_range"),
        max_sim_frames=_integer(
            rally_raw, "max_sim_frames", path=rally_path, minimum=1
        ),
        output_fps=_integer(rally_raw, "output_fps", path=rally_path, minimum=1),
        sim_fps=_integer(rally_raw, "sim_fps", path=rally_path, minimum=1),
        max_rallies=_integer(rally_raw, "max_rallies", path=rally_path, minimum=1),
        max_total_frames=_integer(
            rally_raw, "max_total_frames", path=rally_path, minimum=1
        ),
        hit_timing_range=rally_pair("hit_timing_range"),
        return_z_range=rally_pair("return_z_range"),
        serve_probability=_number(rally_raw, "serve_probability", path=rally_path),
        serve_z_range=rally_pair("serve_z_range"),
        toss_vz_range=rally_pair("toss_vz_range"),
        toss_xy_noise_range=rally_pair("toss_xy_noise_range"),
        toss_max_frames=_integer(
            rally_raw, "toss_max_frames", path=rally_path, minimum=1
        ),
        toss_z0_tolerance=_number(
            rally_raw, "toss_z0_tolerance", path=rally_path
        ),
        volley_probability=_number(
            rally_raw, "volley_probability", path=rally_path
        ),
        normal_return_probability=_number(
            rally_raw, "normal_return_probability", path=rally_path
        ),
        late_return_probability=_number(
            rally_raw, "late_return_probability", path=rally_path
        ),
        out_court_target_probability=_number(
            rally_raw, "out_court_target_probability", path=rally_path
        ),
    )
    camera_path = f"{path}.camera"
    camera_raw = _exact(
        raw["camera"],
        path=camera_path,
        keys={
            "z_min",
            "z_max",
            "hfov_deg",
            "image_size",
            "fixed_look_at",
            "fixed_baseline_clear_extra",
            "fixed_position_noise_radius",
            "fixed_look_at_xy_radius",
            "layout",
            "broadcast_setback",
            "broadcast_height",
            "broadcast_hfov_deg",
            "broadcast_look_at_y",
            "broadcast_look_at_height",
            "broadcast_position_noise_radius",
            "broadcast_look_at_xy_radius",
            "broadcast_hfov_jitter_deg",
            "broadcast_setback_range",
            "broadcast_height_range",
            "broadcast_court_width_frac_range",
        },
    )
    image_size = _fixed_integer_tuple(
        camera_raw, "image_size", path=camera_path, length=2
    )
    camera = CameraConfig(
        z_min=_number(camera_raw, "z_min", path=camera_path),
        z_max=_number(camera_raw, "z_max", path=camera_path),
        hfov_deg=_number(camera_raw, "hfov_deg", path=camera_path),
        image_size=cast(tuple[int, int], image_size),
        fixed_look_at=cast(
            tuple[float, float, float],
            _fixed_number_tuple(
                camera_raw, "fixed_look_at", path=camera_path, length=3
            ),
        ),
        fixed_baseline_clear_extra=_number(
            camera_raw, "fixed_baseline_clear_extra", path=camera_path
        ),
        fixed_position_noise_radius=_number(
            camera_raw, "fixed_position_noise_radius", path=camera_path
        ),
        fixed_look_at_xy_radius=_number(
            camera_raw, "fixed_look_at_xy_radius", path=camera_path
        ),
        layout=_text(camera_raw, "layout", path=camera_path),
        broadcast_setback=_number(camera_raw, "broadcast_setback", path=camera_path),
        broadcast_height=_number(camera_raw, "broadcast_height", path=camera_path),
        broadcast_hfov_deg=_number(
            camera_raw, "broadcast_hfov_deg", path=camera_path
        ),
        broadcast_look_at_y=_number(
            camera_raw, "broadcast_look_at_y", path=camera_path
        ),
        broadcast_look_at_height=_number(
            camera_raw, "broadcast_look_at_height", path=camera_path
        ),
        broadcast_position_noise_radius=_number(
            camera_raw, "broadcast_position_noise_radius", path=camera_path
        ),
        broadcast_look_at_xy_radius=_number(
            camera_raw, "broadcast_look_at_xy_radius", path=camera_path
        ),
        broadcast_hfov_jitter_deg=_number(
            camera_raw, "broadcast_hfov_jitter_deg", path=camera_path
        ),
        broadcast_setback_range=_optional_number_pair(
            camera_raw, "broadcast_setback_range", path=camera_path
        ),
        broadcast_height_range=_optional_number_pair(
            camera_raw, "broadcast_height_range", path=camera_path
        ),
        broadcast_court_width_frac_range=_optional_number_pair(
            camera_raw, "broadcast_court_width_frac_range", path=camera_path
        ),
    )
    target_path = f"{path}.targeted_velocity"
    target_raw = _exact(
        raw["targeted_velocity"],
        path=target_path,
        keys={
            "drive_elevation_range_deg",
            "lob_elevation_range_deg",
            "lob_probability",
            "max_ballistic_apex_height_m",
            "gravity",
            "net_elevation_step_deg",
            "landing_refine_enabled",
            "landing_refine_max_iters",
            "landing_refine_tolerance_m",
            "landing_sim_max_frames",
            "target_margin_m",
        },
    )

    def target_pair(key: str) -> tuple[float, float]:
        return cast(
            tuple[float, float],
            _fixed_number_tuple(target_raw, key, path=target_path, length=2),
        )

    targeted = TargetedVelocityConfig(
        drive_elevation_range_deg=target_pair("drive_elevation_range_deg"),
        lob_elevation_range_deg=target_pair("lob_elevation_range_deg"),
        lob_probability=_number(target_raw, "lob_probability", path=target_path),
        max_ballistic_apex_height_m=_number(
            target_raw, "max_ballistic_apex_height_m", path=target_path
        ),
        gravity=_number(target_raw, "gravity", path=target_path),
        net_elevation_step_deg=_number(
            target_raw, "net_elevation_step_deg", path=target_path
        ),
        landing_refine_enabled=_flag(
            target_raw, "landing_refine_enabled", path=target_path
        ),
        landing_refine_max_iters=_integer(
            target_raw, "landing_refine_max_iters", path=target_path, minimum=1
        ),
        landing_refine_tolerance_m=_number(
            target_raw, "landing_refine_tolerance_m", path=target_path
        ),
        landing_sim_max_frames=_integer(
            target_raw, "landing_sim_max_frames", path=target_path, minimum=1
        ),
        target_margin_m=_number(target_raw, "target_margin_m", path=target_path),
    )
    court_path = f"{path}.court"
    court_raw = _exact(
        raw["court"],
        path=court_path,
        keys={"net_post_offset_x", "net_post_offset_x_range"},
    )
    court = CourtConfig(
        net_post_offset_x=_number(court_raw, "net_post_offset_x", path=court_path),
        net_post_offset_x_range=_optional_number_pair(
            court_raw, "net_post_offset_x_range", path=court_path
        ),
    )
    return GeneratorConfig(
        physics=physics,
        rally=rally,
        camera=camera,
        targeted_velocity=targeted,
        court=court,
    )


def _gaussian_asset(value: object, *, path: str) -> GaussianAsset:
    raw = _exact(
        value,
        path=path,
        keys={
            "schema",
            "asset_id",
            "asset_class",
            "role",
            "coordinates",
            "gaussian_count",
            "feature_dim",
            "floating_dtype",
            "appearance_model",
            "appearance_space",
            "tensor_encoding",
        },
    )
    coordinates_path = f"{path}.coordinates"
    coordinates_raw = _exact(
        raw["coordinates"],
        path=coordinates_path,
        keys={"frame", "unit", "convention"},
    )
    try:
        role = GaussianAssetRole(_text(raw, "role", path=path))
        frame = GaussianCoordinateFrame(
            _text(coordinates_raw, "frame", path=coordinates_path)
        )
        unit = GaussianUnit(_text(coordinates_raw, "unit", path=coordinates_path))
    except ValueError as error:
        raise SemanticConfigurationError(f"{path} contains an unknown Gaussian enum.") from error
    dtype = _text(raw, "floating_dtype", path=path)
    if dtype not in {"float32", "float64"}:
        raise SemanticConfigurationError(f"{path}.floating_dtype is unsupported.")
    return GaussianAsset(
        schema=_text(raw, "schema", path=path),
        asset_id=_text(raw, "asset_id", path=path),
        asset_class=_text(raw, "asset_class", path=path),
        role=role,
        coordinates=GaussianCoordinates(
            frame=frame,
            unit=unit,
            convention=_text(coordinates_raw, "convention", path=coordinates_path),
        ),
        gaussian_count=_integer(raw, "gaussian_count", path=path, minimum=1),
        feature_dim=_integer(raw, "feature_dim", path=path, minimum=1),
        floating_dtype=cast(Literal["float32", "float64"], dtype),
        appearance_model=_text(raw, "appearance_model", path=path),
        appearance_space=_text(raw, "appearance_space", path=path),
        tensor_encoding=_text(raw, "tensor_encoding", path=path),
    )


def _blcs_assets(value: object) -> BLCSCompositionAssets:
    from src.synthetic_data_generation.dataset.blcs.contracts import (
        BLCSCompositionAssets,
    )

    path = "dataset.blcs.assets"
    raw = _exact(
        value,
        path=path,
        keys={"background", "ball", "ball_radius_m"},
    )
    return BLCSCompositionAssets(
        background=_gaussian_asset(raw["background"], path=f"{path}.background"),
        ball=_gaussian_asset(raw["ball"], path=f"{path}.ball"),
        ball_radius_m=_number(raw, "ball_radius_m", path=path),
    )


@dataclass(frozen=True, slots=True)
class BLCSDatasetConfiguration:
    """Complete BLCS physics, semantic asset, render, and timeline authority."""

    timeline: FullFrameChunkPolicy
    trajectory_source: BLCSTrajectorySourceSettings
    generator: GeneratorConfig
    assets: BLCSCompositionAssets
    render_timeout_seconds: float
    performance: DatasetPerformanceBudget
    metadata_fields: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> BLCSDatasetConfiguration:
        raw = _exact(
            value,
            path="dataset.blcs",
            keys={
                "timeline",
                "trajectory_source",
                "generator",
                "assets",
                "render_timeout_seconds",
                "performance",
                "metadata_fields",
            },
        )
        metadata = _text_sequence(raw, "metadata_fields", path="dataset.blcs")
        if not _BLCS_METADATA_FIELDS.issubset(metadata):
            raise SemanticConfigurationError("dataset.blcs.metadata_fields omits required provenance.")
        source = _blcs_source_settings(raw["trajectory_source"])
        generator = _blcs_generator_config(raw["generator"])
        assets = _blcs_assets(raw["assets"])
        timeout = _number(raw, "render_timeout_seconds", path="dataset.blcs")
        if timeout <= 0.0:
            raise SemanticConfigurationError(
                "dataset.blcs.render_timeout_seconds must be positive."
            )
        performance = _performance_budget(
            raw["performance"],
            path="dataset.blcs.performance",
        )
        timeline = FullFrameChunkPolicy.from_mapping(
            raw["timeline"], path="dataset.blcs.timeline"
        )
        if (
            not performance.require_cuda
            or not performance.execution_device.startswith("cuda")
            or performance.maximum_nht_invocations != source.scene_count
            or performance.maximum_batch_frames > timeline.chunk_size_frames
            or performance.maximum_published_fraction_of_dense_reference > 0.2
        ):
            raise SemanticConfigurationError(
                "BLCS production performance requires CUDA, one NHT call per "
                "trajectory, bounded frame batches, and <=20% dense publication."
            )
        return cls(
            timeline=timeline,
            trajectory_source=source,
            generator=generator,
            assets=assets,
            render_timeout_seconds=timeout,
            performance=performance,
            metadata_fields=metadata,
        )


@dataclass(frozen=True, slots=True)
class PLCSObjectConfiguration:
    """One config-owned ACCAD category and global-timeline placement."""

    category: MotionCategory
    start_frame: int
    anchor_position_court_m: tuple[float, float, float]
    yaw_radians: float

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> PLCSObjectConfiguration:
        path = f"dataset.plcs.objects[{index}]"
        raw = _exact(
            value,
            path=path,
            keys={"category", "start_frame", "anchor_position_court_m", "yaw_radians"},
        )
        anchor = _number_sequence(
            raw,
            "anchor_position_court_m",
            path=path,
            minimum_length=3,
        )
        if len(anchor) != 3:
            raise ConfigurationTypeError(
                f"{path}.anchor_position_court_m must contain exactly three values."
            )
        try:
            category = MotionCategory(_text(raw, "category", path=path))
        except ValueError as error:
            raise SemanticConfigurationError(
                f"{path}.category is not a production motion category."
            ) from error
        return cls(
            category=category,
            start_frame=_integer(raw, "start_frame", path=path, minimum=0),
            anchor_position_court_m=anchor,
            yaw_radians=_number(raw, "yaw_radians", path=path),
        )

    def to_runtime_request(self) -> PLCSObjectRequest:
        """Construct the handler value without creating a configuration import cycle."""
        from src.synthetic_data_generation.dataset.plcs.handler import PLCSObjectRequest

        return PLCSObjectRequest(
            category=self.category,
            start_frame=self.start_frame,
            anchor_position_court_m=self.anchor_position_court_m,
            yaw_radians=self.yaw_radians,
        )


@dataclass(frozen=True, slots=True)
class LinearRGBPaletteSettings:
    """Explicit deterministic avatar appearance source in renderer colour space."""

    source: Literal["palette"]
    assignment: Literal["object_index_modulo_palette"]
    gaussian_fill: Literal["uniform"]
    appearance_model: Literal["rgb"]
    appearance_space: Literal["linear_rgb"]
    colors: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_mapping(cls, value: object) -> LinearRGBPaletteSettings:
        path = "dataset.plcs.appearance"
        raw = _exact(
            value,
            path=path,
            keys={
                "source",
                "assignment",
                "gaussian_fill",
                "appearance_model",
                "appearance_space",
                "colors",
            },
        )
        source = _text(raw, "source", path=path)
        assignment = _text(raw, "assignment", path=path)
        gaussian_fill = _text(raw, "gaussian_fill", path=path)
        appearance_model = _text(raw, "appearance_model", path=path)
        appearance_space = _text(raw, "appearance_space", path=path)
        if (
            source != "palette"
            or assignment != "object_index_modulo_palette"
            or gaussian_fill != "uniform"
            or appearance_model != "rgb"
            or appearance_space != "linear_rgb"
        ):
            raise SemanticConfigurationError(
                "PLCS appearance must use the explicit uniform object-index palette "
                "in linear RGB."
            )
        raw_colors = _sequence(
            _value(raw, "colors", (list, tuple), path=path),
            path=f"{path}.colors",
        )
        if not raw_colors:
            raise SemanticConfigurationError("dataset.plcs.appearance.colors must not be empty.")
        colors: list[tuple[float, float, float]] = []
        for index, raw_color in enumerate(raw_colors):
            values = _sequence(raw_color, path=f"{path}.colors[{index}]")
            if len(values) != 3 or any(type(item) not in (int, float) for item in values):
                raise ConfigurationTypeError(
                    f"{path}.colors[{index}] must contain three numeric values."
                )
            color = cast(
                tuple[float, float, float],
                tuple(float(cast("int | float", item)) for item in values),
            )
            if any(not math.isfinite(channel) or not 0.0 <= channel <= 1.0 for channel in color):
                raise SemanticConfigurationError(
                    f"{path}.colors[{index}] must contain finite values in [0, 1]."
                )
            colors.append(color)
        if len(colors) != len(set(colors)):
            raise SemanticConfigurationError("PLCS palette colors must be unique.")
        return cls(
            source="palette",
            assignment="object_index_modulo_palette",
            gaussian_fill="uniform",
            appearance_model="rgb",
            appearance_space="linear_rgb",
            colors=tuple(colors),
        )

    def color_for_object(self, object_index: int) -> tuple[float, float, float]:
        """Return the sole deterministic palette assignment for one object."""
        if isinstance(object_index, bool) or not isinstance(object_index, int) or object_index < 0:
            raise ValueError("PLCS appearance object_index must be non-negative.")
        return self.colors[object_index % len(self.colors)]

    def preflight(self, *, gaussian_count: int) -> None:
        """Validate the explicit uniform palette source before stage mutation."""
        if isinstance(gaussian_count, bool) or not isinstance(gaussian_count, int) or gaussian_count <= 0:
            raise ValueError("PLCS appearance gaussian_count must be positive.")

    def load_avatar_appearance(
        self,
        *,
        clip: PLCSMotionClip,
        object_id: str,
        gaussian_count: int,
        seed: int,
        device: torch.device,
    ) -> AvatarAppearance:
        """Build one explicit uniform linear-RGB feature set with no fallback."""
        import torch

        from src.synthetic_data_generation.dataset.plcs.composition import (
            AvatarAppearance,
        )
        from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
            PLCSMotionClip,
        )

        self.preflight(gaussian_count=gaussian_count)
        if not isinstance(clip, PLCSMotionClip):
            raise TypeError("PLCS palette source requires a PLCSMotionClip.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("PLCS appearance seed must be non-negative.")
        prefix, separator, suffix = object_id.rpartition("-")
        if prefix != "player" or separator != "-" or len(suffix) != 3 or not suffix.isdigit():
            raise ValueError(
                "PLCS palette source requires the canonical player-NNN object ID."
            )
        object_index = int(suffix) - 1
        if object_index < 0:
            raise ValueError("PLCS palette object numbering starts at one.")
        typed_device = torch.device(device)
        color = torch.tensor(
            self.color_for_object(object_index),
            dtype=torch.float32,
            device=typed_device,
        )
        return AvatarAppearance(
            features=color.expand(gaussian_count, 3).clone(),
            appearance_model=self.appearance_model,
            appearance_space=self.appearance_space,
        )


@dataclass(frozen=True, slots=True)
class PLCSDatasetConfiguration:
    """Complete PLCS motion, SMPL-H, appearance, raster, and render authority."""

    timeline: FullFrameChunkPolicy
    motion_categories: tuple[str, ...]
    require_articulated_motion: bool
    multi_object_global_timeline: bool
    accad_root: Path
    split: str
    scene_splits: Mapping[str, str]
    objects: tuple[PLCSObjectConfiguration, ...]
    smplh_model_root: Path
    gaussian_count: int
    smplh_batch_size: int
    device: str
    appearance: LinearRGBPaletteSettings
    foreground_compositor: PLCSForegroundCompositor
    render_timeout_seconds: float
    performance: DatasetPerformanceBudget
    metadata_fields: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        resolver: PathResolver,
    ) -> PLCSDatasetConfiguration:
        raw = _exact(
            value,
            path="dataset.plcs",
            keys={
                "timeline",
                "motion_categories",
                "require_articulated_motion",
                "multi_object_global_timeline",
                "accad_root",
                "split",
                "scene_splits",
                "objects",
                "smplh_model_root",
                "gaussian_count",
                "smplh_batch_size",
                "device",
                "appearance",
                "foreground_rasterizer",
                "render_timeout_seconds",
                "performance",
                "metadata_fields",
            },
        )
        categories = _text_sequence(raw, "motion_categories", path="dataset.plcs")
        if set(categories) != {"running", "walking", "general"}:
            raise SemanticConfigurationError(
                "dataset.plcs.motion_categories must be running, walking, and general."
            )
        articulated = _flag(raw, "require_articulated_motion", path="dataset.plcs")
        global_timeline = _flag(raw, "multi_object_global_timeline", path="dataset.plcs")
        _require_true(articulated, path="dataset.plcs.require_articulated_motion")
        _require_true(global_timeline, path="dataset.plcs.multi_object_global_timeline")
        metadata = _text_sequence(raw, "metadata_fields", path="dataset.plcs")
        if not _PLCS_METADATA_FIELDS.issubset(metadata):
            raise SemanticConfigurationError("dataset.plcs.metadata_fields omits required provenance.")
        scene_splits_raw = _mapping(raw["scene_splits"], path="dataset.plcs.scene_splits")
        if not scene_splits_raw:
            raise SemanticConfigurationError("dataset.plcs.scene_splits must not be empty.")
        scene_splits: dict[str, str] = {}
        for scene_id in sorted(scene_splits_raw):
            split_value = scene_splits_raw[scene_id]
            if (
                not scene_id
                or scene_id != scene_id.strip()
                or type(split_value) is not str
                or split_value not in {"train", "validation", "test"}
            ):
                raise SemanticConfigurationError(
                    "dataset.plcs.scene_splits must map trimmed scene IDs to "
                    "train, validation, or test."
                )
            scene_splits[scene_id] = split_value
        raw_objects = _sequence(
            _value(raw, "objects", (list, tuple), path="dataset.plcs"),
            path="dataset.plcs.objects",
        )
        if len(raw_objects) < 2:
            raise SemanticConfigurationError(
                "Production PLCS configuration requires multiple object requests."
            )
        objects = tuple(
            PLCSObjectConfiguration.from_mapping(item, index=index)
            for index, item in enumerate(raw_objects)
        )
        if {item.category.value for item in objects} != set(categories):
            raise SemanticConfigurationError(
                "PLCS objects must explicitly request every configured motion category."
            )
        raster_path = "dataset.plcs.foreground_rasterizer"
        raster_raw = _exact(
            raw["foreground_rasterizer"],
            path=raster_path,
            keys={
                "sigma_extent",
                "minimum_pixel_variance",
                "near_plane",
                "visibility_threshold",
                "maximum_alpha",
            },
        )
        compositor = PLCSForegroundCompositor(
            sigma_extent=_number(raster_raw, "sigma_extent", path=raster_path),
            minimum_pixel_variance=_number(
                raster_raw, "minimum_pixel_variance", path=raster_path
            ),
            near_plane=_number(raster_raw, "near_plane", path=raster_path),
            visibility_threshold=_number(
                raster_raw, "visibility_threshold", path=raster_path
            ),
            maximum_alpha=_number(raster_raw, "maximum_alpha", path=raster_path),
        )
        split = _text(raw, "split", path="dataset.plcs")
        if split not in {"train", "validation", "test"}:
            raise SemanticConfigurationError(
                "dataset.plcs.split must be train, validation, or test."
            )
        timeout = _number(raw, "render_timeout_seconds", path="dataset.plcs")
        if timeout <= 0.0:
            raise SemanticConfigurationError(
                "dataset.plcs.render_timeout_seconds must be positive."
            )
        performance = _performance_budget(
            raw["performance"],
            path="dataset.plcs.performance",
        )
        timeline = FullFrameChunkPolicy.from_mapping(
            raw["timeline"], path="dataset.plcs.timeline"
        )
        if (
            not performance.require_cuda
            or not performance.execution_device.startswith("cuda")
            or performance.maximum_nht_invocations != 1
            or performance.maximum_batch_frames > timeline.chunk_size_frames
            or performance.maximum_batch_frames
            != _integer(
                raw,
                "smplh_batch_size",
                path="dataset.plcs",
                minimum=1,
            )
            or performance.execution_device != _text(
                raw,
                "device",
                path="dataset.plcs",
            )
            or performance.maximum_published_fraction_of_dense_reference > 0.25
        ):
            raise SemanticConfigurationError(
                "PLCS production performance requires CUDA, one NHT background "
                "call, bounded frame batches, and <=25% dense publication."
            )
        return cls(
            timeline=timeline,
            motion_categories=categories,
            require_articulated_motion=articulated,
            multi_object_global_timeline=global_timeline,
            accad_root=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _text(raw, "accad_root", path="dataset.plcs"),
            ),
            split=split,
            scene_splits=scene_splits,
            objects=objects,
            smplh_model_root=resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                _text(raw, "smplh_model_root", path="dataset.plcs"),
            ),
            gaussian_count=_integer(
                raw, "gaussian_count", path="dataset.plcs", minimum=1
            ),
            smplh_batch_size=_integer(
                raw, "smplh_batch_size", path="dataset.plcs", minimum=1
            ),
            device=_text(raw, "device", path="dataset.plcs"),
            appearance=LinearRGBPaletteSettings.from_mapping(raw["appearance"]),
            foreground_compositor=compositor,
            render_timeout_seconds=timeout,
            performance=performance,
            metadata_fields=metadata,
        )

    def build_stage_parameters(self, *, seed: int) -> PLCSStageParameters:
        """Construct handler parameters while keeping its import one-way."""
        from src.synthetic_data_generation.dataset.plcs.handler import (
            PLCSStageParameters,
        )

        return PLCSStageParameters(
            seed=seed,
            split=self.split,
            scene_splits=self.scene_splits,
            objects=tuple(item.to_runtime_request() for item in self.objects),
            smplh_model_root=self.smplh_model_root,
            gaussian_count=self.gaussian_count,
            smplh_batch_size=self.smplh_batch_size,
            device=self.device,
        )


@dataclass(frozen=True, slots=True)
class ScenePipelineConfiguration:
    """Resolved canonical request and every config-owned stage/domain policy."""

    profile: str
    resolver: PathResolver
    workspace: SceneWorkspace
    request: ScenePipelineRequest
    stages: PipelineStageSettings
    camera: CameraProfileConfig
    nht: NHTCommandPaths
    alignment: AlignmentConfiguration
    court: CourtDatasetConfiguration
    blcs: BLCSDatasetConfiguration
    plcs: PLCSDatasetConfiguration

    @classmethod
    def from_config(cls, value: object) -> ScenePipelineConfiguration:
        root = _exact(
            value,
            path="configuration",
            keys={"profile", "roots", "request", "pipeline", "camera", "nht", "alignment", "dataset"},
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
            keys={"scene_id", "source_video", "targets", "from_stage"},
        )
        target_values = _text_sequence(request_raw, "targets", path="request")
        try:
            targets = tuple(DatasetTarget(item) for item in target_values)
            from_stage = StageName(_text(request_raw, "from_stage", path="request"))
        except ValueError as error:
            raise SemanticConfigurationError(f"request contains an unknown target or stage: {error}") from error
        if len(targets) != len(set(targets)):
            raise SemanticConfigurationError("request.targets must not contain duplicates.")
        source_video = resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            _text(request_raw, "source_video", path="request"),
        )
        request = ScenePipelineRequest(
            scene_id=_text(request_raw, "scene_id", path="request"),
            source_video=source_video,
            targets=frozenset(targets),
            from_stage=from_stage,
            config_schema=stages.config_schema,
        )
        dataset = _exact(root["dataset"], path="dataset", keys={"court", "blcs", "plcs"})
        plcs = PLCSDatasetConfiguration.from_mapping(
            dataset["plcs"],
            resolver=resolver,
        )
        if (
            request.scene_id not in plcs.scene_splits
            or plcs.scene_splits[request.scene_id] != plcs.split
        ):
            raise SemanticConfigurationError(
                "dataset.plcs.scene_splits must explicitly bind request.scene_id "
                "to dataset.plcs.split."
            )
        return cls(
            profile=_text(root, "profile", path="configuration"),
            resolver=resolver,
            workspace=SceneWorkspace.resolve(resolver, request.scene_id),
            request=request,
            stages=stages,
            camera=_camera_profile(root["camera"]),
            nht=NHTCommandPaths.from_mapping(root["nht"]),
            alignment=AlignmentConfiguration.from_mapping(
                root["alignment"],
                resolver=resolver,
            ),
            court=CourtDatasetConfiguration.from_mapping(dataset["court"]),
            blcs=BLCSDatasetConfiguration.from_mapping(dataset["blcs"]),
            plcs=plcs,
        )


def validate_scene_pipeline_boundary(config: DictConfig) -> None:
    """Fail closed before the canonical CLI performs any mutation."""
    ScenePipelineConfiguration.from_config(config)


register_boundary_validator(SCENE_PIPELINE_BOUNDARY, validate_scene_pipeline_boundary)

__all__ = [
    "AlignmentConfiguration",
    "BLCSDatasetConfiguration",
    "CourtDatasetConfiguration",
    "CourtSamplingPolicy",
    "CourtTrajectoryPolicy",
    "CourtViewPolicy",
    "FullFrameChunkPolicy",
    "LinearRGBPaletteSettings",
    "NHTCommandPaths",
    "PipelineStageSettings",
    "PLCSObjectConfiguration",
    "PLCSDatasetConfiguration",
    "SCENE_PIPELINE_BOUNDARY",
    "SCENE_PIPELINE_SCHEMA",
    "ScenePipelineConfiguration",
    "validate_scene_pipeline_boundary",
]
