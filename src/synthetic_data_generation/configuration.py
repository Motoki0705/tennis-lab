"""Strict runtime configuration contracts for synthetic-data entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import (
    ConfigField,
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

ROOT_FIELDS = {
    "project_root": ConfigField.of(str),
    "data_root": ConfigField.of(str),
    "checkpoint_root": ConfigField.of(str),
    "artifact_root": ConfigField.of(str),
    "output_root": ConfigField.of(str),
    "cache_root": ConfigField.of(str),
    "external_asset_root": ConfigField.of(str),
}
ROOT_SCHEMA = StrictConfigSchema(name="roots", fields=ROOT_FIELDS)

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RENDERER_PLACEHOLDERS = {
    "input",
    "output",
    "reference",
    "source_root",
    "artifact_root",
    "dataset_root",
}


@dataclass(frozen=True, slots=True)
class VerifiedSystemExecutable:
    """One content-pinned executable beneath an explicit narrow system root."""

    root: Path
    relative_path: Path
    sha256: str
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise PathContractError("System executable root must be absolute.")
        if self.root == Path(self.root.anchor):
            raise PathContractError(
                "System executable root must not grant the filesystem root."
            )
        if self.root.resolve(strict=False) != self.root:
            raise PathContractError(
                "System executable root must be an already resolved path."
            )
        if not self.root.is_dir():
            raise PathContractError(
                f"System executable root is not a directory: {self.root}"
            )
        rendered = str(self.relative_path)
        if (
            not rendered.strip()
            or rendered != rendered.strip()
            or self.relative_path.is_absolute()
            or self.relative_path in {Path("."), Path("..")}
            or ".." in self.relative_path.parts
        ):
            raise PathContractError(
                "System executable path must be a non-empty trimmed relative child."
            )
        if not _SHA256.fullmatch(self.sha256):
            raise SemanticConfigurationError(
                "System executable sha256 must be a lowercase SHA-256 digest."
            )
        try:
            resolved = (self.root / self.relative_path).resolve(strict=True)
        except FileNotFoundError as error:
            raise PathContractError(
                "Configured system executable does not exist: "
                f"{self.root / self.relative_path}"
            ) from error
        if not resolved.is_relative_to(self.root):
            raise PathContractError(
                "Configured system executable resolves outside its declared root: "
                f"{resolved} (root: {self.root})."
            )
        object.__setattr__(self, "path", resolved)
        self.verify()

    def verify(self) -> Path:
        """Recheck identity and executable kind immediately before process use."""
        if not self.root.is_dir() or self.root.resolve(strict=False) != self.root:
            raise PathContractError(
                f"System executable root is not a resolved directory: {self.root}"
            )
        try:
            resolved = (self.root / self.relative_path).resolve(strict=True)
        except FileNotFoundError as error:
            raise PathContractError(
                f"Configured system executable no longer exists: {self.path}"
            ) from error
        if resolved != self.path or not resolved.is_relative_to(self.root):
            raise PathContractError(
                "Configured system executable target changed or escaped its root."
            )
        if not resolved.is_file():
            raise PathContractError(
                f"Configured system executable is not a regular file: {resolved}"
            )
        if not resolved.stat().st_mode & (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        ):
            raise PathContractError(
                f"Configured system executable is not executable: {resolved}"
            )
        with resolved.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != self.sha256:
            raise PathContractError(
                "Configured system executable SHA-256 mismatch: "
                f"expected {self.sha256}, computed {digest}."
            )
        return resolved


def _mapping_schema(
    name: str,
    fields: Mapping[str, ConfigField],
    *,
    semantic_checks: tuple[Callable[[Mapping[str, object]], None], ...] = (),
) -> StrictConfigSchema:
    return StrictConfigSchema(
        name=name,
        fields=fields,
        semantic_checks=semantic_checks,
    )


def _numbers(*, required: bool = True) -> ConfigField:
    return ConfigField.of(int, float, required=required)


def _semantic_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


def _finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SemanticConfigurationError(f"{name} must be finite.")
    return float(value)


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SemanticConfigurationError(
            f"{name} must be a non-empty trimmed string."
        )
    return value


def _path_safe_id(value: object, *, name: str) -> str:
    rendered = _nonempty_text(value, name=name)
    if _PATH_SAFE_ID.fullmatch(rendered) is None:
        raise SemanticConfigurationError(f"{name} must be path-safe.")
    return rendered


def _sha256(value: object, *, name: str) -> str:
    rendered = _nonempty_text(value, name=name)
    if _SHA256.fullmatch(rendered) is None:
        raise SemanticConfigurationError(
            f"{name} must be a lower-case SHA-256 digest."
        )
    return rendered


def _nonnegative_seed(value: object, *, name: str) -> None:
    if type(value) is not int or value < 0:
        raise SemanticConfigurationError(f"{name} must be a non-negative integer.")


def _positive_integer(value: object, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise SemanticConfigurationError(f"{name} must be a positive integer.")


def _group_ids(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise SemanticConfigurationError(f"{name} must be a non-empty sequence.")
    if any(type(item) is not int or item < 0 for item in value):
        raise SemanticConfigurationError(
            f"{name} must contain only non-negative integers."
        )
    if len(set(value)) != len(value):
        raise SemanticConfigurationError(f"{name} must not contain duplicates.")
    return cast(tuple[int, ...], value)


def _validate_renderer_template(value: str) -> None:
    try:
        parsed = tuple(Formatter().parse(value))
    except ValueError as error:
        raise SemanticConfigurationError(
            f"renderer command contains an invalid template: {value!r}."
        ) from error
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _RENDERER_PLACEHOLDERS:
            raise SemanticConfigurationError(
                f"renderer command uses unknown path placeholder {field_name!r}."
            )
        if format_spec or conversion:
            raise SemanticConfigurationError(
                "renderer path placeholders do not accept formatting."
            )


PATH_MANIFEST_SCHEMA = _mapping_schema(
    "paths",
    {
        name: ConfigField.of(str)
        for name in (
            "source_root",
            "artifact_root",
            "execution_root",
            "dataset_root",
            "alignment_observations",
            "render_jobs",
            "pipeline_manifest",
            "alignment_metrics",
            "dataset_plan",
            "render_manifest",
            "quality_metrics",
            "visualization",
        )
    },
)
RENDERER_SCHEMA = _mapping_schema(
    "renderer",
    {
        "mode": ConfigField.of(str),
        "command": ConfigField.sequence(ConfigField.of(str)),
        "working_directory": ConfigField.of(str),
    },
)


def _renderer_semantics(value: Mapping[str, object]) -> None:
    renderer = _semantic_mapping(value["renderer"], name="renderer")
    mode = renderer["mode"]
    command = cast(tuple[object, ...], renderer["command"])
    if mode not in {"execute", "prepared_outputs"}:
        raise SemanticConfigurationError(
            "renderer.mode must be 'execute' or 'prepared_outputs'."
        )
    if mode == "execute" and not command:
        raise SemanticConfigurationError(
            "renderer.command must be non-empty when renderer.mode='execute'."
        )
    if mode == "prepared_outputs" and command:
        raise SemanticConfigurationError(
            "renderer.command must be empty when renderer.mode='prepared_outputs'."
        )
    if any(
        type(token) is not str or not token or token != token.strip()
        for token in command
    ):
        raise SemanticConfigurationError(
            "renderer.command must contain only non-empty trimmed strings."
        )
    for token in command:
        if not isinstance(token, str):
            raise AssertionError("Validated renderer command token is not a string.")
        _validate_renderer_template(token)
    _nonempty_text(renderer["working_directory"], name="renderer.working_directory")


PIPELINE_SCHEMA = StrictConfigSchema(
    name="synthetic.dataset.pipeline",
    fields={
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "execute": ConfigField.of(bool),
        "paths": ConfigField.mapping(PATH_MANIFEST_SCHEMA),
        "renderer": ConfigField.mapping(RENDERER_SCHEMA),
    },
    semantic_checks=(_renderer_semantics,),
)


def _feature_fit_semantics(value: Mapping[str, object]) -> None:
    if value["source_format"] not in {
        "independent_nht_tensor_pack_v1",
        "vanilla_3dgs_ply_v1",
    }:
        raise SemanticConfigurationError("feature_fit.source_format is unsupported.")
    if cast(int, value["optimization_steps"]) <= 0:
        raise SemanticConfigurationError(
            "feature_fit.optimization_steps must be positive."
        )
    feature_lr = _finite(value["feature_lr"], name="feature_fit.feature_lr")
    final_fraction = _finite(
        value["final_lr_fraction"], name="feature_fit.final_lr_fraction"
    )
    minimum_psnr = _finite(
        value["min_validation_psnr_db"],
        name="feature_fit.min_validation_psnr_db",
    )
    if feature_lr <= 0.0:
        raise SemanticConfigurationError("feature_fit.feature_lr must be positive.")
    if not 0.0 < final_fraction <= 1.0:
        raise SemanticConfigurationError(
            "feature_fit.final_lr_fraction must lie in (0, 1]."
        )
    if minimum_psnr < 20.0:
        raise SemanticConfigurationError(
            "feature_fit.min_validation_psnr_db must be at least 20."
        )
    _nonnegative_seed(value["seed"], name="feature_fit.seed")
    _sha256(
        value["target_appearance_space_sha256"],
        name="feature_fit.target_appearance_space_sha256",
    )
    device = _nonempty_text(value["device"], name="feature_fit.device")
    if not device.startswith("cuda:"):
        raise SemanticConfigurationError(
            "feature_fit.device must explicitly select a CUDA device."
        )


FEATURE_FIT_SCHEMA = StrictConfigSchema(
    name="synthetic.dataset.blcs.feature_fit",
    fields={
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "source": ConfigField.of(str),
        "source_format": ConfigField.of(str),
        "calibration_bundle": ConfigField.of(str),
        "target_appearance": ConfigField.of(str),
        "target_appearance_space_sha256": ConfigField.of(str),
        "output_dir": ConfigField.of(str),
        "optimization_steps": ConfigField.of(int),
        "feature_lr": ConfigField.of(float),
        "final_lr_fraction": ConfigField.of(float),
        "min_validation_psnr_db": ConfigField.of(float),
        "seed": ConfigField.of(int),
        "device": ConfigField.of(str),
        "runtime_pins": ConfigField.of(str),
        "nht_repository": ConfigField.of(str),
        "gsplat_repository": ConfigField.of(str),
        "worker_source": ConfigField.of(str),
    },
    semantic_checks=(_feature_fit_semantics,),
)


def _ground_plane_semantics(value: Mapping[str, object]) -> None:
    _nonnegative_seed(value["seed"], name="ground_plane.seed")
    for name in (
        "footprint_quantile",
        "min_normal_up_cosine",
        "support_bounds_quantile",
    ):
        number = _finite(value[name], name=f"ground_plane.{name}")
        if not 0.0 <= number < 1.0:
            raise SemanticConfigurationError(
                f"ground_plane.{name} must lie in [0, 1)."
            )
    positive_fraction = _finite(
        value["min_positive_camera_fraction"],
        name="ground_plane.min_positive_camera_fraction",
    )
    if not 0.0 <= positive_fraction <= 1.0:
        raise SemanticConfigurationError(
            "ground_plane.min_positive_camera_fraction must lie in [0, 1]."
        )
    for name in (
        "footprint_margin",
        "min_camera_height",
        "max_camera_height",
        "histogram_bin_width",
        "candidate_half_width",
        "ransac_threshold",
        "refine_threshold",
    ):
        if _finite(value[name], name=f"ground_plane.{name}") <= 0.0:
            raise SemanticConfigurationError(f"ground_plane.{name} must be positive.")
    if _finite(
        value["min_camera_height"], name="ground_plane.min_camera_height"
    ) >= _finite(value["max_camera_height"], name="ground_plane.max_camera_height"):
        raise SemanticConfigurationError(
            "ground_plane.min_camera_height must be smaller than max_camera_height."
        )
    for name in (
        "ransac_iterations",
        "ransac_sample_limit",
        "refine_iterations",
        "min_candidate_points",
        "min_support_points",
    ):
        _positive_integer(value[name], name=f"ground_plane.{name}")


def _line_projection_semantics(value: Mapping[str, object]) -> None:
    probability = _finite(
        value["probability_threshold"],
        name="line_projection.probability_threshold",
    )
    if not 0.0 <= probability <= 1.0:
        raise SemanticConfigurationError(
            "line_projection.probability_threshold must lie in [0, 1]."
        )
    for name in (
        "proximity_scale",
        "proximity_power",
        "min_ray_plane_cosine",
        "max_ray_distance",
        "bounds_margin",
        "grid_spacing",
    ):
        if _finite(value[name], name=f"line_projection.{name}") <= 0.0:
            raise SemanticConfigurationError(
                f"line_projection.{name} must be positive."
            )
    if _finite(
        value["min_ray_plane_cosine"],
        name="line_projection.min_ray_plane_cosine",
    ) >= 1.0:
        raise SemanticConfigurationError(
            "line_projection.min_ray_plane_cosine must be smaller than one."
        )
    _positive_integer(
        value["min_projected_pixels"],
        name="line_projection.min_projected_pixels",
    )


def _infer_semantics(value: Mapping[str, object]) -> None:
    if value["stage"] != "ground_line":
        raise SemanticConfigurationError("stage must be 'ground_line'.")
    _path_safe_id(value["artifact_id"], name="artifact_id")
    _nonempty_text(value["device"], name="device")
    _nonnegative_seed(value["seed"], name="seed")
    _positive_integer(value["expected_short_side"], name="expected_short_side")
    _group_ids(value["holdout_group_ids"], name="holdout_group_ids")


GROUND_PLANE_SCHEMA = _mapping_schema(
    "ground_plane",
    {
        "seed": ConfigField.of(int),
        **{
            name: _numbers()
            for name in (
                "footprint_quantile",
                "footprint_margin",
                "min_camera_height",
                "max_camera_height",
                "histogram_bin_width",
                "candidate_half_width",
                "ransac_threshold",
                "refine_threshold",
                "min_normal_up_cosine",
                "min_positive_camera_fraction",
                "support_bounds_quantile",
            )
        },
        **{
            name: ConfigField.of(int)
            for name in (
                "ransac_iterations",
                "ransac_sample_limit",
                "refine_iterations",
                "min_candidate_points",
                "min_support_points",
            )
        },
    },
    semantic_checks=(_ground_plane_semantics,),
)
LINE_PROJECTION_SCHEMA = _mapping_schema(
    "line_projection",
    {
        name: _numbers()
        for name in (
            "probability_threshold",
            "proximity_scale",
            "proximity_power",
            "min_ray_plane_cosine",
            "max_ray_distance",
            "bounds_margin",
            "grid_spacing",
        )
    }
    | {"min_projected_pixels": ConfigField.of(int)},
    semantic_checks=(_line_projection_semantics,),
)
INFER_SCHEMA = _mapping_schema(
    "synthetic.alignment.infer_ground_line_map",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "stage": ConfigField.of(str),
        "artifact_id": ConfigField.of(str),
        "provider_bundle": ConfigField.of(str),
        "line_checkpoint": ConfigField.of(str),
        "backbone_repository": ConfigField.of(str),
        "backbone_checkpoint": ConfigField.of(str),
        "output_dir": ConfigField.of(str),
        "device": ConfigField.of(str),
        "seed": ConfigField.of(int),
        "expected_short_side": ConfigField.of(int),
        "verify_provider_files": ConfigField.of(bool),
        "holdout_group_ids": ConfigField.sequence(ConfigField.of(int)),
        "ground_plane": ConfigField.mapping(GROUND_PLANE_SCHEMA),
        "line_projection": ConfigField.mapping(LINE_PROJECTION_SCHEMA),
    },
    semantic_checks=(_infer_semantics,),
)


def _fit_semantics(value: Mapping[str, object]) -> None:
    _nonnegative_seed(value["seed"], name="fit.seed")
    fractions = tuple(
        _finite(value[name], name=f"fit.{name}")
        for name in (
            "proposal_fraction",
            "uniform_fraction",
            "orthogonal_fraction",
        )
    )
    if any(number <= 0.0 for number in fractions) or not math.isclose(
        sum(fractions), 1.0, abs_tol=1.0e-9
    ):
        raise SemanticConfigurationError(
            "fit proposal fractions must be positive and sum to one."
        )
    for name in (
        "num_global_runs",
        "bootstrap_runs",
        "max_instances",
        "optimizer_max_iterations",
        "optimizer_population_size",
        "local_optimizer_max_iterations",
        "local_optimizer_population_size",
        "bootstrap_block_rows",
        "bootstrap_block_columns",
    ):
        _positive_integer(value[name], name=f"fit.{name}")
    for name in (
        "min_cluster_support_rate",
        "min_bootstrap_survival_rate",
        "orientation_ambiguity_margin",
        "residual_suppression_strength",
        "min_residual_gain",
        "min_confidence",
        "min_line_coverage",
        "min_internal_line_coverage",
        "bootstrap_keep_fraction",
    ):
        number = _finite(value[name], name=f"fit.{name}")
        if not 0.0 < number <= 1.0:
            raise SemanticConfigurationError(f"fit.{name} must lie in (0, 1].")
    for name in (
        "cluster_center_tolerance_m",
        "cluster_orientation_tolerance_deg",
        "cluster_scale_relative_tolerance",
        "duplicate_center_tolerance_m",
        "optimizer_tolerance",
        "min_template_score",
        "template_support_width_m",
        "background_ring_width_m",
        "blur_sigma_cells",
        "samples_per_metre",
    ):
        if _finite(value[name], name=f"fit.{name}") <= 0.0:
            raise SemanticConfigurationError(f"fit.{name} must be positive.")
    minimum_scale = _finite(
        value["min_scale_scene_per_metre"],
        name="fit.min_scale_scene_per_metre",
    )
    maximum_scale = _finite(
        value["max_scale_scene_per_metre"],
        name="fit.max_scale_scene_per_metre",
    )
    if not 0.0 < minimum_scale < maximum_scale:
        raise SemanticConfigurationError("fit court scale bounds are invalid.")
    minimum_orientation = _finite(
        value["orientation_min_radians"], name="fit.orientation_min_radians"
    )
    maximum_orientation = _finite(
        value["orientation_max_radians"], name="fit.orientation_max_radians"
    )
    if minimum_orientation >= maximum_orientation:
        raise SemanticConfigurationError("fit court orientation bounds are invalid.")
    if _finite(
        value["cluster_scale_relative_tolerance"],
        name="fit.cluster_scale_relative_tolerance",
    ) >= 1.0:
        raise SemanticConfigurationError(
            "fit.cluster_scale_relative_tolerance must be smaller than one."
        )
    if _finite(
        value["background_ring_width_m"], name="fit.background_ring_width_m"
    ) <= _finite(
        value["template_support_width_m"], name="fit.template_support_width_m"
    ):
        raise SemanticConfigurationError(
            "fit.background_ring_width_m must exceed template_support_width_m."
        )


def _fit_ground_semantics(value: Mapping[str, object]) -> None:
    if value["stage"] != "court_fit":
        raise SemanticConfigurationError("stage must be 'court_fit'.")
    _path_safe_id(value["artifact_id"], name="artifact_id")


FIT_SCHEMA = _mapping_schema(
    "fit",
    {
        "seed": ConfigField.of(int),
        **{
            name: ConfigField.of(int)
            for name in (
                "num_global_runs",
                "bootstrap_runs",
                "max_instances",
                "optimizer_max_iterations",
                "optimizer_population_size",
                "local_optimizer_max_iterations",
                "local_optimizer_population_size",
                "bootstrap_block_rows",
                "bootstrap_block_columns",
            )
        },
        **{
            name: _numbers()
            for name in (
                "proposal_fraction",
                "uniform_fraction",
                "orthogonal_fraction",
                "cluster_center_tolerance_m",
                "cluster_orientation_tolerance_deg",
                "cluster_scale_relative_tolerance",
                "min_cluster_support_rate",
                "min_bootstrap_survival_rate",
                "orientation_ambiguity_margin",
                "residual_suppression_strength",
                "min_residual_gain",
                "blur_sigma_cells",
                "samples_per_metre",
                "min_scale_scene_per_metre",
                "max_scale_scene_per_metre",
                "orientation_min_radians",
                "orientation_max_radians",
                "optimizer_tolerance",
                "min_template_score",
                "min_confidence",
                "min_line_coverage",
                "min_internal_line_coverage",
                "duplicate_center_tolerance_m",
                "bootstrap_keep_fraction",
                "template_support_width_m",
                "background_ring_width_m",
            )
        },
    },
    semantic_checks=(_fit_semantics,),
)
FIT_GROUND_SCHEMA = _mapping_schema(
    "synthetic.alignment.fit_ground_courts",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "stage": ConfigField.of(str),
        "artifact_id": ConfigField.of(str),
        "ground_line_artifact": ConfigField.of(str),
        "output_dir": ConfigField.of(str),
        "fit": ConfigField.mapping(FIT_SCHEMA),
    },
    semantic_checks=(_fit_ground_semantics,),
)


def _evaluation_semantics(value: Mapping[str, object]) -> None:
    for name in (
        "line_inlier_distance_m",
        "template_sample_spacing_m",
        "point_cloud_vertical_tolerance_m",
        "point_cloud_grid_spacing_m",
    ):
        if _finite(value[name], name=f"evaluation.{name}") <= 0.0:
            raise SemanticConfigurationError(f"evaluation.{name} must be positive.")
    if _finite(
        value["court_roi_margin_m"], name="evaluation.court_roi_margin_m"
    ) < 0.0:
        raise SemanticConfigurationError(
            "evaluation.court_roi_margin_m must be non-negative."
        )


def _local_refit_semantics(value: Mapping[str, object]) -> None:
    _nonnegative_seed(value["seed"], name="local_refit.seed")
    for name in (
        "centre_radius_m",
        "orientation_tolerance_radians",
        "blur_sigma_cells",
        "samples_per_metre",
        "optimizer_tolerance",
    ):
        if _finite(value[name], name=f"local_refit.{name}") <= 0.0:
            raise SemanticConfigurationError(f"local_refit.{name} must be positive.")
    relative_tolerance = _finite(
        value["scale_relative_tolerance"],
        name="local_refit.scale_relative_tolerance",
    )
    if not 0.0 < relative_tolerance < 1.0:
        raise SemanticConfigurationError(
            "local_refit.scale_relative_tolerance must lie in (0, 1)."
        )
    _positive_integer(
        value["optimizer_max_iterations"],
        name="local_refit.optimizer_max_iterations",
    )
    _positive_integer(
        value["optimizer_population_size"],
        name="local_refit.optimizer_population_size",
    )


def _metric_threshold_semantics(value: Mapping[str, object]) -> None:
    for name in (
        "minimum_accepted_view_fraction",
        "minimum_weighted_inlier_fraction",
        "minimum_template_coverage_fraction",
        "maximum_stability_relative_scale_difference",
        "minimum_point_grid_coverage_fraction",
    ):
        number = _finite(value[name], name=f"metric_thresholds.{name}")
        if not 0.0 <= number <= 1.0:
            raise SemanticConfigurationError(
                f"metric_thresholds.{name} must lie in [0, 1]."
            )
    for name in (
        "maximum_distance_weighted_q95_m",
        "maximum_stability_centre_shift_m",
        "maximum_stability_orientation_deg",
        "maximum_point_support_rms_m",
    ):
        if _finite(value[name], name=f"metric_thresholds.{name}") < 0.0:
            raise SemanticConfigurationError(
                f"metric_thresholds.{name} must be non-negative."
            )
    _positive_integer(
        value["minimum_point_support_count"],
        name="metric_thresholds.minimum_point_support_count",
    )


def _calibrate_semantics(value: Mapping[str, object]) -> None:
    if value["stage"] != "calibration":
        raise SemanticConfigurationError("stage must be 'calibration'.")
    _path_safe_id(value["artifact_id"], name="artifact_id")
    validation_dice = _finite(
        value["checkpoint_val_dice"], name="checkpoint_val_dice"
    )
    best_dice = _finite(
        value["checkpoint_best_val_dice"], name="checkpoint_best_val_dice"
    )
    if not 0.0 <= validation_dice <= best_dice <= 1.0:
        raise SemanticConfigurationError(
            "checkpoint dice values must satisfy 0 <= val <= best <= 1."
        )
    _positive_integer(value["expected_short_side"], name="expected_short_side")
    _nonempty_text(value["device"], name="device")
    _nonnegative_seed(value["seed"], name="seed")
    holdouts = set(_group_ids(value["holdout_group_ids"], name="holdout_group_ids"))
    subsets = value["stability_subsets"]
    if not isinstance(subsets, tuple) or not subsets:
        raise SemanticConfigurationError(
            "stability_subsets must be a non-empty sequence."
        )
    normalized: list[tuple[int, ...]] = []
    for index, subset in enumerate(subsets):
        group_ids = _group_ids(subset, name=f"stability_subsets[{index}]")
        if not set(group_ids).isdisjoint(holdouts):
            raise SemanticConfigurationError(
                f"stability_subsets[{index}] must exclude holdout groups."
            )
        normalized.append(group_ids)
    if len(set(normalized)) != len(normalized):
        raise SemanticConfigurationError("stability_subsets must be distinct.")


EVALUATION_SCHEMA = _mapping_schema(
    "evaluation",
    {
        name: _numbers()
        for name in (
            "line_inlier_distance_m",
            "court_roi_margin_m",
            "template_sample_spacing_m",
            "point_cloud_vertical_tolerance_m",
            "point_cloud_grid_spacing_m",
        )
    },
    semantic_checks=(_evaluation_semantics,),
)
LOCAL_REFIT_SCHEMA = _mapping_schema(
    "local_refit",
    {
        "seed": ConfigField.of(int),
        "optimizer_max_iterations": ConfigField.of(int),
        "optimizer_population_size": ConfigField.of(int),
        **{
            name: _numbers()
            for name in (
                "centre_radius_m",
                "orientation_tolerance_radians",
                "scale_relative_tolerance",
                "blur_sigma_cells",
                "samples_per_metre",
                "optimizer_tolerance",
            )
        },
    },
    semantic_checks=(_local_refit_semantics,),
)
METRIC_THRESHOLDS_SCHEMA = _mapping_schema(
    "metric_thresholds",
    {
        **{
            name: _numbers()
            for name in (
                "minimum_accepted_view_fraction",
                "minimum_weighted_inlier_fraction",
                "maximum_distance_weighted_q95_m",
                "minimum_template_coverage_fraction",
                "maximum_stability_centre_shift_m",
                "maximum_stability_orientation_deg",
                "maximum_stability_relative_scale_difference",
                "maximum_point_support_rms_m",
                "minimum_point_grid_coverage_fraction",
            )
        },
        "minimum_point_support_count": ConfigField.of(int),
    },
    semantic_checks=(_metric_threshold_semantics,),
)


CALIBRATE_SCHEMA = _mapping_schema(
    "synthetic.alignment.calibrate_court_alignment",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "stage": ConfigField.of(str),
        "artifact_id": ConfigField.of(str),
        "provider_bundle": ConfigField.of(str),
        "ground_line_artifact": ConfigField.of(str),
        "geometry_artifact": ConfigField.of(str),
        "output_dir": ConfigField.of(str),
        "line_checkpoint": ConfigField.of(str),
        "backbone_repository": ConfigField.of(str),
        "backbone_checkpoint": ConfigField.of(str),
        "checkpoint_val_dice": _numbers(),
        "checkpoint_best_val_dice": _numbers(),
        "expected_short_side": ConfigField.of(int),
        "device": ConfigField.of(str),
        "seed": ConfigField.of(int),
        "verify_provider_files": ConfigField.of(bool),
        "holdout_group_ids": ConfigField.sequence(ConfigField.of(int)),
        "evaluation": ConfigField.mapping(EVALUATION_SCHEMA),
        "stability_subsets": ConfigField.sequence(
            ConfigField.sequence(ConfigField.of(int))
        ),
        "local_refit": ConfigField.mapping(LOCAL_REFIT_SCHEMA),
        "metric_thresholds": ConfigField.mapping(METRIC_THRESHOLDS_SCHEMA),
    },
    semantic_checks=(_calibrate_semantics,),
)

def _source_artifact_semantics(value: Mapping[str, object]) -> None:
    _path_safe_id(value["artifact_id"], name="source_artifact.artifact_id")
    _sha256(value["sha256"], name="source_artifact.sha256")


def _expectations_semantics(value: Mapping[str, object]) -> None:
    for name in ("camera_count", "image_width", "image_height"):
        _positive_integer(value[name], name=f"expectations.{name}")
    for name in (
        "camera_array_sha256",
        "shared_intrinsics_sha256",
        "normalization_sha256",
    ):
        _sha256(value[name], name=f"expectations.{name}")


def _export_semantics(value: Mapping[str, object]) -> None:
    _path_safe_id(value["bundle_id"], name="bundle_id")
    _nonempty_text(value["provider_backend"], name="provider_backend")
    _positive_integer(value["factor"], name="factor")
    _positive_integer(value["group_size"], name="group_size")
    artifacts = value["source_artifacts"]
    if not isinstance(artifacts, tuple) or not artifacts:
        raise SemanticConfigurationError("source_artifacts must be non-empty.")
    artifact_ids = tuple(
        _semantic_mapping(item, name="source_artifact")["artifact_id"]
        for item in artifacts
    )
    if len(set(artifact_ids)) != len(artifact_ids):
        raise SemanticConfigurationError("source_artifact ids must be unique.")


def _system_executable_semantics(value: Mapping[str, object]) -> None:
    _nonempty_text(value["root"], name="geometry_executable.root")
    _nonempty_text(value["path"], name="geometry_executable.path")
    _sha256(value["sha256"], name="geometry_executable.sha256")


SOURCE_ARTIFACT_SCHEMA = _mapping_schema(
    "source_artifact",
    {
        "artifact_id": ConfigField.of(str),
        "path": ConfigField.of(str),
        "sha256": ConfigField.of(str),
    },
    semantic_checks=(_source_artifact_semantics,),
)
EXPECTATIONS_SCHEMA = _mapping_schema(
    "expectations",
    {
        "camera_count": ConfigField.of(int),
        "image_width": ConfigField.of(int),
        "image_height": ConfigField.of(int),
        "camera_array_sha256": ConfigField.of(str),
        "shared_intrinsics_sha256": ConfigField.of(str),
        "normalization_sha256": ConfigField.of(str),
    },
    semantic_checks=(_expectations_semantics,),
)
SYSTEM_EXECUTABLE_SCHEMA = _mapping_schema(
    "system_executable",
    {
        "root": ConfigField.of(str),
        "path": ConfigField.of(str),
        "sha256": ConfigField.of(str),
    },
    semantic_checks=(_system_executable_semantics,),
)
EXPORT_SCHEMA = _mapping_schema(
    "synthetic.alignment.export_scene_provider",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "bundle_id": ConfigField.of(str),
        "provider_backend": ConfigField.of(str),
        "output_dir": ConfigField.of(str),
        "factor": ConfigField.of(int),
        "group_size": ConfigField.of(int),
        "external_asset_scope": ConfigField.of(str),
        "dataset_root": ConfigField.of(str),
        "cameras_bin": ConfigField.of(str),
        "images_bin": ConfigField.of(str),
        "points3d_bin": ConfigField.of(str),
        "original_image_dir": ConfigField.of(str),
        "factor_image_dir": ConfigField.of(str),
        "geometry_executable": ConfigField.mapping(SYSTEM_EXECUTABLE_SCHEMA),
        "geometry_bridge": ConfigField.of(str),
        "source_artifacts": ConfigField.sequence(
            ConfigField.mapping(SOURCE_ARTIFACT_SCHEMA)
        ),
        "expectations": ConfigField.mapping(EXPECTATIONS_SCHEMA),
    },
    semantic_checks=(_export_semantics,),
)
GEOMETRY_BRIDGE_SCHEMA = _mapping_schema(
    "synthetic.alignment.geometry_bridge",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "request": ConfigField.of(str),
        "output": ConfigField.of(str),
    },
)
VALIDATION_MATRIX_SCHEMA = _mapping_schema(
    "synthetic.validation_matrix",
    {
        "roots": ConfigField.mapping(ROOT_SCHEMA),
        "enabled": ConfigField.of(bool),
    },
)

SCHEMAS = {
    "synthetic.dataset.pipeline": PIPELINE_SCHEMA,
    "synthetic.dataset.blcs.feature_fit": FEATURE_FIT_SCHEMA,
    "synthetic.alignment.infer_ground_line_map": INFER_SCHEMA,
    "synthetic.alignment.fit_ground_courts": FIT_GROUND_SCHEMA,
    "synthetic.alignment.calibrate_court_alignment": CALIBRATE_SCHEMA,
    "synthetic.alignment.export_scene_provider": EXPORT_SCHEMA,
    "synthetic.alignment.geometry_bridge": GEOMETRY_BRIDGE_SCHEMA,
    "synthetic.validation_matrix": VALIDATION_MATRIX_SCHEMA,
}

SYNTHETIC_PATH_ROLE_MAP: Mapping[str, Mapping[str, PathRole]] = MappingProxyType(
    {
        "synthetic.dataset.pipeline": MappingProxyType(
            {
                "paths.source_root": PathRole.EXTERNAL_ASSET,
                "paths.artifact_root": PathRole.ARTIFACT,
                "paths.execution_root": PathRole.OUTPUT,
                "paths.dataset_root": PathRole.DATA,
                "paths.alignment_observations": PathRole.EXTERNAL_ASSET,
                "paths.render_jobs": PathRole.EXTERNAL_ASSET,
                "paths.pipeline_manifest": PathRole.OUTPUT,
                "paths.alignment_metrics": PathRole.ARTIFACT,
                "paths.dataset_plan": PathRole.ARTIFACT,
                "paths.render_manifest": PathRole.ARTIFACT,
                "paths.quality_metrics": PathRole.ARTIFACT,
                "paths.visualization": PathRole.OUTPUT,
                "renderer.working_directory": PathRole.EXTERNAL_ASSET,
            }
        ),
        "synthetic.dataset.blcs.feature_fit": MappingProxyType(
            {
                "source": PathRole.EXTERNAL_ASSET,
                "calibration_bundle": PathRole.ARTIFACT,
                "target_appearance": PathRole.ARTIFACT,
                "output_dir": PathRole.ARTIFACT,
                "runtime_pins": PathRole.EXTERNAL_ASSET,
                "nht_repository": PathRole.EXTERNAL_ASSET,
                "gsplat_repository": PathRole.EXTERNAL_ASSET,
                "worker_source": PathRole.PROJECT,
            }
        ),
        "synthetic.alignment.infer_ground_line_map": MappingProxyType(
            {
                "provider_bundle": PathRole.DATA,
                "line_checkpoint": PathRole.CHECKPOINT,
                "backbone_repository": PathRole.EXTERNAL_ASSET,
                "backbone_checkpoint": PathRole.EXTERNAL_ASSET,
                "output_dir": PathRole.DATA,
            }
        ),
        "synthetic.alignment.fit_ground_courts": MappingProxyType(
            {
                "ground_line_artifact": PathRole.DATA,
                "output_dir": PathRole.DATA,
            }
        ),
        "synthetic.alignment.calibrate_court_alignment": MappingProxyType(
            {
                "provider_bundle": PathRole.DATA,
                "ground_line_artifact": PathRole.DATA,
                "geometry_artifact": PathRole.DATA,
                "output_dir": PathRole.DATA,
                "line_checkpoint": PathRole.CHECKPOINT,
                "backbone_repository": PathRole.EXTERNAL_ASSET,
                "backbone_checkpoint": PathRole.EXTERNAL_ASSET,
            }
        ),
        "synthetic.alignment.export_scene_provider": MappingProxyType(
            {
                "output_dir": PathRole.DATA,
                "external_asset_scope": PathRole.EXTERNAL_ASSET,
                "dataset_root": PathRole.EXTERNAL_ASSET,
                "cameras_bin": PathRole.EXTERNAL_ASSET,
                "images_bin": PathRole.EXTERNAL_ASSET,
                "points3d_bin": PathRole.EXTERNAL_ASSET,
                "original_image_dir": PathRole.EXTERNAL_ASSET,
                "factor_image_dir": PathRole.EXTERNAL_ASSET,
                "geometry_bridge": PathRole.PROJECT,
            }
        ),
        "synthetic.alignment.geometry_bridge": MappingProxyType(
            {"request": PathRole.CACHE, "output": PathRole.CACHE}
        ),
        "synthetic.validation_matrix": MappingProxyType({}),
    }
)


def _mapping_value(value: Mapping[str, object], dotted_key: str) -> object:
    current: object = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping):
            raise AssertionError(
                f"Validated configuration parent for {dotted_key!r} is not a mapping."
            )
        current = current[part]
    return current


def _resolve_boundary_paths(
    boundary: str,
    values: Mapping[str, object],
    *,
    resolver: PathResolver,
) -> Mapping[str, Path]:
    roles = SYNTHETIC_PATH_ROLE_MAP[boundary]
    if boundary == "synthetic.dataset.pipeline":
        from src.synthetic_data_generation.dataset.pipeline import (
            PATH_PIPELINE_FIELDS,
            PathPipelineManifest,
        )

        raw_paths = _semantic_mapping(values["paths"], name="paths")
        manifest = PathPipelineManifest.from_config(raw_paths, resolver=resolver)
        resolved = {
            f"paths.{name}": getattr(manifest, name) for name in PATH_PIPELINE_FIELDS
        }
        working_directory = _mapping_value(values, "renderer.working_directory")
        resolved["renderer.working_directory"] = resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            cast(str, working_directory),
        )
        return MappingProxyType(resolved)

    resolved = {
        key: resolver.resolve(role, cast(str, _mapping_value(values, key)))
        for key, role in roles.items()
    }
    if boundary == "synthetic.alignment.export_scene_provider":
        asset_scope = resolved["external_asset_scope"]
        for key, role in roles.items():
            path = resolved[key]
            if (
                role is PathRole.EXTERNAL_ASSET
                and key != "external_asset_scope"
                and (path == asset_scope or not path.is_relative_to(asset_scope))
            ):
                raise PathContractError(
                    f"Configured external asset {key!r} is outside the export asset "
                    f"scope: {path} (scope: {asset_scope})."
                )
        artifacts = values["source_artifacts"]
        if not isinstance(artifacts, tuple):
            raise AssertionError("Validated source_artifacts are not a tuple.")
        for index, item in enumerate(artifacts):
            artifact = _semantic_mapping(item, name=f"source_artifacts[{index}]")
            artifact_path = resolver.resolve(
                PathRole.EXTERNAL_ASSET,
                cast(str, artifact["path"]),
            )
            if artifact_path == asset_scope or not artifact_path.is_relative_to(
                asset_scope
            ):
                raise PathContractError(
                    f"Configured source_artifacts[{index}] is outside the export "
                    f"asset scope: {artifact_path} (scope: {asset_scope})."
                )
    return MappingProxyType(resolved)


def _resolve_system_executables(
    boundary: str,
    values: Mapping[str, object],
) -> Mapping[str, VerifiedSystemExecutable]:
    if boundary != "synthetic.alignment.export_scene_provider":
        return MappingProxyType({})
    raw = _semantic_mapping(values["geometry_executable"], name="geometry_executable")
    executable = VerifiedSystemExecutable(
        root=Path(cast(str, raw["root"])),
        relative_path=Path(cast(str, raw["path"])),
        sha256=cast(str, raw["sha256"]),
    )
    return MappingProxyType({"geometry_executable": executable})


@dataclass(frozen=True, slots=True)
class SyntheticRuntimeConfig:
    """A closed configuration with role paths and typed system executables."""

    values: Mapping[str, object]
    resolver: PathResolver
    path_roles: Mapping[str, PathRole]
    resolved_paths: Mapping[str, Path]
    system_executables: Mapping[str, VerifiedSystemExecutable]

    def path(self, role: PathRole, key: str) -> Path:
        """Return one prevalidated path only through its declared role."""
        declared_role = self.path_roles[key]
        if declared_role is not role:
            raise PathContractError(
                f"Configured path {key!r} is declared as {declared_role.value}, "
                f"not {role.value}."
            )
        return self.resolved_paths[key]

    def system_executable(self, key: str) -> VerifiedSystemExecutable:
        """Return a content-pinned executable outside storage path roles."""
        return self.system_executables[key]


def add_path_roots_argument(parser: argparse.ArgumentParser) -> None:
    """Require the shared seven-role authority at every raw CLI boundary."""
    parser.add_argument(
        "--path-roots",
        required=True,
        help=(
            "JSON object containing exactly project_root, data_root, "
            "checkpoint_root, artifact_root, output_root, cache_root, and "
            "external_asset_root as absolute paths."
        ),
    )


def non_hydra_path_resolver(value: object) -> PathResolver:
    """Parse one explicit root contract without filesystem access or CWD use."""
    if type(value) is not str or not value:
        raise TypeError("path_roots must be a non-empty JSON string.")
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("path_roots must be valid JSON.") from error
    if not isinstance(raw, dict) or any(type(key) is not str for key in raw):
        raise TypeError("path_roots must decode to an object with string keys.")
    roots = ROOT_SCHEMA.validate(cast(Mapping[str, object], raw))
    for role in PathRole:
        root_name = f"{role.value}_root"
        root_value = roots[root_name]
        if type(root_value) is not str or not Path(root_value).is_absolute():
            raise ValueError(f"path_roots.{root_name} must be an absolute path.")
    project_root = cast(str, roots["project_root"])
    return PathResolver(
        RuntimePathRoots.from_mapping(
            roots,
            repository_root=Path(project_root),
        )
    )


def validate_config(boundary: str, config: DictConfig) -> SyntheticRuntimeConfig:
    """Validate one Hydra boundary without reading files or creating outputs."""
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"{boundary} configuration must resolve to a mapping.")
    schema = SCHEMAS[boundary]
    values = schema.validate(cast(Mapping[str, object], raw))
    roots_raw = values["roots"]
    if not isinstance(roots_raw, Mapping):
        raise AssertionError("Validated roots are not a mapping.")
    roots = RuntimePathRoots.from_mapping(roots_raw, repository_root=PROJECT_ROOT)
    resolver = PathResolver(roots)
    resolved_paths = _resolve_boundary_paths(boundary, values, resolver=resolver)
    system_executables = _resolve_system_executables(boundary, values)
    return SyntheticRuntimeConfig(
        values=values,
        resolver=resolver,
        path_roles=SYNTHETIC_PATH_ROLE_MAP[boundary],
        resolved_paths=resolved_paths,
        system_executables=system_executables,
    )


def _boundary_validator(boundary: str) -> Callable[[DictConfig], None]:
    def validate(config: DictConfig) -> None:
        validate_config(boundary, config)

    return validate


for _boundary in SCHEMAS:
    register_boundary_validator(_boundary, _boundary_validator(_boundary))
