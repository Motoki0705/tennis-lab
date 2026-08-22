"""Current-source audit policy and runtime boundary inventory.

The audit deliberately derives findings from the checked-out source instead of
persisting line-number-based snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "AuditInventory",
    "AuditRule",
    "BoundaryKind",
    "DEFAULT_AUDIT_INVENTORY",
    "EXPECTED_RUNTIME_BOUNDARIES",
    "RuntimeBoundary",
]


class AuditRule(StrEnum):
    """High-signal configuration/path practices rejected in current source."""

    GET_WITH_FALLBACK = "get-with-fallback"
    CHAINED_GET = "chained-get"
    GETATTR_WITH_FALLBACK = "getattr-with-fallback"
    SETDEFAULT = "setdefault"
    HYDRA_ABSOLUTE_PATH = "hydra-to-absolute-path"
    FILE_PARENT_INDEX = "file-parent-index"
    NULL_COALESCING = "null-coalescing"
    RUNTIME_PATH_LITERAL = "runtime-path-literal"
    RAW_PATH_CONSTRUCTION = "raw-configured-path"
    PROCESS_CWD = "process-cwd"


class BoundaryKind(StrEnum):
    """How a discoverable runtime boundary is invoked."""

    HYDRA = "hydra"
    ARGPARSE = "argparse"
    CALLABLE = "callable"
    SUBPROCESS_MODULE = "subprocess-module"


@dataclass(frozen=True, slots=True)
class RuntimeBoundary:
    """Record configuration-policy authorities for one runtime boundary."""

    domain: str
    module: str
    callable_name: str
    kind: BoundaryKind
    executable_module: bool
    validator_key: str | None
    validator_callable: str | None
    configuration_authority: str | None
    path_authority: str
    validation_target: str
    required_policy: str
    optional_policy: str
    default_authority: str
    precedence_authority: str


@dataclass(frozen=True, slots=True)
class AuditInventory:
    """Current audit rules and explicit runtime boundary contracts."""

    boundaries: tuple[RuntimeBoundary, ...] = field(default_factory=tuple)
    rules: tuple[AuditRule, ...] = field(default_factory=lambda: tuple(AuditRule))

    def __post_init__(self) -> None:
        if not self.rules or len(self.rules) != len(set(self.rules)):
            raise ValueError("Audit rules must be non-empty and unique.")
        boundary_keys = tuple(
            (boundary.module, boundary.callable_name) for boundary in self.boundaries
        )
        if len(boundary_keys) != len(set(boundary_keys)):
            raise ValueError("Runtime boundary inventory entries must be unique.")
        invalid_boundaries = tuple(
            boundary
            for boundary in self.boundaries
            if any(
                not value.strip()
                for value in (
                    boundary.domain,
                    boundary.module,
                    boundary.callable_name,
                    boundary.path_authority,
                    boundary.validation_target,
                    boundary.required_policy,
                    boundary.optional_policy,
                    boundary.default_authority,
                    boundary.precedence_authority,
                )
            )
        )
        if invalid_boundaries:
            raise ValueError(
                "Every runtime boundary needs explicit schema, path, field, "
                "default, and precedence authorities."
            )
        invalid_validator_pairs = tuple(
            boundary
            for boundary in self.boundaries
            if len(
                {
                    boundary.validator_key is None,
                    boundary.validator_callable is None,
                    boundary.configuration_authority is None,
                }
            )
            != 1
            or (
                boundary.configuration_authority is not None
                and not boundary.configuration_authority.strip()
            )
            or (
                boundary.validator_key is not None
                and not boundary.validator_key.strip()
            )
            or (
                boundary.validator_callable is not None
                and not boundary.validator_callable.strip()
            )
        )
        if invalid_validator_pairs:
            raise ValueError(
                "Runtime validator key, callable, and configuration authority "
                "must be declared as a complete binding."
            )


_PATH_AUTHORITY = "src.utils.configuration.paths.PathResolver.resolve"


_BOUNDARY_VALIDATOR_KEYS: Mapping[str, str] = {
    "src.synthetic_data_generation.scripts.run_scene_pipeline": "synthetic.scene_pipeline",
    "src.synthetic_data_generation.scripts.visualize_dataset": "synthetic.dataset_visualization",
    "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio": "ball.web_tool",
    "src.tasks.ball_detection.scripts.convert_web_dataset": "ball.web_tool",
    "src.tasks.ball_detection.scripts.eval": "ball.eval",
    "src.tasks.ball_detection.scripts.evaluate_manifest": "ball.evaluate_manifest",
    "src.tasks.ball_detection.scripts.preview_augmentation": "ball.preview",
    "src.tasks.ball_detection.scripts.preview_heatmaps": "ball.preview",
    "src.tasks.ball_detection.scripts.train": "ball.train",
    "src.tasks.ball_detection.scripts.train_staged": "ball.train_staged",
    "src.tasks.ball_detection.scripts.visualize": "ball.visualize",
    "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball": "ball.annotation",
    "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset": "ball.youtube",
    "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images": "ball.youtube",
    "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset": "ball.youtube",
    "src.tasks.court_detection.scripts.annotate_youtube_keypoints": "court_detection.annotate_youtube_keypoints",
    "src.tasks.court_detection.scripts.evaluate_homography_annotations": "court_detection.evaluate_homography_annotations",
    "src.tasks.court_detection.scripts.generate_line_masks": "court_detection.generate_line_masks",
    "src.tasks.court_detection.scripts.generate_masks": "court_detection.generate_masks",
    "src.tasks.court_detection.scripts.materialize_targets": "court_detection.materialize_targets",
    "src.tasks.court_detection.scripts.prepare_youtube_dataset": "court_detection.prepare_youtube_dataset",
    "src.tasks.court_detection.scripts.preview_augmentation": "court_detection.preview_augmentation",
    "src.tasks.court_detection.scripts.preview_heatmaps": "court_detection.preview_heatmaps",
    "src.tasks.court_detection.scripts.train": "court_detection.train",
    "src.tasks.court_detection.scripts.visualize": "court_detection.visualize",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "blcs.api_server",
    "src.tasks.blcs.scripts.generate_dataset": "blcs.generate_dataset",
    "src.tasks.blcs.scripts.preview_augmentation": "blcs.preview_augmentation",
    "src.tasks.blcs.scripts.train": "blcs.train",
    "src.tasks.blcs.scripts.visualize": "blcs.visualize",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "plcs.analyze_angle_velocity",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "plcs.analyze_dataset_distribution",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "plcs.analyze_loss_dominance",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "plcs.analyze_rotation_error_samples",
    "src.tasks.plcs.scripts.generate_dataset": "plcs.generate_dataset",
    "src.tasks.plcs.scripts.preview_augmentation": "plcs.preview_augmentation",
    "src.tasks.plcs.scripts.train": "plcs.train",
    "src.tasks.plcs.scripts.visualize": "plcs.visualize",
    "src.tasks.slcs.scripts.analyze_predictions": "slcs.analyze_predictions",
    "src.tasks.slcs.scripts.evaluate": "slcs.evaluate",
    "src.tasks.slcs.scripts.make_splits": "slcs.make_splits",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "slcs.precompute_dino_tokens",
    "src.tasks.slcs.scripts.predict_clip": "slcs.predict_clip",
    "src.tasks.slcs.scripts.train": "slcs.train",
    "src.tennis_scene.scripts.clip_studio": "tennis_scene.clip_studio",
    "src.tennis_scene.scripts.export_clips": "tennis_scene.export_clips",
    "src.tennis_scene.scripts.generate_dataset": "tennis_scene.generate_dataset",
    "src.tennis_scene.scripts.run_pipeline": "tennis_scene.pipeline",
    "src.tennis_scene.scripts.visualization": "tennis_scene.visualization",
    "src.tennis_scene.scripts.visualize_tasks": "tennis_scene.visualize_tasks",
    "src.submodules.scripts.demo_gvhmr": "submodules.demo_gvhmr",
}

_BOUNDARY_VALIDATOR_CALLABLES: Mapping[str, str] = {
    "src.synthetic_data_generation.scripts.run_scene_pipeline": (
        "src.synthetic_data_generation.configuration.validate_scene_pipeline_boundary"
    ),
    "src.synthetic_data_generation.scripts.visualize_dataset": (
        "src.synthetic_data_generation.visualization.configuration."
        "validate_dataset_visualization_boundary"
    ),
    "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio": "src.tasks.ball_detection.configuration.validate_web_tool",
    "src.tasks.ball_detection.scripts.convert_web_dataset": "src.tasks.ball_detection.configuration.validate_web_tool",
    "src.tasks.ball_detection.scripts.eval": "src.tasks.ball_detection.configuration.validate_eval",
    "src.tasks.ball_detection.scripts.evaluate_manifest": "src.tasks.ball_detection.configuration.validate_manifest_boundary",
    "src.tasks.ball_detection.scripts.preview_augmentation": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.scripts.preview_heatmaps": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.scripts.train": "src.tasks.ball_detection.configuration.validate_training",
    "src.tasks.ball_detection.scripts.train_staged": "src.tasks.ball_detection.configuration.validate_training",
    "src.tasks.ball_detection.scripts.visualize": "src.tasks.ball_detection.configuration.validate_visualization",
    "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball": "src.tasks.ball_detection.configuration.validate_annotation_boundary",
    "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.court_detection.scripts.annotate_youtube_keypoints": "src.tasks.court_detection.scripts.annotate_youtube_keypoints._validate_boundary",
    "src.tasks.court_detection.scripts.evaluate_homography_annotations": "src.tasks.court_detection.scripts.evaluate_homography_annotations._validate_boundary",
    "src.tasks.court_detection.scripts.generate_line_masks": "src.tasks.court_detection.scripts.generate_line_masks._validate_boundary",
    "src.tasks.court_detection.scripts.generate_masks": "src.tasks.court_detection.scripts.generate_masks._validate_boundary",
    "src.tasks.court_detection.scripts.materialize_targets": "src.tasks.court_detection.scripts.materialize_targets._validate_boundary",
    "src.tasks.court_detection.scripts.prepare_youtube_dataset": "src.tasks.court_detection.scripts.prepare_youtube_dataset._validate_boundary",
    "src.tasks.court_detection.scripts.preview_augmentation": "src.tasks.court_detection.scripts.preview_augmentation._validate_boundary",
    "src.tasks.court_detection.scripts.preview_heatmaps": "src.tasks.court_detection.scripts.preview_heatmaps._validate_boundary",
    "src.tasks.court_detection.scripts.train": "src.tasks.court_detection.configuration.validate_train_boundary",
    "src.tasks.court_detection.scripts.visualize": "src.tasks.court_detection.scripts.visualize._validate_boundary",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "src.tasks.blcs.configuration.validate_api_boundary",
    "src.tasks.blcs.scripts.generate_dataset": "src.tasks.blcs.configuration.validate_generation_boundary",
    "src.tasks.blcs.scripts.preview_augmentation": "src.tasks.blcs.configuration.validate_preview_boundary",
    "src.tasks.blcs.scripts.train": "src.tasks.blcs.configuration._validate_training_for_hydra",
    "src.tasks.blcs.scripts.visualize": "src.tasks.blcs.configuration.validate_visualization_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "src.tasks.plcs.configuration._validate_angle_velocity_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "src.tasks.plcs.configuration._validate_distribution_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "src.tasks.plcs.configuration._validate_loss_dominance_boundary",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "src.tasks.plcs.configuration._validate_rotation_error_boundary",
    "src.tasks.plcs.scripts.generate_dataset": "src.tasks.plcs.generate_dataset.config._validate_boundary",
    "src.tasks.plcs.scripts.preview_augmentation": "src.tasks.plcs.configuration._validate_preview_boundary",
    "src.tasks.plcs.scripts.train": "src.tasks.plcs.configuration._validate_training_boundary",
    "src.tasks.plcs.scripts.visualize": "src.tasks.plcs.configuration._validate_visualization_boundary",
    "src.tasks.slcs.scripts.analyze_predictions": "src.tasks.slcs.configuration.validate_analysis_boundary",
    "src.tasks.slcs.scripts.evaluate": "src.tasks.slcs.configuration.validate_evaluation_boundary",
    "src.tasks.slcs.scripts.make_splits": "src.tasks.slcs.configuration.validate_split_boundary",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "src.tasks.slcs.configuration.validate_precompute_boundary",
    "src.tasks.slcs.scripts.predict_clip": "src.tasks.slcs.configuration.validate_prediction_boundary",
    "src.tasks.slcs.scripts.train": "src.tasks.slcs.configuration.validate_training_boundary",
    "src.tennis_scene.scripts.clip_studio": "src.tennis_scene.configuration.validate_clip_studio_boundary",
    "src.tennis_scene.scripts.export_clips": "src.tennis_scene.configuration.validate_export_clips_boundary",
    "src.tennis_scene.scripts.generate_dataset": "src.tennis_scene.configuration.validate_generate_dataset_boundary",
    "src.tennis_scene.scripts.run_pipeline": "src.tennis_scene.configuration.validate_pipeline_boundary",
    "src.tennis_scene.scripts.visualization": "src.tennis_scene.configuration.validate_visualization_boundary",
    "src.tennis_scene.scripts.visualize_tasks": "src.tennis_scene.configuration.validate_visualize_tasks_boundary",
    "src.submodules.scripts.demo_gvhmr": "src.submodules.scripts.demo_gvhmr._validate_boundary",
}


def _runtime_boundary(
    domain: str,
    module: str,
    *,
    callable_name: str = "main",
) -> RuntimeBoundary:
    validator_key = _BOUNDARY_VALIDATOR_KEYS.get(module)
    validator_callable = _BOUNDARY_VALIDATOR_CALLABLES.get(module)
    return RuntimeBoundary(
        domain=domain,
        module=module,
        callable_name=callable_name,
        kind=BoundaryKind.HYDRA,
        executable_module=True,
        validator_key=validator_key,
        validator_callable=validator_callable,
        configuration_authority=validator_callable,
        path_authority=_PATH_AUTHORITY,
        validation_target="validated typed runtime contract before side effects",
        required_policy="present after composition; missing values are errors",
        optional_policy="declared optional and absent without value synthesis",
        default_authority="composed configuration only; no Python runtime default",
        precedence_authority="single composed value; no fallback or alias precedence",
    )


_NON_HYDRA_BOUNDARY_BINDINGS: Mapping[str, tuple[str, str]] = {
    "src.automation.chatgpt_mcp.cli": (
        "automation.chatgpt_mcp",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
}


def _non_hydra_boundary(
    module: str,
    callable_name: str,
    *,
    kind: BoundaryKind = BoundaryKind.ARGPARSE,
    domain: str = "synthetic_data_generation",
    executable_module: bool = False,
) -> RuntimeBoundary:
    validator_key, validator_callable = _NON_HYDRA_BOUNDARY_BINDINGS[module]
    return RuntimeBoundary(
        domain=domain,
        module=module,
        callable_name=callable_name,
        kind=kind,
        executable_module=executable_module,
        validator_key=validator_key,
        validator_callable=validator_callable,
        configuration_authority=validator_callable,
        path_authority=validator_callable,
        validation_target="validated typed runtime contract before side effects",
        required_policy="all declared path arguments are present",
        optional_policy="optional non-path values use an explicit typed contract",
        default_authority="caller-owned explicit values only; no boundary fallback",
        precedence_authority="one role/direction declaration per explicit path",
    )


_RUNTIME_BOUNDARIES = (
    _non_hydra_boundary(
        "src.automation.chatgpt_mcp.cli",
        "main",
        domain="automation",
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.run_scene_pipeline",
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.visualize_dataset",
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.convert_web_dataset"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.eval"),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.evaluate_manifest"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.preview_augmentation"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.preview_heatmaps"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.train"),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.train_staged"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.visualize"),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset",
    ),
    _runtime_boundary("blcs", "src.tasks.blcs.generate_dataset.api_server.__main__"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.generate_dataset"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.preview_augmentation"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.train"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.visualize"),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.annotate_youtube_keypoints",
    ),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.evaluate_homography_annotations",
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.generate_line_masks"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.generate_masks"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.materialize_targets"
    ),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.prepare_youtube_dataset",
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.preview_augmentation"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.preview_heatmaps"
    ),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.train"),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.visualize"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_angle_velocity"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_loss_dominance"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.generate_dataset"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.preview_augmentation"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.train"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.visualize"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.analyze_predictions"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.evaluate"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.make_splits"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.precompute_dino_tokens"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.predict_clip"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.train"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.clip_studio"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.export_clips"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.generate_dataset"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.run_pipeline"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualization"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualize_tasks"),
    _runtime_boundary("submodules", "src.submodules.scripts.demo_gvhmr"),
)


EXPECTED_RUNTIME_BOUNDARIES = _RUNTIME_BOUNDARIES


DEFAULT_AUDIT_INVENTORY = AuditInventory(
    boundaries=_RUNTIME_BOUNDARIES,
)
