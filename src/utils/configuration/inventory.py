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
    "EXTERNAL_RUNTIME_MODULES",
    "RuntimeBoundary",
]


EXTERNAL_RUNTIME_MODULES: Mapping[str, str] = {
    "src.tennis_scene.visualization.blender.shots.ball_rays": (
        "Blender bundled Python; displays saved ball rays without repo imports."
    ),
    "src.tennis_scene.visualization.blender.shots.validate_rays": (
        "Blender bundled Python; validates evaluated rays without repo imports."
    ),
    "src.tennis_scene.visualization.blender.shots.reveal": (
        "Blender bundled Python; consumes a plain-file bundle without repo imports."
    ),
    "src.tennis_scene.visualization.blender.shots.validate_reveal": (
        "Blender bundled Python; validates a saved scene without repo imports."
    ),
}


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
    routes: tuple[tuple[str, str, str], ...] = ()


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
    "src.tasks.ball_refiner_3d.scripts.train": "ball_refiner_3d.train",
    "src.tasks.ball_refiner_3d.scripts.generate_dataset": "ball_refiner_3d.generate_dataset",
    "src.synthetic_data_generation.scripts.run_appearance_variant": "synthetic.appearance_variant",
    "src.synthetic_data_generation.scripts.generate_publication_visualizations": "synthetic.publication_visualization",
    "src.synthetic_data_generation.scripts.run_scene_pipeline": "synthetic.scene_pipeline",
    "src.synthetic_data_generation.scripts.visualize_dataset": "synthetic.dataset_visualization",
    "src.synthetic_data_generation.scripts.visualize_scene_reveal": "synthetic.scene_reveal",
    "src.tasks.ball_detection.evaluation.heatmap_cli": "ball.eval",
    "src.tasks.ball_detection.evaluation.manifest_cli": "ball.evaluate_manifest",
    "src.tasks.ball_detection.scripts.generate_dataset": "ball_detection.generate_dataset",
    "src.tasks.ball_detection.visualization.preview.spatial": "ball.preview",
    "src.tasks.ball_detection.visualization.preview.heatmaps": "ball.preview",
    "src.tasks.ball_detection.training.convnext_heatmap.cli": "ball.train",
    "src.tasks.ball_detection.training.hybrid.runner": "ball.hybrid",
    "src.tasks.ball_detection.scripts.visualize": "ball.visualize",
    "src.tasks.ball_detection.scripts.visualize_annotation_correction": "ball.annotation_correction",
    "src.tasks.ball_detection.scripts.visualize_learning_stages": "ball.learning_stages",
    "src.tasks.court_detection.scripts.preview_augmentation": "court_detection.preview_augmentation",
    "src.tasks.court_detection.scripts.train": "court_detection.train",
    "src.tasks.court_detection.scripts.visualize": "court_detection.visualize",
    "src.tasks.court_detection.scripts.visualize_learning_stages": "court_detection.learning_stages",
    "src.tasks.player_detection.scripts.generate_dataset": "player_detection.generate_dataset",
    "src.tasks.player_detection.scripts.train": "player_detection.train",
    "src.tasks.player_detection.scripts.export_checkpoint": "player_detection.export_checkpoint",
    "src.tasks.player_detection.scripts.evaluate": "player_detection.evaluate",
    "src.tasks.player_detection.scripts.preview_dataset": "player_detection.preview_dataset",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "blcs.api_server",
    "src.tasks.blcs.scripts.generate_dataset": "blcs.generate_dataset",
    "src.tasks.blcs.scripts.generate_dataset_samples": "blcs.generate_dataset_samples",
    "src.tasks.blcs.scripts.preview_augmentation": "blcs.preview_augmentation",
    "src.tasks.blcs.scripts.train": "blcs.train",
    "src.tasks.blcs.scripts.visualize": "blcs.visualize",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "plcs.analyze_angle_velocity",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "plcs.analyze_dataset_distribution",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "plcs.analyze_loss_dominance",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "plcs.analyze_rotation_error_samples",
    "src.tasks.plcs.scripts.generate_dataset": "plcs.generate_dataset",
    "src.tasks.plcs.scripts.generate_dataset_samples": "plcs.generate_dataset_samples",
    "src.tasks.plcs.scripts.preview_augmentation": "plcs.preview_augmentation",
    "src.tasks.plcs.scripts.train": "plcs.train",
    "src.tasks.plcs.scripts.visualize": "plcs.visualize",
    "src.tasks.slcs.scripts.analyze_predictions": "slcs.analyze_predictions",
    "src.tasks.slcs.scripts.evaluate": "slcs.evaluate",
    "src.tasks.slcs.scripts.generate_dataset": "slcs.generate_dataset",
    "src.tasks.slcs.scripts.make_splits": "slcs.make_splits",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "slcs.precompute_dino_tokens",
    "src.tasks.slcs.scripts.predict_clip": "slcs.predict_clip",
    "src.tasks.slcs.scripts.train": "slcs.train",
    "src.tennis_scene.scripts.clip_studio": "tennis_scene.clip_studio",
    "src.tennis_scene.scripts.export_blender": "tennis_scene.blender_export",
    "src.tennis_scene.scripts.export_blender_rays": "tennis_scene.export_blender_rays",
    "src.tennis_scene.scripts.generate_dataset": "tennis_scene.generate_dataset",
    "src.tennis_scene.scripts.run_pipeline": "tennis_scene.pipeline",
    "src.tennis_scene.scripts.preflight": "tennis_scene.preflight",
    "src.tennis_scene.scripts.visualization": "tennis_scene.visualization",
    "src.tennis_scene.scripts.visualize_tasks": "tennis_scene.visualize_tasks",
}

_BOUNDARY_VALIDATOR_CALLABLES: Mapping[str, str] = {
    "src.tasks.ball_refiner_3d.scripts.train": "src.tasks.ball_refiner_3d.configuration.training.validate_training",
    "src.tasks.ball_refiner_3d.scripts.generate_dataset": "src.tasks.ball_refiner_3d.configuration.generation.validate_generation",
    "src.synthetic_data_generation.scripts.run_appearance_variant": "src.synthetic_data_generation.appearance.configuration.validate_appearance_boundary",
    "src.synthetic_data_generation.scripts.generate_publication_visualizations": "src.synthetic_data_generation.visualization.publication.configuration.validate_publication_boundary",
    "src.synthetic_data_generation.scripts.run_scene_pipeline": "src.synthetic_data_generation.configuration.validate_scene_pipeline_boundary",
    "src.synthetic_data_generation.scripts.visualize_dataset": "src.synthetic_data_generation.visualization.configuration.validate_dataset_visualization_boundary",
    "src.synthetic_data_generation.scripts.visualize_scene_reveal": "src.synthetic_data_generation.visualization.reveal.configuration.validate_scene_reveal_boundary",
    "src.tasks.ball_detection.evaluation.heatmap_cli": "src.tasks.ball_detection.configuration.validate_eval",
    "src.tasks.ball_detection.evaluation.manifest_cli": "src.tasks.ball_detection.configuration.validate_manifest_boundary",
    "src.tasks.ball_detection.scripts.generate_dataset": "src.tasks.ball_detection.generate_dataset.frame_store.config.validate_generate_boundary",
    "src.tasks.ball_detection.visualization.preview.spatial": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.visualization.preview.heatmaps": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.training.convnext_heatmap.cli": "src.tasks.ball_detection.configuration.validate_training",
    "src.tasks.ball_detection.training.hybrid.runner": "src.tasks.ball_detection.training.hybrid.configuration.validate_training_boundary",
    "src.tasks.ball_detection.scripts.visualize": "src.tasks.ball_detection.configuration.validate_visualization",
    "src.tasks.ball_detection.scripts.visualize_annotation_correction": "src.tasks.ball_detection.scripts.visualize_annotation_correction._validate_boundary",
    "src.tasks.ball_detection.scripts.visualize_learning_stages": "src.tasks.ball_detection.visualization.presentation.configuration.validate_learning_stages_boundary",
    "src.tasks.court_detection.scripts.preview_augmentation": "src.tasks.court_detection.scripts.preview_augmentation._validate_boundary",
    "src.tasks.court_detection.scripts.train": "src.tasks.court_detection.training.runner.validate_train_boundary",
    "src.tasks.court_detection.scripts.visualize": "src.tasks.court_detection.scripts.visualize._validate_boundary",
    "src.tasks.court_detection.scripts.visualize_learning_stages": "src.tasks.court_detection.visualization.presentation.configuration.validate_learning_stages_boundary",
    "src.tasks.player_detection.scripts.generate_dataset": "src.tasks.player_detection.configuration.validate_generate_boundary",
    "src.tasks.player_detection.scripts.train": "src.tasks.player_detection.configuration.validate_train_boundary",
    "src.tasks.player_detection.scripts.export_checkpoint": "src.tasks.player_detection.configuration.validate_export_boundary",
    "src.tasks.player_detection.scripts.evaluate": "src.tasks.player_detection.configuration.validate_evaluate_boundary",
    "src.tasks.player_detection.scripts.preview_dataset": "src.tasks.player_detection.configuration.validate_preview_boundary",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "src.tasks.blcs.configuration.validate_api_boundary",
    "src.tasks.blcs.scripts.generate_dataset": "src.tasks.blcs.configuration.validate_generation_boundary",
    "src.tasks.blcs.scripts.generate_dataset_samples": "src.tasks.blcs.generate_dataset.samples.validate_dataset_samples_boundary",
    "src.tasks.blcs.scripts.preview_augmentation": "src.tasks.blcs.configuration.validate_preview_boundary",
    "src.tasks.blcs.scripts.train": "src.tasks.blcs.configuration._validate_training_for_hydra",
    "src.tasks.blcs.scripts.visualize": "src.tasks.blcs.configuration.validate_visualization_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "src.tasks.plcs.configuration._validate_angle_velocity_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "src.tasks.plcs.configuration._validate_distribution_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "src.tasks.plcs.configuration._validate_loss_dominance_boundary",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "src.tasks.plcs.configuration._validate_rotation_error_boundary",
    "src.tasks.plcs.scripts.generate_dataset": "src.tasks.plcs.generate_dataset.config._validate_boundary",
    "src.tasks.plcs.scripts.generate_dataset_samples": "src.tasks.plcs.generate_dataset.samples.validate_dataset_samples_boundary",
    "src.tasks.plcs.scripts.preview_augmentation": "src.tasks.plcs.configuration._validate_preview_boundary",
    "src.tasks.plcs.scripts.train": "src.tasks.plcs.configuration._validate_training_boundary",
    "src.tasks.plcs.scripts.visualize": "src.tasks.plcs.configuration._validate_visualization_boundary",
    "src.tasks.slcs.scripts.analyze_predictions": "src.tasks.slcs.configuration.validate_analysis_boundary",
    "src.tasks.slcs.scripts.evaluate": "src.tasks.slcs.configuration.validate_evaluation_boundary",
    "src.tasks.slcs.scripts.generate_dataset": "src.tasks.slcs.configuration.validate_generation_boundary",
    "src.tasks.slcs.scripts.make_splits": "src.tasks.slcs.configuration.validate_split_boundary",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "src.tasks.slcs.configuration.validate_precompute_boundary",
    "src.tasks.slcs.scripts.predict_clip": "src.tasks.slcs.configuration.validate_prediction_boundary",
    "src.tasks.slcs.scripts.train": "src.tasks.slcs.configuration.validate_training_boundary",
    "src.tennis_scene.scripts.clip_studio": "src.tennis_scene.configuration.validate_clip_studio_boundary",
    "src.tennis_scene.scripts.export_blender": "src.tennis_scene.configuration.validate_blender_export_boundary",
    "src.tennis_scene.scripts.export_blender_rays": "src.tennis_scene.visualization.blender.export.rays_configuration.validate_rays_boundary",
    "src.tennis_scene.scripts.generate_dataset": "src.tennis_scene.configuration.validate_generate_dataset_boundary",
    "src.tennis_scene.scripts.run_pipeline": "src.tennis_scene.configuration.validate_pipeline_boundary",
    "src.tennis_scene.scripts.preflight": "src.tennis_scene.configuration.validate_pipeline_boundary",
    "src.tennis_scene.scripts.visualization": "src.tennis_scene.configuration.validate_visualization_boundary",
    "src.tennis_scene.scripts.visualize_tasks": "src.tennis_scene.configuration.validate_visualize_tasks_boundary",
}


def _runtime_boundary(
    domain: str,
    module: str,
    *,
    callable_name: str = "main",
    path_authority: str = _PATH_AUTHORITY,
    executable_module: bool = True,
) -> RuntimeBoundary:
    validator_key = _BOUNDARY_VALIDATOR_KEYS.get(module)
    validator_callable = _BOUNDARY_VALIDATOR_CALLABLES.get(module)
    return RuntimeBoundary(
        domain=domain,
        module=module,
        callable_name=callable_name,
        kind=BoundaryKind.HYDRA,
        executable_module=executable_module,
        validator_key=validator_key,
        validator_callable=validator_callable,
        configuration_authority=validator_callable,
        path_authority=path_authority,
        validation_target="validated typed runtime contract before side effects",
        required_policy="present after composition; missing values are errors",
        optional_policy="declared optional and absent without value synthesis",
        default_authority="composed configuration only; no Python runtime default",
        precedence_authority="single composed value; no fallback or alias precedence",
    )


_SLCS_REAL_RGB_ENTRYPOINTS = ("evaluate_run",)


_NON_HYDRA_BOUNDARY_BINDINGS: Mapping[str, tuple[str, str]] = {
    "src.utils.sample_dataset.__init__": (
        "sample_dataset.download",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.utils.model_weights.__init__": (
        "model_weights.download",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.annotation.__main__": (
        "annotation.workflow.cli",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.base.annotation.execution.__main__": (
        "base.annotation.execution.cli",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.base.annotation.workers.__main__": (
        "base.annotation.workers.cli",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.base.annotation.screening.__main__": (
        "base.annotation.screening.cli",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.base.annotation.workers.review.__main__": (
        "base.annotation.review_worker.cli",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_detection.scripts.train_blcs_ablation": (
        "blcs.ablation.train",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_detection.scripts.evaluate_blcs_ablation": (
        "blcs.ablation.evaluation",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_detection.visualization.review.cli": ("ball_detection.review_dataset", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.visualization.inference.cli": ("ball_detection.inference_ui", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.training.dpt_pretrain.runner": ("ball_detection.pretrain_mdd_dpt", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.evaluation.hybrid_cli": ("ball_detection.evaluate_hybrid", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.training.coordinates.finetune": ("ball_detection.posttrain_mdd_query", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.training.coordinates.full_model": ("ball_detection.full_model_mdd_query", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.training.annotation.__main__": ("ball_detection.annotation_learning", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.player_detection.training.annotation.__main__": ("player_detection.annotation_learning", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.player_detection.scripts.preview_predictions": ("player_detection.preview_predictions", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.visualization.preview.video": ("ball_detection.preview_posttraining_augmentation", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),
    "src.tasks.ball_detection.evaluation.coordinate_cli": (
        "ball_detection.evaluate_mdd_coordinates", "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_detection.scripts.prepare_data": (
        "ball_detection.prepare_mdd_training", "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_detection.training.coordinates.scratch": (
        "ball_detection.train_mdd_pose", "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_refiner_3d.scripts.review_dataset": (
        "ball_refiner_3d.review_dataset",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.submodules.visualization.review.__main__": (
        "submodules.pose_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.submodules.scripts.install_smpl_assets": (
        "submodules.install_smpl_assets",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tennis_scene.clip_studio.__main__": (
        "tennis_scene.clip_studio_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_refiner_3d.scripts.evaluate": (
        "ball_refiner_3d.evaluate",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.ball_refiner_3d.scripts.predict": (
        "ball_refiner_3d.predict",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.player_detection.annotation.pose.__main__": (
        "player_detection.annotation.pose",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    **{
        f"src.tasks.slcs.scripts.{script}": (
            f"slcs.{script}",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        )
        for script in _SLCS_REAL_RGB_ENTRYPOINTS
    },
    "src.tasks.court_detection.scripts.audit_hybrid_inference": (
        "court_detection.hybrid_inference_audit",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.court_detection.scripts.evaluate_ablation": (
        "court_detection.ablation_evaluation",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.court_detection.scripts.compare_ablation": (
        "court_detection.ablation_comparison",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.court_detection.scripts.profile_ablation": (
        "court_detection.ablation_profile",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.base.scripts.inference_worker": (
        "base.inference_worker",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    **{
        f"src.tasks.{task}.scripts.{script}": (
            f"{task}.{script}",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        )
        for task in ("court_detection",)
        for script in ("review_dataset", "inference_ui")
    },
    "src.synthetic_data_generation.scripts.edit_alignment": (
        "synthetic.manual_court_alignment",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.synthetic_data_generation.scripts.compact_court_storage": (
        "synthetic.court_storage_compaction",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.synthetic_data_generation.scripts.review_court_dataset": (
        "synthetic.court_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.plcs.scripts.review_accad_motion": (
        "plcs.accad_motion_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.player_detection.scripts.migrate_store": (
        "player_detection.migrate_store",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.player_detection.scripts.review_dataset": (
        "player_detection.review_dataset",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.plcs.scripts.review_dataset": (
        "plcs.dataset_scene_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.blcs.scripts.review_dataset": (
        "blcs.dataset_scene_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.slcs.scripts.review_dataset": (
        "slcs.dataset_scene_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.person_tracking.scripts.migrate_features": (
        "person_tracking.migrate_features",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.person_tracking.scripts.review_dataset": (
        "person_tracking.dataset_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.player_association.scripts.review_dataset": (
        "player_association.dataset_review",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.plcs.scripts.serve_inference_ui": (
        "plcs.inference_ui",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.blcs.scripts.inference_ui": (
        "blcs.inference_ui",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tennis_scene.scripts.visualize_component_store": (
        "tennis_scene.component_gallery",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.court_side.scripts.benchmark_synthetic": (
        "court_side.benchmark_synthetic",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
    "src.tasks.court_side.scripts.review_dataset": (
        "court_side.dataset_review",
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
    optional_policy: str = "optional non-path values use an explicit typed contract",
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
        required_policy="all required path arguments are present",
        optional_policy=optional_policy,
        default_authority="caller-owned explicit values only; no boundary fallback",
        precedence_authority="one role/direction declaration per explicit path",
    )


_RUNTIME_BOUNDARIES: tuple[RuntimeBoundary, ...] = (
    _non_hydra_boundary(
        "src.tasks.annotation.__main__", "main",
        domain="base", executable_module=True,
        optional_policy="runtime/candidates/operation/configuration are required by their subcommands; schema has no path input; test_root is an explicit test-only boundary",
    ),
    _non_hydra_boundary(
        "src.tasks.base.annotation.execution.__main__", "main",
        domain="base", executable_module=True,
        optional_policy="queued selects the shared queue owner; lock/resource FDs apply only to CPU stages; test_root selects test-only storage",
    ),
    _non_hydra_boundary(
        "src.tasks.base.annotation.workers.__main__", "main",
        domain="base", executable_module=True,
        optional_policy="test_root selects an explicit test-only layout; production uses the shared main-repository root",
    ),
    _non_hydra_boundary(
        "src.tasks.base.annotation.screening.__main__", "main",
        domain="base", executable_module=True,
        optional_policy="decision is required only for check; interval arguments are required only for images",
    ),
    _non_hydra_boundary(
        "src.tasks.base.annotation.workers.review.__main__", "main",
        domain="base", executable_module=True,
    ),
    _non_hydra_boundary("src.submodules.visualization.review.__main__", "main", domain="submodules", executable_module=True),
    _non_hydra_boundary(
        "src.submodules.scripts.install_smpl_assets", "main", domain="submodules",
        executable_module=True,
        optional_policy="gvhmr_root is required when installing; --list may omit it, but any supplied checkout is validated",
    ),
    _non_hydra_boundary("src.tennis_scene.clip_studio.__main__", "main", domain="tennis_scene", executable_module=True),
    _non_hydra_boundary("src.tasks.ball_refiner_3d.scripts.evaluate", "main", domain="ball_refiner_3d", executable_module=True),
    _non_hydra_boundary("src.tasks.ball_refiner_3d.scripts.predict", "main", domain="ball_refiner_3d", executable_module=True),
    _non_hydra_boundary("src.tasks.ball_refiner_3d.scripts.review_dataset", "main", domain="ball_refiner_3d", executable_module=True),
    _runtime_boundary("ball_refiner_3d", "src.tasks.ball_refiner_3d.scripts.train"),
    _runtime_boundary("ball_refiner_3d", "src.tasks.ball_refiner_3d.scripts.generate_dataset"),
    _non_hydra_boundary("src.tasks.player_detection.annotation.pose.__main__", "main",
                        domain="player_detection", executable_module=True),
    _non_hydra_boundary(
        "src.tasks.court_detection.scripts.audit_hybrid_inference",
        "main",
        domain="court_detection",
        executable_module=True,
        optional_policy="scene_root may be omitted; supplied paths are validated before side effects",
    ),
    _non_hydra_boundary(
        "src.tasks.court_detection.scripts.evaluate_ablation", "main",
        domain="court_detection", executable_module=True,
        optional_policy="data/checkpoint/external-asset roots may be omitted; checkpoint config remains authoritative otherwise",
    ),
    _non_hydra_boundary(
        "src.tasks.court_detection.scripts.compare_ablation", "main",
        domain="court_detection", executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.court_detection.scripts.profile_ablation", "main",
        domain="court_detection", executable_module=True,
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.run_appearance_variant",
        path_authority="src.synthetic_data_generation.appearance.configuration.require_absolute_path",
    ),
    _non_hydra_boundary(
        "src.tasks.base.scripts.inference_worker",
        "main",
        domain="base",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.synthetic_data_generation.scripts.compact_court_storage",
        "main",
        domain="synthetic_data_generation",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.synthetic_data_generation.scripts.edit_alignment",
        "main",
        domain="manual_court_alignment",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.synthetic_data_generation.scripts.review_court_dataset",
        "main",
        domain="synthetic_data_generation",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.plcs.scripts.review_accad_motion",
        "main",
        domain="plcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.plcs.scripts.review_dataset",
        "main",
        domain="plcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.ball_detection.scripts.train_blcs_ablation",
        "main",
        domain="blcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.ball_detection.scripts.evaluate_blcs_ablation",
        "main",
        domain="blcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.blcs.scripts.review_dataset",
        "main",
        domain="blcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.slcs.scripts.review_dataset",
        "main",
        domain="slcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.person_tracking.scripts.migrate_features", "main",
        domain="person_tracking", executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.person_tracking.scripts.review_dataset",
        "main",
        domain="person_tracking",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.player_association.scripts.review_dataset",
        "main",
        domain="player_association",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.plcs.scripts.serve_inference_ui",
        "main",
        domain="plcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.blcs.scripts.inference_ui",
        "main",
        domain="blcs",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tennis_scene.scripts.visualize_component_store",
        "main",
        domain="tennis_scene",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.court_side.scripts.benchmark_synthetic",
        "main",
        domain="court_side",
        executable_module=True,
    ),
    _non_hydra_boundary(
        "src.tasks.court_side.scripts.review_dataset",
        "main",
        domain="court_side",
        executable_module=True,
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.generate_publication_visualizations",
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
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.visualize_scene_reveal",
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.evaluation.heatmap_cli", executable_module=False),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.generate_dataset"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.evaluation.manifest_cli", executable_module=False
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.visualization.preview.spatial", executable_module=False
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.visualization.preview.heatmaps", executable_module=False
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.training.convnext_heatmap.cli", executable_module=False),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.training.hybrid.runner", executable_module=False),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.visualize"),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.visualize_annotation_correction"),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.visualize_learning_stages"),
    _runtime_boundary("blcs", "src.tasks.blcs.generate_dataset.api_server.__main__"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.generate_dataset"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.generate_dataset_samples"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.preview_augmentation"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.train"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.visualize"),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.preview_augmentation"
    ),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.train"),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.visualize"),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.visualize_learning_stages"),
    _runtime_boundary(
        "player_detection", "src.tasks.player_detection.scripts.generate_dataset"
    ),
    _runtime_boundary("player_detection", "src.tasks.player_detection.scripts.train"),
    _non_hydra_boundary(
        "src.tasks.player_detection.scripts.migrate_store", "main",
        domain="player_detection", executable_module=True,
    ),
    _runtime_boundary(
        "player_detection", "src.tasks.player_detection.scripts.export_checkpoint"
    ),
    _runtime_boundary(
        "player_detection", "src.tasks.player_detection.scripts.evaluate"
    ),
    _runtime_boundary(
        "player_detection", "src.tasks.player_detection.scripts.preview_dataset"
    ),
    _non_hydra_boundary(
        "src.tasks.player_detection.scripts.review_dataset",
        "main",
        domain="player_detection",
        executable_module=True,
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_angle_velocity"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_loss_dominance"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.generate_dataset"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.generate_dataset_samples"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.preview_augmentation"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.train"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.visualize"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.analyze_predictions"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.evaluate"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.generate_dataset"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.make_splits"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.precompute_dino_tokens"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.predict_clip"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.train"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.clip_studio"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.export_blender"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.export_blender_rays"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.generate_dataset"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.run_pipeline"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.preflight"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualization"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualize_tasks"),
)


_RUNTIME_BOUNDARIES += tuple(
    _non_hydra_boundary(
        f"src.tasks.{task}.scripts.{script}",
        "main",
        domain=task,
        executable_module=True,
    )
    for task in ("court_detection",)
    for script in ("review_dataset", "inference_ui")
)

_RUNTIME_BOUNDARIES += tuple(
    _non_hydra_boundary(
        f"src.tasks.slcs.scripts.{script}",
        "main",
        domain="slcs",
        executable_module=True,
    )
    for script in _SLCS_REAL_RGB_ENTRYPOINTS
)

_RUNTIME_BOUNDARIES += tuple(
    _non_hydra_boundary(module, "main", domain="ball_detection", executable_module=False)
    for module in ("src.tasks.ball_detection.training.coordinates.scratch",
                   "src.tasks.ball_detection.evaluation.coordinate_cli",
                   "src.tasks.ball_detection.visualization.review.cli",
                   "src.tasks.ball_detection.visualization.inference.cli")
)
_RUNTIME_BOUNDARIES += (_non_hydra_boundary("src.tasks.ball_detection.scripts.prepare_data", "main", domain="ball_detection", executable_module=True),)



_RUNTIME_BOUNDARIES += (
    _non_hydra_boundary("src.tasks.ball_detection.training.dpt_pretrain.runner", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=False),
    _non_hydra_boundary("src.tasks.ball_detection.evaluation.hybrid_cli", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=False),
    _non_hydra_boundary("src.tasks.ball_detection.training.coordinates.finetune", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=False),
    _non_hydra_boundary("src.tasks.ball_detection.training.coordinates.full_model", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=False),
    _non_hydra_boundary("src.tasks.ball_detection.training.annotation.__main__", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=True),
    _non_hydra_boundary("src.tasks.player_detection.training.annotation.__main__", "main", domain="player_detection", kind=BoundaryKind.ARGPARSE, executable_module=True),
    _non_hydra_boundary("src.tasks.player_detection.scripts.preview_predictions", "main", domain="player_detection", kind=BoundaryKind.ARGPARSE, executable_module=True),
    _non_hydra_boundary("src.tasks.ball_detection.visualization.preview.video", "main", domain="ball_detection", kind=BoundaryKind.ARGPARSE, executable_module=False),
)



def _cli_dispatch(module: str, selector: str, entries: dict[str, str], *, executable: bool) -> RuntimeBoundary:
    return RuntimeBoundary(domain="ball_detection", module=module, callable_name="main", kind=BoundaryKind.CALLABLE,
        executable_module=executable, validator_key=f"{module}.main.routes",
        validator_callable="src.utils.configuration.cli.dispatch_cli",
        configuration_authority="literal selector map; every terminal command's validator",
        path_authority="source-verified terminal commands and their path contracts",
        validation_target="explicit selector and all reachable leaf validators",
        required_policy="selector required; terminal command owns remaining arguments",
        optional_policy="terminal command's declared policy", default_authority="terminal command's configuration",
        precedence_authority="one literal route per selector value",
        routes=tuple(sorted((selector, choice, target) for choice, target in entries.items())))


_RUNTIME_BOUNDARIES += (
    _cli_dispatch("src.tasks.ball_detection.scripts.train", "stage", {
        "convnext_heatmap": "src.tasks.ball_detection.training.convnext_heatmap.cli",
        "dpt_pretrain": "src.tasks.ball_detection.training.dpt_pretrain.runner",
        "coordinates": "src.tasks.ball_detection.training.coordinates",
        "hybrid": "src.tasks.ball_detection.training.hybrid.runner",
    }, executable=True),
    _cli_dispatch("src.tasks.ball_detection.training.coordinates.__init__", "initialize", {
        "scratch": "src.tasks.ball_detection.training.coordinates.scratch",
        "encoder": "src.tasks.ball_detection.training.coordinates.finetune",
        "full_model": "src.tasks.ball_detection.training.coordinates.full_model",
    }, executable=False),
    _cli_dispatch("src.tasks.ball_detection.scripts.evaluate", "mode", {
        "heatmap": "src.tasks.ball_detection.evaluation.heatmap_cli",
        "manifest": "src.tasks.ball_detection.evaluation.manifest_cli",
        "coordinates": "src.tasks.ball_detection.evaluation.coordinate_cli",
        "hybrid": "src.tasks.ball_detection.evaluation.hybrid_cli",
    }, executable=True),
    _cli_dispatch("src.tasks.ball_detection.scripts.review", "mode", {
        "dataset": "src.tasks.ball_detection.visualization.review.cli",
        "inference": "src.tasks.ball_detection.visualization.inference.cli",
    }, executable=True),
    _cli_dispatch("src.tasks.ball_detection.scripts.preview", "kind", {
        "spatial": "src.tasks.ball_detection.visualization.preview.spatial",
        "heatmaps": "src.tasks.ball_detection.visualization.preview.heatmaps",
        "video": "src.tasks.ball_detection.visualization.preview.video",
    }, executable=True),
)

_RUNTIME_BOUNDARIES += (
    _non_hydra_boundary(
        "src.utils.sample_dataset.__init__", "_data_resolver", domain="sample_dataset",
        kind=BoundaryKind.CALLABLE,
    ),
    _non_hydra_boundary(
        "src.utils.model_weights.__init__", "_checkpoint_resolver", domain="model_weights",
        kind=BoundaryKind.CALLABLE,
    ),
)

EXPECTED_RUNTIME_BOUNDARIES = _RUNTIME_BOUNDARIES


DEFAULT_AUDIT_INVENTORY = AuditInventory(
    boundaries=_RUNTIME_BOUNDARIES,
)
