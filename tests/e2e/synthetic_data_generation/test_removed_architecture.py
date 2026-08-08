"""Repository-wide absence checks for the prohibited synthetic architecture."""

from __future__ import annotations

import ast
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

REMOVED_FILES = (
    "src/synthetic_data_generation/dataset/pipeline.py",
    "src/synthetic_data_generation/dataset/execution.py",
    "src/synthetic_data_generation/dataset/algorithms.py",
    "src/synthetic_data_generation/scripts/dataset/run_pipeline.py",
    "src/synthetic_data_generation/scripts/dataset/fit_blcs_features.py",
    "src/synthetic_data_generation/scripts/validate_configuration.py",
    "src/synthetic_data_generation/alignment/stage_result.py",
    "src/synthetic_data_generation/alignment/scene_provider/bundle.py",
    "src/synthetic_data_generation/alignment/scene_provider/export.py",
    "src/synthetic_data_generation/alignment/scene_provider/geometry_bridge.py",
    "src/synthetic_data_generation/rendering/nht/runtime_probe.py",
    "src/synthetic_data_generation/rendering/nht/composition_smoke.py",
    "src/tasks/base/data/chunk_manager.py",
    "src/tasks/base/data/chunked_datamodule.py",
    "src/tasks/base/data/datamodule.py",
    "src/tasks/base/data/dataset_writer.py",
    "src/tasks/base/data/scene_dataset.py",
    "src/tasks/base/training/chunk_rotation_callback.py",
    "src/utils/data/scene_io.py",
    "src/tasks/blcs/data/chunk_manager.py",
    "src/tasks/blcs/data/chunked_datamodule.py",
    "src/tasks/blcs/generate_dataset/io/dataset_io.py",
    "src/tasks/blcs/generate_dataset/utils/parallel_runner.py",
    "src/tasks/blcs/scripts/generate_dataset.py",
    "src/tasks/blcs/scripts/preview_augmentation.py",
    "src/tasks/blcs/scripts/visualize.py",
    "src/tasks/blcs/visualization/orchestrator.py",
    "src/tasks/blcs/visualization/io/scene.py",
    "src/tasks/blcs/visualization/api/predict.py",
    "src/tasks/plcs/data/chunk_manager.py",
    "src/tasks/plcs/data/chunked_datamodule.py",
    "src/tasks/plcs/generate_dataset/config.py",
    "src/tasks/plcs/generate_dataset/io/dataset_io.py",
    "src/tasks/plcs/generate_dataset/io/scene_loader.py",
    "src/tasks/plcs/generate_dataset/scene_generator.py",
    "src/tasks/plcs/generate_dataset/multi_object_scene_generator.py",
    "src/tasks/plcs/generate_dataset/utils/parallel_runner.py",
    "src/tasks/plcs/scripts/generate_dataset.py",
    "src/tasks/plcs/scripts/preview_augmentation.py",
    "src/tasks/plcs/scripts/analysis/analyze_angle_velocity.py",
    "src/tasks/plcs/scripts/analysis/analyze_dataset_distribution.py",
    "src/tasks/plcs/scripts/analysis/analyze_loss_dominance.py",
    "src/tasks/plcs/scripts/visualize.py",
    "src/tasks/plcs/scripts/analysis/visualize_rotation_error_samples.py",
    "src/tasks/plcs/visualization/orchestrator.py",
    "src/tasks/plcs/visualization/io/scene.py",
    "src/tasks/plcs/visualization/api/predict.py",
)

REMOVED_MODULE_PREFIXES = (
    "src.synthetic_data_generation.alignment.artifacts",
    "src.synthetic_data_generation.alignment.scene_provider",
    "src.synthetic_data_generation.dataset.blcs.artifacts",
    "src.synthetic_data_generation.dataset.plcs.artifacts",
    "src.tasks.base.data.chunked_datamodule",
    "src.tasks.base.data.datamodule",
    "src.tasks.base.data.dataset_writer",
    "src.tasks.base.data.scene_dataset",
    "src.utils.data.scene_io",
    "src.tasks.blcs.data.chunked_datamodule",
    "src.tasks.plcs.data.chunked_datamodule",
    "src.tasks.blcs.visualization.orchestrator",
    "src.tasks.plcs.visualization.orchestrator",
)


def _active_python_files() -> tuple[Path, ...]:
    roots = (
        PROJECT_ROOT / "src/synthetic_data_generation",
        PROJECT_ROOT / "src/tasks/base/generate_dataset",
        PROJECT_ROOT / "src/tasks/blcs",
        PROJECT_ROOT / "src/tasks/plcs",
    )
    return tuple(
        path
        for root in roots
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def test_old_files_and_production_entrypoints_are_deleted() -> None:
    present = [
        relative for relative in REMOVED_FILES if (PROJECT_ROOT / relative).exists()
    ]

    assert not present, f"prohibited legacy files still exist: {present}"
    scripts = {
        path.name
        for path in (PROJECT_ROOT / "src/synthetic_data_generation/scripts").glob(
            "*.py"
        )
    }
    assert scripts == {"__init__.py", "run_scene_pipeline.py"}


def test_no_active_import_targets_a_removed_module() -> None:
    violations: list[str] = []
    for path in _active_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in REMOVED_MODULE_PREFIXES
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {module}"
                    )

    assert not violations, "removed module imports remain:\n" + "\n".join(violations)


def test_no_identity_or_fixed_pose_architecture_remains_in_active_generation() -> None:
    forbidden = {
        "artifactref",
        "fingerprint",
        "sha256",
        "sha-256",
        "content-addressed",
        "content_addressed",
        "pose_ids",
        "pose_indices",
        "selected_camera",
        "immutable publication",
        "overwrite refusal",
        "overwrite rejection",
    }
    violations: list[str] = []
    for path in _active_python_files():
        relative = path.relative_to(PROJECT_ROOT)
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{relative}: {token}")

    assert not violations, "prohibited active architecture remains:\n" + "\n".join(
        violations
    )
