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
)

REMOVED_MODULE_PREFIXES = (
    "src.synthetic_data_generation.alignment.artifacts",
    "src.synthetic_data_generation.alignment.scene_provider",
    "src.synthetic_data_generation.dataset.blcs.artifacts",
    "src.synthetic_data_generation.dataset.plcs.artifacts",
)

APPROVED_TASK_BOUNDARIES = frozenset(
    {
        "src.tasks.base.model_io",
        "src.tasks.blcs.generate_dataset.source_api",
        "src.tasks.plcs.data.targets",
        "src.tasks.plcs.generate_dataset.sampling.motion_source",
    }
)

FORBIDDEN_ACTIVE_ARCHITECTURE_TOKENS = frozenset(
    {
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
)

# The court-line cache hashes model inputs and images only to reuse deterministic raw
# probabilities. It is deliberately separate from the removed artifact publication
# and scene-identity architecture guarded by this test.
ALLOWED_ACTIVE_ARCHITECTURE_TOKENS = {
    Path("src/synthetic_data_generation/alignment/line_inference_cache.py"): frozenset(
        {"fingerprint", "sha256"}
    ),
}


def _active_python_files() -> tuple[Path, ...]:
    roots = (
        PROJECT_ROOT / "src/synthetic_data_generation",
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
    assert scripts == {
        "__init__.py",
        "run_scene_pipeline.py",
        "visualize_dataset.py",
    }


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


def test_canonical_production_imports_only_public_task_boundaries() -> None:
    violations: list[str] = []
    production_root = PROJECT_ROOT / "src/synthetic_data_generation"
    for path in sorted(production_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                if not module.startswith("src.tasks."):
                    continue
                if module in APPROVED_TASK_BOUNDARIES:
                    continue
                if module.startswith("src.tasks.court_detection."):
                    continue
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {module}"
                )

    assert not violations, "task-internal imports remain:\n" + "\n".join(violations)


def test_no_identity_or_fixed_pose_architecture_remains_in_active_generation() -> None:
    violations: list[str] = []
    for path in _active_python_files():
        relative = path.relative_to(PROJECT_ROOT)
        text = path.read_text(encoding="utf-8").lower()
        allowed = ALLOWED_ACTIVE_ARCHITECTURE_TOKENS.get(relative, frozenset())
        for token in FORBIDDEN_ACTIVE_ARCHITECTURE_TOKENS - allowed:
            if token in text:
                violations.append(f"{relative}: {token}")

    assert not violations, "prohibited active architecture remains:\n" + "\n".join(
        violations
    )


def test_active_architecture_token_exceptions_are_current() -> None:
    stale: list[str] = []
    for relative, allowed in ALLOWED_ACTIVE_ARCHITECTURE_TOKENS.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            stale.append(f"{relative}: file is absent")
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in allowed:
            if token not in FORBIDDEN_ACTIVE_ARCHITECTURE_TOKENS:
                stale.append(f"{relative}: {token} is not otherwise forbidden")
            elif token not in text:
                stale.append(f"{relative}: {token} is no longer present")

    assert not stale, "stale active-architecture token exceptions:\n" + "\n".join(stale)
