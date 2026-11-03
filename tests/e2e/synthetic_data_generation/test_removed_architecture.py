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

# These modules hash detector inputs, manual-alignment evidence, or review cache revisions.
# The digests validate/reuse the exact measured source; they do not restore the
# removed artifact-reference, scene-identity, or content-addressed publication model.
ALLOWED_ACTIVE_ARCHITECTURE_TOKENS = {
    # Read-only saved-train validation compares historical recipe/index hash
    # fields; it cannot import or generate a new source publication.
    Path("src/synthetic_data_generation/dataset/court/scale/saved_train.py"): frozenset(
        {"sha256"}
    ),
    # Appearance variants use explicit scene IDs/paths. These hashes verify
    # fixed inputs, paid API responses and imported SfM, not scene identity.
    Path("src/synthetic_data_generation/appearance/batch.py"): frozenset({"sha256"}),
    Path("src/synthetic_data_generation/appearance/comparison.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/contracts.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/derived.py"): frozenset({"sha256"}),
    Path("src/synthetic_data_generation/appearance/generation.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/nht.py"): frozenset({"sha256"}),
    Path("src/synthetic_data_generation/appearance/openai_api.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/reporting.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/validation.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/appearance/workspace.py"): frozenset(
        {"sha256"}
    ),
    # Migration resume binds only this attempt to its unchanged source manifest;
    # this does not introduce scene identity or content-addressed publication.
    Path("src/synthetic_data_generation/dataset/court/storage.py"): frozenset(
        {"fingerprint", "sha256"}
    ),
    # One-time JPEG migration reports the source manifest's digest. Source
    # changes are checked by byte equality; the digest never selects an owner.
    Path("src/synthetic_data_generation/dataset/court/migration.py"): frozenset(
        {"sha256"}
    ),
    # Portable assets verify copied bytes and the explicitly pinned input bundle;
    # their digests never derive scene IDs or choose scene-workspace owners.
    Path("src/synthetic_data_generation/dataset/court/assets.py"): frozenset(
        {"sha256", "sha-256"}
    ),
    # The asset CLI forwards the caller's expected transfer-verification digest;
    # the source scenes and destination remain explicit paths.
    Path("src/synthetic_data_generation/scripts/court_scene_assets.py"): frozenset(
        {"sha256"}
    ),
    # Stable scale-point IDs and field-local RNG streams pin experimental views;
    # the explicit source scene ID and canonical workspace remain unchanged.
    Path("src/synthetic_data_generation/dataset/court/scale/core.py"): frozenset(
        {"sha256"}
    ),
    # Candidate plans bind source geometry and the experiment's recipe to stable
    # views; these keys never derive source scene IDs or choose workspace owners.
    Path("src/synthetic_data_generation/dataset/court/scale/planning.py"): frozenset(
        {"sha256"}
    ),
    # Hash-addressed requests/stores live under an explicitly selected experiment
    # output root, outside the scene workspace's fixed datasets/court owner.
    Path("src/synthetic_data_generation/dataset/court/scale/publication.py"): frozenset(
        {"sha256", "content-addressed"}
    ),
    # Durable transport pins copied cache bytes inside an explicit experiment
    # output root; it does not select or change canonical scene-workspace owners.
    Path("src/synthetic_data_generation/dataset/court/scale/durable_storage.py"): frozenset(
        {"sha256", "sha-256"}
    ),
    Path("src/synthetic_data_generation/dataset/court/scale/durability.py"): frozenset(
        {"sha256"}
    ),
    # Each QA record binds the experiment's exact quality-policy version.
    Path("src/synthetic_data_generation/dataset/court/scale/quality.py"): frozenset(
        {"sha256"}
    ),
    # Production recipe digests verify the explicitly selected experiment policy.
    Path("src/synthetic_data_generation/dataset/court/scale/recipes.py"): frozenset(
        {"sha256"}
    ),
    # Asset and renderer hashes verify frozen experimental inputs/runtime;
    # source scenes are read only and retain their explicit IDs and owners.
    Path("src/synthetic_data_generation/dataset/court/scale/runtime.py"): frozenset(
        {"sha256", "sha-256", "fingerprint"}
    ),
    # The serialized RNG algorithm version fixes experimental sampling streams.
    Path("src/synthetic_data_generation/dataset/court/scale/strategies.py"): frozenset(
        {"sha256"}
    ),
    # Strict plan validation checks the experiment's frozen geometry recipe.
    Path("src/synthetic_data_generation/dataset/court/scale/validation.py"): frozenset(
        {"sha256"}
    ),
    # Review exposes the selected experiment's request/recipe digests; its
    # catalog keys and cache revisions never choose canonical scene owners.
    Path("src/synthetic_data_generation/dataset/court/review/scale_service.py"): frozenset(
        {"sha256"}
    ),
    # The CLI pins asset bytes and renderer conditions for an explicit data
    # version outside canonical scene publication.
    Path("src/synthetic_data_generation/scripts/generate_court_scale.py"): frozenset(
        {"sha256", "fingerprint"}
    ),
    Path("src/synthetic_data_generation/alignment/heatmaps.py"): frozenset({"sha256"}),
    Path("src/synthetic_data_generation/dataset/court/review/service.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/alignment/line_inference_cache.py"): frozenset(
        {"fingerprint", "sha256"}
    ),
    Path("src/synthetic_data_generation/alignment/line_inputs.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/alignment/manual/artifacts.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/alignment/manual/service.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/alignment/manual/source.py"): frozenset(
        {"sha256"}
    ),
    Path("src/synthetic_data_generation/dataset/court/review/service.py"): frozenset(
        {"sha256"}
    ),
}


def _active_python_files() -> tuple[Path, ...]:
    roots = (PROJECT_ROOT / "src/synthetic_data_generation",)
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
        "compact_court_storage.py",
        "court_scene_assets.py",
        "edit_alignment.py",
        "generate_court_scale.py",
        "generate_publication_visualizations.py",
        "review_court_dataset.py",
        "run_appearance_variant.py",
        "run_scene_pipeline.py",
        "visualize_dataset.py",
        "visualize_scene_reveal.py",
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
