"""Keep real-RGB operational entrypoints in their owning src packages."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINTS = (
    "src.tasks.slcs.scripts.evaluate_run",
    "src.tasks.blcs.scripts.evaluate_real",
    "src.tennis_scene.scripts.import_broadcast_ball",
    "src.tennis_scene.scripts.prepare_blcs_real_dataset",
    "src.tennis_scene.scripts.build_real_rgb",
    "src.tennis_scene.scripts.build_slcs_dataset",
    "src.tennis_scene.scripts.assemble_slcs_dataset",
    "src.tennis_scene.scripts.report_slcs_dataset_quality",
    "src.tennis_scene.scripts.render_reconstruction_review",
)
REMOVED_ENTRYPOINTS = (
    "scripts/analysis/benchmark_vitpose_precision.py",
    "scripts/analysis/calibrate_slcs_ball_velocity.py",
    "scripts/analysis/compare_slcs_ball_anchors.py",
    "scripts/analysis/compare_slcs_ball_transitions.py",
    "scripts/analysis/compare_slcs_conditions.py",
    "scripts/analysis/evaluate_blcs_real.py",
    "scripts/analysis/evaluate_refinement.py",
    "scripts/analysis/evaluate_slcs_run.py",
    "scripts/analysis/import_broadcast_ball.py",
    "scripts/analysis/meiji_court_probe.py",
    "scripts/analysis/meiji_court_probe.yaml",
    "scripts/analysis/prepare_blcs_real_dataset.py",
    "scripts/analysis/prepare_plcs_motion_split.py",
    "scripts/analysis/prepare_plcs_subset.py",
    "scripts/analysis/render_reconstruction_review.py",
    "scripts/analysis/report_slcs_validation.py",
    "scripts/datasets/build_real_rgb.sh",
    "scripts/visualization/README.md",
    "scripts/visualization/slcs_pr_clip.py",
)


@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_real_rgb_help_is_cpu_only_and_executable(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), module


def test_every_real_rgb_entrypoint_has_a_validated_runtime_boundary() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.callable_name == "main"
    }
    for module in ENTRYPOINTS:
        assert module in boundaries, module
        boundary = boundaries[module]
        assert boundary.executable_module, module
        assert boundary.validator_key and boundary.validator_callable, module
        assert boundary.configuration_authority and boundary.path_authority, module


def test_removed_real_rgb_entrypoints_have_no_root_shims() -> None:
    assert not [name for name in REMOVED_ENTRYPOINTS if (REPO_ROOT / name).exists()]
    # Existing, unrelated operational tooling is outside this migration.
    for name in (
        "scripts/audit_configuration.py",
        "scripts/run_in_repo_venv.sh",
        "scripts/analysis/models/attention_maps.py",
    ):
        assert (REPO_ROOT / name).is_file(), name


def test_production_code_has_no_reverse_imports_from_root_scripts() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            if any(
                name == "scripts" or name.startswith("scripts.") for name in modules
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, offenders
