"""Integration checks for the destructive normalized-court artifact contract."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tasks.plcs.visualization.rendering.scene_renderer import PLCSSceneRenderer

pytestmark = pytest.mark.integration


def test_contract_documentation_has_one_authoritative_breaking_policy() -> None:
    utils_readme = Path("src/utils/README.md").read_text()
    blcs_readme = Path("src/tasks/blcs/README.md").read_text()
    plcs_readme = Path("src/tasks/plcs/README.md").read_text()

    assert "S = HALF_LENGTH = 11.885 m" in utils_readme
    assert "scale_xyz = (S, S, S)" in utils_readme
    assert "再生成" in utils_readme and "再学習" in utils_readme
    assert "自動推測・自動変換は行わない" in utils_readme
    assert "src/utils/README.md" in blcs_readme
    assert "src/utils/README.md" in plcs_readme
    assert re.search(r"\bv[12]\b", utils_readme) is None


def test_plcs_render_boundary_scales_only_court_translation() -> None:
    position_norm = np.asarray([[1.0, -0.5, 0.25]], dtype=np.float32)
    canonical_pose_m = np.asarray(
        [[0.2, -0.3, 0.4], [-0.6, 0.7, 1.1]],
        dtype=np.float32,
    )
    scene = SimpleNamespace(
        position=position_norm,
        rotation=np.asarray([[1.0, 0.0]], dtype=np.float32),
        canonical_pose_3d=canonical_pose_m[None],
    )
    renderer = object.__new__(PLCSSceneRenderer)
    translation_m = position_norm[0].astype(np.float64) * 11.885

    np.testing.assert_allclose(
        renderer._world_positions(scene)[0],
        translation_m,
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        renderer._compute_world_pose(scene, 0),
        canonical_pose_m + translation_m,
        atol=1e-6,
        rtol=0.0,
    )
