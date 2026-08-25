"""Integration checks for the destructive normalized-court artifact contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.data.tracking_dataset import BLCSTrackingDataset
from src.tasks.plcs.visualization.rendering.scene_renderer import PLCSSceneRenderer
from src.utils.schema.court_normalization import (
    CourtCoordinateContractError,
    court_coordinate_normalization_metadata,
)

pytestmark = pytest.mark.integration


def test_mixed_scene_contract_is_rejected_before_sample_loading(
    tmp_path: Path,
) -> None:
    scenes = tmp_path / "scenes"
    valid = scenes / "valid"
    mismatched = scenes / "mismatched"
    valid.mkdir(parents=True)
    mismatched.mkdir()
    current = court_coordinate_normalization_metadata()
    old = court_coordinate_normalization_metadata()
    old["scale_xyz_m"] = [5.485, 11.885, 1.07]
    for path, contract in ((valid, current), (mismatched, old)):
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "num_frames": 8,
                    "court_coordinate_normalization": contract,
                }
            )
        )
        (path / "scalars.json").write_text(json.dumps({"num_cameras": 2}))
    (tmp_path / "train.txt").write_text("valid\nmismatched\n")

    config_dir = Path("src/tasks/blcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train_tracking")
    config.data.seq_len_range = [8, 8]
    config.data.num_views_range = [2, 2]

    with pytest.raises(CourtCoordinateContractError, match="mismatched"):
        BLCSTrackingDataset(
            scene_dir=tmp_path,
            split_file="train.txt",
            config=config,
            augment=False,
        )


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
