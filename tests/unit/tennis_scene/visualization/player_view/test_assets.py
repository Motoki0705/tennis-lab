"""Reject unusable body skeletons before publishing a player-view bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.visualization.player_view.assets import load_body_kinematics


@pytest.mark.parametrize(
    "invalid",
    [
        "dimensions",
        "parents",
        "root-parent",
        "wrong-skeleton",
        "joint-order",
        "nonfinite",
        "regressor-sum",
    ],
)
def test_rejects_invalid_kinematic_assets(tmp_path: Path, invalid: str) -> None:
    template: np.ndarray = np.ones((3, 3), np.float32)
    shapedirs: np.ndarray = np.ones((3, 3, 10), np.float32)
    regressor = np.ones((22, 3), np.float32) / 3
    tree = np.array(
        [
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
            range(22),
        ],
        np.int32,
    )
    if invalid == "dimensions":
        shapedirs = shapedirs[..., :9]
    elif invalid == "parents":
        tree[0, 10] = 10
    elif invalid == "root-parent":
        tree[0, 0] = 12
    elif invalid == "wrong-skeleton":
        tree[0, 15] = 9
    elif invalid == "joint-order":
        tree[1, 5] = 4
    elif invalid == "nonfinite":
        template[0, 0] = np.nan
    elif invalid == "regressor-sum":
        regressor[0] = 0
    path = tmp_path / "model.npz"
    np.savez(
        path,
        v_template=template,
        shapedirs=shapedirs,
        J_regressor=regressor,
        kintree_table=tree,
    )
    with pytest.raises(ValueError):
        load_body_kinematics(path)
