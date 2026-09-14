"""Boundary tests for the pipeline's player-motion selection module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.motion_alignment.similarity import SimilarityConfig
from src.tennis_scene.pipeline.components.motion_alignment import (
    MotionAlignmentModule,
    PlayerMotionConfig,
    load_joint_regressor,
)
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationApplied,
)

_NUM_PLAYERS = 1
_NUM_CAMERAS = 1
_NUM_FRAMES = 3
_NUM_VERTICES = 5


def _config(
    path: Path,
    *,
    source: str = "plcs",
    scale_mode: str = "fixed",
) -> PlayerMotionConfig:
    return PlayerMotionConfig(
        source=source,  # type: ignore[arg-type]
        scale_mode=scale_mode,  # type: ignore[arg-type]
        smpl_joint_regressor=path,
        similarity=SimilarityConfig(),
    )


def _regressor_file(tmp_path: Path, *, rows: int = 24) -> Path:
    path = tmp_path / "smpl_neutral_J_regressor.pt"
    tensor = torch.zeros((rows, _NUM_VERTICES), dtype=torch.float32)
    tensor[0, 0] = 1.0
    torch.save(tensor, path)
    return path


def _associated(*, with_world_motion: bool) -> PlayerAssociationApplied:
    transl_incam: np.ndarray | None = None
    transl_world: np.ndarray | None = None
    orient_world: np.ndarray | None = None
    if with_world_motion:
        shape = (_NUM_PLAYERS, _NUM_FRAMES, 3)
        transl_incam = np.zeros(shape, dtype=np.float32)
        transl_world = np.zeros(shape, dtype=np.float32)
        orient_world = np.zeros(shape, dtype=np.float32)
    return PlayerAssociationApplied(
        human_kp_2d=np.zeros(
            (_NUM_PLAYERS, _NUM_CAMERAS, _NUM_FRAMES, 17, 2), dtype=np.float32
        ),
        human_kp_vis=np.ones(
            (_NUM_PLAYERS, _NUM_CAMERAS, _NUM_FRAMES, 17), dtype=np.float32
        ),
        smpl_body_pose=np.zeros((_NUM_PLAYERS, _NUM_FRAMES, 63), dtype=np.float32),
        smpl_global_orient=np.zeros(
            (_NUM_PLAYERS, _NUM_FRAMES, 3), dtype=np.float32
        ),
        smpl_betas=np.zeros((_NUM_PLAYERS, 10), dtype=np.float32),
        smpl_vertices_local=np.zeros(
            (_NUM_PLAYERS, _NUM_FRAMES, _NUM_VERTICES, 3), dtype=np.float32
        ),
        track_ids=np.arange(_NUM_PLAYERS, dtype=np.int32),
        track_ids_by_camera=[np.arange(_NUM_PLAYERS, dtype=np.int32)],
        smpl_transl_incam=transl_incam,
        smpl_transl_world=transl_world,
        smpl_global_orient_world=orient_world,
    )


def test_plcs_source_returns_the_inputs_unchanged(tmp_path: Path) -> None:
    module = MotionAlignmentModule(
        _config(tmp_path / "unused_regressor.pt", source="plcs")
    )
    associated = _associated(with_world_motion=False)
    position: NDArray[np.float32] = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    yaw = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    applied = module.process(
        associated=associated,
        plcs_position=position,
        plcs_yaw=yaw,
        court_visibility=np.ones((_NUM_CAMERAS, _NUM_FRAMES, 20), dtype=np.float32),
        reference_camera_index=0,
    )

    assert applied.player_position is position
    assert applied.player_yaw is yaw
    assert applied.smpl_global_orient is associated.smpl_global_orient
    assert applied.smpl_vertices_local is associated.smpl_vertices_local
    np.testing.assert_array_equal(applied.player_position, position)
    np.testing.assert_array_equal(applied.player_yaw, yaw)
    assert applied.metadata == {"player_motion": {"source": "plcs"}}


def test_plcs_source_never_reads_the_regressor(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pt"
    assert not missing.exists()

    # Constructing a source='plcs' module must not require the aligned asset.
    MotionAlignmentModule(_config(missing, source="plcs"))


def test_alignment_source_raises_when_world_fields_are_absent(tmp_path: Path) -> None:
    module = MotionAlignmentModule(
        _config(_regressor_file(tmp_path), source="gvhmr_alignment")
    )

    with pytest.raises(ValueError) as error:
        module.process(
            associated=_associated(with_world_motion=False),
            plcs_position=np.zeros((_NUM_PLAYERS, _NUM_FRAMES, 3), dtype=np.float32),
            plcs_yaw=np.zeros((_NUM_PLAYERS, _NUM_FRAMES), dtype=np.float32),
            court_visibility=np.ones(
                (_NUM_CAMERAS, _NUM_FRAMES, 20), dtype=np.float32
            ),
            reference_camera_index=0,
        )

    message = str(error.value)
    assert "smpl_transl_incam" in message
    assert "smpl_transl_world" in message
    assert "smpl_global_orient_world" in message
    assert "before world-motion support" in message


def test_joint_regressor_accepts_a_full_matrix(tmp_path: Path) -> None:
    path = _regressor_file(tmp_path)

    loaded = load_joint_regressor(path)

    assert loaded.shape == (24, _NUM_VERTICES)
    assert loaded.dtype == np.float64


def test_joint_regressor_accepts_a_single_joint_row(tmp_path: Path) -> None:
    path = tmp_path / "root_row.pt"
    torch.save(torch.ones(_NUM_VERTICES, dtype=torch.float32), path)

    loaded = load_joint_regressor(path)

    assert loaded.shape == (_NUM_VERTICES,)


def test_joint_regressor_rejects_a_missing_asset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="joint regressor"):
        load_joint_regressor(tmp_path / "missing.pt")


def test_joint_regressor_rejects_a_non_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "not_a_tensor.pt"
    torch.save({"joints": torch.zeros(1)}, path)

    with pytest.raises(TypeError, match="must contain a tensor"):
        load_joint_regressor(path)


@pytest.mark.parametrize(
    "shape",
    [(3, 4, 5), (0, _NUM_VERTICES), (2, 0)],
)
def test_joint_regressor_rejects_invalid_shapes(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(torch.zeros(shape, dtype=torch.float32), path)

    with pytest.raises(ValueError, match=r"shape \(24, V\) or \(V,\)"):
        load_joint_regressor(path)


@pytest.mark.parametrize(
    ("source", "scale_mode"),
    [("world", "fixed"), ("plcs", "big"), ("gvhmr", "free")],
)
def test_player_motion_config_rejects_unknown_choices(
    tmp_path: Path, source: str, scale_mode: str
) -> None:
    with pytest.raises(ValueError):
        _config(tmp_path / "regressor.pt", source=source, scale_mode=scale_mode)


def test_fit_config_applies_the_selected_scale_mode(tmp_path: Path) -> None:
    fixed = _config(tmp_path / "regressor.pt", scale_mode="fixed")
    free = _config(tmp_path / "regressor.pt", scale_mode="free")

    assert fixed.fit_config().fixed_scale == 1.0
    assert free.fit_config().fixed_scale is None
