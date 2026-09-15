"""Source-adapter tests for ACCAD and GVHMR motion producers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.submodules.models import SmplCoco17Reconstructor
from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip
from src.tasks.plcs.motion.sources import AccadCoco17Adapter, GvhmrCoco17Adapter
from src.tasks.plcs.motion.sources.gvhmr import GVHMR_Y_UP_TO_PLCS_Z_UP
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix


def _global_parameters(frames: int) -> dict[str, torch.Tensor]:
    return {
        "body_pose": torch.zeros((frames, 63), dtype=torch.float32),
        "betas": torch.zeros((frames, 10), dtype=torch.float32),
        "global_orient": torch.zeros((frames, 3), dtype=torch.float32),
        "transl": torch.tensor(
            [[1.0 + frame, 2.0, 3.0] for frame in range(frames)],
            dtype=torch.float32,
        ),
    }


def test_gvhmr_adapter_changes_basis_but_preserves_global_trajectory(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    frames = 2
    parameters = _global_parameters(frames)
    joints_y_up = parameters["transl"][:, None].repeat(1, 17, 1)
    reconstructor = object.__new__(SmplCoco17Reconstructor)
    monkeypatch.setattr(reconstructor, "reconstruct", lambda _params: joints_y_up)
    adapter = GvhmrCoco17Adapter(reconstructor)

    clip = adapter.convert(
        parameters,
        source_id="clip:cam1:near",
        source_path=tmp_path / "cam1.mp4",
        fps=60.0,
        joint_confidence=np.ones((frames, 17), dtype=np.float32),
        frame_valid=np.ones(frames, dtype=np.bool_),
        provenance={"track_id": 7},
    )

    expected_translation = np.asarray(
        [[1.0, -3.0, 2.0], [2.0, -3.0, 2.0]], dtype=np.float32
    )
    np.testing.assert_allclose(clip.root_translation_m, expected_translation)
    np.testing.assert_allclose(clip.joints_3d_m[:, 0], expected_translation, atol=1e-6)
    np.testing.assert_allclose(
        clip.root_rotation,
        np.repeat(GVHMR_Y_UP_TO_PLCS_Z_UP[None], frames, axis=0),
    )


def test_gvhmr_adapter_left_multiplies_active_root_rotation(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    parameters = _global_parameters(1)
    parameters["global_orient"][0, 1] = torch.pi / 2
    joints_y_up = parameters["transl"][:, None].repeat(1, 17, 1)
    reconstructor = object.__new__(SmplCoco17Reconstructor)
    monkeypatch.setattr(reconstructor, "reconstruct", lambda _params: joints_y_up)

    clip = GvhmrCoco17Adapter(reconstructor).convert(
        parameters,
        source_id="clip:cam1:near",
        source_path=tmp_path / "cam1.mp4",
        fps=60.0,
        joint_confidence=np.ones((1, 17), dtype=np.float32),
        frame_valid=np.ones(1, dtype=np.bool_),
        provenance={"track_id": 7},
    )

    rotation_y_up = np.asarray(
        axis_angle_to_matrix(parameters["global_orient"]).numpy(),
        dtype=np.float32,
    )
    expected = np.einsum("ij,tjk->tik", GVHMR_Y_UP_TO_PLCS_Z_UP, rotation_y_up)
    np.testing.assert_allclose(clip.root_rotation, expected, atol=1e-6)


def test_accad_adapter_uses_the_exact_coco17_vertex_regressor(tmp_path: Path) -> None:
    frames = 2
    source = PLCSMotionClip.from_amass_arrays(
        source_path=tmp_path / "walk_poses.npz",
        category="walking",
        gender="neutral",
        fps=120.0,
        poses=np.zeros((frames, 156), dtype=np.float32),
        trans=np.zeros((frames, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
    )

    class FakeModel:
        num_betas = 16

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            count = int(torch.as_tensor(kwargs["transl"]).shape[0])
            vertices = torch.zeros((count, 6890, 3), dtype=torch.float32)
            vertices[:, 0] = torch.tensor([1.0, 2.0, 3.0])
            return SimpleNamespace(vertices=vertices)

    adapter = AccadCoco17Adapter(
        smplh_model_path=tmp_path / "SMPLH_NEUTRAL.pkl",
        coco17_regressor_path=tmp_path / "regressor.pt",
        device="cpu",
    )
    adapter._models["neutral"] = FakeModel()
    regressor = torch.zeros((17, 6890), dtype=torch.float32)
    regressor[:, 0] = 1.0
    adapter._regressor = regressor

    clip = adapter.convert(source)

    np.testing.assert_array_equal(
        clip.joints_3d_m,
        np.tile(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), (frames, 17, 1)),
    )
    assert clip.fps == 120.0
    assert clip.frame_count == frames
