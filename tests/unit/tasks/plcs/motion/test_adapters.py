"""Source-adapter tests for ACCAD and GVHMR motion producers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip
from src.tasks.plcs.motion.sources import AccadCoco17Adapter


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
