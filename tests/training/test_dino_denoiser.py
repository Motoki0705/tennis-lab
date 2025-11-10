from __future__ import annotations

import torch

from src.datasets.dancetrack import TargetFrame
from src.training.scene_model.dino_denoiser import DinoDenoiser


def test_denoiser_generates_expected_shape() -> None:
    target = TargetFrame(
        center=torch.tensor([[5.0, 6.0]]),
        size=torch.tensor([[2.0, 3.0]]),
        track_ids=torch.tensor([7], dtype=torch.long),
        confidence=torch.tensor([1.0]),
    )
    denoiser = DinoDenoiser({"num_noisy_queries": 2, "box_noise_scale": 0.1, "label_noise_scale": 0.5})
    state = denoiser.make_noise([[target]])
    assert state.boxes.shape == (1, 2, 4)
    assert state.labels.shape == (1, 2)
    assert state.pad_size == 2
