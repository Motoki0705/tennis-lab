"""Tests for explicit renderer-ready PLCS avatar appearance."""

import pytest
import torch

from src.synthetic_data_generation.dataset.plcs.composition import AvatarAppearance


def test_avatar_appearance_requires_explicit_linear_rgb() -> None:
    appearance = AvatarAppearance(
        features=torch.tensor(((0.1, 0.2, 0.3),), dtype=torch.float32),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )

    assert appearance.features.shape == (1, 3)
    with pytest.raises(ValueError, match=r"shape \[N,3\]"):
        AvatarAppearance(
            features=torch.zeros((1, 4), dtype=torch.float32),
            appearance_model="rgb",
            appearance_space="linear_rgb",
        )
    with pytest.raises(ValueError, match="appearance_model"):
        AvatarAppearance(
            features=torch.zeros((1, 3), dtype=torch.float32),
            appearance_model="spherical-harmonics",
            appearance_space="linear_rgb",
        )
    with pytest.raises(ValueError, match="unit range"):
        AvatarAppearance(
            features=torch.tensor(((1.1, 0.0, 0.0),), dtype=torch.float32),
            appearance_model="rgb",
            appearance_space="linear_rgb",
        )
