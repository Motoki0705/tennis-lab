from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel


def _build_inputs(
    *,
    batch_size: int = 2,
    n_cams: int = 3,
    seq_len: int = 8,
    num_court_tokens: int = 4,
) -> dict[str, torch.Tensor]:
    ball_uv = torch.randn(batch_size, n_cams, seq_len, 2)
    court_kp = torch.randn(batch_size, n_cams, seq_len, num_court_tokens, 2)
    ball_vis = torch.ones(batch_size, n_cams, seq_len, dtype=torch.bool)
    ball_mask = torch.ones(batch_size, n_cams, seq_len, dtype=torch.bool)
    ball_mask[:, :, -1] = False
    ball_vis[:, :, -1] = False
    return {
        "ball_uv": ball_uv,
        "court_kp": court_kp,
        "ball_vis": ball_vis,
        "ball_mask": ball_mask,
    }


@pytest.mark.parametrize(
    ("time_global_every", "num_layers"),
    [
        (0, 2),
        (2, 5),
    ],
)
def test_forward_supports_sliding_and_periodic_global_time_attention(
    time_global_every: int,
    num_layers: int,
) -> None:
    model = BLCSMultiViewAxialModel(
        hidden_dim=32,
        num_heads=4,
        ffn_dim=64,
        num_layers=num_layers,
        predict_velocity=True,
        max_seq_len=16,
        max_num_cameras=4,
        num_court_tokens=4,
        time_window_radius=2,
        time_global_every=time_global_every,
    ).eval()

    with torch.no_grad():
        out = model(**_build_inputs())

    assert out["position"].shape == (2, 8, 3)
    assert out["velocity"].shape == (2, 8, 3)


def test_sliding_attention_mask_respects_window_radius() -> None:
    valid = torch.ones(1, 5, dtype=torch.bool)

    mask = BLCSMultiViewAxialModel._build_sliding_attn_mask(valid, radius=1)

    expected = torch.tensor(
        [
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


@pytest.mark.parametrize(
    ("time_global_every", "expected"),
    [
        (0, [False, False, False, False, False, False]),
        (2, [False, False, True, False, False, True]),
    ],
)
def test_periodic_global_schedule_matches_config(
    time_global_every: int,
    expected: list[bool],
) -> None:
    model = BLCSMultiViewAxialModel(
        hidden_dim=32,
        num_heads=4,
        num_layers=6,
        max_seq_len=16,
        max_num_cameras=4,
        num_court_tokens=4,
        time_window_radius=2,
        time_global_every=time_global_every,
    )

    actual = [model._is_global_time_layer(layer_index) for layer_index in range(6)]

    assert actual == expected