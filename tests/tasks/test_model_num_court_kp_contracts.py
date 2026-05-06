from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel


NUM_COURT_TOKENS = 12


@pytest.mark.parametrize(
    ("model_cls", "model_cfg"),
    [
        (
            BLCSMultiViewAxialModel,
            {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 8,
                "max_num_cameras": 2,
            },
        ),
        (
            BLCSMultiViewModel,
            {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 8,
                "max_num_cameras": 2,
            },
        ),
    ],
)
def test_blcs_multiview_models_accept_data_num_court_kp(
    model_cls: type[BLCSMultiViewAxialModel] | type[BLCSMultiViewModel],
    model_cfg: dict[str, int],
) -> None:
    config = OmegaConf.create(
        {
            "data": {"num_court_kp": NUM_COURT_TOKENS, "max_seq_len": 8},
            "model": model_cfg,
        }
    )

    model = model_cls.from_config(config)

    assert model.num_court_tokens == NUM_COURT_TOKENS

    outputs = model(
        ball_uv=torch.rand(1, 2, 4, 2),
        court_kp=torch.rand(1, 2, 4, NUM_COURT_TOKENS, 2),
        ball_vis=torch.ones(1, 2, 4),
        ball_mask=torch.ones(1, 2, 4),
        court_vis=torch.ones(1, 2, 4, NUM_COURT_TOKENS),
    )

    assert outputs.keys() == {"position"}
    assert tuple(outputs["position"].shape) == (1, 4, 3)


@pytest.mark.parametrize(
    ("model_cls", "model_cfg"),
    [
        (
            PLCSMultiViewAxialModel,
            {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 8,
                "max_views": 2,
            },
        ),
        (
            PLCSMultiViewModel,
            {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 8,
                "max_views": 2,
            },
        ),
    ],
)
def test_plcs_multiview_models_accept_data_num_court_kp(
    model_cls: type[PLCSMultiViewAxialModel] | type[PLCSMultiViewModel],
    model_cfg: dict[str, int],
) -> None:
    config = OmegaConf.create(
        {
            "data": {"num_court_kp": NUM_COURT_TOKENS, "max_seq_len": 8},
            "model": model_cfg,
        }
    )

    model = model_cls.from_config(config)

    assert model.num_court_tokens == NUM_COURT_TOKENS

    outputs = model(
        human_kp=torch.rand(1, 2, 4, 17, 2),
        court_kp=torch.rand(1, 2, 4, NUM_COURT_TOKENS, 2),
        human_vis=torch.ones(1, 2, 4, 17),
        human_mask=torch.ones(1, 2, 4),
        court_vis=torch.ones(1, 2, 4, NUM_COURT_TOKENS),
    )

    assert outputs.keys() == {"position", "rotation"}
    assert tuple(outputs["position"].shape) == (1, 4, 3)
    assert tuple(outputs["rotation"].shape) == (1, 4, 2)