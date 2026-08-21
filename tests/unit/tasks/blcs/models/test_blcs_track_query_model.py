from __future__ import annotations

import inspect

import pytest
import torch

from src.tasks.blcs.configuration import TrackQueryModelConfig, parse_model_config
from src.tasks.blcs.models import BLCSTrackQueryModel


def _config() -> TrackQueryModelConfig:
    parsed = parse_model_config(
        {
            "model": {
                "name": "blcs_track_query",
                "hidden_dim": 16,
                "num_heads": 4,
                "num_stages": 4,
                "ffn_dim": 32,
                "num_queries": 4,
                "rope_dim": 4,
                "dropout": 0.0,
                "role_rope_enabled": True,
                "invisible_init_std": 0.02,
                "mhc": {
                    "coefficient_dim": 8,
                    "sinkhorn_iters": 5,
                    "eps": 1.0e-6,
                    "residual_identity_bias": 4.0,
                    "update_scale_init": 0.0,
                },
                "cswa": {
                    "compression_ratio": 2,
                    "window_radius": 1,
                    "backend": "reference",
                },
            }
        }
    )
    assert isinstance(parsed, TrackQueryModelConfig)
    return parsed


def _model() -> BLCSTrackQueryModel:
    return BLCSTrackQueryModel(_config()).eval()


def _inputs(*, views: int = 2, frames: int = 3) -> dict[str, torch.Tensor]:
    return {
        "ball_uv": torch.rand(1, views, frames, 4, 2),
        "ball_vis": torch.ones(1, views, frames, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, views, frames, 14, 2),
        "court_vis": torch.ones(1, views, frames, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, views, frames, dtype=torch.bool),
    }


def test_forward_public_contract_has_exactly_five_tensors() -> None:
    parameters = list(inspect.signature(BLCSTrackQueryModel.forward).parameters)
    assert parameters == [
        "self",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


def test_forward_returns_fixed_query_outputs() -> None:
    with torch.no_grad():
        output = _model()(**_inputs())
    assert output["position"].shape == (1, 3, 4, 3)
    assert output["presence_logits"].shape == (1, 3, 4)


def test_nonpadding_invisible_queries_use_learned_token_and_receive_gradient() -> None:
    model = _model()
    inputs = _inputs(views=1, frames=2)
    inputs["ball_vis"].zero_()

    output = model(**inputs)
    output["position"].sum().backward()

    gradient = model.observation_encoder.invisible_token.token.grad
    assert gradient is not None
    assert bool((gradient.abs() > 0).any())


def test_visibility_never_changes_attention_participation() -> None:
    model = _model()
    inputs = _inputs(views=1, frames=2)
    inputs["ball_vis"].zero_()
    observed: dict[str, torch.Tensor] = {}

    def capture_state(
        _module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        del args
        value = kwargs["camera_state_valid"]
        assert isinstance(value, torch.Tensor)
        observed["valid"] = value

    hook = model.stages[0].register_forward_pre_hook(capture_state, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        hook.remove()

    assert observed["valid"].all()


def test_padding_isolates_values_and_zeroes_padded_frame_outputs() -> None:
    model = _model()
    first = _inputs(views=2, frames=3)
    first["padding_mask"][:, :, -1] = True
    second = {key: value.clone() for key, value in first.items()}
    second["ball_uv"][:, :, -1] = 1.0
    second["court_kp"][:, :, -1] = 1.0
    second["ball_vis"][:, :, -1] = False
    second["court_vis"][:, :, -1] = False

    with torch.no_grad():
        output_first = model(**first)
        output_second = model(**second)

    torch.testing.assert_close(
        output_first["position"][:, :-1], output_second["position"][:, :-1]
    )
    torch.testing.assert_close(
        output_first["presence_logits"][:, :-1],
        output_second["presence_logits"][:, :-1],
    )
    assert not output_first["position"][:, -1].any()
    assert not output_first["presence_logits"][:, -1].any()


def test_nonrectangular_and_all_padding_inputs_remain_finite() -> None:
    model = _model()
    nonrectangular = _inputs(views=2, frames=3)
    nonrectangular["padding_mask"] = torch.tensor(
        [[[False, True, True], [True, False, True]]]
    )
    all_padding = _inputs(views=1, frames=2)
    all_padding["padding_mask"].fill_(True)

    with torch.no_grad():
        outputs = (model(**nonrectangular), model(**all_padding))

    for output in outputs:
        assert torch.isfinite(output["position"]).all()
        assert torch.isfinite(output["presence_logits"]).all()
    assert not outputs[1]["position"].any()
    assert not outputs[1]["presence_logits"].any()


@pytest.mark.parametrize("width", [3, 5])
def test_forward_rejects_nonexact_query_width(width: int) -> None:
    inputs = _inputs()
    inputs["ball_uv"] = torch.rand(1, 2, 3, width, 2)
    inputs["ball_vis"] = torch.ones(1, 2, 3, width, dtype=torch.bool)
    with pytest.raises(ValueError, match="model.num_queries"):
        _model()(**inputs)
