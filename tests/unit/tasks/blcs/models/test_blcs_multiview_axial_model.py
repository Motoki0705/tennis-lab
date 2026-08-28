"""Shape tests for BLCS axial multiview models."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.tasks.base.model_io import ModelCall
from src.tasks.blcs.configuration import AxialModelConfig, parse_model_config
from src.tasks.blcs.model_io import compose_blcs_trajectory_model_io


def _config(*, max_num_cameras: int, max_seq_len: int) -> dict[str, object]:
    return {
        "court_keypoints": {"selector": "physical_v1"},
        "model": {
            "name": "blcs_multiview_axial",
            "io": {"input_profile": "multiview"},
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "attention_type": "mha",
            "num_kv_heads": None,
            "ffn_dim": 64,
            "ffn_type": "swiglu",
            "camera_layers_per_stage": [1],
            "time_layers_per_stage": [1],
            "time_global_stage_mask": [False],
            "max_num_cameras": max_num_cameras,
            "max_seq_len": max_seq_len,
            "dropout": 0.0,
            "rope_dim": 4,
            "rope_theta_time": 10000.0,
            "rope_theta_camera": 10000.0,
            "predict_velocity": False,
            "invisible_init_std": 0.02,
            "num_court_tokens": 20,
            "time_window_radius": 2,
        }
    }


def test_multiview_axial_model_forward_accepts_single_view() -> None:
    torch.manual_seed(0)
    binding = compose_blcs_trajectory_model_io(
        OmegaConf.create(_config(max_num_cameras=1, max_seq_len=4))
    )
    binding.model.eval()
    inputs = {
        "ball_uv": torch.rand(2, 1, 4, 2),
        "court_kp": torch.rand(2, 1, 4, 20, 2),
        "ball_vis": torch.ones(2, 1, 4),
        "padding_mask": torch.zeros(2, 1, 4, dtype=torch.bool),
        "court_vis": torch.ones(2, 1, 4, 20),
    }

    with torch.no_grad():
        out = binding.execute_call(binding.build_call(inputs))

    assert out["position"].shape == (2, 4, 3)


def test_multiview_axial_config_preserves_checkpoint_attention_contract() -> None:
    config = OmegaConf.create(_config(max_num_cameras=1, max_seq_len=4))

    parsed = parse_model_config(config)

    assert isinstance(parsed, AxialModelConfig)
    assert parsed.time_window_radius == 2
    assert parsed.time_global_stage_mask == (False,)


def test_multiview_axial_model_constructs_configured_local_global_schedule() -> None:
    config = _config(max_num_cameras=1, max_seq_len=4)
    model = config["model"]
    assert isinstance(model, dict)
    model["num_layers"] = 2
    model["camera_layers_per_stage"] = [1, 1]
    model["time_layers_per_stage"] = [2, 2]
    model["time_global_stage_mask"] = [False, True]

    binding = compose_blcs_trajectory_model_io(OmegaConf.create(config))

    schedule = [
        [type(layer).__name__ for layer in stage.time_layers]
        for stage in binding.model.stages
    ]
    assert schedule == [
        ["_SlidingTimeAttention", "_SlidingTimeAttention"],
        ["_SlidingTimeAttention", "_GlobalTimeAttention"],
    ]


def test_multiview_axial_model_masks_invisible_court_coordinates() -> None:
    torch.manual_seed(1)
    binding = compose_blcs_trajectory_model_io(
        OmegaConf.create(_config(max_num_cameras=2, max_seq_len=3))
    )
    binding.model.eval()
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 2),
        "court_kp": torch.rand(1, 2, 3, 20, 2),
        "ball_vis": torch.ones(1, 2, 3),
        "padding_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
        "court_vis": torch.ones(1, 2, 3, 20),
    }
    inputs["court_vis"][:, 1, :, 4] = 0
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["court_kp"][:, 1, :, 4] = torch.nan

    with torch.no_grad():
        prepared = binding.build_call(inputs)
        output = binding.execute_call(prepared)
        changed_call = dict(prepared.kwargs)
        changed_call["court_kp"] = changed["court_kp"]
        changed_output = binding.execute_call(ModelCall(kwargs=changed_call))

    torch.testing.assert_close(output["position"], changed_output["position"])
