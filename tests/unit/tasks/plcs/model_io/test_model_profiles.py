"""Valid-shape regressions for standard PLCS model/adapter profiles."""

from __future__ import annotations

import torch
from torch import nn

from src.tasks.plcs.model_io import (
    PLCSInputProfile,
    PLCSModelIOAdapter,
    bind_plcs_model_io,
)
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel


def _adapter(
    model_type: type[nn.Module],
    *,
    profile: PLCSInputProfile,
    output_rank: int,
    canonical: bool = False,
    auxiliary: bool = False,
) -> PLCSModelIOAdapter:
    return PLCSModelIOAdapter(
        model_type=model_type,
        profile=profile,
        num_court_tokens=20,
        camera_index=0,
        output_rank=output_rank,
        predict_canonical_pose=canonical,
        predict_auxiliary_position=auxiliary,
        max_views=2,
        max_sequence_length=3,
    )


def test_frame_model_runs_only_through_its_bound_profile() -> None:
    model = PLCSModel(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        rope_dim=4,
        rope_theta=10000.0,
        rope_theta_time=10000.0,
        rope_theta_camera=10000.0,
        rope_theta_type=10000.0,
        num_register_tokens=0,
        use_kp_id_embedding=False,
        use_rope=True,
        ffn_type="swiglu",
        predict_canonical_pose=False,
        invisible_init_std=0.02,
        num_court_tokens=20,
    ).eval()
    bound = bind_plcs_model_io(
        model,
        _adapter(
            PLCSModel,
            profile=PLCSInputProfile.FRAME,
            output_rank=2,
        ),
    )
    decoded = bound.run(
        {
            "human_kp": torch.rand(2, 17, 2),
            "court_kp": torch.rand(2, 20, 2),
            "human_vis": torch.ones(2, 17, dtype=torch.bool),
            "padding_mask": torch.zeros(2, dtype=torch.bool),
            "court_vis": torch.ones(2, 20, dtype=torch.bool),
        }
    )
    assert decoded.position.shape == (2, 3)
    assert decoded.rotation.shape == (2, 2)


def test_interleaved_multiview_model_runs_through_multiview_profile() -> None:
    model = PLCSMultiViewModel(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        rope_dim=4,
        rope_theta=10000.0,
        rope_theta_time=10000.0,
        rope_theta_camera=10000.0,
        rope_theta_type=10000.0,
        ffn_type="swiglu",
        predict_canonical_pose=False,
        max_views=2,
        max_seq_len=3,
        invisible_init_std=0.02,
        num_court_tokens=20,
    ).eval()
    bound = bind_plcs_model_io(
        model,
        _adapter(
            PLCSMultiViewModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
        ),
    )
    prefix = (1, 2, 3)
    decoded = bound.run(
        {
            "human_kp": torch.rand(*prefix, 17, 2),
            "court_kp": torch.rand(*prefix, 20, 2),
            "human_vis": torch.ones(*prefix, 17, dtype=torch.bool),
            "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
            "court_vis": torch.ones(*prefix, 20, dtype=torch.bool),
        }
    )
    assert decoded.position.shape == (1, 3, 3)
    assert decoded.rotation.shape == (1, 3, 2)


def test_split_model_output_strategy_matches_bound_adapter() -> None:
    model = PLCSMultiViewAxialSplitModel(
        hidden_dim=16,
        num_layers=0,
        num_task_layers=1,
        rot_num_task_layers=1,
        pose_num_task_layers=1,
        canonical_on_rotation_branch=True,
        aux_position_on_rotation_branch=True,
        detach_pose_branch=False,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        rope_dim=4,
        rope_theta_time=10000.0,
        rope_theta_camera=10000.0,
        ffn_type="swiglu",
        predict_canonical_pose=True,
        max_views=2,
        max_seq_len=3,
        invisible_init_std=0.02,
        num_court_tokens=20,
    ).eval()
    bound = bind_plcs_model_io(
        model,
        _adapter(
            PLCSMultiViewAxialSplitModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
            canonical=True,
            auxiliary=True,
        ),
    )
    prefix = (1, 2, 3)
    decoded = bound.run(
        {
            "human_kp": torch.rand(*prefix, 17, 2),
            "court_kp": torch.rand(*prefix, 20, 2),
            "human_vis": torch.ones(*prefix, 17, dtype=torch.bool),
            "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
            "court_vis": torch.ones(*prefix, 20, dtype=torch.bool),
        }
    )
    assert decoded.canonical_pose is not None
    assert decoded.canonical_pose.shape == (1, 3, 17, 3)
    assert decoded.auxiliary_position is not None
    assert decoded.auxiliary_position.shape == (1, 3, 3)
