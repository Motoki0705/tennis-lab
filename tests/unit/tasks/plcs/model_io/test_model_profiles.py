"""Valid-shape regressions for standard PLCS model/adapter profiles."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.model_io import (
    PLCSInputProfile,
    PLCSModelIOAdapter,
    bind_plcs_model_io,
)
from src.tasks.plcs.model_io.attention_masks import prepare_axial_attention_masks
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)
from src.utils.models.components.ffn_layers import DeepSeekV4SwiGLU


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
        court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
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


def _split_model() -> PLCSMultiViewAxialSplitModel:
    """Small split model with every optional readout enabled and dropout off."""
    model: PLCSMultiViewAxialSplitModel = PLCSMultiViewAxialSplitModel(
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
        ffn_type="deepseek_v4_swiglu",
        predict_canonical_pose=True,
        max_views=2,
        max_seq_len=3,
        invisible_init_std=0.02,
        num_court_tokens=20,
    )
    model.eval()
    return model


def _assert_head_readouts_preserved(
    model: PLCSMultiViewAxialSplitModel,
    batch: dict[str, torch.Tensor],
    camera_mask: torch.Tensor,
    time_mask: torch.Tensor,
    *,
    label: str,
    autocast_dtype: torch.dtype | None,
) -> None:
    """Forward ``model`` and compare its readouts against the raw head outputs."""
    aux_position_head = model.aux_position_head
    assert aux_position_head is not None
    captured: dict[str, torch.Tensor] = {}

    def _hook(
        key: str,
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def capture(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[key] = output.detach().clone()

        return capture

    handles = [
        model.position_head.register_forward_hook(_hook("position")),
        aux_position_head.register_forward_hook(_hook("aux_position")),
    ]
    context = (
        torch.autocast("cpu", dtype=torch.float32, enabled=False)
        if autocast_dtype is None
        else torch.autocast("cpu", dtype=autocast_dtype)
    )
    try:
        with torch.no_grad(), context:
            output = model(
                batch["human_kp"],
                batch["court_kp"],
                batch["human_vis"],
                batch["padding_mask"],
                batch["court_vis"],
                camera_mask,
                time_mask,
            )
    finally:
        for handle in handles:
            handle.remove()
    for key in ("position", "aux_position"):
        assert output[key].dtype == captured[key].dtype, f"{label}: {key}"
        torch.testing.assert_close(output[key], captured[key], rtol=0.0, atol=0.0)


def test_split_model_output_strategy_matches_bound_adapter() -> None:
    model = _split_model()
    split_blocks = (
        *model.rot_camera_layers,
        *model.rot_time_layers,
        *model.pose_camera_layers,
        *model.pose_time_layers,
    )
    assert split_blocks
    assert all(isinstance(block.ffn, DeepSeekV4SwiGLU) for block in split_blocks)
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


def test_split_model_forward_preserves_head_readouts() -> None:
    """Default observations must keep the position readouts of their heads.

    ``_embed_observations`` used to return a float32 zero anchor for every
    profile and ``forward`` always added it, which promoted bfloat16-autocast
    readouts to float32. The default profile now returns ``None`` and leaves the
    head outputs untouched in values and dtype.
    """
    prefix = (1, 2, 3)
    batch = {
        "human_kp": torch.rand(*prefix, 17, 2),
        "court_kp": torch.rand(*prefix, 20, 2),
        "human_vis": torch.ones(*prefix, 17, dtype=torch.bool),
        "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
        "court_vis": torch.ones(*prefix, 20, dtype=torch.bool),
    }
    camera_mask, time_mask = prepare_axial_attention_masks(batch["padding_mask"])
    for label, autocast_dtype in (("float32", None), ("bfloat16", torch.bfloat16)):
        _assert_head_readouts_preserved(
            _split_model(),
            batch,
            camera_mask,
            time_mask,
            label=label,
            autocast_dtype=autocast_dtype,
        )
