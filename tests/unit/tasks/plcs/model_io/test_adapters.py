"""Unit tests for strict PLCS model-I/O boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    ModelOutputContractError,
)
from src.tasks.plcs.model_io import (
    PLCSInputProfile,
    PLCSModelIOAdapter,
    PLCSTrackQueryIOAdapter,
    bind_plcs_model_io,
)


class _StandardModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        human_mask: Tensor,
        court_vis: Tensor,
    ) -> dict[str, Tensor]:
        del court_kp, human_vis, human_mask, court_vis
        return {
            "position": torch.zeros(*human_kp.shape[:-2], 3),
            "rotation": torch.ones(*human_kp.shape[:-2], 2),
        }


class _OtherModel(nn.Module):
    pass


class _TrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        camera_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        temporal_attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        self.forward_calls += 1
        del (
            detection_mask,
            court_kp,
            court_vis,
            frame_mask,
            camera_state_valid,
            spatial_attention_mask,
            temporal_attention_mask,
        )
        batch_size, _, frames = human_kp.shape[:3]
        return {
            "position": human_kp.new_zeros(batch_size, frames, 3, 3),
            "rotation": human_kp.new_zeros(batch_size, frames, 3, 2),
            "presence_logits": human_kp.new_zeros(batch_size, frames, 3),
        }


def _standard_adapter(
    *,
    profile: PLCSInputProfile = PLCSInputProfile.MULTIVIEW,
    output_rank: int = 3,
    min_views: int = 1,
) -> PLCSModelIOAdapter:
    return PLCSModelIOAdapter(
        model_type=_StandardModel,
        profile=profile,
        num_court_tokens=20,
        camera_index=0,
        output_rank=output_rank,
        predict_canonical_pose=False,
        predict_auxiliary_position=False,
        max_views=4,
        max_sequence_length=8,
        min_views=min_views,
    )


def _canonical_batch(
    *, batch_size: int = 2, views: int = 2, frames: int = 3
) -> dict[str, Tensor]:
    prefix = (batch_size, views, frames)
    return {
        "human_kp": torch.rand(*prefix, 17, 2),
        "court_kp": torch.rand(*prefix, 20, 2),
        "human_vis": torch.ones(*prefix, 17, dtype=torch.bool),
        "human_mask": torch.ones(*prefix, dtype=torch.bool),
        "court_vis": torch.ones(*prefix, 20, dtype=torch.bool),
        "position": torch.rand(batch_size, frames, 3),
        "rotation": torch.rand(batch_size, frames, 2),
    }


def test_exact_model_adapter_pair_is_bound_once() -> None:
    bound = bind_plcs_model_io(_StandardModel(), _standard_adapter())
    assert type(bound.model) is _StandardModel


def test_exact_model_adapter_pair_rejects_another_model_before_forward() -> None:
    with pytest.raises(ModelAdapterMismatchError, match="exact model type"):
        bind_plcs_model_io(_OtherModel(), _standard_adapter())


def test_sequence_profile_flattens_call_and_restores_output() -> None:
    adapter = _standard_adapter(
        profile=PLCSInputProfile.SEQUENCE,
        output_rank=2,
    )
    prepared = adapter.prepare_training_batch(_canonical_batch())
    human_kp = cast(Tensor, prepared.call.kwargs["human_kp"])
    assert human_kp.shape == (6, 17, 2)

    decoded = adapter.decode_prepared_output(
        {
            "position": torch.zeros(6, 3),
            "rotation": torch.ones(6, 2),
        },
        prepared,
    )
    assert decoded.position.shape == (2, 3, 3)
    assert decoded.rotation.shape == (2, 3, 2)


def test_frame_profile_rejects_temporal_training_batch() -> None:
    adapter = _standard_adapter(profile=PLCSInputProfile.FRAME, output_rank=2)
    with pytest.raises(ModelInputContractError, match="exactly one frame"):
        adapter.prepare_training_batch(_canonical_batch(frames=2))


def test_profile_rejects_static_output_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="requires output_rank=3"):
        _standard_adapter(output_rank=2)


def _empty_time_axis(batch: dict[str, Tensor]) -> None:
    batch["human_kp"] = torch.empty(2, 2, 0, 17, 2)
    batch["court_kp"] = torch.empty(2, 2, 0, 20, 2)
    batch["human_vis"] = torch.empty(2, 2, 0, 17, dtype=torch.bool)
    batch["human_mask"] = torch.empty(2, 2, 0, dtype=torch.bool)
    batch["court_vis"] = torch.empty(2, 2, 0, 20, dtype=torch.bool)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda batch: batch.pop("human_vis"), "human_vis.*missing"),
        (
            lambda batch: batch.__setitem__(
                "human_mask", batch["human_mask"].to(torch.int16) + 2
            ),
            "explicit 0/1",
        ),
        (
            lambda batch: batch.__setitem__(
                "court_kp", torch.rand(2, 2, 3, 19, 2)
            ),
            "axis 3",
        ),
        (
            lambda batch: batch["human_kp"].fill_(1.1),
            "normalized UV",
        ),
        (_empty_time_axis, "non-empty"),
    ],
)
def test_multiview_boundary_rejects_invalid_input(
    mutation: object,
    message: str,
) -> None:
    batch: dict[str, Tensor] = _canonical_batch()
    callable_mutation = mutation
    assert callable(callable_mutation)
    callable_mutation(batch)
    with pytest.raises(ModelInputContractError, match=message):
        _standard_adapter().build_call(batch)


def test_camera_token_profile_rejects_single_view_before_forward() -> None:
    adapter = _standard_adapter(min_views=2)
    with pytest.raises(ModelInputContractError, match="at least 2 views"):
        adapter.build_call(_canonical_batch(views=1))


def test_output_contract_rejects_unknown_or_malformed_outputs() -> None:
    adapter = _standard_adapter()
    with pytest.raises(ModelOutputContractError, match="unknown=.*legacy"):
        adapter.decode_output(
            {
                "position": torch.zeros(2, 3, 3),
                "rotation": torch.zeros(2, 3, 2),
                "legacy": torch.zeros(1),
            }
        )
    with pytest.raises(ModelOutputContractError, match="rotation"):
        adapter.decode_output(
            {
                "position": torch.zeros(2, 3, 3),
                "rotation": torch.zeros(2, 4, 2),
            }
        )


def test_numpy_multiview_boundary_broadcasts_explicit_shared_court() -> None:
    prepared = _standard_adapter().prepare_multiview_observations(
        human_kp=np.zeros((2, 2, 3, 17, 2), dtype=np.float32),
        court_kp=np.zeros((2, 3, 20, 2), dtype=np.float32),
        human_vis=np.ones((2, 2, 3, 17), dtype=np.bool_),
        human_mask=np.ones((2, 2, 3), dtype=np.bool_),
        court_vis=np.ones((2, 3, 20), dtype=np.bool_),
    )
    court_kp = cast(Tensor, prepared.call.kwargs["court_kp"])
    assert court_kp.shape == (2, 2, 3, 20, 2)


def _tracking_adapter() -> PLCSTrackQueryIOAdapter:
    return PLCSTrackQueryIOAdapter(
        model_type=_TrackingModel,
        num_queries=3,
        num_court_tokens=14,
        num_joints=17,
        mask_invisible_observations=True,
    )


def _tracking_batch() -> dict[str, Tensor]:
    return {
        "human_kp": torch.rand(1, 2, 3, 2, 17, 2),
        "detection_mask": torch.ones(1, 2, 3, 2, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_position": torch.rand(1, 3, 2, 3),
        "target_rotation": torch.rand(1, 3, 2, 2),
        "target_presence": torch.ones(1, 3, 2, dtype=torch.bool),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_instance_id": torch.ones(1, 3, 2, dtype=torch.int64),
    }


def test_tracking_boundary_validates_inputs_targets_and_decodes_required_presence() -> None:
    adapter = _tracking_adapter()
    prepared = adapter.prepare_training_batch(_tracking_batch())
    decoded = adapter.decode_prepared_output(
        {
            "position": torch.zeros(1, 3, 3, 3),
            "rotation": torch.zeros(1, 3, 3, 2),
            "presence_logits": torch.zeros(1, 3, 3),
        },
        prepared,
    )
    assert decoded.presence_logits.shape == (1, 3, 3)


def test_tracking_boundary_rejects_incomplete_court_and_mask_dtype() -> None:
    adapter = _tracking_adapter()
    batch = _tracking_batch()
    batch["court_kp"] = torch.rand(1, 2, 3, 13, 2)
    batch["court_vis"] = torch.ones(1, 2, 3, 13, dtype=torch.bool)
    with pytest.raises(ModelInputContractError, match="axis 3"):
        adapter.build_call(batch)

    batch = _tracking_batch()
    batch["detection_mask"] = batch["detection_mask"].float()
    with pytest.raises(ModelInputContractError, match="torch.bool"):
        adapter.build_call(batch)


@pytest.mark.parametrize(
    ("padding_key", "padding_index"),
    [("frame_mask", (0, 1)), ("view_mask", (0, 1))],
)
def test_tracking_boundary_rejects_detections_in_padding_before_model_entry(
    padding_key: str,
    padding_index: tuple[int, int],
) -> None:
    model = _TrackingModel()
    adapter = _tracking_adapter()
    bound = bind_plcs_model_io(model, adapter)
    batch = _tracking_batch()
    batch[padding_key][padding_index] = False

    with pytest.raises(ModelInputContractError, match="padded view or frame"):
        bound.execute_call(adapter.build_call(batch))

    assert model.forward_calls == 0


def test_tracking_boundary_accepts_explicitly_empty_padded_observations() -> None:
    model = _TrackingModel()
    adapter = _tracking_adapter()
    bound = bind_plcs_model_io(model, adapter)
    batch = _tracking_batch()
    batch["frame_mask"][0, -1] = False
    batch["view_mask"][0, -1] = False
    batch["detection_mask"][:, -1] = False
    batch["detection_mask"][:, :, -1] = False

    decoded = bound.run(batch)

    assert decoded.position.shape == (1, 3, 3, 3)
    assert model.forward_calls == 1


def test_tracking_boundary_rejects_inactive_non_sentinel_instance_id() -> None:
    batch = _tracking_batch()
    batch["target_presence"][:, 0, 0] = False
    with pytest.raises(ModelInputContractError, match="target_instance_id=-1"):
        _tracking_adapter().prepare_training_batch(batch)


def test_tracking_output_rejects_missing_presence() -> None:
    output: Mapping[str, object] = {
        "position": torch.zeros(1, 3, 3, 3),
        "rotation": torch.zeros(1, 3, 3, 2),
    }
    with pytest.raises(ModelOutputContractError, match="presence_logits"):
        _tracking_adapter().decode_output(output)
