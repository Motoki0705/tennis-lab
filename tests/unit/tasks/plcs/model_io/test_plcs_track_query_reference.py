"""PLCS v2 adapter, factory-contract, and checkpoint metadata tests."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMismatchError,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.model_io import (
    PLCSTrackQueryReferenceIOAdapter,
    bind_plcs_model_io,
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_track_query_reference,
    write_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.model_io.contracts import PLCSInputProfile
from src.tasks.plcs.models.plcs_track_query_reference_model import (
    PLCSTrackQueryReferenceModel,
)


class _ReferenceTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        reference_view_index: Tensor,
    ) -> dict[str, Tensor]:
        del human_vis, court_kp, court_vis, padding_mask, reference_view_index
        batch_size, _, frames = human_kp.shape[:3]
        return {
            "position": human_kp.new_zeros(batch_size, frames, 2, 3),
            "rotation": human_kp.new_zeros(batch_size, frames, 2, 2),
            "presence_logits": human_kp.new_zeros(batch_size, frames, 2),
        }


def _model_config(
    *, selector_mode: str = "reference"
) -> PLCSModelConfig:
    name = (
        "plcs_track_query_reference"
        if selector_mode == "reference"
        else "plcs_track_query_reference_ablation"
    )
    raw: dict[str, object] = {
        "name": name,
        "hidden_dim": 24,
        "num_heads": 4,
        "ffn_dim": 48,
        "num_queries": 2,
        "num_stages": 4,
        "num_joints": 17,
        "rope_dim": 6,
        "rope_theta": 10_000.0,
        "ffn_type": "swiglu",
        "dropout": 0.0,
        "invisible_init_std": 0.02,
        "target_frame_contract": "reference_camera_court_rzpi_v1",
        "track_query_rope_contract": "time_camera_reference_selector_v1",
        "reference_selector_mode": selector_mode,
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
    if selector_mode != "reference":
        raw["ffn_mode"] = "shared"
        raw["mhc_writeback"] = "layer_end"
    return PLCSModelConfig.from_mapping(raw)


def _adapter(
    *, selector_mode: str = "reference"
) -> PLCSTrackQueryReferenceIOAdapter:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    return PLCSTrackQueryReferenceIOAdapter(
        model_type=_ReferenceTrackingModel,
        num_queries=2,
        num_court_tokens=14,
        num_joints=17,
        court_keypoint_contract=contract,
        target_frame_contract="reference_camera_court_rzpi_v1",
        track_query_rope_contract="time_camera_reference_selector_v1",
        reference_selector_mode=selector_mode,
    )


def _batch(*, selector_mode: str = "reference") -> dict[str, object]:
    court_contract = resolve_court_keypoint_contract("camera_view_v2")
    negative = build_court_view_record(
        camera_id="camera_0",
        camera_center_court_m=(0.0, -10.0, 3.0),
        contract=court_contract,
    )
    positive = build_court_view_record(
        camera_id="camera_1",
        camera_center_court_m=(0.0, 10.0, 3.0),
        contract=court_contract,
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            ("camera_0", "camera_1")
        ),
        selected_views=(negative, positive),
        reference_camera_id="camera_1",
    )
    provenance = selection.provenance
    batch: dict[str, object] = {
        "human_kp": torch.rand(1, 2, 3, 2, 17, 2),
        "human_vis": torch.ones(1, 2, 3, 2, 17, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
        "reference_view_index": torch.tensor([1], dtype=torch.int64),
        "view_camera_ids": torch.tensor([[0, 1]], dtype=torch.int64),
        "reference_camera_id": torch.tensor([1], dtype=torch.int64),
        "reference_from_physical": torch.tensor(
            provenance.reference_from_physical,
            dtype=torch.float32,
        ).unsqueeze(0),
        "physical_from_reference": torch.tensor(
            provenance.physical_from_reference,
            dtype=torch.float32,
        ).unsqueeze(0),
        "court_keypoint_metadata": court_keypoint_contract_document(
            court_contract
        ),
        "court_reference_provenance": provenance,
        "reference_view_selection": (selection,),
        "stable_camera_id_table": (selection.stable_camera_id_table,),
    }
    write_track_query_reference_contract(
        batch,
        TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode(selector_mode)
        ),
        location="test batch",
    )
    return batch


def test_v2_adapter_builds_exact_six_tensor_call() -> None:
    adapter = _adapter()
    call = adapter.build_call(_batch())

    assert adapter.profile is PLCSInputProfile.TRACK_QUERY
    assert tuple(call.kwargs) == (
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
    )
    assert cast(Tensor, call.kwargs["reference_view_index"]).dtype == torch.int64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda batch: batch.pop("reference_view_index"), "missing"),
        (
            lambda batch: batch.__setitem__(
                "reference_camera_id", torch.tensor([0], dtype=torch.int64)
            ),
            "exactly equal",
        ),
        (
            lambda batch: cast(Tensor, batch["padding_mask"]).__setitem__(
                (0, 1, 0), True
            ),
            "unmasked reference-view",
        ),
        (
            lambda batch: cast(
                Tensor, batch["physical_from_reference"]
            ).zero_(),
            "must equal",
        ),
    ],
)
def test_v2_adapter_rejects_missing_identity_mismatch_and_padded_reference(
    mutation: object,
    message: str,
) -> None:
    batch = _batch()
    assert callable(mutation)
    mutation(batch)
    with pytest.raises(ValueError, match=message):
        _adapter().build_call(batch)


def test_v2_adapter_rejects_selector_metadata_mismatch_before_call() -> None:
    with pytest.raises(ModelInputContractError, match="does not exactly match"):
        _adapter(selector_mode="selector_zero").build_call(_batch())


def test_plcs_contract_resolver_matches_model_type_court_target_rope_selector() -> None:
    court = resolve_court_keypoint_contract("camera_view_v2")
    resolved = resolve_plcs_track_query_reference_contract(
        _model_config(),
        court,
    )
    assert resolved == TrackQueryReferenceContract.reference_v2(
        ReferenceSelectorMode.REFERENCE
    )

    with pytest.raises(ValueError, match="CourtKP20"):
        resolve_plcs_track_query_reference_contract(
            _model_config(),
            resolve_court_keypoint_contract("physical_v1"),
        )


def test_checkpoint_metadata_is_exact_and_shape_independent() -> None:
    selector = TrackQueryReferenceContract.reference_v2(
        ReferenceSelectorMode.REFERENCE
    )
    selector_zero = TrackQueryReferenceContract.reference_v2(
        ReferenceSelectorMode.SELECTOR_ZERO
    )
    checkpoint: dict[str, object] = {"state_dict": {"same.shape": torch.ones(1)}}
    write_plcs_checkpoint_track_query_reference(checkpoint, selector)
    validate_plcs_checkpoint_track_query_reference(checkpoint, selector)

    with pytest.raises(TrackQueryReferenceContractMismatchError):
        validate_plcs_checkpoint_track_query_reference(checkpoint, selector_zero)


def test_metadata_free_checkpoint_is_allowed_only_for_explicit_legacy_v1() -> None:
    validate_plcs_checkpoint_track_query_reference(
        {},
        TrackQueryReferenceContract.legacy_v1(),
    )
    with pytest.raises(ValueError, match="metadata is absent"):
        validate_plcs_checkpoint_track_query_reference(
            {},
            TrackQueryReferenceContract.reference_v2(
                ReferenceSelectorMode.REFERENCE
            ),
        )


def test_factory_model_type_remains_exact_for_reference_class() -> None:
    config = _model_config()
    model = PLCSTrackQueryReferenceModel(config)
    assert type(model) is PLCSTrackQueryReferenceModel


def test_binding_rejects_shape_compatible_model_adapter_selector_mismatch() -> None:
    model = PLCSTrackQueryReferenceModel(_model_config())
    court = resolve_court_keypoint_contract("camera_view_v2")
    mismatched = PLCSTrackQueryReferenceIOAdapter(
        model_type=PLCSTrackQueryReferenceModel,
        num_queries=2,
        num_court_tokens=14,
        num_joints=17,
        court_keypoint_contract=court,
        target_frame_contract="reference_camera_court_rzpi_v1",
        track_query_rope_contract="time_camera_reference_selector_v1",
        reference_selector_mode="selector_zero",
    )

    with pytest.raises(ModelAdapterMismatchError, match="do not match exactly"):
        bind_plcs_model_io(model, mismatched)
