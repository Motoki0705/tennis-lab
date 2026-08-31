from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor, nn

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    ModelInputContractError,
    TrackQueryReferenceContract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io import (
    PLCSReferenceMetadata,
    PLCSTrackQueryIOAdapter,
    PLCSTrackQueryReferenceIOAdapter,
    build_plcs_model_io,
)
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)
from src.utils.configuration import PathResolver


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            human_vis,
            court_kp,
            court_vis,
            padding_mask,
        )
        batch, _, frames = human_kp.shape[:3]
        rotation = torch.tensor([0.0, 1.0], device=human_kp.device)
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": rotation.expand(batch, frames, 2, -1),
            "presence_logits": torch.tensor([2.0, -2.0], device=human_kp.device).expand(
                batch, frames, -1
            ),
        }


class _FixedReferenceTrackingModel(nn.Module):
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
        batch, _, frames = human_kp.shape[:3]
        rotation = torch.tensor([0.0, 1.0], device=human_kp.device)
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": rotation.expand(batch, frames, 2, -1),
            "presence_logits": torch.tensor([2.0, -2.0], device=human_kp.device).expand(
                batch, frames, -1
            ),
        }


def _reference_metadata() -> PLCSReferenceMetadata:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(0.0, -10.0, 3.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(0.0, 10.0, 3.0),
            contract=contract,
        ),
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(("camera_0", "camera_1"))
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=views,
        reference_camera_id="camera_1",
    )
    forward = torch.tensor(
        selection.provenance.reference_from_physical,
        dtype=torch.float32,
    ).unsqueeze(0)
    return PLCSReferenceMetadata(
        selections=(selection,),
        stable_camera_id_tables=(table,),
        reference_view_index=torch.tensor([1], dtype=torch.int64),
        view_camera_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        reference_camera_id=torch.tensor([1], dtype=torch.int64),
        reference_from_physical=forward,
        physical_from_reference=forward.transpose(-1, -2),
        track_query_contract=TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode.REFERENCE
        ),
    )


def test_predictor_returns_cpu_lifecycle_and_yaw_outputs() -> None:
    predictor = PLCSTrackingPredictor(
        model=_FixedTrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedTrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
            court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
        ),
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        human_vis=torch.ones(*shape, 17, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        denormalize=True,
    )

    position_meters = cast("Tensor", result["position_meters"])
    presence = cast("Tensor", result["presence"])
    assert position_meters.shape == (1, 3, 2, 3)
    assert presence[..., 0].all()
    assert not presence[..., 1].any()
    torch.testing.assert_close(
        position_meters,
        torch.full((1, 3, 2, 3), 11.885),
    )
    torch.testing.assert_close(
        cast("Tensor", result["yaw_radians"]),
        torch.full((1, 3, 2), torch.pi / 2),
    )
    assert all(
        value.device.type == "cpu"
        for value in result.values()
        if isinstance(value, Tensor)
    )


def test_reference_tracking_predictor_requires_and_round_trips_typed_metadata() -> None:
    court_contract = resolve_court_keypoint_contract("camera_view_v2")
    adapter = PLCSTrackQueryReferenceIOAdapter(
        model_type=_FixedReferenceTrackingModel,
        num_queries=2,
        num_court_tokens=14,
        num_joints=17,
        court_keypoint_contract=court_contract,
        target_frame_contract="reference_camera_court_rzpi_v1",
        track_query_rope_contract="time_camera_reference_selector_v1",
        reference_selector_mode="reference",
    )
    model = _FixedReferenceTrackingModel()
    model.target_frame_contract = adapter.target_frame_contract
    model.track_query_rope_contract = adapter.track_query_rope_contract
    model.reference_selector_mode = adapter.reference_selector_mode
    predictor = PLCSTrackingPredictor(
        model=model,
        adapter=adapter,
        device=torch.device("cpu"),
    )
    metadata = _reference_metadata()
    inputs: dict[str, Any] = {
        "human_kp": torch.zeros(1, 2, 2, 2, 17, 2),
        "human_vis": torch.ones(1, 2, 2, 2, 17, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "tracking_metrics": TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        "denormalize": True,
        "court_keypoint_metadata": court_keypoint_contract_document(court_contract),
    }

    result = predictor.predict(**inputs, reference_metadata=metadata)

    returned = result["reference_metadata"]
    assert isinstance(returned, PLCSReferenceMetadata)
    assert returned.reference_camera_ids == ("camera_1",)
    assert returned.reference_view_index.tolist() == [1]
    torch.testing.assert_close(
        returned.physical_from_reference,
        returned.reference_from_physical.transpose(-1, -2),
    )
    assert returned.track_query_contract == adapter.reference_contract
    assert all(
        value.device.type == "cpu"
        for value in result.values()
        if isinstance(value, Tensor)
    )

    with pytest.raises(ModelInputContractError, match="requires explicit typed"):
        predictor.predict(**inputs)


def test_checkpoint_restoration_retains_exact_ablation_model_adapter_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=track_query_ablation_d",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ],
        )
    binding = build_plcs_model_io(PLCSTrainingConfig.from_config(config))
    assert isinstance(binding.adapter, PLCSTrackQueryIOAdapter)
    checkpoint = tmp_path / "ablation.ckpt"
    document: dict[str, object] = {"hyper_parameters": {"config": config}}
    physical_v1 = resolve_court_keypoint_contract("physical_v1")
    write_model_artifact_court_keypoint_contract(document, physical_v1)
    observed: dict[str, object] = {}

    def load_module(
        cls: type[PLCSTrackingPredictor],
        path: Path,
        module_type: type[nn.Module],
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, torch.device]:
        del cls, module_type
        observed["path"] = path
        observed["strict"] = kwargs["strict"]
        observed["weights_only"] = kwargs["weights_only"]
        return (
            SimpleNamespace(model=binding.model, io_adapter=binding.adapter),
            torch.device("cpu"),
        )

    monkeypatch.setattr(
        PLCSTrackingPredictor,
        "_load_single_lightning_module",
        classmethod(load_module),
    )
    monkeypatch.setattr(
        PLCSTrackingPredictor,
        "_ensure_checkpoint",
        staticmethod(lambda value, *, resolver: [checkpoint]),
    )
    monkeypatch.setattr(
        "src.tasks.plcs.inference.tracking_predictor.load_and_validate_checkpoint",
        lambda path: document,
    )

    predictor = PLCSTrackingPredictor.load_from_checkpoint(
        checkpoint,
        resolver=cast("PathResolver", object()),
        device="cpu",
        court_keypoint_contract=physical_v1,
    )

    assert observed == {
        "path": checkpoint,
        "strict": True,
        "weights_only": False,
    }
    assert type(predictor.model) is PLCSTrackQueryAblationModel
    assert predictor.io_adapter is binding.adapter
    assert predictor.io_adapter.model_type is PLCSTrackQueryAblationModel


def test_tracking_checkpoint_factory_rejects_mismatched_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mismatched.ckpt"
    torch.save(
        {
            "court_coordinate_normalization": {
                "identity": "isotropic_half_length",
                "scale_xyz_m": [5.485, 11.885, 1.07],
                "position_unit": "m / scale_xyz_m",
                "velocity_unit": "m/s / scale_xyz_m",
            }
        },
        checkpoint,
    )
    monkeypatch.setattr(
        PLCSTrackingPredictor,
        "_ensure_checkpoint",
        staticmethod(lambda value, *, resolver: [checkpoint]),
    )

    with pytest.raises(ValueError, match="mismatched"):
        PLCSTrackingPredictor.load_from_checkpoint(
            checkpoint,
            resolver=cast("PathResolver", object()),
            device="cpu",
        )
