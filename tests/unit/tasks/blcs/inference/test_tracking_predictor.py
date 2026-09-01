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
    bind_model_io,
    write_checkpoint_track_query_reference_contract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor
from src.tasks.blcs.model_io import (
    BLCSReferenceMetadata,
    TrackQueryBoundModelIO,
    TrackQueryModelIOAdapter,
    TrackQueryReferenceModelIOAdapter,
    compose_blcs_track_query_model_io,
)
from src.tasks.blcs.models import (
    BLCSTrackQueryModel,
)
from src.tasks.blcs.models.blcs_track_query_reference_model import (
    BLCSTrackQueryReferenceModel,
)
from src.utils.configuration import PathResolver
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


class _FixedTrackingModel(BLCSTrackQueryModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> BLCSTrackingPrediction:
        del (
            ball_vis,
            court_kp,
            court_vis,
            padding_mask,
        )
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.tensor([-2.0, 2.0], device=ball_uv.device).expand(
                batch, frames, -1
            ),
        }


class _FixedReferenceTrackingModel(BLCSTrackQueryReferenceModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(  # type: ignore[override]
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        reference_view_index: Tensor,
    ) -> BLCSTrackingPrediction:
        del ball_vis, court_kp, court_vis, padding_mask, reference_view_index
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.ones(batch, frames, 2, device=ball_uv.device),
        }


def _reference_metadata() -> BLCSReferenceMetadata:
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
    return BLCSReferenceMetadata(
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


def test_predictor_returns_cpu_query_presence_and_positions() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(
        model_io=binding,
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_vis=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        denormalize=False,
    )

    assert result.position.shape == (1, 3, 2, 3)
    assert not result.presence[..., 0].any()
    assert result.presence[..., 1].all()
    assert result.position.device.type == "cpu"
    assert result.presence_logits.device.type == "cpu"
    assert result.presence_probability.device.type == "cpu"
    assert result.presence.device.type == "cpu"

    physical = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_vis=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        denormalize=True,
    )
    torch.testing.assert_close(physical.position, torch.full((1, 3, 2, 3), 11.885))


def test_reference_predictor_requires_and_round_trips_typed_metadata() -> None:
    reference_contract = TrackQueryReferenceContract.reference_v2(
        ReferenceSelectorMode.REFERENCE
    )
    court_contract = resolve_court_keypoint_contract("camera_view_v2")
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedReferenceTrackingModel(),
            TrackQueryReferenceModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
                court_keypoint_contract=court_contract,
                track_query_reference_contract=reference_contract,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(
        binding,
        torch.device("cpu"),
        court_keypoint_contract=court_contract,
    )
    metadata = _reference_metadata()
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(document, court_contract)
    inputs = {
        "ball_uv": torch.zeros(1, 2, 2, 2, 2),
        "ball_vis": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "denormalize": False,
        "court_keypoint_document": document,
    }

    result = predictor.predict(**inputs, reference_metadata=metadata)

    assert result.reference_metadata is not None
    assert result.reference_metadata.reference_camera_ids == ("camera_1",)
    assert result.reference_metadata.reference_view_index.tolist() == [1]
    assert result.reference_metadata.reference_view_index.device.type == "cpu"
    assert result.court_reference_provenance == tuple(
        selection.provenance for selection in metadata.selections
    )

    with pytest.raises(ModelInputContractError, match="requires explicit typed"):
        predictor.predict(**inputs)

    mismatched = ReferenceViewSelection.create(
        stable_camera_id_table=metadata.stable_camera_id_tables[0],
        selected_views=metadata.selections[0].selected_views,
        reference_camera_id="camera_0",
    )
    with pytest.raises(ModelInputContractError, match="do not match"):
        predictor.predict(
            **inputs,
            reference_metadata=metadata,
            court_reference_provenance=(mismatched.provenance,),
        )


def test_legacy_tracking_predictor_rejects_v2_reference_metadata() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(binding, torch.device("cpu"))

    with pytest.raises(ModelInputContractError, match="Legacy BLCS"):
        predictor.predict(
            ball_uv=torch.zeros(1, 2, 2, 2, 2),
            ball_vis=torch.ones(1, 2, 2, 2, dtype=torch.bool),
            court_kp=torch.zeros(1, 2, 2, 14, 2),
            court_vis=torch.ones(1, 2, 2, 14, dtype=torch.bool),
            padding_mask=torch.zeros(1, 2, 2, dtype=torch.bool),
            denormalize=False,
            reference_metadata=_reference_metadata(),
        )


def test_predictor_is_the_only_boundary_that_pads_short_candidates() -> None:
    binding = cast(
        "TrackQueryBoundModelIO",
        bind_model_io(
            _FixedTrackingModel(),
            TrackQueryModelIOAdapter(
                num_court_tokens=14,
                num_queries=2,
                presence_threshold=0.5,
            ),
        ),
    )
    predictor = BLCSTrackingPredictor(binding, torch.device("cpu"))
    court_kp = torch.zeros(1, 1, 3, 14, 2)
    court_vis = torch.ones(1, 1, 3, 14, dtype=torch.bool)
    padding_mask = torch.zeros(1, 1, 3, dtype=torch.bool)

    result = predictor.predict(
        ball_uv=torch.zeros(1, 1, 3, 1, 2),
        ball_vis=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        court_kp=court_kp,
        court_vis=court_vis,
        padding_mask=padding_mask,
        denormalize=False,
    )
    assert result.position.shape == (1, 3, 2, 3)

    with pytest.raises(ValueError, match="exceed model.num_queries"):
        predictor.predict(
            ball_uv=torch.zeros(1, 1, 3, 3, 2),
            ball_vis=torch.ones(1, 1, 3, 3, dtype=torch.bool),
            court_kp=court_kp,
            court_vis=court_vis,
            padding_mask=padding_mask,
            denormalize=False,
        )


def test_checkpoint_restoration_dispatches_to_exact_canonical_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = Path("src/tasks/blcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query",
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.backend=reference",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ],
        )
    binding = compose_blcs_track_query_model_io(config)
    checkpoint = tmp_path / "tracking.ckpt"
    checkpoint_payload: dict[str, object] = {
        "hyper_parameters": {"config": config},
        "court_coordinate_normalization": (court_coordinate_normalization_metadata()),
    }
    write_model_artifact_court_keypoint_contract(
        checkpoint_payload,
        resolve_court_keypoint_contract("physical_v1"),
    )
    write_checkpoint_track_query_reference_contract(
        checkpoint_payload,
        TrackQueryReferenceContract.physical_v1(),
    )
    torch.save(checkpoint_payload, checkpoint)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        BLCSTrackingPredictor,
        "_ensure_checkpoint",
        classmethod(lambda cls, value, *, resolver: [checkpoint]),
    )

    def compose_binding(value: object) -> TrackQueryBoundModelIO:
        observed["config"] = value
        return binding

    def load_module(
        cls: type[BLCSTrackingPredictor],
        path: Path,
        module_type: type[nn.Module],
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, torch.device]:
        del cls, module_type
        observed["path"] = path
        observed["strict"] = kwargs["strict"]
        observed["model_io"] = kwargs["model_io"]
        return SimpleNamespace(model_io=binding), torch.device("cpu")

    monkeypatch.setattr(
        "src.tasks.blcs.inference.tracking_predictor.compose_blcs_track_query_model_io",
        compose_binding,
    )
    monkeypatch.setattr(
        BLCSTrackingPredictor,
        "_load_single_lightning_module",
        classmethod(load_module),
    )

    predictor = BLCSTrackingPredictor.load_from_checkpoint(
        checkpoint,
        resolver=cast("PathResolver", object()),
        device="cpu",
    )

    assert observed == {
        "config": config,
        "path": checkpoint,
        "strict": True,
        "model_io": binding,
    }
    assert type(predictor.model) is BLCSTrackQueryModel
    assert type(predictor.model_io.adapter) is TrackQueryModelIOAdapter
    assert (
        predictor.model_io.adapter.track_query_reference_contract
        == TrackQueryReferenceContract.physical_v1()
    )
