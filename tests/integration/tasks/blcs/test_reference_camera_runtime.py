"""BLCS reference-camera v2 runtime composition and persistence tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    TrackQueryReferenceContractMismatchError,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.data.tracking_datamodule import BLCSTrackingDataModule
from src.tasks.blcs.model_io import BLCSReferenceMetadata
from src.tasks.blcs.model_io.adapters import (
    TrackQueryReferenceAblationModelIOAdapter,
)
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _config(profile: str = "track_query_ablation_d_v2_selector") -> Any:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_queries=2",
                "model.num_stages=4",
                "model.dropout=0.0",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
                "model.cswa.backend=reference",
            ],
        )


def _selection() -> ReferenceViewSelection:
    court_contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="cam_0",
            camera_center_court_m=(0.0, -10.0, 3.0),
            contract=court_contract,
        ),
        build_court_view_record(
            camera_id="cam_1",
            camera_center_court_m=(0.0, 10.0, 3.0),
            contract=court_contract,
        ),
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(("cam_0", "cam_1"))
    return ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=views,
        reference_camera_id="cam_1",
    )


def _batch() -> dict[str, object]:
    selection = _selection()
    matrix = torch.tensor(
        selection.provenance.reference_from_physical,
        dtype=torch.float32,
    ).unsqueeze(0)
    metadata = BLCSReferenceMetadata(
        selections=(selection,),
        stable_camera_id_tables=(selection.stable_camera_id_table,),
        reference_view_index=torch.tensor([1], dtype=torch.int64),
        view_camera_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        reference_camera_id=torch.tensor([1], dtype=torch.int64),
        reference_from_physical=matrix,
        physical_from_reference=matrix.transpose(-1, -2),
        track_query_contract=TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode.REFERENCE
        ),
    )
    batch: dict[str, object] = {
        "ball_uv": torch.rand(1, 2, 2, 2, 2),
        "ball_vis": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "target_position": torch.zeros(1, 2, 2, 3),
        "target_velocity": torch.zeros(1, 2, 2, 3),
        "target_presence": torch.ones(1, 2, 2, dtype=torch.bool),
        "target_instance_id": torch.tensor([[[0, 1], [0, 1]]]),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
        "court_reference_provenance": (selection.provenance,),
        **metadata.to_batch_fields(),
    }
    return batch


def test_v2_composition_binds_tracking_runtime_and_reference_adapter() -> None:
    composition = compose_blcs_training(_config(), generator_config=None)

    assert isinstance(composition.datamodule, BLCSTrackingDataModule)
    assert isinstance(composition.lightning_module, BLCSTrackingLightningModule)
    assert isinstance(
        composition.lightning_module.io_adapter,
        TrackQueryReferenceAblationModelIOAdapter,
    )


def test_datamodule_passes_eval_reference_but_training_keeps_seeded_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object]] = []

    def _dataset(**kwargs: object) -> object:
        received.append(dict(kwargs))
        return object()

    monkeypatch.setattr(
        "src.tasks.blcs.data.tracking_datamodule.BLCSTrackingDataset",
        _dataset,
    )
    datamodule = BLCSTrackingDataModule(_config())

    datamodule._build_dataset(Path("dataset"), "train.txt", True)
    datamodule._build_dataset(Path("dataset"), "val.txt", False)

    assert received[0]["reference_camera_id"] is None
    assert received[1]["reference_camera_id"] == "cam_1"


def test_checkpoint_and_prediction_metadata_round_trip_and_reject_selector_mismatch() -> None:
    selector = cast(
        BLCSTrackingLightningModule,
        compose_blcs_training(_config(), generator_config=None).lightning_module,
    )
    selector_zero = cast(
        BLCSTrackingLightningModule,
        compose_blcs_training(
            _config("track_query_ablation_d_v2_selector_zero"),
            generator_config=None,
        ).lightning_module,
    )
    checkpoint: dict[str, object] = {"state_dict": {}}

    selector.on_save_checkpoint(checkpoint)
    selector.on_load_checkpoint(checkpoint)
    with pytest.raises(TrackQueryReferenceContractMismatchError):
        selector_zero.on_load_checkpoint(checkpoint)

    step = selector.compute_tracking_step(
        cast("dict[str, torch.Tensor]", _batch()),
        compute_metrics=False,
    )
    assert step.prediction.reference_metadata is not None
    assert step.prediction.reference_metadata.reference_camera_ids == ("cam_1",)
    payload = selector.test_prediction_payload(
        _batch(),
        {"prediction": step.prediction},
    )
    assert np.asarray(payload["reference_camera_id_string"]).tolist() == ["cam_1"]
    assert np.asarray(payload["reference_view_index"]).tolist() == [1]
