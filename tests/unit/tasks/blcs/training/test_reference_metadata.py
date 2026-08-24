"""BLCS prediction-bundle reference metadata tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import torch

from src.tasks.blcs.model_io import (
    BLCSReferenceMetadata,
    BLCSTrackQueryPrediction,
    BLCSTrajectoryPrediction,
)
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)


def _metadata_mock() -> BLCSReferenceMetadata:
    metadata = Mock(spec=BLCSReferenceMetadata)
    metadata.prediction_payload.return_value = {
        "reference_view_index": torch.tensor([1], dtype=torch.int64),
        "reference_camera_id_string": np.asarray(["camera_1"]),
        "target_frame_contract": np.asarray(
            ["reference_camera_court_rzpi_v1"]
        ),
    }
    return cast("BLCSReferenceMetadata", metadata)


def test_standard_test_payload_includes_reference_prediction_contract() -> None:
    metadata = _metadata_mock()
    module = object.__new__(BLCSLightningModule)
    object.__setattr__(
        module,
        "io_adapter",
        SimpleNamespace(
            build_training_batch=lambda batch: SimpleNamespace(
                position=torch.zeros(1, 2, 3)
            )
        ),
    )
    object.__setattr__(
        module,
        "config",
        SimpleNamespace(data=SimpleNamespace(num_views_range=(1, 2))),
    )
    result = {
        "outputs": BLCSTrajectoryPrediction(
            position=torch.zeros(1, 2, 3),
            velocity=None,
            reference_metadata=metadata,
        ),
        "mask": torch.ones(1, 2, dtype=torch.bool),
    }

    payload = module.test_prediction_payload({}, result)

    assert payload["reference_view_index"].tolist() == [1]
    assert payload["reference_camera_id_string"].tolist() == ["camera_1"]
    assert payload["target_frame_contract"].tolist() == [
        "reference_camera_court_rzpi_v1"
    ]


def test_tracking_test_payload_includes_reference_prediction_contract() -> None:
    metadata = _metadata_mock()
    prepared = SimpleNamespace(
        target_position=torch.zeros(1, 2, 1, 3),
        target_presence=torch.ones(1, 2, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 2, 1, dtype=torch.int64),
        frame_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    module = object.__new__(BLCSTrackingLightningModule)
    object.__setattr__(
        module,
        "io_adapter",
        SimpleNamespace(build_training_batch=lambda batch: prepared),
    )
    object.__setattr__(
        module,
        "config",
        SimpleNamespace(data=SimpleNamespace(num_views_range=(1, 2))),
    )
    logits = torch.zeros(1, 2, 1)
    result = {
        "prediction": BLCSTrackQueryPrediction(
            position=torch.zeros(1, 2, 1, 3),
            presence_logits=logits,
            presence_probability=logits.sigmoid(),
            presence=torch.ones(1, 2, 1, dtype=torch.bool),
            reference_metadata=metadata,
        )
    }

    payload = module.test_prediction_payload({}, result)

    assert payload["reference_view_index"].tolist() == [1]
    assert payload["reference_camera_id_string"].tolist() == ["camera_1"]
    assert payload["target_frame_contract"].tolist() == [
        "reference_camera_court_rzpi_v1"
    ]
