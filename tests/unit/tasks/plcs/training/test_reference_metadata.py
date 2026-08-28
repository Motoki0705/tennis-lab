"""PLCS training prediction metadata persistence tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import torch

from src.tasks.base.model_io import ModelCall
from src.tasks.plcs.model_io import (
    PLCSDecodedPrediction,
    PLCSPreparedBatch,
    PLCSReferenceMetadata,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)


def _metadata_mock() -> PLCSReferenceMetadata:
    metadata = Mock(spec=PLCSReferenceMetadata)
    metadata.prediction_payload.return_value = {
        "reference_view_index": torch.tensor([1], dtype=torch.int64),
        "reference_camera_id_string": np.asarray(["camera_1"]),
        "target_frame_contract": np.asarray(
            ["reference_camera_court_rzpi_v1"]
        ),
    }
    return cast("PLCSReferenceMetadata", metadata)


def test_standard_test_payload_includes_reference_prediction_contract() -> None:
    metadata = _metadata_mock()
    prepared = PLCSPreparedBatch(
        call=ModelCall(),
        target_position=torch.zeros(1, 2, 3),
        target_rotation=torch.zeros(1, 2, 2),
        reference_metadata=metadata,
    )
    result = {
        "outputs": PLCSDecodedPrediction(
            position=torch.zeros(1, 2, 3),
            rotation=torch.zeros(1, 2, 2),
            reference_metadata=metadata,
        ),
        "prepared": prepared,
        "gan_padding_mask": torch.zeros(1, 2, dtype=torch.bool),
    }

    payload = PLCSLightningModule.test_prediction_payload(
        cast(
            "PLCSLightningModule",
            SimpleNamespace(
                plcs_runtime=SimpleNamespace(
                    data=SimpleNamespace(values={"num_views_range": (1, 2)})
                )
            ),
        ),
        {},
        result,
    )

    assert np.asarray(payload["reference_view_index"]).tolist() == [1]
    assert np.asarray(payload["reference_camera_id_string"]).tolist() == ["camera_1"]
    assert np.asarray(payload["target_frame_contract"]).tolist() == [
        "reference_camera_court_rzpi_v1"
    ]


def test_tracking_test_payload_includes_reference_prediction_contract(
    monkeypatch,
) -> None:
    metadata = _metadata_mock()
    monkeypatch.setattr(
        "src.tasks.plcs.training.tracking_lightning_module."
        "plcs_reference_metadata_from_batch",
        lambda batch: metadata,
    )
    batch = {
        "target_position": torch.zeros(1, 2, 1, 3),
        "target_rotation": torch.zeros(1, 2, 1, 2),
        "target_presence": torch.ones(1, 2, 1, dtype=torch.bool),
        "target_instance_id": torch.ones(1, 2, 1, dtype=torch.int64),
        "padding_mask": torch.zeros(1, 1, 2, dtype=torch.bool),
    }
    result = {
        "position": torch.zeros(1, 2, 1, 3),
        "rotation": torch.zeros(1, 2, 1, 2),
        "presence_logits": torch.zeros(1, 2, 1),
    }
    module = object.__new__(PLCSTrackingLightningModule)
    object.__setattr__(
        module,
        "plcs_runtime",
        SimpleNamespace(
            data=SimpleNamespace(values={"num_views_range": (1, 2)})
        ),
    )

    payload = module.test_prediction_payload(batch, result)

    assert np.asarray(payload["reference_view_index"]).tolist() == [1]
    assert np.asarray(payload["reference_camera_id_string"]).tolist() == ["camera_1"]
