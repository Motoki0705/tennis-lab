"""BLCS prediction-bundle reference metadata tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
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


def test_counterfactual_tracking_payload_preserves_strict_pair_inputs(
    tmp_path: Path,
) -> None:
    metadata = _metadata_mock()
    prepared = SimpleNamespace(
        target_position=torch.zeros(1, 2, 1, 3),
        target_velocity=torch.zeros(1, 2, 1, 3),
        target_presence=torch.ones(1, 2, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 2, 1, dtype=torch.int64),
        target_slot_mask=torch.ones(1, 1, dtype=torch.bool),
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
    output_dir = tmp_path.resolve()
    module.set_counterfactual_prediction_dir(output_dir)
    batch = {
        "ball_uv": torch.zeros(1, 2, 2, 1, 2),
        "ball_vis": torch.ones(1, 2, 2, 1, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "clean_ball_uv": torch.zeros(1, 2, 2, 1, 2),
        "clean_ball_vis": torch.ones(1, 2, 2, 1, dtype=torch.bool),
        "candidate_gt_index": torch.zeros(1, 2, 2, 1, dtype=torch.int64),
    }
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

    payload = module.test_prediction_payload(batch, result)

    assert module._test_predictions_dir() == output_dir
    for field in (
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "target_velocity",
        "target_slot_mask",
        "clean_ball_uv",
        "clean_ball_vis",
        "candidate_gt_index",
    ):
        assert field in payload


def test_counterfactual_tracking_output_requires_absolute_path() -> None:
    module = object.__new__(BLCSTrackingLightningModule)
    with pytest.raises(ValueError, match="absolute Path"):
        module.set_counterfactual_prediction_dir(Path("relative"))
