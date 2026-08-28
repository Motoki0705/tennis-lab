"""Frozen-provenance-aware Lightning batch transfer tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch import Tensor

from src.tasks.base.training.batch_transfer import (
    move_batch_to_device_preserving_frozen_metadata,
)


@dataclass(frozen=True, slots=True)
class _FrozenReferenceMetadata:
    camera_ids: tuple[str, ...]
    diagnostic_tensor: Tensor


def test_batch_transfer_moves_tensors_without_reconstructing_frozen_metadata() -> None:
    metadata = _FrozenReferenceMetadata(
        camera_ids=("camera_0", "camera_1"),
        diagnostic_tensor=torch.tensor([7]),
    )
    batch = {
        "model_input": torch.ones(2),
        "nested_loss_inputs": [torch.zeros(1), {"target": torch.ones(1)}],
        "reference_metadata": metadata,
    }
    device = torch.device("meta")

    with pytest.raises(ValueError, match="frozen dataclass"):
        move_data_to_device(batch, device)

    moved = move_batch_to_device_preserving_frozen_metadata(batch, device)

    assert moved["model_input"].device == device
    assert moved["nested_loss_inputs"][0].device == device
    assert moved["nested_loss_inputs"][1]["target"].device == device
    moved_metadata = moved["reference_metadata"]
    assert moved_metadata.camera_ids == metadata.camera_ids
    assert moved_metadata.diagnostic_tensor.device.type == "cpu"
