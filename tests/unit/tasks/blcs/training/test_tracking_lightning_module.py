from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)


def test_migrate_legacy_group_embedding_checkpoint_keys() -> None:
    weight = torch.randn(3, 2)
    state_dict = {
        "model.group_encoder.proj.layers.0.weight": weight,
        "model.position_head.weight": torch.randn(3, 4),
    }

    BLCSTrackingLightningModule._migrate_legacy_group_embedding_keys(state_dict)

    assert "model.group_encoder.proj.layers.0.weight" not in state_dict
    assert state_dict["model.group_embed.proj.layers.0.weight"] is weight


def test_migrate_legacy_group_embedding_rejects_key_collisions() -> None:
    state_dict = {
        "model.group_encoder.proj.layers.0.weight": torch.randn(3, 2),
        "model.group_embed.proj.layers.0.weight": torch.randn(3, 2),
    }

    with pytest.raises(RuntimeError, match="both legacy group_encoder"):
        BLCSTrackingLightningModule._migrate_legacy_group_embedding_keys(state_dict)
