"""Export a LoRA-fine-tuned DINOv3 backbone for downstream tasks.

The exported checkpoint merges the LoRA adapters into the base ViT weights and
is saved in the same ``{"model": state_dict}`` layout that the shared DINOv3
loaders (``court_detection`` / ``ball_detection``) already consume, so the
tennis-adapted backbone can be dropped in via a ``checkpoint_path`` override.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.dino_ssl.training.lightning_module import DinoSSLLightningModule


def export_backbone(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Merge LoRA adapters and save a DINOv3-compatible backbone checkpoint."""
    config = OmegaConf.load(str(config_path))
    # The trained weights live in the Lightning checkpoint, so skip reloading the
    # original pretrained file here.
    config.model.load_pretrained = False

    module = DinoSSLLightningModule(config, steps_per_epoch=1)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"], strict=False)

    merged = module.network.student["backbone"].merge_and_unload()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged.state_dict()}, output)
    print(f"[dino_ssl] exported tennis-adapted DINOv3 backbone to {output}")
    return output


__all__ = ["export_backbone"]
