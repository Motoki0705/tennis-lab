"""Extract DINOv3 backbone weights from a WASB Lightning checkpoint.

Example:
    uv run python -m src.wasb.scripts.tools.extract_dinov3_backbone \
      checkpoint_path=outputs/wasb/ball_detection/dinov3_heatmap/logs/version_0/checkpoints/last.ckpt \
      output_path=outputs/wasb/ball_detection/dinov3_heatmap/dinov3_backbone.pth

Hydra parameters:
    - checkpoint_path: Path to a Lightning checkpoint (.ckpt or .pth.tar).
    - output_path: Destination for the extracted backbone state dict.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))

def _extract_backbone_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("model.backbone.", "backbone.")
    for prefix in prefixes:
        keys = [k for k in state_dict if k.startswith(prefix)]
        if not keys:
            continue
        return {k[len(prefix) :]: state_dict[k] for k in keys}
    raise KeyError("No backbone parameters found in the checkpoint state_dict.")


@hydra_main(
    config_path="../../configs",
    config_name="extract_dinov3_backbone",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint_path)))
    output_path = Path(to_absolute_path(str(cfg.output_path)))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    backbone_state = _extract_backbone_state(state_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_state, output_path)

    print(f"Saved backbone weights to {output_path}")


if __name__ == "__main__":
    cast(Callable[[], None], main)()
