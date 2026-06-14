"""Helpers for importing the vendored DINOv3 sources.

The DINOv3 backbone, self-supervised losses, and projection heads are loaded
from the vendored ``third_party/dinov3`` repository and remain subject to the
DINOv3 License Agreement in ``third_party/dinov3/LICENSE.md``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DINOV3_REPOSITORY = _PROJECT_ROOT / "third_party" / "dinov3"
DEFAULT_CHECKPOINT = (
    DINOV3_REPOSITORY / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def ensure_dinov3_importable() -> Path:
    """Add the vendored DINOv3 repository to ``sys.path`` (idempotent)."""
    if not DINOV3_REPOSITORY.is_dir():
        raise FileNotFoundError(
            f"Vendored DINOv3 repository not found: {DINOV3_REPOSITORY}"
        )
    repo_str = str(DINOV3_REPOSITORY)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return DINOV3_REPOSITORY


@lru_cache(maxsize=1)
def load_dinov3_losses_and_head():
    """Return ``(DINOHead, DINOLoss, iBOTPatchLoss, KoLeoLoss)`` from DINOv3."""
    ensure_dinov3_importable()
    from dinov3.layers.dino_head import DINOHead
    from dinov3.loss.dino_clstoken_loss import DINOLoss
    from dinov3.loss.ibot_patch_loss import iBOTPatchLoss
    from dinov3.loss.koleo_loss import KoLeoLoss

    return DINOHead, DINOLoss, iBOTPatchLoss, KoLeoLoss


__all__ = [
    "DINOV3_REPOSITORY",
    "DEFAULT_CHECKPOINT",
    "ensure_dinov3_importable",
    "load_dinov3_losses_and_head",
]
