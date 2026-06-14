"""One-shot generator for refactor characterization goldens.

Run with the current (pre-refactor) implementations to capture reference
outputs, then commit the resulting ``goldens/*.pt`` files. The equivalence
tests compare the new shared implementations against these frozen goldens, so
the tests survive deletion of the old per-task helpers in Phase 2.

Usage:
    .venv/bin/python -m tests.refactor._generate_goldens
"""

from __future__ import annotations

from pathlib import Path

import torch

GOLDEN_DIR = Path(__file__).parent / "goldens"


def _focal_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    targets = (torch.rand(2, 3, 8, 8) > 0.7).float()
    return logits, targets


def _mask_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "values_bt": torch.randn(4, 6),
        "values_btj3": torch.randn(4, 6, 5, 3),
        "mask_b": (torch.rand(4) > 0.3).float(),
        "mask_bt": (torch.rand(4, 6) > 0.3).float(),
        "mask_bnt": (torch.rand(4, 2, 6) > 0.3).float(),
        "mask_bntj": (torch.rand(4, 2, 6, 5) > 0.3).float(),
    }


def generate() -> None:
    from src.tasks.ball_detection.training.losses import BallDetectionFocalLoss
    from src.tasks.court_detection.training.losses import (
        FocalBCEWithLogitsLoss as CourtFocal,
    )
    from src.tasks.blcs.training.losses import _masked_mean as blcs_masked_mean
    from src.tasks.plcs.training.losses import _masked_mean as plcs_masked_mean
    from src.tasks.plcs.training.losses import _to_frame_mask
    from src.tasks.plcs.training.metrics import _valid_from_human_mask

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    logits, targets = _focal_inputs()
    ball = BallDetectionFocalLoss({"gamma": 2.0})(logits, targets)
    court = CourtFocal(gamma=2.0)(logits, targets)
    torch.save(
        {"logits": logits, "targets": targets, "ball": ball, "court": court},
        GOLDEN_DIR / "focal_loss.pt",
    )

    mi = _mask_inputs()
    # PLCS masked_mean: binarize + clamp_min(1.0), equal-rank values.
    plcs_mm = plcs_masked_mean(mi["values_bt"], mi["mask_bt"])
    # BLCS masked_mean: raw float mask, broadcast, +1e-8 denom.
    blcs_mm = blcs_masked_mean(mi["values_btj3"], mi["mask_bt"])
    # Frame-mask normalization (flatten=False) and metrics variant (flatten=True).
    norm = {
        f"to_frame_{k}": _to_frame_mask(mi[k])
        for k in ("mask_b", "mask_bt", "mask_bnt", "mask_bntj")
    }
    norm_flat = {
        f"valid_{k}": _valid_from_human_mask(mi[k])
        for k in ("mask_b", "mask_bt", "mask_bnt", "mask_bntj")
    }
    torch.save(
        {**mi, "plcs_mm": plcs_mm, "blcs_mm": blcs_mm, **norm, **norm_flat},
        GOLDEN_DIR / "masks.pt",
    )
    print(f"Goldens written to {GOLDEN_DIR}")


if __name__ == "__main__":
    generate()
