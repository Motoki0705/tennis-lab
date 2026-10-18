"""One float precision contract for registration, training and evaluation."""

from __future__ import annotations

import torch

from src.tasks.base.configuration import BaseTrainingConfig


def precision_contract(training: BaseTrainingConfig) -> dict[str, str | bool]:
    """Require the matmul precision and TF32 flag to describe the same runtime."""
    if training.allow_tf32 != (training.matmul_precision != "highest"):
        raise ValueError(
            "player matmul_precision and allow_tf32 must agree: "
            "highest requires false; high and medium require true"
        )
    return {
        "matmul_precision": training.matmul_precision,
        "cuda_matmul_allow_tf32": training.allow_tf32,
        "cudnn_allow_tf32": training.allow_tf32,
    }


def apply_precision(training: BaseTrainingConfig) -> None:
    expected = precision_contract(training)
    # This setter also configures CUDA matmul TF32. Setting allow_tf32 again
    # overwrites the precision (including medium) on supported PyTorch versions.
    torch.set_float32_matmul_precision(training.matmul_precision)
    torch.backends.cudnn.allow_tf32 = training.allow_tf32
    actual = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if actual != expected:
        raise RuntimeError(f"player float runtime differs from its precision contract: {actual}")
