"""Unsupported precision and invalid resource settings never fall back silently."""

import pytest
import torch

from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime


@pytest.mark.parametrize("options", [dict(precision="fp16"), dict(num_workers=-1),
                                    dict(prefetch_factor=0), dict(cpu_threads=0)])
def test_invalid_runtime_controls_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        CoordinateRuntime(**options)


def test_cpu_does_not_silently_replace_cuda_precision_or_pinned_inputs() -> None:
    with pytest.raises(ValueError, match="no fallback"):
        CoordinateRuntime(precision="bf16").configure(torch.device("cpu"))
    with pytest.raises(ValueError, match="Pinned"):
        CoordinateRuntime(pin_memory=True).configure(torch.device("cpu"))
