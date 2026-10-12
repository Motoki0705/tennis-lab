import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.training.runtime.diagnostics import (
    batch_identity,
    record_failure,
)


def test_failure_receipt_uses_only_cpu_metadata_and_keeps_original_exception(tmp_path: Path) -> None:
    metadata = batch_identity(dict(clip_id=["clip-a"], start=[10], frame_step=[4], frame_indices=torch.tensor([[10, 14, 18]])))
    context = dict(epoch=0, completed_updates=5800, attempted_update=5801, surface_phase="gradient_norm", batch=metadata)
    error = RuntimeError("CUDA unknown error")
    with pytest.raises(RuntimeError) as caught, patch("torch.cuda.synchronize", side_effect=AssertionError("broken CUDA context")), \
            record_failure(tmp_path, context):
        raise error
    assert caught.value is error
    receipt = json.loads((tmp_path / "failure.json").read_text())
    assert receipt["batch"]["frame_indices"] == [[10, 14, 18]]
    assert receipt["completed_updates"] == 5800
    assert "not proof" in receipt["caveat"]
    assert "CUDA unknown error" in receipt["traceback"]


def test_metadata_rejects_non_cpu_tensor_without_transferring_it() -> None:
    with pytest.raises(ValueError, match="CPU"):
        batch_identity(dict(clip_id=[], start=[], frame_step=[], frame_indices=torch.empty(1, device="meta")))


def test_receipt_io_failure_does_not_hide_the_training_error(tmp_path: Path) -> None:
    error = RuntimeError("training failure")
    with patch.object(Path, "write_text", side_effect=OSError("disk full")), pytest.raises(RuntimeError) as caught, \
            record_failure(tmp_path, {}):
        raise error
    assert caught.value is error
    assert "disk full" in error.__notes__[0]
