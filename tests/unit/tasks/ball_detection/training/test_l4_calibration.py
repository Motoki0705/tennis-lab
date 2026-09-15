"""GPU selection and batch-search failures must remain explicit."""

from __future__ import annotations

import pytest

from src.tasks.ball_detection.training import l4_calibration as calibration


def test_search_checks_the_next_power_before_claiming_maximum() -> None:
    attempted = []

    def fits(batch: int) -> bool:
        attempted.append(batch)
        return batch <= 16

    assert calibration.largest_power_of_two(fits) == 16
    assert attempted == [1, 2, 4, 8, 16, 32]


def test_no_batch_fits_does_not_silently_select_one() -> None:
    with pytest.raises(RuntimeError, match="Batch size 1"):
        calibration.largest_power_of_two(lambda batch: False)


def test_non_memory_errors_propagate() -> None:
    def fits(batch: int) -> bool:
        raise ValueError("bad annotation")

    with pytest.raises(ValueError, match="bad annotation"):
        calibration.largest_power_of_two(fits)


@pytest.mark.parametrize("name", ["NVIDIA T4", "NVIDIA A100-SXM4-40GB", "NVIDIA L4"])
def test_only_l4_is_accepted(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(
        calibration.subprocess, "check_output", lambda *args, **kwargs: name
    )
    if name == "NVIDIA L4":
        assert calibration.require_l4() == name
    else:
        with pytest.raises(RuntimeError, match="Requested NVIDIA L4"):
            calibration.require_l4()


def test_wrapped_cuda_oom_is_detected_but_unrelated_errors_are_not() -> None:
    wrapped = RuntimeError("backend compiler failed")
    wrapped.__cause__ = calibration.torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert calibration.is_cuda_oom(wrapped)
    assert not calibration.is_cuda_oom(RuntimeError("bad compiler settings"))
    assert not calibration.is_cuda_oom(MemoryError("host memory exhausted"))
