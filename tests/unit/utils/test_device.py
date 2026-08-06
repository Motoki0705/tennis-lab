"""Tests for strict shared device selection."""

from __future__ import annotations

import pytest
import torch

from src.utils.device import (
    DeviceSelectionError,
    resolve_device,
    select_accelerator,
)

pytestmark = pytest.mark.unit


def test_auto_is_the_only_availability_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(DeviceSelectionError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_auto_selects_cuda_only_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")


def test_explicit_cuda_rejects_inconsistent_zero_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(DeviceSelectionError, match="no CUDA devices are visible"):
        resolve_device(torch.device("cuda"))


def test_explicit_cuda_index_must_be_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(DeviceSelectionError, match="index 1 is unavailable"):
        resolve_device("cuda:1")


def test_select_accelerator_rejects_unavailable_positive_gpu_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_accelerator(0) == ("cpu", 1)
    with pytest.raises(DeviceSelectionError, match="explicitly requests GPU"):
        select_accelerator(1)


def test_select_accelerator_rejects_excess_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(DeviceSelectionError, match="visible CUDA device count 1"):
        select_accelerator(2)


@pytest.mark.parametrize("gpus", [-1, True, 1.0])
def test_select_accelerator_rejects_invalid_gpu_counts(gpus: object) -> None:
    with pytest.raises(DeviceSelectionError):
        select_accelerator(gpus)  # type: ignore[arg-type]
