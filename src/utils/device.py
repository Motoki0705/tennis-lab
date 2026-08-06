"""Strict device-resolution helpers shared by inference and training."""

from __future__ import annotations

import torch


class DeviceSelectionError(RuntimeError):
    """Raised when an explicitly requested compute device is unavailable."""


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    ``"auto"`` is the only availability-based selector. Explicit CUDA requests
    never change meaning: unavailable devices fail before model construction.

    Args:
        device: Device string (``"auto"``/``"cpu"``/``"cuda"``/...) or
            ``torch.device``.
    Returns:
        The resolved :class:`torch.device`.

    Raises:
        DeviceSelectionError: If the device specification is invalid or an
            explicitly requested CUDA device is unavailable.
    """
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DeviceSelectionError(
            f"Invalid device specification: {device!r}"
        ) from error
    if resolved.type != "cuda":
        return resolved
    if not torch.cuda.is_available():
        raise DeviceSelectionError(
            f"Explicit CUDA device {str(resolved)!r} was requested, but CUDA is unavailable."
        )
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise DeviceSelectionError(
            f"Explicit CUDA device {str(resolved)!r} was requested, but no CUDA devices are visible."
        )
    if resolved.index is not None and resolved.index >= device_count:
        raise DeviceSelectionError(
            f"CUDA device index {resolved.index} is unavailable; visible device count is {device_count}."
        )
    return resolved


def select_accelerator(gpus: int) -> tuple[str, int]:
    """Return a Lightning ``(accelerator, devices)`` pair from a GPU count.

    Zero explicitly selects CPU. A positive GPU request is strict and raises
    when CUDA or the requested number of devices is unavailable.
    """
    if type(gpus) is not int:
        raise DeviceSelectionError(
            f"GPU count must be exactly int, got {type(gpus).__name__}."
        )
    if gpus < 0:
        raise DeviceSelectionError(f"GPU count must be non-negative, got {gpus}.")
    if gpus == 0:
        return "cpu", 1
    if not torch.cuda.is_available():
        raise DeviceSelectionError(
            f"run.gpus={gpus} explicitly requests GPU training, but CUDA is unavailable."
        )
    device_count = torch.cuda.device_count()
    if gpus > device_count:
        raise DeviceSelectionError(
            f"run.gpus={gpus} exceeds the visible CUDA device count {device_count}."
        )
    return "gpu", gpus


__all__ = ["DeviceSelectionError", "resolve_device", "select_accelerator"]
