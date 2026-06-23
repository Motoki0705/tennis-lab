"""Device-resolution helpers shared by inference, scripts and training.

Previously each entry point re-derived ``"cuda" if torch.cuda.is_available()
else "cpu"`` (sometimes returning a ``str``, sometimes a ``torch.device``,
sometimes handling an ``"auto"`` sentinel, sometimes not). Use
:func:`resolve_device` for a single, consistent resolution.
"""

from __future__ import annotations

import torch


def resolve_device(
    device: str | torch.device,
    *,
    allow_fallback: bool = True,
) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    - The ``"auto"`` sentinel selects CUDA when available, otherwise CPU.
    - A requested CUDA device falls back to CPU when CUDA is unavailable, unless
      ``allow_fallback`` is ``False`` (in which case ``RuntimeError`` is raised).

    Args:
        device: Device string (``"auto"``/``"cpu"``/``"cuda"``/...) or
            ``torch.device``.
        allow_fallback: Fall back to CPU when CUDA is requested but unavailable.

    Returns:
        The resolved :class:`torch.device`.

    Raises:
        RuntimeError: If CUDA is requested, unavailable, and fallback is off.
    """
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            return torch.device("cpu")
        raise RuntimeError("CUDA is not available")
    return resolved


def select_accelerator(gpus: int) -> tuple[str, int]:
    """Return a Lightning ``(accelerator, devices)`` pair from a GPU count.

    Mirrors the historical ``select_devices`` behaviour: use ``("gpu", gpus)``
    when ``gpus > 0`` and CUDA is available, otherwise ``("cpu", 1)``.
    """
    if int(gpus) > 0 and torch.cuda.is_available():
        return "gpu", int(gpus)
    return "cpu", 1


__all__ = ["resolve_device", "select_accelerator"]
