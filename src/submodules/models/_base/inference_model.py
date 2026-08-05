"""Abstract base class shared by all submodule inference models.

Every model in :mod:`src.submodules.models` follows the same contract:

- construction is cheap (no weights are touched),
- :meth:`BaseInferenceModel.load` is idempotent and loads weights onto
  :attr:`device`,
- :meth:`BaseInferenceModel.predict` takes a single request object and returns
  a single result object (auto-loading and ``torch.no_grad`` included), and
- :meth:`BaseInferenceModel.unload` releases the weights.

Downstream code can therefore hold heterogeneous models in one place and drive
them uniformly::

    model: BaseInferenceModel[RequestT, ResultT]
    result = model.predict(request)
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from src.utils.device import resolve_device

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class BaseInferenceModel(ABC, Generic[RequestT, ResultT]):
    """Lifecycle + typed ``predict`` contract for inference models."""

    def __init__(
        self,
        device: str | torch.device,
        *,
        allow_device_fallback: bool,
    ) -> None:
        if type(allow_device_fallback) is not bool:
            raise TypeError("allow_device_fallback must be a bool.")
        self._device = resolve_device(
            device,
            allow_fallback=allow_device_fallback,
        )
        self._loaded = False

    @property
    def device(self) -> torch.device:
        """Device the model weights live on once loaded."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load model weights (idempotent)."""
        if self._loaded:
            return
        self._load_impl()
        self._loaded = True

    def unload(self) -> None:
        """Release model weights (idempotent)."""
        if not self._loaded:
            return
        self._unload_impl()
        self._loaded = False
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, request: RequestT) -> ResultT:
        """Run inference on one request, loading weights on first use."""
        self.load()
        with torch.no_grad():
            return self._predict_impl(request)

    def __call__(self, request: RequestT) -> ResultT:
        return self.predict(request)

    @abstractmethod
    def _load_impl(self) -> None:
        """Load weights onto :attr:`device`."""

    @abstractmethod
    def _unload_impl(self) -> None:
        """Drop references to loaded weights."""

    @abstractmethod
    def _predict_impl(self, request: RequestT) -> ResultT:
        """Model-specific inference (called under ``torch.no_grad``)."""
