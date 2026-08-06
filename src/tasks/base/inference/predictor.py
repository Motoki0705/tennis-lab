"""Abstract base class for inference predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor

from src.utils.configuration import PathResolver, PathRole
from src.utils.device import resolve_device

PredictionT_co = TypeVar("PredictionT_co", covariant=True)


class BasePredictor(ABC, Generic[PredictionT_co]):
    """Abstract base class for inference predictors.

    All predictors implement ``predict``. Checkpoint factories remain
    task-specific because their validated options differ by model family.

    Attributes:
        model: The model used for inference.
        device: The device for inference.

    Example:
        >>> predictor = MyPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> results = predictor.predict(inputs)

    """

    model: torch.nn.Module
    device: torch.device

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> PredictionT_co:
        """Run batch inference.

        Input and decoded output types are owned by the task's selected model
        I/O adapter. Tensor fields in decoded results must be on CPU; callers
        do not perform device transfer or model-specific key decoding.

        Returns:
            The task-specific decoded prediction type ``PredictionT_co``.

        """
        ...

    @staticmethod
    def _ensure_checkpoint(
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
    ) -> list[Path]:
        """Normalize and validate checkpoint paths.

        Args:
            checkpoint_path: Path or iterable of paths to checkpoint files.

        Returns:
            List of resolved checkpoint paths.

        Raises:
            FileNotFoundError: If any checkpoint file does not exist.
            ValueError: If no checkpoints are provided.
        """
        paths = checkpoint_path
        if isinstance(paths, (str, Path)):
            paths = [paths]
        checkpoints = [
            resolver.validate(PathRole.CHECKPOINT, candidate)
            if (candidate := Path(path)).is_absolute()
            else resolver.resolve(PathRole.CHECKPOINT, candidate)
            for path in paths
        ]
        if not checkpoints:
            raise ValueError("checkpoint_path must be non-empty")
        for path in checkpoints:
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        return checkpoints

    @classmethod
    def _load_single_lightning_module(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        lightning_module_cls: Any,
        *,
        resolver: PathResolver,
        device: str | torch.device,
        **kwargs: Any,
    ) -> tuple[Any, torch.device]:
        """Load a single Lightning checkpoint and return the Lightning module.

        Enforces exactly one checkpoint, resolves the device, and loads the
        Lightning module. Use this variant when the caller needs access to the
        Lightning module itself (e.g. its ``config``); ``kwargs`` are forwarded
        to ``load_from_checkpoint`` verbatim, so callers control ``strict`` /
        ``weights_only`` explicitly.

        Args:
            checkpoint_path: Path or iterable of paths to checkpoint files.
            lightning_module_cls: LightningModule class with
                ``load_from_checkpoint``.
            device: Inference device.
            **kwargs: Forwarded to ``load_from_checkpoint`` unchanged.

        Returns:
            Tuple of (Lightning module, resolved ``torch.device``).

        Raises:
            ValueError: If not exactly one checkpoint is provided.
        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        resolved_device = resolve_device(device)
        lightning_module = lightning_module_cls.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
            **kwargs,
        )
        return lightning_module, resolved_device

    def _denormalize_coords(self, coords: Tensor, scale_xyz: Iterable[float]) -> Tensor:
        """Scale normalized coordinates to physical units."""
        return coords * torch.tensor(
            list(scale_xyz),
            device=coords.device,
            dtype=coords.dtype,
        )

    @staticmethod
    def _to_device(
        device: torch.device, *tensors: Tensor | None
    ) -> tuple[Tensor | None, ...]:
        """Move tensors to the requested device, preserving None entries.

        Args:
            device: Target device.
            *tensors: Tensors or None values.

        Returns:
            Tuple with tensors moved to device (None preserved).
        """
        moved: list[Tensor | None] = []
        for tensor in tensors:
            moved.append(tensor.to(device) if tensor is not None else None)
        return tuple(moved)


__all__ = ["BasePredictor", "PredictionT_co"]
