"""Abstract base class for inference predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.utils.device import resolve_device


class BasePredictor(ABC):
    """Abstract base class for inference predictors.

    All predictors must implement two methods:
    - load_from_checkpoint: Create instance from checkpoint file.
    - predict: Run batch inference.

    Attributes:
        model: The model used for inference.
        device: The device for inference.

    Example:
        >>> predictor = MyPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> results = predictor.predict(inputs)

    """

    model: torch.nn.Module
    device: torch.device

    @classmethod
    @abstractmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Create a predictor instance from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device for inference ("cpu" or "cuda").
            **kwargs: Implementation-specific additional arguments.

        Returns:
            Initialized predictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        ...

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Run batch inference.

        Input/output formats vary by implementation.
        See subclass documentation for details.

        Returns:
            Dictionary of inference results. All predictors must follow this contract:
            
            **Return Type Contract:**
            - All values MUST be `torch.Tensor` (not numpy arrays)
            - All tensors MUST be on CPU (callers should not handle device transfers)
            - Batch dimension MUST be preserved in outputs
            
            **Key Naming Contract:**
            - Use snake_case for all keys
            - Use descriptive names matching the semantic meaning
            - Suffix denormalized/physical units (e.g., `_meters`, `_radians`)
            - Common keys across tasks:
              - `position`: 3D position in normalized or world coordinates
              - `position_meters`: 3D position in meters (denormalized)
              - `velocity`: Velocity vector
              - `rotation`: Rotation representation (e.g., sin/cos, quaternion)
              - `yaw_radians`: Yaw angle in radians
              - `keypoints`: 2D/3D keypoint coordinates
              - `visibility`: Visibility flags/probabilities
              - `heatmaps`: Spatial probability maps
            
            Implementation-specific keys are allowed but should follow the naming convention.

        """
        ...

    @staticmethod
    def _resolve_device(
        device: str | torch.device,
        *,
        allow_fallback: bool = True,
    ) -> torch.device:
        """Resolve device string to torch.device with optional CUDA fallback.

        Args:
            device: Device string or torch.device.
            allow_fallback: If True, fall back to CPU when CUDA is unavailable.

        Returns:
            Resolved torch.device.

        Raises:
            RuntimeError: If CUDA is requested but unavailable and fallback is disabled.
        """
        return resolve_device(device, allow_fallback=allow_fallback)

    @staticmethod
    def _ensure_checkpoint(
        checkpoint_path: str | Path | Iterable[str | Path],
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
        checkpoints: list[Path] = [Path(p) for p in paths]
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
        device: str | torch.device = "cpu",
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
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        resolved_device = cls._resolve_device(device)
        lightning_module = lightning_module_cls.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
            **kwargs,
        )
        return lightning_module, resolved_device

    @classmethod
    def _load_single_lightning_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        lightning_module_cls: Any,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> tuple[nn.Module, torch.device]:
        """Load a single Lightning checkpoint and return its inner model.

        Same as :meth:`_load_single_lightning_module` but returns the inner
        ``.model`` and applies the strict/weights_only defaults shared across
        task predictors.

        Args:
            checkpoint_path: Path or iterable of paths to checkpoint files.
            lightning_module_cls: LightningModule class with
                ``load_from_checkpoint``.
            device: Inference device.
            **kwargs: Forwarded to ``load_from_checkpoint``. ``strict`` and
                ``weights_only`` are popped (defaulting to False).

        Returns:
            Tuple of (inner ``nn.Module``, resolved ``torch.device``).

        Raises:
            ValueError: If not exactly one checkpoint is provided.
        """
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoint_path,
            lightning_module_cls,
            device,
            strict=bool(kwargs.pop("strict", False)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        return lightning_module.model, resolved_device

    def _denormalize_coords(self, coords: Tensor, scale_xyz: Iterable[float]) -> Tensor:
        """Scale normalized coordinates to physical units."""
        return coords * torch.tensor(
            list(scale_xyz),
            device=coords.device,
            dtype=coords.dtype,
        )

    @staticmethod
    def _to_device(device: torch.device, *tensors: Tensor | None) -> tuple[Tensor | None, ...]:
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
