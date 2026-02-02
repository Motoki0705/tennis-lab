"""Inference predictor for court keypoint detection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.base.inference.predictor import BasePredictor
from src.court_detection.models.court_keypoint_model import CourtKeypointModel
from src.court_detection.training.lightning_module import CourtKeypointLightningModule


class CourtKeypointPredictor(BasePredictor):
    """Predictor for court keypoint detection.

    Provides a simple API for running inference with trained models.

    Attributes:
        model: CourtKeypointModel instance.
        device: Device to run inference on.
        input_size: Input image size [H, W].
    """

    def __init__(
        self,
        model: CourtKeypointModel,
        device: torch.device,
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.model = model
        self.device = device
        self.input_size = input_size

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Load predictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file(s).
            device: Device to run inference on.
            **kwargs: Additional arguments (unused).

        Returns:
            CourtKeypointPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path)
        resolved_device = cls._resolve_device(device)

        # Load Lightning module from first checkpoint
        lightning_module = CourtKeypointLightningModule.load_from_checkpoint(
            checkpoints[0],
            map_location=resolved_device,
        )

        # Extract model and config
        model = lightning_module.model
        model_config = lightning_module.config.get("model", {})
        input_size = tuple(model_config.get("input_size", [256, 256]))

        return cls(model=model, device=resolved_device, input_size=input_size)

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        weights_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> Self:
        """Create predictor from configuration.

        Args:
            model_config: Model configuration dict.
            weights_path: Optional path to model weights.
            device: Device to run inference on.

        Returns:
            CourtKeypointPredictor instance.
        """
        resolved_device = cls._resolve_device(device)
        model = CourtKeypointModel(model_config)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=resolved_device)
            model.load_state_dict(state_dict)

        input_size = tuple(model_config.get("input_size", [256, 256]))

        return cls(model=model, device=resolved_device, input_size=input_size)

    def preprocess(self, image: np.ndarray | Image.Image) -> Tensor:
        """Preprocess image for inference.

        Args:
            image: Input image (numpy array HWC or PIL Image).

        Returns:
            Preprocessed tensor of shape (1, 3, H, W).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize
        image = image.resize((self.input_size[1], self.input_size[0]))

        # Convert to tensor
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

        return image_tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | Image.Image | Tensor,
        return_heatmaps: bool = False,
    ) -> dict[str, np.ndarray | Tensor]:
        """Run inference on a single image.

        Args:
            image: Input image.
            return_heatmaps: Whether to return heatmaps.

        Returns:
            Dictionary with:
                - 'keypoints': Keypoint coordinates in pixel space (K, 2)
                - 'visibility': Visibility probabilities (K,)
                - 'heatmaps': Optional heatmaps (K, H, W)
        """
        # Preprocess if needed
        if isinstance(image, (np.ndarray, Image.Image)):
            if isinstance(image, np.ndarray):
                orig_h, orig_w = image.shape[:2]
            else:
                orig_w, orig_h = image.size
            image_tensor = self.preprocess(image)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            orig_h, orig_w = self.input_size

        # Run inference
        outputs = self.model(image_tensor)

        keypoints = self._heatmaps_to_coords(outputs["heatmaps"])[0].cpu().numpy()
        keypoints[:, 0] *= orig_w
        keypoints[:, 1] *= orig_h

        # Extract visibility
        visibility = torch.sigmoid(outputs["visibility"][0]).cpu().numpy()  # (K,)

        result = {
            "keypoints": keypoints,
            "visibility": visibility,
        }

        if return_heatmaps:
            result["heatmaps"] = outputs["heatmaps"][0].cpu().numpy()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[np.ndarray | Image.Image] | Tensor,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on a batch of images.

        Args:
            images: List of images or batched tensor.

        Returns:
            List of prediction dictionaries.
        """
        if isinstance(images, Tensor):
            batch_tensor = images.to(self.device)
            orig_sizes = [(self.input_size[0], self.input_size[1])] * len(images)
        else:
            batch_tensors = []
            orig_sizes = []
            for img in images:
                if isinstance(img, np.ndarray):
                    orig_sizes.append((img.shape[0], img.shape[1]))
                else:
                    orig_sizes.append((img.size[1], img.size[0]))
                batch_tensors.append(self.preprocess(img))
            batch_tensor = torch.cat(batch_tensors, dim=0)

        # Run inference
        outputs = self.model(batch_tensor)

        # Extract results
        results = []
        for i in range(len(batch_tensor)):
            orig_h, orig_w = orig_sizes[i]

            keypoints = self._heatmaps_to_coords(outputs["heatmaps"][i : i + 1])[0].cpu().numpy()
            keypoints[:, 0] *= orig_w
            keypoints[:, 1] *= orig_h

            visibility = torch.sigmoid(outputs["visibility"][i]).cpu().numpy()

            results.append({
                "keypoints": keypoints,
                "visibility": visibility,
            })

        return results

    @staticmethod
    def _heatmaps_to_coords(heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to keypoint coordinates using soft-argmax.

        Args:
            heatmaps: Heatmaps of shape (B, K, H, W).

        Returns:
            Coordinates of shape (B, K, 2) in normalized [0, 1] range.
        """
        bsz, num_kp, height, width = heatmaps.shape
        device = heatmaps.device

        heatmaps_flat = heatmaps.view(bsz, num_kp, -1)
        probs = F.softmax(heatmaps_flat, dim=-1)

        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        x = (probs * xx_flat.view(1, 1, -1)).sum(dim=-1)
        y = (probs * yy_flat.view(1, 1, -1)).sum(dim=-1)

        return torch.stack([x, y], dim=-1)
