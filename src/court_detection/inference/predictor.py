"""Inference predictor for court keypoint detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.court_detection.models.court_keypoint_model import CourtKeypointModel
from src.court_detection.training.lightning_module import CourtKeypointLightningModule


class CourtKeypointPredictor:
    """Predictor for court keypoint detection.

    Provides a simple API for running inference with trained models.

    Args:
        model: CourtKeypointModel instance.
        device: Device to run inference on.
        input_size: Input image size [H, W].
    """

    def __init__(
        self,
        model: CourtKeypointModel,
        device: str | torch.device = "cpu",
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.input_size = input_size

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "CourtKeypointPredictor":
        """Load predictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Device to run inference on.

        Returns:
            CourtKeypointPredictor instance.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load Lightning module
        lightning_module = CourtKeypointLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        # Extract model and config
        model = lightning_module.model
        input_size = tuple(lightning_module.model_config.get("input_size", [256, 256]))

        return cls(model=model, device=device, input_size=input_size)

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        weights_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> "CourtKeypointPredictor":
        """Create predictor from configuration.

        Args:
            model_config: Model configuration dict.
            weights_path: Optional path to model weights.
            device: Device to run inference on.

        Returns:
            CourtKeypointPredictor instance.
        """
        model = CourtKeypointModel(
            backbone=model_config.get("backbone", {}),
            head=model_config.get("head", {}),
            input_size=model_config.get("input_size", [256, 256]),
        )

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)

        input_size = tuple(model_config.get("input_size", [256, 256]))

        return cls(model=model, device=device, input_size=input_size)

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

        # Extract keypoints (scale to original image size)
        keypoints = outputs["keypoints"][0].cpu().numpy()  # (K, 2) normalized
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

            keypoints = outputs["keypoints"][i].cpu().numpy()
            keypoints[:, 0] *= orig_w
            keypoints[:, 1] *= orig_h

            visibility = torch.sigmoid(outputs["visibility"][i]).cpu().numpy()

            results.append({
                "keypoints": keypoints,
                "visibility": visibility,
            })

        return results
