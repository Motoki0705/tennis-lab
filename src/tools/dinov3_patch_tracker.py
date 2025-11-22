"""DINOv3-based single-object tracking and segmentation using patch tokens.

This module provides a minimal utility around the local DINOv3 backbone
(`src.models.utils.load_dinov3`) to:

- extract patch-level features from RGB frames
- initialize a template from a user-specified bounding box
- compute cosine-similarity maps for subsequent frames
- derive a binary segmentation mask and a tracking bounding box

The intended front-end is an interactive OpenCV-based CLI that lets the user
select an ROI in the first frame and then runs this tracker over a video.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from src.models.utils.load_dinov3 import load_dinov3


@dataclass
class TrackerConfig:
    """Configuration for ``Dinov3PatchTracker``.

    Attributes:
        arch: DINOv3 architecture name passed to ``load_dinov3``.
        weights_path: Optional path to pretrained weights.
        img_size: Input resolution for DINOv3 (typically 224).
        threshold: Cosine-similarity threshold for foreground mask.
        template_update_alpha: EMA factor for template updates per frame.
        device: Torch device string (e.g. "cuda" or "cpu").

    """

    arch: str = "dinov3_vits16"
    weights_path: str | None = (
        "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    img_size: int = 224
    threshold: float = 0.6
    template_update_alpha: float = 0.0
    device: str = "cuda"


class Dinov3PatchTracker:
    """Patch-token-based tracker and segmenter using a DINOv3 ViT backbone.

    Usage (high-level):

    - Call :meth:`set_template` once with a reference frame and user-specified
      bounding box to initialize the template embedding.
    - Then call :meth:`track` for each subsequent frame to obtain a binary mask
      and a tracked bounding box in the original frame resolution.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model: nn.Module = load_dinov3(
            arch=cfg.arch,
            weights_path=cfg.weights_path,
        ).to(self.device)
        self.model.eval()

        # Template embedding in feature space (L2-normalized), shape [C].
        self._template_vec: torch.Tensor | None = None

        # Cache for feature-map spatial size (H_p, W_p) once observed.
        self._feat_hw: tuple[int, int] | None = None

        # ImageNet normalization used by DINOv3.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_template(
        self,
        frame_bgr: np.ndarray,
        bbox_xywh: tuple[float, float, float, float],
    ) -> None:
        """Initialize the template from a reference frame and bounding box.

        Args:
            frame_bgr (np.ndarray): ``H x W x 3`` uint8 BGR image from OpenCV.
            bbox_xywh (tuple[float, float, float, float]): Bounding box
                ``(x, y, w, h)`` in *original* pixel coordinates of
                ``frame_bgr``.

        Returns:
            None: The internal template embedding is updated in-place.

        Raises:
            ValueError: If the bounding box dimensions are invalid.
            RuntimeError: If the bounding box does not cover any patch tokens.

        """
        x, y, w, h = bbox_xywh
        if w <= 1 or h <= 1:
            msg = f"Invalid template bbox dimensions: (x={x}, y={y}, w={w}, h={h})"
            raise ValueError(msg)

        frame_proc, scale_x, scale_y = self._preprocess_frame(frame_bgr)
        feats = self._extract_patch_features(frame_proc)
        # feats: [1, C, H_p, W_p]
        _, c, h_p, w_p = feats.shape
        self._feat_hw = (h_p, w_p)

        # Map bbox into the resized (img_size x img_size) coordinate system.
        x_resized0 = x * scale_x
        y_resized0 = y * scale_y
        x_resized1 = (x + w) * scale_x
        y_resized1 = (y + h) * scale_y

        # Convert to inclusive patch-index range.
        j0 = int(np.floor(x_resized0 * w_p / float(self.cfg.img_size)))
        j1 = int(np.ceil(x_resized1 * w_p / float(self.cfg.img_size))) - 1
        i0 = int(np.floor(y_resized0 * h_p / float(self.cfg.img_size)))
        i1 = int(np.ceil(y_resized1 * h_p / float(self.cfg.img_size))) - 1

        j0 = max(0, min(w_p - 1, j0))
        j1 = max(0, min(w_p - 1, j1))
        i0 = max(0, min(h_p - 1, i0))
        i1 = max(0, min(h_p - 1, i1))
        if j1 < j0 or i1 < i0:
            msg = "Template bbox does not overlap any patch tokens."
            raise RuntimeError(msg)

        patch_tokens = feats[0, :, i0 : i1 + 1, j0 : j1 + 1].reshape(c, -1)
        template_vec = patch_tokens.mean(dim=1)
        template_vec = torch.nn.functional.normalize(template_vec, dim=0)
        self._template_vec = template_vec

    def track(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray]:
        """Run tracking and segmentation for a single frame.

        Args:
            frame_bgr (np.ndarray): ``H x W x 3`` uint8 BGR image.

        Returns:
            tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray]:
                ``mask`` is a binary ``H x W`` uint8 mask (0 or 255) in the
                original resolution, ``bbox`` is the tracked bounding box as
                ``(x, y, w, h)`` in original coordinates or ``None`` if no
                foreground is found, and ``sim_map`` is a cosine-similarity
                map in patch space ``H_p x W_p`` with float32 values in
                ``[-1, 1]``.

        Raises:
            RuntimeError: If the template has not been initialized.

        """
        if self._template_vec is None:
            raise RuntimeError(
                "Template is not initialized. Call set_template() first."
            )

        frame_proc, scale_x, scale_y = self._preprocess_frame(frame_bgr)
        feats = self._extract_patch_features(frame_proc)
        _, c, h_p, w_p = feats.shape
        self._feat_hw = (h_p, w_p)

        # Compute cosine similarity per patch.
        feat_flat = feats.reshape(c, h_p * w_p)
        feat_flat = torch.nn.functional.normalize(feat_flat, dim=0)
        sim = torch.matmul(self._template_vec.to(feat_flat.device), feat_flat)
        sim_map = sim.reshape(h_p, w_p).detach().cpu().numpy().astype(np.float32)

        # Threshold in patch space.
        mask_patch = (sim_map >= float(self.cfg.threshold)).astype(np.uint8)

        # Upsample to resized image size, then back to original size.
        mask_resized = cv2.resize(
            mask_patch,
            (self.cfg.img_size, self.cfg.img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        h0, w0 = frame_bgr.shape[:2]
        mask_full = cv2.resize(
            mask_resized,
            (w0, h0),
            interpolation=cv2.INTER_NEAREST,
        )

        # Post-process mask and derive bounding box.
        mask_uint8 = (mask_full > 0).astype(np.uint8) * 255
        bbox = self._largest_component_bbox(mask_uint8)

        # Optional template update using the new bbox.
        if bbox is not None and self.cfg.template_update_alpha > 0.0:
            self._update_template(frame_proc, bbox, scale_x, scale_y)

        return mask_uint8, bbox, sim_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[torch.Tensor, float, float]:
        """Resize and normalize a BGR frame for DINOv3.

        Args:
            frame_bgr (np.ndarray): ``H x W x 3`` uint8 BGR frame from OpenCV.

        Returns:
            tuple[torch.Tensor, float, float]:
                ``tensor`` is a ``1 x 3 x img_size x img_size`` float32 tensor
                on the configured device, ``scale_x`` maps original x
                coordinates into the resized space, and ``scale_y`` does the
                same for y coordinates.

        Raises:
            ValueError: If the frame is not a valid ``HxWx3`` image or has
                non-positive spatial dimensions.

        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            msg = f"Expected HxWx3 BGR frame, got shape {frame_bgr.shape}"
            raise ValueError(msg)

        h0, w0 = frame_bgr.shape[:2]
        if h0 <= 0 or w0 <= 0:
            msg = f"Invalid frame size: (h={h0}, w={w0})"
            raise ValueError(msg)

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb,
            (self.cfg.img_size, self.cfg.img_size),
            interpolation=cv2.INTER_LINEAR,
        )

        scale_x = self.cfg.img_size / float(w0)
        scale_y = self.cfg.img_size / float(h0)

        img_f32 = img_resized.astype(np.float32) / 255.0
        img_f32 = (img_f32 - self._mean) / self._std
        img_chw = np.transpose(img_f32, (2, 0, 1))  # [3, H, W]

        tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        return tensor, scale_x, scale_y

    def _extract_patch_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Run DINOv3 and return patch features of the last layer.

        Args:
            img_tensor (torch.Tensor): ``1 x 3 x H x W`` float32 tensor.

        Returns:
            torch.Tensor: Patch feature map of shape ``[1, C, H_p, W_p]``.

        Raises:
            RuntimeError: If the backbone does not return a 4-D feature map.

        """
        with torch.inference_mode():
            features = self.model.get_intermediate_layers(
                img_tensor,
                n=1,
                reshape=True,
                return_class_token=False,
            )
        # get_intermediate_layers returns a tuple of length ``n``
        feat = features[0]
        if feat.ndim != 4:
            msg = f"Unexpected feature shape from DINOv3: {tuple(feat.shape)}"
            raise RuntimeError(msg)
        return feat

    def _largest_component_bbox(
        self,
        mask_uint8: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """Return bounding box of the largest connected component in ``mask``.

        Args:
            mask_uint8 (np.ndarray): Binary mask ``H x W`` with values 0 or 255.

        Returns:
            tuple[int, int, int, int] | None: ``(x, y, w, h)`` for the largest
            component, or ``None`` if no foreground pixels are present.

        """
        if mask_uint8.max() == 0:
            return None

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_idx = int(np.argmax(areas))
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        return int(x), int(y), int(w), int(h)

    def _update_template(
        self,
        frame_proc: torch.Tensor,
        bbox_xywh: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> None:
        """EMA update of the template vector using the current bbox.

        The bbox is provided in *original* coordinates, so we map it to resized
        coordinates in the same way as :meth:`set_template`.
        """
        if self._template_vec is None:
            return

        x, y, w, h = bbox_xywh
        if w <= 1 or h <= 1:
            return

        feats = self._extract_patch_features(frame_proc)
        _, c, h_p, w_p = feats.shape

        x_resized0 = x * scale_x
        y_resized0 = y * scale_y
        x_resized1 = (x + w) * scale_x
        y_resized1 = (y + h) * scale_y

        j0 = int(np.floor(x_resized0 * w_p / float(self.cfg.img_size)))
        j1 = int(np.ceil(x_resized1 * w_p / float(self.cfg.img_size))) - 1
        i0 = int(np.floor(y_resized0 * h_p / float(self.cfg.img_size)))
        i1 = int(np.ceil(y_resized1 * h_p / float(self.cfg.img_size))) - 1

        j0 = max(0, min(w_p - 1, j0))
        j1 = max(0, min(w_p - 1, j1))
        i0 = max(0, min(h_p - 1, i0))
        i1 = max(0, min(h_p - 1, i1))
        if j1 < j0 or i1 < i0:
            return

        patch_tokens = feats[0, :, i0 : i1 + 1, j0 : j1 + 1].reshape(c, -1)
        new_vec = patch_tokens.mean(dim=1)
        new_vec = torch.nn.functional.normalize(new_vec, dim=0)

        alpha = float(self.cfg.template_update_alpha)
        old_vec = self._template_vec.to(new_vec.device)
        mixed = (1.0 - alpha) * old_vec + alpha * new_vec
        mixed = torch.nn.functional.normalize(mixed, dim=0)
        self._template_vec = mixed.detach()
