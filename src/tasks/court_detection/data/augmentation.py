"""Data augmentation pipelines for court detection.

Provides joint spatial transforms that operate on both the image and
accompanying target (segmentation mask or keypoints).

Two public factory functions build the full pipeline:

* :func:`build_seg_transforms` — image + segmentation mask.
* :func:`build_kp_transforms` — image + keypoint array ``(N, 2)``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from src.utils.data.augmentation import (
    IMAGENET_MEAN as IMAGENET_MEAN,
)
from src.utils.data.augmentation import (
    IMAGENET_STD as IMAGENET_STD,
)

# ── Helpers ──────────────────────────────────────────────────────


def _pil_to_tensor_image(img: Image.Image) -> Tensor:
    """Convert PIL Image to float tensor ``[C, H, W]`` in ``[0, 1]``."""
    return TF.to_tensor(img)


def _mask_pil_to_tensor(mask: Image.Image) -> Tensor:
    """Convert a single-channel PIL mask to ``int64`` tensor ``[H, W]``."""
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def _sample_perspective_h(
    w: int,
    h: int,
    distortion_scale: float,
) -> np.ndarray | None:
    """Sample a random perspective homography matrix."""
    if distortion_scale <= 0.0:
        return None

    src = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )
    max_offset = distortion_scale * min(w, h)
    jitter = np.random.uniform(-max_offset, max_offset, size=(4, 2)).astype(np.float32)
    dst = src + jitter
    dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
    dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)

    if cv2.contourArea(dst.astype(np.float32)) < 1.0:
        return None

    h_mat = cv2.getPerspectiveTransform(src, dst)
    return h_mat.astype(np.float32)


def _warp_perspective_pil(
    img: Image.Image,
    h_mat: np.ndarray,
    *,
    interpolation: int,
    fill_value: int,
) -> Image.Image:
    """Apply OpenCV perspective warp to a PIL image."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    warped = cv2.warpPerspective(
        arr,
        h_mat,
        dsize=(w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )
    return Image.fromarray(warped)


# ── Joint Spatial Transforms (Segmentation) ──────────────────────


class SegMultiScaleResize:
    """Resize image + mask so the short side equals a random scale value."""

    def __init__(self, scales: Sequence[int]) -> None:
        self.scales = list(scales)

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        short_side = random.choice(self.scales)
        w, h = img.size
        if h <= w:
            new_h = short_side
            new_w = int(round(w * new_h / h))
        else:
            new_w = short_side
            new_h = int(round(h * new_w / w))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return img, mask


class SegFixedResize:
    """Resize image + mask so the short side is a fixed value."""

    def __init__(self, short_side: int) -> None:
        self.short_side = short_side

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = img.size
        if h <= w:
            new_h = self.short_side
            new_w = int(round(w * new_h / h))
        else:
            new_w = self.short_side
            new_h = int(round(h * new_w / w))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return img, mask


class SegRandomResizedCrop:
    """Random resized crop applied jointly to image + mask."""

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        ratio: tuple[float, float] = (0.75, 1.333),
    ) -> None:
        self.scale = scale
        self.ratio = ratio

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = img.size
        area = h * w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                img = TF.crop(img, top, left, crop_h, crop_w)
                mask = TF.crop(mask, top, left, crop_h, crop_w)
                return img, mask
        crop_h = min(h, w)
        crop_w = crop_h
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        img = TF.crop(img, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        return img, mask


class SegRandomHorizontalFlip:
    """Random horizontal flip with label swap for segmentation masks."""

    def __init__(self, p: float, swap_pairs: list[tuple[int, int]]) -> None:
        self.p = p
        self.swap_pairs = swap_pairs

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            mask_np = np.array(mask)
            swapped = mask_np.copy()
            for a, b in self.swap_pairs:
                swapped[mask_np == a] = b
                swapped[mask_np == b] = a
            mask = Image.fromarray(swapped)
        return img, mask


class SegRandomAffine:
    """Random affine transform applied jointly to image + mask."""

    def __init__(
        self,
        degrees: float = 15.0,
        translate: tuple[float, float] = (0.1, 0.1),
        scale: tuple[float, float] = (0.8, 1.2),
        shear: float = 10.0,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        angle = random.uniform(-self.degrees, self.degrees)
        w, h = img.size
        max_dx = self.translate[0] * w
        max_dy = self.translate[1] * h
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear_val = random.uniform(-self.shear, self.shear)

        img = TF.affine(
            img, angle=angle, translate=[tx, ty],
            scale=scale_factor, shear=[shear_val],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask, angle=angle, translate=[tx, ty],
            scale=scale_factor, shear=[shear_val],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        return img, mask


class SegRandomPerspective:
    """Random perspective transform applied jointly to image + mask."""

    def __init__(self, distortion_scale: float = 0.15, p: float = 0.3) -> None:
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(
        self, img: Image.Image, mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() >= self.p:
            return img, mask

        w, h = img.size
        h_mat = _sample_perspective_h(w, h, self.distortion_scale)
        if h_mat is None:
            return img, mask

        img = _warp_perspective_pil(img, h_mat, interpolation=cv2.INTER_LINEAR, fill_value=0)
        mask = _warp_perspective_pil(mask, h_mat, interpolation=cv2.INTER_NEAREST, fill_value=0)
        return img, mask


# ── Joint Spatial Transforms (Keypoints) ─────────────────────────


class KPMultiScaleResize:
    """Resize image and scale keypoints to match new dimensions."""

    def __init__(self, scales: Sequence[int]) -> None:
        self.scales = list(scales)

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        short_side = random.choice(self.scales)
        w, h = img.size
        if h <= w:
            new_h = short_side
            new_w = int(round(w * new_h / h))
        else:
            new_w = short_side
            new_h = int(round(h * new_w / w))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        sx = new_w / w
        sy = new_h / h
        img = img.resize((new_w, new_h), Image.BILINEAR)
        kps = kps.copy()
        kps[:, 0] *= sx
        kps[:, 1] *= sy
        return img, kps


class KPFixedResize:
    """Resize image + keypoints to fixed short side."""

    def __init__(self, short_side: int) -> None:
        self.short_side = short_side

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        w, h = img.size
        if h <= w:
            new_h = self.short_side
            new_w = int(round(w * new_h / h))
        else:
            new_w = self.short_side
            new_h = int(round(h * new_w / w))
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        sx = new_w / w
        sy = new_h / h
        img = img.resize((new_w, new_h), Image.BILINEAR)
        kps = kps.copy()
        kps[:, 0] *= sx
        kps[:, 1] *= sy
        return img, kps


class KPRandomResizedCrop:
    """Random resized crop applied jointly to image + keypoints."""

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        ratio: tuple[float, float] = (0.75, 1.333),
    ) -> None:
        self.scale = scale
        self.ratio = ratio

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        w, h = img.size
        area = h * w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                img = TF.crop(img, top, left, crop_h, crop_w)
                kps = kps.copy()
                kps[:, 0] -= left
                kps[:, 1] -= top
                return img, kps
        crop_h = min(h, w)
        crop_w = crop_h
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        img = TF.crop(img, top, left, crop_h, crop_w)
        kps = kps.copy()
        kps[:, 0] -= left
        kps[:, 1] -= top
        return img, kps


class KPRandomHorizontalFlip:
    """Random horizontal flip with keypoint index swap."""

    def __init__(self, p: float, swap_pairs: list[tuple[int, int]]) -> None:
        self.p = p
        self.swap_pairs = swap_pairs

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        if random.random() < self.p:
            w, _ = img.size
            img = TF.hflip(img)
            kps = kps.copy()
            kps[:, 0] = w - 1 - kps[:, 0]
            for i, j in self.swap_pairs:
                kps[[i, j]] = kps[[j, i]]
        return img, kps


class KPRandomAffine:
    """Random affine transform applied jointly to image + keypoints."""

    def __init__(
        self,
        degrees: float = 15.0,
        translate: tuple[float, float] = (0.1, 0.1),
        scale: tuple[float, float] = (0.8, 1.2),
        shear: float = 10.0,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        angle = random.uniform(-self.degrees, self.degrees)
        w, h = img.size
        max_dx = self.translate[0] * w
        max_dy = self.translate[1] * h
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        shear_deg = random.uniform(-self.shear, self.shear)

        img = TF.affine(
            img, angle=angle, translate=[tx, ty],
            scale=scale_factor, shear=[shear_deg],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        cx, cy = w / 2.0, h / 2.0
        angle_rad = math.radians(-angle)
        shear_rad = math.radians(shear_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        m00 = scale_factor * (cos_a + sin_a * math.tan(shear_rad))
        m01 = scale_factor * (-sin_a + cos_a * math.tan(shear_rad))
        m10 = scale_factor * sin_a
        m11 = scale_factor * cos_a

        kps = kps.copy().astype(np.float64)
        kps[:, 0] -= cx
        kps[:, 1] -= cy
        new_x = m00 * kps[:, 0] + m01 * kps[:, 1]
        new_y = m10 * kps[:, 0] + m11 * kps[:, 1]
        kps[:, 0] = new_x + cx + tx
        kps[:, 1] = new_y + cy + ty
        kps = kps.astype(np.float32)
        return img, kps


class KPRandomPerspective:
    """Random perspective transform applied jointly to image + keypoints."""

    def __init__(self, distortion_scale: float = 0.15, p: float = 0.3) -> None:
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        if random.random() >= self.p:
            return img, kps

        w, h = img.size
        h_mat = _sample_perspective_h(w, h, self.distortion_scale)
        if h_mat is None:
            return img, kps

        img = _warp_perspective_pil(img, h_mat, interpolation=cv2.INTER_LINEAR, fill_value=0)

        kps_in = kps.astype(np.float32).reshape(-1, 1, 2)
        kps_out = cv2.perspectiveTransform(kps_in, h_mat).reshape(-1, 2)
        return img, kps_out.astype(np.float32)


# ── Image-only Transforms ────────────────────────────────────────


class ImageColorJitter:
    """Random colour jitter applied only to the image."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Image.Image) -> Image.Image:
        from torchvision.transforms import ColorJitter as _CJ

        return _CJ(self.brightness, self.contrast, self.saturation, self.hue)(img)


class ImageGaussianBlur:
    """Random Gaussian blur applied only to the image."""

    def __init__(
        self,
        kernel_size: list[int] | None = None,
        sigma: tuple[float, float] = (0.1, 2.0),
        p: float = 0.3,
    ) -> None:
        self.kernel_size = kernel_size or [3, 7]
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            k = random.choice(self.kernel_size)
            if k % 2 == 0:
                k += 1
            s = random.uniform(self.sigma[0], self.sigma[1])
            img = TF.gaussian_blur(img, kernel_size=[k, k], sigma=[s, s])
        return img


# ── Pipeline Builders ─────────────────────────────────────────────


def build_seg_transforms(
    *,
    is_train: bool,
    train_scales: list[int] | None = None,
    val_short_side: int = 640,
    crop_scale: tuple[float, float] = (0.3, 1.0),
    crop_ratio: tuple[float, float] = (0.75, 1.333),
    hflip_prob: float = 0.5,
    swap_pairs: list[tuple[int, int]] | None = None,
    affine_degrees: float = 15.0,
    affine_translate: tuple[float, float] = (0.1, 0.1),
    affine_scale: tuple[float, float] = (0.8, 1.2),
    affine_shear: float = 10.0,
    perspective_distortion: float = 0.15,
    perspective_prob: float = 0.3,
    color_jitter: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1),
    gaussian_blur_kernel: list[int] | None = None,
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0),
    gaussian_blur_prob: float = 0.3,
) -> tuple[list, list]:
    """Build joint spatial + image-only transforms for segmentation."""
    if swap_pairs is None:
        swap_pairs = [(1, 2), (3, 4), (5, 6)]

    spatial: list = []
    image_only: list = []

    if is_train:
        spatial.append(SegMultiScaleResize(train_scales or [480, 512, 544, 576, 608, 640]))
        spatial.append(SegRandomResizedCrop(scale=crop_scale, ratio=crop_ratio))
        spatial.append(SegRandomHorizontalFlip(p=hflip_prob, swap_pairs=swap_pairs))
        spatial.append(SegRandomAffine(
            degrees=affine_degrees, translate=affine_translate,
            scale=affine_scale, shear=affine_shear,
        ))
        spatial.append(SegRandomPerspective(
            distortion_scale=perspective_distortion, p=perspective_prob,
        ))
        image_only.append(ImageColorJitter(*color_jitter))
        image_only.append(ImageGaussianBlur(
            kernel_size=gaussian_blur_kernel, sigma=gaussian_blur_sigma, p=gaussian_blur_prob,
        ))
    else:
        spatial.append(SegFixedResize(val_short_side))

    return spatial, image_only


def build_kp_transforms(
    *,
    is_train: bool,
    train_scales: list[int] | None = None,
    val_short_side: int = 640,
    crop_scale: tuple[float, float] = (0.3, 1.0),
    crop_ratio: tuple[float, float] = (0.75, 1.333),
    hflip_prob: float = 0.5,
    swap_pairs: list[tuple[int, int]] | None = None,
    affine_degrees: float = 15.0,
    affine_translate: tuple[float, float] = (0.1, 0.1),
    affine_scale: tuple[float, float] = (0.8, 1.2),
    affine_shear: float = 10.0,
    perspective_distortion: float = 0.15,
    perspective_prob: float = 0.3,
    color_jitter: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1),
    gaussian_blur_kernel: list[int] | None = None,
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0),
    gaussian_blur_prob: float = 0.3,
) -> tuple[list, list]:
    """Build joint spatial + image-only transforms for keypoint heatmaps."""
    if swap_pairs is None:
        swap_pairs = [(0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 11)]

    spatial: list = []
    image_only: list = []

    if is_train:
        spatial.append(KPMultiScaleResize(train_scales or [480, 512, 544, 576, 608, 640]))
        spatial.append(KPRandomResizedCrop(scale=crop_scale, ratio=crop_ratio))
        spatial.append(KPRandomHorizontalFlip(p=hflip_prob, swap_pairs=swap_pairs))
        spatial.append(KPRandomAffine(
            degrees=affine_degrees, translate=affine_translate,
            scale=affine_scale, shear=affine_shear,
        ))
        spatial.append(KPRandomPerspective(
            distortion_scale=perspective_distortion, p=perspective_prob,
        ))
        image_only.append(ImageColorJitter(*color_jitter))
        image_only.append(ImageGaussianBlur(
            kernel_size=gaussian_blur_kernel, sigma=gaussian_blur_sigma, p=gaussian_blur_prob,
        ))
    else:
        spatial.append(KPFixedResize(val_short_side))

    return spatial, image_only
