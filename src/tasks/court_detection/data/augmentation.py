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
from dataclasses import dataclass
from typing import Generic, TypeVar

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
from src.utils.geometry.affine import (
    AffineMatrix,
    build_centered_affine_matrix,
    invert_homogeneous_matrix,
    to_pil_affine_coefficients,
    transform_points,
)
from src.utils.geometry.image_size import resize_short_side_aligned

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

    h_mat: np.ndarray = np.asarray(cv2.getPerspectiveTransform(src, dst), dtype=np.float32)
    return h_mat


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


def _random_affine_matrix(
    *,
    width: int,
    height: int,
    degrees: float,
    translate: tuple[float, float],
    scale: tuple[float, float],
    shear: float,
) -> AffineMatrix:
    """Sample one source->destination affine matrix shared by image/mask warps."""
    angle = random.uniform(-degrees, degrees)
    max_dx = translate[0] * width
    max_dy = translate[1] * height
    tx = random.uniform(-max_dx, max_dx)
    ty = random.uniform(-max_dy, max_dy)
    scale_factor = random.uniform(scale[0], scale[1])
    shear_deg = random.uniform(-shear, shear)
    return build_centered_affine_matrix(
        width=width,
        height=height,
        rotation_degrees=angle,
        translate=(tx, ty),
        scale=scale_factor,
        shear_degrees=(shear_deg, shear_deg),
        shear_mode="torchvision",
    )


def _warp_pil_affine(img: Image.Image, matrix: AffineMatrix, *, resample: int) -> Image.Image:
    """Warp a PIL image with the inverse of a source->destination affine matrix."""
    coeffs = to_pil_affine_coefficients(invert_homogeneous_matrix(matrix))
    return img.transform(img.size, Image.AFFINE, coeffs, resample, fillcolor=0)


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
        new_w, new_h = resize_short_side_aligned(w, h, short_side)
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
        new_w, new_h = resize_short_side_aligned(w, h, self.short_side)
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
        w, h = img.size
        matrix = _random_affine_matrix(
            width=w, height=h, degrees=self.degrees,
            translate=self.translate, scale=self.scale, shear=self.shear,
        )
        img = _warp_pil_affine(img, matrix, resample=Image.BILINEAR)
        mask = _warp_pil_affine(mask, matrix, resample=Image.NEAREST)
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
#
# Keypoint transforms are split into parameter sampling and application so
# that a whole parameter chain can be evaluated on the keypoints alone
# (cheap ndarray math) before the image is warped once. This enables
# :class:`KPVisibilityConstrainedPipeline` to reject parameter draws that
# push too many keypoints out of bounds without paying for image warps.

P = TypeVar("P")


def kp_in_bounds_mask(kps: np.ndarray, w: int, h: int) -> np.ndarray:
    """Boolean mask of keypoints inside ``[0, w-1] x [0, h-1]``."""
    mask: np.ndarray = (
        (kps[:, 0] >= 0.0)
        & (kps[:, 0] <= w - 1.0)
        & (kps[:, 1] >= 0.0)
        & (kps[:, 1] <= h - 1.0)
    )
    return mask


class KPParamTransform(Generic[P]):
    """Keypoint spatial transform split into sampling and application."""

    def sample_params(self, w: int, h: int) -> tuple[P, int, int]:
        """Draw parameters; return ``(params, out_w, out_h)``."""
        raise NotImplementedError

    def apply_to_image(self, img: Image.Image, params: P) -> Image.Image:
        raise NotImplementedError

    def apply_to_kps(self, kps: np.ndarray, params: P) -> np.ndarray:
        raise NotImplementedError

    def apply_to_mask(self, mask: np.ndarray, params: P) -> np.ndarray:
        """Propagate a per-keypoint boolean mask through index permutations."""
        return mask

    def out_size(self, params: P, w: int, h: int) -> tuple[int, int]:
        """Output image size for the given params and input size ``(w, h)``."""
        return w, h

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        w, h = img.size
        params, _, _ = self.sample_params(w, h)
        return self.apply_to_image(img, params), self.apply_to_kps(kps, params)


@dataclass(frozen=True)
class ResizeParams:
    sx: float
    sy: float
    new_w: int
    new_h: int


@dataclass(frozen=True)
class CropParams:
    top: int
    left: int
    crop_h: int
    crop_w: int


@dataclass(frozen=True)
class FlipParams:
    flip: bool
    w: int


@dataclass(frozen=True)
class AffineParams:
    angle: float
    tx: float
    ty: float
    scale: float
    shear: float
    w: int
    h: int


@dataclass(frozen=True)
class PerspectiveParams:
    h_mat: np.ndarray | None


class _KPResizeBase(KPParamTransform[ResizeParams]):
    """Shared application logic for short-side resizes."""

    def _params_for(self, w: int, h: int, short_side: int) -> tuple[ResizeParams, int, int]:
        new_w, new_h = resize_short_side_aligned(w, h, short_side)
        return ResizeParams(sx=new_w / w, sy=new_h / h, new_w=new_w, new_h=new_h), new_w, new_h

    def apply_to_image(self, img: Image.Image, params: ResizeParams) -> Image.Image:
        return img.resize((params.new_w, params.new_h), Image.BILINEAR)

    def apply_to_kps(self, kps: np.ndarray, params: ResizeParams) -> np.ndarray:
        kps = kps.copy()
        kps[:, 0] *= params.sx
        kps[:, 1] *= params.sy
        return kps

    def out_size(self, params: ResizeParams, w: int, h: int) -> tuple[int, int]:
        return params.new_w, params.new_h


class KPMultiScaleResize(_KPResizeBase):
    """Resize image and scale keypoints so the short side matches a random scale."""

    def __init__(self, scales: Sequence[int]) -> None:
        self.scales = list(scales)

    def sample_params(self, w: int, h: int) -> tuple[ResizeParams, int, int]:
        return self._params_for(w, h, random.choice(self.scales))


class KPFixedResize(_KPResizeBase):
    """Resize image + keypoints to fixed short side."""

    def __init__(self, short_side: int) -> None:
        self.short_side = short_side

    def sample_params(self, w: int, h: int) -> tuple[ResizeParams, int, int]:
        return self._params_for(w, h, self.short_side)


class KPRandomResizedCrop(KPParamTransform[CropParams]):
    """Random resized crop applied jointly to image + keypoints."""

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        ratio: tuple[float, float] = (0.75, 1.333),
    ) -> None:
        self.scale = scale
        self.ratio = ratio

    def sample_params(self, w: int, h: int) -> tuple[CropParams, int, int]:
        area = h * w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                return CropParams(top=top, left=left, crop_h=crop_h, crop_w=crop_w), crop_w, crop_h
        crop_h = min(h, w)
        crop_w = crop_h
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return CropParams(top=top, left=left, crop_h=crop_h, crop_w=crop_w), crop_w, crop_h

    def apply_to_image(self, img: Image.Image, params: CropParams) -> Image.Image:
        return TF.crop(img, params.top, params.left, params.crop_h, params.crop_w)

    def apply_to_kps(self, kps: np.ndarray, params: CropParams) -> np.ndarray:
        kps = kps.copy()
        kps[:, 0] -= params.left
        kps[:, 1] -= params.top
        return kps

    def out_size(self, params: CropParams, w: int, h: int) -> tuple[int, int]:
        return params.crop_w, params.crop_h


class KPRandomHorizontalFlip(KPParamTransform[FlipParams]):
    """Random horizontal flip with keypoint index swap."""

    def __init__(self, p: float, swap_pairs: list[tuple[int, int]]) -> None:
        self.p = p
        self.swap_pairs = swap_pairs

    def sample_params(self, w: int, h: int) -> tuple[FlipParams, int, int]:
        return FlipParams(flip=random.random() < self.p, w=w), w, h

    def apply_to_image(self, img: Image.Image, params: FlipParams) -> Image.Image:
        return TF.hflip(img) if params.flip else img

    def apply_to_kps(self, kps: np.ndarray, params: FlipParams) -> np.ndarray:
        if not params.flip:
            return kps
        kps = kps.copy()
        kps[:, 0] = params.w - 1 - kps[:, 0]
        for i, j in self.swap_pairs:
            kps[[i, j]] = kps[[j, i]]
        return kps

    def apply_to_mask(self, mask: np.ndarray, params: FlipParams) -> np.ndarray:
        if not params.flip:
            return mask
        mask = mask.copy()
        for i, j in self.swap_pairs:
            mask[[i, j]] = mask[[j, i]]
        return mask


class KPRandomAffine(KPParamTransform[AffineParams]):
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

    def sample_params(self, w: int, h: int) -> tuple[AffineParams, int, int]:
        params = AffineParams(
            angle=random.uniform(-self.degrees, self.degrees),
            tx=random.uniform(-self.translate[0] * w, self.translate[0] * w),
            ty=random.uniform(-self.translate[1] * h, self.translate[1] * h),
            scale=random.uniform(self.scale[0], self.scale[1]),
            shear=random.uniform(-self.shear, self.shear),
            w=w,
            h=h,
        )
        return params, w, h

    @staticmethod
    def _matrix(params: AffineParams) -> AffineMatrix:
        return build_centered_affine_matrix(
            width=params.w,
            height=params.h,
            rotation_degrees=params.angle,
            translate=(params.tx, params.ty),
            scale=params.scale,
            shear_degrees=(params.shear, params.shear),
            shear_mode="torchvision",
        )

    def apply_to_image(self, img: Image.Image, params: AffineParams) -> Image.Image:
        return _warp_pil_affine(img, self._matrix(params), resample=Image.BILINEAR)

    def apply_to_kps(self, kps: np.ndarray, params: AffineParams) -> np.ndarray:
        return transform_points(kps, self._matrix(params))


class KPRandomPerspective(KPParamTransform[PerspectiveParams]):
    """Random perspective transform applied jointly to image + keypoints."""

    def __init__(self, distortion_scale: float = 0.15, p: float = 0.3) -> None:
        self.distortion_scale = distortion_scale
        self.p = p

    def sample_params(self, w: int, h: int) -> tuple[PerspectiveParams, int, int]:
        h_mat: np.ndarray | None = None
        if random.random() < self.p:
            h_mat = _sample_perspective_h(w, h, self.distortion_scale)
        return PerspectiveParams(h_mat=h_mat), w, h

    def apply_to_image(self, img: Image.Image, params: PerspectiveParams) -> Image.Image:
        if params.h_mat is None:
            return img
        return _warp_perspective_pil(
            img, params.h_mat, interpolation=cv2.INTER_LINEAR, fill_value=0,
        )

    def apply_to_kps(self, kps: np.ndarray, params: PerspectiveParams) -> np.ndarray:
        if params.h_mat is None:
            return kps
        kps_in: np.ndarray = kps.astype(np.float32).reshape(-1, 1, 2)
        kps_out: np.ndarray = np.asarray(
            cv2.perspectiveTransform(kps_in, params.h_mat), dtype=np.float32,
        ).reshape(-1, 2)
        return kps_out


class KPVisibilityConstrainedPipeline:
    """Chain of keypoint transforms with a minimum-visibility guarantee."""

    def __init__(
        self,
        transforms: Sequence[KPParamTransform],
        min_visible_kp: int = 0,
        max_retries: int = 20,
    ) -> None:
        if min_visible_kp < 0:
            raise ValueError(f"min_visible_kp must be >= 0, got {min_visible_kp}")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.transforms = list(transforms)
        self.min_visible_kp = min_visible_kp
        self.max_retries = max_retries

    def _sample_chain(
        self, w: int, h: int, kps: np.ndarray,
    ) -> tuple[list[object], np.ndarray, np.ndarray]:
        """Draw one parameter chain and apply it to the keypoints only."""
        chain: list[object] = []
        out = kps
        mask = kp_in_bounds_mask(kps, w, h)
        for t in self.transforms:
            params, w, h = t.sample_params(w, h)
            out = t.apply_to_kps(out, params)
            mask = t.apply_to_mask(mask, params) & kp_in_bounds_mask(out, w, h)
            chain.append(params)
        return chain, out, mask

    def draw_params(
        self, w: int, h: int, kps: np.ndarray,
    ) -> tuple[list[object], np.ndarray, np.ndarray, int]:
        """Sample parameter chains until the visibility constraint is met."""
        target = min(self.min_visible_kp, int(kp_in_bounds_mask(kps, w, h).sum()))

        best: tuple[list[object], np.ndarray, np.ndarray] | None = None
        best_visible = -1
        attempts = 0
        while attempts < self.max_retries:
            attempts += 1
            chain, out, mask = self._sample_chain(w, h, kps)
            visible = int(mask.sum())
            if visible > best_visible:
                best, best_visible = (chain, out, mask), visible
            if visible >= target:
                break

        assert best is not None
        return best[0], best[1], best[2], attempts

    def transform_with_visibility(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        """Transform and additionally return the cumulative visibility mask."""
        w, h = img.size
        chain, out_kps, mask, _ = self.draw_params(w, h, kps)
        for t, params in zip(self.transforms, chain, strict=True):
            img = t.apply_to_image(img, params)
        return img, out_kps, mask

    def __call__(
        self, img: Image.Image, kps: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        img, kps, _ = self.transform_with_visibility(img, kps)
        return img, kps


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
    min_visible_kp: int = 0,
    visibility_max_retries: int = 20,
) -> tuple[KPVisibilityConstrainedPipeline, list]:
    """Build the spatial pipeline + image-only transforms for keypoint heatmaps."""
    if swap_pairs is None:
        swap_pairs = [(0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 11)]

    transforms: list[KPParamTransform] = []
    image_only: list = []

    if is_train:
        transforms.append(KPMultiScaleResize(train_scales or [480, 512, 544, 576, 608, 640]))
        transforms.append(KPRandomResizedCrop(scale=crop_scale, ratio=crop_ratio))
        transforms.append(KPRandomHorizontalFlip(p=hflip_prob, swap_pairs=swap_pairs))
        transforms.append(KPRandomAffine(
            degrees=affine_degrees, translate=affine_translate,
            scale=affine_scale, shear=affine_shear,
        ))
        transforms.append(KPRandomPerspective(
            distortion_scale=perspective_distortion, p=perspective_prob,
        ))
        image_only.append(ImageColorJitter(*color_jitter))
        image_only.append(ImageGaussianBlur(
            kernel_size=gaussian_blur_kernel, sigma=gaussian_blur_sigma, p=gaussian_blur_prob,
        ))
        spatial = KPVisibilityConstrainedPipeline(
            transforms,
            min_visible_kp=min_visible_kp,
            max_retries=visibility_max_retries,
        )
    else:
        spatial = KPVisibilityConstrainedPipeline([KPFixedResize(val_short_side)])

    return spatial, image_only
