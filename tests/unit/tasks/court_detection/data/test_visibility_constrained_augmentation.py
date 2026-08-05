from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.tasks.court_detection.data.augmentation import (
    CropParams,
    FlipParams,
    KPParamTransform,
    KPRandomHorizontalFlip,
    KPRandomResizedCrop,
    KPVisibilityConstrainedPipeline,
    build_kp_transforms,
)
from src.utils.geometry.image_size import resize_short_side_aligned

W, H = 320, 180


class _FixedCrop(KPParamTransform):
    def __init__(self, params: CropParams) -> None:
        self.params = params

    def sample_params(self, w: int, h: int):  # noqa: ANN201
        return self.params, self.params.crop_w, self.params.crop_h

    def apply_to_image(self, img: Image.Image, params: CropParams) -> Image.Image:
        return img.crop((
            params.left,
            params.top,
            params.left + params.crop_w,
            params.top + params.crop_h,
        ))

    def apply_to_kps(self, kps: np.ndarray, params: CropParams) -> np.ndarray:
        kps = kps.copy()
        kps[:, 0] -= params.left
        kps[:, 1] -= params.top
        return kps


class _FixedShift(KPParamTransform):
    def __init__(self, dx: float) -> None:
        self.dx = dx

    def sample_params(self, w: int, h: int):  # noqa: ANN201
        return None, w, h

    def apply_to_image(self, img: Image.Image, params: None) -> Image.Image:
        return img

    def apply_to_kps(self, kps: np.ndarray, params: None) -> np.ndarray:
        kps = kps.copy()
        kps[:, 0] += self.dx
        return kps


def test_flip_mask_follows_swapped_keypoint_identity() -> None:
    transform = KPRandomHorizontalFlip(p=1.0, swap_pairs=[(0, 1)])
    params = FlipParams(flip=True, w=W)
    kps = np.array([[10.0, 5.0], [300.0, 5.0], [100.0, 5.0]], dtype=np.float32)
    mask = np.array([True, False, True])

    out_kps = transform.apply_to_kps(kps, params)
    out_mask = transform.apply_to_mask(mask, params)

    assert out_mask.tolist() == [False, True, True]
    np.testing.assert_allclose(out_kps[0, 0], W - 1 - 300.0)
    np.testing.assert_allclose(out_kps[1, 0], W - 1 - 10.0)


def test_cumulative_visibility_rejects_reentering_keypoint() -> None:
    img = Image.new("RGB", (W, H))
    kps = np.array([[10.0, 50.0], [200.0, 50.0]], dtype=np.float32)
    crop = _FixedCrop(CropParams(top=0, left=100, crop_h=H, crop_w=200))
    shift = _FixedShift(dx=95.0)
    pipeline = KPVisibilityConstrainedPipeline(
        [crop, shift], min_visible_kp=0, max_retries=1
    )

    _, out_kps, mask = pipeline.transform_with_visibility(img, kps)

    assert out_kps[0, 0] == pytest.approx(5.0)
    assert mask.tolist() == [False, True]


def test_min_visible_target_is_capped_by_input_visibility() -> None:
    img = Image.new("RGB", (W, H))
    kps = np.array([[-500.0, 50.0], [100.0, 50.0]], dtype=np.float32)
    pipeline = KPVisibilityConstrainedPipeline(
        [_FixedShift(dx=0.0)], min_visible_kp=2, max_retries=3,
    )

    _, _, mask = pipeline.transform_with_visibility(img, kps)

    assert mask.tolist() == [False, True]


def test_constraint_raises_visible_keypoint_count() -> None:
    random.seed(0)
    np.random.seed(0)
    img = Image.new("RGB", (W, H))
    kps = np.stack(
        [
            np.linspace(W * 0.35, W * 0.65, 14).astype(np.float32),
            np.linspace(H * 0.35, H * 0.65, 14).astype(np.float32),
        ],
        axis=1,
    )
    crop = KPRandomResizedCrop(scale=(0.2, 0.6), ratio=(0.8, 1.25))

    def run(min_visible: int) -> float:
        counts = []
        for _ in range(100):
            pipeline = KPVisibilityConstrainedPipeline(
                [crop], min_visible_kp=min_visible, max_retries=30,
            )
            _, _, mask = pipeline.transform_with_visibility(img, kps)
            counts.append(mask.sum())
        return float(np.mean(counts))

    assert run(10) > run(0)


def test_invalid_visibility_constraint_args() -> None:
    with pytest.raises(ValueError):
        KPVisibilityConstrainedPipeline([], min_visible_kp=-1, max_retries=1)
    with pytest.raises(ValueError):
        KPVisibilityConstrainedPipeline([], min_visible_kp=0, max_retries=0)


def test_val_pipeline_visibility_mask_matches_resize_bounds() -> None:
    pipeline, image_only = build_kp_transforms(
        is_train=False,
        train_scales=[288],
        val_short_side=90,
        crop_scale=(0.2, 1.0),
        crop_ratio=(0.5, 2.0),
        hflip_prob=0.7,
        swap_pairs=[(0, 1)],
        affine_degrees=25.0,
        affine_translate=(0.18, 0.18),
        affine_scale=(0.65, 1.5),
        affine_shear=18.0,
        perspective_distortion=0.25,
        perspective_prob=0.6,
        color_jitter=(0.5, 0.5, 0.5, 0.2),
        gaussian_blur_kernel=[3, 5, 7, 9],
        gaussian_blur_sigma=(0.1, 3.0),
        gaussian_blur_prob=0.5,
        min_visible_kp=12,
        visibility_max_retries=10,
    )
    assert image_only == []
    img = Image.new("RGB", (W, H))
    kps = np.array(
        [[0.0, 0.0], [W - 3.0, H - 3.0], [-5.0, 10.0]], dtype=np.float32,
    )

    out_img, _, mask = pipeline.transform_with_visibility(img, kps)

    expected_w, expected_h = resize_short_side_aligned(W, H, 90)
    assert out_img.size == (expected_w, expected_h)
    assert mask.tolist() == [True, True, False]
