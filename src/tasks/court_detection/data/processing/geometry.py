"""One sampled geometry plan shared by RGB and every selected Court target."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ColorJitter, GaussianBlur
from torchvision.transforms import functional as TF

from src.tasks.court_detection.configuration import CourtAugmentationConfig
from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtInstance2D,
    CourtKeypointChannels,
    CourtRawSample,
    CourtTransformedSample,
)
from src.tasks.court_detection.geometry.pose import (
    build_pose_target,
    validate_projection_round_trip,
)
from src.utils.geometry.affine import build_centered_affine_matrix


@dataclass(frozen=True, slots=True)
class CourtGeometryPlan:
    """A source-pixel to output-pixel homography sampled exactly once."""

    matrix: Tensor
    output_size_hw: tuple[int, int]
    horizontal_flipped: bool

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3) or not self.matrix.is_floating_point():
            raise ValueError("Court geometry matrix must be floating [3,3].")
        if not bool(torch.isfinite(self.matrix).all()):
            raise ValueError("Court geometry matrix must be finite.")
        if self.output_size_hw[0] <= 0 or self.output_size_hw[1] <= 0:
            raise ValueError("Court output geometry must be positive.")


class CourtProcessingGeometry:
    """Sample one transform and apply it to every geometric payload."""

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        config: CourtAugmentationConfig,
        *,
        is_train: bool,
        require_pose: bool = False,
    ) -> None:
        self.config = config
        self.is_train = is_train
        self.require_pose = require_pose
        if require_pose and not config.preserve_fx_fy:
            raise ValueError(
                "Court pose geometry requires augmentation preserve_fx_fy=true."
            )
        self.color_jitter = ColorJitter(*config.color_jitter)

    def sample(self, raw: CourtRawSample) -> CourtGeometryPlan:
        """Return one accepted plan; retries only choose that single final plan."""
        attempts = self.config.visibility_max_retries if self.is_train else 1
        fallback: CourtGeometryPlan | None = None
        for _ in range(attempts):
            candidate = self._sample_once(raw.image.size)
            fallback = candidate
            if self._has_required_visibility(raw, candidate):
                return candidate
        if fallback is None:  # pragma: no cover - positive retry count is validated
            raise RuntimeError("Court geometry did not sample a candidate.")
        return fallback

    def apply(
        self,
        raw: CourtRawSample,
        *,
        dense_targets: Mapping[CourtDenseTargetKind, Tensor],
        plan: CourtGeometryPlan | None = None,
    ) -> CourtTransformedSample:
        """Apply the same plan to RGB, points, instances, and dense targets."""
        selected = self.sample(raw) if plan is None else plan
        height, width = selected.output_size_hw
        matrix = selected.matrix.detach().cpu().numpy().astype(np.float64)

        image_array = np.asarray(raw.image, dtype=np.uint8)
        warped = cv2.warpPerspective(
            image_array,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        image = Image.fromarray(warped, mode="RGB")
        if self.is_train:
            image = self.color_jitter(image)
            if random.random() < self.config.gaussian_blur_prob:
                kernel = random.choice(self.config.gaussian_blur_kernel)
                sigma = random.uniform(*self.config.gaussian_blur_sigma)
                image = GaussianBlur(kernel_size=kernel, sigma=sigma)(image)
        image_tensor = TF.normalize(TF.to_tensor(image), self._MEAN, self._STD)

        transformed_dense: dict[CourtDenseTargetKind, Tensor] = {}
        for kind, target in dense_targets.items():
            transformed_dense[kind] = self._warp_dense(
                target,
                matrix=matrix,
                output_size_hw=selected.output_size_hw,
                preserve_binary_coverage=kind == "line",
            )

        channels = (
            None
            if raw.keypoint_channels is None
            else self._transform_channels(
                raw.keypoint_channels,
                matrix=selected.matrix,
                output_size_hw=selected.output_size_hw,
                horizontal_flipped=selected.horizontal_flipped,
            )
        )
        instances = tuple(
            self._transform_instance(
                instance,
                matrix=selected.matrix,
                output_size_hw=selected.output_size_hw,
            )
            for instance in raw.court_instances
        )
        pose_target = None
        if self.require_pose:
            if raw.pose_authority is None:
                raise ValueError("Court pose geometry received no typed V3 authority.")
            if channels is None or channels.points_xy.shape != (14, 1, 2):
                raise ValueError(
                    "Court query pose geometry requires singleton target-court KP14."
                )
            pose_target = build_pose_target(
                raw.pose_authority,
                source_to_output=selected.matrix,
            )
            if not torch.equal(
                channels.physical_indices[:, 0],
                pose_target.semantic_to_physical,
            ):
                raise ValueError(
                    "Court V3 KP14 physical order disagrees with pose authority."
                )
            validate_projection_round_trip(
                pose_target,
                channels.points_xy[:, 0],
            )
        return CourtTransformedSample(
            sample_id=raw.sample_id,
            image_tensor=image_tensor,
            image_size=torch.tensor([height, width], dtype=torch.long),
            keypoint_channels=channels,
            court_instances=instances,
            dense_targets=MappingProxyType(transformed_dense),
            horizontal_flipped=selected.horizontal_flipped,
            metadata=raw.metadata,
            pose_target=pose_target,
        )

    def _sample_once(self, image_size_wh: tuple[int, int]) -> CourtGeometryPlan:
        width, height = image_size_wh
        if self.require_pose:
            # Pose-safe query inputs preserve the source aspect ratio.  The target
            # size is the long side; only the minimal right/bottom patch alignment
            # is added so DINOv3/16 receives an integral patch grid.  In particular,
            # do not letterbox every portrait frame into a square canvas.
            long_side = (
                random.choice(self.config.train_scales)
                if self.is_train
                else self.config.val_short_side
            )
            scale = long_side / float(max(width, height))
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            padded_width = resized_width + (-resized_width) % self.config.patch_size
            padded_height = resized_height + (-resized_height) % self.config.patch_size
            matrix = torch.tensor(
                [
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )
            return CourtGeometryPlan(
                matrix,
                (padded_height, padded_width),
                False,
            )
        if not self.is_train:
            short_side = self.config.val_short_side
            scale = short_side / float(min(width, height))
            out_width = max(1, int(round(width * scale)))
            out_height = max(1, int(round(height * scale)))
            matrix = torch.tensor(
                [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            )
            return CourtGeometryPlan(matrix, (out_height, out_width), False)

        output = random.choice(self.config.train_scales)
        top, left, crop_height, crop_width = self._random_resized_crop(height, width)
        crop_resize = np.array(
            [
                [output / float(crop_width), 0.0, -left * output / float(crop_width)],
                [0.0, output / float(crop_height), -top * output / float(crop_height)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        flipped = random.random() < self.config.hflip_prob
        if flipped:
            flip = np.array(
                [[-1.0, 0.0, float(output - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            matrix = flip @ crop_resize
        else:
            matrix = crop_resize

        translate_x = random.uniform(
            -self.config.affine_translate[0], self.config.affine_translate[0]
        ) * output
        translate_y = random.uniform(
            -self.config.affine_translate[1], self.config.affine_translate[1]
        ) * output
        affine = build_centered_affine_matrix(
            width=output,
            height=output,
            rotation_degrees=random.uniform(
                -self.config.affine_degrees, self.config.affine_degrees
            ),
            translate=(translate_x, translate_y),
            scale=random.uniform(*self.config.affine_scale),
            shear_degrees=random.uniform(
                -self.config.affine_shear, self.config.affine_shear
            ),
            shear_mode="torchvision",
            dtype=np.float64,
        )
        matrix = affine @ matrix

        if random.random() < self.config.perspective_prob:
            amount = self.config.perspective_distortion * float(output) * 0.5
            source = np.array(
                [[0.0, 0.0], [output - 1.0, 0.0], [output - 1.0, output - 1.0], [0.0, output - 1.0]],
                dtype=np.float32,
            )
            destination = source + np.array(
                [
                    [random.uniform(0.0, amount), random.uniform(0.0, amount)],
                    [random.uniform(-amount, 0.0), random.uniform(0.0, amount)],
                    [random.uniform(-amount, 0.0), random.uniform(-amount, 0.0)],
                    [random.uniform(0.0, amount), random.uniform(-amount, 0.0)],
                ],
                dtype=np.float32,
            )
            perspective = cv2.getPerspectiveTransform(source, destination).astype(
                np.float64
            )
            matrix = perspective @ matrix

        return CourtGeometryPlan(
            matrix=torch.from_numpy(matrix),
            output_size_hw=(output, output),
            horizontal_flipped=flipped,
        )

    def _random_resized_crop(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = float(height * width)
        log_ratio = tuple(math.log(value) for value in self.config.crop_ratio)
        for _ in range(10):
            target_area = area * random.uniform(*self.config.crop_scale)
            ratio = math.exp(random.uniform(*log_ratio))
            crop_width = int(round(math.sqrt(target_area * ratio)))
            crop_height = int(round(math.sqrt(target_area / ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = random.randint(0, height - crop_height)
                left = random.randint(0, width - crop_width)
                return top, left, crop_height, crop_width
        source_ratio = width / float(height)
        if source_ratio < self.config.crop_ratio[0]:
            crop_width = width
            crop_height = int(round(crop_width / self.config.crop_ratio[0]))
        elif source_ratio > self.config.crop_ratio[1]:
            crop_height = height
            crop_width = int(round(crop_height * self.config.crop_ratio[1]))
        else:
            crop_height, crop_width = height, width
        return (
            max(0, (height - crop_height) // 2),
            max(0, (width - crop_width) // 2),
            crop_height,
            crop_width,
        )

    def _has_required_visibility(
        self, raw: CourtRawSample, plan: CourtGeometryPlan
    ) -> bool:
        channels = raw.keypoint_channels
        if channels is None or self.config.min_visible_kp == 0:
            return True
        transformed = self._transform_channels(
            channels,
            matrix=plan.matrix,
            output_size_hw=plan.output_size_hw,
            horizontal_flipped=plan.horizontal_flipped,
        )
        source_visible: int = int(channels.point_visible.sum().item())
        required: int = min(self.config.min_visible_kp, source_visible)
        transformed_visible: int = int(transformed.point_visible.sum().item())
        return transformed_visible >= required

    @staticmethod
    def _transform_points(points: Tensor, matrix: Tensor) -> Tensor:
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError("Court geometry points must have shape [...,2].")
        if not points.is_floating_point():
            raise TypeError("Court geometry points must use a floating dtype.")
        if matrix.shape != (3, 3) or not matrix.is_floating_point():
            raise ValueError("Court geometry matrix must be floating [3,3].")
        if points.device != matrix.device:
            raise ValueError("Court geometry points and matrix must share one device.")
        if not bool(torch.isfinite(points).all()) or not bool(
            torch.isfinite(matrix).all()
        ):
            raise ValueError("Court geometry points and matrix must be finite.")
        original_shape = points.shape
        flat = points.to(dtype=torch.float64).reshape(-1, 2)
        ones = torch.ones(
            (flat.shape[0], 1),
            dtype=torch.float64,
            device=points.device,
        )
        homogeneous = torch.cat((flat, ones), dim=1)
        transformed = homogeneous @ matrix.to(dtype=torch.float64).T
        denominator = transformed[:, 2:3]
        if bool(torch.any(torch.isclose(denominator, torch.zeros_like(denominator)))):
            raise ValueError("Court geometry produced a zero homogeneous denominator.")
        return (transformed[:, :2] / denominator).reshape(original_shape).to(
            dtype=points.dtype
        )

    @classmethod
    def _transform_channels(
        cls,
        channels: CourtKeypointChannels,
        *,
        matrix: Tensor,
        output_size_hw: tuple[int, int],
        horizontal_flipped: bool,
    ) -> CourtKeypointChannels:
        points = cls._transform_points(channels.points_xy, matrix)
        height, width = output_size_hw
        visible = (
            channels.point_visible
            & (points[..., 0] >= 0.0)
            & (points[..., 0] <= float(width - 1))
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= float(height - 1))
        )
        physical = channels.physical_indices
        names = channels.channel_names
        if horizontal_flipped:
            permutation = torch.tensor(
                channels.horizontal_flip_permutation, dtype=torch.long
            )
            points = points.index_select(0, permutation)
            visible = visible.index_select(0, permutation)
            physical = physical.index_select(0, permutation)
        return CourtKeypointChannels(
            channel_names=names,
            points_xy=points,
            point_visible=visible,
            physical_indices=physical,
            horizontal_flip_permutation=channels.horizontal_flip_permutation,
        )

    @classmethod
    def _transform_instance(
        cls,
        instance: CourtInstance2D,
        *,
        matrix: Tensor,
        output_size_hw: tuple[int, int],
    ) -> CourtInstance2D:
        points = cls._transform_points(instance.points_xy, matrix)
        height, width = output_size_hw
        visible = (
            instance.point_visible
            & (points[:, 0] >= 0.0)
            & (points[:, 0] <= float(width - 1))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= float(height - 1))
        )
        return CourtInstance2D(
            court_instance_id=instance.court_instance_id,
            physical_indices=instance.physical_indices,
            points_xy=points,
            point_in_front=instance.point_in_front,
            point_visible=visible,
        )

    @staticmethod
    def _warp_dense(
        target: Tensor,
        *,
        matrix: np.ndarray,
        output_size_hw: tuple[int, int],
        preserve_binary_coverage: bool,
    ) -> Tensor:
        height, width = output_size_hw
        array = target.detach().cpu().numpy()
        had_channel = array.ndim == 3
        if had_channel:
            if array.shape[0] != 1:
                raise ValueError("Dense Court targets must be [H,W] or [1,H,W].")
            array = array[0]
        warped = cv2.warpPerspective(
            array,
            matrix,
            (width, height),
            flags=(cv2.INTER_LINEAR if preserve_binary_coverage else cv2.INTER_NEAREST),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if preserve_binary_coverage:
            warped = (warped > 0).astype(array.dtype)
        tensor = torch.from_numpy(np.ascontiguousarray(warped)).to(dtype=target.dtype)
        return tensor.unsqueeze(0) if had_channel else tensor


__all__ = ["CourtGeometryPlan", "CourtProcessingGeometry"]
