"""FastHMR inference over already selected people, images and square boxes."""

from __future__ import annotations

import math
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from src.submodules.models.fasthmr.assets import (
    IMAGE_SIZE,
    WINDOW_LENGTH,
    FastHMRConfig,
)
from src.submodules.models.fasthmr.upstream import (
    OfficialSMPL,
    load_official_camera_models,
)
from src.utils.inference_recording import span
from src.utils.video import FramePacket, OpenCVVideoFrameReader

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)
_LEFT_JOINTS = (10, 7, 4, 1, 13, 16, 18, 20, 22, 25, 27)
_RIGHT_JOINTS = (11, 8, 5, 2, 14, 17, 19, 21, 23, 26, 28)
Frame: TypeAlias = NDArray[np.uint8]
SelectionRows: TypeAlias = tuple[NDArray[np.int64], NDArray[np.float32]]


@dataclass(frozen=True)
class FastHMRPrediction:
    rotations: NDArray[np.float32]  # (T,24,3,3), root followed by 23 local SMPL joints
    betas: NDArray[np.float32]  # (T,10)
    translation: NDArray[np.float32]  # (T,3), OpenCV camera axes, metres


def _letterbox_bgr(image: Frame) -> Frame:
    height, width = image.shape[:2]
    long_edge = max(height, width)
    resized_w, resized_h = int(IMAGE_SIZE * width / long_edge), int(IMAGE_SIZE * height / long_edge)
    left, top = (IMAGE_SIZE - resized_w) // 2, (IMAGE_SIZE - resized_h) // 2
    square: Frame = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    square[top:top + resized_h, left:left + resized_w] = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_AREA,
    )
    return square


def camera_translation(camera: Tensor, boxes: Tensor, size: tuple[int, int]) -> Tensor:
    """Convert the demo's crop camera using its diagonal focal length convention."""
    width, height = size
    scale = camera[:, 0:1] * boxes[:, 2:3]
    if not bool(torch.isfinite(camera).all()) or not bool((scale > 0).all()):
        raise ValueError("FastHMR camera requires finite parameters and positive scale")
    image_center = boxes.new_tensor((width / 2, height / 2))
    xy = camera[:, 1:] + 2 * (boxes[:, :2] - image_center) / scale
    depth = (2 * math.hypot(width, height)) / scale
    return torch.cat((xy, depth), dim=1)


def _smooth_crop_camera(camera: Tensor) -> Tensor:
    if len(camera) <= 6:
        return camera
    kernel = camera.new_full((3, 1, 5), 1 / 5)
    filtered = F.conv1d(camera.T[None], kernel, padding=2, groups=3)[0].T
    filtered[:3] = camera[:3]
    filtered[-3:] = camera[-3:]
    return filtered


def fuse_flip_pose(predicted: Tensor, flipped: Tensor) -> Tensor:
    """Retain the main prediction's twist and average the matched skeleton."""
    joints = predicted[:, :87].reshape(-1, 29, 3)
    mirrored = flipped[:, :87].reshape(-1, 29, 3).clone()
    mirrored[..., 0].neg_()
    mirrored[:, _LEFT_JOINTS + _RIGHT_JOINTS] = mirrored[:, _RIGHT_JOINTS + _LEFT_JOINTS]
    return torch.cat(((0.5 * (joints + mirrored)).flatten(1), predicted[:, 87:]), dim=1)


class FastHMR:
    def __init__(self, config: FastHMRConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.half = config.backend == "pt16"
        self.models: tuple[Any, Any, Any, Any, Any] | None = None

    def load(self) -> None:
        if self.half and self.device.type != "cuda":
            raise ValueError("FastHMR PT16 requires CUDA")
        self.models = load_official_camera_models(self.config)

    def unload(self) -> None:
        self.models = None

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def predict(
        self,
        video: Path,
        size: tuple[int, int],
        selections: Sequence[SelectionRows],
        body: OfficialSMPL,
    ) -> tuple[FastHMRPrediction, ...]:
        if self.models is None:
            raise RuntimeError("FastHMR models have not been loaded")
        if not selections:
            return ()
        by_segment: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        packets = iter(OpenCVVideoFrameReader(video, max_frames=int(selections[-1][0][-1]) + 1))
        with closing(cast(Generator[FramePacket[Frame], None, None], packets)) as reader:
            for frames, boxes in selections:
                features: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
                for offset in range(0, len(frames), self.config.feature_batch_size):
                    rows = slice(offset, offset + self.config.feature_batch_size)
                    with span("preprocess", operation="fasthmr_source_frames"):
                        images = self._source_frames(reader, frames[rows], size)
                    features.append(self._encode(images, boxes[rows], size))
                original, flipped, betas, camera = zip(*features, strict=True)
                by_segment.append((torch.cat(original), torch.cat(flipped), torch.cat(betas), torch.cat(camera)))

        betas = torch.cat([item[2] for item in by_segment]).float().mean(dim=0, keepdim=True)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(self.config.seed)
            else:
                torch.random.default_generator.manual_seed(self.config.seed)
            return tuple(
                self._decode(original, flipped, betas, camera, boxes, size, body)
                for (original, flipped, _, camera), (_, boxes) in zip(by_segment, selections, strict=True)
            )

    @staticmethod
    def _source_frames(reader: Iterator[FramePacket[Frame]], requested: NDArray[np.int64], size: tuple[int, int]) -> list[Frame]:
        images: list[Frame] = []
        for frame_index in requested:
            for packet in reader:
                if packet.index >= frame_index:
                    break
            else:
                raise ValueError(f"FastHMR video ended before selected frame {frame_index}")
            if packet.index != frame_index or packet.frame.shape != (size[1], size[0], 3) or packet.frame.dtype != np.uint8:
                raise ValueError(f"FastHMR source frame {frame_index} disagrees with its selection")
            images.append(packet.frame)
        return images

    def _encode(self, images: Sequence[Frame], boxes: NDArray[np.float32], size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert self.models is not None
        backbone, fov, _, segmentor, crop = self.models
        with span("preprocess", operation="fasthmr_rgb_crop_and_fov_input"):
            crops = [
                crop(image, cx, cy, side, side, IMAGE_SIZE, IMAGE_SIZE, False, 1., 0)[0][..., ::-1]
                for image, (cx, cy, side) in zip(images, boxes, strict=True)
            ]
            pixels = self._images(crops)
            # The release feeds BGR from OpenCV to HumanFoV.
            fov_images = self._images([_letterbox_bgr(image) for image in images])
            box = torch.from_numpy(np.array(boxes, copy=True)).to(self.device)
            mean = pixels.new_tensor(_IMAGE_MEAN)[None, :, None, None]
            std = pixels.new_tensor(_IMAGE_STD)[None, :, None, None]

        with span("inference", model="fasthmr_yolo11x_seg", batch_shape=list(pixels.shape), batch_items=len(images)):
            masks = self._person_masks(segmentor, pixels)
        with span("inference", model="fasthmr_humanfov", batch_shape=list(fov_images.shape), batch_items=len(images)), \
                torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            angles, _ = fov((fov_images - mean) / std)
        width, height = size
        focal = height / (2 * (angles[:, 1].float() / 2).tan())
        if not bool(torch.isfinite(focal).all()) or not bool((focal > 0).all()):
            raise ValueError("FastHMR HumanFoV produced an invalid focal length")
        intrinsic = torch.zeros((len(images), 3, 3), device=self.device, dtype=torch.float32)
        intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = focal
        intrinsic[:, 0, 2], intrinsic[:, 1, 2], intrinsic[:, 2, 2] = width / 2, height / 2, 1
        normalized = (pixels - mean) / std
        batch = {
            "img": normalized,
            "person_masks": masks,
            "box_center": box[:, :2],
            "box_size": box[:, 2],
            "img_size": box.new_tensor((height, width)).expand(len(images), 2),
            "cam_int": intrinsic,
        }
        with span("inference", model="fasthmr_camerahmr", batch_shape=list(pixels.shape), batch_items=len(images)), \
                torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            params, camera, _, original = backbone(batch)
            # The release flips the image and retains the original mask.
            _, _, _, flipped = backbone({**batch, "img": normalized.flip(-1)})
        return original.float(), flipped.float(), params["betas"].float(), camera.float()

    def _images(self, images: Sequence[Frame]) -> Tensor:
        return torch.from_numpy(np.stack(images)).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255

    def _person_masks(self, segmentor: Any, pixels: Tensor) -> Tensor:
        results = segmentor.predict(pixels, verbose=False, conf=0.1, imgsz=IMAGE_SIZE,
                                    device=str(self.device), half=self.half)
        if len(results) != len(pixels):
            raise ValueError("FastHMR segmentation count differs from the crop count")
        masks = torch.zeros((len(pixels), IMAGE_SIZE, IMAGE_SIZE), device=self.device)
        for row, result in enumerate(results):
            people = result.boxes.cls == 0
            if bool(people.any()):
                if result.masks is None or result.masks.data.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                    raise ValueError("FastHMR requires a segmentation mask in crop coordinates")
                masks[row] = result.masks.data[people].bool().any(dim=0)
        return masks

    def _decode(
        self, original: Tensor, flipped: Tensor, betas: Tensor, camera: Tensor,
        boxes: NDArray[np.float32], size: tuple[int, int], body: OfficialSMPL,
    ) -> FastHMRPrediction:
        assert self.models is not None
        diffusion = self.models[2]
        rotations = []
        shape = betas.expand(len(original), 10)
        for start in range(0, len(original), WINDOW_LENGTH):
            window = slice(start, start + WINDOW_LENGTH)
            features = original[window][None]
            with span("inference", model="fasthmr_diffusion", batch_shape=list(features.shape), batch_items=len(features[0])), \
                    torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
                pose = diffusion(features)[0]
                flip_pose = diffusion(flipped[window][None])[0]
            with span("postprocess", operation="fasthmr_hybrik"):
                fused = fuse_flip_pose(pose.float(), flip_pose.float())
                rotations.append(body.rotations(fused, shape[window]))
        translated = camera_translation(_smooth_crop_camera(camera), camera.new_tensor(boxes), size)
        return FastHMRPrediction(torch.cat(rotations).float().cpu().numpy(),
                                 shape.cpu().numpy(), translated.cpu().numpy())
