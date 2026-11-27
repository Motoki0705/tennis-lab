# The affine crop and SimCC decode are adapted from OpenMMLab MMPose (Apache-2.0).
# See ../../vendor/rtmpose/NOTICE.md and origin.json.
"""Top-down RTMPose-X with native COCO17 pixels and raw SimCC peak scores."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from src.submodules.configuration import InferencePrecision, require_absolute_path
from src.submodules.models._base.crops import iter_requested_frames
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models.vitpose.pose2d import (
    Pose2DFrameSequenceRequest,
    Pose2DRequest,
    Pose2DResult,
)
from src.submodules.people_model_assets import PEOPLE_MODEL_CHECKPOINTS
from src.submodules.vendor.rtmpose import network
from src.submodules.vendor.rtmpose.network import (
    HALPE_JOINTS,
    HEIGHT,
    SIMCC_SPLIT,
    WIDTH,
    RTMPoseXNetwork,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_recording import span
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP
from src.utils.video import OpenCVVideoFrameReader

MODEL_ID = "rtmpose-x"
SOURCE_COMMIT = "7bea17cdf362cdd731fc5e79b982e4b79be5cca9"
NETWORK_SOURCE = Path(network.__file__).resolve()
ONNX_OPSET = 17
SCORE_KIND = "raw_simcc_peak"
BBOX_PADDING = 1.25
HALPE_NAMES = (*COCO_KP_NAMES, "head", "neck", "hip", "left_big_toe", "right_big_toe",
               "left_small_toe", "right_small_toe", "left_heel", "right_heel")
HALPE_FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24)
_RGB_MEAN = np.array([123.675, 116.28, 103.53], np.float32)[:, None, None]
_RGB_STD = np.array([58.395, 57.12, 57.375], np.float32)[:, None, None]
_DST_POINTS = np.array([(WIDTH * .5, HEIGHT * .5), (WIDTH * .5, (HEIGHT - WIDTH) * .5),
                        (0, (HEIGHT - WIDTH) * .5)], np.float32)
INPUTS = {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, HEIGHT, WIDTH))}
OUTPUTS = {
    "simcc_x": TensorSpec(dtype="float16", shape=(-1, HALPE_JOINTS, WIDTH * SIMCC_SPLIT)),
    "simcc_y": TensorSpec(dtype="float16", shape=(-1, HALPE_JOINTS, HEIGHT * SIMCC_SPLIT)),
}
EXPORT_SETTINGS = {"mmpose_commit": SOURCE_COMMIT, "opset": ONNX_OPSET, "graph": "CSPNeXt-X and RTMCC raw SimCC"}


@dataclass(frozen=True)
class RTMPoseBoxSequenceRequest:
    """Ordered BGR frames with native rectangular xyxy detection boxes."""

    frames_bgr: Iterable[tuple[int, NDArray[np.uint8]]]
    boxes_xyxy: Tensor
    frame_indices: Tensor


def _validate_halpe(metadata: dict[str, Any]) -> None:
    if (metadata["keypoint_id2name"] != dict(enumerate(HALPE_NAMES))
        or tuple(metadata["flip_indices"]) != HALPE_FLIP):
        raise ValueError("RTMPose-X checkpoint Halpe26 names or flip indices differ")


def load_rtmpose_x(checkpoint: Path) -> RTMPoseXNetwork:
    PEOPLE_MODEL_CHECKPOINTS[MODEL_ID].verify(checkpoint)
    reconstruct = import_module("numpy._core.multiarray")._reconstruct
    with torch.serialization.safe_globals([
        (reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray,
        np.dtype, np.dtypes.Float32DType, np.dtypes.UInt8DType,
    ]):
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _validate_halpe(weights["meta"]["dataset_meta"])
    model = RTMPoseXNetwork()
    model.load_state_dict(weights["state_dict"], strict=True)
    model.eval()
    return model


def prepare_rtmpose_crop(frame: NDArray[np.uint8], box: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply MMPose's zero-rotation affine crop and RGB normalization."""
    start, end = box[:2], box[2:]
    center = (end + start) * np.float32(0.5)
    w, h = (end - start) * np.float32(BBOX_PADDING)
    aspect = np.float32(WIDTH / HEIGHT)
    scale = np.array((w, w / aspect) if w > h * aspect else (h * aspect, h), np.float32)
    src: NDArray[np.float32] = np.empty((3, 2), np.float32)
    src[0] = center
    src[1] = center.astype(np.float64) + np.array([0, -0.5 * float(scale[0])])
    delta = src[0] - src[1]
    src[2] = src[1] + np.array([-delta[1], delta[0]], np.float32)
    warp = cv2.getAffineTransform(src, _DST_POINTS)
    crop = cv2.warpAffine(frame, warp, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    rgb = np.ascontiguousarray(crop[..., ::-1].transpose(2, 0, 1), dtype=np.float32)
    return (rgb - _RGB_MEAN) / _RGB_STD, np.concatenate((center, scale))


def iter_rtmpose_crops(request: RTMPoseBoxSequenceRequest, *, batch_size: int) -> Generator[tuple[Tensor, Tensor], None, None]:
    """Yield prepared pixels and (cx, cy, width, height) in request order."""
    boxes = request.boxes_xyxy
    if boxes.shape != (len(request.frame_indices), 4) or boxes.device.type != "cpu" or not boxes.is_floating_point():
        raise ValueError("RTMPose-X requires one CPU floating xyxy box per requested frame row")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("RTMPose-X batch size must be a positive integer")
    array = boxes.float().numpy()
    if not np.isfinite(array).all() or (array[:, 2:] <= array[:, :2]).any():
        raise ValueError("RTMPose-X xyxy boxes must be finite with positive area")
    crops: list[NDArray[np.float32]] = []
    geometry: list[NDArray[np.float32]] = []
    for image, rows in iter_requested_frames(request.frames_bgr, request.frame_indices):
        for box in array[rows]:
            with span("preprocess", model="RTMPose-X", operation="person_crop"):
                pixels, meta = prepare_rtmpose_crop(image, box)
            crops.append(pixels)
            geometry.append(meta)
            if len(crops) == batch_size:
                yield torch.from_numpy(np.stack(crops)), torch.from_numpy(np.stack(geometry))
                crops, geometry = [], []
    if crops:
        yield torch.from_numpy(np.stack(crops)), torch.from_numpy(np.stack(geometry))


def decode_simcc(x: NDArray[np.float32], y: NDArray[np.float32], geometry: NDArray[np.float32]) -> Tensor:
    expected = (len(geometry), HALPE_JOINTS)
    if x.shape != (*expected, WIDTH * SIMCC_SPLIT) or y.shape != (*expected, HEIGHT * SIMCC_SPLIT):
        raise ValueError(f"Unexpected RTMPose-X SimCC shapes: {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite RTMPose-X SimCC logits")
    x, y = x[:, :NUM_HUMAN_KP], y[:, :NUM_HUMAN_KP]
    scores = np.minimum(x.max(axis=-1), y.max(axis=-1))
    crop_xy = np.stack((x.argmax(axis=-1), y.argmax(axis=-1)), axis=-1).astype(np.float32) / SIMCC_SPLIT
    center, scale = geometry[:, None, :2], geometry[:, None, 2:]
    xy = crop_xy / np.array([WIDTH, HEIGHT], np.float64) * scale + center - 0.5 * scale
    # A nonpositive peak has no native location. The finite out-of-image
    # sentinel supplies that information to existing scene and tracking consumers.
    xy[scores <= 0] = -1
    return torch.from_numpy(np.concatenate((xy.astype(np.float32), scores[..., None]), axis=-1))


def _xyxy_from_squares(boxes: Tensor) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 3:
        raise ValueError("RTMPose-X square crop requests require (cx, cy, size)")
    centers, radius = boxes[:, :2], boxes[:, 2:] * .5
    return torch.cat((centers - radius, centers + radius), dim=1)


class RTMPoseXPose2D(BaseInferenceModel[Pose2DRequest | Pose2DFrameSequenceRequest | RTMPoseBoxSequenceRequest, Pose2DResult]):
    """RTMPose-X with explicit FP32, FP16 weights or typed TensorRT execution."""

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device, batch_size: int,
                 precision: InferencePrecision = "fp32", flip_test: bool = True,
                 show_progress: bool = True, engine: EngineBundle | None = None) -> None:
        super().__init__(device)
        self.checkpoint = require_absolute_path(checkpoint, name="RTMPose-X checkpoint")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("RTMPose-X batch size must be a positive integer")
        if type(flip_test) is not bool or type(show_progress) is not bool:
            raise TypeError("RTMPose-X flip and progress settings must be boolean")
        if precision not in {"fp32", "fp16"} or (engine is not None and precision != "fp16"):
            raise ValueError("RTMPose-X supports PT32, PT16 and TRT16")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("RTMPose-X FP16 requires CUDA")
        self.batch_size, self.flip_test, self.show_progress = batch_size, flip_test, show_progress
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self._engine = engine
        self._runner: TensorRTRunner | None = None
        self._network: RTMPoseXNetwork | None = None

    def _load_impl(self) -> None:
        if self._engine is None:
            self._network = load_rtmpose_x(self.checkpoint).to(device=self.device, dtype=self.dtype)
            return
        checkpoint = PEOPLE_MODEL_CHECKPOINTS[MODEL_ID]
        checkpoint.verify(self.checkpoint)
        bundle = self._engine.for_model(
            model=MODEL_ID, settings=EXPORT_SETTINGS,
            sources={"checkpoint": checkpoint.sha256, "network": dual_sha256(NETWORK_SOURCE)},
            inputs=INPUTS, outputs=OUTPUTS,
        )
        profile = bundle.engine.profiles["pixels"]
        for batch in (1, self.batch_size):
            profile.require((batch, 3, HEIGHT, WIDTH), tensor="pixels")
        self._runner = TensorRTRunner(bundle, self.device)

    def _unload_impl(self) -> None:
        self._runner = None
        self._network = None

    def recording_assets(self) -> dict[str, Path]:
        return {"checkpoint": self.checkpoint} if self._engine is None else {"engine": self._engine.root / PLAN_FILE}

    def _predict_impl(self, request: Pose2DRequest | Pose2DFrameSequenceRequest | RTMPoseBoxSequenceRequest) -> Pose2DResult:
        if isinstance(request, Pose2DRequest):
            stream = ((row.index, row.frame) for row in OpenCVVideoFrameReader(Path(request.video_path)))
            indices = torch.arange(len(request.bbx_xys), dtype=torch.int64) if request.frame_indices is None else request.frame_indices
            request = RTMPoseBoxSequenceRequest(stream, _xyxy_from_squares(request.bbx_xys), indices)
        elif isinstance(request, Pose2DFrameSequenceRequest):
            request = RTMPoseBoxSequenceRequest(request.frames_bgr, _xyxy_from_squares(request.bbx_xys), request.frame_indices)
        crops = iter_rtmpose_crops(request, batch_size=self.batch_size)
        batches = tqdm(crops, desc="RTMPose-X") if self.show_progress else crops
        results = [self._predict_prepared(pixels, geometry) for pixels, geometry in batches]
        return Pose2DResult(torch.cat(results, dim=0) if results else torch.empty((0, NUM_HUMAN_KP, 3), dtype=torch.float32))

    def predict_crops(self, images: Tensor, geometry: Tensor) -> Pose2DResult:
        """Infer prepared RTMPose-X RGB crops with their center and scale metadata."""
        self.load()
        with torch.no_grad():
            return Pose2DResult(self._predict_prepared(images, geometry))

    def _predict_prepared(self, images: Tensor, geometry: Tensor) -> Tensor:
        if (images.ndim != 4 or images.shape[1:] != (3, HEIGHT, WIDTH) or images.dtype != torch.float32
            or geometry.shape != (len(images), 4) or geometry.device.type != "cpu" or geometry.dtype != torch.float32
            or not bool(torch.isfinite(geometry).all()) or bool((geometry[:, 2:] <= 0).any()) or len(images) == 0):
            raise ValueError("Invalid prepared RTMPose-X pixel batch or crop geometry")
        with span("transfer", model="RTMPose-X", operation="input_to_device"):
            # Match the reference NCHW layout so FP16 selects the same convolution kernels.
            pixels = images.to(device=self.device, dtype=self.dtype, memory_format=torch.contiguous_format)
        with span("inference", model="RTMPose-X", batch_shape=list(pixels.shape), flip_test=self.flip_test):
            x, y = self._forward(pixels)
            if self.flip_test:
                flipped_x, flipped_y = self._forward(pixels.flip(-1))
                x = (x.float() + flipped_x[:, HALPE_FLIP].flip(-1).float()) * 0.5
                y = (y.float() + flipped_y[:, HALPE_FLIP].float()) * 0.5
        with span("postprocess", model="RTMPose-X", operation="decode_simcc"):
            return decode_simcc(x.float().cpu().numpy(), y.float().cpu().numpy(), geometry.numpy())

    def _forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if self._runner is not None:
            outputs = self._runner({"pixels": images})
            return outputs["simcc_x"], outputs["simcc_y"]
        if self._network is None:
            raise RuntimeError("RTMPose-X model did not load before prediction")
        with torch.autocast(self.device.type, enabled=False):
            return cast(tuple[Tensor, Tensor], self._network(images))
