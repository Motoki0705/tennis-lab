"""Use pinned EdgeCrafter code without installing upstream training components."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.ops import box_convert

from src.submodules.models.ecpose.releases import (
    IMAGE_SIZE,
    SOURCE_COMMIT,
    EdgeCrafterRelease,
)
from src.utils.checksum import dual_sha256
from src.utils.schema.player import NUM_HUMAN_KP

_LOAD_LOCK = Lock()
_NORMALIZE = transforms.Compose(
    [
        transforms.Resize(
            IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def require_source(repository: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repository.resolve():
        raise ValueError(f"Expected the EdgeCrafter checkout root: {repository}")
    if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain"):
        raise ValueError(
            f"EdgeCrafter requires clean source at {SOURCE_COMMIT}: {repository}"
        )


def require_weight(checkpoint: Path, release: EdgeCrafterRelease) -> None:
    if (
        checkpoint.stat().st_size != release.size
        or dual_sha256(checkpoint) != release.sha256
    ):
        raise ValueError(
            f"EdgeCrafter weight does not match {release.filename}: {checkpoint}"
        )


def _yaml_config(engine_dir: Path) -> Any:
    package_name = f"_tennis_lab_edgecrafter_{engine_dir.parent.name}"
    existing = sys.modules.get(package_name)
    if existing is None:
        # Import the model registry through a namespace. ECPose's regular
        # engine initializer imports its training datasets and globally aliases
        # pycocotools to faster_coco_eval, which inference does not use.
        spec = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        spec.submodule_search_locations = [str(engine_dir)]
        sys.modules[package_name] = importlib.util.module_from_spec(spec)
        pillow_max_pixels = Image.MAX_IMAGE_PIXELS
        try:
            importlib.import_module(f"{package_name}.edgecrafter")
        except BaseException:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            raise
        finally:
            # The detection registry imports an upstream dataset module that
            # changes Pillow's process-wide image size limit on import.
            Image.MAX_IMAGE_PIXELS = pillow_max_pixels
    elif list(existing.__path__) != [str(engine_dir)]:
        raise ValueError(
            f"An EdgeCrafter model is already loaded from {list(existing.__path__)}"
        )
    return importlib.import_module(f"{package_name}.core").YAMLConfig


class EdgeCrafterNetwork(nn.Module):
    """Return only the two heads used by postprocessing, including in ONNX."""

    def __init__(self, model: nn.Module, has_pose: bool) -> None:
        super().__init__()
        self.model = model
        self.coordinate_output = "pred_keypoints" if has_pose else "pred_boxes"

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(pixels)
        return output["pred_logits"], output[self.coordinate_output]


def load_network(
    repository: Path, checkpoint: Path, release: EdgeCrafterRelease
) -> EdgeCrafterNetwork:
    """Strictly load the published EMA or model state into the deployed model."""
    repository = repository.resolve()
    require_source(repository)
    require_weight(checkpoint, release)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["ema"]["module"] if "ema" in payload else payload["model"]
    family = repository / release.family
    with _LOAD_LOCK:
        cfg = _yaml_config(family / "engine")(str(family / "configs" / release.config))
        if tuple(cfg.yaml_cfg["eval_spatial_size"]) != IMAGE_SIZE:
            raise ValueError("Unexpected EdgeCrafter input size")
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
        model = cfg.model
        model.load_state_dict(state, strict=True)
        model = model.deploy()
    network = EdgeCrafterNetwork(model, release.has_pose)
    network.float()
    network.eval()
    return network


def preprocess_images(
    frames_bgr: Sequence[NDArray[np.uint8]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official normalized RGB NCHW pixels and each native (width, height)."""
    pixels, sizes = [], []
    for frame in frames_bgr:
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError(f"Expected a nonempty HWC BGR image, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise TypeError(f"Expected uint8 BGR input, got {frame.dtype}")
        sizes.append((frame.shape[1], frame.shape[0]))
        pixels.append(
            _NORMALIZE(Image.fromarray(np.ascontiguousarray(frame[..., ::-1])))
        )
    if not pixels:
        raise ValueError("EdgeCrafter requires at least one image per batch")
    return torch.stack(pixels), torch.tensor(sizes, dtype=torch.float32)


@dataclass(frozen=True)
class EdgeCrafterDetections:
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    keypoints_xyc: torch.Tensor | None


def validate_decode_policy(
    release: EdgeCrafterRelease,
    confidence: float,
    pose_box_margin: float | None,
) -> None:
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Person confidence cutoff must be in [0, 1]")
    if release.has_pose:
        if (
            pose_box_margin is None
            or not math.isfinite(pose_box_margin)
            or pose_box_margin < 0
        ):
            raise ValueError(
                "ECPose requires an explicit nonnegative joint-envelope margin"
            )
    elif pose_box_margin is not None:
        raise ValueError("A joint-envelope margin only applies to ECPose")


def decode_predictions(
    release: EdgeCrafterRelease,
    output: tuple[torch.Tensor, torch.Tensor],
    sizes_wh: torch.Tensor,
    *,
    confidence: float,
    pose_box_margin: float | None = None,
) -> list[EdgeCrafterDetections]:
    """Decode top queries into aligned native-pixel boxes, scores and optional poses."""
    validate_decode_policy(release, confidence, pose_box_margin)
    logits, coordinates = output
    num_queries, num_classes, person_label = (
        (60, 2, 1) if release.has_pose else (300, 80, 0)
    )
    batch = logits.shape[0]
    coordinate_shape = (NUM_HUMAN_KP, 2) if release.has_pose else (4,)
    if (
        logits.shape != (batch, num_queries, num_classes)
        or coordinates.shape != (batch, num_queries, *coordinate_shape)
        or sizes_wh.shape != (batch, 2)
    ):
        raise ValueError("Unexpected EdgeCrafter batch or output head shape")
    scores, indices = logits.float().sigmoid().flatten(1).topk(num_queries, dim=1)
    labels = indices % num_classes
    query_indices = indices // num_classes
    results = []
    for image, wh in enumerate(sizes_wh):
        image_scores = scores[image]
        people = (labels[image] == person_label) & (image_scores > confidence)
        people_coordinates = coordinates[image, query_indices[image, people]].float()
        keypoints = None
        if pose_box_margin is not None:
            xy = people_coordinates * wh
            present = torch.isfinite(xy).all(dim=-1) & ((xy >= 0) & (xy < wh)).all(
                dim=-1
            )
            lower = xy.masked_fill(~present[..., None], torch.inf).amin(dim=1)
            upper = xy.masked_fill(~present[..., None], -torch.inf).amax(dim=1)
            eligible = (present.sum(dim=1) >= 2) & (upper > lower).all(dim=-1)
            lower, upper = lower[eligible], upper[eligible]
            padding = (upper - lower) * pose_box_margin
            boxes = torch.cat((lower - padding, upper + padding), dim=-1)
            # The upstream COCO evaluation uses availability 1 for each joint.
            xy = xy[eligible].masked_fill(~present[eligible, :, None], 0.0)
            keypoints = torch.cat((xy, present[eligible, :, None].to(xy.dtype)), dim=-1)
        else:
            boxes = box_convert(people_coordinates, "cxcywh", "xyxy") * wh.repeat(2)
            eligible = torch.isfinite(boxes).all(dim=-1)
            boxes = boxes[eligible]
        boxes = torch.minimum(boxes.clamp_min(0), wh.repeat(2))
        positive_area = (boxes[:, 2:] > boxes[:, :2]).all(dim=-1)
        results.append(
            EdgeCrafterDetections(
                boxes[positive_area],
                image_scores[people][eligible][positive_area],
                keypoints[positive_area] if keypoints is not None else None,
            )
        )
    return results
