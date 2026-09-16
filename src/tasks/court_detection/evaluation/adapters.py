"""CPU inference adapters that feed one shared keypoint contract.

``ours`` loads the repo checkpoint through the production predictor.  ``tcd``
loads the external yastrebksv/TennisCourtDetector code and weights from the
paths the operator passes in: the upstream repository has no licence, so no
file is copied into this repo, vendored, or committed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.evaluation.contracts import (
    KEYPOINT_COUNT,
    KeypointPrediction,
    ModelPrediction,
)
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelOutput,
)
from src.utils.configuration import PathResolver, RuntimePathRoots

# Upstream pre-processing constants.  The network input is a 640x360 canvas and
# its output heatmaps live on the same 640x360 grid, so a predicted point is
# mapped to the original frame with ``x * width / 640`` (never a fixed ``* 2``).
TCD_INPUT_WIDTH = 640
TCD_INPUT_HEIGHT = 360
TCD_HEATMAP_SCALE = 1
TCD_HEATMAP_CHANNELS = 15
# Upstream trains keypoints 0..13 plus one extra court-centre map, and the
# centre map is the last channel.  Only the first 14 channels are keypoints.
TCD_CENTER_CHANNEL_INDEX = TCD_HEATMAP_CHANNELS - 1
TCD_CENTER_CHANNEL_ROLE = "court_center_excluded_from_keypoint_metrics"

_REQUIRED_PYTHON_MODULES = ("tracknet.py", "postprocess.py")

# Implementation files whose content defines this benchmark's inference.  They
# are hashed into every model fingerprint so a code change invalidates cached
# predictions instead of silently mixing two behaviours.
_OURS_INFERENCE_SOURCES = (Path(__file__).resolve(),)

# The repo's single-image predictor resizes so the *short* side reaches
# ``spec.short_side``, while a pose-safe checkpoint trains and validates with an
# isotropic *long*-side resize (``CourtProcessingGeometry(require_pose=True)``).
# ``training_geometry`` reproduces the latter exactly; ``predictor_default``
# keeps the former.  The choice is recorded in the run provenance and in the
# model fingerprint, so cached predictions never mix the two.
Preprocessing = Literal["training_geometry", "predictor_default"]


class KeypointModelAdapter(Protocol):
    """One benchmarked model behind a model-agnostic prediction contract."""

    name: str

    def predict(self, image_rgb: NDArray[np.uint8]) -> ModelPrediction: ...

    def provenance(self) -> Mapping[str, object]: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(paths: object) -> dict[str, str]:
    """Hash a small, explicit set of implementation files by relative path."""
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("source_sha256 requires a non-empty sequence of paths.")
    hashes: dict[str, str] = {}
    for value in paths:
        path = Path(cast("str | Path", value))
        if not path.is_file():
            raise FileNotFoundError(f"Implementation source is missing: {path}")
        hashes[path.name] = sha256_file(path)
    return hashes


def benchmark_adapter_source() -> dict[str, str]:
    """Return the benchmark inference-implementation hashes for this checkout."""
    return source_sha256(_OURS_INFERENCE_SOURCES)


def require_cpu(device: str | torch.device) -> torch.device:
    """Reject every non-CPU device before any model or data work starts."""
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ValueError(
            f"The court-alignment benchmark is CPU-only; requested device {resolved!r}."
        )
    return resolved


class CourtCheckpointAdapter:
    """The repo's DINOv3 + DPT multi-head checkpoint as a keypoint model."""

    name = "ours"

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        project_root: Path,
        device: str | torch.device = "cpu",
        max_peaks: int = 1,
        subpixel_refine: bool = False,
        preprocessing: Preprocessing = "training_geometry",
    ) -> None:
        if max_peaks <= 0:
            raise ValueError("ours max_peaks must be positive.")
        if preprocessing not in ("training_geometry", "predictor_default"):
            raise ValueError(f"Unsupported preprocessing mode: {preprocessing!r}.")
        self.device = require_cpu(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root)
        self.max_peaks = int(max_peaks)
        self.subpixel_refine = bool(subpixel_refine)
        self.preprocessing = preprocessing
        self.config = checkpoint_config(
            self.checkpoint_path, project_root=self.project_root
        )
        roots = RuntimePathRoots.from_mapping(
            _infer_roots_payload(self.config), repository_root=self.project_root
        )
        self.predictor = CourtKeypointPredictor.load_from_checkpoint(
            self.checkpoint_path,
            resolver=PathResolver(roots),
            device=self.device,
            subpixel_refine=self.subpixel_refine,
            max_peaks=self.max_peaks,
            config=self.config,
        )
        self._checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self._adapter_source_sha256 = benchmark_adapter_source()
        self._code_commit = _git_commit(self.project_root)
        self._model_io_adapter = self.predictor.adapter
        self._model = self.predictor.model
        spec = self._model_io_adapter.spec
        self._short_side = int(spec.short_side)
        augmentation = self.config.data.augmentation
        self._patch_size = int(augmentation.patch_size)
        self._pose_safe = bool(self.config.loss.pose.enabled)
        if self.preprocessing == "training_geometry" and not self._pose_safe:
            raise ValueError(
                "training_geometry preprocessing reproduces the pose-safe "
                "long-side resize contract, but this checkpoint does not enable "
                "pose supervision; pass preprocessing='predictor_default'."
            )

    def predict(self, image_rgb: NDArray[np.uint8]) -> ModelPrediction:
        start = time.perf_counter()
        image = _require_rgb(image_rgb, name="ours")
        if self.preprocessing == "training_geometry":
            decoded = self._predict_training_geometry(image)
        else:
            decoded = self.predictor.predict(image)
        elapsed = time.perf_counter() - start
        keypoints = decoded.keypoints[:, 0, :].detach().cpu().numpy()
        valid = decoded.valid[:, 0].detach().cpu().numpy().astype(bool)
        scores = decoded.scores[:, 0].detach().cpu().numpy()
        return ModelPrediction(
            model="ours",
            keypoints=KeypointPrediction(
                keypoints_xy=_masked_points(keypoints, valid),
                scores=scores.astype(np.float64),
                valid=valid,
            ),
            elapsed_seconds=elapsed,
            extras={
                "preprocessing": self.preprocessing,
                "resize_side": self._short_side,
                "pose_safe_geometry": self._pose_safe,
                "max_peaks": self.max_peaks,
                "subpixel_refine": self.subpixel_refine,
            },
        )

    def _predict_training_geometry(
        self, image: NDArray[np.uint8]
    ) -> CourtKeypointPrediction:
        """Reproduce the checkpoint's pose-safe validation geometry exactly.

        The training contract resizes isotropically so the *long* side equals the
        configured size, pads right/bottom to a patch multiple with edge
        replication, and normalizes with the ImageNet statistics.  Decoded
        coordinates are mapped back through the same scale factor.
        """
        import cv2
        import torchvision.transforms.functional as functional

        height, width = image.shape[:2]
        long_side = self._short_side
        scale = long_side / float(max(width, height))
        content_width = max(1, int(round(width * scale)))
        content_height = max(1, int(round(height * scale)))
        matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        warped = cv2.warpPerspective(
            image,
            matrix,
            (content_width, content_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        padded_width = content_width + (-content_width) % self._patch_size
        padded_height = content_height + (-content_height) % self._patch_size
        if (padded_width, padded_height) != (content_width, content_height):
            warped = cv2.copyMakeBorder(
                warped,
                0,
                padded_height - content_height,
                0,
                padded_width - content_width,
                borderType=cv2.BORDER_REPLICATE,
            )
        from PIL import Image

        tensor = functional.normalize(
            functional.to_tensor(Image.fromarray(warped, mode="RGB")),
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
        images = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            call = self._model_io_adapter.prepare_images(images)
            output = self._model(*call.model_args)
            self._model_io_adapter.validate_logits(output, call)
            logits = (
                output.dense_logits if isinstance(output, CourtModelOutput) else output
            )
            decoded = decode_padded_keypoint_logits(
                self._model_io_adapter,
                logits["kp"],
                content_size_hw=(content_height, content_width),
                subpixel_refine=self.subpixel_refine,
                max_peaks=self.max_peaks,
            )
        return CourtKeypointPrediction(
            keypoints=decoded.keypoints / scale,
            scores=decoded.scores,
            valid=decoded.valid,
            heatmaps=decoded.heatmaps,
        )

    def provenance(self) -> Mapping[str, object]:
        return {
            "model": self.name,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
            "config_source": "checkpoint hyper_parameters",
            "project_root_override": str(self.project_root),
            # Repo code identity: the adapter implementation plus the checkout
            # the checkpoint config was replayed against.
            "adapter_source_sha256": self._adapter_source_sha256,
            "code_root": str(self.project_root),
            "code_commit": self._code_commit,
            "preprocessing": self.preprocessing,
            "resize_side": self._short_side,
            "pose_safe_geometry": self._pose_safe,
            "patch_size": self._patch_size,
            "max_peaks": self.max_peaks,
            "subpixel_refine": self.subpixel_refine,
        }


class TennisCourtDetectorAdapter:
    """The upstream yastrebksv/TennisCourtDetector baseline on CPU."""

    name = "tcd"

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        low_thresh: int = 155,
        min_radius: int = 10,
        max_radius: int = 30,
        input_width: int = TCD_INPUT_WIDTH,
        input_height: int = TCD_INPUT_HEIGHT,
    ) -> None:
        self.device = require_cpu(device)
        self.repo_path = Path(repo_path).resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        missing = [
            name
            for name in _REQUIRED_PYTHON_MODULES
            if not (self.repo_path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"External baseline repo is missing {missing} under {self.repo_path}."
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"External baseline checkpoint is missing: {self.checkpoint_path}"
            )
        self.low_thresh = int(low_thresh)
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.input_width = int(input_width)
        self.input_height = int(input_height)

        self._tracknet = _load_external_module(self.repo_path, "tracknet.py")
        self._postprocess = _load_external_module(self.repo_path, "postprocess.py")
        model_factory = getattr(self._tracknet, "BallTrackerNet", None)
        if model_factory is None:
            raise AttributeError("External tracknet.py must expose BallTrackerNet.")
        if not hasattr(self._postprocess, "postprocess"):
            raise AttributeError("External postprocess.py must expose postprocess.")
        model = cast(
            "torch.nn.Module", model_factory(out_channels=TCD_HEATMAP_CHANNELS)
        )
        if getattr(model, "out_channels", None) != TCD_HEATMAP_CHANNELS:
            raise ValueError(
                "External baseline must declare "
                f"out_channels={TCD_HEATMAP_CHANNELS} (14 keypoints plus the "
                "court-centre map), got "
                f"{getattr(model, 'out_channels', None)!r}."
            )
        state = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        self._checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self._adapter_source_sha256 = benchmark_adapter_source()
        self._external_source_sha256 = source_sha256(
            [self.repo_path / name for name in _REQUIRED_PYTHON_MODULES]
        )
        self._repo_commit = _git_commit(self.repo_path)

    def predict(self, image_rgb: NDArray[np.uint8]) -> ModelPrediction:
        import cv2
        import torch.nn.functional as functional

        start = time.perf_counter()
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("TCD predictions require an H x W x 3 RGB image.")
        resized = cv2.resize(image, (self.input_width, self.input_height))
        # Upstream reads frames with cv2.imread, so the network expects BGR.
        bgr = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(np.rollaxis(bgr, 2, 0)))
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
            heatmaps = functional.sigmoid(logits)[0].detach().cpu().numpy()
        expected_shape = (
            TCD_HEATMAP_CHANNELS,
            self.input_height,
            self.input_width,
        )
        if heatmaps.shape != expected_shape:
            raise ValueError(
                "External baseline output contract changed: expected "
                f"(channels, height, width)={expected_shape} after removing the "
                f"batch dimension, got {tuple(heatmaps.shape)}. The upstream "
                "network must emit 15 heatmaps on the 640x360 grid with the "
                "court-centre map in the last channel."
            )
        height, width = image.shape[:2]
        points: NDArray[np.float64] = np.full(
            (KEYPOINT_COUNT, 2), np.nan, dtype=np.float64
        )
        scores: NDArray[np.float64] = np.zeros(KEYPOINT_COUNT, dtype=np.float64)
        valid: NDArray[np.bool_] = np.zeros(KEYPOINT_COUNT, dtype=bool)
        for channel in range(KEYPOINT_COUNT):
            heatmap = (heatmaps[channel] * 255.0).astype(np.uint8)
            x_pred, y_pred = self._postprocess.postprocess(
                heatmap,
                scale=TCD_HEATMAP_SCALE,
                low_thresh=self.low_thresh,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
            )
            if x_pred is None or y_pred is None:
                continue
            points[channel, 0] = float(x_pred) * width / float(self.input_width)
            points[channel, 1] = float(y_pred) * height / float(self.input_height)
            scores[channel] = float(heatmaps[channel].max())
            valid[channel] = bool(np.isfinite(points[channel]).all())
        elapsed = time.perf_counter() - start
        return ModelPrediction(
            model="tcd",
            keypoints=KeypointPrediction(
                keypoints_xy=_masked_points(points, valid),
                scores=scores,
                valid=valid,
            ),
            elapsed_seconds=elapsed,
            extras={
                "preprocessing": "resize_640x360_BGR_div255",
                "postprocess": {
                    "low_thresh": self.low_thresh,
                    "min_radius": self.min_radius,
                    "max_radius": self.max_radius,
                    "heatmap_scale": TCD_HEATMAP_SCALE,
                    "method": "cv2.HoughCircles on thresholded sigmoid heatmap",
                },
                "coordinate_normalization": ("x * width / 640, y * height / 360"),
                "score_definition": "sigmoid_heatmap_channel_max",
                "center_channel_index": TCD_CENTER_CHANNEL_INDEX,
            },
        )

    def provenance(self) -> Mapping[str, object]:
        return {
            "model": self.name,
            "repo_path": str(self.repo_path),
            "repo_commit": self._repo_commit,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
            "input_size": [self.input_width, self.input_height],
            "expected_output_shape": [
                TCD_HEATMAP_CHANNELS,
                self.input_height,
                self.input_width,
            ],
            "heatmap_channels_used": KEYPOINT_COUNT,
            "center_channel_index": TCD_CENTER_CHANNEL_INDEX,
            "center_channel_role": TCD_CENTER_CHANNEL_ROLE,
            "adapter_source_sha256": self._adapter_source_sha256,
            "external_source_sha256": self._external_source_sha256,
            "postprocess": {
                "low_thresh": self.low_thresh,
                "min_radius": self.min_radius,
                "max_radius": self.max_radius,
                "heatmap_scale": TCD_HEATMAP_SCALE,
            },
            "licence": (
                "external repository has no licence; code and weights are "
                "referenced in place and never copied into this repo"
            ),
        }


def _masked_points(
    points: NDArray[np.floating], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (KEYPOINT_COUNT, 2):
        raise ValueError(f"Predictions must have shape ({KEYPOINT_COUNT}, 2).")
    mask = np.asarray(valid, dtype=bool)[:, None]
    masked = np.where(mask, array, np.nan)
    if np.any(valid & ~np.isfinite(masked).all(axis=1)):
        raise ValueError("A valid prediction must be finite.")
    return masked


def _require_rgb(image: NDArray[np.uint8], *, name: str) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} predictions require a uint8 H x W x 3 RGB image.")
    return array


def decode_padded_keypoint_logits(
    model_io_adapter: CourtModelIOAdapter,
    keypoint_logits: torch.Tensor,
    *,
    content_size_hw: tuple[int, int],
    subpixel_refine: bool,
    max_peaks: int,
) -> CourtKeypointPrediction:
    """Decode KP predictions from patch-padded logits.

    The pose-safe geometry replicates a few border pixels so the input is an
    integral DINOv3 patch grid.  Those padded columns/rows must be removed before
    peak extraction: ``heatmaps_to_peaks`` normalizes coordinates by the heatmap
    it is given, so decoding padded logits against a content-sized target both
    rescales every coordinate and lets a padded (edge-replicated) peak win the
    arg-max.  The production qualitative renderer crops the same way before
    decoding.
    """
    if keypoint_logits.ndim != 4 or keypoint_logits.shape[0] != 1:
        raise ValueError(
            "Court KP logits must have shape (1, C, H, W), got "
            f"{tuple(keypoint_logits.shape)}."
        )
    content_height, content_width = content_size_hw
    if content_height <= 0 or content_width <= 0:
        raise ValueError("Court KP content size must be positive.")
    padded_height, padded_width = keypoint_logits.shape[-2:]
    if content_height > padded_height or content_width > padded_width:
        raise ValueError(
            "Court KP content size must fit inside the padded logits, got "
            f"content=({content_height}, {content_width}) "
            f"padded=({padded_height}, {padded_width})."
        )
    cropped = keypoint_logits[:, :, :content_height, :content_width]
    decoded = model_io_adapter.decode_prediction(
        "kp",
        cropped,
        original_size_hw=(content_height, content_width),
        subpixel_refine=subpixel_refine,
        max_peaks=max_peaks,
    )
    if not isinstance(decoded, CourtKeypointPrediction):
        raise TypeError("Court KP decoding must return a keypoint prediction.")
    return decoded


def checkpoint_config(checkpoint_path: Path, *, project_root: Path) -> DictConfig:
    """Replay the exact serialized training config with explicit roots."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Court checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "hyper_parameters" not in payload:
        raise ValueError("Court checkpoint does not publish hyper_parameters.")
    hyper = payload["hyper_parameters"]
    if not isinstance(hyper, Mapping) or "config" not in hyper:
        raise ValueError("Court checkpoint does not publish a training config.")
    config = OmegaConf.create(OmegaConf.to_container(hyper["config"], resolve=False))
    del payload
    if not isinstance(config, DictConfig):
        raise ValueError("Court checkpoint config must be a mapping.")
    config.paths.project_root = str(project_root)
    config.paths.external_asset_root = str(project_root / "third_party")
    return config


def checkpoint_training_config(
    checkpoint_path: Path, *, project_root: Path
) -> CourtTrainingConfig:
    """Validate that a checkpoint's serialized config replays under repo code."""
    return CourtTrainingConfig.from_config(
        checkpoint_config(checkpoint_path, project_root=project_root)
    )


def _infer_roots_payload(config: DictConfig) -> Mapping[str, str]:
    paths = config.get("paths")
    if paths is None:
        raise ValueError("Court checkpoint config must publish paths.")
    return {str(key): str(value) for key, value in paths.items()}


@contextmanager
def _import_path(directory: Path) -> Iterator[None]:
    """Temporarily expose one directory as the first import location."""
    entry = str(directory)
    sys.path.insert(0, entry)
    saved: dict[str, ModuleType] = {}
    for name in ("utils", "postprocess", "tracknet"):
        module = sys.modules.pop(name, None)
        if module is not None:
            saved[name] = module
    try:
        yield
    finally:
        if sys.path and sys.path[0] == entry:
            sys.path.pop(0)
        for name in ("utils", "postprocess", "tracknet"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _load_external_module(repo_path: Path, filename: str) -> ModuleType:
    """Import one file from the external repo without leaving modules behind."""
    module_path = repo_path / filename
    with _import_path(repo_path):
        spec = importlib.util.spec_from_file_location(
            f"_court_benchmark_tcd_{module_path.stem}", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import external module {module_path}.")
        module = importlib.util.module_from_spec(spec)
        module_name = spec.name
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != module_path.resolve():
            raise ImportError(
                f"External module {filename} did not load from {repo_path}."
            )
        sys.modules.pop(module_name, None)
    return module


def _git_commit(repo_path: Path) -> str | None:
    head = repo_path / ".git"
    if not head.exists():
        return None
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


__all__ = [
    "TCD_CENTER_CHANNEL_INDEX",
    "TCD_CENTER_CHANNEL_ROLE",
    "TCD_HEATMAP_CHANNELS",
    "TCD_HEATMAP_SCALE",
    "TCD_INPUT_HEIGHT",
    "TCD_INPUT_WIDTH",
    "CourtCheckpointAdapter",
    "KeypointModelAdapter",
    "Preprocessing",
    "TennisCourtDetectorAdapter",
    "benchmark_adapter_source",
    "checkpoint_config",
    "checkpoint_training_config",
    "decode_padded_keypoint_logits",
    "require_cpu",
    "sha256_file",
    "source_sha256",
]
