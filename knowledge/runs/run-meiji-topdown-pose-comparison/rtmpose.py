"""The pinned MMPose RTMPose-M/X graphs and shared SimCC pose adapter."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pose_contract import PoseBatch, PoseInput, PoseStages
from pydantic import JsonValue

from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    BenchmarkAdapter,
    ConditionUnavailable,
    Stage,
    Status,
)
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import TensorRTRunner, tensorrt_module
from src.utils.schema.player import COCO_KP_NAMES

SOURCE_COMMIT = "7bea17cdf362cdd731fc5e79b982e4b79be5cca9"
HALPE = Path("configs/_base_/datasets/halpe26.py")
WIDTH, HEIGHT = 288, 384
HALPE_JOINTS = 26
COCO_JOINTS = len(COCO_KP_NAMES)
SIMCC_SPLIT = 2
ONNX_OPSET = 17


@dataclass(frozen=True)
class ModelSpec:
    config: str
    checkpoint_name: str
    sha256: str
    parameters: int


MODELS = {
    "rtmpose-m": ModelSpec(
        "rtmpose-m_8xb512-700e_body8-halpe26-384x288.py",
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth",
        "89e6428b38901b5003d0f753619e94d269999e1f5a922349f9068430387863f2",
        14_056_356,
    ),
    "rtmpose-x": ModelSpec(
        "rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
        "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth",
        "7fb6e239601082a06ca8442feb7a9774b8d77a1c37911b2ad0f27c1e84be4d21",
        49_998_772,
    ),
}

EXPORT_SETTINGS: dict[str, JsonValue] = {
    "mmpose_commit": SOURCE_COMMIT,
    "opset": ONNX_OPSET,
    "graph": "MMPose tensor forward with raw SimCC outputs",
    "precision": "fp16",
}
INPUTS = {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, HEIGHT, WIDTH))}
OUTPUTS = {
    "simcc_x": TensorSpec(
        dtype="float16", shape=(-1, HALPE_JOINTS, WIDTH * SIMCC_SPLIT)
    ),
    "simcc_y": TensorSpec(
        dtype="float16", shape=(-1, HALPE_JOINTS, HEIGHT * SIMCC_SPLIT)
    ),
}

FloatArray: TypeAlias = NDArray[np.float32]
SimCC = tuple[torch.Tensor, torch.Tensor]
Decode = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]
Transform = Callable[[dict[str, Any]], dict[str, Any]]
Infer = Callable[[tuple[PoseInput, ...], PoseStages], PoseBatch]


@dataclass(frozen=True)
class Recipe:
    checkpoint: Path
    config_path: Path
    config: Any
    metainfo: Mapping[str, Any]


def _validate_joint_order(names: Mapping[int, str]) -> None:
    if (
        set(names) != set(range(HALPE_JOINTS))
        or tuple(names[joint] for joint in range(COCO_JOINTS)) != COCO_KP_NAMES
    ):
        raise ConditionUnavailable(
            Status.UNSUPPORTED, "Halpe26 order differs from COCO17"
        )


def load_recipe(model_id: str, checkpoint: Path, source_root: Path | None) -> Recipe:
    spec = MODELS[model_id]
    if not checkpoint.is_file() or dual_sha256(checkpoint) != spec.sha256:
        raise ConditionUnavailable(
            Status.UNAVAILABLE,
            f"{model_id} requires the official checkpoint with SHA-256 {spec.sha256}: {checkpoint}",
        )
    if source_root is None:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, "Supply the pinned MMPose source root"
        )
    root = source_root.resolve()
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        module = import_module("mmpose")
        imported = Path(module.__file__ or "").resolve()
        if head != SOURCE_COMMIT or dirty or imported != root / "mmpose/__init__.py":
            raise ConditionUnavailable(
                Status.UNAVAILABLE,
                f"Expected clean MMPose {SOURCE_COMMIT} on PYTHONPATH from {root}; imported {imported}",
            )
        config_path = root / "configs/body_2d_keypoint/rtmpose/body8" / spec.config
        config = import_module("mmengine.config").Config.fromfile(config_path)
        # This revision includes OpenMMLab's own CSPNeXt port. Select the local
        # registry explicitly and resolve the metadata from the pinned checkout.
        config.model.backbone._scope_ = "mmpose"
        config.train_dataloader.dataset.metainfo.from_file = str(root / HALPE)
        metainfo = import_module("mmpose.datasets.datasets.utils").parse_pose_metainfo(
            {"from_file": str(root / HALPE)}
        )
    except (
        AssertionError,
        ImportError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"MMPose setup failed: {error}"
        ) from error
    _validate_joint_order(metainfo["keypoint_id2name"])
    return Recipe(checkpoint, config_path, config, metainfo)


def load_estimator(recipe: Recipe) -> Any:
    """Load the published tensors strictly and verify their Halpe metadata."""
    reconstruct = import_module("numpy._core.multiarray")._reconstruct
    numpy_globals = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        np.dtypes.Float32DType,
        np.dtypes.UInt8DType,
    ]
    with torch.serialization.safe_globals(numpy_globals):
        weights = torch.load(recipe.checkpoint, map_location="cpu", weights_only=True)
    metadata = weights["meta"]["dataset_meta"]
    # The released checkpoints call the dataset "halpe". Joint names and flip
    # indices match the config's "halpe26" definition.
    for name in ("keypoint_id2name", "flip_indices"):
        if metadata[name] != recipe.metainfo[name]:
            raise ConditionUnavailable(
                Status.UNSUPPORTED, f"Checkpoint Halpe26 {name} mismatch"
            )
    model = import_module("mmpose.apis").init_model(recipe.config, device="cpu")
    model.load_state_dict(weights["state_dict"], strict=True)
    return model


class RTMPoseGraph(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> SimCC:
        return cast(SimCC, self.model(pixels, None))


def load_tensorrt(
    model_id: str, recipe: Recipe, engine: Path, device: torch.device
) -> TensorRTRunner:
    try:
        bundle = EngineBundle.read(engine).for_model(
            model=model_id,
            settings=EXPORT_SETTINGS,
            sources={
                "checkpoint": MODELS[model_id].sha256,
                "config": dual_sha256(recipe.config_path),
            },
            inputs=INPUTS,
            outputs=OUTPUTS,
        )
        runner = TensorRTRunner(bundle, device)
        # Keep inspector metadata on the plan and disable per-launch NVTX events.
        runner._context.nvtx_verbosity = tensorrt_module().ProfilingVerbosity.NONE
        return runner
    except torch.OutOfMemoryError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"RTMPose TensorRT engine: {error}"
        ) from error


def _to_native_coco17(
    crop_xy: FloatArray,
    scores: FloatArray,
    centers: FloatArray,
    scales: FloatArray,
    finite_axes: NDArray[np.bool_],
) -> PoseBatch:
    # The official SimCC decoder returns negative crop coordinates for missing
    # peaks. Keep the raw score and mark the missing native coordinates as NaN.
    crop_xy = crop_xy[:, :COCO_JOINTS]
    scores = scores[:, :COCO_JOINTS]
    coords = crop_xy / np.array((WIDTH, HEIGHT), dtype=np.float64)
    coords = coords * scales[:, None] + centers[:, None] - 0.5 * scales[:, None]
    valid = (
        finite_axes[:, :COCO_JOINTS] & (crop_xy >= 0).all(axis=-1) & np.isfinite(scores)
    )
    result: FloatArray = np.empty((*crop_xy.shape[:2], 3), dtype=np.float32)
    result[..., :2] = np.where(valid[..., None], coords, np.nan)
    result[..., 2] = np.where(np.isfinite(scores), scores, np.nan)
    return PoseBatch(result)


@dataclass(frozen=True)
class Preprocess:
    center: Transform
    affine: Transform
    mean: FloatArray
    std: FloatArray

    def __call__(
        self, rows: Sequence[PoseInput]
    ) -> tuple[torch.Tensor, FloatArray, FloatArray]:
        if not rows:
            raise ValueError("RTMPose requires at least one crop")
        pixels: FloatArray = np.empty((len(rows), 3, HEIGHT, WIDTH), np.float32)
        centers: FloatArray = np.empty((len(rows), 2), np.float32)
        scales = np.empty_like(centers)
        for index, row in enumerate(rows):
            sample = self.affine(
                self.center({"img": row.frame_bgr, "bbox": row.bbox_xyxy[None]})
            )
            pixels[index] = sample["img"][..., ::-1].transpose(2, 0, 1)
            centers[index] = sample["input_center"]
            scales[index] = sample["input_scale"]
        pixels -= self.mean
        pixels /= self.std
        return torch.from_numpy(pixels), centers, scales


def _make_infer(
    recipe: Recipe,
    backend: Backend,
    forward: Callable[[torch.Tensor], SimCC],
    device: torch.device,
) -> Infer:
    transforms = import_module("mmpose.datasets.transforms")
    preprocess_cfg = recipe.config.model.data_preprocessor
    preprocess = Preprocess(
        transforms.GetBBoxCenterScale(),
        transforms.TopdownAffine(input_size=(WIDTH, HEIGHT)),
        np.array(preprocess_cfg.mean, dtype=np.float32).reshape(3, 1, 1),
        np.array(preprocess_cfg.std, dtype=np.float32).reshape(3, 1, 1),
    )
    codec = import_module("mmpose.registry").KEYPOINT_CODECS.build(recipe.config.codec)
    decode = cast(Decode, codec.decode)
    flip_vectors = import_module("mmpose.models.utils.tta").flip_vectors
    flip_indices = recipe.metainfo["flip_indices"]
    dtype = torch.float32 if backend is Backend.PT32 else torch.float16

    def fuse(original: SimCC, flipped: SimCC) -> SimCC:
        x, y = flip_vectors(*flipped, flip_indices=flip_indices)
        return (original[0].float() + x.float()) * 0.5, (
            original[1].float() + y.float()
        ) * 0.5

    def restore(
        axes: tuple[FloatArray, FloatArray], centers: FloatArray, scales: FloatArray
    ) -> PoseBatch:
        coordinates, scores = decode(*axes)
        finite_axes = np.isfinite(axes[0]).all(axis=-1) & np.isfinite(axes[1]).all(
            axis=-1
        )
        return _to_native_coco17(coordinates, scores, centers, scales, finite_axes)

    def infer(rows: tuple[PoseInput, ...], stages: PoseStages) -> PoseBatch:
        cpu_pixels, centers, scales = stages.stage(
            Stage.PREPROCESS, lambda: preprocess(rows)
        )
        pixels = stages.stage(
            Stage.H2D, lambda: cpu_pixels.to(device=device, dtype=dtype)
        )
        flipped = stages.stage(
            Stage.PREPROCESS, lambda: pixels.flip(-1), name="horizontal_flip"
        )
        original_axes = stages.measure_forward(
            backend, lambda: forward(pixels), name="original"
        )
        flipped_axes = stages.measure_forward(
            backend, lambda: forward(flipped), name="horizontal_flip"
        )
        x, y = stages.stage(
            Stage.POSTPROCESS,
            lambda: fuse(original_axes, flipped_axes),
            name="flip_fusion",
        )
        arrays = stages.stage(
            Stage.D2H, lambda: (x.detach().cpu().numpy(), y.detach().cpu().numpy())
        )
        return stages.stage(
            Stage.POSTPROCESS,
            lambda: restore(arrays, centers, scales),
            name="simcc_decode",
        )

    return infer


def make_adapter(
    *,
    model_id: str,
    backend: Backend,
    checkpoint: Path,
    device: str,
    engine: Path | None = None,
    source_root: Path | None = None,
) -> BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]:
    if model_id not in MODELS:
        raise ValueError(f"Unknown RTMPose model: {model_id}")
    if backend is not Backend.TRT16 and engine is not None:
        raise ValueError("An RTMPose engine is only used by TRT16")

    def load() -> Infer:
        if backend is Backend.TRT16 and engine is None:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, "Supply an RTMPose TensorRT bundle"
            )
        target = torch.device(device)
        if target.type != "cuda" and (
            target.type != "cpu" or backend is not Backend.PT32
        ):
            raise ConditionUnavailable(
                Status.UNSUPPORTED, f"RTMPose {backend} requires CUDA"
            )
        recipe = load_recipe(model_id, checkpoint, source_root)
        if engine is not None:
            runner = load_tensorrt(model_id, recipe, engine, target)

            def forward(pixels: torch.Tensor) -> SimCC:
                try:
                    axes = runner({"pixels": pixels})
                except ValueError as error:
                    raise ConditionUnavailable(
                        Status.UNSUPPORTED, str(error)
                    ) from error
                return axes["simcc_x"], axes["simcc_y"]

            return _make_infer(recipe, backend, forward, target)

        dtype = torch.float32 if backend is Backend.PT32 else torch.float16
        graph = (
            RTMPoseGraph(load_estimator(recipe)).eval().to(device=target, dtype=dtype)
        )
        return _make_infer(recipe, backend, graph.forward, target)

    engines = {} if engine is None else {"plan": engine / PLAN_FILE}
    return BenchmarkAdapter(backend, load, {"checkpoint": checkpoint}, engines)
