"""BBox-only PMPose adapter for the fixed Meiji comparison."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pmpose_assets import (
    LICENSE,
    SOURCE_COMMIT,
    WEIGHT_REVISION,
    WEIGHTS,
    verify_checkpoint,
    verify_source,
)
from pose_contract import PoseBatch, PoseInput, PoseStages

from src.utils.checksum import FileIntegrityError
from src.utils.inference_benchmark import (
    Backend,
    BatchStages,
    BenchmarkAdapter,
    ConditionUnavailable,
    Stage,
    Status,
)
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import TensorRTRunner, tensorrt_module
from src.utils.schema.player import COCO_KP_NAMES

INPUT_HW = (256, 192)
HEATMAP_HW = (64, 48)
NATIVE_JOINTS = 23
JOINT_ORDER = (
    *COCO_KP_NAMES,
    "thorax_mpii",
    "pelvis_mpii",
    "neck_mpii",
    "head_top_mpii",
    "neck_aic",
    "head_top_aic",
)
FLIP_INDICES = (
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
)
EXTRA_FIELDS = ("presence", "visibility", "expected_oks", "normalized_error")
INPUT_SPEC = {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, *INPUT_HW))}
HEATMAP_NAME = "heatmaps"
OUTPUT_SPEC = {
    HEATMAP_NAME: TensorSpec(dtype="float32", shape=(-1, NATIVE_JOINTS, *HEATMAP_HW)),
    **{
        name: TensorSpec(dtype="float16", shape=(-1, NATIVE_JOINTS, 1, 1))
        for name in EXTRA_FIELDS
    },
}
TensorOutputs = tuple[torch.Tensor, ...]
ArrayOutputs = tuple[NDArray[np.float32], ...]
Infer = Callable[[tuple[PoseInput, ...], PoseStages | BatchStages], PoseBatch]


def benchmark_settings(model_id: str, backend: Backend) -> dict[str, Any]:
    return {
        "source_commit": SOURCE_COMMIT,
        "weights_revision": WEIGHT_REVISION,
        "checkpoint_sha256": WEIGHTS[model_id].sha256,
        "license": LICENSE,
        "input_hw": list(INPUT_HW),
        "mask": "bbox_rectangle",
        "flip_test": True,
        "forwards_per_batch": 2,
        "autocast": False,
        "forward_dtype": "float32" if backend is Backend.PT32 else "float16",
        "sparsemax_dtype": "float32",
        "normalize_fuse_decode_dtype": "float32",
    }


def _bbox_contour(row: PoseInput) -> list[list[int]]:
    """External contour of the official demo's floor/ceil BBox rectangle."""
    height, width = row.frame_bgr.shape[:2]
    box = row.bbox_xyxy
    x1, y1 = np.clip(np.floor(box[:2]), 0, (width, height)).astype(int).tolist()
    x2, y2 = np.clip(np.ceil(box[2:]), 0, (width, height)).astype(int).tolist()
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(
            f"PMPose BBox does not yield a polygon inside the frame: {box}"
        )
    right, bottom = x2 - 1, y2 - 1
    return [[x1, y1, x1, bottom, right, bottom, right, y1]]


@dataclass(frozen=True)
class OfficialPipeline:
    config: Any
    api: Any
    meta: Mapping[str, Any]
    pipeline: Callable[[dict[str, Any]], Any]
    collate: Callable[[list[Any]], Any]
    preprocessor: Any
    decode: Callable[[NDArray[np.float32]], tuple[NDArray[Any], NDArray[Any]]]

    def preprocess(self, rows: tuple[PoseInput, ...]) -> tuple[torch.Tensor, list[Any]]:
        samples = [
            self.pipeline(
                {
                    **self.meta,
                    "img": row.frame_bgr,
                    "bbox": row.bbox_xyxy[None].copy(),
                    "bbox_score": np.ones(1, dtype=np.float32),
                    "segmentation": _bbox_contour(row),
                }
            )
            for row in rows
        ]
        batch = self.preprocessor(self.collate(samples), training=False)
        return batch["inputs"], batch["data_samples"]


def _verify_joint_schema(meta: Mapping[str, Any]) -> None:
    joint_ids = meta["keypoint_id2name"]
    if (
        len(joint_ids) != NATIVE_JOINTS
        or tuple(joint_ids.get(index) for index in range(NATIVE_JOINTS)) != JOINT_ORDER
        or tuple(meta["flip_indices"]) != FLIP_INDICES
    ):
        raise ValueError(
            "PMPose metadata differs from the pinned COCO17 and flip order"
        )


def load_official_pipeline(model_id: str, root: Path) -> OfficialPipeline:
    weight = WEIGHTS[model_id]
    root = root.resolve()
    verify_source(root)
    source_module = root / "mmpose" / "__init__.py"
    entry = str(root)
    sys.path.insert(0, entry)
    try:
        mmpose = importlib.import_module("mmpose")
        if mmpose.__file__ is None or Path(mmpose.__file__).resolve() != source_module:
            raise ImportError(
                f"mmpose was imported from {mmpose.__file__}, expected {source_module}"
            )
        api = importlib.import_module("mmpose.apis.inference")
        registry = importlib.import_module("mmpose.registry")
        mmengine = importlib.import_module("mmengine.config")
        dataset = importlib.import_module("mmengine.dataset")
        config = mmengine.Config.fromfile(
            root / "mmpose/configs/ProbMaskPose" / f"{Path(weight.filename).stem}.py"
        )
        train_meta = config.train_dataloader.dataset.metainfo
        train_meta.from_file = str(root / "mmpose" / train_meta.from_file)
        api.init_default_scope(config.default_scope)
        meta = api.dataset_meta_from_config(config)
        _verify_joint_schema(meta)
        if (
            not config.model.test_cfg.flip_test
            or config.model.test_cfg.shift_heatmap
            or config.model.head.freeze_oks
        ):
            raise ValueError("Unexpected PMPose flip or confidence configuration")
        transforms = dataset.Compose(config.test_dataloader.dataset.pipeline)
        preprocessor = registry.MODELS.build(config.model.data_preprocessor)
        codec = registry.KEYPOINT_CODECS.build(config.model.head.decoder)
        return OfficialPipeline(
            config,
            api,
            meta,
            transforms,
            dataset.pseudo_collate,
            preprocessor,
            codec.decode,
        )
    finally:
        sys.path.remove(entry)


class _Float32Sparsemax(torch.nn.Module):
    def __init__(self, sparsemax: torch.nn.Module) -> None:
        super().__init__()
        self.sparsemax = sparsemax

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.sparsemax(values.float()))


class _Forward(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> TensorOutputs:
        return cast(TensorOutputs, self.model(pixels, None, mode="tensor"))


def load_model(
    upstream: OfficialPipeline,
    checkpoint: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> _Forward:
    model = upstream.api.init_model(upstream.config, device="cpu")
    # The caller has verified the published SHA-256. MMEngine metadata uses
    # Python objects that are outside torch.load's tensor-only allowlist.
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    meta = weights.get("meta", {}).get("dataset_meta")
    if meta is not None:
        _verify_joint_schema(meta)
    model.load_state_dict(weights["state_dict"], strict=True, assign=True)
    del weights
    if dtype == torch.float16:
        # The probability-map projection sorts and accumulates over 3,072
        # bins. Keep the stock Sparsemax calculation in FP32 for both FP16 backends.
        model.head.normalize_layer = _Float32Sparsemax(model.head.normalize_layer)
    forward = _Forward(model.to(device=device, dtype=dtype))
    forward.eval()
    return forward


def _fuse(normal: TensorOutputs, flipped: TensorOutputs) -> TensorOutputs:
    # Apply the official left/right permutation to all five heads. Only the
    # spatial heatmaps are mirrored. Accumulate in FP32 for every backend.
    restored = (flipped[0].flip(-1), *flipped[1:])
    return tuple(
        (base.float() + flip[:, FLIP_INDICES].float()) * 0.5
        for base, flip in zip(normal, restored, strict=True)
    )


def _to_numpy(outputs: TensorOutputs) -> ArrayOutputs:
    return tuple(value.detach().cpu().numpy() for value in outputs)


def _decode(
    decode: Callable[[NDArray[np.float32]], tuple[NDArray[Any], NDArray[Any]]],
    outputs: ArrayOutputs,
    samples: list[Any],
) -> PoseBatch:
    heatmaps, *scalars = outputs
    n = len(samples)
    if heatmaps.shape != (n, NATIVE_JOINTS, *HEATMAP_HW):
        raise ValueError(f"Unexpected PMPose heatmaps {heatmaps.shape}")
    extras = {
        name: values.reshape(n, NATIVE_JOINTS)
        for name, values in zip(EXTRA_FIELDS, scalars, strict=True)
    }
    extras["normalized_error"] = extras["normalized_error"] / np.float32(
        np.hypot(*HEATMAP_HW)
    )
    peak = heatmaps.max(axis=(-1, -2))
    native: NDArray[np.float32] = np.full(
        (n, NATIVE_JOINTS, 2), np.nan, dtype=np.float32
    )
    for index, (maps, sample) in enumerate(zip(heatmaps, samples, strict=True)):
        valid = (peak[index] > 0) & np.isfinite(maps).all(axis=(-1, -2))
        if valid.any():
            crop_xy, _ = decode(maps[valid])
            meta = sample.metainfo
            center, scale = meta["input_center"], meta["input_scale"]
            native[index, valid] = (
                crop_xy[0] / meta["input_size"] * scale + center - 0.5 * scale
            )
    extras["heatmap_peak"] = peak
    extras["native_keypoints_xy"] = native
    joints = len(COCO_KP_NAMES)
    return PoseBatch(
        np.concatenate(
            (native[:, :joints], extras["expected_oks"][:, :joints, None]), axis=-1
        ),
        extras,
    )


def _infer(
    upstream: OfficialPipeline,
    forward: Callable[[torch.Tensor], TensorOutputs],
    backend: Backend,
    device: torch.device,
    dtype: torch.dtype,
) -> Infer:
    def infer(
        rows: tuple[PoseInput, ...], stages: PoseStages | BatchStages
    ) -> PoseBatch:
        if not rows:
            raise ValueError("PMPose requires at least one input row")
        cpu, metadata = stages.stage(
            Stage.PREPROCESS, lambda: upstream.preprocess(rows)
        )
        pixels = stages.stage(Stage.H2D, lambda: cpu.to(device=device, dtype=dtype))
        flipped = stages.stage(Stage.PREPROCESS, lambda: pixels.flip(-1), name="flip")
        measure = (
            stages.measure_forward
            if isinstance(stages, BatchStages)
            else stages.forward
        )
        normal = measure(backend, lambda: forward(pixels), name="original")
        mirror = measure(backend, lambda: forward(flipped), name="flipped")
        fused = stages.stage(
            Stage.POSTPROCESS, lambda: _fuse(normal, mirror), name="flip_restore"
        )
        outputs = stages.stage(Stage.D2H, lambda: _to_numpy(fused))
        return stages.stage(
            Stage.POSTPROCESS,
            lambda: _decode(upstream.decode, outputs, metadata),
            name="dark_udp",
        )

    return infer


def load_engine(model_id: str, engine: Path, device: torch.device) -> TensorRTRunner:
    bundle = EngineBundle.read(engine).for_model(
        model=model_id,
        settings=benchmark_settings(model_id, Backend.TRT16),
        sources={"checkpoint": WEIGHTS[model_id].sha256},
        inputs=INPUT_SPEC,
        outputs=OUTPUT_SPEC,
    )
    runner = TensorRTRunner(bundle, device)
    # Retain the detailed layer inspector built into the plan and turn off
    # TensorRT's runtime NVTX ranges for the timed forwards.
    runner._context.nvtx_verbosity = tensorrt_module().ProfilingVerbosity.NONE
    return runner


def make_adapter(
    *,
    model_id: str,
    backend: Backend,
    checkpoint: Path,
    device: str,
    engine: Path | None = None,
    source_root: Path | None = None,
) -> BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]:
    if model_id not in WEIGHTS:
        raise ValueError(f"Unknown PMPose model ID {model_id!r}")
    if backend is not Backend.TRT16 and engine is not None:
        raise ValueError("A TensorRT engine can only be supplied to TRT16")
    device_spec = torch.device(device)
    dtype = torch.float32 if backend is Backend.PT32 else torch.float16

    def load() -> Infer:
        if backend is Backend.TRT16 and engine is None:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, "Provide a PMPose TensorRT engine bundle"
            )
        if source_root is None:
            raise ConditionUnavailable(
                Status.UNAVAILABLE,
                "Provide a pinned BBoxMaskPose checkout as source_root",
            )
        try:
            verify_checkpoint(checkpoint, WEIGHTS[model_id])
            upstream = load_official_pipeline(model_id, source_root.resolve())
        except (
            OSError,
            ImportError,
            ValueError,
            AssertionError,
            FileIntegrityError,
            subprocess.CalledProcessError,
        ) as error:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, f"PMPose assets or upstream runtime: {error}"
            ) from error
        if engine is None:
            return _infer(
                upstream,
                load_model(upstream, checkpoint, dtype=dtype, device=device_spec),
                backend,
                device_spec,
                dtype,
            )
        try:
            runner = load_engine(model_id, engine, device_spec)
        except (OSError, ImportError, ValueError, FileIntegrityError) as error:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, f"PMPose TensorRT engine: {error}"
            ) from error

        def forward(pixels: torch.Tensor) -> TensorOutputs:
            try:
                runner.bundle.engine.profiles["pixels"].require(
                    tuple(pixels.shape), tensor="pixels"
                )
            except ValueError as error:
                raise ConditionUnavailable(Status.UNSUPPORTED, str(error)) from error
            raw = runner({"pixels": pixels})
            return tuple(raw[name] for name in OUTPUT_SPEC)

        return _infer(upstream, forward, backend, device_spec, dtype)

    return BenchmarkAdapter(
        backend,
        load,
        {"checkpoint": checkpoint},
        {} if engine is None else {"plan": engine / PLAN_FILE},
    )
