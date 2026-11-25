"""Sapiens2 top-down COCO17 adapter for the fixed-box benchmark."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pose_contract import PoseBatch, PoseInput, PoseStages

from src.utils.checksum import dual_sha256
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.inference_benchmark import (
    Backend,
    BenchmarkAdapter,
    ConditionUnavailable,
    Stage,
    Status,
)
from src.utils.models.tensorrt_artifacts import (
    ENGINE_MANIFEST,
    EXPORT_MANIFEST,
    PLAN_FILE,
    EngineBundle,
    TensorSpec,
)
from src.utils.models.tensorrt_runtime import TensorRTRunner, tensorrt_module
from src.utils.schema.player import COCO_KP_NAMES

SOURCE_COMMIT = "7e5bae88456ac418ff0e58e74106c9fe192055d4"
INPUT_HW = (1024, 768)
HEATMAP_CHW = (308, 256, 192)
ROPE_CONFIG_DTYPE = "fp32"
# The released COCO-WholeBody to Goliath mapping includes the face landmarks.
COCO17_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
TRT_SETTINGS = {
    "source_commit": SOURCE_COMMIT,
    "forward": "single_view",
    "pos_embed_rope_dtype": ROPE_CONFIG_DTYPE,
}
TRT_INPUTS = {"pixels": TensorSpec(dtype="float16", shape=(-1, 3, *INPUT_HW))}
TRT_OUTPUTS = {"heatmaps": TensorSpec(dtype="float16", shape=(-1, *HEATMAP_CHW))}


@dataclass(frozen=True)
class Checkpoint:
    size: str
    revision: str
    sha256: str
    file_bytes: int
    float32_elements: int

    @property
    def filename(self) -> str:
        return f"sapiens2_{self.size}_pose.safetensors"


CHECKPOINTS = {
    "sapiens2-1b": Checkpoint(
        "1b",
        "f5fed8b97b99698d5eea1d14ff0855d0b4c3f000",
        "2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f",
        6079194752,
        1519780164,
    ),
    "sapiens2-5b": Checkpoint(
        "5b",
        "ada1f29aa1fd454ca28665c700923a0101b6b24f",
        "b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52",
        20480899148,
        5120198663,
    ),
}

HeatmapDecoder = Callable[
    [NDArray[np.float32]], tuple[NDArray[np.floating], NDArray[np.floating]]
]
Forward = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OfficialPose:
    build: Callable[[Mapping[str, Any]], torch.nn.Module]
    model_config: Mapping[str, Any]
    pipeline: Callable[[dict[str, Any]], dict[str, Any]]
    preprocessor: torch.nn.Module
    decode: HeatmapDecoder
    flipped_channels: tuple[int, ...]


def verify_checkpoint(spec: Checkpoint, path: Path) -> None:
    if not path.is_file():
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"Missing {spec.filename}: {path}"
        )
    if path.stat().st_size != spec.file_bytes or dual_sha256(path) != spec.sha256:
        raise ConditionUnavailable(
            Status.UNAVAILABLE,
            f"File does not match pinned SHA-256 for {spec.filename}: {path}",
        )


def _coco_flip_channels(meta: Mapping[str, Any]) -> tuple[int, ...]:
    mapping = meta["coco_wholebody_to_goliath_mapping"]
    channels = tuple(mapping.get(index) for index in range(len(COCO_KP_NAMES)))
    names = meta["keypoint_id2name"]
    if (
        meta["num_keypoints"] != HEATMAP_CHW[0]
        or channels != COCO17_CHANNELS
        or tuple(names[index] for index in channels) != COCO_KP_NAMES
    ):
        raise ValueError("Official Sapiens2 metadata does not map every COCO17 joint")

    flipped = tuple(meta["flip_indices"][index] for index in COCO17_CHANNELS)
    # Every COCO17 joint after the nose belongs to a left/right pair.
    expected = (COCO17_CHANNELS[0],) + tuple(
        COCO17_CHANNELS[index + 1] if index % 2 else COCO17_CHANNELS[index - 1]
        for index in range(1, len(COCO_KP_NAMES))
    )
    if flipped != expected:
        raise ValueError(
            "Official Sapiens2 flip indices disagree with COCO17 laterality"
        )
    return flipped


def load_official(spec: Checkpoint, source_root: Path | None) -> OfficialPose:
    if source_root is None or not source_root.is_dir():
        raise ConditionUnavailable(
            Status.UNAVAILABLE,
            f"Provide --source-root at Sapiens2 commit {SOURCE_COMMIT}",
        )
    root = source_root.resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        identity = git("rev-parse", "--show-toplevel", "HEAD").splitlines()
        if identity != [str(root), SOURCE_COMMIT] or git(
            "status", "--porcelain", "--untracked-files=normal", "--", "sapiens"
        ):
            raise ValueError(
                f"Expected an unmodified Sapiens2 checkout at {SOURCE_COMMIT}"
            )
        sys.path.insert(0, str(root))
        try:
            package = find_spec("sapiens")
            if package is None or package.origin != str(
                root / "sapiens" / "__init__.py"
            ):
                raise ValueError(
                    "The active sapiens package differs from --source-root"
                )
            config_class = importlib.import_module("sapiens.engine.config").Config
            compose = importlib.import_module("sapiens.engine.datasets").Compose
            datasets = importlib.import_module("sapiens.pose.datasets")
            registry = importlib.import_module("sapiens.registry").MODELS
        finally:
            sys.path.remove(str(root))

        configs = root / "sapiens" / "pose" / "configs"
        cohort = "shutterstock_goliath_3po"
        config = config_class.fromfile(
            str(
                configs
                / "keypoints308"
                / cohort
                / f"sapiens2_{spec.size}_keypoints308_{cohort}-1024x768.py"
            )
        )
        model_config = config.model
        model_config["backbone"].pop("init_cfg")
        model_config["backbone"]["pos_embed_rope_dtype"] = ROPE_CONFIG_DTYPE
        codec = dict(config.codec)
        if (
            tuple(config.image_size) != INPUT_HW
            or config.num_keypoints != HEATMAP_CHW[0]
            or tuple(codec["heatmap_size"]) != HEATMAP_CHW[1:][::-1]
            or codec.pop("type") != "UDPHeatmap"
            or not config.val_cfg["flip_test"]
        ):
            raise ValueError(
                "The Sapiens2 config differs from the pinned demo contract"
            )
        meta = datasets.parse_pose_metainfo(
            {"from_file": str(configs / "_base_" / "keypoints308.py")}
        )
        return OfficialPose(
            registry.build,
            model_config,
            compose(config.test_pipeline),
            registry.build(config.data_preprocessor).eval(),
            datasets.UDPHeatmap(**codec).decode,
            _coco_flip_channels(meta),
        )
    except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"Cannot load pinned Sapiens2 source {root}: {error}"
        ) from error


def load_model(
    official: OfficialPose, checkpoint: Path, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    try:
        load_file = importlib.import_module("safetensors.torch").load_file
    except ImportError as error:
        raise ConditionUnavailable(Status.UNAVAILABLE, str(error)) from error
    # The official initialization builds CPU weights before reading the checkpoint.
    # Meta initialization and assignment avoid a second 20 GB host allocation.
    try:
        with torch.device("meta"):
            model = official.build(official.model_config)
        model.load_state_dict(
            load_file(checkpoint, device="cpu"), strict=True, assign=True
        )
        rope = model.backbone.rope_embed
        periods = rope.periods
        model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        # The comparison's FP32 RoPE configuration also needs the original periods.
        rope.periods = periods.to(device=device, dtype=torch.float32)
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE,
            f"Cannot load {checkpoint} as {dtype} on {device}: {error}",
        ) from error
    return model


def _prepare(
    rows: tuple[PoseInput, ...], official: OfficialPose
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...]]:
    samples = tuple(
        official.pipeline(
            {
                "img": row.frame_bgr,
                "bbox": row.bbox_xyxy[None].copy(),
                "bbox_score": np.ones(1, dtype=np.float32),
            }
        )
        for row in rows
    )
    packed = torch.stack([sample["inputs"] for sample in samples])
    pixels: torch.Tensor = official.preprocessor({"inputs": packed})["inputs"]
    return pixels, tuple(sample["data_samples"]["meta"] for sample in samples)


def _decode(
    heatmaps: NDArray[np.float32],
    transforms: tuple[Mapping[str, Any], ...],
    decode: HeatmapDecoder,
) -> PoseBatch:
    poses: NDArray[np.float32] = np.full(
        (len(transforms), len(COCO_KP_NAMES), 3), np.nan, np.float32
    )
    for pose, maps, meta in zip(poses, heatmaps, transforms, strict=True):
        scores = maps.max(axis=(1, 2))
        pose[:, 2] = np.where(np.isfinite(scores), scores, np.nan)
        # Upstream's peak finder uses an invalid coordinate for nonpositive peaks.
        valid = (scores > 0) & np.isfinite(maps).all(axis=(1, 2))
        if valid.any():
            crop_xy, _ = decode(maps[valid])
            scale = meta["bbox_scale"]
            xy = (
                crop_xy[0] / meta["input_size"] * scale
                + meta["bbox_center"]
                - 0.5 * scale
            )
            pose[valid, :2] = np.where(
                np.isfinite(xy).all(axis=1, keepdims=True), xy, np.nan
            )
    return PoseBatch(poses)


def _run_batch(
    rows: tuple[PoseInput, ...],
    stages: PoseStages,
    official: OfficialPose,
    run: Forward,
    backend: Backend,
    device: torch.device,
    dtype: torch.dtype,
    channels: torch.Tensor,
    flipped_channels: torch.Tensor,
) -> PoseBatch:
    def selected(
        pixels: torch.Tensor, indices: torch.Tensor, name: str
    ) -> torch.Tensor:
        maps = stages.forward(backend, lambda: run(pixels), name=name)
        if tuple(maps.shape) != (len(rows), *HEATMAP_CHW):
            raise ValueError(f"Unexpected Sapiens2 heatmap shape: {tuple(maps.shape)}")
        return stages.stage(
            Stage.POSTPROCESS,
            lambda: maps.index_select(1, indices).float(),
            name=f"{name}_coco17",
        )

    with torch.inference_mode():
        pixels, transforms = stages.stage(
            Stage.PREPROCESS, lambda: _prepare(rows, official)
        )
        pixels = stages.stage(Stage.H2D, lambda: pixels.to(device=device, dtype=dtype))
        original = selected(pixels, channels, "original")
        mirrored = stages.stage(
            Stage.PREPROCESS, lambda: pixels.flip(-1), name="horizontal_flip"
        )
        flipped = selected(mirrored, flipped_channels, "flipped")
        merged = stages.stage(
            Stage.POSTPROCESS,
            lambda: (original + flipped.flip(-1)) * 0.5,
            name="flip_average",
        )
        maps_cpu = stages.stage(Stage.D2H, lambda: merged.cpu().numpy())
        return stages.stage(
            Stage.POSTPROCESS,
            lambda: _decode(maps_cpu, transforms, official.decode),
            name="udp_decode",
        )


def _engine_forward(
    model_id: str, spec: Checkpoint, engine: Path, device: torch.device
) -> Forward:
    try:
        bundle = EngineBundle.read(engine).for_model(
            model=model_id,
            settings=TRT_SETTINGS,
            sources={"checkpoint": spec.sha256},
            inputs=TRT_INPUTS,
            outputs=TRT_OUTPUTS,
        )
        runner = TensorRTRunner(bundle, device)
        # Keep the engine's inspector metadata and disable NVTX during inference.
        runner._context.nvtx_verbosity = tensorrt_module().ProfilingVerbosity.NONE
    except torch.OutOfMemoryError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"Sapiens2 TensorRT engine {engine}: {error}"
        ) from error

    def forward(pixels: torch.Tensor) -> torch.Tensor:
        try:
            bundle.engine.profiles["pixels"].require(
                tuple(pixels.shape), tensor="pixels"
            )
        except ValueError as error:
            raise ConditionUnavailable(Status.UNSUPPORTED, str(error)) from error
        return runner({"pixels": pixels})["heatmaps"]

    return forward


def make_adapter(
    *,
    model_id: str,
    backend: Backend,
    checkpoint: Path,
    device: str,
    engine: Path | None = None,
    source_root: Path | None = None,
) -> BenchmarkAdapter[tuple[PoseInput, ...], PoseBatch]:
    spec = CHECKPOINTS[model_id]
    if backend is not Backend.TRT16 and engine is not None:
        raise ValueError("An engine can only be supplied to TRT16")
    dtype = torch.float32 if backend is Backend.PT32 else torch.float16

    def load() -> Callable[[tuple[PoseInput, ...], PoseStages], PoseBatch]:
        if backend is Backend.TRT16 and engine is None:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, "Provide a Sapiens2 TensorRT bundle"
            )
        verify_checkpoint(spec, checkpoint)
        try:
            target = resolve_device(device)
            if target.type != "cuda":
                raise DeviceSelectionError(
                    f"Sapiens2 benchmark requires a CUDA device: {device}"
                )
        except DeviceSelectionError as error:
            raise ConditionUnavailable(Status.UNAVAILABLE, str(error)) from error
        official = load_official(spec, source_root)
        run: Forward
        if engine is None:
            run = load_model(official, checkpoint, target, dtype)
        else:
            run = _engine_forward(model_id, spec, engine, target)
        channels = torch.tensor(COCO17_CHANNELS, device=target)
        flipped = torch.tensor(official.flipped_channels, device=target)

        def infer(rows: tuple[PoseInput, ...], stages: PoseStages) -> PoseBatch:
            return _run_batch(
                rows, stages, official, run, backend, target, dtype, channels, flipped
            )

        return infer

    return BenchmarkAdapter(
        backend,
        load,
        {"checkpoint": checkpoint},
        {}
        if engine is None
        else {
            name: engine / name
            for name in (PLAN_FILE, ENGINE_MANIFEST, EXPORT_MANIFEST)
        },
    )
