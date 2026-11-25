"""Shared EdgeCrafter execution backends and the opt-in ECPose asset contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from pydantic import JsonValue

from src.submodules.models.ecpose.inference import (
    load_network,
    require_source,
    require_weight,
    validate_decode_policy,
)
from src.submodules.models.ecpose.releases import (
    IMAGE_SIZE,
    RELEASES,
    SOURCE_COMMIT,
    EdgeCrafterRelease,
)
from src.utils.checksum import dual_sha256
from src.utils.models.loading.source_assets import python_source_assets
from src.utils.models.tensorrt_artifacts import EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import TensorRTRunner

HEADS = ("logits", "coordinates")
INPUT_NAME = "pixels"
ONNX_OPSET = 19

EdgeCrafterInference = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def export_settings() -> dict[str, JsonValue]:
    return {
        "source_commit": SOURCE_COMMIT,
        "image_size_hw": list(IMAGE_SIZE),
        "opset": ONNX_OPSET,
        "precision_policy": "cuda_fp16_autocast",
        "batch_axis": "dynamic",
    }


def read_engine(
    directory: Path, model_id: str, *, digest: Callable[[Path], str] = dual_sha256
) -> EngineBundle:
    release = RELEASES[model_id]
    bundle = EngineBundle.read(directory, digest=digest)
    shapes = (
        ((-1, 60, 2), (-1, 60, 17, 2))
        if release.has_pose
        else ((-1, 300, 80), (-1, 300, 4))
    )
    if set(bundle.export.outputs) != set(HEADS):
        raise ValueError(f"Unexpected EdgeCrafter TensorRT outputs: {directory}")
    outputs = {}
    for name, shape in zip(HEADS, shapes, strict=True):
        spec = bundle.export.outputs[name]
        if spec.dtype not in {"float16", "float32"}:
            raise ValueError(
                f"Unexpected EdgeCrafter TensorRT dtype {name}={spec.dtype}"
            )
        outputs[name] = TensorSpec(dtype=spec.dtype, shape=shape)
    return bundle.for_model(
        model=model_id,
        settings=export_settings(),
        sources={"weights": release.sha256},
        inputs={INPUT_NAME: TensorSpec(dtype="float32", shape=(-1, 3, *IMAGE_SIZE))},
        outputs=outputs,
    )


def load_backend(
    repository: Path,
    checkpoint: Path,
    model_id: str,
    backend: str,
    device: torch.device,
    engine: Path | None,
) -> EdgeCrafterInference:
    _validate_backend(backend, device, engine)
    release = RELEASES[model_id]
    if engine is not None:
        require_weight(checkpoint, release)
        runner = TensorRTRunner(read_engine(engine, model_id), device)

        def run_trt(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = runner({INPUT_NAME: pixels})
            return outputs[HEADS[0]], outputs[HEADS[1]]

        return run_trt
    network = load_network(repository, checkpoint, release).to(device)

    def run_pytorch(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=backend == "pt16"
        ):
            logits, coordinates = network(pixels)
        return logits, coordinates

    return run_pytorch


def _validate_backend(backend: str, device: torch.device, engine: Path | None) -> None:
    if backend not in {"pt32", "pt16", "trt16"}:
        raise ValueError(f"Unknown EdgeCrafter backend: {backend}")
    if device.type not in {"cpu", "cuda"} or (
        backend != "pt32" and device.type != "cuda"
    ):
        raise ValueError(f"EdgeCrafter {backend} requires CUDA or explicit PT32 CPU")
    if (engine is not None) != (backend == "trt16"):
        raise ValueError(
            "An EdgeCrafter engine directory is required exactly for TRT16"
        )


@dataclass(frozen=True)
class ECPoseConfig:
    repository: Path
    checkpoint: Path
    model_id: str
    backend: str
    device: str
    batch_size: int
    confidence: float
    box_margin: float
    engine: Path | None

    def __post_init__(self) -> None:
        if self.model_id not in RELEASES or not self.release.has_pose:
            raise ValueError("ECPose requires ecpose-l-o365 or ecpose-x-o365")
        _validate_backend(self.backend, torch.device(self.device), self.engine)
        if self.batch_size < 1:
            raise ValueError("ECPose batch_size must be positive")
        validate_decode_policy(self.release, self.confidence, self.box_margin)

    @property
    def release(self) -> EdgeCrafterRelease:
        return RELEASES[self.model_id]

    @property
    def model_config(self) -> Path:
        return self.repository.joinpath(self.release.family, "configs", self.release.config)

    @cached_property
    def source_assets(self) -> dict[str, Path]:
        require_source(self.repository)
        assets = python_source_assets(self.repository, prefix="edgecrafter_source")
        configs = self.repository / self.release.family / "configs"
        return {
            **assets,
            **{
                f"edgecrafter_source/{file.relative_to(self.repository).as_posix()}": file
                for file in sorted(configs.rglob("*.yml"))
            },
        }
