"""Explicit ONNX export bookkeeping and strongly typed TensorRT builds."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import JsonValue

from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    ENGINE_MANIFEST,
    EXPORT_MANIFEST,
    ONNX_FILE,
    PLAN_FILE,
    EngineBundle,
    EngineManifest,
    ExportManifest,
    InputProfile,
    TensorDtype,
    TensorSpec,
    require_digests,
    require_profiles,
)
from src.utils.models.tensorrt_runtime import (
    engine_environment,
    require_engine_contract,
    tensorrt_module,
)


def _binding(info: Any, onnx: Any) -> TensorSpec:
    tensor = info.type.tensor_type
    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type).name
    return TensorSpec(
        dtype=cast(TensorDtype, dtype),
        shape=tuple(
            dim.dim_value if dim.HasField("dim_value") else -1
            for dim in tensor.shape.dim
        ),
    )


def export_bundle(
    directory: Path,
    *,
    model: str,
    settings: dict[str, JsonValue],
    sources: Mapping[str, Path],
    export: Callable[[Path], None],
) -> None:
    """Write a complete export once the typed ONNX and its sources verify."""
    onnx = importlib.import_module("onnx")
    source_hashes = {name: dual_sha256(path) for name, path in sources.items()}
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / ONNX_FILE
    export(path)
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(str(path))
    locations = {
        item.value
        for tensor in graph.graph.initializer
        for item in tensor.external_data
        if item.key == "location"
    }
    if locations - {f"{ONNX_FILE}.data"}:
        raise ValueError(f"Unexpected ONNX external weights: {locations}")
    if source_hashes != {name: dual_sha256(p) for name, p in sources.items()}:
        raise ValueError(f"Model sources changed during TensorRT export: {model}")
    manifest = ExportManifest(
        model=model,
        settings=settings,
        sources=source_hashes,
        onnx_files={
            name: dual_sha256(directory / name) for name in {ONNX_FILE, *locations}
        },
        inputs={info.name: _binding(info, onnx) for info in graph.graph.input},
        outputs={info.name: _binding(info, onnx) for info in graph.graph.output},
    )
    write_json_atomic(directory / EXPORT_MANIFEST, manifest.model_dump(mode="json"))


def set_profile_shapes(profile: Any, profiles: Mapping[str, InputProfile]) -> None:
    """Check the registered dimensions after the TensorRT setter returns."""
    for name, spec in profiles.items():
        profile.set_shape(name, *spec.shapes)
        actual = tuple(tuple(shape) for shape in profile.get_shape(name))
        if actual != spec.shapes:
            raise ValueError(
                f"TensorRT profile {name}={actual} differs from {spec.shapes}"
            )


def build_bundle(
    directory: Path,
    profiles: dict[str, InputProfile],
    *,
    device: torch.device,
    workspace_bytes: int,
    detailed_profiling: bool = False,
) -> EngineManifest:
    """Build the typed export on the selected GPU and publish its checked plan."""
    export_digest = dual_sha256(directory / EXPORT_MANIFEST)
    export = ExportManifest.model_validate_json(
        (directory / EXPORT_MANIFEST).read_bytes()
    )
    require_digests(directory, export.onnx_files, dual_sha256)
    require_profiles(profiles, export.inputs)
    plan_path = directory / PLAN_FILE
    manifest_path = directory / ENGINE_MANIFEST
    if plan_path.exists() or manifest_path.exists():
        raise FileExistsError(f"TensorRT engine output already exists: {directory}")

    trt = tensorrt_module()
    environment = engine_environment(device, trt)
    logger = trt.Logger(trt.Logger.WARNING)
    with torch.cuda.device(device):
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(directory / ONNX_FILE)):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(
                f"TensorRT failed to parse {directory / ONNX_FILE}\n{errors}"
            )
        config = builder.create_builder_config()
        if detailed_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        if any(-1 in spec.shape for spec in export.inputs.values()):
            profile = builder.create_optimization_profile()
            set_profile_shapes(profile, profiles)
            if config.add_optimization_profile(profile) != 0:
                raise RuntimeError(
                    "TensorRT did not accept the first optimization profile"
                )
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT build failed: {directory}")
        # Release ONNX weights and the builder arena before deserializing the plan.
        del parser, network, config, builder
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(
                f"TensorRT could not deserialize the built plan: {directory}"
            )
        with tempfile.TemporaryDirectory(prefix=".build-", dir=directory) as stage:
            temporary_plan = Path(stage) / PLAN_FILE
            with temporary_plan.open("wb") as file:
                file.write(memoryview(serialized))
            manifest = EngineManifest(
                export_sha256=export_digest,
                plan_sha256=dual_sha256(temporary_plan),
                environment=environment,
                profiles=profiles,
            )
            require_engine_contract(
                EngineBundle(directory, export, manifest), engine, trt
            )
            require_digests(
                directory,
                {EXPORT_MANIFEST: export_digest, **export.onnx_files},
                dual_sha256,
            )
            # A concurrent builder cannot replace a published plan.
            os.link(temporary_plan, plan_path)
            write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return manifest
