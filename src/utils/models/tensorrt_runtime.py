"""Execute one verified TensorRT profile with PyTorch CUDA tensor bindings."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from hashlib import sha256
from types import ModuleType
from typing import Any

import torch

from src.utils.models.tensorrt_artifacts import (
    PLAN_FILE,
    EngineBundle,
    EngineEnvironment,
)


def tensorrt_module() -> ModuleType:
    try:
        return importlib.import_module("tensorrt")
    except ImportError as error:
        raise RuntimeError(
            "TensorRT inference requires the 'tensorrt' extra"
        ) from error


def engine_environment(device: torch.device, trt: ModuleType) -> EngineEnvironment:
    if device.type != "cuda":
        raise ValueError(f"TensorRT requires CUDA, got {device}")
    return EngineEnvironment(
        tensorrt_version=trt.__version__,
        device_name=torch.cuda.get_device_name(device),
        compute_capability=torch.cuda.get_device_capability(device),
    )


def require_engine_contract(bundle: EngineBundle, engine: Any, trt: ModuleType) -> None:
    inputs, outputs = bundle.export.inputs, bundle.export.outputs
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    if names != inputs.keys() | outputs.keys() or engine.num_optimization_profiles != 1:
        raise ValueError(
            f"Unexpected TensorRT bindings or profile count: {bundle.root}"
        )
    for name, spec in {**inputs, **outputs}.items():
        mode = trt.TensorIOMode.INPUT if name in inputs else trt.TensorIOMode.OUTPUT
        if (
            engine.get_tensor_mode(name) != mode
            or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
            or engine.get_tensor_dtype(name) != getattr(trt, spec.dtype)
            or tuple(engine.get_tensor_shape(name)) != spec.shape
            or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
        ):
            raise ValueError(f"Unexpected TensorRT binding {name}: {bundle.root}")
    for name, spec in inputs.items():
        if -1 in spec.shape:
            observed = tuple(
                tuple(shape) for shape in engine.get_tensor_profile_shape(name, 0)
            )
            if observed != bundle.engine.profiles[name].shapes:
                raise ValueError(
                    f"TensorRT engine profile mismatch for {name}: {observed}"
                )


class TensorRTRunner:
    """Single execution context bound to the CUDA stream of its first caller."""

    def __init__(self, bundle: EngineBundle, device: torch.device) -> None:
        trt = tensorrt_module()
        bundle.require_environment(engine_environment(device, trt))
        plan = (bundle.root / PLAN_FILE).read_bytes()
        if sha256(plan).hexdigest() != bundle.engine.plan_sha256:
            raise ValueError(f"TensorRT plan SHA-256 mismatch: {bundle.root}")
        self.bundle = bundle
        self.device = device
        self._stream: torch.cuda.Stream | None = None
        self._logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(device):
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(plan)
            if self._engine is None:
                raise RuntimeError(
                    f"Could not deserialize TensorRT engine: {bundle.root}"
                )
            require_engine_contract(bundle, self._engine, trt)
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(f"Could not create TensorRT context: {bundle.root}")

    def __call__(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        specs = self.bundle.export.inputs
        if inputs.keys() != specs.keys():
            raise ValueError(
                f"Expected TensorRT inputs {sorted(specs)}, got {sorted(inputs)}"
            )
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self._stream is not None and stream != self._stream:
                raise RuntimeError("TensorRT context cannot switch CUDA streams")
            self._stream = stream
            bindings = {}
            for name, spec in specs.items():
                tensor = inputs[name]
                shape = tuple(tensor.shape)
                self.bundle.engine.profiles[name].require(shape, tensor=name)
                if -1 in spec.shape and not self._context.set_input_shape(name, shape):
                    raise ValueError(f"TensorRT rejected input shape {name}={shape}")
                bindings[name] = tensor.to(
                    device=self.device, dtype=getattr(torch, spec.dtype)
                ).contiguous()
            self._bind(bindings)
            if unresolved := self._context.infer_shapes():
                raise RuntimeError(f"Unresolved TensorRT tensors: {unresolved}")
            outputs = {}
            for name, spec in self.bundle.export.outputs.items():
                shape = tuple(self._context.get_tensor_shape(name))
                if len(shape) != len(spec.shape) or any(
                    dim <= 0 or (expected != -1 and dim != expected)
                    for dim, expected in zip(shape, spec.shape, strict=True)
                ):
                    raise RuntimeError(
                        f"Unexpected TensorRT output shape {name}={shape}"
                    )
                outputs[name] = torch.empty(
                    shape, dtype=getattr(torch, spec.dtype), device=self.device
                )
            self._bind(outputs)
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"TensorRT inference failed: {self.bundle.root}")
            for tensor in bindings.values():
                tensor.record_stream(stream)
            return outputs

    def _bind(self, tensors: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in tensors.items():
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT tensor: {name}")
