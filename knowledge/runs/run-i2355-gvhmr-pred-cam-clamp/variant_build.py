"""Build one strongly typed static TensorRT plan with explicit builder options.

The builder matches ``convert.build_checked_plan`` for static diagnosis builds
(strongly typed network, 512 MiB workspace, TF32 cleared, detailed profiling),
and additionally applies the builder options named in ``--options``.  Every
option is read back from the config; an option TensorRT did not apply stops the
build with a non-zero exit instead of silently building a different variant.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

WORKSPACE_BYTES = 512 * 1024**2
OPTION_KEYS = {
    "builder_optimization_level",
    "tactic_sources",
    "max_aux_streams",
    "tiling_optimization_level",
    "max_num_tactics",
    "set_flags",
}
TILING_LEVELS = ("NONE", "FAST", "MODERATE", "FULL")


@dataclass(frozen=True)
class BuildOptions:
    builder_optimization_level: int = 3
    tactic_sources: tuple[str, ...] | None = None
    max_aux_streams: int | None = None
    tiling_optimization_level: str | None = None
    max_num_tactics: int | None = None
    set_flags: tuple[str, ...] = field(default_factory=tuple)


def parse_options(raw: dict[str, object]) -> BuildOptions:
    """Validate a JSON option mapping without importing TensorRT."""
    if unknown := raw.keys() - OPTION_KEYS:
        raise ValueError(f"Unknown TensorRT builder options: {sorted(unknown)}")
    level = raw.get("builder_optimization_level", 3)
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 5:
        raise ValueError(f"builder_optimization_level must be 0..5, got {level!r}")

    def names(key: str) -> tuple[str, ...]:
        value = raw.get(key, ())
        if not isinstance(value, list | tuple) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"{key} must be a list of names")
        if len(set(value)) != len(value):
            raise ValueError(f"{key} has duplicate names")
        return tuple(value)

    def optional_count(key: str) -> int | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        return value

    tiling = raw.get("tiling_optimization_level")
    if tiling is not None and tiling not in TILING_LEVELS:
        raise ValueError(f"tiling_optimization_level must be one of {TILING_LEVELS}")
    return BuildOptions(
        builder_optimization_level=level,
        tactic_sources=names("tactic_sources") if "tactic_sources" in raw else None,
        max_aux_streams=optional_count("max_aux_streams"),
        tiling_optimization_level=tiling,
        max_num_tactics=optional_count("max_num_tactics"),
        set_flags=names("set_flags"),
    )


def apply_options(trt, config, options: BuildOptions) -> dict[str, object]:
    """Apply options to a builder config and return the read-back values."""
    config.builder_optimization_level = options.builder_optimization_level
    applied: dict[str, object] = {
        "builder_optimization_level": config.builder_optimization_level
    }
    if options.tactic_sources is not None:
        mask = 0
        for name in options.tactic_sources:
            mask |= 1 << int(getattr(trt.TacticSource, name))
        config.set_tactic_sources(mask)
        applied["tactic_sources_mask"] = int(config.get_tactic_sources())
        if applied["tactic_sources_mask"] != mask:
            raise RuntimeError(f"TensorRT tactic sources {applied} differ from {mask}")
    if options.max_aux_streams is not None:
        config.max_aux_streams = options.max_aux_streams
        applied["max_aux_streams"] = config.max_aux_streams
    if options.tiling_optimization_level is not None:
        level = getattr(trt.TilingOptimizationLevel, options.tiling_optimization_level)
        config.tiling_optimization_level = level
        applied["tiling_optimization_level"] = config.tiling_optimization_level.name
    if options.max_num_tactics is not None:
        config.max_num_tactics = options.max_num_tactics
        applied["max_num_tactics"] = config.max_num_tactics
    for name in options.set_flags:
        flag = getattr(trt.BuilderFlag, name)
        config.set_flag(flag)
        if not config.get_flag(flag):
            raise RuntimeError(f"TensorRT did not set builder flag {name}")
    applied["set_flags"] = list(options.set_flags)
    expected = {
        key: value
        for key, value in asdict(options).items()
        if key
        in {
            "builder_optimization_level",
            "max_aux_streams",
            "tiling_optimization_level",
            "max_num_tactics",
        }
        and value is not None
    }
    for key, value in expected.items():
        if applied[key] != value:
            raise RuntimeError(f"TensorRT applied {key}={applied[key]}, not {value}")
    return applied


def build(onnx_file: Path, plan: Path, options: BuildOptions) -> dict[str, object]:
    import tensorrt as trt
    import torch
    from convert import inspect_engine

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("The pinned TensorRT build requires one CC 12.0 CUDA GPU")
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    applied = apply_options(trt, config, options)
    print(f"Applied TensorRT builder options: {json.dumps(applied)}", flush=True)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_file}\n{errors}")
    for index in range(network.num_inputs):
        shape = tuple(network.get_input(index).shape)
        if any(axis < 0 for axis in shape):
            raise ValueError(f"Diagnosis builds must be static, got {shape}")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_file}")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the built engine")
    layers = inspect_engine(engine)["Layers"]
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.with_suffix(".layers.json").write_text(json.dumps(layers, indent=2) + "\n")
    with plan.open("xb") as stream:
        stream.write(memoryview(engine_bytes))
    return {
        "applied_options": applied,
        "engine_layers": len(layers),
        "layer_types": dict(Counter(layer["LayerType"] for layer in layers)),
    }


def main() -> None:
    from benchmark import write_json
    from convert import require_queue
    from inputs import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--options", default="{}", help="JSON builder options")
    args = parser.parse_args()
    require_queue()
    options = parse_options(json.loads(args.options))
    evidence = build(args.onnx, args.plan, options)
    write_json(
        args.report,
        {
            "onnx": str(args.onnx),
            "onnx_sha256": sha256_file(args.onnx),
            "plan": str(args.plan),
            "plan_sha256": sha256_file(args.plan),
            "options": asdict(options),
            **evidence,
        },
    )


if __name__ == "__main__":
    main()
