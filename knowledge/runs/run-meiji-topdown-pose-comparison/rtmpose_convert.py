"""Export RTMPose SimCC tensors, build a TensorRT plan, and inspect its layers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from rtmpose import (
    EXPORT_SETTINGS,
    HEIGHT,
    MODELS,
    ONNX_OPSET,
    OUTPUTS,
    WIDTH,
    RTMPoseGraph,
    load_estimator,
    load_recipe,
    load_tensorrt,
)

from src.utils.inference_benchmark import Conversion, measure_conversion
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import InputProfile
from src.utils.models.tensorrt_build import build_bundle, export_bundle


def _shape(batch: int) -> tuple[int, ...]:
    return batch, 3, HEIGHT, WIDTH


def _export(args: argparse.Namespace) -> None:
    recipe = load_recipe(args.model, args.checkpoint, args.source_root)
    graph = RTMPoseGraph(load_estimator(recipe)).eval().half()
    sample = torch.zeros(_shape(1), dtype=torch.float16)

    def write_onnx(output: Path) -> None:
        with torch.inference_mode():
            torch.onnx.export(
                graph,
                (sample,),
                str(output),
                input_names=["pixels"],
                output_names=list(OUTPUTS),
                dynamic_axes={name: {0: "batch"} for name in ("pixels", *OUTPUTS)},
                opset_version=ONNX_OPSET,
                dynamo=False,
            )

    export_bundle(
        args.engine,
        model=args.model,
        settings=EXPORT_SETTINGS,
        sources={"checkpoint": args.checkpoint, "config": recipe.config_path},
        export=write_onnx,
    )


def _build(args: argparse.Namespace) -> None:
    profile = InputProfile(
        minimum=_shape(1),
        optimum=_shape(args.opt_batch),
        maximum=_shape(args.max_batch),
    )
    build_bundle(
        args.engine,
        {"pixels": profile},
        device=torch.device(args.device),
        workspace_bytes=args.workspace_mib * 1024 * 1024,
        detailed_profiling=True,
    )


def _inspect(args: argparse.Namespace) -> None:
    recipe = load_recipe(args.model, args.checkpoint, args.source_root)
    device = torch.device(args.device)
    runner = load_tensorrt(args.model, recipe, args.engine, device)
    tensors = torch.zeros(_shape(args.batch), device=device, dtype=torch.float16)
    with torch.inference_mode():
        runner({"pixels": tensors})
        torch.cuda.synchronize(device)
    write_json_atomic(args.receipt, runner.inspect_layers())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    export = subcommands.add_parser("export")
    build = subcommands.add_parser("build")
    inspect = subcommands.add_parser("inspect")
    for subcommand in (export, build, inspect):
        subcommand.add_argument("--engine", type=Path, required=True)
        subcommand.add_argument("--receipt", type=Path, required=True)
    for subcommand in (export, inspect):
        subcommand.add_argument("--model", choices=MODELS, required=True)
        subcommand.add_argument("--checkpoint", type=Path, required=True)
        subcommand.add_argument("--source-root", type=Path, required=True)
    for subcommand in (build, inspect):
        subcommand.add_argument("--device", default="cuda:0")
    build.add_argument("--opt-batch", type=int, required=True)
    build.add_argument("--max-batch", type=int, required=True)
    build.add_argument("--workspace-mib", type=int, required=True)
    inspect.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        _inspect(args)
        return
    if args.command == "export":
        result = measure_conversion(Conversion.ONNX_EXPORT, lambda: _export(args))
    else:
        if not 1 <= args.opt_batch <= args.max_batch or args.workspace_mib < 1:
            parser.error(
                "Require 1 <= opt-batch <= max-batch and positive workspace-mib"
            )
        result = measure_conversion(
            Conversion.ENGINE_BUILD, lambda: _build(args), device=args.device
        )
    write_json_atomic(args.receipt, asdict(result))
    if result.failure is not None:
        raise SystemExit(result.failure.message)


if __name__ == "__main__":
    main()
