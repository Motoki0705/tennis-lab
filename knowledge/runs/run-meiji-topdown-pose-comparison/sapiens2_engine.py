"""Export or build the Sapiens2 single-view FP16 TensorRT bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from sapiens2 import (
    CHECKPOINTS,
    INPUT_HW,
    TRT_SETTINGS,
    load_model,
    load_official,
    verify_checkpoint,
)

from src.utils.inference_benchmark import (
    ConditionUnavailable,
    Conversion,
    Status,
    measure_conversion,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import InputProfile
from src.utils.models.tensorrt_build import build_bundle, export_bundle


def _export(
    directory: Path, model_id: str, checkpoint: Path, source_root: Path
) -> None:
    spec = CHECKPOINTS[model_id]
    verify_checkpoint(spec, checkpoint)
    official = load_official(spec, source_root)
    model = load_model(official, checkpoint, torch.device("cpu"), torch.float16)

    def export(path: Path) -> None:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (torch.zeros((2, 3, *INPUT_HW), dtype=torch.float16),),
                str(path),
                input_names=["pixels"],
                output_names=["heatmaps"],
                dynamic_shapes=({0: torch.export.Dim("person_batch")},),
                opset_version=19,
                dynamo=True,
                external_data=True,
                optimize=False,
            )

    try:
        export_bundle(
            directory,
            model=model_id,
            settings=TRT_SETTINGS,
            sources={"checkpoint": checkpoint},
            export=export,
        )
    except ImportError as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"ONNX export dependency: {error}"
        ) from error


def _build(
    directory: Path,
    device: str,
    optimum: int,
    maximum: int,
    workspace_mib: int,
) -> None:
    profile = InputProfile(
        minimum=(1, 3, *INPUT_HW),
        optimum=(optimum, 3, *INPUT_HW),
        maximum=(maximum, 3, *INPUT_HW),
    )
    build_bundle(
        directory,
        {"pixels": profile},
        device=torch.device(device),
        workspace_bytes=workspace_mib * 1024**2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in Conversion:
        command = commands.add_parser(phase)
        command.add_argument("--directory", required=True, type=Path)
        command.add_argument("--receipt", required=True, type=Path)
        if phase is Conversion.ONNX_EXPORT:
            command.add_argument("--model", required=True, choices=tuple(CHECKPOINTS))
            command.add_argument("--checkpoint", required=True, type=Path)
            command.add_argument("--source-root", required=True, type=Path)
        else:
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--optimum-batch", type=int, required=True)
            command.add_argument("--maximum-batch", type=int, required=True)
            command.add_argument("--workspace-mib", type=int, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error(f"Receipt already exists: {args.receipt}")
    phase = Conversion(args.phase)
    if phase is Conversion.ONNX_EXPORT:
        result = measure_conversion(
            phase,
            lambda: _export(
                args.directory, args.model, args.checkpoint, args.source_root
            ),
        )
    else:
        if not 1 <= args.optimum_batch <= args.maximum_batch or args.workspace_mib < 1:
            parser.error(
                "The optimum and maximum batch and workspace must be positive and ordered"
            )
        result = measure_conversion(
            phase,
            lambda: _build(
                args.directory,
                args.device,
                args.optimum_batch,
                args.maximum_batch,
                args.workspace_mib,
            ),
            device=args.device,
        )
    write_json_atomic(args.receipt, asdict(result))
    if result.failure is not None:
        raise SystemExit(result.failure.message)


if __name__ == "__main__":
    main()
