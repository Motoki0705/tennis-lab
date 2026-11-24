"""Export typed ONNX and build TensorRT FP16 plans with separate receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from edgecrafter import ASSETS, HEADS, INPUT_NAME, ONNX_OPSET, export_settings

from src.submodules.models.ecpose.inference import load_network
from src.submodules.models.ecpose.releases import IMAGE_SIZE, RELEASES
from src.utils.inference_benchmark import Conversion, measure_conversion
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import InputProfile
from src.utils.models.tensorrt_build import build_bundle, export_bundle


def export_onnx(
    model_id: str,
    assets: Path,
    directory: Path,
    device: torch.device,
    trace_batch: int,
) -> dict[str, int]:
    if directory.exists():
        raise FileExistsError(directory)
    release = RELEASES[model_id]
    checkpoint = assets / release.filename
    network = load_network(assets / "EdgeCrafter", checkpoint, release).to(device)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    pixels = torch.zeros(
        (trace_batch, 3, *IMAGE_SIZE), dtype=torch.float32, device=device
    )
    precision: dict[str, int] = {}

    def write_graph(path: Path) -> None:
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.float16, cache_enabled=False),
        ):
            torch.onnx.export(
                network,
                (pixels,),
                str(path),
                input_names=[INPUT_NAME],
                output_names=list(HEADS),
                dynamic_axes={name: {0: "batch"} for name in (INPUT_NAME, *HEADS)},
                opset_version=ONNX_OPSET,
                dynamo=False,
            )
        graph = onnx.load(path, load_external_data=False).graph
        for name, dtype in (
            ("float16", onnx.TensorProto.FLOAT16),
            ("float32", onnx.TensorProto.FLOAT),
        ):
            precision[f"{name}_initializers"] = sum(
                t.data_type == dtype for t in graph.initializer
            )
            precision[f"casts_to_{name}"] = sum(
                node.op_type == "Cast"
                and any(
                    attr.name == "to" and attr.i == dtype for attr in node.attribute
                )
                for node in graph.node
            )
        if not precision["float16_initializers"] and not precision["casts_to_float16"]:
            raise ValueError(
                "The exported EdgeCrafter graph has no typed FP16 parameters or casts"
            )

    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-", dir=directory.parent) as staging:
        bundle = Path(staging) / "bundle"
        try:
            export_bundle(
                bundle,
                model=model_id,
                settings=export_settings(),
                sources={"weights": checkpoint},
                export=write_graph,
            )
        except onnx.checker.ValidationError as error:
            raise ValueError(f"Invalid EdgeCrafter ONNX graph: {error}") from error
        bundle.rename(directory)
    return precision


def engine_profile(opt_batch: int, max_batch: int) -> dict[str, InputProfile]:
    shape = (3, *IMAGE_SIZE)
    return {
        INPUT_NAME: InputProfile(
            minimum=(1, *shape),
            optimum=(opt_batch, *shape),
            maximum=(max_batch, *shape),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "build"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--engine", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "export":
            command.add_argument("--assets", type=Path, default=ASSETS)
            command.add_argument("--model", choices=RELEASES, required=True)
            command.add_argument("--trace-batch", type=int, default=2)
        else:
            command.add_argument("--opt-batch", type=int, default=1)
            command.add_argument("--max-batch", type=int, required=True)
            command.add_argument("--workspace-mib", type=int, default=1536)

    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("Use an explicit CUDA device for conversion")
    details: dict[str, object] = {}
    if args.command == "export":
        if args.trace_batch < 1:
            parser.error("Trace batch must be positive")
        details.update(
            model_id=args.model,
            weight_sha256=RELEASES[args.model].sha256,
            settings=export_settings(),
            trace_batch=args.trace_batch,
        )

        def operation() -> None:
            details["onnx_precision"] = export_onnx(
                args.model,
                args.assets,
                args.engine,
                device,
                args.trace_batch,
            )

        receipt = measure_conversion(Conversion.ONNX_EXPORT, operation, device=device)
    else:
        if args.workspace_mib < 1 or not 1 <= args.opt_batch <= args.max_batch:
            parser.error(
                "Expected a positive workspace and 1 <= opt-batch <= max-batch"
            )
        profiles = engine_profile(args.opt_batch, args.max_batch)
        details.update(
            profile=profiles[INPUT_NAME].shapes, workspace_mib=args.workspace_mib
        )

        def build() -> None:
            manifest = build_bundle(
                args.engine,
                profiles,
                device=device,
                workspace_bytes=args.workspace_mib * 1024**2,
            )
            details["engine_sha256"] = manifest.plan_sha256

        receipt = measure_conversion(Conversion.ENGINE_BUILD, build, device=device)

    write_json_atomic(args.receipt, {**asdict(receipt), **details})
    if receipt.failure is not None:
        raise SystemExit(receipt.failure.message)
    print(
        f"{receipt.phase} completed in {receipt.wall_ms:.0f} ms. Receipt {args.receipt}"
    )


if __name__ == "__main__":
    main()
