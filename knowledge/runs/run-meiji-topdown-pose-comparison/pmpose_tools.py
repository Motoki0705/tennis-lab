"""Run PMPose source parity and measure ONNX export and TensorRT builds."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from pmpose import (
    HEATMAP_HW,
    INPUT_HW,
    INPUT_SPEC,
    JOINT_ORDER,
    OUTPUT_SPEC,
    _infer,
    benchmark_settings,
    load_engine,
    load_model,
    load_official_pipeline,
)
from pmpose_assets import WEIGHTS, verify_checkpoint
from pose_contract import PoseInput

from src.utils.inference_benchmark import Backend, Conversion, Stage, measure_conversion
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import InputProfile
from src.utils.models.tensorrt_build import build_bundle, export_bundle

Value = TypeVar("Value")


class _ParityStages:
    def stage(
        self, stage: Stage, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        return operation()

    def forward(
        self, backend: Backend, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        if backend is not Backend.PT32:
            raise ValueError("The upstream API smoke must use PT32")
        return operation()


def _export(args: argparse.Namespace) -> None:
    weight = WEIGHTS[args.model]
    verify_checkpoint(args.checkpoint, weight)
    upstream = load_official_pipeline(args.model, args.source_root)

    def export(path: Path) -> None:
        with torch.no_grad():
            model = load_model(
                upstream,
                args.checkpoint,
                dtype=torch.float16,
                device=torch.device("cpu"),
            )
            torch.onnx.export(
                model,
                (torch.zeros(1, 3, *INPUT_HW, dtype=torch.float16),),
                str(path),
                input_names=list(INPUT_SPEC),
                output_names=list(OUTPUT_SPEC),
                dynamic_axes={
                    name: {0: "batch"} for name in (*INPUT_SPEC, *OUTPUT_SPEC)
                },
                opset_version=19,
                dynamo=False,
            )

    export_bundle(
        args.directory,
        model=args.model,
        settings=benchmark_settings(args.model, Backend.TRT16),
        sources={"checkpoint": args.checkpoint},
        export=export,
    )


def _build(args: argparse.Namespace) -> None:
    if not 1 <= args.opt_batch <= args.max_batch:
        raise ValueError("Expected 1 <= opt-batch <= max-batch")
    dimensions = (3, *INPUT_HW)
    build_bundle(
        args.directory,
        {
            "pixels": InputProfile(
                minimum=(1, *dimensions),
                optimum=(args.opt_batch, *dimensions),
                maximum=(args.max_batch, *dimensions),
            )
        },
        device=torch.device(args.device),
        workspace_bytes=args.workspace_mib * 1024**2,
        detailed_profiling=True,
    )


def _smoke(args: argparse.Namespace) -> None:
    weight = WEIGHTS[args.model]
    verify_checkpoint(args.checkpoint, weight)
    upstream = load_official_pipeline(args.model, args.source_root)
    device = torch.device(args.device)
    frame = np.random.default_rng(2490).integers(0, 256, (180, 210, 3), dtype=np.uint8)
    boxes = np.array(
        [[12.75, 10, 82.1, 150], [100, 14.5, 190, 169.3]], dtype=np.float32
    )
    original_frame, original_boxes = frame.copy(), boxes.copy()
    masks = np.zeros((len(boxes), *frame.shape[:2]), dtype=np.uint8)
    for mask, (x1, y1, x2, y2) in zip(masks, boxes, strict=True):
        mask[
            int(np.floor(y1)) : int(np.ceil(y2)), int(np.floor(x1)) : int(np.ceil(x2))
        ] = 1
    rows = tuple(PoseInput(frame, box) for box in boxes)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    with torch.no_grad():
        model = load_model(
            upstream, args.checkpoint, dtype=torch.float32, device=device
        )
        adapter = _infer(upstream, model, Backend.PT32, device, torch.float32)
        observed = adapter(rows, _ParityStages())
        np.testing.assert_array_equal(frame, original_frame)
        np.testing.assert_array_equal(boxes, original_boxes)
        samples = upstream.api.inference_topdown(
            model.model, frame, bboxes=boxes, masks=masks
        )
    fields = {
        "native_keypoints_xy": "keypoints",
        "presence": "keypoints_probs",
        "visibility": "keypoints_visible",
        "expected_oks": "keypoints_oks",
        "normalized_error": "keypoints_error",
        "heatmap_peak": "keypoints_conf",
    }
    differences = {}
    for name, field in fields.items():
        target = np.concatenate(
            [getattr(sample.pred_instances, field) for sample in samples]
        )
        observed_field = observed.extras[name]
        np.testing.assert_allclose(observed_field, target, rtol=1e-5, atol=2e-4)
        differences[name] = float(np.max(np.abs(observed_field - target)))
    write_json_atomic(
        args.receipt,
        {
            "model": args.model,
            "backend": Backend.PT32,
            "device": str(device),
            "torch": torch.__version__,
            "joint_order": JOINT_ORDER,
            "heatmap_hw": HEATMAP_HW,
            "checkpoint_sha256": weight.sha256,
            "parameters": sum(p.numel() for p in model.parameters()),
            "rows": len(rows),
            "max_absolute_difference": differences,
        },
    )


def _inspect(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    runner = load_engine(args.model, args.directory, device)
    pixels = torch.zeros(args.batch, 3, *INPUT_HW, device=device, dtype=torch.float16)
    runner({"pixels": pixels})
    torch.cuda.synchronize(device)
    write_json_atomic(args.receipt, runner.inspect_layers())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export, build, smoke, inspect = (
        actions.add_parser(name) for name in ("export", "build", "smoke", "inspect")
    )
    for command in (export, build, smoke, inspect):
        command.add_argument("--model", choices=tuple(WEIGHTS), required=True)
        command.add_argument("--receipt", type=Path, required=True)
    for command in (export, smoke):
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
    for command in (export, build, inspect):
        command.add_argument("--directory", type=Path, required=True)
    for command in (build, inspect):
        command.add_argument("--device", default="cuda:0")
    build.add_argument("--opt-batch", type=int, required=True)
    build.add_argument("--max-batch", type=int, required=True)
    build.add_argument("--workspace-mib", type=int, required=True)
    inspect.add_argument("--batch", type=int, required=True)
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    if args.action == "smoke":
        _smoke(args)
    elif args.action == "inspect":
        _inspect(args)
    else:
        conversion, operation, device = (
            (Conversion.ONNX_EXPORT, _export, None)
            if args.action == "export"
            else (Conversion.ENGINE_BUILD, _build, args.device)
        )
        result = measure_conversion(conversion, lambda: operation(args), device=device)
        write_json_atomic(args.receipt, asdict(result))
        if result.failure is not None:
            raise SystemExit(result.failure.message)


if __name__ == "__main__":
    main()
