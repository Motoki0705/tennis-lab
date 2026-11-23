"""Check exported GVHMR mask parity and probe the TensorRT engine at profile boundaries."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.submodules.models.gvhmr.onnx_attention import ATTENTION_MASK_ENCODING
from src.submodules.models.gvhmr.onnx_export import MASK_REFERENCE_FILE
from src.submodules.models.gvhmr.tensorrt_validation import (
    probe_inputs,
    verify_bit_equal,
    verify_fp16_layers,
    verify_outputs,
)
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathContractError,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import ONNX_FILE, EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.gvhmr_tensorrt_qualification",
    fields=(
        BoundaryPathField(
            "bundle", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
            must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "progress", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
            must_exist=True, allow_role_root=True, many=True, required=False,
        ),
        BoundaryPathField("report", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
    ),
)

Shape = tuple[int, int]


def _recorded_shapes(progress: Path) -> set[Shape]:
    document = json.loads(progress.read_text())
    counts = document["batch_shapes"]["GVHMR-network"]
    result = set()
    for shape in counts:
        batch, frames, features = json.loads(shape)
        if features != 1024:
            raise ValueError(f"Unexpected GVHMR batch shape in {progress}: {shape}")
        result.add((batch, frames))
    return result


def _ort_session(ort: Any, path: Path, threads: int) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _ort_outputs(session: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = session.run(None, inputs)
    return {output.name: value for output, value in zip(session.get_outputs(), values, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ort-threads", type=int, default=2)
    parser.add_argument("--shape", action="append", nargs=2, type=int, default=[], metavar=("BATCH", "FRAMES"))
    parser.add_argument("--progress", action="append", type=Path, default=[], help="Add all GVHMR shapes from a video batch")
    parser.add_argument("--ort-only", action="store_true", help="Compare ONNX outputs at explicitly selected shapes")
    args = parser.parse_args()
    if args.ort_threads < 1:
        parser.error("--ort-threads must be positive")

    report_path = PROJECT_ROOT / args.report.expanduser()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=report_path.parent.resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    arguments: dict[str, object] = {
        "bundle": PROJECT_ROOT / args.bundle.expanduser(),
        "report": report_path,
    }
    if args.progress:
        arguments["progress"] = [PROJECT_ROOT / path.expanduser() for path in args.progress]
    paths = PATH_BOUNDARY.validate(
        arguments, resolver=PathResolver(roots), independent_artifact_inputs=True,
    )
    root = paths.declared("bundle").path
    report_path = paths.declared("report").path
    if report_path.exists() and not report_path.is_file():
        raise PathContractError(f"Qualification report output must be a file: {report_path}")

    bundle = EngineBundle.read(root)
    if bundle.export.settings.get("attention_mask") != ATTENTION_MASK_ENCODING:
        raise ValueError("Qualification requires the finite additive mask GVHMR export")
    mask_reference = root / MASK_REFERENCE_FILE
    profile = bundle.engine.profiles["obs"]
    used = {tuple(shape) for shape in args.shape}
    if "progress" in paths:
        for progress in paths.declared_many("progress"):
            used.update(_recorded_shapes(progress.path))
    boundaries = {(batch, frames) for batch, frames, *_ in profile.shapes}
    boundaries.update((batch, frames) for batch in (profile.minimum[0], profile.maximum[0])
                      for frames in (profile.minimum[1], profile.maximum[1]))
    if profile.minimum[1] <= 177 <= profile.maximum[1]:
        boundaries.update((batch, 177) for batch in (profile.minimum[0], profile.maximum[0]))
    cases = used if args.ort_only else used | boundaries
    if not cases:
        parser.error("Pass --shape or --progress to qualify explicit shapes")
    for batch, frames in cases:
        profile.require((batch, frames, *profile.optimum[2:]), tensor="obs")

    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")
    graph_path = root / ONNX_FILE
    graph = onnx.load(graph_path, load_external_data=False).graph
    operations = {node.name for node in graph.node if node.op_type in {"MatMul", "Gemm", "Conv"}}
    attention_blocks = sum(node.op_type == "Softmax" and "/attn/" in node.name for node in graph.node)
    del graph
    device = torch.device(args.device)
    runner = None if args.ort_only else TensorRTRunner(bundle, device)
    original = _ort_session(ort, mask_reference, args.ort_threads)
    rewritten = _ort_session(ort, graph_path, args.ort_threads)
    report: dict[str, Any] = {
        "status": "running", "engine": bundle.engine.model_dump(mode="json"),
        "export": bundle.export.model_dump(mode="json"),
        "mask_reference_sha256": dual_sha256(mask_reference), "onnxruntime": ort.__version__,
        "ort_provider": "CPUExecutionProvider", "shapes": {},
    }
    for batch, frames in sorted(cases):
        shape = f"B{batch}_L{frames}"
        row: dict[str, Any] = {}
        report["shapes"][shape] = row
        print(f"Qualifying GVHMR {shape}", flush=True)
        try:
            inputs = probe_inputs(batch, frames)
            before, after = _ort_outputs(original, inputs), _ort_outputs(rewritten, inputs)
            verify_outputs(before, batch=batch, frames=frames)
            verify_outputs(after, batch=batch, frames=frames)
            verify_bit_equal(before, after)
            row["onnx_mask_parity"] = "bit_equal"
            if runner is not None and (batch, frames) in boundaries:
                outputs = {name: value.cpu().numpy() for name, value in runner(
                    {name: torch.from_numpy(array) for name, array in inputs.items()}).items()}
                verify_outputs(outputs, batch=batch, frames=frames)
                row["tensorrt_finite"] = True
                layers = runner.inspect_layers()
                inspector = report_path.with_name(f"{report_path.stem}-{shape}-layers.json")
                write_json_atomic(inspector, layers)
                row["inspector"] = inspector.name
                row["tensorrt_layer_precision"] = verify_fp16_layers(layers, operations, attention_blocks=attention_blocks)
        except Exception as error:
            report["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json_atomic(report_path, report)
    report["status"] = "complete"
    write_json_atomic(report_path, report)


if __name__ == "__main__":
    main()
