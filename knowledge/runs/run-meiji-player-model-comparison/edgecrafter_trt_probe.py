"""Expose ECDet-L intermediates in a diagnostic copy of the typed ONNX graph."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from edgecrafter import ASSETS, HEADS, INPUT_NAME
from edgecrafter_diagnose import (
    BATCHES,
    FRAMES,
    capture,
    compare_images,
    engine_bundle,
    head_stats,
    tensor_difference,
)
from numpy.typing import NDArray
from speed_inputs import InputSet

from src.submodules.models.ecpose.inference import preprocess_images
from src.submodules.models.ecpose.releases import RELEASES
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import ONNX_FILE, EngineBundle
from src.utils.models.tensorrt_build import build_bundle, export_bundle
from src.utils.models.tensorrt_runtime import TensorRTRunner

MODEL = "ecdet-l-coco"
PROBE_NODES = (
    "/model/backbone/backbone/patch_embed/proj/Conv",
    "/model/backbone/backbone/blocks.0/Add_1",
    "/model/backbone/backbone/blocks.5/Add_1",
    "/model/backbone/backbone/blocks.11/Add_1",
    "/model/backbone/projector.0/conv_bn_fused/Conv",
    "/model/encoder/fpn_blocks.1/cv4/act/Mul",
    "/model/decoder/enc_score_head/Add",
    "/model/decoder/TopK",
    "/model/decoder/GatherElements_1",
)
BATCH_NAMES = ("pair1_b", "pair1_ab", "pair1_ba", "pair1_ab_repeat")


def add_probe_outputs(
    model: Any, node_names: tuple[str, ...]
) -> tuple[Any, tuple[str, ...]]:
    onnx = import_module("onnx")
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    nodes = {node.name: node for node in inferred.graph.node}
    values = {value.name: value for value in inferred.graph.value_info}
    names = tuple(output for node in node_names for output in nodes[node].output)
    if len(names) != len(set(names)) or set(names) & {
        v.name for v in inferred.graph.output
    }:
        raise ValueError("The probe must add each intermediate value exactly once")
    inferred.graph.output.extend(values[name] for name in names)
    return inferred, names


def export_probe(original: EngineBundle, directory: Path) -> tuple[str, ...]:
    onnx = import_module("onnx")
    source = original.root / ONNX_FILE
    graph, names = add_probe_outputs(onnx.load(source), PROBE_NODES)
    export_bundle(
        directory,
        model=MODEL,
        settings={**original.export.settings, "diagnostic_outputs": list(names)},
        sources={"weights": ASSETS / RELEASES[MODEL].filename, "base_onnx": source},
        export=lambda path: onnx.save(graph, path),
    )
    return names


def probe(inputs: Path, output: Path, archive: Path) -> None:
    device = torch.device("cuda:0")
    original = engine_bundle(MODEL)
    previous = json.loads((archive / "cuda.json").read_bytes())["models"][MODEL]
    raw_path = archive / previous["raw_heads"]["path"]
    if (
        previous["engine_sha256"] != original.engine.plan_sha256
        or dual_sha256(raw_path) != previous["raw_heads"]["sha256"]
    ):
        raise ValueError("The archived raw heads refer to a different plan")

    fixed = InputSet(inputs)
    pixels, sizes = preprocess_images(tuple(fixed.frame(frame) for frame in FRAMES[:2]))
    tensors = {name: pixels[list(BATCHES[name])].contiguous() for name in BATCH_NAMES}
    input_hashes = {
        name: sha256(value.numpy().tobytes()).hexdigest()
        for name, value in tensors.items()
    }
    for name in ("pair1_b", "pair1_ab", "pair1_ba"):
        if input_hashes[name] != previous["batches"][name]["input_tensor_sha256"]:
            raise ValueError(
                f"Fixed input bytes differ from the original diagnosis for {name}"
            )

    directory = output / "engine"
    names = (*HEADS, *export_probe(original, directory))
    build_bundle(
        directory,
        original.engine.profiles,
        device=device,
        workspace_bytes=1536 * 1024**2,
    )
    instrumented = EngineBundle.read(directory)
    runtime = TensorRTRunner(instrumented, device)
    ort = import_module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(directory / ONNX_FILE),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("The alternate ONNX interpreter must run on CPU")

    pairs = {}
    candidate_arrays = {}
    cpu_arrays: dict[str, dict[str, NDArray[Any]]] = {}
    with (
        np.load(raw_path, allow_pickle=False) as old,
        torch.no_grad(),
        torch.cuda.stream(torch.cuda.Stream(device=device)),
    ):
        for name, tensor in tensors.items():
            reference = (
                cpu_arrays["pair1_ab"]
                if name == "pair1_ab_repeat"
                else dict(
                    zip(
                        names,
                        session.run(list(names), {INPUT_NAME: tensor.numpy()}),
                        strict=True,
                    )
                )
            )
            cpu_arrays[name] = reference
            values = runtime({INPUT_NAME: tensor.to(device)})
            candidate = {key: values[key].cpu().numpy() for key in names}
            candidate_arrays[name] = candidate
            selection = list(BATCHES[name])
            frames = tuple(FRAMES[index] for index in selection)
            base_name = "pair1_ab" if name == "pair1_ab_repeat" else name
            old_heads = (
                torch.from_numpy(old[f"trt16_{base_name}_{HEADS[0]}"]),
                torch.from_numpy(old[f"trt16_{base_name}_{HEADS[1]}"]),
            )
            new_heads = (
                torch.from_numpy(candidate[HEADS[0]]),
                torch.from_numpy(candidate[HEADS[1]]),
            )
            pairs[name] = {
                "frames": frames,
                "input_sha256": input_hashes[name],
                "original_vs_instrumented_trt_heads": compare_images(
                    capture(MODEL, old_heads, sizes[selection], frames),
                    capture(MODEL, new_heads, sizes[selection], frames),
                ),
                "onnx_cpu_vs_instrumented_trt": {
                    key: {
                        "shape": candidate[key].shape,
                        "cpu": head_stats(reference[key]),
                        "trt": head_stats(candidate[key]),
                        "difference": tensor_difference(
                            reference[key], candidate[key], label=key
                        ),
                    }
                    for key in names
                },
            }

    report = {
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_plan_sha256": original.engine.plan_sha256,
        "source_onnx_sha256": original.export.onnx_files[ONNX_FILE],
        "probe_plan_sha256": instrumented.engine.plan_sha256,
        "probe_onnx_sha256": instrumented.export.onnx_files[ONNX_FILE],
        "cpu_onnxruntime_version": ort.__version__,
        "profile": original.engine.profiles[INPUT_NAME].model_dump(),
        "tap_order": [*names[2:], *HEADS],
        "input_manifest_sha256": fixed.sha256,
        "cases": pairs,
        "batch_order": {
            backend: {
                key: tensor_difference(
                    arrays["pair1_ab"][key], arrays["pair1_ba"][key][::-1], label=key
                )
                for key in names
            }
            for backend, arrays in (
                ("onnx_cpu", cpu_arrays),
                ("trt16", candidate_arrays),
            )
        },
        "trt_ab_repeat": {
            key: tensor_difference(
                candidate_arrays["pair1_ab"][key],
                candidate_arrays["pair1_ab_repeat"][key],
                label=key,
            )
            for key in names
        },
    }
    write_json_atomic(output / "probes.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.inputs, args.output, args.archive)


if __name__ == "__main__":
    main()
