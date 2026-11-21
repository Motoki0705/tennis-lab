"""Prepare the #2303 ViT graph and build a separately verified batch plan."""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from batching import require_dynamic_axis
from benchmark import write_json
from convert import (
    TRT_PRECISIONS,
    WORKSPACE_BYTES,
    build_engine,
    require_queue,
    verify_qdq_scope,
)
from export_dino import graph_files
from inputs import sha256_file
from vision_inputs import CONDITIONS, HMR2, PTQ_RECIPES, VITPOSE, PinnedInputs, read


def verify_batch_axes(path):
    graph = onnx.load(path, load_external_data=False).graph
    for value in (*graph.input, *graph.output):
        require_dynamic_axis(value, 0, "batch")
    return graph


def prepare(data, condition, output):
    exported = output / "export.onnx"
    data.adapter.export(
        data.checkpoint, exported, fp16=condition == "2", dynamic_batch=True
    )
    verify_batch_axes(exported)
    graph, quantization, selected = exported, None, set()
    mode = TRT_PRECISIONS[condition]
    if condition in PTQ_RECIPES:
        graph = output / "quantized.onnx"
        calibration = np.load(data.directory / "calibration.npy", allow_pickle=False)
        shape = "x".join(map(str, data.adapter.engine_shape))
        quantization, selected = data.adapter.quantize(
            exported,
            graph,
            calibration,
            mode,
            PTQ_RECIPES[condition],
            calibration_shapes=f"pixels:{shape}",
        )
        verify_qdq_scope(verify_batch_axes(graph), selected)
    requirements = data.adapter.compute_requirements(exported, mode, selected)
    write_json(output / "precision.json", requirements)
    write_json(
        output / "graph.json",
        {
            "identity": data.identity,
            "condition": condition,
            "graph": graph.name,
            "files": graph_files(graph),
            "precision_manifest_sha256": sha256_file(output / "precision.json"),
            "quantization": quantization,
            "queue_job": os.environ["TENNIS_RUN_ID"],
        },
    )


def build(data, condition, source, output, profile, optimization_level):
    prepared = read(source / "graph.json")
    graph = source / prepared["graph"]
    manifest = source / "precision.json"
    if (
        prepared["identity"] != data.identity
        or prepared["condition"] != condition
        or graph_files(graph) != prepared["files"]
        or sha256_file(manifest) != prepared["precision_manifest_sha256"]
    ):
        raise ValueError("Prepared ViT graph differs from the requested build")
    requirements = read(manifest)
    mode = TRT_PRECISIONS[condition]
    if condition in PTQ_RECIPES:
        verify_qdq_scope(
            verify_batch_axes(graph),
            {
                name
                for name, dtype in requirements["expected_operations"].items()
                if dtype == mode
            },
        )
    plan = output / f"{data.model}.plan"
    evidence = build_engine(
        graph,
        plan,
        required_precision=mode,
        profile_shapes=profile,
        builder_optimization_level=optimization_level,
        min_free_gib=12,
        **requirements,
    )
    write_json(output / "precision.json", requirements)
    write_json(
        output / "conversion.json",
        {
            **prepared,
            "graph_queue_job": prepared["queue_job"],
            "queue_job": os.environ["TENNIS_RUN_ID"],
            "checkpoint_sha256": data.identity["checkpoint_sha256"],
            "input_profile": profile,
            "compute_precision": evidence,
            "workspace_bytes": WORKSPACE_BYTES,
            "builder_optimization_level": optimization_level,
            "engine_sha256": sha256_file(plan),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build"))
    parser.add_argument("--model", choices=(VITPOSE, HMR2), required=True)
    parser.add_argument("--condition", choices=TRT_PRECISIONS, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--batch-profile", nargs=2, type=int)
    parser.add_argument("--builder-optimization-level", choices=range(6), type=int)
    args = parser.parse_args()
    if args.condition not in CONDITIONS[args.model]:
        parser.error("Condition unavailable for this model")
    if args.stage == "build" and (args.source is None or args.batch_profile is None):
        parser.error("The build stage requires a source graph and batch profile")
    require_queue()
    data = PinnedInputs(args.model, args.inputs, verify=True)
    args.output.mkdir(parents=True)
    if args.stage == "prepare":
        prepare(data, args.condition, args.output)
    else:
        build(
            data,
            args.condition,
            args.source,
            args.output,
            data.profile(*args.batch_profile),
            args.builder_optimization_level,
        )


if __name__ == "__main__":
    main()
