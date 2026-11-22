"""Isolate the first GVHMR transformer block's TensorRT build failure.

Each ``--source LABEL=PATH`` is a static GVHMR ONNX (for example FP16 B=2,
FP32 B=2 and FP16 B=1).  Every case below is sliced from every source and built
in its own child process, so a crashing build still leaves the next result.
``--probe-source`` additionally rebuilds the full first-block prefix with each
builder option variant in ``PROBES``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
REPO = HERE.parents[2]
COMMON = RUNS / "run-i2297-clipreid-tensorrt"
GVHMR = RUNS / "run-i2305-gvhmr-tensorrt"
for dependency in (REPO, COMMON, GVHMR, HERE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from benchmark import write_json  # noqa: E402
from convert import require_queue  # noqa: E402
from graph_slices import infer_tensor_specs, slice_graph  # noqa: E402

BUILD = HERE / "variant_build.py"

EMBEDDING = "/network/Add_3_output_0"
LOCAL_MASK = "/Or_output_0"
PADDING_MASK = "/network/Not_2_output_0"
NORM1 = "/network/blocks.0/norm1/LayerNormalization_output_0"
QUERY_ROPE = "/network/blocks.0/attn/Concat_1_output_0"
KEY_ROPE = "/network/blocks.0/attn/Concat_3_output_0"
VALUE = "/network/blocks.0/attn/Transpose_2_output_0"
SCORES = "/network/blocks.0/attn/Div_output_0"
LOCAL_MASKED = "/network/blocks.0/attn/Where_1_output_0"
FULLY_MASKED = "/network/blocks.0/attn/Where_3_output_0"
PROBABILITIES = "/network/blocks.0/attn/Softmax_output_0"
WEIGHTED_VALUE = "/network/blocks.0/attn/MatMul_1_output_0"
ATTENTION_PROJECTION = "/network/blocks.0/attn/proj/Add_output_0"
ATTENTION_RESIDUAL = "/network/blocks.0/Add_output_0"
NORM2 = "/network/blocks.0/norm2/LayerNormalization_output_0"
MLP_FC1 = "/network/blocks.0/mlp/fc1/Add_output_0"
MLP_ACTIVATION = "/network/blocks.0/mlp/act/Mul_5_output_0"
MLP_FC2 = "/network/blocks.0/mlp/fc2/Add_output_0"
BLOCK_OUTPUT = "/network/blocks.0/Add_1_output_0"
FULL_PREFIX = "full-prefix-block0"

PROBES: tuple[tuple[str, dict[str, object]], ...] = (
    ("opt1", {"builder_optimization_level": 1}),
    ("opt2", {"builder_optimization_level": 2}),
    ("opt4", {"builder_optimization_level": 4}),
    ("opt5", {"builder_optimization_level": 5}),
    ("no-tactic-sources", {"tactic_sources": []}),
    ("no-aux-streams", {"max_aux_streams": 0}),
    ("one-tactic", {"max_num_tactics": 1}),
    ("tiling-fast", {"tiling_optimization_level": "FAST"}),
    ("distributive-independence", {"set_flags": ["DISTRIBUTIVE_INDEPENDENCE"]}),
    ("no-compilation-cache", {"set_flags": ["DISABLE_COMPILATION_CACHE"]}),
)


def case_definitions() -> tuple[tuple[str, str, tuple[str, ...], int], ...]:
    """Describe prefixes, isolated cores, and boundary-interaction comparisons."""
    isolated = (EMBEDDING, LOCAL_MASK, PADDING_MASK)
    return (
        ("local-mask-only", LOCAL_MASK, (), 3),
        ("padding-mask-only", PADDING_MASK, (), 3),
        ("block0-norm1", NORM1, isolated, 3),
        ("block0-query-rope", QUERY_ROPE, isolated, 3),
        ("block0-key-rope", KEY_ROPE, isolated, 3),
        ("block0-qk-score", SCORES, isolated, 3),
        ("block0-local-mask-applied", LOCAL_MASKED, isolated, 3),
        ("block0-padding-mask-applied", FULLY_MASKED, isolated, 3),
        ("block0-softmax", PROBABILITIES, isolated, 3),
        ("block0-weighted-value", WEIGHTED_VALUE, isolated, 3),
        ("block0-attention-projection", ATTENTION_PROJECTION, isolated, 3),
        ("block0-attention-residual", ATTENTION_RESIDUAL, isolated, 3),
        ("block0-norm2", NORM2, isolated, 3),
        ("block0-mlp-fc1", MLP_FC1, isolated, 3),
        ("block0-mlp-activation", MLP_ACTIVATION, isolated, 3),
        ("block0-mlp-fc2", MLP_FC2, isolated, 3),
        ("block0-output", BLOCK_OUTPUT, isolated, 3),
        ("qk-core", SCORES, (QUERY_ROPE, KEY_ROPE), 3),
        ("local-mask-core", LOCAL_MASKED, (SCORES, LOCAL_MASK), 3),
        (
            "padding-mask-core",
            FULLY_MASKED,
            (LOCAL_MASKED, PADDING_MASK),
            3,
        ),
        ("softmax-core", PROBABILITIES, (FULLY_MASKED,), 3),
        (
            "weighted-value-core",
            WEIGHTED_VALUE,
            (PROBABILITIES, VALUE),
            3,
        ),
        ("generated-masks-plus-block", BLOCK_OUTPUT, (EMBEDDING,), 3),
        (
            "embedding-plus-block-promoted-masks",
            BLOCK_OUTPUT,
            (LOCAL_MASK, PADDING_MASK),
            3,
        ),
        (FULL_PREFIX, BLOCK_OUTPUT, (), 3),
        ("block0-output-opt0", BLOCK_OUTPUT, isolated, 0),
        ("full-prefix-block0-opt0", BLOCK_OUTPUT, (), 0),
    )


def parse_source(text: str) -> tuple[str, Path]:
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {text!r}")
    return label, Path(path).resolve(strict=True)


def log_summary(text: str) -> dict[str, object]:
    timing = [line.strip() for line in text.splitlines() if "Timing Runner:" in line]
    costs = [
        line.strip() for line in text.splitlines() if "Computing costs for" in line
    ]
    return {
        "segmentation_fault_reported": "Segmentation fault" in text,
        "timing_runners": len(timing),
        "last_timing_runner": timing[-1] if timing else None,
        "last_computing_costs": costs[-1] if costs else None,
    }


def run_variant(
    report: dict[str, object],
    root: Path,
    graph: Path,
    name: str,
    options: dict[str, object],
) -> dict[str, object]:
    destination = root / "cases" / name
    plan = destination / "model.plan"
    receipt = destination / "build.json"
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        "-X",
        "faulthandler",
        str(BUILD),
        "--onnx",
        str(graph),
        "--plan",
        str(plan),
        "--report",
        str(receipt),
        "--options",
        json.dumps(options, sort_keys=True),
    ]
    started = time.monotonic()
    with log.open("x") as stream:
        print(subprocess.list2cmdline(command), file=stream, flush=True)
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(map(str, (REPO, COMMON, GVHMR, HERE))),
            },
        )
    result = {
        "exit_code": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "graph": str(graph),
        "options": options,
        "log": str(log.relative_to(root)),
        "plan_published": plan.is_file(),
        "receipt_published": receipt.is_file(),
        **log_summary(log.read_text(errors="replace")),
    }
    report["builds"][name] = result
    write_json(root / "diagnosis.json", report)
    print(f"{name}: exit {completed.returncode}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, action="append", type=parse_source)
    parser.add_argument("--probe-source", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = dict(args.source)
    if len(sources) != len(args.source):
        parser.error("Source labels must be unique")
    if args.probe_source not in sources:
        parser.error(f"Unknown probe source {args.probe_source!r}")
    require_queue()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")

    definitions = case_definitions()
    tensor_names = {
        name
        for _case, output, promoted, _optimization in definitions
        for name in (output, *promoted)
    }
    report: dict[str, object] = {
        "schema_version": 2,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "sources": {label: str(path) for label, path in sources.items()},
        "case_order": [name for name, *_rest in definitions],
        "probe_source": args.probe_source,
        "probe_order": [name for name, _options in PROBES],
        "graphs": {},
        "builds": {},
    }
    write_json(root / "diagnosis.json", report)

    specs_by_source = {
        label: infer_tensor_specs(source, tensor_names)
        for label, source in sources.items()
    }
    graphs: dict[str, Path] = {}
    for label, source in sources.items():
        specs = specs_by_source[label]
        for case, output, promoted_names, optimization in definitions:
            name = f"{label}-{case}"
            output_type, output_shape = specs[output]
            graph = root / "graphs" / f"{name}.onnx"
            report["graphs"][name] = slice_graph(
                source,
                graph,
                output_name=output,
                output_type=output_type,
                output_shape=output_shape,
                promoted_inputs={item: specs[item] for item in promoted_names},
            )
            write_json(root / "diagnosis.json", report)
            graphs[name] = graph
            run_variant(
                report,
                root,
                graph,
                name,
                {"builder_optimization_level": optimization},
            )

    prefix = graphs[f"{args.probe_source}-{FULL_PREFIX}"]
    for probe, options in PROBES:
        run_variant(
            report, root, prefix, f"{args.probe_source}-{FULL_PREFIX}-{probe}", options
        )
    report["status"] = "done"
    write_json(root / "diagnosis.json", report)


if __name__ == "__main__":
    main()
