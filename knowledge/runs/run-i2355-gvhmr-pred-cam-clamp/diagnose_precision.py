"""Build FP16 GVHMR graphs whose attention operators are pinned to FP32.

Each variant pins one operator set in all 12 blocks (``pin_precision``), then
the first-block attention slice, the first-block prefix and the whole graph are
built in separate child processes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
for dependency in (
    HERE.parents[2],
    RUNS / "run-i2297-clipreid-tensorrt",
    RUNS / "run-i2305-gvhmr-tensorrt",
    HERE,
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from benchmark import write_json  # noqa: E402
from convert import require_queue  # noqa: E402
from diagnose_block import (  # noqa: E402
    BLOCK_OUTPUT,
    EMBEDDING,
    LOCAL_MASK,
    PADDING_MASK,
    run_variant,
)
from graph_slices import infer_tensor_specs, slice_graph  # noqa: E402
from pin_precision import pin_fp32  # noqa: E402

BLOCKS = 12
PINNED = {
    "pin-pv": ("MatMul_1",),
    "pin-softmax": ("Softmax",),
    "pin-softmax-pv": ("Softmax", "MatMul_1"),
    "pin-attention": ("MatMul", "Div", "Where_1", "Where_3", "Softmax", "MatMul_1"),
}


def pv_output(variant: str) -> str:
    """The first block's P*V output, renamed when MatMul_1 is pinned."""
    name = "/network/blocks.0/attn/MatMul_1_output_0"
    return f"{name}__fp32" if "MatMul_1" in PINNED[variant] else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_queue()
    source = args.source.resolve(strict=True)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "source": str(source),
        "variants": {},
        "graphs": {},
        "builds": {},
    }
    variants = {}
    for variant, suffixes in PINNED.items():
        target = root / "variants" / f"{variant}.onnx"
        report["variants"][variant] = pin_fp32(
            source, target, suffixes=suffixes, blocks=BLOCKS
        )
        variants[variant] = target
    write_json(root / "diagnosis.json", report)

    isolated = (EMBEDDING, LOCAL_MASK, PADDING_MASK)
    for variant, graph_file in variants.items():
        cases = (
            ("pv-from-embedding", pv_output(variant), isolated),
            ("block0-output", BLOCK_OUTPUT, isolated),
            ("full-prefix-block0", BLOCK_OUTPUT, ()),
        )
        specs = infer_tensor_specs(
            graph_file, {name for _c, out, pro in cases for name in (out, *pro)}
        )
        for case, output, promoted in cases:
            name = f"{variant}-{case}"
            graph = root / "graphs" / f"{name}.onnx"
            report["graphs"][name] = slice_graph(
                graph_file,
                graph,
                output_name=output,
                output_type=specs[output][0],
                output_shape=specs[output][1],
                promoted_inputs={item: specs[item] for item in promoted},
            )
            write_json(root / "diagnosis.json", report)
            run_variant(report, root, graph, name, {"builder_optimization_level": 3})
        run_variant(
            report,
            root,
            graph_file,
            f"{variant}-full-graph",
            {"builder_optimization_level": 3},
        )
    report["status"] = "done"
    write_json(root / "diagnosis.json", report)


if __name__ == "__main__":
    main()
