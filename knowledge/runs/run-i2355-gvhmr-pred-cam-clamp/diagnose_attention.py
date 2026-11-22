"""Isolate the first-block attention failure and try exact mask rewrites.

From one static ONNX, this builds graph variants (the source, two exact mask
rewrites, and three non-equivalent diagnostic mask removals).  Each variant is
sliced at the first block's attention boundaries and also built whole, every
build in its own child process.
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

import onnx  # noqa: E402
from benchmark import write_json  # noqa: E402
from convert import require_queue  # noqa: E402
from diagnose_block import (  # noqa: E402
    BLOCK_OUTPUT,
    EMBEDDING,
    FULLY_MASKED,
    KEY_ROPE,
    LOCAL_MASK,
    PADDING_MASK,
    QUERY_ROPE,
    SCORES,
    VALUE,
    WEIGHTED_VALUE,
    run_variant,
)
from graph_slices import infer_tensor_specs, slice_graph  # noqa: E402
from rewrite_attention_mask import (  # noqa: E402
    remove_masks_for_diagnosis,
    rewrite_attention_mask,
)

BLOCKS = 12
CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pv-from-embedding", WEIGHTED_VALUE, (EMBEDDING, LOCAL_MASK, PADDING_MASK)),
    (
        "pv-from-qkv",
        WEIGHTED_VALUE,
        (QUERY_ROPE, KEY_ROPE, VALUE, LOCAL_MASK, PADDING_MASK),
    ),
    ("pv-from-scores", WEIGHTED_VALUE, (SCORES, VALUE, LOCAL_MASK, PADDING_MASK)),
    ("pv-from-masked-scores", WEIGHTED_VALUE, (FULLY_MASKED, VALUE)),
    ("block0-output", BLOCK_OUTPUT, (EMBEDDING, LOCAL_MASK, PADDING_MASK)),
    ("full-prefix-block0", BLOCK_OUTPUT, ()),
)


def make_variants(source: Path, root: Path) -> dict[str, tuple[Path, dict]]:
    directory = root / "variants"
    variants: dict[str, tuple[Path, dict]] = {"source": (source, {"equivalent": True})}
    for mode in ("drop-expand", "additive"):
        target = directory / f"{mode}.onnx"
        report = rewrite_attention_mask(source, target, mode=mode, expected=2 * BLOCKS)
        variants[mode] = (target, {**report, "equivalent": True})
    for name, fills, count in (
        ("no-local-mask", ("Where_1",), BLOCKS),
        ("no-padding-mask", ("Where_3",), BLOCKS),
        ("no-masks", ("Where_1", "Where_3"), 2 * BLOCKS),
    ):
        target = directory / f"{name}.onnx"
        variants[name] = (
            target,
            remove_masks_for_diagnosis(source, target, fills=fills, expected=count),
        )
    return variants


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
        "case_order": [name for name, *_rest in CASES] + ["full-graph"],
        "variants": {},
        "graphs": {},
        "dropped_boundaries": {},
        "builds": {},
    }
    variants = make_variants(source, root)
    report["variants"] = {name: info for name, (_path, info) in variants.items()}
    write_json(root / "diagnosis.json", report)

    for variant, (graph_file, _info) in variants.items():
        model = onnx.load(graph_file, load_external_data=False)
        produced = {name for node in model.graph.node for name in node.output}
        del model
        cases = []
        for case, output, promoted in CASES:
            # A removed mask also removes its producer; drop it from the boundary.
            if missing := sorted({output, *promoted} - produced):
                report["dropped_boundaries"][f"{variant}-{case}"] = missing
            if output not in produced:
                continue
            cases.append(
                (case, output, tuple(item for item in promoted if item in produced))
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
