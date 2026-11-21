"""Audit the original #2305 measured graphs against the current Q/DQ scope rule."""

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
from benchmark import write_json
from convert import (
    QUANTIZABLE_OPS,
    dequantized_tensors,
    quantized_matrix_ops,
    verify_qdq_scope,
)
from convert_gvhmr import BLOCK_SCOPE, major_operations
from inputs import sha256_file
from sweep import GVHMR


def audit_graph(path):
    onnx.checker.check_model(str(path))
    model = onnx.load(path)
    graph = model.graph
    selected = {
        name for name in major_operations(model, path) if BLOCK_SCOPE.search(name)
    }
    observed = quantized_matrix_ops(graph)
    try:
        verify_qdq_scope(graph, selected)
    except RuntimeError as exc:
        scope = {"passed": False, "reason": str(exc).split(": ", 1)[0]}
    else:
        scope = {"passed": True}
    dequantized = dequantized_tensors(graph)
    outside_inputs = Counter()
    outside_nodes = Counter()
    examples = {}
    for node in graph.node:
        outside = [
            i
            for i, value in enumerate(node.input)
            if value in dequantized
            and (
                node.name not in selected
                or node.op_type not in QUANTIZABLE_OPS
                or i >= 2
            )
        ]
        if outside:
            outside_inputs[node.op_type] += len(outside)
            outside_nodes[node.op_type] += 1
            examples.setdefault(
                node.op_type, {"node": node.name, "input_indices": outside}
            )
    return {
        "onnx_checker_passed": True,
        "verify_qdq_scope": scope,
        "selected_compute_nodes": len(selected),
        "observed_dual_dq_compute_nodes": len(observed),
        "missing_selected_compute_nodes": sorted(selected - observed),
        "unexpected_dual_dq_compute_nodes": sorted(observed - selected),
        "qdq_nodes": dict(
            Counter(
                node.op_type
                for node in graph.node
                if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
            )
        ),
        "outside_dq_inputs_by_op": dict(sorted(outside_inputs.items())),
        "outside_dq_consumer_nodes_by_op": dict(sorted(outside_nodes.items())),
        "outside_dq_examples": dict(sorted(examples.items())),
        "dq_graph_outputs": [
            output.name for output in graph.output if output.name in dequantized
        ],
    }


def audit(artifact_root):
    validation = json.loads((GVHMR / "validation.json").read_text())
    conversions = validation["successful_tensor_rt_conversions_and_boundaries"]
    cases = json.loads((GVHMR / "metrics.json").read_text())["conditions"]
    report = {"schema_version": 1, "source_issue": 2305, "conditions": {}}
    for condition in ("3a", "3b"):
        entry = conversions[condition]
        receipt = entry["conversion_receipt"]
        folder = Path(receipt["path"]).parent.name
        local_receipt = artifact_root / folder / "conversion.json"
        path = local_receipt.with_name("quantized.onnx")
        expected = entry["conversion"]
        if (
            sha256_file(local_receipt) != receipt["sha256"]
            or sha256_file(path) != expected["graph_sha256"]
            or set(cases[condition]) != {"145", "177"}
            or any(
                case["measurement"]["engine_sha256"] != expected["engine_sha256"]
                for case in cases[condition].values()
            )
        ):
            raise ValueError(
                f"The graph or conversion receipt differs from the measured {condition} case"
            )
        report["conditions"][condition] = {
            "measured_graph": str(Path(receipt["path"]).with_name(path.name)),
            "graph_sha256": expected["graph_sha256"],
            "measurement_engine_sha256": expected["engine_sha256"],
            "conversion_receipt_sha256": receipt["sha256"],
            "queue_job": f"{entry['queue']['queue_run_id']}.job",
            **audit_graph(path),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.output, audit(args.artifact_root))


if __name__ == "__main__":
    main()
