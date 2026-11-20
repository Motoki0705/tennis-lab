"""Paired CLIP calibration experiments and activation Q/DQ interventions."""

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from benchmark import normalize, verify_inputs, write_json
from convert import constant_tensors, linear_names, modelopt_ptq, require_queue
from inputs import sha256_file

BLOCK = re.compile(r"/resblocks/resblocks\.(\d+)/")
MATRIX = {"MatMul", "Gemm"}
NONLINEAR = {"Add", "LayerNormalization", "Softmax", "Gelu", "Erf", "Mul", "Div"}
LINEAR_GROUPS = {
    "qkv": ("/attn/MatMul",),
    "attention-out": ("/attn/Gemm",),
    "mlp-in": ("/mlp/c_fc/MatMul",),
    "mlp-out": ("/mlp/c_proj/MatMul",),
    "normalized-inputs": ("/attn/MatMul", "/mlp/c_fc/MatMul"),
    "without-mlp-out": ("/attn/MatMul", "/attn/Gemm", "/mlp/c_fc/MatMul"),
}


def selected_nodes(graph, scope):
    if scope == "default":
        return None
    if scope not in {"matmul", "linear", *LINEAR_GROUPS}:
        raise ValueError(f"Unknown CLIP quantization scope: {scope}")
    linear = linear_names(graph)
    return [
        node.name
        for node in graph.node
        if BLOCK.search(node.name)
        and (node.op_type in MATRIX if scope == "matmul" else node.name in linear)
        and (scope not in LINEAR_GROUPS or node.name.endswith(LINEAR_GROUPS[scope]))
    ]


def evaluate(path, samples):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}[
        session.get_inputs()[0].type
    ]
    return np.concatenate(
        [
            session.run(["features"], {"pixels": sample[None].astype(dtype)})[0]
            for sample in samples
        ]
    )


def cosine_summary(test, baseline):
    cosine = np.clip(np.sum(normalize(test) * normalize(baseline), axis=-1), -1, 1)
    return {
        "p05": float(np.quantile(cosine, 0.05)),
        "min": float(cosine.min()),
        "median": float(np.median(cosine)),
        "per_crop": cosine.tolist(),
    }


def qdq_edges(graph):
    producer = {out: node for node in graph.node for out in node.output}
    for node in graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        for i, name in enumerate(node.input):
            dq = producer.get(name)
            if dq is not None and dq.op_type == "DequantizeLinear":
                q = producer.get(dq.input[0])
                if q is not None and q.op_type == "QuantizeLinear":
                    yield node, i, q.input[0]


def interventions(graph):
    linear = linear_names(graph)
    yield "patch", lambda node: node.op_type == "Conv"
    yield "nonlinear_residual", lambda node: node.op_type in NONLINEAR
    yield "residual_add", lambda node: node.op_type == "Add"
    yield "layer_norm", lambda node: node.op_type == "LayerNormalization"
    yield "quick_gelu", lambda node: "/mlp/gelu/" in node.name
    yield "softmax", lambda node: node.op_type == "Softmax"
    yield (
        "dynamic_attention",
        lambda node: node.op_type == "MatMul" and node.name not in linear,
    )
    yield "weighted_linear", lambda node: node.name in linear
    for block in range(12):
        prefix = f"/resblocks/resblocks.{block}/"
        yield f"block_{block:02d}", lambda node, prefix=prefix: prefix in node.name


def ablate(float_graph, legacy_graph, output, held_out, baseline, report):
    report["variants"].update(
        fp32_export=cosine_summary(evaluate(float_graph, held_out), baseline),
        legacy_full=cosine_summary(evaluate(legacy_graph, held_out), baseline),
    )
    report["interventions"] = {}
    original_graph = onnx.load(legacy_graph)
    constants = constant_tensors(original_graph.graph)
    edges = list(qdq_edges(original_graph.graph))
    activation = [edge for edge in edges if edge[2] not in constants]
    static = [edge for edge in edges if edge[2] in constants]
    report["legacy_quantized_activation_consumers"] = dict(
        Counter(node.op_type for node, _, _ in activation)
    )
    report["legacy_quantized_static_consumers"] = dict(
        Counter(node.op_type for node, _, _ in static)
    )
    temporary = output / "intervention.onnx"
    trials = [
        ("all_activation_edges", activation, []),
        ("all_static_edges", [], static),
        *[
            (name, [edge for edge in activation if pick(edge[0])], [])
            for name, pick in interventions(original_graph.graph)
        ],
    ]
    for name, runtime_edges, static_edges in trials:
        changed = [
            (node, idx, node.input[idx], source)
            for node, idx, source in (*runtime_edges, *static_edges)
        ]
        print(f"Legacy Q/DQ ablation {name}: {len(changed)} edges", flush=True)
        for node, idx, _, source in changed:
            node.input[idx] = source
        if changed:
            onnx.save(original_graph, temporary)
            stats = cosine_summary(evaluate(temporary, held_out), baseline)
        else:
            stats = report["variants"]["legacy_full"]
        report["interventions"][name] = {
            "activation_edges_restored": len(runtime_edges),
            "static_edges_restored": len(static_edges),
            **stats,
        }
        for node, idx, quantized, _ in changed:
            node.input[idx] = quantized
    temporary.unlink()
    write_json(output / "ablation.json", report)


def compare_calibrations(
    float_graph, output, calibration, held_out, baseline, report, variants
):
    # ModelOpt infers shapes in place. Keep the original #2297 export intact.
    source = output / "calibration-source.onnx"
    shutil.copyfile(float_graph, source)
    float_model = onnx.load(source, load_external_data=False)
    scopes = {scope: selected_nodes(float_model.graph, scope) for scope, _ in variants}
    del float_model
    for scope, method in variants:
        label = f"{scope}-{method}"
        print(f"ModelOpt scope {scope}, calibration {method}", flush=True)
        directory = output / label
        directory.mkdir()
        target = directory / "quantized.onnx"
        quant = modelopt_ptq(
            source,
            target,
            calibration,
            "int8",
            input_name="pixels",
            method=method,
            nodes=scopes[scope],
        )
        report["variants"][label] = {
            "explicit_compute_operations": len(scopes[scope])
            if scopes[scope]
            else None,
            "quantization": quant,
            **cosine_summary(evaluate(target, held_out), baseline),
        }
        write_json(output / "metrics.json", report)


def run(root, output, only):
    lock, addresses = verify_inputs(root / "inputs")
    reference = root / "0" / "features.npy"
    original = json.loads((root / "0" / "result.json").read_text())
    if (
        original["identity"]["tensors"] != lock["tensors"]
        or sha256_file(reference) != original["features_sha256"]
    ):
        raise ValueError("CLIP reference output differs from #2297")
    float_graph = root / "3a" / "export.onnx"
    legacy_graph = root / "3a" / "quantized.onnx"
    output.mkdir(parents=True, exist_ok=False)
    inputs = np.load(root / "inputs" / "eval.npy", mmap_mode="r")
    indices = np.arange(0, len(inputs), 20)
    held_out = inputs[indices]
    baseline = np.load(reference)[indices]
    calibration_indices = np.arange(0, lock["tensors"]["calibration"]["shape"][0], 8)
    calibration = np.load(root / "inputs" / "calibration.npy", mmap_mode="r")[
        calibration_indices
    ]
    report = {
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "held_out_indices": indices.tolist(),
        "held_out_addresses": [addresses[i] for i in indices],
        "calibration_indices": calibration_indices.tolist(),
        "tensors": lock["tensors"],
        "onnx_sha256": {
            path.name: sha256_file(path) for path in (float_graph, legacy_graph)
        },
        "variants": {},
    }
    if only in {None, "ablation"}:
        ablate(float_graph, legacy_graph, output, held_out, baseline, report)
    if only != "ablation":
        variants = (
            [(scope, "max") for scope in LINEAR_GROUPS]
            if only == "sensitivity"
            else [
                ("default", "max"),
                ("matmul", "max"),
                ("linear", "max"),
                ("linear", "entropy"),
            ]
        )
        compare_calibrations(
            float_graph, output, calibration, held_out, baseline, report, variants
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="completed #2297 output root"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=("ablation", "calibration", "sensitivity"))
    args = parser.parse_args()
    require_queue()
    run(args.source, args.output, args.only)


if __name__ == "__main__":
    main()
