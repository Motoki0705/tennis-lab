"""Compare exported ONNX graphs on CPU to locate accuracy loss before TensorRT."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cosine_summary(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    v = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    x = np.load(root / "inputs/eval.npy", allow_pickle=False)
    indices = np.arange(0, len(x), 20)
    ref = np.load(root / "0/features.npy", allow_pickle=False)[indices]
    result = {
        "provider": "CPUExecutionProvider",
        "sample_indices": indices.tolist(),
        "graphs": {},
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for condition, graphs in {
        "3a": ("export.onnx", "quantized.onnx"),
        "3b": ("quantized.onnx",),
    }.items():
        trt_out = np.load(root / condition / "features.npy", allow_pickle=False)[
            indices
        ]
        for graph in graphs:
            session = ort.InferenceSession(
                str(root / condition / graph),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            input_type = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
            }[session.get_inputs()[0].type]
            match = np.concatenate(
                [
                    session.run(
                        ["features"], {"pixels": x[i : i + 1].astype(input_type)}
                    )[0]
                    for i in indices
                ]
            )
            result["graphs"][f"{condition}/{graph}"] = {
                "input_type": str(input_type),
                "cosine_vs_pytorch_fp32": cosine_summary(match, ref),
                "cosine_vs_tensorrt_condition": cosine_summary(match, trt_out),
            }
            del session
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
