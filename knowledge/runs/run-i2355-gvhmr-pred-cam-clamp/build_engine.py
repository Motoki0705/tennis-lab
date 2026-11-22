"""Build one minimal or full TensorRT plan in an isolated child process."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from additive_precision import verify_additive_attention
from benchmark import write_json
from convert import build_checked_plan, build_engine, require_queue
from inputs import sha256_file


def summarize_layers(details):
    layers = details["Layers"]
    return {
        "engine_layers": len(layers),
        "layer_types": dict(Counter(layer["LayerType"] for layer in layers)),
        "layer_names": [layer["Name"] for layer in layers],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-operations", type=Path)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6))
    parser.add_argument(
        "--attention-mask", choices=("bool", "additive"), default="bool"
    )
    args = parser.parse_args()
    if (args.profile is None) != (args.expected_operations is None):
        parser.error("Full builds require both profile and expected operations")
    require_queue()
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    if args.profile is None:
        evidence = build_checked_plan(
            args.onnx,
            args.plan,
            allow_tf32=False,
            verbose=True,
            builder_optimization_level=args.builder_optimization_level,
            verify=summarize_layers,
        )
    else:
        profile = json.loads(args.profile.read_text())
        expected = json.loads(args.expected_operations.read_text())
        if args.attention_mask == "bool":
            evidence = build_engine(
                args.onnx,
                args.plan,
                required_precision="fp16",
                expected_operations=expected,
                profile_shapes=profile["inputs"],
                builder_optimization_level=args.builder_optimization_level,
            )
        else:
            evidence = build_checked_plan(
                args.onnx,
                args.plan,
                verify=lambda details: verify_additive_attention(
                    details, "fp16", expected_operations=expected
                ),
                profile_shapes=profile["inputs"],
                builder_optimization_level=args.builder_optimization_level,
            )
    write_json(
        args.report,
        {
            "onnx": str(args.onnx),
            "onnx_sha256": sha256_file(args.onnx),
            "plan": str(args.plan),
            "plan_sha256": sha256_file(args.plan),
            "builder_optimization_level": args.builder_optimization_level,
            "attention_mask": args.attention_mask,
            "compute_precision": evidence,
        },
    )


if __name__ == "__main__":
    main()
