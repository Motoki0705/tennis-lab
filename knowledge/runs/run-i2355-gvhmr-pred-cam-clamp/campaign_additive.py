"""Validate and measure batched GVHMR TensorRT FP16 with additive attention masks.

Stages, each recorded in ``campaign.json``:

1. FP32 static B=2, L=145: the additive-mask graph must equal the source graph
   bit for bit under ONNX Runtime, and both must match PyTorch FP32.
2. FP16 dynamic export (B=1..4, L=145..177): pred_cam and additive-mask
   rewrites, then bit parity with the unrewritten FP16 graph under ONNX
   Runtime for every B in {1, 2, 4} and L in {145, 177}.
3. One dynamic-profile FP16 plan with per-layer FP16 checks at min/opt/max,
   TensorRT boundary probes, and the #2335 measurement harness.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

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
from campaign import (  # noqa: E402
    BATCHES,
    BUILD,
    GVHMR,
    LENGTHS,
    child_command,
    difference,
    measure_engine,
    probe_engine,
    representatives,
    run_child,
)
from convert import require_queue  # noqa: E402
from convert_gvhmr import export_network, network_args  # noqa: E402
from inputs import sha256_file  # noqa: E402
from locked_inputs import checked_segments, sweep_profile  # noqa: E402
from minimal_repro import run_onnx  # noqa: E402
from model import PLAN, load_network, pack_sequences  # noqa: E402
from rewrite_attention_mask import rewrite_attention_mask  # noqa: E402
from rewrite_pred_cam import rewrite_pred_cam_clamp  # noqa: E402

OUTPUTS = ("pred_x", "pred_cam", "static_conf_logits", "mask_out", "length_out")
INPUTS = ("length", "obs", "f_cliffcam", "f_cam_angvel", "f_imgseq")
MASKS = 24
EXPERIMENT = HERE / "experiment_additive.py"


def eval_batch(segments, length: int, batch: int):
    selected = [
        arrays
        for row, arrays in segments
        if row["split"] == "eval" and row["length"] == length
    ][:batch]
    if len(selected) != batch:
        raise ValueError(f"Only {len(selected)} eval segments at L={length}")
    return pack_sequences(selected)


def feeds(arrays, dtype) -> dict[str, np.ndarray]:
    return {
        name: arrays[name].astype(np.int64 if name == "length" else dtype)
        for name in INPUTS
    }


def assert_bit_equal(actual, expected) -> None:
    for name in OUTPUTS:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def verify_fp32(
    root: Path, segments, checkpoint: Path, report: dict, mode: str
) -> None:
    arrays = eval_batch(segments, 145, 2)
    directory = root / "fp32-equivalence"
    directory.mkdir(parents=True)
    scatter = directory / "scatter.onnx"
    concat = directory / "concat.onnx"
    additive = directory / "additive.onnx"
    network = load_network(checkpoint).to(dtype=torch.float32)
    with torch.inference_mode():
        native = {
            name: value.detach().cpu().numpy()
            for name, value in zip(
                OUTPUTS, network(*network_args(arrays, torch.float32)), strict=True
            )
        }
    export_network(
        network, {145: arrays}, scatter, torch.float32, 145, dynamic_length=False
    )
    pred_cam = rewrite_pred_cam_clamp(scatter, concat)
    masks = rewrite_attention_mask(concat, additive, mode=mode, expected=MASKS)
    if masks["unverified_broadcast"]:
        raise RuntimeError("Static FP32 masks must have folded broadcast shapes")
    source_values = run_onnx(concat, feeds(arrays, np.float32))
    rewritten_values = run_onnx(additive, feeds(arrays, np.float32))
    assert_bit_equal(rewritten_values, source_values)
    comparisons = {}
    for name, expected in native.items():
        if expected.dtype.kind in {"b", "i", "u"}:
            np.testing.assert_array_equal(rewritten_values[name], expected)
        else:
            np.testing.assert_allclose(
                rewritten_values[name], expected, rtol=5e-4, atol=5e-4
            )
        comparisons[name] = {
            "rewrite_vs_pytorch": difference(rewritten_values[name], expected),
            "rewrite_vs_source_onnx": difference(
                rewritten_values[name], source_values[name]
            ),
        }
    report["fp32_equivalence"] = {
        "batch": 2,
        "length": 145,
        "pred_cam_rewrite": pred_cam,
        "mask_rewrite": masks,
        "bit_equal_to_source_onnx": True,
        "comparisons": comparisons,
    }
    write_json(root / "campaign.json", report)
    del network, native, source_values, rewritten_values
    gc.collect()


def export_dynamic(root: Path, segments, checkpoint: Path, report: dict, mode: str):
    profile = sweep_profile(4, 4)
    samples = representatives(segments, profile)
    conversion = root / "engine-additive"
    conversion.mkdir(parents=True)
    scatter = conversion / "export-scatter.onnx"
    concat = conversion / "export-concat.onnx"
    graph = conversion / "export.onnx"
    network = load_network(checkpoint).to(dtype=torch.float16)
    operations = export_network(
        network,
        samples,
        scatter,
        torch.float16,
        profile["profile_lengths"]["opt"],
        dynamic_batch=True,
        dynamic_length=True,
    )
    del network
    pred_cam = rewrite_pred_cam_clamp(scatter, concat)
    masks = rewrite_attention_mask(
        concat,
        graph,
        mode=mode,
        expected=MASKS,
        allow_input_dependent_expand=True,
    )
    parity = {}
    for length in LENGTHS:
        for batch in BATCHES:
            arrays = feeds(eval_batch(segments, length, batch), np.float16)
            source_values = run_onnx(concat, arrays)
            rewritten_values = run_onnx(graph, arrays)
            assert_bit_equal(rewritten_values, source_values)
            parity[f"b{batch}-l{length}"] = "bit-equal"
    if len(parity) != len(LENGTHS) * len(BATCHES):
        raise RuntimeError("FP16 parity did not cover every measured shape")
    report["fp16_dynamic_parity"] = {
        "pred_cam_rewrite": pred_cam,
        "mask_rewrite": masks,
        "shapes": parity,
    }
    write_json(root / "campaign.json", report)
    return conversion, graph, profile, operations, samples, masks, pred_cam


def build(root: Path, inputs: Path, exported, report: dict) -> Path:
    conversion, graph, profile, operations, samples, masks, pred_cam = exported
    expected_file = conversion / "expected-precision.json"
    profile_file = conversion / "profile.json"
    write_json(expected_file, dict.fromkeys(operations, "fp16"))
    write_json(profile_file, profile)
    build_report = conversion / "build.json"
    result = run_child(
        report,
        "full-dynamic-additive-build",
        child_command(
            BUILD,
            "--onnx",
            graph,
            "--plan",
            conversion / PLAN,
            "--report",
            build_report,
            "--profile",
            profile_file,
            "--expected-operations",
            expected_file,
            "--attention-mask",
            "additive",
        ),
        root,
    )
    if result["exit_code"] != 0 or not build_report.is_file():
        raise RuntimeError("The additive-mask GVHMR engine did not build")
    built = json.loads(build_report.read_text())
    certificate = {
        "condition": "2",
        "dynamic_profile": profile,
        "tensors_sha256": sha256_file(GVHMR / "tensor-lock.json"),
        "graph_sha256": sha256_file(graph),
        "engine_sha256": sha256_file(conversion / PLAN),
        "expected_precision_sha256": sha256_file(expected_file),
        "compute_precision": built["compute_precision"],
        "native_mask_adapter_parity": sorted(samples),
        "graph_rewrite": {"pred_cam": pred_cam, "attention_mask": masks},
    }
    write_json(conversion / "conversion.json", certificate)
    report["full_engine"] = {**certificate, "inputs": str(inputs)}
    write_json(root / "campaign.json", report)
    return conversion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument(
        "--mask-mode", required=True, choices=("additive", "additive-finite")
    )
    args = parser.parse_args()
    require_queue()
    inputs = args.inputs.resolve(strict=True)
    reference_root = args.reference_root.resolve(strict=True)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Campaign output already exists: {root}")
    root.mkdir(parents=True)
    (Path(os.environ["TENNIS_REPRO_DIR"]) / "output_dir.txt").write_text(f"{root}\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "inputs": str(inputs),
        "reference_root": str(reference_root),
        "mask_mode": args.mask_mode,
        "stages": {},
    }
    write_json(root / "campaign.json", report)
    segments, checkpoint = checked_segments(inputs, args.checkpoints)
    verify_fp32(root, segments, checkpoint, report, args.mask_mode)
    exported = export_dynamic(root, segments, checkpoint, report, args.mask_mode)
    conversion = build(root, inputs, exported, report)
    probe_engine(root, inputs, conversion, report)
    measure_engine(
        root, inputs, conversion, reference_root, report, experiment=EXPERIMENT
    )
    report["status"] = "done"
    write_json(root / "campaign.json", report)


if __name__ == "__main__":
    main()
