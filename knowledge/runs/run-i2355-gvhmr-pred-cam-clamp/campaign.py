"""Reproduce the batched ScatterND build crash and validate its graph workaround."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
from convert_gvhmr import export_network, network_args  # noqa: E402
from inputs import sha256_file  # noqa: E402
from locked_inputs import checked_segments, sweep_profile  # noqa: E402
from minimal_repro import compare_minimal, export_minimal, run_onnx  # noqa: E402
from model import PLAN, load_network, pack_sequences  # noqa: E402
from rewrite_pred_cam import rewrite_pred_cam_clamp  # noqa: E402

BUILD = HERE / "build_engine.py"
EXPERIMENT = GVHMR / "experiment.py"
RUNTIME = GVHMR / "run_gvhmr.py"
BATCHES = (1, 2, 4)
LENGTHS = (145, 177)


def child_command(script: Path, *arguments: object) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-X",
        "faulthandler",
        str(script),
        *map(str, arguments),
    ]


def run_child(
    report: dict[str, object],
    name: str,
    command: list[str],
    root: Path,
) -> dict[str, object]:
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
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
        "log": str(log.relative_to(root)),
    }
    report.setdefault("stages", {})[name] = result
    write_json(root / "campaign.json", report)
    print(f"{name}: exit {completed.returncode}", flush=True)
    return result


def prepare_minimal(root: Path, report: dict[str, object]) -> None:
    cases = (
        ("fp16-b1-opt3-scatter", "fp16", 1, 3, False),
        ("fp16-b2-opt3-scatter", "fp16", 2, 3, False),
        ("fp16-b2-opt0-scatter", "fp16", 2, 0, False),
        ("fp32-b2-opt3-scatter", "fp32", 2, 3, False),
        ("fp16-b2-opt3-concat", "fp16", 2, 3, True),
    )
    minimal = root / "minimal"
    parity = {}
    for name, dtype, batch, optimization, rewritten in cases:
        case = minimal / name
        original = case / "scatter.onnx"
        raw_cam = export_minimal(original, batch=batch, frames=145, dtype=dtype)
        concat = case / "concat.onnx"
        rewrite = rewrite_pred_cam_clamp(original, concat)
        parity[name] = {
            "rewrite": rewrite,
            "equivalence": compare_minimal(original, concat, raw_cam),
        }
        graph = concat if rewritten else original
        result = run_child(
            report,
            f"minimal-{name}",
            child_command(
                BUILD,
                "--onnx",
                graph,
                "--plan",
                case / "model.plan",
                "--report",
                case / "build.json",
                "--builder-optimization-level",
                optimization,
            ),
            root,
        )
        result.update(
            {
                "dtype": dtype,
                "batch": batch,
                "length": 145,
                "graph": "concat" if rewritten else "scatter",
                "builder_optimization_level": optimization,
                "onnx_bytes": graph.stat().st_size,
                "onnx_sha256": sha256_file(graph),
            }
        )
        write_json(root / "campaign.json", report)
    report["minimal_equivalence"] = parity
    write_json(root / "campaign.json", report)


def representatives(segments, profile):
    batch_size = profile["profile_batches"]["opt"]
    return {
        length: pack_sequences(
            [
                arrays
                for row, arrays in segments
                if row["split"] == "eval" and row["length"] == length
            ][:batch_size]
        )
        for length in sorted(set(profile["profile_lengths"].values()))
    }


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-6)
    return {
        "max_abs": float(delta.max(initial=0)),
        "max_rel": float((delta / scale).max(initial=0)),
    }


def verify_full_fp32(
    root: Path, segments, checkpoint: Path, report: dict[str, object]
) -> None:
    selected = [
        arrays
        for row, arrays in segments
        if row["split"] == "eval" and row["length"] == 145
    ][:2]
    arrays = pack_sequences(selected)
    directory = root / "fp32-equivalence"
    directory.mkdir(parents=True)
    source = directory / "scatter.onnx"
    rewritten = directory / "concat.onnx"
    network = load_network(checkpoint).to(dtype=torch.float32)
    args = network_args(arrays, torch.float32)
    with torch.inference_mode():
        native = {
            name: value.detach().cpu().numpy()
            for name, value in zip(
                ("pred_x", "pred_cam", "static_conf_logits", "mask_out", "length_out"),
                network(*args),
                strict=True,
            )
        }
    operations = export_network(
        network,
        {145: arrays},
        source,
        torch.float32,
        145,
        dynamic_length=False,
    )
    rewrite = rewrite_pred_cam_clamp(source, rewritten)
    feeds = {
        name: arrays[name].astype(np.int64 if name == "length" else np.float32)
        for name in ("length", "obs", "f_cliffcam", "f_cam_angvel", "f_imgseq")
    }
    source_values = run_onnx(source, feeds)
    rewritten_values = run_onnx(rewritten, feeds)
    comparisons = {}
    for name, expected in native.items():
        if expected.dtype.kind in {"b", "i", "u"}:
            np.testing.assert_array_equal(source_values[name], expected)
            np.testing.assert_array_equal(rewritten_values[name], expected)
        else:
            np.testing.assert_allclose(
                source_values[name], expected, rtol=5e-4, atol=5e-4
            )
            np.testing.assert_allclose(
                rewritten_values[name], expected, rtol=5e-4, atol=5e-4
            )
        np.testing.assert_array_equal(rewritten_values[name], source_values[name])
        comparisons[name] = {
            "source_vs_pytorch": difference(source_values[name], expected),
            "rewrite_vs_pytorch": difference(rewritten_values[name], expected),
            "rewrite_vs_source": difference(
                rewritten_values[name], source_values[name]
            ),
        }
    report["full_fp32_equivalence"] = {
        "batch": 2,
        "length": 145,
        "operations": len(operations),
        "rewrite": rewrite,
        "comparisons": comparisons,
    }
    write_json(root / "campaign.json", report)
    del network, native, source_values, rewritten_values
    gc.collect()


def build_full_engine(
    root: Path, inputs: Path, segments, checkpoint: Path, report: dict[str, object]
) -> Path:
    profile = sweep_profile(4, 4)
    samples = representatives(segments, profile)
    conversion = root / "engine-2-concat"
    conversion.mkdir(parents=True)
    source = conversion / "export-scatter.onnx"
    graph = conversion / "export.onnx"
    network = load_network(checkpoint).to(dtype=torch.float16)
    operations = export_network(
        network,
        samples,
        source,
        torch.float16,
        profile["profile_lengths"]["opt"],
        dynamic_batch=True,
        dynamic_length=True,
    )
    del network
    rewrite = rewrite_pred_cam_clamp(source, graph)
    expected = dict.fromkeys(operations, "fp16")
    expected_file = conversion / "expected-precision.json"
    profile_file = conversion / "profile.json"
    write_json(expected_file, expected)
    write_json(profile_file, profile)
    build_report = conversion / "build.json"
    result = run_child(
        report,
        "full-dynamic-concat-build",
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
        ),
        root,
    )
    if result["exit_code"] != 0 or not build_report.is_file():
        raise RuntimeError("The rewritten full GVHMR engine did not build")
    build = json.loads(build_report.read_text())
    certificate = {
        "condition": "2",
        "dynamic_profile": profile,
        "tensors_sha256": sha256_file(GVHMR / "tensor-lock.json"),
        "graph_sha256": sha256_file(graph),
        "engine_sha256": sha256_file(conversion / PLAN),
        "expected_precision_sha256": sha256_file(expected_file),
        "compute_precision": build["compute_precision"],
        "native_mask_adapter_parity": sorted(samples),
        "graph_rewrite": rewrite,
    }
    write_json(conversion / "conversion.json", certificate)
    report["full_engine"] = certificate
    report["full_engine"]["inputs"] = str(inputs)
    write_json(root / "campaign.json", report)
    return conversion


def probe_engine(
    root: Path,
    inputs: Path,
    conversion: Path,
    report: dict[str, object],
) -> None:
    cases = (
        ("valid-b1-l145", 1, 145, True, "GVHMR executed B=1, L=145"),
        ("valid-b4-l177", 4, 177, True, "GVHMR executed B=4, L=177"),
        ("reject-b0", 0, 145, False, "TensorRT rejected input shape length=(0,)"),
        ("reject-b5", 5, 145, False, "TensorRT rejected input shape length=(5,)"),
        (
            "reject-l144",
            1,
            144,
            False,
            "TensorRT rejected input shape obs=(1, 144, 17, 3)",
        ),
        (
            "reject-l178",
            1,
            178,
            False,
            "TensorRT rejected input shape obs=(1, 178, 17, 3)",
        ),
    )
    evidence = {}
    for name, batch, length, should_pass, marker in cases:
        result = run_child(
            report,
            f"probe-{name}",
            child_command(
                RUNTIME,
                "--condition",
                "2",
                "--inputs",
                inputs,
                "--conversion",
                conversion,
                "--mode",
                "probe",
                "--batch-size",
                batch,
                "--length",
                length,
            ),
            root,
        )
        text = (root / result["log"]).read_text(errors="replace")
        verified = (
            result["exit_code"] == 0 if should_pass else result["exit_code"] != 0
        ) and marker in text
        evidence[name] = {
            **result,
            "batch": batch,
            "length": length,
            "expected_success": should_pass,
            "marker": marker,
            "verified": verified,
        }
    report["boundary_probes"] = evidence
    write_json(root / "campaign.json", report)
    if not all(item["verified"] for item in evidence.values()):
        raise RuntimeError(
            "TensorRT profile boundary probes did not match expectations"
        )


def measure_engine(
    root: Path,
    inputs: Path,
    conversion: Path,
    reference_root: Path,
    report: dict[str, object],
    *,
    experiment: Path = EXPERIMENT,
) -> None:
    measurements = {}
    for length in LENGTHS:
        reference = reference_root / f"c0-b1-l{length}"
        for batch in BATCHES:
            name = f"c2-b{batch}-l{length}"
            destination = root / "measurements" / name
            result = run_child(
                report,
                f"measure-{name}",
                child_command(
                    experiment,
                    "--condition",
                    "2",
                    "--length",
                    length,
                    "--batch-size",
                    batch,
                    "--inputs",
                    inputs,
                    "--output",
                    destination,
                    "--conversion",
                    conversion,
                    "--reference",
                    reference,
                    "--profiles",
                    "nsys",
                ),
                root,
            )
            if result["exit_code"] != 0:
                measurements[name] = {**result, "verified": False}
                continue
            timing = json.loads((destination / "measurement.json").read_text())
            accuracy = json.loads((destination / "accuracy.json").read_text())
            profiling = json.loads((destination / "profiles.json").read_text())
            measurements[name] = {
                **result,
                "verified": True,
                "median_ms": timing["median_ms"],
                "p95_ms": timing["p95_ms"],
                "sequences_per_second": timing["sequences_per_second"],
                "frames_per_second": timing["frames_per_second"],
                "torch_peak_allocated_bytes": timing["torch_peak_allocated_bytes"],
                "torch_peak_reserved_bytes": timing["torch_peak_reserved_bytes"],
                "nvml_device_peak_sampled_bytes": timing[
                    "nvml_device_peak_sampled_bytes"
                ],
                "nvml_device_peak_minus_idle_bytes": timing[
                    "nvml_device_peak_minus_idle_bytes"
                ],
                "nsys_gpu_kernels": profiling["nsys"]["gpu_kernels"],
                "mask_matches_fp32": accuracy["mask_matches_fp32"],
                "length_matches_fp32": accuracy["length_matches_fp32"],
                "incam_coco17_l2_mm": accuracy["incam"]["coco17_l2_mm"],
                "global_coco17_l2_mm": accuracy["global"]["coco17_l2_mm"],
                "artifact": str(destination),
            }
            report["measurements"] = measurements
            write_json(root / "campaign.json", report)
    report["measurements"] = measurements
    write_json(root / "campaign.json", report)
    if not all(item["verified"] for item in measurements.values()):
        raise RuntimeError("One or more rewritten-engine measurements failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    args = parser.parse_args()
    require_queue()
    inputs = args.inputs.resolve(strict=True)
    reference_root = args.reference_root.resolve(strict=True)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Campaign output already exists: {root}")
    root.mkdir(parents=True)
    repro = Path(os.environ["TENNIS_REPRO_DIR"])
    (repro / "output_dir.txt").write_text(f"{root}\n")
    report: dict[str, object] = {
        "schema_version": 1,
        "queue_job": os.environ["TENNIS_RUN_ID"],
        "inputs": str(inputs),
        "reference_root": str(reference_root),
        "stages": {},
    }
    write_json(root / "campaign.json", report)
    segments, checkpoint = checked_segments(inputs, args.checkpoints)
    prepare_minimal(root, report)
    verify_full_fp32(root, segments, checkpoint, report)
    conversion = build_full_engine(root, inputs, segments, checkpoint, report)
    probe_engine(root, inputs, conversion, report)
    measure_engine(root, inputs, conversion, reference_root, report)
    report["status"] = "done"
    write_json(root / "campaign.json", report)


if __name__ == "__main__":
    main()
