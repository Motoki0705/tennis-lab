"""Bounded source-video execute/fresh-load/cache parity check; CUDA only in queue."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner.deployment import InferenceBundle, load_inference_bundle
from src.tasks.ball_refiner.pipeline_options import select_ball_path
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.ball_refiner_recipe import RefinerEvidenceModule
from src.tennis_scene.pipeline.components.ball_refiner import (
    CalibratedBallRefiner2DOutput,
)
from src.tennis_scene.pipeline.input_assembly.ball_refiner import (
    BallRefiner2DInputAssembler,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.resource_guard import (
    HostRAMGuard,
    available_ram_bytes,
    process_tree_rss_bytes,
)
from src.utils.video import probe_video_info

GMM_FIELDS = ("means", "scale_tril", "mixture_logits", "presence_logits",
              "covariance", "weights", "presence_probability")


def verify_inputs(plan: dict[str, Any]) -> None:
    for name, digest in plan["input_sha256"].items():
        if dual_sha256(Path(name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    if available_ram_bytes() < 8 * 1024**3:
        raise RuntimeError("Launch requires at least 8 GiB MemAvailable")
    if shutil.disk_usage(Path(plan["report"]).parent).free < plan["disk_budget_bytes"]:
        raise RuntimeError("Insufficient disk headroom")
    if Path(plan["report"]).exists():
        raise FileExistsError(plan["report"])
    if plan["clip_id"] != "meiji/video_000/clip_010" or [r["camera"] for r in plan["cameras"]] != ["cam0", "cam1", "cam2"]:
        raise ValueError("Only the predeclared three-camera validation clip is allowed")
    verify_inputs(plan)
    bundle = load_inference_bundle(Path(plan["bundle"]))
    artifact = select_ball_path(plan["ball_path"], bundle, Path(plan["calibration_artifact"]))
    if artifact is None:
        raise ValueError("The candidate check requires explicit covariance calibration")
    results = []
    for row in plan["cameras"]:
        video = Path(row["video"])
        if "video_001" in video.parts or "video_000" not in video.parts or "clip_010" not in video.parts:
            raise ValueError(f"Unexpected source video: {video}")
        info = probe_video_info(video)
        if info.frame_count != 270 or (info.width, info.height) != (1920, 1080):
            raise ValueError("Unexpected source clip dimensions/frame count")
        with np.load(row["cached_prediction"], allow_pickle=False) as saved:
            if saved["means"].shape != (270, 4, 2) or not np.array_equal(saved["frame_index"], np.arange(270)):
                raise ValueError("Cached predictions do not cover the same complete clip")
        results.append({"camera": row["camera"], "frames": info.frame_count, "source_sha256": dual_sha256(video)})
    return {"status": "passed", "cameras": results, "fixed_calibration": artifact.load().covariance_multiplier,
            "input_files": len(plan["input_sha256"]), "ram_available_bytes": available_ram_bytes()}


def compare_arrays(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], tolerances: dict[str, float],
) -> dict[str, Any]:
    """Every element is checked; nonfinite or misaligned arrays never pass."""
    rows: dict[str, Any] = {}
    for name, tolerance in tolerances.items():
        a, b = actual[name], expected[name]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            rows[name] = {"passed": False, "reason": "shape/nonfinite mismatch"}
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        count = int(np.count_nonzero(delta > tolerance))
        rows[name] = {"passed": count == 0, "max_abs": float(delta.max()), "elements": delta.size,
                      "exceeded": count, "atol": tolerance, "rtol": 0,
                      "p50_p90_p95_abs": np.quantile(delta, [0.5, 0.9, 0.95]).tolist()}
    return {"passed": all(row["passed"] for row in rows.values()), "fields": rows}


def snapshot(plan: dict[str, Any], row: dict[str, Any], phase: str, receipt: dict[str, Any]) -> None:
    root = Path(plan["report"]) / row["camera"]
    store = ClipStore(root / "store", receipt["source"], memory_entries=0)
    reference = store.active(f"ball_refiner_2d/{row['camera']}")
    if reference is None:
        raise ValueError("Missing calibrated refiner artifact after pipeline execution/load")
    output = store.load(reference, ArtifactCodec(CalibratedBallRefiner2DOutput))
    arrays = {name: getattr(output.prediction.distribution, name)[0].numpy() for name in GMM_FIELDS}
    arrays.update(
        frame_indices=output.frame_indices, pts=output.pts, timestamps_seconds=output.timestamps_seconds,
        source_size_wh=np.asarray(output.source_size_wh), time_base=np.asarray(output.time_base),
        camera_id=np.asarray(output.camera_id), detector_window_start=output.detector_window_start,
        detector_time_index=output.detector_time_index, refiner_window_start=output.prediction.window_start,
        refiner_time_index=output.prediction.time_index,
    )
    np.savez_compressed(root / f"{phase}.npz", **arrays)
    write_json_atomic(root / f"{phase}.json", receipt)


def child_phase(plan: dict[str, Any], row: dict[str, Any], phase: str) -> None:
    from src.tasks.ball_refiner.scripts.run_pipeline import main as run_pipeline

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Fresh load-only must not load models or assemble video inputs")

    torch.set_num_threads(2)
    if phase == "execute":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            plan["allocator_limit_bytes"] / torch.cuda.get_device_properties(device).total_memory, device,
        )
        torch.cuda.reset_peak_memory_stats(device)
    else:
        InferenceBundle.load_model = forbidden
        RefinerEvidenceModule.load = forbidden
        BallRefiner2DInputAssembler.assemble = forbidden
    sys.argv = [
        "run_pipeline", "--video", row["video"], "--camera-id", row["camera"],
        "--bundle", plan["bundle"], "--detector-checkpoint", plan["detector_checkpoint"],
        "--store", str(Path(plan["report"]) / row["camera"] / "store"),
        "--device", "cuda", "--source", phase, "--detector-batch-size", "4", "--refiner-batch-size", "32",
        "--ball-path", plan["ball_path"], "--calibration-artifact", plan["calibration_artifact"],
    ]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        run_pipeline()
    receipt = json.loads(captured.getvalue().splitlines()[-1])
    if set(receipt["components"].values()) != {"executed" if phase == "execute" else "loaded"}:
        raise ValueError("Unexpected execute/load component status")
    snapshot(plan, row, phase, receipt)
    if phase == "execute":
        receipt["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        write_json_atomic(Path(plan["report"]) / row["camera"] / "execute.json", receipt)
    print(json.dumps(receipt), flush=True)


def compare_camera(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    root = Path(plan["report"]) / row["camera"]
    receipts = [json.loads((root / f"{phase}.json").read_text()) for phase in ("execute", "load")]
    if receipts[0]["references"] != receipts[1]["references"] or receipts[0]["source"] != receipts[1]["source"]:
        raise ValueError("Fresh load changed source/artifact identities")
    with np.load(root / "execute.npz", allow_pickle=False) as ex, np.load(root / "load.npz", allow_pickle=False) as ld:
        if set(ex.files) != set(ld.files):
            raise ValueError("Fresh load changed snapshot fields")
        for key in ex.files:
            if ex[key].dtype != ld[key].dtype or not np.array_equal(ex[key], ld[key]):
                raise ValueError(f"Fresh load changed {key}")
        actual = {key: ex[key] for key in ex.files}
    artifact = select_ball_path(plan["ball_path"], load_inference_bundle(Path(plan["bundle"])),
                                Path(plan["calibration_artifact"]))
    if artifact is None:
        raise ValueError("Missing fixed calibration")
    with np.load(row["cached_prediction"], allow_pickle=False) as saved:
        for name, key in (("frame_indices", "frame_index"), ("pts", "pts")):
            np.testing.assert_array_equal(actual[name], saved[key])
        raw = BallGMM2D(**{field.name: torch.from_numpy(saved[field.name][None].copy()) for field in fields(BallGMM2D)})
        calibrated = artifact.load().apply(raw)
    expected = {name: getattr(calibrated, name)[0].numpy() for name in GMM_FIELDS}
    result = compare_arrays(actual, expected, plan["tolerances"])
    result.update(camera=row["camera"], source_load_arrays_bit_equal=True, artifact_references_equal=True,
                  frame_pts_exact=True, compared_frames=len(actual["pts"]),
                  max_mean_coordinate_delta_px=float(np.max(np.abs(actual["means"] - expected["means"]) * [1919, 1079])))
    return result


def output_bytes(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                continue  # Artifact staging directory was atomically published.
    return total


def run(plan_path: Path, plan: dict[str, Any]) -> None:
    started = time.monotonic()
    checked = preflight(plan)
    root = Path(plan["report"])
    root.mkdir(parents=True)
    write_json_atomic(root / "preflight.json", checked)
    shutil.copyfile(plan_path, root / "plan.json")
    ram = HostRAMGuard()
    peak_device = 0
    status = "running"
    failure: str | None = None
    phases: list[dict[str, Any]] = []
    child: subprocess.Popen[bytes] | None = None

    def terminate(signum: int, frame: Any) -> None:
        raise RuntimeError(f"Queue timeout/resource signal {signum}")

    signal.signal(signal.SIGTERM, terminate)

    def sample() -> None:
        nonlocal peak_device
        ram_failure = ram.sample(available_ram_bytes(), time.monotonic() - started, rss_bytes=process_tree_rss_bytes())
        result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=5)
        used = int(result.stdout.strip()) * 1024**2
        peak_device = max(peak_device, used)
        if ram_failure or used > plan["gpu_device_used_limit_bytes"]:
            raise RuntimeError(f"Resource guard: {ram_failure}; device_used_bytes={used}")
        if output_bytes(root) > plan["disk_budget_bytes"]:
            raise RuntimeError("Output disk budget exceeded")

    def publish() -> None:
        write_json_atomic(root / "resource_usage.json", {
            "status": status, "failure": failure, "seconds": time.monotonic() - started, "phases": phases,
            "peak_device_used_bytes": peak_device, "host_ram": ram.report(),
            "output_bytes": output_bytes(root), "queue_job": os.environ.get("TENNIS_RUN_ID"),
        })

    comparisons: list[dict[str, Any]] = []
    try:
        sample()
        for row in plan["cameras"]:
            directory = root / row["camera"]
            directory.mkdir()
            for phase in ("execute", "load"):
                begin = time.monotonic()
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--plan", str(plan_path),
                           "--phase", phase, "--camera", row["camera"]]
                with (directory / f"{phase}.log").open("wb") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        try:
                            code = child.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            sample()
                    sample()
                    if code:
                        raise RuntimeError(f"{row['camera']} {phase} exited {code}; see {phase}.log")
                phases.append({"camera": row["camera"], "phase": phase, "seconds": time.monotonic() - begin})
                publish()
            comparisons.append(compare_camera(plan, row))
            write_json_atomic(root / "comparison.json", {"cameras": comparisons, "all_passed": all(r["passed"] for r in comparisons)})
        verify_inputs(plan)
        if not all(result["passed"] for result in comparisons):
            raise RuntimeError("Source/cache parity exceeds predeclared tolerances; pipeline default must stay unchanged")
        status = "complete"
    except BaseException as error:
        failure = repr(error)
        status = "failed"
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        publish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--phase", choices=("execute", "load"))
    parser.add_argument("--camera")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(2)
    if args.preflight_output is not None:
        write_json_atomic(args.preflight_output, preflight(plan))
    elif args.phase:
        row = next(row for row in plan["cameras"] if row["camera"] == args.camera)
        child_phase(plan, row, args.phase)
    else:
        run(args.plan, plan)


if __name__ == "__main__":
    main()
