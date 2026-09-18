"""Two fixed ViTPose predictions; run only through training queue.

Use .venv/bin/python -B with CUDA_VISIBLE_DEVICES=0 and an external
`timeout --kill-after=5s 590s`. No retries or determinism interventions.
Differences measure repeatability, not ground-truth accuracy or hardware health.
"""

import _sha256
import argparse
import hashlib
import json
import os
import signal
import ssl
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = Path(
    "/home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/"
    "vitpose/vitpose-h-multi-coco.pth"
)
EXPECTED = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
OBSERVATIONS = ROOT / (
    "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/"
    "video_000/clip_007/cam0_people.npz"
)
CHUNK = 1024 * 1024
LIMIT_SECONDS = 540


def stat_record(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        key: getattr(stat, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    }


class Probe:
    def __init__(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.started = time.monotonic()
        self.initial_stats: dict[str, dict[str, int]] = {}
        self.baselines: dict[str, str] = {}
        self.data: dict[str, Any] = {
            "status": "running",
            "phase": "initialization",
            "hash_rows": [],
            "predictions": [],
            "errors": [],
            "internal_limit_seconds": LIMIT_SECONDS,
            "interpretation": "Exact numerical repeatability only; not GT accuracy. "
            "Any positive difference is a difference, not a hardware fault diagnosis. "
            "No retry, deterministic-algorithm setting, or production hash change.",
            "source_code": [
                str(ROOT / path)
                for path in (
                    "knowledge/runs/run-slcs-meiji-checkpoint-hash-gpu-control-v1/probe.py",
                    "knowledge/runs/run-slcs-meiji-checkpoint-hash-probe-v3/probe.py",
                    "src/submodules/models/vitpose/pose2d.py",
                    "src/submodules/models/_base/inference_model.py",
                )
            ],
            "provenance": {
                "python": sys.version,
                "executable": sys.executable,
                "openssl": ssl.OPENSSL_VERSION,
                "hashlib_type": str(type(hashlib.sha256())),
                "software_hash_type": str(type(_sha256.sha256())),
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUBLAS_WORKSPACE_CONFIG",
                        "OPENSSL_ia32cap",
                        "LD_LIBRARY_PATH",
                        "PYTHONPATH",
                    )
                },
            },
        }
        self.save()

    def save(self) -> None:
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        temporary = self.output / "results.tmp"
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "results.json")

    def phase(self, name: str) -> None:
        self.data["phase"] = name
        self.save()
        self.check_deadline()

    def check_deadline(self) -> None:
        if time.monotonic() - self.started >= LIMIT_SECONDS:
            raise TimeoutError("Internal 540-second diagnostic deadline reached")

    def hash_file(self, path: Path, expected: str | None = None) -> None:
        self.check_deadline()
        started = time.monotonic()
        key = str(path)
        before = stat_record(path)
        initial = self.initial_stats.setdefault(key, before)
        first, second = hashlib.sha256(), _sha256.sha256()
        count = 0
        with path.open("rb") as stream:
            while block := stream.read(CHUNK):
                self.check_deadline()
                # Exactly the same immutable bytes object goes to both implementations.
                first.update(block)
                second.update(block)
                count += len(block)
        after = stat_record(path)
        openssl, software = first.hexdigest(), second.hexdigest()
        target = expected or self.baselines.get(key)
        match = (
            openssl == software
            and (target is None or openssl == target)
            and initial == before == after
            and count == before["st_size"]
        )
        row = {
            "phase": self.data["phase"],
            "path": key,
            "hashlib": openssl,
            "_sha256": software,
            "expected": target,
            "expected_source": "supplied_known_sha" if expected else "start_baseline",
            "stat_initial": initial,
            "stat_before": before,
            "stat_after": after,
            "stat_changed": not initial == before == after,
            "bytes": count,
            "chunk_bytes": CHUNK,
            "match": match,
            "seconds": time.monotonic() - started,
        }
        self.data["hash_rows"].append(row)
        self.save()
        if not match:
            raise ValueError(f"Hash or stat mismatch: {path}")
        self.baselines.setdefault(key, openssl)

    def error(self) -> None:
        self.data["errors"].append(
            {"phase": self.data["phase"], "traceback": traceback.format_exc()}
        )
        self.data["status"] = "error"
        self.save()


def cuda_workload(probe: Probe) -> None:
    import numpy as np
    import torch

    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models import Pose2DRequest, ViTPosePose2D
    from src.submodules.models.tracker.common import TrackResult
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    source = ROOT / "data/tennis_multivew/processed/meiji_3cam/dataset"
    manifest = load_dataset_manifest(source)
    clip = ClipManifest.load(source / manifest.clips["video_000/clip_007"].path)
    if clip.num_frames != 652:
        raise ValueError(f"Expected 652 frames, got {clip.num_frames}")
    video = Path(clip.media_path("cam0"))
    metadata_path = OBSERVATIONS.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata["pose_sha256"] != EXPECTED:
        raise ValueError("Stored pose checkpoint provenance differs")
    video_sha = metadata["video_sha256"]
    probe.data["input_metadata"] = {
        "path": str(metadata_path),
        "contents": metadata,
        "track_known_sha": None,
        "track_known_sha_reason": "Not present in saved people metadata",
    }
    probe.phase("inputs_start")
    probe.hash_file(video, video_sha)
    probe.hash_file(OBSERVATIONS)
    with np.load(OBSERVATIONS, allow_pickle=False) as archive:
        boxes = archive["boxes"][0].copy()
    if boxes.shape != (652, 4) or not np.isfinite(boxes).all():
        raise ValueError(f"Invalid saved track boxes: {boxes.shape}")
    tracks = TrackResult({0: torch.from_numpy(boxes)}, clip.num_frames)
    request = Pose2DRequest(video, tracks.bbx_xys(0, base_enlarge=1.2))
    request_copy = request.bbx_xys.clone()
    probe.data["workload"] = {
        "video": str(video),
        "observations": str(OBSERVATIONS),
        "num_frames": 652,
        "track_id": 0,
        "boxes_shape": list(boxes.shape),
        "request_shape": list(request.bbx_xys.shape),
        "precision": "bfloat16",
        "batch_size": 8,
        "flip_test": True,
        "seed": 42,
        "predictions_planned": 2,
        "same_request_object": True,
        "frame_order": "clip-local indices 0 through 651",
    }
    probe.data["provenance"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    )
    pose = ViTPosePose2D(
        CHECKPOINT,
        device="cuda",
        flip_test=True,
        batch_size=8,
        precision="bfloat16",
        head_config=ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ()),
    )
    outputs = []
    try:
        probe.phase("model_loading")
        pose.load()
        probe.phase("model_loaded")
        probe.data["provenance"]["gpu"] = torch.cuda.get_device_name()
        probe.hash_file(CHECKPOINT, EXPECTED)
        for index in range(2):
            probe.phase(f"predict_{index + 1}")
            started = time.monotonic()
            result = pose.predict(request)
            torch.cuda.synchronize()
            array = result.keypoints.detach().cpu().numpy().copy()
            outputs.append(array)
            destination = probe.output / f"prediction_{index + 1}.npz"
            np.savez(destination, frame_indices=np.arange(652), keypoints=array)
            row = {
                "index": index + 1,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.isfinite(array).all()),
                "path": str(destination),
                "request_unchanged": bool(torch.equal(request.bbx_xys, request_copy)),
                "seconds_including_sync_and_save": time.monotonic() - started,
            }
            probe.data["predictions"].append(row)
            probe.save()
            if (
                array.shape != (652, 17, 3)
                or not row["finite"]
                or not row["request_unchanged"]
            ):
                raise ValueError("Invalid prediction or mutated input; evidence saved")
        probe.phase("after_two_predicts")
        delta = np.abs(outputs[0].astype(np.float64) - outputs[1].astype(np.float64))
        equal = bool(np.array_equal(outputs[0], outputs[1]))
        probe.data["comparison"] = {
            "shape_equal": outputs[0].shape == outputs[1].shape,
            "both_finite": True,
            "array_equal": equal,
            "coordinate_max_absolute_difference": float(delta[..., :2].max()),
            "coordinate_mean_absolute_difference": float(delta[..., :2].mean()),
            "confidence_max_absolute_difference": float(delta[..., 2].max()),
            "differing_elements": int(np.count_nonzero(outputs[0] != outputs[1])),
            "coordinate_differing_elements": int(np.count_nonzero(delta[..., :2])),
            "confidence_differing_elements": int(np.count_nonzero(delta[..., 2])),
        }
        probe.data["status"] = "passed" if equal and not np.any(delta) else "difference"
        probe.save()
        probe.hash_file(CHECKPOINT, EXPECTED)
    except BaseException:
        probe.error()
    finally:
        # Never load another model after an anomaly. Keep any prior evidence/errors.
        try:
            probe.data["phase"] = "unloading"
            probe.save()
            pose.unload()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            probe.phase("after_unload")
            probe.hash_file(CHECKPOINT, EXPECTED)
        except BaseException:
            probe.error()
        for path, expected in ((video, video_sha), (OBSERVATIONS, None)):
            try:
                probe.phase("inputs_end")
                probe.hash_file(path, expected)
            except BaseException:
                probe.error()


def deadline_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Internal 540-second diagnostic deadline reached")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New, nonexistent directory"
    )
    args = parser.parse_args()
    probe = Probe(args.output_dir)
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_SECONDS)
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Use repository .venv/bin/python (Python 3.11)")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("Queue must set CUDA_VISIBLE_DEVICES=0")
        probe.phase("before_torch_import")
        probe.hash_file(CHECKPOINT, EXPECTED)
        sys.path.insert(0, str(ROOT))
        cuda_workload(probe)
    except BaseException:
        probe.error()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        probe.save()
    return 0 if probe.data["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
