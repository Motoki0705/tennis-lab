"""Fast v1-like differential probe; run CUDA ONLY through training queue.

Wrap the invocation in `timeout --kill-after=5s 590s` for a <10 minute hard
envelope (including inference). The retained checkpoint consumes <3 GiB;
software SHA runs once for baseline blocks and only on subsequent anomalies.
No explicit CUDA synchronization precedes the first post-predict disk passes;
the pose implementation may itself synchronize internally.
"""

import _hashlib
import _sha256
import argparse
import hashlib
import json
import os
import platform
import ssl
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any

EXPECTED = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
CHUNK = 1024 * 1024
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = Path(
    "/home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/"
    "vitpose/vitpose-h-multi-coco.pth"
)


def stat_record(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        key: getattr(stat, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    }


def process_snapshot() -> dict[str, Any]:
    status = Path("/proc/self/status").read_text().splitlines()
    maps = Path("/proc/self/maps").read_text().splitlines()
    return {
        "thread_count": next(
            int(line.split()[1]) for line in status if line.startswith("Threads:")
        ),
        "selected_native_maps": [
            line
            for line in maps
            if any(
                name in line.lower()
                for name in ("libcrypto", "libssl", "libpython", "opencv", "/cv2/")
            )
        ],
    }


class Probe:
    def __init__(self, output: Path, cpu_only: bool, passes: int = 6) -> None:
        if not 1 <= passes <= 8:
            raise ValueError("diagnostic passes must be between 1 and 8")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "results.json"
        # Atomic reservation also prevents concurrent probes from sharing output.
        with self.output.open("x") as stream:
            stream.write('{"status": "initializing"}\n')
        self.started = time.monotonic()
        self.passes = passes
        self.blocks: tuple[bytes, ...] = ()
        self.block_digests: tuple[str, ...] = ()
        self.data: dict[str, Any] = {
            "status": "running",
            "cpu_only": cpu_only,
            "rows": [],
            "checkpoint": str(CHECKPOINT),
            "expected": EXPECTED,
            "chunk_bytes": CHUNK,
            "diagnostic_passes": passes,
            "anomalies": [],
            "interpretation": "Phases are chronological and confounded by elapsed time; "
            "explicit synchronization is not a clean A/B intervention. "
            "ViTPose predict already transfers heatmaps to CPU, implicitly synchronizing.",
            "provenance": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "openssl": ssl.OPENSSL_VERSION,
                "hashlib_sha256_type": str(type(hashlib.sha256())),
                "hashlib_sha256_module": hashlib.sha256.__module__,
                "software_sha256_type": str(type(_sha256.sha256())),
                "initial_process": process_snapshot(),
                "modules": {
                    module.__name__: getattr(module, "__file__", None)
                    or getattr(getattr(module, "__spec__", None), "origin", None)
                    for module in (_hashlib, _sha256, hashlib)
                },
                "environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(("CUDA", "NVIDIA", "OMP", "MKL", "OPENSSL"))
                    or key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH")
                },
                "sha256sum": subprocess.run(
                    ["sha256sum", "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.splitlines()[0],
            },
        }
        self.save()

    def save(self) -> None:
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.output)

    def row(self, phase: str, method: str, iteration: int, **values: Any) -> None:
        row = {"phase": phase, "method": method, "iteration": iteration, **values}
        self.data["rows"].append(row)
        self.save()
        print(json.dumps(row), flush=True)

    def system_reference(self, phase: str) -> None:
        started = time.monotonic()
        before = stat_record(CHECKPOINT)
        digest = subprocess.run(
            ["sha256sum", str(CHECKPOINT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        ).stdout.split()[0]
        after = stat_record(CHECKPOINT)
        self.row(
            phase,
            "sha256sum",
            0,
            digest=digest,
            expected=EXPECTED,
            stat_before=before,
            stat_after=after,
            seconds=time.monotonic() - started,
            match=digest == EXPECTED and before == after == self.data["initial_stat"],
        )

    def load_baseline(self) -> None:
        started = time.monotonic()
        initial = stat_record(CHECKPOINT)
        if not 0 < initial["st_size"] < 3 * 1024**3:
            raise ValueError("Checkpoint must be nonempty and below 3 GiB")
        self.data["initial_stat"] = initial
        self.save()
        self.system_reference("before_imports_and_cuda")
        with CHECKPOINT.open("rb") as stream:
            self.blocks = tuple(iter(lambda: stream.read(CHUNK), b""))
        # Per-block references are computed with software SHA before torch imports.
        self.block_digests = tuple(
            _sha256.sha256(block).hexdigest() for block in self.blocks
        )
        after = stat_record(CHECKPOINT)
        self.row(
            "baseline",
            "load_immutable_blocks",
            0,
            stat_before=initial,
            stat_after=after,
            block_count=len(self.blocks),
            retained_bytes=sum(map(len, self.blocks)),
            seconds=time.monotonic() - started,
            match=initial == after,
        )
        self.fast_pass("before_imports_and_cuda", 0, memory=True)

    def anomaly(
        self,
        phase: str,
        iteration: int,
        index: int,
        block: bytes,
        digest: str,
        expected: str | None,
        method: str,
    ) -> None:
        # Hash immediately, before formatting / I/O can perturb the failure state.
        software = _sha256.sha256(block).hexdigest()
        reference = self.blocks[index] if index < len(self.blocks) else b""
        same = block == reference
        differences = list(
            islice(
                (
                    {
                        "block_offset": offset,
                        "file_offset": index * CHUNK + offset,
                        "actual": actual,
                        "retained": retained,
                    }
                    for offset, (actual, retained) in enumerate(
                        zip_longest(block, reference)
                    )
                    if actual != retained
                ),
                128,
            )
        )
        record = {
            "phase": phase,
            "method": method,
            "iteration": iteration,
            "index": index,
            "offset": index * CHUNK,
            "length": len(block),
            "retained_length": len(reference),
            "hashlib": digest,
            "software_sha256": software,
            "baseline_software_sha256": expected,
            "same_as_retained_bytes": same,
            "first_byte_differences_limit_128": differences,
        }
        if not same:
            destination = (
                self.output.parent / f"anomaly-{phase}-{method}-{iteration}-{index}.bin"
            )
            with destination.open("xb") as stream:
                stream.write(block)
            record["preserved_actual_block"] = str(destination)
        self.data["anomalies"].append(record)
        self.save()
        print(json.dumps(record), flush=True)

    def hash_blocks(
        self, blocks: Iterable[bytes], phase: str, iteration: int, method: str
    ) -> tuple[list[str], int, int]:
        first, second = hashlib.sha256(), hashlib.sha256()
        mismatches, count = 0, 0
        for index, block in enumerate(blocks):
            first.update(block)
            second.update(block)
            digest = hashlib.sha256(block).hexdigest()
            expected = (
                self.block_digests[index] if index < len(self.block_digests) else None
            )
            same = index < len(self.blocks) and block == self.blocks[index]
            if digest != expected or not same:
                self.anomaly(phase, iteration, index, block, digest, expected, method)
                mismatches += 1
            count += 1
        return [first.hexdigest(), second.hexdigest()], count, mismatches

    def fast_pass(self, phase: str, iteration: int, *, memory: bool = False) -> None:
        started = time.monotonic()
        before = stat_record(CHECKPOINT)
        method = "retained_memory" if memory else "disk"
        if memory:
            digests, count, mismatches = self.hash_blocks(
                self.blocks, phase, iteration, method
            )
        else:
            with CHECKPOINT.open("rb") as stream:
                digests, count, mismatches = self.hash_blocks(
                    iter(lambda: stream.read(CHUNK), b""),
                    phase,
                    iteration,
                    method,
                )
        after = stat_record(CHECKPOINT)
        self.row(
            phase,
            method,
            iteration,
            stream_digests=digests,
            expected=EXPECTED,
            mismatched_blocks=mismatches,
            block_count=count,
            stat_before=before,
            stat_after=after,
            seconds=time.monotonic() - started,
            process_at_pass_end=process_snapshot(),
            match=all(digest == EXPECTED for digest in digests)
            and not mismatches
            and count == len(self.blocks)
            and before == after == self.data["initial_stat"],
        )


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

    probe.data["provenance"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_initialized_after_imports": torch.cuda.is_initialized(),
            "process_after_imports": process_snapshot(),
        }
    )
    source = ROOT / "data/tennis_multivew/processed/meiji_3cam/dataset"
    manifest = load_dataset_manifest(source)
    clip = ClipManifest.load(source / manifest.clips["video_000/clip_007"].path)
    observations = ROOT / (
        "outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/"
        "video_000/clip_007/cam0_people.npz"
    )
    with np.load(observations, allow_pickle=False) as archive:
        boxes = archive["boxes"][0]
    tracks = TrackResult({0: torch.from_numpy(boxes)}, clip.num_frames)
    probe.data["workload"] = {
        "video": str(clip.media_path("cam0")),
        "observations": str(observations),
        "num_frames": clip.num_frames,
    }
    probe.save()
    pose = ViTPosePose2D(
        CHECKPOINT,
        device="cuda",
        flip_test=True,
        batch_size=8,
        precision="bfloat16",
        head_config=ViTPoseHeadConfig(1280, 17, 2, (256, 256), (4, 4), 1, 0, ()),
    )
    try:
        started = time.monotonic()
        pose.predict(
            Pose2DRequest(clip.media_path("cam0"), tracks.bbx_xys(0, base_enlarge=1.2))
        )
        probe.data["predict_return_seconds"] = time.monotonic() - started
        # No CUDA API call, subprocess, software hash, or JSON write before first pass.
        for iteration in range(probe.passes):
            probe.fast_pass("after_predict_no_explicit_sync", iteration)
        probe.data["provenance"]["process_after_inference_and_disk_passes"] = (
            process_snapshot()
        )
        probe.fast_pass("after_predict_no_explicit_sync", 0, memory=True)
        torch.cuda.synchronize()
        probe.fast_pass("after_explicit_sync", 0)
        probe.system_reference("after_explicit_sync")
    finally:
        pose.unload()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    probe.fast_pass("after_unload", 0)
    probe.system_reference("after_unload")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Never import torch/model code."
    )
    parser.add_argument("--diagnostic-passes", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    probe = Probe(args.output_dir, args.cpu_only, args.diagnostic_passes)
    try:
        probe.load_baseline()
        if args.cpu_only:
            for iteration in range(probe.passes):
                probe.fast_pass("cpu_only", iteration)
            probe.fast_pass("cpu_only", 0, memory=True)
            probe.system_reference("after_cpu_only")
        else:
            cuda_workload(probe)
        probe.data["status"] = (
            "passed" if all(row["match"] for row in probe.data["rows"]) else "mismatch"
        )
    except BaseException:
        probe.data["status"] = "error"
        probe.data["error"] = traceback.format_exc()
        raise
    finally:
        probe.save()
    return 0 if probe.data["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
