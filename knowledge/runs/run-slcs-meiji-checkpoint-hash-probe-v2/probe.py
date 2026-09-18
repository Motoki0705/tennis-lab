"""Bounded SHA256 differential diagnosis; CUDA execution must use training queue.

Retains one checkpoint copy (2.55 GB) as immutable 1 MiB bytes, plus 64 MiB
of deterministic bytes. Two memory passes and one disk pass per phase; exactly
one real pose prediction. No production integrity checks are modified.
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
from pathlib import Path
from typing import Any

EXPECTED = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
CHUNK = 1024 * 1024
REPEATS = 2
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


def system_memory(blocks: Iterable[bytes]) -> str:
    # Stream the SAME bytes objects, without concatenating a second 2.55 GB copy.
    with subprocess.Popen(
        ["sha256sum"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        for block in blocks:
            process.stdin.write(block)
        process.stdin.close()
        output = process.stdout.read()
        error = process.stderr.read()
        if process.wait(timeout=30) != 0:
            raise RuntimeError(f"sha256sum failed: {error!r}")
    return output.decode().split()[0]


class Probe:
    def __init__(self, output: Path, cpu_only: bool) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "results.json"
        if self.output.exists():
            raise FileExistsError(f"Use a fresh output directory: {self.output}")
        self.started = time.monotonic()
        self.data: dict[str, Any] = {
            "status": "running",
            "cpu_only": cpu_only,
            "rows": [],
            "checkpoint": str(CHECKPOINT),
            "expected": EXPECTED,
            "chunk_bytes": CHUNK,
            "memory_iterations": REPEATS,
            "provenance": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "openssl": ssl.OPENSSL_VERSION,
                "hashlib_sha256_type": str(type(hashlib.sha256())),
                "hashlib_sha256_module": hashlib.sha256.__module__,
                "software_sha256_type": str(type(_sha256.sha256())),
                "modules": {
                    m.__name__: m.__file__ for m in (_hashlib, _sha256, hashlib)
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

    def phase(self, phase: str, blocks: tuple[bytes, ...], known: bytes) -> None:
        start = time.monotonic()
        before = stat_record(CHECKPOINT)
        for label, buffers, expected in (
            ("checkpoint_memory", blocks, EXPECTED),
            ("known_64mib", (known,), self.data["known_expected"]),
        ):
            for iteration in range(REPEATS):
                openssl, software = hashlib.sha256(), _sha256.sha256()
                for block in buffers:
                    openssl.update(block)
                    software.update(block)
                digests = {
                    "hashlib": openssl.hexdigest(),
                    "_sha256": software.hexdigest(),
                }
                self.row(
                    phase,
                    label,
                    iteration,
                    digests=digests,
                    expected=expected,
                    match=all(value == expected for value in digests.values()),
                )
            digest = system_memory(buffers)
            self.row(
                phase,
                label + "_sha256sum_stdin",
                0,
                digest=digest,
                expected=expected,
                match=digest == expected,
            )

        openssl, software = hashlib.sha256(), _sha256.sha256()
        mismatches = []
        count = 0
        with CHECKPOINT.open("rb") as stream:
            while block := stream.read(CHUNK):
                openssl.update(block)
                software.update(block)
                block_openssl = hashlib.sha256(block).hexdigest()
                block_software = _sha256.sha256(block).hexdigest()
                # Exact byte comparison detects changed reads independently of hashing.
                same_bytes = count < len(blocks) and block == blocks[count]
                if block_openssl != block_software or not same_bytes:
                    mismatches.append(
                        {
                            "index": count,
                            "offset": count * CHUNK,
                            "length": len(block),
                            "same_as_retained_bytes": same_bytes,
                            "hashlib": block_openssl,
                            "_sha256": block_software,
                        }
                    )
                count += 1
        digests = {"hashlib": openssl.hexdigest(), "_sha256": software.hexdigest()}
        self.row(
            phase,
            "disk_same_chunk_dual_hash",
            0,
            digests=digests,
            expected=EXPECTED,
            block_count=count,
            mismatched_blocks=mismatches,
            match=not mismatches
            and count == len(blocks)
            and all(value == EXPECTED for value in digests.values()),
        )
        system = subprocess.run(
            ["sha256sum", str(CHECKPOINT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        ).stdout.split()[0]
        self.row(
            phase,
            "disk_sha256sum",
            0,
            digest=system,
            expected=EXPECTED,
            match=system == EXPECTED,
        )
        after = stat_record(CHECKPOINT)
        self.row(
            phase,
            "file_stat",
            0,
            before=before,
            after=after,
            match=before == after == self.data["initial_stat"],
            phase_seconds=time.monotonic() - start,
        )


def cuda_workload(probe: Probe, blocks: tuple[bytes, ...], known: bytes) -> None:
    # Imports intentionally occur AFTER the stdlib-only baseline.
    import numpy as np
    import torch

    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models import Pose2DRequest, ViTPosePose2D
    from src.submodules.models.tracker.common import TrackResult
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )

    probe.data["provenance"]["torch"] = torch.__version__
    probe.data["provenance"]["torch_cuda"] = torch.version.cuda
    probe.data["provenance"]["cuda_initialized_after_imports"] = (
        torch.cuda.is_initialized()
    )
    probe.save()
    imported_digests = {
        "hashlib": hashlib.sha256(known).hexdigest(),
        "_sha256": _sha256.sha256(known).hexdigest(),
        "sha256sum_stdin": system_memory((known,)),
    }
    probe.row(
        "after_imports_before_cuda_pose",
        "known_64mib",
        0,
        digests=imported_digests,
        expected=probe.data["known_expected"],
        match=all(
            value == probe.data["known_expected"] for value in imported_digests.values()
        ),
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
        torch.cuda.synchronize()
        probe.data["inference_seconds"] = time.monotonic() - started
        probe.data["provenance"]["gpu"] = torch.cuda.get_device_name()
        probe.save()
        probe.phase("after_cuda_pose_synchronized", blocks, known)
    finally:
        pose.unload()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    probe.phase("after_unload_synchronized", blocks, known)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run stdlib hashes only; do not import torch or model code.",
    )
    args = parser.parse_args()
    probe = Probe(args.output_dir, args.cpu_only)
    try:
        initial = stat_record(CHECKPOINT)
        if initial["st_size"] > 3 * 1024**3:
            raise ValueError(
                "Checkpoint exceeds the fixed 3 GiB retained-memory budget"
            )
        probe.data["initial_stat"] = initial
        with CHECKPOINT.open("rb") as stream:
            blocks = tuple(iter(lambda: stream.read(CHUNK), b""))
        probe.row(
            "load",
            "file_stat",
            0,
            before=initial,
            after=stat_record(CHECKPOINT),
            match=initial == stat_record(CHECKPOINT),
        )
        known = bytes(range(256)) * (64 * 1024 * 1024 // 256)
        probe.data["known_pattern"] = "bytes(range(256)) repeated to 64 MiB"
        probe.data["known_expected"] = _sha256.sha256(known).hexdigest()
        probe.phase("before_imports_and_cuda", blocks, known)
        if not args.cpu_only:
            cuda_workload(probe, blocks, known)
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
