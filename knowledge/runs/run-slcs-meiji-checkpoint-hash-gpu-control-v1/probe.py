"""One CUDA inference followed by the exact fresh-process CPU hash controls.

Run only through training queue, with `timeout --kill-after=5s 590s`.
This diagnostic retains the model during 16 alternating file checks and two
known-memory children, then checks each environment once after unloading.
No production checks, global environment, or OS settings are changed.
"""

import argparse
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CPU_SCRIPT = ROOT / "knowledge/runs/run-slcs-meiji-checkpoint-hash-cpu-v1/probe.py"
CPU = runpy.run_path(str(CPU_SCRIPT))
BaseProbe = CPU["Probe"]
CONDITIONS = CPU["CONDITIONS"]
POST_PREDICT = "after_predict_synchronized_model_loaded"
AFTER_UNLOAD = "after_unload"


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


class Probe(BaseProbe):
    def __init__(self, output: Path) -> None:
        self.phase = "initialization"
        super().__init__(output, iterations=8)
        self.data["control_implementation"] = str(CPU_SCRIPT)
        self.data["scope"] = (
            "GPU-following CPU hash diagnostic; no production workaround"
        )
        self.data["provenance"]["initial_process"] = process_snapshot()
        self.data["phase_interpretation"] = (
            "Phases are chronological and confounded by elapsed time. "
            "Fresh hash children do not import torch; only the parent holds the CUDA model."
        )
        self.save()

    def command(
        self, arguments: list[str], condition: str, timeout: float
    ) -> dict[str, Any]:
        row = super().command(arguments, condition, timeout)
        row["phase"] = self.phase
        return row

    def summarize(self) -> None:
        phase_reports = {}
        for phase, expected_per_condition, expected_memory_children in (
            (POST_PREDICT, 8, 2),
            (AFTER_UNLOAD, 1, 0),
        ):
            files = [row for row in self.data["file_rows"] if row["phase"] == phase]
            memory = [row for row in self.data["memory_rows"] if row["phase"] == phase]
            digests = {
                item[method]
                for row in memory
                if isinstance(row["report"], dict)
                for item in row["report"]["rows"]
                for method in ("hashlib", "_sha256")
            }
            file_summary = {
                condition: {
                    "runs": sum(row["condition"] == condition for row in files),
                    "expected_runs": expected_per_condition,
                    "mismatches_or_errors": sum(
                        row["condition"] == condition and not row["match"]
                        for row in files
                    ),
                }
                for condition in CONDITIONS
            }
            memory_match = len(memory) == expected_memory_children and all(
                row["match"] for row in memory
            )
            if expected_memory_children:
                memory_match = memory_match and digests == {
                    CPU["KNOWN_MEMORY_EXPECTED"]
                }
            passed = (
                all(
                    item["runs"] == expected_per_condition
                    and item["mismatches_or_errors"] == 0
                    for item in file_summary.values()
                )
                and memory_match
            )
            phase_reports[phase] = {
                "file": file_summary,
                "known_memory": {
                    "children": len(memory),
                    "expected_children": expected_memory_children,
                    "expected_digest": CPU["KNOWN_MEMORY_EXPECTED"],
                    "digests": sorted(digests),
                    "match": memory_match,
                    "performed": bool(expected_memory_children),
                },
                "match": passed,
            }
        self.data["summary"] = phase_reports
        provenance_failed = any(
            row["exit_code"] != 0 for row in self.data["provenance_commands"]
        )
        complete = (
            len(self.data["file_rows"]) == 18 and len(self.data["memory_rows"]) == 2
        )
        self.data["status"] = (
            "error"
            if provenance_failed
            else "passed"
            if complete and all(report["match"] for report in phase_reports.values())
            else "mismatch"
        )
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

    probe.data["provenance"].update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "process_after_imports": process_snapshot(),
        }
    )
    source = ROOT / "data/tennis_multivew/processed/meiji_3cam/dataset"
    manifest = load_dataset_manifest(source)
    clip = ClipManifest.load(source / manifest.clips["video_000/clip_007"].path)
    if clip.num_frames != 652:
        raise ValueError(
            f"Expected the fixed 652-frame workload; got {clip.num_frames}"
        )
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
        "precision": "bfloat16",
        "batch_size": 8,
        "flip_test": True,
    }
    probe.save()
    pose = ViTPosePose2D(
        CPU["CHECKPOINT"],
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
        probe.data["inference_and_sync_seconds"] = time.monotonic() - started
        probe.data["provenance"]["gpu"] = torch.cuda.get_device_name()
        probe.data["provenance"]["process_after_predict_sync"] = process_snapshot()
        probe.phase = POST_PREDICT
        probe.save()
        for iteration in range(8):
            for condition in CONDITIONS:
                probe.file_pass(condition, iteration)
        for condition in CONDITIONS:
            probe.memory_pass(condition)
    finally:
        pose.unload()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    probe.phase = AFTER_UNLOAD
    probe.data["provenance"]["process_after_unload"] = process_snapshot()
    for condition in CONDITIONS:
        probe.file_pass(condition, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the repository .venv/bin/python (Python 3.11)")
    probe = Probe(args.output_dir)
    try:
        probe.data["initial_stat"] = CPU["stat_record"]()
        probe.provenance()
        cuda_workload(probe)
        probe.summarize()
    except BaseException:
        probe.data["status"] = "error"
        probe.data["error"] = traceback.format_exc()
        raise
    finally:
        probe.save()
    return 0 if probe.data["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
