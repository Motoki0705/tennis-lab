"""Collect read-only environment, planned input sizes and Drive acknowledgements."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from src.synthetic_data_generation.dataset.court.scale.durability import _input_files
from src.synthetic_data_generation.dataset.court.scale.publication import (
    iter_plan_groups,
)


def command(arguments: list[str]) -> str:
    return subprocess.run(
        arguments, check=True, capture_output=True, text=True
    ).stdout.strip()


def plan_inputs(root: Path) -> dict[str, object]:
    requests = list((root / "requests").glob("*.json"))
    assert len(requests) == 1
    request = requests[0]
    files = _input_files(root, request)
    sizes = [(root / name).stat().st_size for name in sorted(set(files))]
    tar_payload = sum(512 + ((size + 511) // 512) * 512 for size in sizes)
    archive_bytes = ((tar_payload + 1024 + 10239) // 10240) * 10240
    frozen = json.loads(request.read_text())
    scenes = {}
    for scene in frozen["scenes"]:
        plan = json.loads((root / scene["plan_path"]).read_text())
        scenes[scene["scene_id"]] = [
            {
                "group_id": group["trajectory_id"],
                "base_count": None
                if group["arc_map"] is None
                else group["arc_map"]["base_count"],
                "candidates": len(group["samples"]),
                "geometry_pass": sum(not item["reasons"] for item in group["samples"]),
            }
            for group in iter_plan_groups(root, plan)
        ]
    return {
        "root": root.as_posix(),
        "input_files": len(sizes),
        "raw_input_bytes": sum(sizes),
        "uncompressed_ustar_bytes": archive_bytes,
        "scenes": scenes,
        "rendered": root.name == "court-scale-pilot-resumed",
    }


def main() -> None:
    target = Path(
        "outputs/synthetic_data_generation/profile/court-scale-pilot/evidence/final"
    )
    assert not target.exists()
    target.mkdir()
    packages = (
        "torch",
        "numpy",
        "scipy",
        "pytorch-lightning",
        "gsplat",
        "nht",
        "opencv-python",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        "source_commit": command(["git", "rev-parse", "HEAD"]),
        "tracked_diff": command(["git", "diff", "--stat"]),
        "submodules": command(
            ["git", "submodule", "status", "third_party/nht", "third_party/dinov3"]
        ),
        "python": sys.version,
        "packages": versions,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    (target / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    plans = {"pilot": plan_inputs(Path("tmp/court-scale-pilot-resumed"))}
    for label in ("small", "dense", "single", "base"):
        plans[label] = plan_inputs(Path(f"tmp/court-pilot-plan-cost-resumed-{label}"))
    (target / "plan-inputs.json").write_text(json.dumps(plans, indent=2) + "\n")
    run = Path("outputs/court_detection/train/court-scale-pilot-l4/s42")
    events = [
        {"path": path.as_posix(), "bytes": path.stat().st_size}
        for path in sorted(run.rglob("events.out.tfevents.*"))
    ]
    (target / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    drive = [sys.executable, ".agents/skills/tennis-drive/scripts/drive.py"]
    checkpoints = run / "logs/version_0/checkpoints"
    verification = json.loads(
        command(drive + ["verify", str(checkpoints), str(checkpoints)])
    )
    (target / "checkpoint-drive-verify.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    paths = [
        str(run),
        "ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "data/court_detection/tennis_court_detector-v1",
    ]
    inspections = {
        path: json.loads(command(drive + ["inspect", path])) for path in paths
    }
    (target / "storage.json").write_text(json.dumps(inspections, indent=2) + "\n")
    print(
        json.dumps(
            {
                "environment": environment,
                "events": events,
                "checkpoints_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
