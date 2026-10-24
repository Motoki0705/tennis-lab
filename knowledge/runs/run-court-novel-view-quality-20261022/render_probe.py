"""Run the public renderer from a training-queue allocation, with batch bounds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from plan_probe import dump


def render(scene: Path, request: Path, output: Path, batch: int) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Refuse to replace renderer output: {output}")
    samples: list[int] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            samples.append(int(result.stdout.strip()))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor)
    command = [
        "nht-render",
        "--scene",
        str(scene),
        "--cameras",
        str(request),
        "--output",
        str(output),
        "--batch-size",
        str(batch),
    ]
    start = time.monotonic()
    thread.start()
    try:
        subprocess.run(command, check=True, timeout=1800)
    finally:
        stop.set()
        thread.join()
    report: dict[str, Any] = {
        "command": command,
        "wall_seconds": time.monotonic() - start,
        "gpu_used_peak_mib": max(samples),
        "gpu_used_initial_mib": samples[0],
        "cameras": len(json.loads(request.read_text())["cameras"]),
        "batch_size": batch,
    }
    dump(output.parent / (output.name + "-timing.json"), report)
    if report["gpu_used_peak_mib"] > 10000:
        raise RuntimeError(
            "Measured GPU usage exceeded the conservative 10000 MiB probe limit"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("GPU renderer must run inside training queue")
    root, output = (
        args.scenes_root.resolve(strict=True),
        args.plan_root.resolve(strict=True),
    )
    if output.is_relative_to(root):
        raise ValueError("Renderer output must be outside read-only input root")
    if args.smoke:
        scene = root / "B00/reconstruction/export/scene.json"
        for batch in [1, 4]:
            print(
                json.dumps(
                    render(
                        scene,
                        output / f"smoke-{batch}.json",
                        output / f"smoke-{batch}",
                        batch,
                    )
                ),
                flush=True,
            )
    else:
        reports = []
        # Require both capacity probes before rendering the paired inventory.
        for batch in [1, 4]:
            probe = json.loads((output / f"smoke-{batch}-timing.json").read_text())
            if probe["gpu_used_peak_mib"] > 10000:
                raise RuntimeError("Capacity probe failed")
        for sid in ["B00", "B01", "B02", "B03"]:
            for condition in ["recommended", "hull5", "hfov45_90", "jitter0"]:
                directory = output / sid / condition
                report = render(
                    root / sid / "reconstruction/export/scene.json",
                    directory / "request.json",
                    directory / "renders",
                    4,
                )
                reports.append({"scene": sid, "condition": condition, **report})
                print(json.dumps(reports[-1]), flush=True)
        dump(output / "render-timings.json", reports)


if __name__ == "__main__":
    main()
