"""Deterministic sweep: N real court photos + N B00 test samples through the classical baseline.

All artifacts stay under /tmp/tcd_work/extra. No repo writes, no threshold tuning.
Each docker run is bounded by PER_RUN_TIMEOUT_S; a client-side hang (observed once with
Docker 29 on this host, container finished but `docker run` never returned) is retried once
and recorded as an infrastructure event rather than a detector failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(
    os.environ.get("TENNIS_LAB_REPO_ROOT", "/home/kamimura/projects/tennis-lab")
).resolve()
WORK_ROOT = Path(os.environ.get("TCD_WORK_DIR", "/tmp/tcd_work")).resolve()

sys.path.insert(0, str(REPOSITORY_ROOT))
from src.utils.data.float32_store import read_float32  # noqa: E402

REAL_DIR = REPOSITORY_ROOT / "data/court/images"
DATASET = Path(
    REPOSITORY_ROOT / "data/synthetic_data_generation/scenes/B00/datasets/court"
)
EXTRA = WORK_ROOT / "extra"
PNG_DIR = EXTRA / "png"
AVI_DIR = EXTRA / "avi"
OUT_DIR = EXTRA / "out"
for d in (PNG_DIR, AVI_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_REAL = 10
N_SYNTH = 10
IMAGE = "tcd-opencv4:local"
PER_RUN_TIMEOUT_S = 180.0


def evenly_spaced(items: list, count: int) -> list:
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def make_avi(png: Path, avi: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(png),
            "-frames:v",
            "3",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-f",
            "avi",
            str(avi),
        ],
        check=True,
    )


def _single_attempt(avi: Path, tag: str, attempt: int) -> dict[str, Any]:
    result_txt = OUT_DIR / f"{tag}.txt"
    overlay = OUT_DIR / f"{tag}_overlay.png"
    log = OUT_DIR / f"{tag}.log"
    name = f"tcd_{tag}_{attempt}"
    started = time.time()
    timed_out = False
    with log.open("w") as stream:
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--name",
                    name,
                    "-e",
                    f"TCD_OVERLAY=/out/{tag}_overlay.png",
                    "-v",
                    f"{OUT_DIR}:/out",
                    "-v",
                    f"{AVI_DIR}:/in",
                    IMAGE,
                    "/usr/bin/time",
                    "-v",
                    "/build/build/detect",
                    f"/in/{avi.name}",
                    f"/out/{tag}.txt",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=WORK_ROOT,
                timeout=PER_RUN_TIMEOUT_S,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    wall = time.time() - started
    log_text = log.read_text()
    stat: dict[str, Any] = {
        "attempt": attempt,
        "exit_code": exit_code,
        "client_timed_out": timed_out,
        "client_wall_s": round(wall, 2),
        "elapsed_wall_s": None,
        "max_rss_kb": None,
        "completed_per_container_log": "Exit status:" in log_text,
    }
    for line in log_text.splitlines():
        if "Elapsed (wall clock) time" in line:
            parts = line.split(": ", 1)[1].strip().split(":")
            stat["elapsed_wall_s"] = (
                float(parts[-1]) + 60 * float(parts[-2]) + 3600 * float(parts[0])
            )
        if "Maximum resident set size" in line:
            stat["max_rss_kb"] = int(line.split(": ", 1)[1].strip())
        if line.startswith("Processing error:"):
            stat["processing_error"] = line
    stat["overlay"] = str(overlay) if overlay.exists() else None
    stat["result_file"] = str(result_txt) if result_txt.exists() else None
    stat["log"] = str(log)
    return stat


def run_detect(avi: Path, tag: str) -> dict[str, Any]:
    first = _single_attempt(avi, tag, 1)
    # Retry only when the container itself completed (client hang) or timed out.
    if first["client_timed_out"] or (first["exit_code"] is None):
        second = _single_attempt(avi, tag, 2)
        second["retry_after"] = first
        return second
    return first


def parse_result(path: Path, width: int, height: int) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    points = []
    malformed = 0
    for line in lines:
        try:
            x, y = line.split(";")
            points.append((float(x), float(y)))
        except ValueError:
            malformed += 1
    if not points:
        return {"line_count": len(lines), "malformed_lines": malformed, "points": 0}
    arr = np.array(points, dtype=np.float64)
    xs, ys = arr[:, 0], arr[:, 1]
    bbox_w = float(xs.max() - xs.min())
    bbox_h = float(ys.max() - ys.min())
    inside = int(((xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)).sum())
    return {
        "line_count": len(lines),
        "malformed_lines": malformed,
        "points": len(points),
        "finite_points": int(np.isfinite(arr).all(axis=1).sum()),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "bbox_w_frac": bbox_w / width,
        "bbox_h_frac": bbox_h / height,
        "points_inside_image": inside,
        # Declared before the sweep as a reporting aid, not a tuned parameter:
        # a fit whose 16 points span <10% of the frame in either axis is collapsed.
        "degenerate_bbox_lt_10pct": bool(
            bbox_w / width < 0.10 or bbox_h / height < 0.10
        ),
    }


def main() -> int:
    real_pngs = evenly_spaced(sorted(REAL_DIR.glob("*.png")), N_REAL)
    dataset = json.loads((DATASET / "dataset.json").read_text())
    tests = sorted(
        (s for s in dataset["samples"] if s["split"] == "test"),
        key=lambda s: s["sample_index"],
    )
    synth = evenly_spaced(tests, N_SYNTH)

    records = []
    for index, png in enumerate(real_pngs):
        tag = f"real_{index:02d}_{png.stem}"
        target = PNG_DIR / f"{tag}.png"
        if not target.exists():
            target.write_bytes(png.read_bytes())
        avi = AVI_DIR / f"{tag}.ffv1.avi"
        make_avi(target, avi)
        stat = run_detect(avi, tag)
        width, height = Image.open(target).size
        record = {
            "kind": "real",
            "tag": tag,
            "source": str(png),
            "source_png": str(target),
            "avi": str(avi),
            "width": width,
            "height": height,
            **stat,
        }
        if stat["result_file"]:
            record["result"] = parse_result(Path(stat["result_file"]), width, height)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    for index, sample in enumerate(synth):
        tag = f"synth_{index:02d}_{sample['sample_id']}"
        target = PNG_DIR / f"{tag}.png"
        if not target.exists():
            rgb = read_float32(DATASET / sample["rgb"])
            u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(u8, mode="RGB").save(target, compress_level=9)
        avi = AVI_DIR / f"{tag}.ffv1.avi"
        make_avi(target, avi)
        stat = run_detect(avi, tag)
        width, height = Image.open(target).size
        record = {
            "kind": "synth",
            "tag": tag,
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "dataset_index": sample["sample_index"],
            "trajectory_group_id": sample["trajectory_group_id"],
            "rgb_store": sample["rgb"],
            "visible_point_count": sample["projection"]["visible_point_count"],
            "coverage_modes": sample["projection"]["coverage_modes"],
            "court_instances": len(sample["projection"]["courts"]),
            "source_png": str(target),
            "avi": str(avi),
            "width": width,
            "height": height,
            **stat,
        }
        if stat["result_file"]:
            record["result"] = parse_result(Path(stat["result_file"]), width, height)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "n_real": len(real_pngs),
        "n_synth": len(synth),
        "per_run_timeout_s": PER_RUN_TIMEOUT_S,
        "real_selection": "sorted(data/court/images/*.png) sampled evenly to 10",
        "synth_selection": "B00 test split, sorted by sample_index, sampled evenly to 10",
        "records": records,
    }
    (EXTRA / "sweep_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
