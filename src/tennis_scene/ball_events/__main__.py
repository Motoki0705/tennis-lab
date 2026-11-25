"""Run the rules on every scene in a pinned multiview benchmark."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.ball_events import RuleThresholds, detect_scene_events
from src.tennis_scene.ball_events.rules import (
    DEFAULT_THRESHOLDS,
    EVENT_KINDS,
    BallEvent,
)
from src.utils.checksum import FileIntegrityError
from src.utils.configuration.output_layout import new_run_id, task_output_path
from src.utils.io import write_json_atomic

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True, slots=True)
class _Clip:
    clip_id: str
    frames: int
    fps: float
    scene_path: Path
    scene_sha256: str


def _read_benchmark(
    root: Path, manifest_sha256: str, clip_ids: list[str] | None
) -> tuple[str, list[_Clip]]:
    data = (root / "manifest.json").read_bytes()
    if sha256(data).hexdigest() != manifest_sha256:
        raise ValueError("Benchmark manifest SHA-256 mismatch")
    document = json.loads(data)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "tennis_multiview_benchmark_v1"
        or not isinstance(document.get("benchmark_id"), str)
        or not isinstance(document.get("clips"), list)
        or not isinstance(document.get("totals"), dict)
        or len(document["clips"]) != document["totals"].get("clips")
        or not document["clips"]
    ):
        raise ValueError("Invalid multiview benchmark manifest")
    clips: dict[str, _Clip] = {}
    for row in document["clips"]:
        clip = _clip_from_row(root, row)
        if clip.clip_id in clips:
            raise ValueError(f"Duplicate clip ID: {clip.clip_id}")
        clips[clip.clip_id] = clip
    if clip_ids is None:
        return str(document["benchmark_id"]), list(clips.values())
    if len(set(clip_ids)) != len(clip_ids) or not set(clip_ids) <= clips.keys():
        raise ValueError("Unknown or duplicate benchmark clip IDs")
    return str(document["benchmark_id"]), [clips[clip] for clip in clip_ids]


def _clip_from_row(root: Path, row: Any) -> _Clip:
    clip_id, frames, fps = row["clip_id"], row["frames"], row["fps"]
    location = Path(row["scene"]["path"])
    scene_sha256 = row["scene"]["sha256"]
    if (
        not isinstance(clip_id, str)
        or _CLIP_ID.fullmatch(clip_id) is None
        or type(frames) is not int
        or frames < 1
        or type(fps) not in (int, float)
        or not isfinite(fps)
        or fps <= 0
        or not isinstance(scene_sha256, str)
        or _SHA256.fullmatch(scene_sha256) is None
    ):
        raise ValueError(
            f"Invalid clip ID, timeline or scene checksum in manifest: {clip_id!r}"
        )
    scene_path = (root / location).resolve()
    if (
        location.is_absolute()
        or location.name != "scene.json"
        or not scene_path.is_relative_to(root)
    ):
        raise ValueError(f"Invalid scene index path in manifest: {location}")
    return _Clip(clip_id, frames, float(fps), scene_path, scene_sha256)


def _predict(clip: _Clip, thresholds: RuleThresholds) -> tuple[list[BallEvent], int]:
    if sha256(clip.scene_path.read_bytes()).hexdigest() != clip.scene_sha256:
        raise ValueError("Scene index SHA-256 mismatch")
    scene = load_scene_result(clip.scene_path)
    if (scene.num_frames, scene.fps) != (clip.frames, clip.fps):
        raise ValueError("Scene timeline disagrees with benchmark manifest")
    events = detect_scene_events(scene, thresholds=thresholds)
    valid_frames = int(np.count_nonzero(scene.ball_3d_valid))
    if valid_frames < 3:
        raise ValueError(
            f"At least three valid ball frames are required, got {valid_frames}"
        )
    return events, valid_frames


def run_benchmark(
    benchmark: Path,
    manifest_sha256: str,
    output: Path,
    *,
    method_version: str,
    clip_ids: list[str] | None = None,
) -> bool:
    """Write one prediction per input plus an atomic progress receipt."""
    benchmark = benchmark.resolve()
    benchmark_id, clips = _read_benchmark(benchmark, manifest_sha256, clip_ids)
    output = output.resolve()
    if output.is_relative_to(benchmark):
        raise ValueError("Output must be outside the read-only benchmark")
    output.mkdir(parents=True, exist_ok=False)
    thresholds = DEFAULT_THRESHOLDS
    rows: dict[str, dict[str, object]] = {}
    summary: dict[str, object] = {
        "schema": "ball_event_prediction_progress.v1",
        "benchmark_id": benchmark_id,
        "manifest_sha256": manifest_sha256,
        "method": "rule",
        "method_version": method_version,
        "thresholds": asdict(thresholds),
        "total": len(clips),
        "clip_ids": [clip.clip_id for clip in clips],
        "clips": rows,
    }

    def save_progress(state: str, successes: int, failures: int) -> None:
        write_json_atomic(
            output / "progress.json",
            {
                **summary,
                "state": state,
                "ok": successes,
                "failed": failures,
                "pending": len(clips) - successes - failures,
            },
        )

    successes = failures = 0
    save_progress("running", successes, failures)
    for index, clip in enumerate(clips, 1):
        events: list[BallEvent] = []
        valid_frames = None
        reason = None
        try:
            events, valid_frames = _predict(clip, thresholds)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            FileIntegrityError,
            BadZipFile,
        ) as error:
            reason = f"{type(error).__name__}: {error}"
        status = "ok" if reason is None else "failed"
        path = f"predictions/{clip.clip_id}.json"
        write_json_atomic(
            output / path,
            {
                "schema": "ball_event_predictions.v1",
                "clip_id": clip.clip_id,
                "fps": clip.fps,
                "frames": clip.frames,
                "method": "rule",
                "method_version": method_version,
                "inputs": {"scene_sha256": clip.scene_sha256},
                "status": status,
                "reason": reason,
                "events": events,
            },
        )
        successes += reason is None
        failures += reason is not None
        rows[clip.clip_id] = {
            "status": status,
            "reason": reason,
            "predictions": path,
            "valid_ball_frames": valid_frames,
            "events": {
                kind: sum(event.kind == kind for event in events)
                for kind in EVENT_KINDS
            },
            "events_spanning_ball_gaps": sum(
                "spans_ball_gap" in event.flags for event in events
            ),
        }
        state = (
            "running" if index < len(clips) else ("failed" if failures else "complete")
        )
        save_progress(state, successes, failures)
    print(f"{successes}/{len(clips)} clips OK. Progress: {output / 'progress.json'}")
    return failures == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=new_run_id())
    parser.add_argument(
        "--clip-id", action="append", help="Select a clip. Repeat to select multiple."
    )
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    run_path = task_output_path(
        "tennis_scene", "analyze", "rule_ball_events", args.run_id
    )
    return (
        0
        if run_benchmark(
            args.benchmark,
            args.manifest_sha256,
            args.output_root / run_path,
            method_version=commit,
            clip_ids=args.clip_id,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
