"""TrackNet ``<game>/<Clip*>/Label.csv`` plus numbered JPEG frames -> :class:`ClipSpec`.

This module is the only reader of the original TrackNet layout; the store is
the format every consumer reads. Label mapping:

* ``visibility`` 0: annotated frame without a visible ball (a negative).
* ``visibility`` 1 and 2 (clearly / hardly identifiable): ``observed``.
* ``visibility`` 3 (occluded, position labelled): ``occlusion_estimated``
  with ``occluded`` set.
* ``status`` 0/1/2 is flying/hit/bounce -> event ``none``/``hit``/``bounce``;
  the blank status of invisible frames is ``unlabeled``.
* Labels are integer pixels. A label exactly on the far image border
  (``x == width`` or ``y == height``; one frame in the release) is moved onto the
  last pixel and listed in the clip's ``provenance["border_clamped"]``; any other
  position outside the image is an error.

TrackNet does not carry a frame rate; the configured nominal ``fps`` (30 in
the TrackNet release) defines ``time_base = 1/fps`` and ``pts = frame_index``.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    ClipSpec,
    JpegSequence,
    SourceFrame,
)
from src.tennis_scene.chat_annotation.runtime.contracts import sha256_file

LABEL_FILE = "Label.csv"
COLUMNS = ("file name", "visibility", "x-coordinate", "y-coordinate", "status")
TRACK_ID = "ball"
_EVENTS = {"0": "none", "1": "hit", "2": "bounce"}


@dataclass(frozen=True, slots=True)
class TrackNetSourceConfig:
    root: Path
    fps: Fraction


def _parse_row(row: dict[str, str], path: Path) -> tuple[str, SourceFrame]:
    name = row["file name"]
    visibility = row["visibility"]
    status = row["status"]
    if visibility == "0":
        if status not in ("", "0"):
            raise ValueError(f"{path}: invisible frame {name} has event status {status!r}")
        return name, SourceFrame(0, True, True, False, "unlabeled" if status == "" else "none", ())
    if visibility not in ("1", "2", "3") or status not in _EVENTS:
        raise ValueError(f"{path}: frame {name} has visibility {visibility!r} / status {status!r}")
    xy = (float(row["x-coordinate"]), float(row["y-coordinate"]))
    occluded = visibility == "3"
    ball = BallInstance(TRACK_ID, "occlusion_estimated" if occluded else "observed", xy, occluded)
    return name, SourceFrame(0, True, True, False, _EVENTS[status], (ball,))


def read_clip(clip_dir: Path, game: str, fps: Fraction) -> ClipSpec:
    """Parse one TrackNet clip; every frame must be listed once, in file order."""
    label_path = clip_dir / LABEL_FILE
    with label_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{label_path}: expected columns {COLUMNS}, got {reader.fieldnames}")
        rows = [_parse_row(row, label_path) for row in reader]
    images = sorted(path.name for path in clip_dir.glob("*.jpg"))
    names = [name for name, _ in rows]
    expected = [f"{index:04d}.jpg" for index in range(len(names))]
    if names != expected or images != expected:
        raise ValueError(f"{clip_dir}: labels and JPEG files must both be 0000.jpg.. in order")
    first = cv2.imread(str(clip_dir / names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"{clip_dir / names[0]}: unreadable JPEG")
    height, width = first.shape[:2]
    labels: list[SourceFrame] = []
    clamped: list[dict[str, object]] = []
    for index, (name, label) in enumerate(rows):
        balls = []
        for ball in label.balls:
            if ball.xy is None:
                raise ValueError(f"{label_path}: TrackNet instance of {name} has no position")
            x, y = ball.xy
            if x == width or y == height:
                clamped.append({"frame": name, "from": [x, y]})
                ball = BallInstance(ball.track_id, ball.point_kind, (min(x, width - 1), min(y, height - 1)), ball.occluded)
            balls.append(ball)
        labels.append(SourceFrame(index, label.annotated, label.is_target, label.segment_break, label.event, tuple(balls)))
    paths = tuple(clip_dir / name for name in names)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return ClipSpec(
        clip_id=f"tracknet/{game}/{clip_dir.name}",
        source="tracknet",
        group_id=game,
        camera_id=None,
        width=width,
        height=height,
        time_base=1 / fps,
        fps=fps,
        has_events=True,
        annotation_path=label_path,
        annotation_sha256=sha256_file(label_path),
        # TrackNet has no container file: hash the frame JPEGs in order.
        media_sha256=digest.hexdigest(),
        media=JpegSequence(paths),
        labels=ClipLabels.from_frames(labels, width=width, height=height),
        provenance={"clip_dir": str(clip_dir), "border_clamped": clamped},
    )


def collect_tracknet(config: TrackNetSourceConfig) -> list[ClipSpec]:
    """Every ``<game>/<Clip*>`` below the root; anything else there is an error."""
    root = config.root
    games = sorted((path for path in root.iterdir()), key=lambda path: path.name)
    if not games:
        raise FileNotFoundError(f"No TrackNet games in {root}")
    specs: list[ClipSpec] = []
    for game in games:
        if not game.is_dir() or not game.name.startswith("game"):
            raise ValueError(f"Unexpected entry in the TrackNet root: {game}")
        clips = sorted(game.iterdir(), key=lambda path: (len(path.name), path.name))
        if not clips or any(not clip.is_dir() or not clip.name.startswith("Clip") for clip in clips):
            raise ValueError(f"{game}: expected only Clip* directories")
        specs.extend(read_clip(clip, game.name, config.fps) for clip in clips)
    located = sum(int(np.count_nonzero(spec.labels.instances["point_kind"] == 0)) for spec in specs)
    if located == 0:
        raise ValueError(f"TrackNet root {root} contains no visible ball")
    return specs
