"""Persist complete, manifest-scoped prediction rows for the comparison."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.utils.schema.player import NUM_HUMAN_KP

SCHEMA = "player_model_predictions_v1"


class PredictionHeader(NamedTuple):
    benchmark_id: str
    manifest_sha256: str
    model_id: str
    backend: str
    has_pose: bool

    def as_json(self) -> dict[str, object]:
        return {"schema": SCHEMA, **self._asdict()}


def _require_cohort(keys: Iterable[FrameKey], expected: Iterable[FrameKey]) -> None:
    actual, cohort = set(keys), set(expected)
    if actual != cohort:
        raise ValueError(
            f"Predictions must contain the full fixed cohort: "
            f"{len(cohort - actual)} missing frames, {len(actual - cohort)} unexpected"
        )


def write_predictions(
    path: Path,
    header: PredictionHeader,
    predictions: Mapping[FrameKey, Predictions],
    *,
    expected_frames: Iterable[FrameKey],
) -> None:
    _require_cohort(predictions, expected_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(header.as_json(), allow_nan=False) + "\n")
            for key, row in sorted(predictions.items()):
                joints = row.keypoints_xyc
                if (joints is not None) != header.has_pose:
                    raise ValueError(f"Unexpected pose availability at {key}")
                record = {
                    **key._asdict(),
                    "boxes_xyxy": row.boxes_xyxy.tolist(),
                    "scores": row.scores.tolist(),
                    "keypoints_xyc": None if joints is None else joints.tolist(),
                }
                handle.write(
                    json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_predictions(
    path: Path,
    header: PredictionHeader,
    *,
    expected_frames: Iterable[FrameKey],
) -> dict[FrameKey, Predictions]:
    result: dict[FrameKey, Predictions] = {}
    with path.open(encoding="utf-8") as handle:
        saved_header = json.loads(handle.readline())
        if (
            saved_header != header.as_json()
            or type(saved_header["has_pose"]) is not bool
        ):
            raise ValueError(
                f"Prediction header differs from the selected condition: {path}"
            )
        for line in handle:
            row = json.loads(line)
            key = FrameKey(row["clip_id"], row["camera_id"], row["source_frame"])
            if type(key.source_frame) is not int:
                raise ValueError(f"Expected an integer source frame: {key}")
            if key in result:
                raise ValueError(f"Duplicate prediction frame: {key}")

            boxes = row["boxes_xyxy"]
            pose = row["keypoints_xyc"]
            if (pose is not None) != header.has_pose:
                raise ValueError(f"Unexpected pose availability at {key}")
            joints = None
            if pose is not None:
                joints = (
                    np.empty((0, NUM_HUMAN_KP, 3), np.float32)
                    if pose == []
                    else np.asarray(pose, dtype=np.float32)
                )
                if not np.isfinite(joints).all():
                    raise ValueError(f"Non-finite pose output at {key}")
            result[key] = Predictions(
                np.empty((0, 4), np.float32)
                if boxes == []
                else np.asarray(boxes, np.float32),
                np.asarray(row["scores"], np.float32),
                joints,
            )
    _require_cohort(result, expected_frames)
    return result
