"""Open an explicitly pinned player evaluation bundle on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.evaluation.player_pose.reference import (
    BoxQuality,
    JointQuality,
    joint_boxes,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, ReferenceFrame
from src.utils.checksum import dual_sha256
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP

MANIFEST_SCHEMA = "multiview_player_evaluation_v1"
BODY_INPUT_SCHEMA = "multiview_body_recovery_inputs_v1"


def pinned_file(root: Path, record: dict[str, Any]) -> Path:
    """Verify a manifest-relative file before use."""
    root = root.resolve()
    relative = Path(record["path"])
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or not path.is_relative_to(root)
        or dual_sha256(path) != record["sha256"]
    ):
        raise ValueError(
            f"Bundle file path or SHA-256 differs from the pin: {relative}"
        )
    return path


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def load_manifest(root: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Pin the cohort and its per-clip file hashes to an expected manifest SHA-256."""
    doc = _object(
        pinned_file(root, {"path": "manifest.json", "sha256": expected_sha256})
    )
    if (
        doc.get("schema") != MANIFEST_SCHEMA
        or tuple(doc.get("joint_order", ())) != COCO_KP_NAMES
    ):
        raise ValueError("Unsupported player evaluation schema or COCO17 order")
    return doc


def load_body_input_manifest(root: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load fixed body recovery inputs using the same outer cohort pin as pose scoring."""
    cohort = load_manifest(root, expected_sha256=expected_sha256)
    body = _object(pinned_file(root, cohort["body_recovery_manifest"]))
    axes = (
        "clip_id",
        "player_ids",
        "camera_ids",
        "evaluation_frames",
        "source_frames",
        "width",
        "height",
        "fps",
    )
    if (
        body.get("schema") != BODY_INPUT_SCHEMA
        or body.get("cohort_id") != cohort["benchmark_id"]
        or body.get("source_benchmark") != cohort["source_benchmark"]
        or body.get("joint_order") != cohort["joint_order"]
        or [[clip[key] for key in axes] for clip in body["clips"]]
        != [[clip[key] for key in axes] for clip in cohort["clips"]]
    ):
        raise ValueError("Body recovery manifest disagrees with the cohort")
    return body


def load_reference_frames(
    root: Path, *, expected_sha256: str, box_margin: float | None = None
) -> dict[FrameKey, ReferenceFrame]:
    """Load every evaluation image, including images with zero eligible players.

    An explicit margin override permits sensitivity analysis with the same
    reference rows and joint masks.
    """
    doc = load_manifest(root, expected_sha256=expected_sha256)
    result = {}
    for clip in doc["clips"]:
        cameras = clip["camera_ids"]
        ids = np.asarray(clip["player_ids"], np.int64)
        frames = np.asarray(clip["evaluation_frames"], np.int64)
        width, height = clip["width"], clip["height"]
        expected = (len(ids), len(cameras), clip["source_frames"])
        if (
            len(frames) == 0
            or np.any((frames < 0) | (frames >= expected[2]))
            or np.any(np.diff(frames) <= 0)
        ):
            raise ValueError(f"Invalid evaluation frame axis for {clip['clip_id']}")
        with np.load(
            pinned_file(root, clip["pseudo_ground_truth"]), allow_pickle=False
        ) as data:
            uv = data["keypoints_px"]
            boxes = data["boxes_xyxy"]
            box_valid = data["box_reasons"] == BoxQuality.ELIGIBLE
            joint_valid = data["joint_reasons"] == JointQuality.ELIGIBLE
            if (
                uv.shape != expected + (NUM_HUMAN_KP, 2)
                or boxes.shape != expected + (4,)
                or box_valid.shape != expected
                or joint_valid.shape != expected + (NUM_HUMAN_KP,)
                or not np.array_equal(ids, data["player_ids"])
                or not np.array_equal(cameras, data["camera_ids"])
                or not np.array_equal(frames, data["evaluation_frames"])
            ):
                raise ValueError(
                    f"Pseudo-ground-truth axes differ from the manifest for {clip['clip_id']}"
                )
            if box_margin is not None:
                boxes, valid = joint_boxes(
                    uv,
                    data["in_image"],
                    (width, height),
                    margin=box_margin,
                    min_joints=doc["reference_policy"]["min_box_joints"],
                )
                if (box_valid & ~valid).any():
                    raise ValueError("Margin override collapses some eligible boxes")
        for view, camera in enumerate(cameras):
            for frame in frames:
                keep = box_valid[:, view, frame]
                key = FrameKey(clip["clip_id"], camera, int(frame))
                if key in result:
                    raise ValueError(f"Duplicate evaluation frame: {key}")
                result[key] = ReferenceFrame(
                    ids[keep],
                    boxes[keep, view, frame],
                    uv[keep, view, frame],
                    joint_valid[keep, view, frame],
                )
    return result
