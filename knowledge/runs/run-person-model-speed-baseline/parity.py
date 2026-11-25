"""Adapt stored baseline outputs to the shared model parity contract."""

from __future__ import annotations

import numpy as np

from src.utils.inference_parity import (
    BodyOutput,
    Coco17Output,
    DetectionOutput,
    compare_bodies,
    compare_coco17,
    compare_detections,
)


def _empty_array(value, empty_shape):
    if isinstance(value, list) and not value:
        return np.empty(empty_shape)
    return value


def _detection(row):
    poses = row.get("poses")
    return DetectionOutput(
        _empty_array(row["boxes"], (0, 4)),
        row["scores"],
        None if poses is None else Coco17Output(_empty_array(poses, (0, 17, 3))),
    )


def _body(row):
    params = row["params"]
    root = np.asarray(params["global_orient"], dtype=np.float64)
    pose = np.asarray(params["body_pose"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or pose.shape != (len(root), 63):
        raise ValueError(
            "GVHMR must contain the root rotation and 21 body rotations per frame"
        )
    return BodyOutput(
        row["joints"],
        np.concatenate((root[:, None], pose.reshape(len(root), 21, 3)), axis=1),
        params["transl"],
        features=row["features"],
        betas=params["betas"],
        frame_ids=None if row.get("frame_ids") is None else tuple(row["frame_ids"]),
    )


def compare(workload, reference, candidate):
    if workload in ("dino", "dino-vitpose"):
        return compare_detections(
            {key: _detection(value) for key, value in reference.items()},
            {key: _detection(value) for key, value in candidate.items()},
            with_pose=workload == "dino-vitpose",
        )
    if workload == "body":
        return compare_bodies(
            {key: _body(value) for key, value in reference.items()},
            {key: _body(value) for key, value in candidate.items()},
        )
    return compare_coco17(
        {key: Coco17Output(value) for key, value in reference.items()},
        {key: Coco17Output(value) for key, value in candidate.items()},
    )
