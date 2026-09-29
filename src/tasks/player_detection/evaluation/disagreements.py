"""Diagnose disagreement with old boxes without treating them as independent GT."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.evaluation.labels import AMBIGUOUS, CameraLabels
from src.tasks.player_detection.evaluation.partial_labels import _unit_matches
from src.utils.geometry.bbox import pairwise_iou


def player_unit_rows(
    boxes: NDArray[np.float32], labels: CameraLabels, roles: NDArray[np.str_],
    frame: int, *, min_iou: float,
) -> list[dict[str, Any]]:
    """Use the metric's one-to-one assignment, including non-player competitors.

    Height uses the largest-area old box per unit (not the best new match).
    Near/far is an image-space proxy: rank the old box bottoms when exactly two
    player units are present. Single-player frames and tied bottoms are unknown.
    Best IoU considers all duplicate old boxes, even for an unmatched unit.
    The caller must apply the same detection score threshold as the evaluation.
    """
    selection = labels.at(frame)
    people, old = labels.person_index[selection], labels.boxes_xyxy[selection]
    units = np.unique(people[people != AMBIGUOUS])
    overlaps = pairwise_iou(boxes, old)
    iou = np.column_stack([overlaps[:, people == person].max(axis=1) for person in units]) \
        if len(units) else np.empty((len(boxes), 0), np.float64)
    _, assigned_units = _unit_matches(iou, min_iou)
    rows = []
    for unit, person in enumerate(units):
        if roles[person] != "player":
            continue
        candidates = old[people == person]
        representative = candidates[np.prod(candidates[:, 2:] - candidates[:, :2], axis=1).argmax()]
        rows.append({"frame": frame, "person": int(person), "matched": bool(unit in assigned_units),
                     "best_iou": float(iou[:, unit].max()) if len(boxes) else 0.,
                     "box_height_px": float(representative[3] - representative[1]),
                     "old_box": representative.tolist(), "near_far": "unknown"})
    if len(rows) == 2 and rows[0]["old_box"][3] != rows[1]["old_box"][3]:
        near = int(rows[1]["old_box"][3] > rows[0]["old_box"][3])
        rows[near]["near_far"], rows[1 - near]["near_far"] = "near", "far"
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [row for row in rows if not row["matched"]]
    ious = [row["best_iou"] for row in missing]
    heights = [row["box_height_px"] for row in missing]
    iou_bins = [0., .1, .2, .3, .4, .5, 1.000001]
    height_bins = [0., 32., 64., 128., 256., 100000.]
    return {"units": len(rows), "unmatched": len(missing),
            "old_box_agreement": 1 - len(missing) / len(rows) if rows else None,
            "best_iou_bins": iou_bins, "best_iou_histogram": np.histogram(ious, iou_bins)[0].tolist(),
            "height_px_bins": height_bins, "height_px_histogram": np.histogram(heights, height_bins)[0].tolist(),
            "unmatched_height_px_p10_p50_p90": np.percentile(heights, [10, 50, 90]).tolist() if heights else None,
            "zero_overlap": sum(value == 0 for value in ious)}
