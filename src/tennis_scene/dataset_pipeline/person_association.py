"""Conservative image-space association for one incumbent in a court half."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PersonAssociation:
    """Fixed gates relative to the last detector box, including after long gaps."""

    min_iou: float
    max_center_distance: float
    max_prediction_frames: int = 0
    max_prediction_distance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("min_iou", "max_center_distance", "max_prediction_distance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"people.association.{name} must be finite numeric")
        if (
            type(self.max_prediction_frames) is not int
            or self.max_prediction_frames < 0
        ):
            raise ValueError(
                "people.association.max_prediction_frames must be a nonnegative integer"
            )
        if not 0 <= self.max_prediction_distance <= 1:
            raise ValueError(
                "people.association.max_prediction_distance must be in [0, 1]"
            )
        if (self.max_prediction_frames == 0) != (self.max_prediction_distance == 0):
            raise ValueError(
                "people.association prediction frame and distance limits must both be zero or positive"
            )
        if not 0 <= self.min_iou <= 1:
            raise ValueError("people.association.min_iou must be in [0, 1]")
        if not 0 < self.max_center_distance <= 1:
            raise ValueError("people.association.max_center_distance must be in (0, 1]")


def association_settings(settings: Mapping[str, Any]) -> PersonAssociation | None:
    """Reject incomplete/unused settings; omitted policy preserves legacy identity."""
    policy = settings.get("selection_policy", "largest")
    if policy not in ("largest", "temporal_continuity"):
        raise ValueError(
            "people.selection_policy must be largest or temporal_continuity"
        )
    raw = settings.get("association")
    if policy == "largest":
        if "association" in settings:
            raise ValueError("people.association requires temporal_continuity")
        return None
    required = {"min_iou", "max_center_distance"}
    prediction = {"max_prediction_frames", "max_prediction_distance"}
    if not isinstance(raw, Mapping) or set(raw) not in (
        required,
        required | prediction,
    ):
        raise ValueError(
            "people.association requires min_iou and max_center_distance with optional paired prediction limits"
        )
    return PersonAssociation(**raw)


def associate_single_person(
    history: list[list[dict[str, Any]]], settings: PersonAssociation
) -> list[list[dict[str, Any]]]:
    """Seed largest once, then retain spatial continuity without gap resets.

    Both fixed gates must pass. IoU ranks accepted candidates before normalized
    center distance; input order breaks exact ties. Normalization uses the last
    selected box diagonal, so a large entrant cannot enlarge its own gate.
    Explicit prediction limits enable median velocity from the last five actual
    accepted transitions, capped in elapsed frames and incumbent diagonals.
    A missing target leaves no detection. Even after long gaps, reacquisition
    uses the last actual or bounded predicted anchor with unchanged gates; it
    never starts a new largest-person track.
    These image-space gates cannot distinguish physically overlapping people.
    """
    result: list[list[dict[str, Any]]] = []
    anchor: np.ndarray | None = None
    anchor_frame = 0
    velocities: deque[np.ndarray] = deque(maxlen=5)
    for frame_index, frame in enumerate(history):
        predicted = anchor
        if anchor is not None and velocities and settings.max_prediction_frames:
            displacement = np.median(np.stack(velocities), axis=0) * min(
                frame_index - anchor_frame, settings.max_prediction_frames
            )
            distance = float(np.linalg.norm(displacement))
            limit = settings.max_prediction_distance * float(
                np.linalg.norm(anchor[2:] - anchor[:2])
            )
            if distance > limit:
                displacement *= limit / distance
            predicted = anchor + np.tile(displacement, 2)
        candidates: list[tuple[tuple[float, float], dict[str, Any], np.ndarray]] = []
        for detection in frame:
            box = np.asarray(detection["bbx_xyxy"], dtype=np.float64)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])
            ):
                raise ValueError("Person boxes must be finite positive xyxy boxes")
            area = float(np.prod(box[2:] - box[:2]))
            if anchor is None:
                rank = (area, 0.0)
            else:
                assert predicted is not None
                matches: list[tuple[float, float]] = []
                # Both endpoints of bounded motion uncertainty retain fixed gates.
                # A stale velocity must never discard a match to the last actual box.
                for reference in (anchor, predicted):
                    distance = float(
                        np.linalg.norm(
                            (box[:2] + box[2:] - reference[:2] - reference[2:]) / 2
                        )
                        / np.linalg.norm(anchor[2:] - anchor[:2])
                    )
                    intersection = float(
                        np.prod(
                            np.maximum(
                                0,
                                np.minimum(box[2:], reference[2:])
                                - np.maximum(box[:2], reference[:2]),
                            )
                        )
                    )
                    iou = intersection / (
                        area + float(np.prod(anchor[2:] - anchor[:2])) - intersection
                    )
                    if (
                        iou >= settings.min_iou
                        and distance <= settings.max_center_distance
                    ):
                        matches.append((iou, -distance))
                if not matches:
                    continue
                rank = max(matches)
            candidates.append((rank, detection, box))
        if not candidates:
            result.append([])
            continue
        _, selected, next_anchor = max(candidates, key=lambda candidate: candidate[0])
        if anchor is not None:
            velocities.append(
                (next_anchor[:2] + next_anchor[2:] - anchor[:2] - anchor[2:])
                / (2 * (frame_index - anchor_frame))
            )
        anchor, anchor_frame = next_anchor, frame_index
        result.append([{**selected, "id": 0, "source_track_id": int(selected["id"])}])
    return result
