"""Original-frame event pairings and test-set summaries for motion ablations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.inference.segmentation import (
    DEFAULT_PICKING,
    pick_class_event_frames,
)
from src.tasks.ball_refiner_3d.post_training.labels import (
    MATCH_TOLERANCE,
    VOLLEY_SUBTYPES,
    Annotations,
)
from src.tasks.ball_refiner_3d.post_training.motion import MotionFeatures
from src.utils.physics.ball.events import match_event_pairs

CONDITIONS = (
    "ball_missing",
    "smpl_missing",
    "all_smpl_missing",
    "near_net",
    "volley",
)


@dataclass(frozen=True, slots=True)
class Decision:
    kind: str
    outcome: str
    prediction_frame: int | None
    probability: float | None
    label: dict[str, Any] | None
    signed_offset_frames: int | None
    absolute_offset_ms: float | None
    in_point: bool
    conditions: tuple[str, ...]


def score_clip(
    probability: NDArray[np.float32],
    annotations: Annotations,
    fps: float,
    motion: MotionFeatures | None = None,
) -> list[Decision]:
    """Assign per clip and class before attributing TP/FN by label and FP by frame."""
    if probability.shape != (len(annotations.coverage), len(EVENT_CLASSES)):
        raise ValueError("Probabilities must cover the original annotation frame grid")
    predictions = pick_class_event_frames(probability, DEFAULT_PICKING)
    ignored = annotations.ignored_near_uncertain()
    point_frames = annotations.point_frames()
    decisions: list[Decision] = []
    for channel, (kind, proposed) in enumerate(predictions.items()):
        labels = {e.frame: e for e in annotations.events if e.kind == kind}
        truth = {
            frame: label
            for frame, label in labels.items()
            if label.status != "uncertain"
        }
        # Explicit labels can resolve peaks even within unlabelable intervals.
        pairs = match_event_pairs(
            proposed, np.asarray(list(truth), dtype=np.int64), MATCH_TOLERANCE
        )
        matched = {int(pred): int(label) for pred, label in pairs}
        counted_labels = set(matched.values())
        for pred_np in proposed:
            pred = int(pred_np)
            label_frame = matched.get(pred)
            label = truth[label_frame] if label_frame is not None else None
            if label is not None:
                outcome = "tp"
            elif not annotations.coverage[pred]:
                outcome = "outside_coverage"
            elif ignored[pred, channel]:
                outcome = "near_uncertain"
            else:
                outcome = "fp"
            frame = pred if label_frame is None else label_frame
            offset = None if label_frame is None else pred - label_frame
            decisions.append(
                Decision(
                    kind,
                    outcome,
                    pred,
                    float(probability[pred, channel]),
                    None if label is None else asdict(label),
                    offset,
                    None if offset is None else abs(offset) * 1_000 / fps,
                    bool(point_frames[pred]) if label is None else label.in_point,
                    _conditions(
                        motion, frame, None if label is None else label.subtype
                    ),
                )
            )
        for frame, event in labels.items():
            if frame in counted_labels:
                continue
            outcome = "uncertain_label" if event.status == "uncertain" else "fn"
            decisions.append(
                Decision(
                    kind,
                    outcome,
                    None,
                    None,
                    asdict(event),
                    None,
                    None,
                    event.in_point,
                    _conditions(motion, frame, event.subtype),
                )
            )
    return decisions


def _conditions(
    motion: MotionFeatures | None,
    frame: int,
    subtype: str | None,
) -> tuple[str, ...]:
    tags = ["volley"] if subtype in VOLLEY_SUBTYPES else []
    if motion is not None:
        present = np.asarray([identity is not None for identity in motion.track_ids])
        valid = motion.smpl_valid[frame, present]
        if not valid.all():
            tags.append("smpl_missing")
        if not valid.any():
            tags.append("all_smpl_missing")
        if not motion.ball_valid[frame]:
            tags.append("ball_missing")
        elif abs(float(motion.ball_m[frame, 1])) <= 1.0:
            tags.append("near_net")
    return tuple(tags)


def summary(
    decisions: Iterable[Decision], *, subtype_only: bool = False
) -> dict[str, Any]:
    records = list(decisions)
    by_class = {}
    for kind in EVENT_CLASSES:
        rows = [d for d in records if d.kind == kind]
        tp = sum(d.outcome == "tp" for d in rows)
        fp = sum(d.outcome == "fp" for d in rows)
        fn = sum(d.outcome == "fn" for d in rows)
        values = [
            d.absolute_offset_ms for d in rows if d.absolute_offset_ms is not None
        ]
        by_class[kind] = {
            "tp": tp,
            "fp": None if subtype_only else fp,
            "fn": fn,
            "events": tp + fn,
            "precision": tp / (tp + fp)
            if tp + fn and tp + fp and not subtype_only
            else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn and not subtype_only else None,
            "time_mae_ms": float(np.mean(values)) if values else None,
            "excluded_predictions": sum(
                d.outcome in ("outside_coverage", "near_uncertain") for d in rows
            ),
            "excluded_labels": sum(d.outcome == "uncertain_label" for d in rows),
        }
    f1 = [m["f1"] for m in by_class.values() if m["f1"] is not None]
    return {"by_class": by_class, "macro_f1": float(np.mean(f1)) if f1 else None}


def report(decisions: list[Decision]) -> dict[str, Any]:
    return {
        "all": summary(decisions),
        "in_point": summary(d for d in decisions if d.in_point),
        "conditions": {
            name: summary(
                (d for d in decisions if name in d.conditions),
                subtype_only=name == "volley",
            )
            for name in CONDITIONS
        },
    }
