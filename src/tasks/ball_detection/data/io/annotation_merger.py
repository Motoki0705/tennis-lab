"""Merge detector/refiner/event outputs into final pseudo-label annotation rows."""

from __future__ import annotations

from src.tasks.ball_detection.data.type import ConfidenceRecord, DetectionRecord, EventRecord, LabelRecord


def merge_annotation_records(
    *,
    file_names: list[str],
    detections: dict[int, DetectionRecord],
    refined_xy: dict[int, tuple[float, float]],
    confidence: dict[int, ConfidenceRecord],
    events: dict[int, EventRecord],
    confidence_threshold: float,
) -> list[LabelRecord]:
    """Build final pseudo labels.

    Priority:
    1) high-confidence detection -> visibility=1
    2) refined trajectory point -> visibility=2
    3) missing -> visibility=0

    `status` uses event hint:
    - 1 if shot probability is dominant
    - 2 if bounce probability is dominant
    - 0 otherwise
    """
    merged: list[LabelRecord] = []
    for idx, file_name in enumerate(file_names):
        evt = events.get(idx)
        status = 0
        if evt is not None:
            if evt.shot_prob >= evt.bounce_prob and evt.shot_prob >= 0.5:
                status = 1
            elif evt.bounce_prob > evt.shot_prob and evt.bounce_prob >= 0.5:
                status = 2

        conf = confidence.get(idx)
        det = detections.get(idx)
        if det is not None and conf is not None and conf.confidence >= confidence_threshold and det.visible:
            merged.append(
                LabelRecord(
                    file_name=file_name,
                    visibility=1,
                    x=det.x,
                    y=det.y,
                    status=status,
                    score=det.score,
                )
            )
            continue

        ref = refined_xy.get(idx)
        if ref is not None:
            merged.append(
                LabelRecord(
                    file_name=file_name,
                    visibility=2,
                    x=float(ref[0]),
                    y=float(ref[1]),
                    status=status,
                    score=0.0,
                )
            )
            continue

        merged.append(
            LabelRecord(
                file_name=file_name,
                visibility=0,
                x=0.0,
                y=0.0,
                status=status,
                score=0.0,
            )
        )

    return merged
