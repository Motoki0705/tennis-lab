"""Diagnostic variants and person/frame units for #964; no deployment defaults.

Agreement is against reviewed old COCO boxes, not independent detection GT.
All variants compete for the same units (duplicates and non-players included).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionResult
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, CameraLabels
from src.tasks.player_detection.evaluation.partial_labels import _unit_matches
from src.utils.geometry.bbox import pairwise_iou


def threshold(result: PersonDetectionResult, score: float) -> PersonDetectionResult:
    keep = result.scores >= score
    return PersonDetectionResult(result.boxes_xyxy[keep], result.scores[keep])


def merge_extra(primary: PersonDetectionResult, extra: PersonDetectionResult, *,
                max_height: float | None = None, dedup_iou: float = .5) -> PersonDetectionResult:
    """Preserve FT boxes; add score-ordered, nonduplicate extra boxes.

    Scores across COCO/FT are not calibrated, so extra boxes never replace FT.
    Caller uses the same spatial scope for both (full frame or one explicit ROI).
    Small means box
    height <= max_height in source pixels; zero-area boxes are rejected.
    """
    if not 0 < dedup_iou <= 1 or (max_height is not None and max_height <= 0):
        raise ValueError('Invalid union height/IoU')
    boxes, scores = list(primary.boxes_xyxy), list(primary.scores)
    for index in np.argsort(-extra.scores, kind='stable'):
        box = extra.boxes_xyxy[index]
        width, height = box[2:] - box[:2]
        if width <= 0 or height <= 0:
            raise ValueError('Degenerate detection in diagnostic union')
        if max_height is not None and height > max_height:
            continue
        if boxes and (pairwise_iou(box[None], np.asarray(boxes)) >= dedup_iou).any():
            continue
        boxes.append(box)
        scores.append(extra.scores[index])
    return PersonDetectionResult(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))


def far_tiles(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """Upper half of image, with 10% horizontal/vertical overlap at seams.

    This is an image-space far-region hypothesis, not court-coordinate side.
    Bottom y reaches 60% of image height to avoid cutting people at the seam.
    """
    return ((0, 0, int(np.ceil(width * .55)), int(np.ceil(height * .6))),
            (int(np.floor(width * .45)), 0, width, int(np.ceil(height * .6))))


def unit_rows(prediction: PersonDetectionResult, labels: CameraLabels, roles: NDArray[np.str_],
              frame: int) -> list[dict[str, Any]]:
    selection = labels.at(frame)
    people, old = labels.person_index[selection], labels.boxes_xyxy[selection]
    units = np.unique(people[people != AMBIGUOUS])
    overlap = pairwise_iou(prediction.boxes_xyxy, old)
    iou = np.column_stack([overlap[:, people == person].max(axis=1) for person in units]) \
        if len(units) else np.empty((len(prediction.scores), 0), np.float64)
    assigned = set(_unit_matches(iou, .5)[1].tolist())
    assigned03 = set(_unit_matches(iou, .3)[1].tolist())
    rows = []
    for unit, person in enumerate(units):
        candidates = old[people == person]
        box = candidates[np.prod(candidates[:, 2:] - candidates[:, :2], axis=1).argmax()]
        candidates03 = prediction.scores[iou[:, unit] >= .3]
        rows.append({'person': int(person), 'role': str(roles[person]), 'old_box': box.tolist(),
                     'near_far': 'unknown', 'matched': unit in assigned, 'matched_iou03': unit in assigned03,
                     'best_score_iou03': float(candidates03.max()) if len(candidates03) else None})
    players = [row for row in rows if row['role'] == 'player']
    if len(players) == 2 and players[0]['old_box'][3] != players[1]['old_box'][3]:
        midpoint = sum(row['old_box'][3] for row in players) / 2
        for row in rows:
            row['near_far'] = 'far' if row['old_box'][3] < midpoint else 'near'
    return rows


def add_counts(counts: dict[str, int], row: dict[str, Any], baseline: dict[str, Any]) -> None:
    role = row['role']
    for key, value in ((f'{role}_units', 1), (f'matched_{role}_units', int(row['matched'])),
                       (f'matched_{role}_units_iou03', int(row['matched_iou03'])),
                       (f'added_{role}_units', int(row['matched'] and not baseline['matched'])),
                       (f'lost_{role}_units', int(baseline['matched'] and not row['matched']))):
        counts[key] = counts.get(key, 0) + value


def rates(counts: dict[str, int]) -> dict[str, Any]:
    players, nonplayers = counts.get('player_units', 0), counts.get('non_player_units', 0)
    return {**counts,
            'old_box_agreement': counts.get('matched_player_units', 0) / players if players else None,
            'old_box_agreement_iou03': counts.get('matched_player_units_iou03', 0) / players if players else None,
            'non_player_hit_rate': counts.get('matched_non_player_units', 0) / nonplayers if nonplayers else None}
