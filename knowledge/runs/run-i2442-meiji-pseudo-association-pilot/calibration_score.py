"""Score identical candidate boxes while aligning anonymous player IDs once per clip."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels


def agreement(reference: ClipLabels, prediction: ClipLabels) -> dict[str, Any]:
    frames = prediction.annotated_frames
    if (
        frames is None
        or reference.clip_id != prediction.clip_id
        or reference.num_frames != prediction.num_frames
        or set(reference.cameras) != set(prediction.cameras)
    ):
        raise ValueError("Calibration needs comparable source frames and cameras")
    pairs = []
    for camera, reference_rows in reference.cameras.items():
        predicted_rows = prediction.cameras[camera]
        for frame in frames:
            for_rows = []
            for labels in (reference_rows, predicted_rows):
                at = labels.at(frame)
                for_rows.append(
                    sorted(
                        zip(
                            labels.boxes_xyxy[at].tolist(),
                            labels.person_index[at].tolist(),
                            strict=True,
                        ),
                        key=lambda row: row[0],
                    )
                )
            left, right = for_rows
            if [box for box, _ in left] != [box for box, _ in right]:
                raise ValueError(
                    f"Compared boxes differ: {prediction.clip_id}/{camera}/{frame}"
                )
            pairs.extend(
                (camera, frame, a, b)
                for (_, a), (_, b) in zip(left, right, strict=True)
            )

    player_ref = [
        index
        for index, person in enumerate(reference.people)
        if person.role == "player"
    ]
    player_pred = [
        index
        for index, person in enumerate(prediction.people)
        if person.role == "player"
    ]
    ref_index = {p: i for i, p in enumerate(player_ref)}
    pred_index = {p: i for i, p in enumerate(player_pred)}
    counts = np.zeros((len(player_ref), len(player_pred)), dtype=np.int64)
    for _, _, truth, answer in pairs:
        if truth in ref_index and answer in pred_index:
            counts[ref_index[truth], pred_index[answer]] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping = {
        player_pred[col]: player_ref[row]
        for row, col in zip(rows, cols, strict=True)
        if counts[row, col] > 0
    }
    role_correct = box_correct = known = ambiguous_prediction = player_boxes = (
        player_boxes_correct
    ) = 0
    units: dict[tuple[str, int, int], list[bool]] = defaultdict(list)
    for camera, frame, truth, answer in pairs:
        if truth == AMBIGUOUS:
            continue
        known += 1
        ambiguous_prediction += answer == AMBIGUOUS
        if truth in ref_index:
            player_boxes += 1
            exact = mapping.get(answer) == truth
            role_correct += answer in pred_index
            player_boxes_correct += exact
            units[camera, frame, truth].append(exact)
        else:
            exact = (
                answer != AMBIGUOUS and prediction.people[answer].role == "non_player"
            )
            role_correct += exact
        box_correct += exact
    return {
        "frames": len(frames),
        "boxes": len(pairs),
        "known_boxes": known,
        "reference_ambiguous_boxes": len(pairs) - known,
        "predicted_ambiguous_on_known": ambiguous_prediction,
        "role_correct_boxes": role_correct,
        "identity_or_role_correct_boxes": box_correct,
        "player_boxes": player_boxes,
        "player_boxes_correct": player_boxes_correct,
        "player_units": len(units),
        "player_units_correct": sum(all(values) for values in units.values()),
        "player_mapping": {
            prediction.people[p].person_id: reference.people[t].person_id
            for p, t in mapping.items()
        },
    }
