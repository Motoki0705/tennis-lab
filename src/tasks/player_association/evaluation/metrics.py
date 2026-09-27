"""Score a cross-camera player association against box labels.

A prediction gives, per camera, tracks ``(D, T)`` of boxes and a per-frame
player ID (``-1`` = excluded). Each observed predicted box is matched to a
labelled box of the same camera and frame (Hungarian, IoU >= ``min_iou``).

Scoring works on *units*: a labelled player in one camera and frame (all its
duplicate boxes together) or one non-player box. A player unit carries the
set of non-negative IDs of its matched boxes. Only units that some predicted
box matched are scored (detection misses are reported as coverage, not as
association errors); ambiguous label boxes are ignored.

* pair F1: over pairs of units from different cameras in the same frame.
  Positive truth = same player; positive prediction = shared ID.
* group accuracy: fraction of frames whose units are all right: every
  non-player excluded, each player carries exactly one ID, one ID per player
  across cameras and distinct IDs for distinct players.
* exclusion: unit-level precision/recall of ``-1``. A non-player unit (one
  box) is a true exclusion when set to ``-1``; a player unit is a false
  exclusion only when none of its boxes carries an ID (a duplicate box set to
  ``-1`` is not an error).
* ID switches: a true switch is a change of the matched labelled person along
  a predicted track (between consecutive labelled frames); a predicted split is
  a change of the predicted ID along the track. They match one-to-one within
  ``switch_tolerance`` frames.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.utils.geometry.bbox import pairwise_iou


@dataclass(frozen=True)
class CameraPrediction:
    track_ids: NDArray[np.int64]  # (D,)
    boxes_xyxy: NDArray[np.float64]  # (D, T, 4)
    observed: NDArray[np.bool_]  # (D, T)
    player_ids: NDArray[np.int64]  # (D, T), -1 = excluded

    def __post_init__(self) -> None:
        tracks, frames = self.observed.shape
        if self.track_ids.shape != (tracks,) or self.boxes_xyxy.shape != (tracks, frames, 4) or self.player_ids.shape != (tracks, frames):
            raise ValueError("Prediction arrays must be (D,), (D, T, 4), (D, T), (D, T)")
        if (self.player_ids < -1).any():
            raise ValueError("Player IDs must be -1 or nonnegative")


def match_to_labels(labels: ClipLabels, predictions: dict[str, CameraPrediction], min_iou: float) -> dict[str, NDArray[np.int64]]:
    """Per camera ``(D, T)`` labelled person index of each observed predicted box.

    ``-2`` = unmatched (or unobserved), ``AMBIGUOUS`` = matched to an ambiguous label box.
    """
    if set(predictions) != set(labels.cameras):
        raise ValueError(f"Predicted cameras {sorted(predictions)} differ from labelled cameras {sorted(labels.cameras)}")
    matched = {}
    for camera, prediction in predictions.items():
        if prediction.observed.shape[1] != labels.num_frames:
            raise ValueError(f"{camera} prediction timeline differs from the labels")
        camera_labels = labels.cameras[camera]
        result = np.full(prediction.observed.shape, -2, np.int64)
        for frame in np.flatnonzero(prediction.observed.any(0)):
            rows = camera_labels.at(int(frame))
            if rows.start == rows.stop:
                continue
            tracks = np.flatnonzero(prediction.observed[:, frame])
            iou = pairwise_iou(prediction.boxes_xyxy[tracks, frame], camera_labels.boxes_xyxy[rows])
            for track, label in zip(*linear_sum_assignment(-iou), strict=True):
                if iou[track, label] >= min_iou:
                    result[tracks[track], frame] = camera_labels.person_index[rows][label]
        matched[camera] = result
    return matched


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _events(sequence: list[tuple[int, int]]) -> list[int]:
    """Frames where the value changes between consecutive entries of ``[(frame, value)]``."""
    return [frame for (_, before), (frame, after) in zip(sequence, sequence[1:], strict=False) if before != after]


def _match_events(truth: list[int], predicted: list[int], tolerance: int) -> int:
    free = sorted(predicted)
    hits = 0
    for frame in sorted(truth):
        near = [candidate for candidate in free if abs(candidate - frame) <= tolerance]
        if near:
            free.remove(min(near, key=lambda candidate: abs(candidate - frame)))
            hits += 1
    return hits


def evaluate(labels: ClipLabels, predictions: dict[str, CameraPrediction], *, min_iou: float = .5,
             switch_tolerance: int = 15) -> dict[str, Any]:
    """Association metrics of one clip; see the module docstring."""
    matched = match_to_labels(labels, predictions, min_iou)
    roles = labels.roles
    cameras = sorted(predictions)
    pair_tp = pair_fp = pair_fn = 0
    frames_scored = frames_correct = 0
    excl_tp = excl_fp = excl_fn = 0
    failures: list[dict[str, Any]] = []
    for frame in range(labels.num_frames):
        # unit key -> (camera, true person or None for a non-player box, predicted ID set)
        units: dict[tuple[Any, ...], tuple[str, int | None, set[int]]] = {}
        player_boxes: dict[tuple[str, int], list[int]] = defaultdict(list)
        for camera in cameras:
            ids = predictions[camera].player_ids[:, frame]
            for track in np.flatnonzero(matched[camera][:, frame] >= 0):
                person, predicted = int(matched[camera][track, frame]), int(ids[track])
                if roles[person] == "player":
                    player_boxes[(camera, person)].append(predicted)
                else:
                    units[(camera, "box", int(track))] = (camera, None, {predicted} - {-1})
                    excl_tp += predicted == -1
                    excl_fn += predicted != -1
        for (camera, person), predicted_ids in player_boxes.items():
            carried = set(predicted_ids) - {-1}
            units[(camera, "player", person)] = (camera, person, carried)
            excl_fp += not carried
        if not units:
            continue
        values = list(units.values())
        for (camera_a, person_a, ids_a), (camera_b, person_b, ids_b) in combinations(values, 2):
            if camera_a == camera_b:
                continue
            truth = person_a is not None and person_a == person_b
            linked = bool(ids_a & ids_b)
            pair_tp += truth and linked
            pair_fp += linked and not truth
            pair_fn += truth and not linked
        identity: dict[int, int] = {}
        correct = all(not ids for _, person, ids in values if person is None)
        for _, unit_person, ids in values:
            if unit_person is None:
                continue
            if len(ids) != 1 or identity.setdefault(unit_person, next(iter(ids))) != next(iter(ids)):
                correct = False
        correct = correct and len(set(identity.values())) == len(identity)
        frames_scored += 1
        frames_correct += correct
        if not correct:
            signature = sorted((camera, "non_player" if unit_person is None else labels.people[unit_person].person_id, tuple(sorted(ids)))
                               for camera, unit_person, ids in values)
            if failures and failures[-1]["end"] == frame and failures[-1]["signature"] == signature:
                failures[-1]["end"] = frame + 1
            else:
                failures.append({"start": frame, "end": frame + 1, "signature": signature})
    switches_true: list[dict[str, Any]] = []
    switches_predicted: list[dict[str, Any]] = []
    hits = 0
    confusion: dict[str, dict[str, dict[str, int]]] = {}
    coverage: dict[str, Any] = {}
    for camera in cameras:
        prediction, camera_match = predictions[camera], matched[camera]
        confusion[camera] = {}
        for row, track in enumerate(prediction.track_ids.tolist()):
            frames = np.flatnonzero(prediction.observed[row])
            labelled = [(int(f), int(camera_match[row, f])) for f in frames if camera_match[row, f] >= 0]
            predicted_sequence = [(int(f), int(prediction.player_ids[row, f])) for f in frames]
            true_events, predicted_events = _events(labelled), _events(predicted_sequence)
            hits += _match_events(true_events, predicted_events, switch_tolerance)
            switches_true += [{"camera": camera, "track_id": track, "frame": f} for f in true_events]
            switches_predicted += [{"camera": camera, "track_id": track, "frame": f} for f in predicted_events]
            counts = Counter((labels.people[p].person_id if p >= 0 else ("ambiguous" if p == AMBIGUOUS else "unmatched"),
                              int(prediction.player_ids[row, f])) for f in frames for p in [int(camera_match[row, f])])
            confusion[camera][str(track)] = {f"{person}->{player}": count for (person, player), count in sorted(counts.items())}
        label_boxes = len(labels.cameras[camera].frames)
        covered = int((camera_match >= 0).sum() + (camera_match == AMBIGUOUS).sum())
        coverage[camera] = {"label_boxes": label_boxes, "matched_label_boxes": covered,
                            "unmatched_predicted_boxes": int((prediction.observed & (camera_match == -2)).sum()),
                            "unmatched_predicted_boxes_with_id": int((prediction.observed & (camera_match == -2) & (prediction.player_ids >= 0)).sum())}
    return {"pairs": _f1(pair_tp, pair_fp, pair_fn),
            "group_accuracy": {"frames_scored": frames_scored, "frames_correct": frames_correct,
                               "accuracy": _ratio(frames_correct, frames_scored)},
            "exclusion": _f1(excl_tp, excl_fp, excl_fn),
            "id_switch": {"true": len(switches_true), "predicted": len(switches_predicted), "matched": hits,
                          "precision": _ratio(hits, len(switches_predicted)), "recall": _ratio(hits, len(switches_true)),
                          "tolerance_frames": switch_tolerance, "true_events": switches_true, "predicted_events": switches_predicted},
            "coverage": coverage, "track_confusion": confusion,
            "failed_frame_runs": [{"start": run["start"], "end": run["end"],
                                   "units": [{"camera": camera, "person": person, "predicted_ids": list(ids)} for camera, person, ids in run["signature"]]}
                                  for run in failures]}
