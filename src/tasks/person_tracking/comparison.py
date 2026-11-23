"""Metrics for comparing tracking routes on the same partially labelled boxes."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import (
    PSEUDO_PROVENANCE,
    CameraLabels,
    ClipLabels,
)
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    match_to_labels,
)


@dataclass(frozen=True)
class TrackingPrediction:
    """One source clip. Global player IDs are on the real raw-box timeline."""

    raw: Mapping[str, CameraTracks]
    selected: Mapping[str, np.ndarray]
    groups: Mapping[str, CameraTracks]
    player_ids: Mapping[str, np.ndarray]  # camera -> (D,T), -1 means unassigned
    valid_3d: Mapping[int, np.ndarray]  # global ID -> (T,), at least one valid 3D joint
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.raw or any(self.raw.keys() != x.keys() for x in (self.selected, self.groups, self.player_ids)):
            raise ValueError("Tracking predictions require the same nonempty camera set at every stage")
        frames = next(iter(self.raw.values())).observed.shape[1]
        for camera, raw in self.raw.items():
            group, selected, ids = self.groups[camera], self.selected[camera], self.player_ids[camera]
            if (raw.camera.camera_id != camera or group.camera.camera_id != camera
                    or raw.observed.shape[1] != frames or group.observed.shape[1] != frames
                    or selected.dtype != np.bool_ or selected.shape != raw.observed.shape
                    or (selected & ~raw.observed).any() or ids.dtype != np.int64
                    or ids.shape != selected.shape or (ids < -1).any() or ((ids >= 0) & ~selected).any()):
                raise ValueError(f"{camera}: selection and player IDs must align to real raw observations")
        if any(identity < 0 or mask.shape != (frames,) or mask.dtype != np.bool_
               for identity, mask in self.valid_3d.items()):
            raise ValueError("3D validity requires a boolean source-frame timeline per global ID")


def _label_subset(labels: ClipLabels, frames: Collection[int]) -> ClipLabels:
    indices = sorted(set(frames))
    cover = labels.annotated_frames
    if (not indices or indices[0] < 0 or indices[-1] >= labels.num_frames
            or (cover is not None and not set(indices).issubset(cover))):
        raise ValueError("The scoring stratum must be a nonempty subset of the labelled timeline")
    cameras = {}
    for name, camera in labels.cameras.items():
        keep = np.isin(camera.frames, indices)
        cameras[name] = CameraLabels(camera.frames[keep], camera.person_index[keep], camera.boxes_xyxy[keep])
    return replace(labels, cameras=cameras, provenance={
        **labels.provenance, PSEUDO_PROVENANCE: {
            **labels.provenance.get(PSEUDO_PROVENANCE, {}), "schema_version": 1, "annotated_frames": indices
        },
    })


def _pairs_and_coverage(
    labels: ClipLabels, prediction: TrackingPrediction, frames: Collection[int]
) -> tuple[Counter[str], Counter[str], int, int, int]:
    mask = np.zeros(labels.num_frames, dtype=bool)
    mask[list(frames)] = True
    observations = {
        name: CameraPrediction(raw.track_ids, raw.boxes_xyxy, raw.observed & mask, prediction.player_ids[name])
        for name, raw in prediction.raw.items()
    }
    matched = match_to_labels(labels, observations, .5)
    pairs: Counter[str] = Counter()
    conditional: Counter[str] = Counter()
    covered_3d = reference_3d = matched_players = 0
    for frame in sorted(frames):
        # (camera, player, -1) groups known duplicate player boxes. Matched
        # non-player boxes get independent keys and cannot make a positive pair.
        units: dict[tuple[str, int, int], set[int]] = {}
        seen: set[tuple[str, int, int]] = set()
        player_support: dict[int, Counter[int]] = defaultdict(Counter)
        for name, reference in labels.cameras.items():
            for person in set(reference.person_index[reference.at(frame)].tolist()):
                if person >= 0 and labels.people[person].role == "player":
                    units[name, person, -1] = set()
            for row in np.flatnonzero(matched[name][:, frame] >= 0):
                person = int(matched[name][row, frame])
                key = (name, person, -1) if labels.people[person].role == "player" else (name, -1, int(row))
                carried = units.setdefault(key, set())
                seen.add(key)
                identity = int(prediction.player_ids[name][row, frame])
                if identity >= 0:
                    carried.add(identity)
        for key, identities in units.items():
            person = key[1]
            if person >= 0:
                matched_players += key in seen
                votes = player_support[person]
                for identity in identities:
                    valid = prediction.valid_3d.get(identity)
                    if valid is not None and valid[frame]:
                        votes[identity] += 1
        for a, b in combinations(units, 2):
            if a[0] == b[0]:
                continue
            truth, linked = a[1] >= 0 and a[1] == b[1], bool(units[a] & units[b])
            if truth or linked:
                result = "tp" if truth and linked else "fn" if truth else "fp"
                pairs[result] += 1
                if prediction.stop_reason is None and a in seen and b in seen:
                    conditional[result] += 1
        reference_3d += len(player_support)
        supported_ids = sorted({identity for votes in player_support.values() for identity in votes})
        support = np.asarray([[votes[identity] for identity in supported_ids]
                              for _, votes in sorted(player_support.items())], dtype=np.int64)
        if support.size:
            # Maximize covered players, then supporting views. A merged ID
            # can account for at most one labelled player in the same frame.
            reward = np.where(support > 0, support + len(labels.cameras) * min(support.shape) + 1, 0)
            covered_3d += int(np.count_nonzero(support[linear_sum_assignment(-reward)]))
    return pairs, conditional, covered_3d, reference_3d, matched_players


def _local_counts(labels: ClipLabels, tracks: Mapping[str, CameraTracks],
                  masks: Mapping[str, np.ndarray]) -> dict[str, int]:
    units = []
    unknown = 0
    for name, camera in tracks.items():
        local, unmatched = tracking_units(camera, masks[name], labels)
        units.extend(local)
        unknown += unmatched
    scored = score_units(units)
    return {
        **{key: scored[key] for key in ("idtp", "idfp", "idfn", "id_switches", "fragments",
                                         "player_units", "player_units_kept", "nonplayer_units_kept")},
        "unmatched_selected_boxes": unknown,
    }


def score_clip(
    labels: ClipLabels, prediction: TrackingPrediction, strata: Mapping[str, Collection[int]]
) -> dict[str, dict[str, Any]]:
    """Score each stratum on its own continuous spans and retain additive counts."""
    if labels.cameras.keys() != prediction.raw.keys():
        raise ValueError("Predictions and labels must name the same cameras")
    result = {}
    pair_keys = ("tp", "fp", "fn")
    for name, frames in strata.items():
        if not frames:
            continue
        subset = _label_subset(labels, frames)
        scored_frames = subset.annotated_frames
        assert scored_frames is not None
        pairs, conditional, covered, reference, matched = _pairs_and_coverage(subset, prediction, scored_frames)
        result[name] = {
            "frames": len(scored_frames), "clips": 1, "stopped_clips": int(prediction.stop_reason is not None),
            "raw": _local_counts(subset, prediction.raw, prediction.selected),
            "group": _local_counts(subset, prediction.groups, {k: t.observed for k, t in prediction.groups.items()}),
            "pairs": {k: pairs[k] for k in pair_keys}, "matched_decided_pairs": {k: conditional[k] for k in pair_keys},
            "matched_player_camera_units": matched, "valid_player_frames": covered, "labelled_player_frames": reference,
        }
    return result


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def pool_scores(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pool integer counts across clips before computing ratios."""
    rows = list(rows)
    boxes = ("frames", "clips", "stopped_clips", "matched_player_camera_units",
             "valid_player_frames", "labelled_player_frames")
    result: dict[str, Any] = {key: sum(row[key] for row in rows) for key in boxes}
    for stage in ("raw", "group"):
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(row[stage])
        result[stage] = dict(counts, idf1=_ratio(2 * counts["idtp"], 2 * counts["idtp"] + counts["idfp"] + counts["idfn"]))
    kept, nonplayers = (result["raw"].get(k, 0) for k in ("player_units_kept", "nonplayer_units_kept"))
    result["selection_precision"] = _ratio(kept, kept + nonplayers)
    result["selection_recall"] = _ratio(kept, result["raw"].get("player_units", 0))
    for name in ("pairs", "matched_decided_pairs"):
        tp, fp, fn = (sum(row[name][k] for row in rows) for k in ("tp", "fp", "fn"))
        result[name] = dict(tp=tp, fp=fp, fn=fn, f1=_ratio(2 * tp, 2 * tp + fp + fn))
    result["coverage_3d"] = _ratio(result["valid_player_frames"], result["labelled_player_frames"])
    return result
