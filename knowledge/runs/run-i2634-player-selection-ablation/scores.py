"""Score all conditions against the same partial 3cam references."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.utils.geometry.triangulation import PinholeCamera


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def prediction(raw: PersonTrackingOutput) -> CameraPrediction:
    return CameraPrediction(
        raw.track_ids,
        raw.boxes_xyxy.astype(np.float64),
        raw.observed,
        np.full(raw.observed.shape, -1, np.int64),
    )


def project_ids(
    raw: CameraPrediction,
    selection: PlayerSelectionOutput,
    ids: NDArray[np.int64],
) -> CameraPrediction:
    origins = selection.origin_rows
    if ids.shape != origins.shape or not np.array_equal(
        raw.track_ids, selection.raw_track_ids
    ):
        raise ValueError("The association IDs must follow the selected groups")
    projected = np.full(raw.observed.shape, -1, np.int64)
    groups, frames = np.nonzero((ids >= 0) & (origins >= 0))
    rows = origins[groups, frames]
    if not raw.observed[rows, frames].all() or len(
        np.unique(np.ravel_multi_index((rows, frames), raw.observed.shape))
    ) != len(rows):
        raise ValueError(
            "Each associated observation must address one distinct raw box"
        )
    projected[rows, frames] = ids[groups, frames]
    return replace(raw, player_ids=projected)


def box_counts(
    labels: ClipLabels,
    raw: dict[str, CameraPrediction],
    matches: dict[str, NDArray[np.int64]],
    selected: dict[str, PlayerSelectionOutput],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    player_ids = np.flatnonzero(labels.roles == "player")
    nonplayer_ids = np.flatnonzero(labels.roles == "non_player")
    for camera, reference in labels.cameras.items():
        matched = matches[camera]
        mask = selected[camera].selected
        if (mask & ~raw[camera].observed).any():
            raise ValueError("Only real raw observations may be selected")
        for name, indices in (("player", player_ids), ("nonplayer", nonplayer_ids)):
            known = np.isin(matched, indices)
            counts[f"{name}_reference"] += int(
                np.isin(reference.person_index, indices).sum()
            )
            counts[f"{name}_matched"] += int(known.sum())
            counts[f"{name}_kept"] += int((known & mask).sum())
            counts[f"{name}_rejected"] += int((known & ~mask).sum())
        counts["selected_ambiguous"] += int((mask & (matched == AMBIGUOUS)).sum())
        counts["selected_unmatched"] += int((mask & (matched == -2)).sum())
        counts["raw_observations"] += int(raw[camera].observed.sum())
        counts["selected_observations"] += int(mask.sum())
    return dict(counts)


def box_rates(counts: dict[str, int]) -> dict[str, float | None]:
    p, n = counts["player_kept"], counts["nonplayer_kept"]
    rejected_p, rejected_n = counts["player_rejected"], counts["nonplayer_rejected"]
    return {
        "player_precision": rate(p, p + n),
        "player_recall": rate(p, counts["player_reference"]),
        "nonplayer_precision": rate(rejected_n, rejected_n + rejected_p),
        "nonplayer_recall": rate(rejected_n, counts["nonplayer_reference"]),
        "nonplayer_recall_matched": rate(rejected_n, counts["nonplayer_matched"]),
    }


def tracking_scores(
    labels: ClipLabels,
    size: tuple[int, int],
    raw: dict[str, PersonTrackingOutput],
    cameras: tuple[PinholeCamera, ...],
    selected: dict[str, PlayerSelectionOutput],
    *,
    groups: bool,
) -> dict[str, int | float | None]:
    units = []
    for camera in cameras:
        choice = selected[camera.camera_id]
        view = choice.tracks if groups else raw[camera.camera_id]
        tracks = CameraTracks(
            camera, size, view.track_ids, view.boxes_xyxy, view.observed
        )
        rows, _ = tracking_units(
            tracks, view.observed if groups else choice.selected, labels
        )
        units.extend(rows)
    scores = score_units(units)
    return {
        key: value
        for key, value in scores.items()
        if key not in {"mapping", "switch_events", "fragment_events"}
    }


def sum_counts(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in keys}


def pool_tracking(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    counts = sum_counts(
        rows,
        (
            "idtp",
            "idfp",
            "idfn",
            "player_units",
            "player_units_kept",
            "player_identities",
            "player_identities_kept50",
            "nonplayer_units",
            "nonplayer_units_kept",
            "id_switches",
            "fragments",
        ),
    )
    tp, fp, fn = counts["idtp"], counts["idfp"], counts["idfn"]
    return {**counts, "idf1": rate(2 * tp, 2 * tp + fp + fn)}


def pool_pairs(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    counts = sum_counts(rows, ("tp", "fp", "fn"))
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
    }
