"""Read the pinned geometry inputs and report extraction coverage across Meiji V4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.analytics.geometry import (
    BIN_VERSION,
    DEFINITION_VERSION,
    DEPTH_BIN_EDGES_M,
    DEPTH_BINS,
    HEIGHT_BIN_WIDTH_M,
    LANES,
    BallTrajectory,
    GeometryEvent,
    GeometryResult,
)
from src.tennis_scene.ball_event_labels import (
    EVENT_CLASSES_V4,
    EVENT_STATUS,
    EVENT_SUBTYPES_V4,
)
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.utils.inference_benchmark_inputs import read_pinned_json

MEIJI_V4_MANIFEST_SHA256 = (
    "8e012944b1407d696a1ab1d325ae3c6b60cf26f071752510ea2fc688b031893c"
)
SPLIT_COUNTS = {"train": 47, "val": 11, "test": 11}


@dataclass(frozen=True, slots=True)
class V4GeometryClip:
    clip_id: str
    split: str
    scene_npz_sha256: str
    annotation_sha256: str
    component_sha256: str
    trajectory: BallTrajectory
    events: tuple[GeometryEvent, ...]


def _pinned_clip(root: Path, row: dict[str, Any], split: str) -> V4GeometryClip:
    clip_id = row["clip_id"]
    index = row["scene"]
    index_path = (root / index["path"]).resolve()
    if not index_path.is_relative_to(root) or index_path.name != "scene.json":
        raise ValueError(f"Invalid V4 scene index path for {clip_id}")
    document = read_pinned_json(index_path, index["sha256"])
    annotation = row["ball_events"]
    reference = annotation["component"]
    archive = row["exports"]["scene/scene"]
    export = document["exports"]["scene"]
    indexed_archive = export["files"]["scene"]
    if (
        document["schema"] != "tennis_scene_index_v1"
        or document["source"]["clip_id"] != clip_id
        or document["artifacts"]["event_detection"] != reference
        or export["inputs"]["event_detection"] != reference
        or indexed_archive["sha256"] != archive["sha256"]
        or (index_path.parent / indexed_archive["path"]).resolve()
        != (root / archive["path"]).resolve()
        or reference["schema"] != "scene_result"
        or reference["version"] != 4
    ):
        raise ValueError(
            f"V4 scene index does not agree with the manifest for {clip_id}"
        )
    descriptor = read_component_descriptor(
        index_path.parent,
        reference,
        node="event_detection",
        source_sha256=index["source_sha256"],
    )
    component_dir = (index_path.parent / reference["path"]).resolve().parent
    payload = descriptor["payload"]
    if (
        payload["metadata"]["scene_schema_version"] != 4
        or payload["metadata"]["dataset_clip_id"] != clip_id
        or payload["metadata"]["players_per_side"] != 2
        or payload["metadata"]["ball_events"]["source"] != "annotation"
        or payload["metadata"]["ball_events"]["provenance"]["annotation_sha256"]
        != annotation["annotation_sha256"]
        or (payload["num_frames"], payload["fps"]) != (row["frames"], row["fps"])
    ):
        raise ValueError(f"V4 geometry timeline or provenance mismatch for {clip_id}")

    columns = {
        "ball_3d": np.float32,
        "ball_3d_valid": np.bool_,
        "ball_event_coverage": np.bool_,
        "ball_event_frame": np.int32,
        "ball_event_kind": np.uint8,
        "ball_event_subtype": np.uint8,
        "ball_event_status": np.uint8,
        "ball_event_frame_uncertainty": np.uint8,
        "ball_event_side": np.int8,
    }
    arrays: dict[str, Any] = unpack_value(
        {name: payload[name] for name in columns}, component_dir, descriptor["arrays"]
    )
    evidence = payload["event_evidence"]
    events, excluded = evidence["events"], evidence["excluded_events"]
    for name, dtype in columns.items():
        value = arrays[name]
        shape: tuple[int, ...]
        if name == "ball_3d":
            shape = (row["frames"], 3)
        elif name in ("ball_3d_valid", "ball_event_coverage"):
            shape = (row["frames"],)
        else:
            shape = (len(events),)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
        ):
            raise ValueError(f"V4 geometry array mismatch for {clip_id}: {name}")

    if (
        len(events) != annotation["events"]
        or len(excluded) != annotation["excluded_events"]
        or int(arrays["ball_3d_valid"].sum()) != row["ball"]["valid_3d_frames"]
    ):
        raise ValueError(
            f"V4 geometry event or valid-ball count disagrees with the manifest for {clip_id}"
        )

    decoded = []
    for offset, event in enumerate(events):
        kind, subtype, status = event["kind"], event["subtype"], event["status"]
        if (
            int(arrays["ball_event_frame"][offset]) != event["frame"]
            or int(arrays["ball_event_kind"][offset]) != EVENT_CLASSES_V4.index(kind)
            or int(arrays["ball_event_subtype"][offset])
            != EVENT_SUBTYPES_V4.index(subtype)
            or int(arrays["ball_event_status"][offset]) != EVENT_STATUS.index(status)
            or int(arrays["ball_event_frame_uncertainty"][offset])
            != event["frame_uncertainty"]
            or type(event["old_in_point"]) is not bool
        ):
            raise ValueError(
                f"V4 event arrays disagree with annotation evidence for {clip_id}"
            )
        decoded.append(
            GeometryEvent(
                event["source_event_id"],
                event["frame"],
                kind,
                subtype,
                float(event["frame_uncertainty"]),
                int(arrays["ball_event_side"][offset]),
                status,
                in_point=event["old_in_point"],
            )
        )
    decoded.extend(
        GeometryEvent(
            event["source_event_id"],
            event["frame"],
            event["old_kind"],
            frame_uncertainty=None,
            in_point=False,
            excluded_reason=event["reason"],
        )
        for event in excluded
    )
    return V4GeometryClip(
        clip_id,
        split,
        archive["sha256"],
        annotation["annotation_sha256"],
        reference["sha256"],
        BallTrajectory(
            arrays["ball_3d"],
            arrays["ball_3d_valid"],
            float(row["fps"]),
            arrays["ball_event_coverage"],
        ),
        tuple(sorted(decoded, key=lambda event: (event.frame, event.source_event_id))),
    )


def read_frozen_v4(
    root: Path,
    manifest_sha256: str = MEIJI_V4_MANIFEST_SHA256,
) -> tuple[V4GeometryClip, ...]:
    """Verify manifest and selected component members, then read only S4 inputs."""
    root = root.resolve(strict=True)
    document = read_pinned_json(root / "manifest.json", manifest_sha256)
    if (
        document["schema"] != "tennis_multiview_benchmark_v1"
        or document["benchmark_id"] != "meiji_4cam/multiview-v4"
        or document["scene_schema_version"] != 4
    ):
        raise ValueError("Geometry audit requires the annotated Meiji V4 release")
    rows = document["clips"]
    split_by_clip: dict[str, str] = {}
    for split, count in SPLIT_COUNTS.items():
        ids = document["splits"][split]["clip_ids"]
        if (
            len(ids) != count
            or len(set(ids)) != count
            or split_by_clip.keys() & set(ids)
        ):
            raise ValueError("Require the fixed, disjoint 47/11/11 V4 split")
        split_by_clip.update(dict.fromkeys(ids, split))
    if (
        len(rows) != document["totals"]["clips"]
        or len(rows) != len(split_by_clip)
        or {row["clip_id"] for row in rows} != split_by_clip.keys()
    ):
        raise ValueError("V4 manifest clip list and split assignments disagree")

    clips = []
    for row in rows:
        try:
            clips.append(_pinned_clip(root, row, split_by_clip[row["clip_id"]]))
        except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot load V4 geometry for {row['clip_id']}: {error}"
            ) from error
    return tuple(clips)


def _bin_counts(
    results: Sequence[GeometryResult],
    height_ids: set[str],
) -> dict[str, dict[str, int]]:
    axes: dict[str, Counter[str]] = {
        "contact_height": Counter(dict.fromkeys(height_ids, 0)),
        "landing_depth": Counter(dict.fromkeys(DEPTH_BINS, 0)),
        "landing_lane": Counter(dict.fromkeys(LANES, 0)),
        "landing_course": Counter({"crosscourt": 0, "down_the_line": 0}),
    }
    for counts in axes.values():
        counts["unassigned"] = 0
    for result in results:
        for shot in result.shots:
            for axis, counts in axes.items():
                counts[shot.bins.get(axis, "unassigned")] += 1
    return {axis: dict(counts) for axis, counts in axes.items()}


def _availability(results: Sequence[GeometryResult]) -> dict[str, Any]:
    count = sum(len(result.shots) for result in results)
    reasons: dict[str, Counter[str]] = {
        field: Counter()
        for field in (
            "contact_height",
            "analysis_contact_position",
            "incoming_speed",
            "outgoing_speed",
            "outgoing_heading",
            "net_crossing",
            "primary_net_crossing",
            "net_center_height",
            "net_clearance_center",
            "net_clearance_bottom",
            "time_to_net",
            "first_bounce",
            "landing_depth",
            "landing_depth_bin",
            "landing_lane",
            "landing_course",
            "line_margin",
            "landing_result",
        )
    }
    shot_statuses: Counter[str] = Counter()
    arc_statuses: Counter[str] = Counter()
    landing_sources: Counter[str] = Counter()
    crossing_directions: Counter[int] = Counter()
    outside_span = 0
    for result in results:
        by_arc = {arc.arc_index: arc.shot_event for arc in result.arcs}
        statuses_by_shot: dict[str, set[str]] = {}
        primary_shots: set[str] = set()
        for crossing in result.crossings:
            arc_statuses[crossing.status] += 1
            shot_event = by_arc[crossing.arc_index]
            if shot_event is not None:
                statuses_by_shot.setdefault(shot_event.source_event_id, set()).add(
                    crossing.status
                )
                if crossing.is_primary:
                    primary_shots.add(shot_event.source_event_id)
            if crossing.direction_y is not None:
                crossing_directions[crossing.direction_y] += 1
                outside_span += crossing.crossing_in_net_span is False

        for shot in result.shots:
            key = shot.shot_event.source_event_id
            found = statuses_by_shot.get(key, set())
            status = next(
                (
                    s
                    for s in ("crossed", "unknown_gap", "no_crossing_confirmed")
                    if s in found
                ),
                "not_applicable",
            )
            shot_statuses[status] += 1
            primary = key in primary_shots
            landing = shot.landing
            if landing.position is not None:
                landing_sources[landing.position.source_kind] += 1
            else:
                landing_sources["missing"] += 1
            landing_reason = (
                landing.missing_reason
                or shot.quality.get("landing_context_reason")
                or "boundary_ambiguous"
            )
            contact_reason = (
                "tracking_gap" if not shot.contact_position else "hitter_side_unknown"
            )
            if shot.contact_height_m is None:
                reasons["contact_height"][contact_reason] += 1
            if shot.analysis_contact_xyz is None:
                reasons["analysis_contact_position"][contact_reason] += 1
            if shot.incoming is None:
                reason = (
                    "insufficient_contiguous_samples"
                    if shot.quality["incoming_speed_applicable"]
                    else "not_applicable"
                )
                reasons["incoming_speed"][reason] += 1
            if shot.outgoing is None:
                reasons["outgoing_speed"]["insufficient_contiguous_samples"] += 1
            if shot.outgoing_heading_rad is None:
                reason = "insufficient_contiguous_samples"
                if shot.outgoing is not None:
                    reason = (
                        "hitter_side_unknown"
                        if shot.shot_event.court_side == 0
                        else "no_horizontal_motion"
                    )
                reasons["outgoing_heading"][reason] += 1

            if status != "crossed":
                for field in (
                    "net_crossing",
                    "net_center_height",
                    "net_clearance_center",
                    "net_clearance_bottom",
                ):
                    reasons[field][status] += 1
            if not primary:
                for field in ("primary_net_crossing", "time_to_net"):
                    reasons[field][
                        "no_opponent_directed_crossing"
                        if status == "crossed"
                        else status
                    ] += 1
            for field, value in (
                ("first_bounce", landing.position),
                ("landing_depth", landing.depth_m),
                ("landing_depth_bin", landing.depth_bin),
                ("landing_lane", landing.lane),
                ("landing_course", landing.course),
                ("line_margin", landing.line_margin_m),
                (
                    "landing_result",
                    None
                    if landing.result in ("uncertain", "not_applicable")
                    else landing.result,
                ),
            ):
                if value is None:
                    bin_field = {
                        "landing_depth_bin": "landing_depth",
                        "landing_lane": "landing_lane",
                    }.get(field)
                    bin_reason = (
                        shot.quality.get("landing_bin_evidence", {})
                        .get(bin_field, {})
                        .get("held_reason")
                        if bin_field is not None
                        else None
                    )
                    reasons[field][bin_reason or landing_reason] += 1

    available = {}
    for field, missing in reasons.items():
        valid = count - missing.total()
        available[field] = {
            "available": valid,
            "shots": count,
            "rate": valid / count if count else None,
            "missing_reasons": dict(missing),
        }
    return {
        "clips": len(results),
        "shots": count,
        "availability": available,
        "shot_crossing_statuses": dict(shot_statuses),
        "arc_crossing_row_statuses": dict(arc_statuses),
        "crossing_directions": {
            str(key): value for key, value in crossing_directions.items()
        },
        "crossings_outside_net_span": outside_span,
        "landing_sources": dict(landing_sources),
    }


def audit_geometry(
    clips: Sequence[V4GeometryClip],
    results: Mapping[str, GeometryResult],
    *,
    manifest_sha256: str,
    ingest_run_id: int | None,
) -> dict[str, Any]:
    by_split = {
        split: [results[clip.clip_id] for clip in clips if clip.split == split]
        for split in SPLIT_COUNTS
    }
    height_ids = {
        height_id
        for split in ("train", "val")
        for result in by_split[split]
        for shot in result.shots
        for height_id in shot.quality["contact_height_bin_candidates"]
    }
    bins = {
        split: _bin_counts(by_split[split], height_ids) for split in ("train", "val")
    }
    profiles = {
        result.profile.profile_key: result.profile.to_record()
        for result in results.values()
    }
    return {
        "benchmark_id": "meiji_4cam/multiview-v4",
        "manifest_sha256": manifest_sha256,
        "ingest_run_id": ingest_run_id,
        "definition_version": DEFINITION_VERSION,
        "bin_version": BIN_VERSION,
        "bin_settings": {
            "contact_height_width_m": HEIGHT_BIN_WIDTH_M,
            "landing_depth_edges_m": list(DEPTH_BIN_EDGES_M),
            "landing_depth_ids": list(DEPTH_BINS),
            "landing_lane_ids": list(LANES),
            "landing_lane_edges": [-1 / 3, 1 / 3],
            "landing_lane_edge_unit": "fraction_of_half_court_width",
        },
        "profile": next(iter(profiles.values())) if len(profiles) == 1 else None,
        "profiles": profiles,
        "overall": _availability(list(results.values())),
        "splits": {split: _availability(group) for split, group in by_split.items()},
        "train_val_bins": bins,
        "clips": {
            clip.clip_id: {
                "split": clip.split,
                "component_sha256": clip.component_sha256,
                "annotation_sha256": clip.annotation_sha256,
                "valid_ball_frames": int(clip.trajectory.valid.sum()),
                "profile_key": results[clip.clip_id].profile.profile_key,
                "excluded_contact_boundaries": sum(
                    e.excluded_reason is not None for e in clip.events
                ),
                "net_contact_subtypes": dict(
                    Counter(
                        e.subtype
                        for e in clip.events
                        if e.kind == "net" and e.subtype is not None
                    )
                ),
                **_availability([results[clip.clip_id]]),
            }
            for clip in clips
        },
    }
