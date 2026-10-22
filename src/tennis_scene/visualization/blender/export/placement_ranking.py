"""Rank placement windows across every identity in an adopted component store.

``mode=rank`` is a read-only selection step for the Blender placement export.  It
re-runs the production placement trace on the CPU, enumerates windows inside each
GVHMR segment, filters them by the ranking rules, and writes only
``ranking.json``.  It never extracts a plate, writes a bundle, or decides which
camera the final export must use.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.tennis_scene.pipeline.components.player_reconstruction import (
    BodyGeometryModel,
    PlacementTrace,
    trace_player_placement,
)
from src.tennis_scene.visualization.blender.export.placement import (
    PlacementInputs,
    _cpu_body_model,
    _git_commit,
    _json_mapping,
    _range_indices,
    _select_identity,
    _validate_plate_video,
    _validate_reproduction,
    _validate_root_validity,
    read_placement_inputs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

RANKING_SCHEMA = "tennis_scene_blender_placement_ranking_v1"
IDENTITY_QUOTA = 3
EXCLUSION_NO_T0 = "no_t0"
EXCLUSION_LOW_VALID_RATIO = "low_valid_ratio"


@dataclass(frozen=True, slots=True)
class PlacementRankConfig:
    """Ranking choices; every value is explicit in ``blender_placement.yaml``."""

    window_seconds: float = 14.0
    stride_seconds: float = 1.0
    min_valid_ratio: float = 0.9
    top_k: int = 10

    def __post_init__(self) -> None:
        for name in ("window_seconds", "stride_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"rank.{name} must be finite and positive")
        ratio = self.min_valid_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0.0 <= ratio <= 1.0
        ):
            raise ValueError("rank.min_valid_ratio must be finite and in [0, 1]")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("rank.top_k must be a positive integer")


@dataclass(frozen=True, slots=True)
class PlacementRankCandidate:
    """One evaluated window and the values rank exposes to the user."""

    identity: int
    segment_id: int
    gvhmr_camera_id: str
    start: int
    stop: int
    t0: int
    t0_valid_joints: int
    valid_ratio: float
    yaw_variation_deg: float
    yaw_correction_range_deg: float
    scale: float
    shortened: bool
    # Filled in by select_top_candidates for the candidates written to
    # ranking.json; values on a freshly evaluated window are placeholders.
    overall_rank: int = 0
    selected_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "segment_id": self.segment_id,
            "gvhmr_camera_id": self.gvhmr_camera_id,
            "start": self.start,
            "stop": self.stop,
            "t0": self.t0,
            "t0_valid_joints": self.t0_valid_joints,
            "valid_ratio": self.valid_ratio,
            "yaw_variation_deg": self.yaw_variation_deg,
            "yaw_correction_range_deg": self.yaw_correction_range_deg,
            "scale": self.scale,
            "shortened": self.shortened,
            "overall_rank": self.overall_rank,
            "selected_by": list(self.selected_by),
        }


@dataclass(frozen=True, slots=True)
class RankTraceResult:
    candidates: tuple[PlacementRankCandidate, ...]
    windows: int
    shortened_windows: int
    exclusions: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _WindowPlan:
    window_frames: int
    stride_frames: int


def _rounded_frames(seconds: float, fps: float, *, name: str) -> int:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"store source fps must be finite and positive, got {fps}.")
    frames = int(round(seconds * fps))
    if frames < 1:
        raise ValueError(
            f"rank.{name}={seconds!r} rounds to {frames} frames at {fps} fps; "
            "window and stride must be at least one frame."
        )
    return frames


def _window_plan(config: PlacementRankConfig, fps: float) -> _WindowPlan:
    return _WindowPlan(
        _rounded_frames(config.window_seconds, fps, name="window_seconds"),
        _rounded_frames(config.stride_seconds, fps, name="stride_seconds"),
    )


def _segment_windows(
    trace: PlacementTrace, plan: _WindowPlan
) -> Iterator[tuple[int, int, bool]]:
    """Yield ``(start, stop, shortened)`` inside one GVHMR segment, stop exclusive."""
    for segment_id in np.unique(trace.segment_ids):
        selected = trace.segment_ids == segment_id
        frames = np.asarray(trace.frames[selected], np.int64)
        if not len(frames):
            continue
        start = int(frames[0])
        stop = int(frames[-1]) + 1
        expected: NDArray[np.int64] = np.arange(start, stop, dtype=np.int64)
        if not np.array_equal(frames, expected):
            raise ValueError(
                f"GVHMR segment {int(segment_id)} has non-contiguous trace frames; "
                "windows cannot be enumerated without crossing a gap."
            )
        length = stop - start
        if length <= plan.window_frames:
            # A segment shorter than one window is itself the only window.
            yield start, stop, length < plan.window_frames
            continue
        for window_start in range(
            start, stop - plan.window_frames + 1, plan.stride_frames
        ):
            yield window_start, window_start + plan.window_frames, False


def _select_t0(
    trace: PlacementTrace,
    positions: NDArray[np.int64],
    *,
    start: int,
    stop: int,
) -> tuple[int, int] | None:
    """Pick the window's representative frame by the agreed tie-break order."""
    valid = np.asarray(trace.valid[positions], bool)
    target = np.asarray(trace.target_joints[positions], np.float64)
    finite = np.isfinite(target).all(-1)
    counts = finite.sum(-1)
    eligible = valid & (counts >= 15)
    if not eligible.any():
        return None
    weights = np.asarray(trace.target_weights[positions], np.float64)
    weight_sums = np.where(finite, weights, 0.0).sum(-1)
    source_frames = np.asarray(trace.frames[positions], np.float64)
    center = (start + stop - 1) / 2.0
    distance = np.abs(source_frames - center)
    candidates = np.flatnonzero(eligible)
    order = np.lexsort(
        (
            source_frames[candidates],
            distance[candidates],
            -weight_sums[candidates],
            -counts[candidates],
        )
    )
    best = int(candidates[int(order[0])])
    return int(positions[best]), int(counts[best])


def _heading_degrees(
    joints_local: NDArray[Any], yaw_correction: NDArray[Any]
) -> NDArray[np.float64]:
    """Rotate the left/right hip vector by ``psi`` and take its XY heading."""
    joints = np.asarray(joints_local, np.float64)
    difference = joints[:, 11, :2] - joints[:, 12, :2]
    angle = np.asarray(yaw_correction, np.float64)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    x = cosine * difference[:, 0] - sine * difference[:, 1]
    y = sine * difference[:, 0] + cosine * difference[:, 1]
    return np.asarray(np.degrees(np.arctan2(y, x)), np.float64)


def _yaw_variation_deg(
    heading: NDArray[Any], valid: NDArray[Any]
) -> float:
    """Sum absolute unwrapped heading changes inside contiguous valid runs."""
    values = np.asarray(heading, np.float64)
    mask = np.asarray(valid, bool)
    total = 0.0
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < len(mask) and mask[end]:
            end += 1
        run = values[index:end]
        if not np.isfinite(run).all():
            raise ValueError(
                "A valid placement frame has a non-finite hip heading; "
                "yaw variation cannot be computed."
            )
        total += float(np.abs(np.diff(np.unwrap(np.radians(run)))).sum())
        index = end
    return float(np.degrees(total))


def _yaw_correction_range_deg(
    yaw_correction: NDArray[Any], valid: NDArray[Any]
) -> float:
    values = np.asarray(yaw_correction, np.float64)[np.asarray(valid, bool)]
    if not len(values):
        return 0.0
    if not np.isfinite(values).all():
        raise ValueError(
            "A valid placement frame has a non-finite yaw correction; "
            "its range cannot be computed."
        )
    return float(np.degrees(np.ptp(values)))


def _evaluate_window(
    trace: PlacementTrace,
    positions: NDArray[np.int64],
    *,
    start: int,
    stop: int,
    shortened: bool,
    config: PlacementRankConfig,
) -> tuple[PlacementRankCandidate | None, str | None]:
    valid = np.asarray(trace.valid[positions], bool)
    valid_ratio = float(valid.mean()) if len(valid) else 0.0
    selected = _select_t0(trace, positions, start=start, stop=stop)
    if selected is None:
        return None, EXCLUSION_NO_T0
    if valid_ratio < config.min_valid_ratio:
        return None, EXCLUSION_LOW_VALID_RATIO
    scale = trace.scale
    if scale is None:
        raise ValueError(
            f"Identity {trace.identity} has valid frames but no fitted scale; "
            "a ranked candidate cannot be exported."
        )
    joints = np.asarray(trace.joints_local[positions], np.float64)
    bad_hips = valid & ~np.isfinite(joints[:, [11, 12], :]).all(axis=(1, 2))
    if bad_hips.any():
        frames = np.asarray(trace.frames[positions], np.int64)[bad_hips]
        raise ValueError(
            f"Identity {trace.identity} has non-finite hip joints on valid frames "
            f"{frames.tolist()}; yaw variation cannot be computed."
        )
    heading = _heading_degrees(joints, trace.yaw_correction[positions])
    yaw_variation = _yaw_variation_deg(heading, valid)
    yaw_range = _yaw_correction_range_deg(trace.yaw_correction[positions], valid)
    segment_ids = np.unique(trace.segment_ids[positions])
    if len(segment_ids) != 1:
        raise ValueError(
            f"Window {start}:{stop} crosses GVHMR segments: "
            f"{sorted(int(value) for value in segment_ids)}."
        )
    segment_id = int(segment_ids[0])
    if not 0 <= segment_id < len(trace.segment_camera_ids):
        raise ValueError(
            f"GVHMR segment {segment_id} has no recorded source camera."
        )
    t0_index, t0_joints = selected
    return (
        PlacementRankCandidate(
            identity=int(trace.identity),
            segment_id=segment_id,
            gvhmr_camera_id=trace.segment_camera_ids[segment_id],
            start=start,
            stop=stop,
            t0=int(trace.frames[t0_index]),
            t0_valid_joints=t0_joints,
            valid_ratio=valid_ratio,
            yaw_variation_deg=yaw_variation,
            yaw_correction_range_deg=yaw_range,
            scale=float(scale),
            shortened=shortened,
        ),
        None,
    )


def rank_trace_candidates(
    trace: PlacementTrace,
    *,
    config: PlacementRankConfig,
    fps: float,
) -> RankTraceResult:
    """Enumerate and evaluate every window of one identity's placement trace."""
    plan = _window_plan(config, fps)
    candidates: list[PlacementRankCandidate] = []
    exclusions = {EXCLUSION_NO_T0: 0, EXCLUSION_LOW_VALID_RATIO: 0}
    windows = 0
    shortened = 0
    for start, stop, is_shortened in _segment_windows(trace, plan):
        windows += 1
        shortened += int(is_shortened)
        positions = _range_indices(trace, (start, stop))
        candidate, exclusion = _evaluate_window(
            trace,
            positions,
            start=start,
            stop=stop,
            shortened=is_shortened,
            config=config,
        )
        if candidate is None:
            if exclusion is None:
                raise AssertionError("Excluded window must carry an exclusion reason.")
            exclusions[exclusion] += 1
        else:
            candidates.append(candidate)
    return RankTraceResult(
        tuple(candidates), windows, shortened, dict(exclusions)
    )


def _sort_key(candidate: PlacementRankCandidate) -> tuple[float, float, int, int, int]:
    # The final two keys only make exact ties deterministic; they are not rank
    # criteria and cannot reorder candidates that the agreed keys separate.
    return (
        -candidate.yaw_variation_deg,
        -candidate.valid_ratio,
        candidate.start,
        candidate.identity,
        candidate.segment_id,
    )


def sort_candidates(
    candidates: Sequence[PlacementRankCandidate],
) -> tuple[PlacementRankCandidate, ...]:
    return tuple(sorted(candidates, key=_sort_key))


def select_top_candidates(
    candidates: Sequence[PlacementRankCandidate],
    *,
    identities: Sequence[int],
    top_k: int,
) -> tuple[tuple[PlacementRankCandidate, ...], dict[str, Any]]:
    """Select the union of the global ``top_k`` and each identity's top three.

    The union may exceed ``top_k``.  Selected candidates are kept in global rank
    order and annotated with their 1-based ``overall_rank`` and the reasons they
    were selected.  Duplicate input windows are rejected: they indicate a window
    enumeration bug upstream.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    identity_values = tuple(int(identity) for identity in identities)
    if len(set(identity_values)) != len(identity_values):
        raise ValueError("identities must be unique.")
    unknown = sorted(
        {
            int(candidate.identity)
            for candidate in candidates
            if int(candidate.identity) not in identity_values
        }
    )
    if unknown:
        raise ValueError(f"Candidates reference unknown identities: {unknown}.")
    seen_windows: set[tuple[int, int, int, int]] = set()
    duplicate_windows: set[tuple[int, int, int, int]] = set()
    for candidate in candidates:
        window = (
            int(candidate.identity),
            int(candidate.segment_id),
            int(candidate.start),
            int(candidate.stop),
        )
        if window in seen_windows:
            duplicate_windows.add(window)
        seen_windows.add(window)
    if duplicate_windows:
        duplicates = ", ".join(
            f"identity={identity} segment_id={segment_id} "
            f"start={start} stop={stop}"
            for identity, segment_id, start, stop in sorted(duplicate_windows)
        )
        raise ValueError(
            "Duplicate rank candidates for the same window: "
            f"{duplicates}. Window enumeration must return each window once."
        )
    ordered = sort_candidates(candidates)
    positions_by_identity: dict[int, list[int]] = {
        identity: [] for identity in identity_values
    }
    for index, candidate in enumerate(ordered):
        positions_by_identity[int(candidate.identity)].append(index)
    overall_indices = set(range(min(top_k, len(ordered))))
    identity_quota_indices: set[int] = set()
    for identity in identity_values:
        identity_quota_indices.update(
            positions_by_identity[identity][:IDENTITY_QUOTA]
        )
    selected_indices = sorted(overall_indices | identity_quota_indices)
    selected_list: list[PlacementRankCandidate] = []
    for index in selected_indices:
        selected_by: list[str] = []
        if index in overall_indices:
            selected_by.append("overall")
        if index in identity_quota_indices:
            selected_by.append("identity_quota")
        selected_list.append(
            replace(
                ordered[index],
                overall_rank=index + 1,
                selected_by=tuple(selected_by),
            )
        )
    selected = tuple(selected_list)
    selected_counts = {
        identity: sum(
            1 for candidate in selected if candidate.identity == identity
        )
        for identity in identity_values
    }
    document = {
        "top_k": top_k,
        "identity_quota": IDENTITY_QUOTA,
        "candidate_count": len(ordered),
        "selected_count": len(selected),
        "selected_by_overall": len(overall_indices),
        "added_by_identity_quota": len(selected_indices) - len(overall_indices),
        "capacity_shortage": max(0, top_k - len(ordered)),
        "identity_shortages": [
            {
                "identity": identity,
                "available": len(positions_by_identity[identity]),
                "required": IDENTITY_QUOTA,
                "missing": max(
                    0,
                    IDENTITY_QUOTA - len(positions_by_identity[identity]),
                ),
            }
            for identity in identity_values
            if len(positions_by_identity[identity]) < IDENTITY_QUOTA
        ],
        "identities": [
            {
                "identity": identity,
                "candidates": len(positions_by_identity[identity]),
                "selected": selected_counts[identity],
            }
            for identity in identity_values
        ],
    }
    return selected, document


def _identity_ids(inputs: PlacementInputs) -> tuple[int, ...]:
    values = tuple(int(value) for value in np.asarray(inputs.grouped.identities).tolist())
    if len(set(values)) != len(values):
        raise ValueError("Store grouped identities are not unique.")
    return values


def _verify_identity(
    inputs: PlacementInputs,
    trace: PlacementTrace,
    player: int,
    identity: int,
) -> dict[str, Any]:
    """Check the saved artifact exactly as the export path does."""
    _validate_root_validity(inputs.placed, trace, player, identity)
    saved_smpl_valid = np.asarray(inputs.placed.smpl_valid[player], bool)
    if trace.scale is None:
        smpl_valid_frames = int(saved_smpl_valid.sum())
        if smpl_valid_frames:
            valid_frames = np.flatnonzero(saved_smpl_valid).astype(np.int64)
            raise ValueError(
                f"identity={identity} has no fitted scale, but the saved "
                f"body_placement artifact has {smpl_valid_frames} valid SMPL "
                "frame(s) that the current code cannot reproduce: "
                f"source_frames={valid_frames.tolist()}."
            )
        return {
            "status": "no_scale",
            "root_valid": "ok",
            "saved_placement_check": "skipped_no_scale",
            "smpl_valid_frames": 0,
            "max_abs_difference_m": None,
        }
    compared_smpl_valid = (
        saved_smpl_valid[trace.frames]
        if len(trace.frames)
        else np.zeros(0, bool)
    )
    smpl_valid_frames = int(compared_smpl_valid.sum())
    _, max_difference = _validate_reproduction(
        inputs,
        inputs.placed,
        trace,
        player,
        float(trace.scale),
        identity,
    )
    return {
        "status": "ok",
        "root_valid": "ok",
        "saved_placement_check": "ok",
        "smpl_valid_frames": smpl_valid_frames,
        "max_abs_difference_m": float(max_difference),
    }


def rank_placement_candidates(
    inputs: PlacementInputs,
    *,
    body: BodyGeometryModel,
    config: PlacementRankConfig,
    plate_camera_id: str,
    plate_video_path: Path,
    output_path: Path,
    resolved_config: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
) -> Path:
    """Rank every identity and write the single new ``ranking.json`` output."""
    if not plate_camera_id.strip():
        raise ValueError("plate_camera_id must be non-empty.")
    plan = _window_plan(config, inputs.source.fps)
    plate_video = _validate_plate_video(
        inputs,
        plate_camera_id=plate_camera_id,
        plate_video_path=plate_video_path,
    )
    identities = _identity_ids(inputs)
    candidates: list[PlacementRankCandidate] = []
    exclusions = {EXCLUSION_NO_T0: 0, EXCLUSION_LOW_VALID_RATIO: 0}
    per_identity: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    windows = 0
    shortened = 0
    for identity in identities:
        player, recovered_body = _select_identity(inputs, identity)
        trace = trace_player_placement(
            recovered_body,
            inputs.skeleton,
            inputs.grouped,
            inputs.cameras,
            player=player,
            body=body,
            fps=inputs.source.fps,
            placement_config=inputs.placement_config,
        )
        verification[str(identity)] = _verify_identity(
            inputs, trace, player, identity
        )
        result = rank_trace_candidates(
            trace, config=config, fps=inputs.source.fps
        )
        windows += result.windows
        shortened += result.shortened_windows
        for name, count in result.exclusions.items():
            exclusions[name] += count
        candidates.extend(result.candidates)
        per_identity[str(identity)] = {
            "candidates": len(result.candidates),
            "windows": result.windows,
            "shortened_windows": result.shortened_windows,
            "exclusions": dict(result.exclusions),
            "scale": None if trace.scale is None else float(trace.scale),
        }
    selected, quota = select_top_candidates(
        candidates, identities=identities, top_k=config.top_k
    )
    document = {
        "schema": RANKING_SCHEMA,
        "store": dict(inputs.store_provenance),
        "artifacts": _json_mapping(inputs.artifact_provenance),
        "plate": {
            "camera_id": plate_camera_id,
            "video": str(plate_video),
            "video_sha256": inputs.source.video(plate_camera_id).sha256,
        },
        "config": {
            "resolved": _json_mapping(resolved_config),
            "rank": {
                "window_seconds": float(config.window_seconds),
                "stride_seconds": float(config.stride_seconds),
                "min_valid_ratio": float(config.min_valid_ratio),
                "top_k": config.top_k,
            },
        },
        "parameters": {
            "fps": float(inputs.source.fps),
            "window_frames": plan.window_frames,
            "stride_frames": plan.stride_frames,
        },
        "counts": {
            "identities": len(identities),
            "windows": windows,
            "shortened_windows": shortened,
            "candidates": len(candidates),
            "selected": len(selected),
        },
        "exclusions": {**exclusions, "per_identity": per_identity},
        "quota": quota,
        "verification": verification,
        "candidates": [candidate.to_dict() for candidate in selected],
        "git_commit": git_commit if git_commit is not None else _git_commit(),
        "implementation": {
            "trace": "src.tennis_scene.pipeline.components.player_reconstruction.trace_player_placement",
            "geometry": "src.tennis_scene.pipeline.components.body_placement.SmplGeometry",
            "world": "src.tennis_scene.visualization.rendering.smpl_court.smpl_vertices_to_court",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Ranking output already exists: {output_path}")
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
    except FileExistsError:
        raise
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def rank_blender_placement(
    *,
    store_dir: Path,
    output_path: Path,
    config: PlacementRankConfig,
    plate_camera_id: str,
    plate_video_path: Path,
    human_vis_threshold: float,
    resolved_config: Mapping[str, Any] | None = None,
    body: BodyGeometryModel | None = None,
) -> Path:
    """Read ``store_dir``, reconstruct on CPU and write ``ranking.json``."""
    inputs = read_placement_inputs(
        store_dir, human_vis_threshold=human_vis_threshold
    )
    owned = body is None
    if body is None:
        body = _cpu_body_model(inputs)
    try:
        return rank_placement_candidates(
            inputs,
            body=body,
            config=config,
            plate_camera_id=plate_camera_id,
            plate_video_path=plate_video_path,
            output_path=output_path,
            resolved_config=resolved_config,
        )
    finally:
        if owned:
            body.unload()


__all__ = [
    "EXCLUSION_LOW_VALID_RATIO",
    "EXCLUSION_NO_T0",
    "IDENTITY_QUOTA",
    "PlacementRankCandidate",
    "PlacementRankConfig",
    "RANKING_SCHEMA",
    "RankTraceResult",
    "rank_blender_placement",
    "rank_placement_candidates",
    "rank_trace_candidates",
    "select_top_candidates",
    "sort_candidates",
]
