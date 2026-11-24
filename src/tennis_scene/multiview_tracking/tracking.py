"""Track frame pose candidates, interpolate short gaps and select players."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates
from src.tennis_scene.multiview_tracking.association import (
    Observations,
    associate,
    trajectory_motion,
)
from src.tennis_scene.multiview_tracking.fragments import (
    FragmentIntegration,
    FragmentJoin,
    PlayerSlot,
    describe_fragments,
    player_slots,
)
from src.tennis_scene.multiview_tracking.schema import (
    PoseTracks,
    SampleEvent,
    TrackEnd,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.selection import select_players


def _identities(predecessor: NDArray[np.int64]) -> NDArray[np.int64]:
    ids: NDArray[np.int64] = np.empty(len(predecessor), np.int64)
    next_id = 0
    for index, previous in enumerate(predecessor):
        if previous >= 0:
            ids[index] = ids[previous]
        else:
            ids[index] = next_id
            next_id += 1
    return ids


def track(
    candidates: FramePoseCandidates, *, fps: float, config: TrackingConfig
) -> PoseTracks:
    return track_with_diagnostics(candidates, fps=fps, config=config)[0]


def track_with_diagnostics(
    candidates: FramePoseCandidates, *, fps: float, config: TrackingConfig
) -> tuple[PoseTracks, FragmentIntegration | None]:
    if not isfinite(fps) or fps <= 0:
        raise ValueError("Tracking requires a finite positive source FPS")
    observations = Observations.from_candidates(candidates)
    links = associate(candidates, observations, fps, config)
    motion = trajectory_motion(observations, links, fps, config)
    frames = observations.frames
    num_frames = len(candidates.frame_offsets) - 1
    ids = _identities(links.predecessor)
    selection = select_players(
        frames,
        ids,
        observations.centres,
        num_frames=num_frames,
        fps=fps,
        players_per_side=config.players_per_side,
    )
    predecessor = links.predecessor.copy()
    link_margin = links.margin.copy()
    link_ambiguous = (link_margin >= 0) & (link_margin <= config.ambiguity_margin)
    integration = None
    if (settings := config.fragment_integration) is not None:
        fragments = describe_fragments(
            candidates, observations, motion, ids, selection, fps, settings
        )
        assignments = player_slots(fragments, selection, fps, settings)
        joined = []
        for slot in assignments:
            joins = []
            for link, margin in slot.links:
                before, after = fragments[link.before], fragments[link.after]
                gap = after.start.frame - before.end.frame - 1
                ambiguous = margin <= settings.ambiguity_margin
                predecessor[after.first] = before.last
                link_margin[after.first] = margin
                link_ambiguous[after.first] = ambiguous
                joins.append(
                    FragmentJoin(
                        before.index,
                        after.index,
                        before.end.frame,
                        after.start.frame,
                        gap,
                        gap / fps > config.max_occlusion_seconds,
                        1.0 - link.cost,
                        margin,
                        ambiguous,
                        link.position_residual_m,
                        link.velocity_difference_mps,
                        link.log_bone_difference,
                        link.shared_bones,
                    )
                )
            joined.append((slot, tuple(joins)))
        ids = _identities(predecessor)
        slots = tuple(
            PlayerSlot(
                fragments[slot.core].side,
                slot.core,
                int(ids[fragments[slot.core].first]),
                slot.fragments,
                joins,
            )
            for slot, joins in joined
        )
        integration = FragmentIntegration(len(fragments), slots)
        selection = select_players(
            frames,
            ids,
            observations.centres,
            num_frames=num_frames,
            fps=fps,
            players_per_side=config.players_per_side,
            player_candidates={slot.track_id for slot in slots},
        )
    count = len(ids)
    track_count = len(selection.decisions)
    ends: NDArray[np.int64] = np.zeros(track_count, np.int64)
    np.maximum.at(ends, ids, frames + 1)
    end_reasons = np.where(
        num_frames - ends >= links.max_link_delta,
        TrackEnd.DISAPPEARED,
        TrackEnd.CLIP_BOUNDARY,
    ).astype(np.uint8)

    reappeared = np.flatnonzero((predecessor >= 0) & (frames - frames[predecessor] > 1))
    previous = predecessor[reappeared]
    gap_size = frames[reappeared] - frames[previous] - 1
    gap_steps = (
        np.arange(gap_size.sum())
        - np.repeat(np.cumsum(gap_size) - gap_size, gap_size)
        + 1
    )
    gap_start: NDArray[np.int64] = np.repeat(previous, gap_size)
    gap_end: NDArray[np.int64] = np.repeat(reappeared, gap_size)
    gap_frames = frames[gap_start] + gap_steps
    span = frames[gap_end] - frames[gap_start]
    weight = gap_steps / span
    predicted = candidates.joint_valid[gap_start] & candidates.joint_valid[gap_end]
    position_prediction = (
        candidates.positions_m[gap_start] * (1 - weight[:, None, None])
        + candidates.positions_m[gap_end] * weight[:, None, None]
    )
    position_prediction[~predicted] = 0
    centre_prediction = (
        observations.centres[gap_start] * (1 - weight[:, None])
        + observations.centres[gap_end] * weight[:, None]
    )
    velocity_prediction = (
        observations.centres[gap_end] - observations.centres[gap_start]
    ) * (fps / span[:, None])
    prediction_motion_valid = (
        np.linalg.norm(velocity_prediction, axis=1) <= config.max_speed_mps
    )
    velocity_prediction[~prediction_motion_valid] = 0

    event = np.where(predecessor < 0, SampleEvent.BORN, SampleEvent.OBSERVED)
    event[reappeared] = SampleEvent.REAPPEARED
    all_frames = np.concatenate((frames, gap_frames))
    all_ids = np.concatenate((ids, ids[gap_start]))
    order = np.lexsort((all_ids, all_frames))
    gap_count = len(gap_frames)
    margins = np.concatenate((link_margin, link_margin[gap_end])).astype(np.float32)
    ambiguous_samples = np.concatenate((link_ambiguous, link_ambiguous[gap_end]))

    poses = PoseTracks(
        camera_ids=candidates.camera_ids,
        frame_offsets=np.r_[
            0, np.cumsum(np.bincount(all_frames, minlength=num_frames))
        ],
        track_ids=all_ids[order],
        candidate_indices=np.r_[np.arange(count), np.full(gap_count, -1, np.int64)][
            order
        ],
        source_rows=np.concatenate(
            (
                candidates.source_rows,
                np.full((gap_count, len(candidates.camera_ids)), -1, np.int64),
            )
        )[order],
        positions_m=np.concatenate(
            (candidates.positions_m, position_prediction.astype(np.float32))
        )[order],
        measured_joints=np.concatenate(
            (candidates.joint_valid, np.zeros_like(predicted))
        )[order],
        predicted_joints=np.concatenate(
            (np.zeros_like(candidates.joint_valid), predicted)
        )[order],
        torso_m=np.concatenate((observations.centres, centre_prediction))[order].astype(
            np.float32
        ),
        velocity_mps=np.concatenate((motion.velocity, velocity_prediction))[
            order
        ].astype(np.float32),
        velocity_valid=np.r_[motion.valid, prediction_motion_valid][order],
        event=np.r_[event, np.full(gap_count, SampleEvent.OCCLUDED)][order].astype(
            np.uint8
        ),
        frame_competing=np.r_[
            candidates.competing_assignment, np.zeros(gap_count, bool)
        ][order],
        frame_gap=np.r_[candidates.relative_score_gap, np.zeros(gap_count, np.float32)][
            order
        ],
        frame_ambiguous=np.r_[candidates.ambiguous, np.zeros(gap_count, bool)][order],
        temporal_margin=margins[order],
        id_ambiguous=ambiguous_samples[order],
        end_frames=ends,
        end_reasons=end_reasons,
        selection=selection,
    )
    return poses, integration
