"""Predeclared signals and deterministic sampling on the source frame grid."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import Field

from src.tasks.player_association.geometry.switches import (
    SwitchConfig,
    switch_candidates,
)
from src.tasks.player_association.pseudo_annotation.job import StrictRecord
from src.utils.geometry.triangulation import PointRejection

NEAR_REJECTED = "selection_rejected_near_court"
SINGLE_VIEW = "single_view_no_3d"
ASSOCIATION = "unassigned_or_reprojection"
TRACKING = "switch_or_fragment"
STOPPED = "stopped_clip"
CONFIDENT = "high_confidence_random"
DOUBT = (NEAR_REJECTED, SINGLE_VIEW, ASSOCIATION, TRACKING, STOPPED)
STRATA = (*DOUBT, CONFIDENT)


class SamplingConfig(StrictRecord):
    schema_version: Literal[1]
    max_abs_x_m: float = Field(gt=0)
    max_abs_y_m: float = Field(gt=0)
    footpoint_bottom_border_px: float = Field(ge=0)
    switch_window_s: float = Field(gt=0)
    switch_jump_m: float = Field(gt=0)
    switch_context_frames: int = Field(ge=0)
    fragment_max_duration_s: float = Field(gt=0)
    fragment_max_gap_s: float = Field(gt=0)
    reprojection_px: float = Field(gt=0)
    reprojection_min_joints: int = Field(ge=1, le=17)
    confident_min_views: int = Field(ge=2, le=4)
    confident_min_valid_joints: int = Field(ge=1, le=17)
    stopped_reasons: list[str] = Field(min_length=1)
    seed: int
    doubt_blocks_per_clip_stratum: int = Field(ge=1)
    doubt_block_frames: int = Field(ge=1)
    confident_frames_per_clip: int = Field(ge=1)

    @classmethod
    def load(cls, path: Path) -> SamplingConfig:
        config: SamplingConfig = cls.model_validate(yaml.safe_load(path.read_text()))
        return config


def near_court(
    points: NDArray[np.float64], valid: NDArray[np.bool_], cfg: SamplingConfig
) -> NDArray[np.bool_]:
    return (
        valid
        & (np.abs(points[..., 0]) <= cfg.max_abs_x_m)
        & (np.abs(points[..., 1]) <= cfg.max_abs_y_m)
    )


def tracking_suspicions(
    observed: NDArray[np.bool_],
    points: NDArray[np.float64],
    valid: NDArray[np.bool_],
    near: NDArray[np.bool_],
    fps: float,
    cfg: SamplingConfig,
) -> NDArray[np.bool_]:
    """Mark near-court real-observation breaks and windows around abrupt footpoint changes."""
    result = np.zeros(observed.shape[1], dtype=bool)
    config = SwitchConfig(cfg.switch_window_s, cfg.switch_jump_m)
    duration = max(1, round(fps * cfg.fragment_max_duration_s))
    gap = round(fps * cfg.fragment_max_gap_s)
    for row in np.flatnonzero(near.any(axis=1)):
        for cut in switch_candidates(points[row], valid[row], fps, config):
            begin = max(0, cut - cfg.switch_context_frames)
            end = min(len(result), cut + cfg.switch_context_frames + 1)
            if near[row, begin:end].any():
                result[begin:end] = True
        edges = np.flatnonzero(np.diff(np.r_[False, observed[row], False]))
        starts, ends = edges[::2], edges[1::2]
        for start, end in zip(starts, ends, strict=True):
            if (
                end - start <= duration
                and (start > 0 or end < len(result))
                and near[row, start:end].any()
            ):
                result[start:end] = True
        for end, start in zip(ends[:-1], starts[1:], strict=True):
            if start - end <= gap and near[row, end - 1] and near[row, start]:
                result[end - 1 : start + 1] = True
    return result


def reconstruction_suspicions(
    views: NDArray[np.int64],
    valid: NDArray[np.bool_],
    reasons: NDArray[np.uint8],
    reprojection: NDArray[np.floating],
    cfg: SamplingConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute one-view, large-residual and high-support flags per frame for four players."""
    if (
        views.ndim != 2
        or views.shape[0] != 4
        or valid.shape != (*views.shape, 17)
        or reasons.shape != valid.shape
    ):
        raise ValueError(
            "Expected triangulation on the same four-player/source-frame grid"
        )
    if (
        reprojection.ndim != 4
        or (reprojection.shape[0], *reprojection.shape[2:]) != valid.shape
    ):
        raise ValueError("Reprojection view axis disagrees with the triangulation grid")
    one = ((views == 1) & ~valid.any(-1)).any(0)
    bad = (reprojection > cfg.reprojection_px).any(1) | (
        reasons == int(PointRejection.REPROJECTION)
    )
    large_error = (bad.sum(-1) >= cfg.reprojection_min_joints).any(0)
    supported = (
        (views >= cfg.confident_min_views)
        & (valid.sum(-1) >= cfg.confident_min_valid_joints)
    ).all(0)
    return one, large_error, supported


def sample_strata(
    clip_id: str,
    masks: dict[str, NDArray[np.bool_]],
    detections: NDArray[np.bool_],
    cfg: SamplingConfig,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Select deterministic blocks from each doubt stratum and single clean controls."""
    if masks.keys() != set(STRATA) or any(
        mask.shape != detections.shape for mask in masks.values()
    ):
        raise ValueError("All strata must share the full source frame grid")
    doubt = np.logical_or.reduce([masks[name] for name in DOUBT])
    all_frames = {
        name: np.flatnonzero(
            masks[name] & detections & (~doubt if name == CONFIDENT else True)
        ).tolist()
        for name in STRATA
    }
    sampled = {}
    for name, frames in all_frames.items():
        width = 1 if name == CONFIDENT else cfg.doubt_block_frames
        count = (
            cfg.confident_frames_per_clip
            if name == CONFIDENT
            else cfg.doubt_blocks_per_clip_stratum
        )
        blocks = {frame // width for frame in frames}
        # Integer hash ranking stays fixed across processes, Python versions and input enumeration order.
        ordered = sorted(
            blocks,
            key=lambda block: (
                sha256(f"{cfg.seed}/{clip_id}/{name}/{block}".encode()).digest(),
                block,
            ),
        )
        chosen = set(ordered[:count])
        sampled[name] = [frame for frame in frames if frame // width in chosen]
    return all_frames, sampled
