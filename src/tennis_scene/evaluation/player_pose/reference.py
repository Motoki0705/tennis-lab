"""Native-pixel COCO17 references from placed SMPL and their evidence masks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, require_smpl_asset
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.rendering.smpl_court import smpl_vertices_to_court
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.player import COCO_KP_IDX, NUM_HUMAN_KP


@dataclass(frozen=True)
class SmplRegressors:
    root: NDArray[np.float32]  # (1,V), SMPL joint 0
    coco17: NDArray[np.float32]  # (17,V), COCO_KP_NAMES order


def load_regressors(directory: Path) -> SmplRegressors:
    """Read the byte-pinned GVHMR auxiliary assets without running inference."""
    import torch

    values = []
    for name, rows in (
        ("smpl_neutral_joint_regressor", 24),
        ("smpl_coco17_regressor", NUM_HUMAN_KP),
    ):
        asset = SMPL_ASSET_SOURCES[name]
        path = directory / asset.checkpoint_path.name
        require_smpl_asset(path, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset.sha256:
            raise ValueError(
                f"Regressor SHA-256 mismatch for {path}; expected {asset.sha256}"
            )
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (rows, 6890):
            raise ValueError(f"Expected a ({rows}, 6890) regressor in {path}")
        value = np.asarray(tensor.float().numpy(), np.float32)
        if not np.isfinite(value).all() or not np.allclose(value.sum(-1), 1, atol=1e-4):
            raise ValueError(f"Invalid regressor weights in {path}")
        values.append(value)
    return SmplRegressors(values[0][:1], values[1])


def placed_smpl_world(
    vertices: NDArray[np.float32],
    orient: NDArray[np.float32],
    position: NDArray[np.float32],
    yaw: NDArray[np.float32],
    regressors: SmplRegressors,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert a batch of saved meshes and COCO17 joints to court metres."""
    relative = smpl_vertices_to_court(
        vertices[:, None],
        orient[:, None],
        np.zeros_like(position[:, None]),
        yaw[:, None],
        regressors.root,
    )[:, 0]
    # Several asset rows sum to 0.99998. Apply world translation after regression.
    translation = position[:, None]
    return relative + translation, regressors.coco17 @ relative + translation


def scene_cameras(scene: SceneResult) -> tuple[PinholeCamera, ...]:
    """Bind the fitted pinhole parameters to the scene observation axis."""
    reference = scene.metadata["court_reference"]
    camera_ids = scene.metadata["camera_ids"]
    if reference["camera_ids"] != camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Camera fits and scene observation camera axis disagree")
    return tuple(
        PinholeCamera(
            camera_id,
            np.asarray(fit["K"], np.float64),
            np.asarray(fit["R"], np.float64),
            np.asarray(fit["t"], np.float64),
        )
        for camera_id, fit in zip(camera_ids, reference["camera_fits"], strict=True)
    )


def joint_boxes(
    uv: NDArray[np.floating],
    valid: NDArray[np.bool_],
    size: tuple[int, int],
    *,
    margin: float,
    min_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return clipped xyxy envelopes with fractional expansion per side."""
    if (
        uv.shape != valid.shape + (2,)
        or valid.shape[-1:] != (NUM_HUMAN_KP,)
        or min(size) <= 0
        or not np.isfinite(margin)
        or margin < 0
        or not 2 <= min_joints <= NUM_HUMAN_KP
    ):
        raise ValueError("Invalid coordinates, image size, joint count or box margin")
    valid = valid & np.isfinite(uv).all(-1)
    enough = valid.sum(-1) >= min_joints
    lower = np.where(valid[..., None], uv, np.inf).min(-2)
    upper = np.where(valid[..., None], uv, -np.inf).max(-2)
    extent = np.where(enough[..., None], upper - lower, 0.0)
    left = np.clip(lower - margin * extent, 0, size)
    right = np.clip(upper + margin * extent, 0, size)
    keep = enough & (extent > 0).all(-1) & (right > left).all(-1)
    return np.where(keep[..., None], np.concatenate((left, right), -1), np.nan), keep


@dataclass(frozen=True)
class Geometry:
    joints_m: NDArray[np.float32]  # (P,T,J,3)
    pixels: NDArray[np.float64]  # (P,C,T,J,2), NaN without positive depth
    front: NDArray[np.bool_]
    inside: NDArray[np.bool_]
    depth_m: NDArray[np.float64]
    mesh_boxes: NDArray[np.float64]  # (P,C,T,4), clipped xyxy or NaN


def project_placed_smpl(
    scene: SceneResult,
    cameras: tuple[PinholeCamera, ...],
    inferred: NDArray[np.bool_],
    regressors: SmplRegressors,
) -> Geometry:
    """Project only saved inference samples with accepted placed bodies."""
    vertices = scene.smpl_vertices_local
    orient = scene.smpl_global_orient
    valid = scene.player_smpl_valid
    if (
        vertices is None
        or orient is None
        or valid is None
        or inferred.shape != valid.shape
    ):
        raise ValueError("Scene is missing placed SMPL or matching inference masks")
    p, t = valid.shape
    shape = (p, len(cameras), t, NUM_HUMAN_KP)
    joints_m = np.full((p, t, NUM_HUMAN_KP, 3), np.nan, np.float32)
    pixels = np.full((*shape, 2), np.nan)
    front = np.zeros(shape, bool)
    depth_m = np.full(shape, np.nan)
    mesh_boxes = np.full((*shape[:-1], 4), np.nan)
    size = (scene.width, scene.height)
    players, frames = np.nonzero(valid & inferred)
    for start in range(0, len(frames), 64):
        row = (players[start : start + 64], frames[start : start + 64])
        world, joints = placed_smpl_world(
            vertices[row],
            orient[row],
            scene.player_position[row],
            scene.player_yaw[row],
            regressors,
        )
        joints_m[row] = joints
        for view, camera in enumerate(cameras):
            index = (row[0], view, row[1])
            uv, positive = camera.project(joints)
            pixels[index] = np.where(positive[..., None], uv, np.nan)
            front[index] = positive
            depth_m[index] = joints @ camera.rotation[2] + camera.translation[2]
            vertex_uv, vertex_front = camera.project(world)
            lower = np.clip(vertex_uv.min(-2), 0, size)
            upper = np.clip(vertex_uv.max(-2), 0, size)
            usable = vertex_front.all(-1) & (upper > lower).all(-1)
            mesh_boxes[index] = np.where(
                usable[:, None], np.concatenate((lower, upper), -1), np.nan
            )
    inside = front & ((pixels >= 0) & (pixels < size)).all(-1)
    return Geometry(joints_m, pixels, front, inside, depth_m, mesh_boxes)


class BoxQuality(IntEnum):
    ELIGIBLE = 0
    NO_GVHMR = 1
    INVALID_SMPL = 2
    INSUFFICIENT_POSITIVE_DEPTH = 3
    INSUFFICIENT_IN_IMAGE = 4
    DEGENERATE_BOX = 5
    LIKELY_OCCLUDED = 6
    INSUFFICIENT_VIEW_SUPPORT = 7


class JointQuality(IntEnum):
    ELIGIBLE = 0
    NO_GVHMR = 1
    INVALID_SMPL = 2
    BEHIND_CAMERA = 3
    OUTSIDE_IMAGE = 4
    LIKELY_OCCLUDED = 5
    UNDETERMINED = 6
    INSUFFICIENT_VIEW_SUPPORT = 7
    BOX_EXCLUDED = 8


@dataclass(frozen=True)
class ReferencePolicy:
    box_margin: float = 0.10
    min_box_joints: int = 12
    min_supporting_views: int = 2
    occlusion_depth_gap_m: float = 0.25


DEFAULT_REFERENCE_POLICY = ReferencePolicy()


def _cross(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


_TORSO_TRIANGLES = tuple(
    [COCO_KP_IDX[joint] for joint in triangle]
    for triangle in (
        ("left_shoulder", "right_shoulder", "left_hip"),
        ("right_shoulder", "right_hip", "left_hip"),
    )
)


def _projects_through_nearer_torso(
    geometry: Geometry, depth_gap_m: float
) -> NDArray[np.bool_]:
    occluded = np.zeros_like(geometry.front)
    for other, (pixels, depth) in enumerate(
        zip(geometry.pixels, geometry.depth_m, strict=True)
    ):
        for triangle in _TORSO_TRIANGLES:
            a, b, c = (pixels[..., index : index + 1, :] for index in triangle)
            area = _cross(b - a, c - a)
            signs = np.stack(
                (
                    _cross(b - a, geometry.pixels - a),
                    _cross(c - b, geometry.pixels - b),
                    _cross(a - c, geometry.pixels - c),
                )
            )
            inside = ((signs >= 0).all(0) | (signs <= 0).all(0)) & (np.abs(area) > 1e-6)
            nearer = (
                geometry.depth_m
                >= depth[..., triangle].max(-1)[..., None] + depth_gap_m
            )
            hidden = inside & nearer
            hidden[other] = False
            occluded |= hidden
    return occluded


@dataclass(frozen=True)
class Quality:
    boxes: NDArray[np.float64]
    box_reasons: NDArray[np.uint8]
    joint_reasons: NDArray[np.uint8]
    likely_occluded: NDArray[np.bool_]
    person_view_support: NDArray[np.int64]  # (P,T)
    joint_view_support: NDArray[np.int64]  # (P,T,J)


def qualify_reference(
    geometry: Geometry,
    inferred: NDArray[np.bool_],
    smpl_valid: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    size: tuple[int, int],
    policy: ReferencePolicy = DEFAULT_REFERENCE_POLICY,
) -> Quality:
    """Apply the fixed gate, with the first failing reason on each scoring unit."""
    if (
        observed.shape != geometry.inside.shape
        or inferred.shape != smpl_valid.shape
        or (observed.shape[0], observed.shape[2]) != smpl_valid.shape
    ):
        raise ValueError("Projection, observation and SMPL axes disagree")
    if not 1 <= policy.min_supporting_views <= observed.shape[1] or (
        not np.isfinite(policy.occlusion_depth_gap_m)
        or policy.occlusion_depth_gap_m <= 0
    ):
        raise ValueError("Invalid view support or occlusion depth gap")

    boxes, box_valid = joint_boxes(
        geometry.pixels,
        geometry.inside,
        size,
        margin=policy.box_margin,
        min_joints=policy.min_box_joints,
    )
    likely_occluded = (
        geometry.inside
        & ~observed
        & _projects_through_nearer_torso(geometry, policy.occlusion_depth_gap_m)
    )
    person_views = observed.any(-1).sum(1)
    joint_views = observed.sum(1)
    min_joints = policy.min_box_joints
    box_reasons = np.select(
        [
            ~inferred[:, None],
            ~smpl_valid[:, None],
            geometry.front.sum(-1) < min_joints,
            geometry.inside.sum(-1) < min_joints,
            ~box_valid,
            (geometry.inside & ~likely_occluded).sum(-1) < min_joints,
            person_views[:, None] < policy.min_supporting_views,
        ],
        [
            BoxQuality.NO_GVHMR,
            BoxQuality.INVALID_SMPL,
            BoxQuality.INSUFFICIENT_POSITIVE_DEPTH,
            BoxQuality.INSUFFICIENT_IN_IMAGE,
            BoxQuality.DEGENERATE_BOX,
            BoxQuality.LIKELY_OCCLUDED,
            BoxQuality.INSUFFICIENT_VIEW_SUPPORT,
        ],
        default=BoxQuality.ELIGIBLE,
    ).astype(np.uint8)
    joint_reasons = np.select(
        [
            ~inferred[:, None, :, None],
            ~smpl_valid[:, None, :, None],
            ~geometry.front,
            ~geometry.inside,
            likely_occluded,
            ~observed,
            joint_views[:, None] < policy.min_supporting_views,
            (box_reasons != BoxQuality.ELIGIBLE)[..., None],
        ],
        [
            JointQuality.NO_GVHMR,
            JointQuality.INVALID_SMPL,
            JointQuality.BEHIND_CAMERA,
            JointQuality.OUTSIDE_IMAGE,
            JointQuality.LIKELY_OCCLUDED,
            JointQuality.UNDETERMINED,
            JointQuality.INSUFFICIENT_VIEW_SUPPORT,
            JointQuality.BOX_EXCLUDED,
        ],
        default=JointQuality.ELIGIBLE,
    ).astype(np.uint8)
    return Quality(
        boxes, box_reasons, joint_reasons, likely_occluded, person_views, joint_views
    )
