"""Geometry-contract tests for the in-pipeline GVHMR/PLCS scene alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.motion_alignment.similarity import SimilarityConfig
from src.tennis_scene.pipeline.components.motion_alignment import (
    AlignedTrack,
    AlignmentWeights,
    CanonicalMotion,
    align_track_to_plcs,
    derive_alignment_weights,
    derive_canonical_motion,
)
from src.utils.geometry.matrices import (
    SMPL_Y_UP_TO_COURT_Z_UP,
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
    smpl_y_up_to_court_z_up,
)
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

_NUM_FRAMES = 24
_NUM_VERTICES = 6
_RECONSTRUCTION_TOLERANCE_M = 1e-4


def _rotation_matrices(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        axis_angle_to_matrix(torch.from_numpy(axis_angle)).numpy(),
        dtype=np.float64,
    )


def _rotation_z(yaw: float) -> NDArray[np.float64]:
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    return np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _SyntheticTrack:
    """One synthetic track whose exact similarity transform is known."""

    joint_regressor: NDArray[np.float64]
    vertices_canonical: NDArray[np.float64]
    canonical_root: NDArray[np.float64]
    world_rotation: NDArray[np.float64]
    world_axis_angle: NDArray[np.float64]
    transl_world: NDArray[np.float64]
    transl_incam: NDArray[np.float64]
    incam_axis_angle: NDArray[np.float64]
    vertices_incam: NDArray[np.float64]
    scale: float
    yaw: float
    translation: NDArray[np.float64]
    source_position: NDArray[np.float64]
    source_heading: NDArray[np.float64]
    target_position: NDArray[np.float64]
    target_yaw: NDArray[np.float64]

    def weights(self) -> AlignmentWeights:
        return AlignmentWeights(
            position=np.ones(_NUM_FRAMES, dtype=np.float64),
            heading=np.ones(_NUM_FRAMES, dtype=np.float64),
            observed=np.ones(_NUM_FRAMES, dtype=np.bool_),
            confidence_clipped=False,
        )


def _synthetic_track(
    *,
    scale: float = 1.25,
    yaw: float = 0.7,
    translation: tuple[float, float, float] = (0.4, -0.2, 0.05),
    seed: int = 7,
) -> _SyntheticTrack:
    """Build a track whose world motion maps exactly onto the PLCS target."""
    rng = np.random.default_rng(seed)
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    joint_regressor: NDArray[np.float64] = np.zeros(
        (24, _NUM_VERTICES), dtype=np.float64
    )
    joint_regressor[0, 0] = 1.0

    vertices_canonical = rng.normal(
        scale=0.3, size=(_NUM_FRAMES, _NUM_VERTICES, 3)
    )
    canonical_root = np.einsum("v,tvc->tc", joint_regressor[0], vertices_canonical)
    world_axis_angle = rng.normal(scale=0.6, size=(_NUM_FRAMES, 3))
    world_rotation = _rotation_matrices(world_axis_angle)
    transl_world = rng.normal(scale=0.2, size=(_NUM_FRAMES, 3))
    world_root = (
        np.einsum("tij,tj->ti", world_rotation, canonical_root) + transl_world
    )
    source_position = world_root @ basis.T
    rotation_court = np.einsum("ij,tjk->tik", basis, world_rotation)
    source_heading = np.arctan2(rotation_court[:, 1, 0], rotation_court[:, 0, 0])

    offset = np.asarray(translation, dtype=np.float64)
    target_position = scale * (source_position @ _rotation_z(yaw).T) + offset
    target_yaw = np.arctan2(
        np.sin(source_heading + yaw), np.cos(source_heading + yaw)
    )

    incam_axis_angle = rng.normal(scale=0.5, size=(_NUM_FRAMES, 3))
    transl_incam = rng.normal(scale=0.3, size=(_NUM_FRAMES, 3))
    vertices_incam = (
        np.einsum("tij,tvj->tvi", _rotation_matrices(incam_axis_angle), vertices_canonical)
        + transl_incam[:, None, :]
    )
    return _SyntheticTrack(
        joint_regressor=joint_regressor,
        vertices_canonical=vertices_canonical,
        canonical_root=canonical_root,
        world_rotation=world_rotation,
        world_axis_angle=world_axis_angle,
        transl_world=transl_world,
        transl_incam=transl_incam,
        incam_axis_angle=incam_axis_angle,
        vertices_incam=vertices_incam,
        scale=scale,
        yaw=yaw,
        translation=offset,
        source_position=source_position,
        source_heading=source_heading,
        target_position=target_position,
        target_yaw=target_yaw,
    )


def _motion(track: _SyntheticTrack) -> CanonicalMotion:
    return derive_canonical_motion(
        vertices_incam=track.vertices_incam,
        global_orient_incam=track.incam_axis_angle,
        transl_incam=track.transl_incam,
        global_orient_world=track.world_axis_angle,
        transl_world=track.transl_world,
        joint_regressor=track.joint_regressor,
    )


def _aligned(
    track: _SyntheticTrack, config: SimilarityConfig
) -> AlignedTrack:
    return align_track_to_plcs(
        motion=_motion(track),
        target_position=track.target_position,
        target_yaw=track.target_yaw,
        weights=track.weights(),
        config=config,
    )


def _renderer_court_vertices(
    aligned: AlignedTrack, joint_regressor: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Reproduce the renderer's authoritative placement rule exactly."""
    vertices_local = np.asarray(aligned.smpl_vertices_local, dtype=np.float32)
    global_orient = np.asarray(aligned.smpl_global_orient, dtype=np.float32)
    player_position = np.asarray(aligned.player_position, dtype=np.float32)
    player_yaw = np.asarray(aligned.player_yaw, dtype=np.float32)

    roots = np.einsum(
        "jv,tvc->tjc", joint_regressor.astype(np.float32), vertices_local
    )[:, 0, :]
    centered = vertices_local - roots[:, None, :]
    orientation = axis_angle_to_rotation_matrix(global_orient)
    pose = np.einsum("tji,tvj->tvi", orientation, centered)
    court_local = smpl_y_up_to_court_z_up(pose)
    yaw_rotation = rotation_matrix_z(player_yaw)
    return (
        np.einsum("tij,tvj->tvi", yaw_rotation, court_local)
        + player_position[:, None, :]
    )


def _direct_similarity_vertices(
    aligned: AlignedTrack, track: _SyntheticTrack
) -> NDArray[np.float64]:
    """Directly similarity-transform the world vertices for the fitted transform."""
    transform = aligned.transform
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    root_relative_world = np.einsum(
        "tij,tvj->tvi",
        track.world_rotation,
        track.vertices_canonical - track.canonical_root[:, None, :],
    )
    geometry = transform.scale * np.einsum(
        "ij,tvj->tvi", transform.rotation() @ basis, root_relative_world
    )
    translated = (
        transform.scale * (track.source_position @ transform.rotation().T)
        + transform.translation
    )
    return geometry + translated[:, None, :]


def test_canonical_derivation_recovers_orientation_free_motion() -> None:
    track = _synthetic_track()
    motion = _motion(track)

    np.testing.assert_allclose(
        motion.canonical_vertices,
        track.vertices_canonical,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        motion.canonical_root,
        track.canonical_root,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        motion.source_position,
        track.source_position,
        atol=1e-10,
    )
    heading_error = np.abs(
        np.arctan2(
            np.sin(motion.source_heading - track.source_heading),
            np.cos(motion.source_heading - track.source_heading),
        )
    )
    assert heading_error.max() < 1e-10


def test_free_scale_round_trip_recovers_the_synthetic_sim3() -> None:
    track = _synthetic_track()
    aligned = _aligned(track, SimilarityConfig(scale_prior=0.0, fixed_scale=None))
    transform = aligned.transform

    assert transform.scale == pytest.approx(track.scale, abs=1e-8)
    assert transform.yaw == pytest.approx(track.yaw, abs=1e-8)
    np.testing.assert_allclose(transform.translation, track.translation, atol=1e-8)
    np.testing.assert_allclose(
        aligned.player_position, track.target_position, atol=1e-8
    )


def test_fixed_scale_pins_the_scale_to_one() -> None:
    track = _synthetic_track()
    aligned = _aligned(track, SimilarityConfig(fixed_scale=1.0, scale_prior=0.0))

    assert aligned.transform.scale == 1.0
    assert aligned.transform.yaw == pytest.approx(track.yaw, abs=1e-8)
    # A 1.25x target cannot be reproduced by scale 1, so the fit must differ.
    assert not np.allclose(
        aligned.player_position, track.target_position, atol=1e-3
    )


@pytest.mark.parametrize("fixed_scale", [1.0, None])
def test_renderer_rule_reproduces_the_direct_similarity_transform(
    fixed_scale: float | None,
) -> None:
    track = _synthetic_track()
    aligned = _aligned(
        track,
        SimilarityConfig(fixed_scale=fixed_scale, scale_prior=0.0),
    )

    rendered = _renderer_court_vertices(aligned, track.joint_regressor)
    expected = _direct_similarity_vertices(aligned, track)

    assert float(np.abs(rendered - expected).max()) < _RECONSTRUCTION_TOLERANCE_M


def test_zero_position_weights_are_rejected() -> None:
    track = _synthetic_track()
    weights = AlignmentWeights(
        position=np.zeros(_NUM_FRAMES, dtype=np.float64),
        heading=np.ones(_NUM_FRAMES, dtype=np.float64),
        observed=np.ones(_NUM_FRAMES, dtype=np.bool_),
        confidence_clipped=False,
    )

    with pytest.raises(ValueError, match="position observation"):
        align_track_to_plcs(
            motion=_motion(track),
            target_position=track.target_position,
            target_yaw=track.target_yaw,
            weights=weights,
            config=SimilarityConfig(fixed_scale=1.0),
        )


def test_unobservable_yaw_is_rejected() -> None:
    track = _synthetic_track()
    motion = _motion(track)
    # Stationary court positions plus zero heading weight leave the yaw
    # undetermined; the fit must raise rather than pick an arbitrary value.
    imbalanced = replace(motion, source_position=np.zeros_like(motion.source_position))
    weights = AlignmentWeights(
        position=np.ones(_NUM_FRAMES, dtype=np.float64),
        heading=np.zeros(_NUM_FRAMES, dtype=np.float64),
        observed=np.ones(_NUM_FRAMES, dtype=np.bool_),
        confidence_clipped=False,
    )
    with pytest.raises(ValueError, match="cannot initialise yaw"):
        align_track_to_plcs(
            motion=imbalanced,
            target_position=track.target_position,
            target_yaw=track.target_yaw,
            weights=weights,
            config=SimilarityConfig(fixed_scale=1.0),
        )


def test_outlier_frames_do_not_dominate_the_robust_fit() -> None:
    track = _synthetic_track()
    corrupted = track.target_position.copy()
    corrupted[3] += np.array([6.0, -5.0, 0.0])
    corrupted[11] += np.array([-7.0, 4.0, 0.0])
    corrupted[17] += np.array([5.0, 6.0, 0.0])

    aligned = align_track_to_plcs(
        motion=_motion(track),
        target_position=corrupted,
        target_yaw=track.target_yaw,
        weights=track.weights(),
        config=SimilarityConfig(fixed_scale=1.0, scale_prior=0.0),
    )

    assert aligned.transform.yaw == pytest.approx(track.yaw, abs=0.05)
    assert aligned.transform.translation == pytest.approx(
        track.translation, abs=0.05
    )


def test_confidence_clipping_is_recorded_in_diagnostics() -> None:
    frames = 4
    target_visibility: NDArray[np.float64] = np.full(
        (frames, 17), 0.8, dtype=np.float64
    )
    source_visibility: NDArray[np.float64] = np.full(
        (frames, 17), 1.4, dtype=np.float64
    )
    court_visibility: NDArray[np.float64] = np.ones(
        (1, frames, 20), dtype=np.float64
    )
    rotation_court = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()

    weights = derive_alignment_weights(
        target_visibility_ref=target_visibility,
        source_visibility_ref=source_visibility,
        court_visibility=court_visibility,
        rotation_court=rotation_court,
        observed=np.ones(frames, dtype=np.bool_),
    )

    assert weights.confidence_clipped
    assert np.all(weights.position >= 0.0)
    # The out-of-range source confidence is clipped to 1, so the hip weight
    # reduces to the clipped target hip confidence.
    np.testing.assert_allclose(weights.position, 0.8)

    clean = derive_alignment_weights(
        target_visibility_ref=target_visibility,
        source_visibility_ref=np.full((frames, 17), 0.5, dtype=np.float64),
        court_visibility=court_visibility,
        rotation_court=rotation_court,
        observed=np.ones(frames, dtype=np.bool_),
    )
    assert not clean.confidence_clipped
