"""Unit tests for the read-only ACCAD review service and its SMPL-H schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.plcs.visualization.review.service import (
    JOINT_COUNT,
    ReviewService,
    SMPLHJointModel,
    display_name,
    sampled_frame_count,
    world_contract,
    world_joint_positions,
)
from src.utils.schema.player import (
    SMPLH_BODY_JOINT_NAMES,
    SMPLH_FINGER_NAMES,
    SMPLH_FULL_SKELETON,
    SMPLH_HAND_JOINT_NAMES,
    SMPLH_JOINT_IDX,
    SMPLH_JOINT_NAMES,
)

_ACCAD_ROOT = "/home/kamimura/projects/tennis-lab/data/ACCAD"
_SMPLH_ROOT = "/home/kamimura/projects/tennis-lab/data/smplh"
_ACCAD_CLIP = "Male1Walking_c3d/Walk B10 - Walk turn left 45_poses.npz"


def _parents_from_skeleton() -> np.ndarray:
    parents: np.ndarray = np.full(JOINT_COUNT, -1, dtype=np.int64)
    for child in range(1, JOINT_COUNT):
        candidates = [parent for parent, target in SMPLH_FULL_SKELETON if target == child]
        assert len(candidates) == 1, child
        parents[child] = candidates[0]
    return parents


def _synthetic_model(seed: int = 0, beta_count: int = 2) -> SMPLHJointModel:
    """A tiny stand-in model whose joints are its first 52 template vertices."""
    rng = np.random.default_rng(seed)
    vertex_count = 96
    template = rng.normal(0.0, 0.4, size=(vertex_count, 3))
    template[0] = 0.0  # keep the root at the origin so rotations are unambiguous
    return SMPLHJointModel(
        gender="neutral",
        template_vertices_m=template,
        shape_directions=rng.normal(0.0, 0.05, size=(vertex_count, 3, beta_count)),
        joint_regressor=np.eye(JOINT_COUNT, vertex_count),
        parents=_parents_from_skeleton(),
    )


def _canonical(model: SMPLHJointModel) -> np.ndarray:
    return cast(
        np.ndarray,
        np.matmul(model.joint_regressor, model.template_vertices_m),
    )


# ------------------------------------------------------------ forward kinematics


def test_identity_pose_places_canonical_joints_plus_root_translation() -> None:
    model = _synthetic_model()
    translation = np.array([[1.0, -2.0, 0.5]])
    joints = world_joint_positions(
        model, np.zeros((1, 156)), translation, np.zeros(model.beta_count)
    )
    np.testing.assert_allclose(joints[0], _canonical(model) + translation[0], atol=1e-5)


def test_shape_blend_moves_the_canonical_joints() -> None:
    model = _synthetic_model(seed=2, beta_count=3)
    betas = np.array([0.5, -0.25, 1.0])
    expected = np.einsum(
        "vdb,b->vd", model.shape_directions, betas
    ) + model.template_vertices_m
    joints = world_joint_positions(
        model, np.zeros((1, 156)), np.zeros((1, 3)), betas
    )
    np.testing.assert_allclose(joints[0], expected[:JOINT_COUNT], atol=1e-5)


def test_posed_frames_preserve_every_bone_length() -> None:
    """SMPL-H joints are rigidly transformed, so bones cannot stretch."""
    model = _synthetic_model(seed=7)
    rng = np.random.default_rng(11)
    poses = rng.normal(0.0, 0.4, size=(5, 156))
    trans = rng.normal(0.0, 1.0, size=(5, 3))
    joints = world_joint_positions(model, poses, trans, np.zeros(model.beta_count))
    canonical = _canonical(model)
    for parent, child in SMPLH_FULL_SKELETON:
        expected = float(np.linalg.norm(canonical[child] - canonical[parent]))
        actual = np.linalg.norm(joints[:, child] - joints[:, parent], axis=-1)
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_global_orientation_rotates_the_body_about_the_root() -> None:
    model = _synthetic_model(seed=5)
    canonical = _canonical(model)
    pose = np.zeros((1, 156))
    pose[0, 2] = np.pi / 2  # +Z quarter turn of the SMPL-H root
    joints = world_joint_positions(model, pose, np.zeros((1, 3)), np.zeros(model.beta_count))
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(joints[0], canonical @ quarter_turn.T, atol=1e-5)


@pytest.mark.parametrize(
    ("poses", "trans", "betas"),
    [
        (np.zeros((1, 155)), np.zeros((1, 3)), np.zeros(2)),
        (np.zeros((1, 156)), np.zeros((2, 3)), np.zeros(2)),
        (np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(3)),
        (np.full((1, 156), np.nan), np.zeros((1, 3)), np.zeros(2)),
        (np.zeros((1, 156)), np.full((1, 3), np.inf), np.zeros(2)),
        (np.zeros((0, 156)), np.zeros((0, 3)), np.zeros(2)),
    ],
)
def test_world_joint_positions_rejects_malformed_input(
    poses: np.ndarray, trans: np.ndarray, betas: np.ndarray
) -> None:
    model = _synthetic_model()
    with pytest.raises(ValueError):
        world_joint_positions(model, poses, trans, betas)


# ------------------------------------------------------------------- schema


def test_the_full_skeleton_is_the_official_52_joint_tree() -> None:
    assert len(SMPLH_JOINT_NAMES) == 52
    assert len(set(SMPLH_JOINT_NAMES)) == 52
    assert SMPLH_JOINT_NAMES[:22] == SMPLH_BODY_JOINT_NAMES
    assert len(SMPLH_HAND_JOINT_NAMES) == 30
    assert len(SMPLH_FULL_SKELETON) == 51
    # Every joint except the root has exactly one parent, which the schema
    # test above already asserts while building the tree.
    children = sorted(child for _, child in SMPLH_FULL_SKELETON)
    assert children == list(range(1, 52))


def test_each_finger_is_a_three_joint_chain_hanging_from_its_wrist() -> None:
    index_of = {name: index for index, name in enumerate(SMPLH_JOINT_NAMES)}
    edges = set(SMPLH_FULL_SKELETON)
    for hand in ("left", "right"):
        wrist = index_of[f"{hand}_wrist"]
        for finger in SMPLH_FINGER_NAMES:
            first, second, third = (
                index_of[f"{hand}_{finger}{joint}"] for joint in (1, 2, 3)
            )
            assert (wrist, first) in edges
            assert (first, second) in edges
            assert (second, third) in edges
    # The two hands must not share any joint.
    assert SMPLH_JOINT_IDX["left_wrist"] != SMPLH_JOINT_IDX["right_wrist"]


@pytest.mark.local_data
def test_hand_joint_names_match_the_model_finger_geometry() -> None:
    """The MANO finger order is checked against the official rest skeleton."""
    model_path = f"{_SMPLH_ROOT}/male/model.npz"
    if not Path(model_path).is_file():
        pytest.skip("Licensed SMPL-H assets are unavailable.")
    with np.load(model_path, allow_pickle=False) as archive:
        rest = np.asarray(archive["J_regressor"]) @ np.asarray(archive["v_template"])
    index_of = {name: index for index, name in enumerate(SMPLH_JOINT_NAMES)}
    for hand in ("left", "right"):
        wrist = index_of[f"{hand}_wrist"]
        reach = {}
        for finger in SMPLH_FINGER_NAMES:
            chain = [index_of[f"{hand}_{finger}{joint}"] for joint in (1, 2, 3)]
            points = [rest[wrist], *(rest[joint] for joint in chain)]
            reach[finger] = sum(
                float(np.linalg.norm(points[i + 1] - points[i]))
                for i in range(len(points) - 1)
            )
        assert reach["middle"] == max(reach.values())
        assert reach["pinky"] < min(reach["index"], reach["middle"], reach["ring"])
        assert reach["thumb"] < reach["pinky"]


# ------------------------------------------------------------------ catalog


def _write_archive(
    root,
    subject: str,
    motion: str,
    *,
    frames: int = 4,
    gender: str = "male",
    fps: float = 120.0,
) -> None:
    path = root / subject / motion
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        gender=np.array(gender),
        mocap_framerate=np.array(fps),
        poses=np.zeros((frames, 156)),
        trans=np.zeros((frames, 3)),
        betas=np.zeros(16),
    )


def _service(tmp_path) -> ReviewService:
    accad = tmp_path / "ACCAD"
    accad.mkdir(exist_ok=True)
    (tmp_path / "smplh").mkdir(exist_ok=True)
    return ReviewService(accad, tmp_path / "smplh")


def test_catalog_reports_archive_metadata_and_world_contract(tmp_path) -> None:
    service = _service(tmp_path)
    root = service.root
    _write_archive(root, "SubjectA", "B1 - walk_poses.npz", frames=4)
    _write_archive(root, "SubjectA", "B2 - run_poses.npz", frames=8, fps=60.0)
    _write_archive(root, "SubjectB", "C1 - jump_poses.npz", frames=2, gender="female")

    catalog = service.catalog()
    assert [subject["id"] for subject in catalog["subjects"]] == [
        "SubjectA",
        "SubjectB",
    ]
    assert [subject["motion_count"] for subject in catalog["subjects"]] == [2, 1]
    assert catalog["world"] == world_contract()
    assert catalog["world"]["schema"] == "plcs_amass_smplh_z_up_v1"
    assert catalog["joints"]["count"] == 52
    assert len(catalog["joints"]["edges"]) == 51

    first = catalog["subjects"][0]["motions"][0]
    assert first["name"] == "B1 - walk"
    assert first["gender"] == "male"
    assert first["fps"] == 120.0
    assert first["frame_count"] == 4
    assert first["duration_s"] == pytest.approx(4 / 120)
    assert first["size_bytes"] > 0

    meta = service.motion_meta("SubjectA", "B2 - run_poses.npz", stride=2)
    assert meta["sampled_frame_count"] == 4
    assert meta["source_frame_count"] == 8
    assert meta["stride"] == 2
    assert meta["joints"]["count"] == 52


def test_motion_path_rejects_identifiers_outside_the_root(tmp_path) -> None:
    service = _service(tmp_path)
    for subject, motion in (
        ("..", "B1 - walk_poses.npz"),
        ("SubjectA", "../B1 - walk_poses.npz"),
        ("SubjectA", "/etc/passwd"),
        ("SubjectA", "B1 - walk.npz"),
        ("", "B1 - walk_poses.npz"),
    ):
        with pytest.raises(ValueError):
            service.motion_path(subject, motion)
    with pytest.raises(FileNotFoundError):
        service.motion_path("SubjectA", "Missing_poses.npz")


def test_helpers_normalize_names_and_strides() -> None:
    assert display_name("Walk B10 - Walk turn left 45_poses.npz") == (
        "Walk B10 - Walk turn left 45"
    )
    with pytest.raises(ValueError):
        display_name("Walk B10.npz")
    assert sampled_frame_count(1011, 1) == 1011
    assert sampled_frame_count(1011, 4) == 253
    assert sampled_frame_count(1010, 4) == 253
    for frame_count, stride in ((0, 1), (4, 0), (4, -1)):
        with pytest.raises(ValueError):
            sampled_frame_count(frame_count, stride)


# ------------------------------------------------------------------ local data


@pytest.mark.local_data
def test_real_accad_joints_match_the_reference_lbs_and_stay_rigid() -> None:
    import torch
    from smplx.lbs import lbs  # type: ignore[import-untyped]

    accad_root = Path(_ACCAD_ROOT)
    smplh_root = Path(_SMPLH_ROOT)
    clip_path = accad_root / _ACCAD_CLIP
    model_path = smplh_root / "male" / "model.npz"
    if not clip_path.is_file() or not model_path.is_file():
        pytest.skip("Licensed ACCAD/SMPL-H assets are unavailable.")

    service = ReviewService(accad_root, smplh_root)
    subject, motion = _ACCAD_CLIP.split("/")
    loaded = service.motion_joints(subject, motion, stride=17, revision=None)
    joints = loaded.joints.astype(np.float64)
    assert joints.shape == (loaded.sampled_frame_count, JOINT_COUNT, 3)
    assert np.isfinite(joints).all()

    # Recompute the same frames through the public smplx linear blend skinning
    # reference, using the official arrays rather than the review service.
    def tensor(value: object) -> Any:
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float64)

    with np.load(clip_path, allow_pickle=False) as archive:
        poses = np.asarray(archive["poses"], dtype=np.float64)[::17]
        trans = np.asarray(archive["trans"], dtype=np.float64)[::17]
        betas = np.asarray(archive["betas"], dtype=np.float64)
    with np.load(model_path, allow_pickle=False) as archive:
        posedirs = np.asarray(archive["posedirs"])
        parents = np.asarray(archive["kintree_table"][0]).astype(np.int64)
        parents[0] = -1
        _, reference = lbs(
            tensor(betas).unsqueeze(0).expand(poses.shape[0], -1),
            tensor(poses),
            tensor(archive["v_template"]),
            tensor(archive["shapedirs"]),
            tensor(posedirs.transpose(2, 0, 1).reshape(posedirs.shape[2], -1)),
            tensor(archive["J_regressor"]),
            torch.as_tensor(parents),
            tensor(archive["weights"]),
        )
    reference = reference.numpy() + trans[:, None, :]
    np.testing.assert_allclose(joints, reference, atol=1e-4)

    # Rigid SMPL-H joints keep every bone length across the whole clip.
    for parent, child in SMPLH_FULL_SKELETON:
        expected = float(np.linalg.norm(joints[0, child] - joints[0, parent]))
        actual = np.linalg.norm(joints[:, child] - joints[:, parent], axis=-1)
        np.testing.assert_allclose(actual, expected, atol=1e-4)
