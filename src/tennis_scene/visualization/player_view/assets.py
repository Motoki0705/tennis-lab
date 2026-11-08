"""Extract portable numeric skeleton assets without constructing a body model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.utils.schema.player import SMPLH_SKELETON


def load_body_kinematics(model_path: Path) -> dict[str, NDArray]:
    """Read the root and 21 articulated body joints from a neutral SMPL-X NPZ.

    These are the joints controlled by SceneResult's 63 body-pose values.
    Hand/face articulation is outside that pose contract. Only numeric NPZ
    fields are read; no licensed model or neural network is initialized.
    """
    with np.load(model_path, allow_pickle=False) as model:
        template = np.asarray(model["v_template"], dtype=np.float32)
        regressor = np.asarray(model["J_regressor"], dtype=np.float32)[:22]
        shapedirs = np.asarray(model["shapedirs"][..., :10], dtype=np.float32)
        tree = np.asarray(model["kintree_table"])
    vertices = len(template)
    if (
        template.shape != (vertices, 3)
        or vertices == 0
        or regressor.shape != (22, vertices)
        or shapedirs.shape != (vertices, 3, 10)
        or tree.ndim != 2
        or tree.shape[0] != 2
        or tree.shape[1] < 22
        or tree.dtype.kind not in "iu"
    ):
        raise ValueError("SMPL-X kinematic asset dimensions disagree")
    if not all(np.isfinite(x).all() for x in (template, regressor, shapedirs)):
        raise ValueError("SMPL-X kinematic assets must be finite")
    if not np.allclose(regressor.sum(1), 1.0, atol=1e-4):
        raise ValueError("SMPL-X joint regressor rows must sum to one")
    if not np.array_equal(tree[1, :22], np.arange(22)):
        raise ValueError("SMPL-X body joint IDs must follow the pose order")
    if int(tree[0, 0]) not in (-1, 2**32 - 1):
        raise ValueError("SMPL-X root must have the standard parent sentinel")
    parents = tree[0, :22].astype(np.int32)
    parents[0] = -1  # NPZ encodes the root sentinel as an unsigned integer.
    if any(not 0 <= int(parents[j]) < j for j in range(1, 22)):
        raise ValueError("SMPL-X body parents must precede their children")
    if {(int(parents[j]), j) for j in range(1, 22)} != set(SMPLH_SKELETON):
        raise ValueError("SMPL-X tree disagrees with the named 22-joint body skeleton")
    return {
        "smplx_parents": parents,
        "smplx_rest_joints": regressor @ template,
        "smplx_joint_shapedirs": np.einsum("jv,vck->jck", regressor, shapedirs),
    }
