"""Replay option 2 on saved COCO17/GVHMR outputs; CPU, no neural inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.submodules.configuration import BundledModelAssetPaths
from src.submodules.models import SmplCoco17Reconstructor, SmplVertexReconstructor
from src.tennis_scene.motion_alignment.mesh_placement import (
    canonicalize_incam_body,
    place_canonical_body,
)
from src.tennis_scene.motion_alignment.temporal import (
    TemporalPlacementConfig,
    fit_supported_track,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    placement_weights,
)
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP
from src.utils.geometry.triangulation import PinholeCamera


def replay(repo: Path, inputs: Path, output: Path) -> dict[str, Any]:
    """Read the fixed Meiji bundle and exercise the production placement/renderer boundary."""
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    vendor = repo / "src/submodules/vendor/gvhmr"
    asset_dir = vendor / "body_model/data"
    assets = BundledModelAssetPaths(
        vendor / "hmr2/smpl_mean_params.npz",
        asset_dir / "smplx2smpl_sparse.pt",
        asset_dir / "smpl_coco17_J_regressor.pt",
        asset_dir / "smplx_verts437.pt",
        asset_dir / "smpl_neutral_J_regressor.pt",
    )
    coco = SmplCoco17Reconstructor(
        repo / "ckpt/body_models", device="cpu", bundled_assets=assets
    )
    mesh = SmplVertexReconstructor(
        repo / "ckpt/body_models", device="cpu", bundled_assets=assets
    )
    root_regressor = (
        torch.load(
            assets.smpl_neutral_joint_regressor, map_location="cpu", weights_only=True
        )
        .double()
        .numpy()
    )
    with np.load(inputs / "triangulation.npz") as z:
        data = {key: z[key] for key in z.files}
    meta = json.loads((inputs / "scene_observations.metadata.json").read_text())
    ids = np.arange(0, data["X_init"].shape[1], 2)
    fps = float(data["fps"]) / 2
    target = np.nan_to_num(data["X_init"][:, ids]).astype(np.float32)
    vis = data["used_views"][:, ids].transpose(0, 3, 1, 2)
    errors = (
        np.nan_to_num(np.linalg.norm(data["reprojection_residual_px"][:, ids], axis=-1))
        .transpose(0, 3, 1, 2)
        .astype(np.float32)
    )
    skeleton = PlayerSkeleton(
        target,
        data["valid"][:, ids],
        np.zeros(target.shape[:-1], np.uint8),
        vis,
        errors,
    )
    grouped = GroupedObservations(
        np.arange(2, dtype=np.int64),
        data["observations_px"][:, ids].transpose(0, 3, 1, 2, 4).astype(np.float32),
        np.clip(data["scores"][:, ids].transpose(0, 3, 1, 2), 0, 1).astype(np.float32),
        vis,
        np.zeros(vis.shape[:3], np.int64),
    )
    config = TemporalPlacementConfig()
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, np.float64)
    report: dict[str, Any] = {
        "players": [],
        "device": "cpu",
        "neural_inference": False,
        "source_sha256": {},
    }
    for path in sorted(inputs.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                report["source_sha256"][path.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    for player, camera_index in ((0, 1), (1, 2)):
        with np.load(inputs / f"cam{camera_index}.gvhmr.npz") as z:
            params = {
                key: np.array(z[key][ids], dtype=np.float32)
                for key in ("body_pose", "betas", "global_orient", "transl")
            }
            observed = z["observed_mask"][ids]
        params["betas"][:] = np.median(params["betas"], axis=0)
        pose_before = params["body_pose"].copy()
        joints_y = coco.reconstruct(
            {key: torch.from_numpy(value) for key, value in params.items()}
        ).numpy()
        court = joints_y @ basis.T
        local = court - court[:, [11, 12]].mean(1)[:, None]
        weights = placement_weights(skeleton, grouped, player, config)
        segments = np.where(observed, 0, -1).astype(np.int64)
        fitted = fit_supported_track(
            local,
            target[player].astype(np.float64),
            weights,
            segments,
            fps=fps,
            config=config,
        )
        assert (
            fitted.scale is not None and fitted.valid.sum() >= observed.sum() * 0.9
        ), fitted.intervals
        np.testing.assert_array_equal(params["body_pose"], pose_before)
        corrected = (
            fitted.scale
            * np.einsum(
                "tij,tkj->tki",
                Rotation.from_euler("z", fitted.yaw_correction[:, None]).as_matrix(),
                local,
            )
            + fitted.hip_position[:, None]
        )
        mask = (weights > 0) & fitted.valid[:, None]
        residual = np.linalg.norm(corrected - target[player], axis=-1)[mask]
        # Reconstruct representative real meshes and compare renderer encoding to a direct transform.
        eligible = np.flatnonzero(fitted.valid)
        sample = eligible[[0, len(eligible) // 2, -1]]
        vertices_y = mesh.reconstruct(
            {key: torch.from_numpy(value[sample]) for key, value in params.items()}
        ).numpy()
        values = meta["court_reference"]["camera_fits"][camera_index]
        camera = PinholeCamera(
            f"cam{camera_index}",
            np.asarray(values["K"]),
            np.asarray(values["R"]),
            np.asarray(values["t"]),
        )
        incam_vertices = (
            vertices_y @ basis.T @ camera.rotation.T + camera.translation
        ).astype(np.float32)
        incam_joints = (
            joints_y[sample] @ basis.T @ camera.rotation.T + camera.translation
        ).astype(np.float32)
        incam_orient = (
            Rotation.from_matrix(
                camera.rotation
                @ basis
                @ Rotation.from_rotvec(params["global_orient"][sample]).as_matrix()
            )
            .as_rotvec()
            .astype(np.float32)
        )
        body = canonicalize_incam_body(
            incam_vertices, incam_joints, incam_orient, camera, root_regressor
        )
        placed = place_canonical_body(
            body,
            fitted.hip_position[sample],
            fitted.yaw_correction[sample],
            fitted.scale,
        )
        delta = Rotation.from_euler(
            "z", fitted.yaw_correction[sample, None]
        ).as_matrix()
        direct = (
            fitted.scale
            * np.einsum(
                "tij,tvj->tvi",
                delta,
                (incam_vertices - incam_joints[:, [11, 12]].mean(1)[:, None])
                @ camera.rotation,
            )
            + fitted.hip_position[sample, None]
        )
        encoded_rotation = (
            Rotation.from_euler("z", placed.yaw[:, None]).as_matrix()
            @ basis
            @ Rotation.from_rotvec(placed.global_orient).as_matrix().transpose(0, 2, 1)
        )
        encoded = (
            np.einsum("tij,tvj->tvi", encoded_rotation, placed.vertices_local)
            + placed.position[:, None]
        )
        error = float(np.max(np.abs(encoded - direct)))
        assert error < 3e-5, error
        report["players"].append(
            {
                "player": player,
                "scale": fitted.scale,
                "valid_frames": int(fitted.valid.sum()),
                "source_observed_frames": int(observed.sum()),
                "residual_median_m": float(np.median(residual)),
                "residual_p95_m": float(np.percentile(residual, 95)),
                "mesh_roundtrip_max_m": error,
                "intervals": fitted.intervals,
            }
        )
        np.savez_compressed(
            output / f"player{player}.npz",
            joints=corrected,
            hip_position=fitted.hip_position,
            yaw_correction=fitted.yaw_correction,
            valid=fitted.valid,
            reasons=fitted.reasons,
            scale=fitted.scale,
        )
    mesh.unload()
    coco.unload()
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            replay(args.repo.resolve(), args.inputs.resolve(), args.output.resolve()),
            indent=2,
        )
    )
