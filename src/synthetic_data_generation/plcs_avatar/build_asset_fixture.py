"""Build a deterministic SMPL-X-controlled Gaussian-avatar geometry fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.submodules.vendor.gvhmr.body_model.smplx_lite import (
    SmplxLite,
    batch_rigid_transform_v2,
)
from src.synthetic_data_generation.plcs.avatar_asset import (
    build_surface_gaussian_asset,
    deform_avatar_gaussians,
)
from src.synthetic_data_generation.plcs.avatar_control import (
    embed_points_on_posed_mesh,
)
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

SCHEMA = "plcs_smplx_gaussian_asset_fixture_v1"
SEED = 20260728
POSE_IDS = ("canonical", "ready", "forehand")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _motion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body = torch.zeros((1, len(POSE_IDS), 63), dtype=torch.float32)

    def set_joint(frame: int, joint: int, xyz: tuple[float, float, float]) -> None:
        body[0, frame, joint * 3 : joint * 3 + 3] = torch.tensor(xyz)

    # Ready stance: flex knees, lower arms, and turn shoulders.
    set_joint(1, 3, (0.32, 0.0, 0.0))
    set_joint(1, 4, (0.32, 0.0, 0.0))
    set_joint(1, 11, (0.0, 0.15, 0.12))
    set_joint(1, 15, (0.05, 0.0, -0.55))
    set_joint(1, 16, (-0.05, 0.0, 0.55))
    set_joint(1, 17, (0.48, 0.08, -0.18))
    set_joint(1, 18, (0.48, -0.08, 0.18))

    # Forehand: asymmetric shoulder/elbow/wrist motion and lower-body transfer.
    set_joint(2, 0, (0.18, 0.0, 0.08))
    set_joint(2, 1, (-0.12, 0.0, -0.05))
    set_joint(2, 3, (0.42, 0.0, 0.0))
    set_joint(2, 4, (0.22, 0.0, 0.0))
    set_joint(2, 11, (0.0, 0.35, 0.22))
    set_joint(2, 15, (0.20, 0.05, -0.65))
    set_joint(2, 16, (-0.10, -0.04, 0.40))
    set_joint(2, 17, (0.55, 0.12, -0.65))
    set_joint(2, 18, (0.36, -0.10, 0.30))
    set_joint(2, 19, (0.10, 0.0, 0.42))
    set_joint(2, 20, (-0.08, 0.0, -0.20))

    betas = torch.zeros((1, len(POSE_IDS), 10), dtype=torch.float32)
    betas[..., 1] = -0.35
    betas[..., 2] = 0.20
    global_orient = torch.zeros((1, len(POSE_IDS), 3), dtype=torch.float32)
    global_orient[0, 1, 1] = -0.12
    global_orient[0, 2, 1] = 0.28
    translations = torch.zeros((1, len(POSE_IDS), 3), dtype=torch.float32)
    return body, betas, global_orient, translations


def _relative_joint_transforms(
    model: SmplxLite,
    *,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
) -> torch.Tensor:
    frame_count = body_pose.shape[1]
    defaults = model.other_default_pose.expand(1, frame_count, -1)
    full_pose = torch.cat((global_orient, body_pose, defaults), dim=-1)
    rotations = axis_angle_to_matrix(full_pose.reshape(1, frame_count, 55, 3))
    joints = model.get_skeleton(betas)
    live = batch_rigid_transform_v2(rotations, joints, model.parents)[1]

    canonical_pose = torch.cat(
        (
            torch.zeros((1, 1, 3)),
            torch.zeros((1, 1, 63)),
            model.other_default_pose.expand(1, 1, -1),
        ),
        dim=-1,
    )
    canonical_rotations = axis_angle_to_matrix(canonical_pose.reshape(1, 1, 55, 3))
    canonical = batch_rigid_transform_v2(
        canonical_rotations,
        model.get_skeleton(betas[:, :1]),
        model.parents,
    )[1]
    return live @ torch.linalg.inv(canonical)


def _save_tensor_set(
    path: Path,
    *,
    means: np.ndarray,
    quaternions: np.ndarray,
    log_scales: np.ndarray,
    opacity_logits: np.ndarray,
    feature_dim: int,
) -> None:
    count = means.shape[0]
    torch.save(
        {
            "means": torch.from_numpy(means.astype(np.float32)),
            "quats": torch.from_numpy(quaternions.astype(np.float32)),
            "scales": torch.from_numpy(log_scales.astype(np.float32)),
            "opacities": torch.from_numpy(opacity_logits.astype(np.float32)),
            "features": torch.zeros((count, feature_dim), dtype=torch.float32),
            "instance_ids": torch.zeros(count, dtype=torch.int64),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-appearance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-count", type=int, default=4096)
    args = parser.parse_args()

    output = args.output.resolve()
    model_path = args.model.resolve()
    appearance_path = args.target_appearance.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    if not model_path.is_file() or not appearance_path.is_file():
        raise SystemExit("SMPL-X model and target NHT appearance must exist.")
    appearance = torch.load(appearance_path, map_location="cpu", weights_only=True)
    if not isinstance(appearance, dict) or set(appearance) != {"config", "state_dict"}:
        raise SystemExit("Target NHT appearance has an unsupported payload.")
    config = appearance["config"]
    if not isinstance(config, dict) or not isinstance(config.get("feature_dim"), int):
        raise SystemExit("Target NHT appearance has no integer feature_dim.")
    feature_dim = int(config["feature_dim"])

    torch.manual_seed(SEED)
    model = SmplxLite(model_path=model_path).eval()
    body, betas, global_orient, translations = _motion()
    with torch.inference_mode():
        vertices = model(body, betas, global_orient, translations)[0]
        canonical_vertices = model(
            torch.zeros((1, 1, 63)),
            betas[:, :1],
            torch.zeros((1, 1, 3)),
            torch.zeros((1, 1, 3)),
        )[0, 0]
        transforms = _relative_joint_transforms(
            model,
            body_pose=body,
            betas=betas,
            global_orient=global_orient,
        )[0]

    asset = build_surface_gaussian_asset(
        canonical_vertices.numpy(),
        faces=np.asarray(model.faces),
        vertex_joint_weights=model.lbs_weights.numpy(),
        gaussian_count=args.gaussian_count,
        seed=SEED,
    )
    deformed = deform_avatar_gaussians(
        asset,
        joint_transforms=transforms.numpy(),
        translations_m=translations[0].numpy(),
    )
    mesh_reference = embed_points_on_posed_mesh(
        vertices.numpy(),
        faces=np.asarray(model.faces),
        face_indices=asset.face_indices,
        barycentric_coordinates=asset.barycentric_coordinates,
    )
    error_mm = np.linalg.norm(deformed.means_m - mesh_reference, axis=-1) * 1000.0
    per_pose_metrics = [
        {
            "pose_id": pose_id,
            "mean_attachment_error_mm": float(error_mm[index].mean()),
            "p95_attachment_error_mm": float(np.quantile(error_mm[index], 0.95)),
            "max_attachment_error_mm": float(error_mm[index].max()),
            "mean_displacement_from_canonical_m": float(
                np.linalg.norm(
                    deformed.means_m[index] - deformed.means_m[0],
                    axis=-1,
                ).mean()
            ),
        }
        for index, pose_id in enumerate(POSE_IDS)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        poses_root = temporary / "poses"
        control_root = temporary / "control"
        poses_root.mkdir()
        control_root.mkdir()
        canonical_source = temporary / "canonical-source.pt"
        _save_tensor_set(
            canonical_source,
            means=asset.means_m,
            quaternions=asset.quaternions_wxyz,
            log_scales=asset.log_scales_m,
            opacity_logits=asset.opacity_logits,
            feature_dim=feature_dim,
        )
        for index, pose_id in enumerate(POSE_IDS):
            _save_tensor_set(
                poses_root / f"{index:03d}-{pose_id}.pt",
                means=deformed.means_m[index],
                quaternions=deformed.quaternions_wxyz[index],
                log_scales=deformed.log_scales_m[index],
                opacity_logits=asset.opacity_logits,
                feature_dim=feature_dim,
            )
        arrays = {
            "body_pose_axis_angle.npy": body[0].numpy(),
            "betas.npy": betas[0].numpy(),
            "global_orient_axis_angle.npy": global_orient[0].numpy(),
            "translations_m.npy": translations[0].numpy(),
            "joint_transforms.npy": transforms.numpy(),
            "point_joint_weights.npy": asset.point_joint_weights,
            "face_indices.npy": asset.face_indices,
            "barycentric_coordinates.npy": asset.barycentric_coordinates,
        }
        for filename, value in arrays.items():
            np.save(control_root / filename, value, allow_pickle=False)
        files = {
            path.relative_to(temporary).as_posix(): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        unsigned = {
            "schema": SCHEMA,
            "asset_id": "codex-smplx-surface-avatar-v1",
            "asset_origin": "codex-generated-smplx-prototype",
            "seed": SEED,
            "smplx_model": {
                "uri": model_path.as_uri(),
                "sha256": _sha256(model_path),
                "copied_into_artifact": False,
            },
            "target_nht_appearance": {
                "uri": appearance_path.as_uri(),
                "sha256": _sha256(appearance_path),
                "feature_dim": feature_dim,
            },
            "construction": {
                "method": "area-weighted-smplx-surface-gaussians-v1",
                "gaussian_count": asset.gaussian_count,
                "joint_count": int(asset.point_joint_weights.shape[1]),
                "body_pose_joint_count": 21,
                "dropped_joint_indices": [],
                "covariance_control": "exact-linear-pushforward-eigendecomposition-v1",
                "appearance_initialization": "all-zero-nht-features",
                "standard_3dgs_features_imported": False,
            },
            "pose_ids": list(POSE_IDS),
            "metrics": {
                "per_pose": per_pose_metrics,
                "max_p95_attachment_error_mm": float(
                    max(item["p95_attachment_error_mm"] for item in per_pose_metrics)
                ),
                "all_frames_finite": bool(
                    np.isfinite(deformed.means_m).all()
                    and np.isfinite(deformed.log_scales_m).all()
                ),
                "all_frames_emitted": len(POSE_IDS),
            },
            "files": files,
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    print(f"content_fingerprint={manifest['content_fingerprint']}")


if __name__ == "__main__":
    main()
