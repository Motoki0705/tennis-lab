"""Compare pinned PLCS avatar control rules on the licensed local SMPL-X model.

This probe is deliberately geometry-only. It does not claim to reproduce the
appearance networks or training procedures of GaussianAvatar or HUGS. Instead,
it measures the two reusable control boundaries needed before an avatar can be
fitted into the repository's independent NHT renderer:

* fixed per-Gaussian SMPL-X LBS weights, following GaussianAvatar's query-LBS
  control boundary; and
* top-k template-vertex transform blending, following HUGS.

The output directory is immutable and contains sufficient arrays and hashes to
repeat the comparison. The licensed SMPL-X model is read in place and is never
copied into the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.submodules.vendor.gvhmr.body_model.smplx_lite import (
    SmplxLite,
    batch_rigid_transform_v2,
)
from src.synthetic_data_generation.plcs.avatar_control import (
    apply_joint_linear_blend_skinning,
    apply_vertex_transform_blend,
    embed_points_on_posed_mesh,
    hugs_topk_neighbor_blend,
    interpolate_face_attributes,
)
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

SCHEMA = "plcs_avatar_control_probe_v1"
FRAME_COUNT = 9
ATTACHMENT_COUNT = 512
SEED = 20260728

OFFICIAL_METHODS = {
    "gaussianavatar_query_lbs": {
        "paper": "GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Hu_GaussianAvatar_Towards_Realistic_Human_Avatar_Modeling_from_a_Single_Video_CVPR_2024_paper.html",
        "official_code": "https://github.com/aipixel/GaussianAvatar",
        "commit": "d981c62238ef64e89dcc04719d2ebbb4758b080a",
    },
    "hugs_topk_lbs": {
        "paper": "HUGS: Human Gaussian Splats",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Kocabas_HUGS_Human_Gaussian_Splats_CVPR_2024_paper.html",
        "official_code": "https://github.com/apple/ml-hugs",
        "commit": "b65721a5946771053e4f1d0d68d06199bc1d8c07",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _motion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = torch.linspace(0.0, 2.0 * torch.pi, FRAME_COUNT, dtype=torch.float32)
    body = torch.zeros((1, FRAME_COUNT, 63), dtype=torch.float32)

    def joint(index: int, xyz: tuple[torch.Tensor | float, ...]) -> None:
        for axis, value in enumerate(xyz):
            body[0, :, index * 3 + axis] = value

    # A compact tennis-like sequence: stride, knee flexion, shoulder turn,
    # asymmetric racket-arm swing, and counterbalancing non-racket arm.
    joint(0, (0.18 * torch.sin(phase), 0.0, 0.10 * torch.cos(phase)))
    joint(1, (-0.16 * torch.sin(phase), 0.0, -0.08 * torch.cos(phase)))
    joint(3, (0.35 + 0.20 * torch.clamp(torch.sin(phase), min=0.0), 0.0, 0.0))
    joint(4, (0.28 + 0.18 * torch.clamp(-torch.sin(phase), min=0.0), 0.0, 0.0))
    joint(11, (0.0, 0.18 * torch.sin(phase), 0.20 * torch.cos(phase)))
    joint(15, (0.15 * torch.sin(phase), 0.0, -0.35 * torch.cos(phase)))
    joint(16, (-0.10 * torch.sin(phase), 0.0, 0.30 * torch.cos(phase)))
    joint(17, (0.20 + 0.45 * torch.sin(phase), 0.10, -0.45 * torch.cos(phase)))
    joint(18, (0.35 - 0.20 * torch.sin(phase), -0.08, 0.25 * torch.cos(phase)))
    joint(19, (0.10 * torch.sin(phase), 0.0, 0.30 * torch.sin(phase)))
    joint(20, (-0.08 * torch.sin(phase), 0.0, -0.20 * torch.sin(phase)))

    global_orient = torch.zeros((1, FRAME_COUNT, 3), dtype=torch.float32)
    global_orient[0, :, 1] = 0.22 * torch.sin(phase)
    translations = torch.stack(
        (
            torch.linspace(-0.7, 0.7, FRAME_COUNT),
            0.10 * torch.sin(phase),
            torch.zeros_like(phase),
        ),
        dim=-1,
    )[None]
    betas = torch.zeros((1, FRAME_COUNT, 10), dtype=torch.float32)
    betas[..., 1] = -0.35
    betas[..., 2] = 0.20
    return body, betas, global_orient, translations


def _joint_transforms(
    model: SmplxLite,
    *,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
) -> torch.Tensor:
    other = model.other_default_pose.expand(*body_pose.shape[:-1], -1)
    full_pose = torch.cat((global_orient, body_pose, other), dim=-1)
    rotations = axis_angle_to_matrix(full_pose.reshape(1, FRAME_COUNT, 55, 3))
    joints = model.get_skeleton(betas)
    live = batch_rigid_transform_v2(rotations, joints, model.parents)[1]

    zero_body = torch.zeros((1, 1, 63), dtype=body_pose.dtype)
    zero_global = torch.zeros((1, 1, 3), dtype=body_pose.dtype)
    canonical_other = model.other_default_pose.expand(1, 1, -1)
    canonical_pose = torch.cat((zero_global, zero_body, canonical_other), dim=-1)
    canonical_rotations = axis_angle_to_matrix(canonical_pose.reshape(1, 1, 55, 3))
    canonical_joints = model.get_skeleton(betas[:, :1])
    canonical = batch_rigid_transform_v2(
        canonical_rotations,
        canonical_joints,
        model.parents,
    )[1]
    return live @ torch.linalg.inv(canonical)


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error_mm = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    per_frame = [
        {
            "frame": frame,
            "mean_mm": float(values.mean()),
            "p95_mm": float(np.quantile(values, 0.95)),
            "max_mm": float(values.max()),
        }
        for frame, values in enumerate(error_mm)
    ]
    return {
        "finite": bool(np.isfinite(prediction).all()),
        "mean_mm": float(error_mm.mean()),
        "median_mm": float(np.median(error_mm)),
        "p95_mm": float(np.quantile(error_mm, 0.95)),
        "max_mm": float(error_mm.max()),
        "per_frame": per_frame,
    }


def _content_fingerprint(files: dict[str, str], metrics: object) -> str:
    payload = {"files": files, "metrics": metrics, "schema": SCHEMA}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def run_probe(*, model_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SmplxLite(model_path=model_path).eval()
    body, betas, global_orient, translations = _motion()

    with torch.inference_mode():
        posed = model(body, betas, global_orient, translations)[0]
        canonical = model(
            torch.zeros((1, 1, 63), dtype=torch.float32),
            betas[:, :1],
            torch.zeros((1, 1, 3), dtype=torch.float32),
            torch.zeros((1, 1, 3), dtype=torch.float32),
        )[0, 0]
        joint_transforms = _joint_transforms(
            model,
            body_pose=body,
            betas=betas,
            global_orient=global_orient,
        )[0]

    faces = np.asarray(model.faces, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    face_indices = rng.choice(
        faces.shape[0],
        size=ATTACHMENT_COUNT,
        replace=False,
    ).astype(np.int64)
    barycentric = rng.dirichlet(np.ones(3), size=ATTACHMENT_COUNT)
    canonical_np = canonical.cpu().numpy().astype(np.float64)
    canonical_points = embed_points_on_posed_mesh(
        canonical_np[None],
        faces=faces,
        face_indices=face_indices,
        barycentric_coordinates=barycentric,
    )[0]
    target = embed_points_on_posed_mesh(
        posed.cpu().numpy().astype(np.float64),
        faces=faces,
        face_indices=face_indices,
        barycentric_coordinates=barycentric,
    )
    lbs_weights = model.lbs_weights.cpu().numpy().astype(np.float64)
    point_weights = interpolate_face_attributes(
        lbs_weights,
        faces=faces,
        face_indices=face_indices,
        barycentric_coordinates=barycentric,
    )
    transforms_np = joint_transforms.cpu().numpy().astype(np.float64)
    translations_np = translations[0].cpu().numpy().astype(np.float64)

    gaussianavatar_prediction = apply_joint_linear_blend_skinning(
        canonical_points,
        point_joint_weights=point_weights,
        joint_transforms=transforms_np,
        translations_m=translations_np,
    )
    neighbor_blend = hugs_topk_neighbor_blend(
        canonical_points,
        template_vertices_m=canonical_np,
        vertex_joint_weights=lbs_weights,
    )
    vertex_transforms = np.einsum(
        "vj,tjkl->tvkl",
        lbs_weights,
        transforms_np,
    )
    hugs_prediction = apply_vertex_transform_blend(
        canonical_points,
        vertex_transforms=vertex_transforms,
        neighbor_blend=neighbor_blend,
        translations_m=translations_np,
    )

    metrics = {
        "gaussianavatar_query_lbs": _error_metrics(
            gaussianavatar_prediction,
            target,
        ),
        "hugs_topk_lbs": _error_metrics(hugs_prediction, target),
    }
    for values in metrics.values():
        values["screening_gate"] = {
            "mean_mm_at_most": 30.0,
            "p95_mm_at_most": 80.0,
            "passed": bool(
                values["finite"]
                and values["mean_mm"] <= 30.0
                and values["p95_mm"] <= 80.0
            ),
            "scope": "geometry-control screening only; not P4 acceptance",
        }

    output_dir.mkdir(parents=True)
    arrays = {
        "face_indices.npy": face_indices,
        "barycentric_coordinates.npy": barycentric,
        "canonical_points_m.npy": canonical_points,
        "point_joint_weights.npy": point_weights,
        "hugs_neighbor_indices.npy": neighbor_blend.indices,
        "hugs_neighbor_weights.npy": neighbor_blend.weights,
        "body_pose_axis_angle.npy": body[0].numpy(),
        "betas.npy": betas[0].numpy(),
        "global_orient_axis_angle.npy": global_orient[0].numpy(),
        "translations_m.npy": translations_np,
        "joint_transforms.npy": transforms_np,
        "target_mesh_attachments_m.npy": target,
        "gaussianavatar_query_lbs_m.npy": gaussianavatar_prediction,
        "hugs_topk_lbs_m.npy": hugs_prediction,
    }
    for filename, value in arrays.items():
        np.save(output_dir / filename, value, allow_pickle=False)
    (output_dir / "metrics.json").write_bytes(_json_bytes(metrics))
    files = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "frame_count": FRAME_COUNT,
        "attachment_count": ATTACHMENT_COUNT,
        "smplx_model": {
            "path": str(model_path.resolve()),
            "sha256": _sha256(model_path),
            "copied_into_artifact": False,
        },
        "official_methods": OFFICIAL_METHODS,
        "method_scope": (
            "Geometry-control comparison only. Upstream appearance networks and "
            "standard-3DGS feature tensors are not imported into NHT."
        ),
        "coco17_policy": (
            "COCO17 is an output label or an explicit IK input; it is not silently "
            "inverted into SMPL-X pose because the inverse is underdetermined."
        ),
        "files": files,
        "metrics": metrics,
        "content_fingerprint": _content_fingerprint(files, metrics),
    }
    (output_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_probe(model_path=args.model, output_dir=args.output)
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    print(f"content_fingerprint={manifest['content_fingerprint']}")


if __name__ == "__main__":
    main()
