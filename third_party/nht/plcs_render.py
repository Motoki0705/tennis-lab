#!/usr/bin/env python3
"""Render export-bound PLCS people and the NHT court as one Gaussian scene."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from gsplat.rendering import rasterization
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.composition.gaussians import (  # noqa: E402
    GaussianTensorSet,
    compose_gaussians,
    transform_gaussians,
)
from src.synthetic_data_generation.scene_contract import (  # noqa: E402
    SceneCamera,
    SimilarityTransform,
)
from third_party.nht.blcs_render import (  # noqa: E402
    _canonical_sha256,
    _git_dirty,
    _git_head,
    _load_shader,
    _load_tensor_set,
    _relative_file_ref,
    _sha256_file,
)

PLAN_SCHEMA = "tennis_plcs_gaussian_scene_plan_v1"
RENDER_SCHEMA = "tennis_plcs_nht_render_v1"
FRAME_LABEL_SCHEMA = "tennis_plcs_nht_frame_labels_v1"
POSE_IDS = ("canonical", "ready", "forehand")
_ARRAY_NAMES = (
    "instance_ids",
    "positions_court_m",
    "velocities_court_mps",
    "yaw_radians",
    "pose_indices",
    "present",
    "scene_from_asset",
    "camera_uv",
    "camera_depth",
    "camera_geometric_visible",
)
_TENSOR_KEYS = {
    "means",
    "quats",
    "scales",
    "opacities",
    "features",
    "instance_ids",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-indices",
        default="0,4,6,8,11",
        help="Strictly increasing comma-separated frame indices.",
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--instance-alpha-threshold", type=float, default=1.0e-4)
    parser.add_argument("--aov-alpha-consistency-tolerance", type=float, default=0.005)
    return parser.parse_args()


def _file_uri_path(uri: object) -> Path:
    if not isinstance(uri, str):
        raise ValueError("Artifact URI must be a string.")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Only local file artifacts are supported: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _parse_frame_indices(value: str, frame_count: int) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise SystemExit("frame-indices must be comma-separated integers.") from error
    if not indices or any(index < 0 or index >= frame_count for index in indices):
        raise SystemExit(f"frame-indices must lie inside [0, {frame_count}).")
    if indices != tuple(sorted(set(indices))):
        raise SystemExit("frame-indices must be unique and strictly increasing.")
    return indices


def _load_plan(
    root: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray], SceneCamera]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != PLAN_SCHEMA:
        raise ValueError("Unsupported PLCS Gaussian plan.")
    unsigned = dict(manifest)
    declared = unsigned.pop("plan_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise ValueError("PLCS plan fingerprint differs.")
    file_refs = manifest.get("files")
    if not isinstance(file_refs, dict) or set(file_refs) != {
        f"{name}.npy" for name in _ARRAY_NAMES
    }:
        raise ValueError("PLCS plan file inventory differs.")
    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_NAMES:
        relative_name = f"{name}.npy"
        reference = file_refs[relative_name]
        if not isinstance(reference, dict):
            raise ValueError(f"Invalid PLCS plan file reference: {relative_name}.")
        relative = PurePosixPath(str(reference.get("relative_path")))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != reference.get("size_bytes")
            or _sha256_file(path) != reference.get("sha256")
        ):
            raise ValueError(f"PLCS plan file changed: {relative_name}.")
        arrays[name] = np.load(path, allow_pickle=False)

    frame_count = manifest.get("frame_count")
    person_count = manifest.get("person_count")
    if not isinstance(frame_count, int) or not isinstance(person_count, int):
        raise ValueError("PLCS frame/person count is invalid.")
    expected_shapes = {
        "instance_ids": (person_count,),
        "positions_court_m": (frame_count, person_count, 3),
        "velocities_court_mps": (frame_count, person_count, 3),
        "yaw_radians": (frame_count, person_count),
        "pose_indices": (frame_count, person_count),
        "present": (frame_count, person_count),
        "scene_from_asset": (frame_count, person_count, 4, 4),
        "camera_uv": (frame_count, person_count, 2),
        "camera_depth": (frame_count, person_count),
        "camera_geometric_visible": (frame_count, person_count),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("PLCS plan array shape differs from the manifest.")
    if not arrays["present"].all():
        raise ValueError("PLCS renderer refuses a silent missing-person fallback.")
    if not np.array_equal(
        arrays["instance_ids"],
        np.arange(1, person_count + 1),
    ):
        raise ValueError("PLCS instance IDs must be contiguous and one-based.")
    if (
        np.any(arrays["pose_indices"] < 0)
        or np.any(arrays["pose_indices"] >= len(POSE_IDS))
        or not arrays["camera_geometric_visible"].all()
    ):
        raise ValueError("PLCS pose indices or selected-camera visibility differ.")

    raw_camera = manifest.get("camera")
    if not isinstance(raw_camera, dict):
        raise ValueError("PLCS selected camera is missing.")
    camera_keys = {
        "camera_id",
        "source_camera_id",
        "image_uri",
        "source_frame_index",
        "group_id",
        "width",
        "height",
        "intrinsics",
        "camera_to_scene",
    }
    camera = SceneCamera.from_dict(
        {key: raw_camera[key] for key in camera_keys},
    )
    return manifest, arrays, camera


def _load_pose_assets(
    manifest: dict[str, object],
    *,
    appearance_space_sha256: str,
    device: torch.device,
) -> tuple[GaussianTensorSet, ...]:
    pose_assets = manifest.get("pose_assets")
    if not isinstance(pose_assets, list) or len(pose_assets) != len(POSE_IDS):
        raise ValueError("PLCS pose asset inventory differs.")
    result = []
    for expected_index, (expected_pose, record) in enumerate(
        zip(POSE_IDS, pose_assets, strict=True)
    ):
        if (
            not isinstance(record, dict)
            or record.get("pose_index") != expected_index
            or record.get("pose_id") != expected_pose
            or record.get("gaussian_count") != 4096
            or record.get("feature_dim") != 48
        ):
            raise ValueError(f"PLCS pose asset {expected_index} differs.")
        path = _file_uri_path(record.get("uri"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"PLCS pose asset changed: {expected_pose}.")
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or set(payload) != _TENSOR_KEYS:
            raise ValueError(f"PLCS pose tensor keys differ: {expected_pose}.")
        count = int(payload["means"].shape[0])
        result.append(
            GaussianTensorSet(
                means=payload["means"].float(),
                quats=payload["quats"].float(),
                log_scales=payload["scales"].float(),
                opacity_logits=payload["opacities"].float(),
                features=payload["features"].float(),
                instance_ids=torch.full(
                    (count,),
                    1,
                    dtype=torch.int64,
                    device=device,
                ),
                appearance_space_sha256=appearance_space_sha256,
            )
        )
    return tuple(result)


def _similarity_from_matrix(matrix: np.ndarray) -> SimilarityTransform:
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale
    return SimilarityTransform(
        scale=scale,
        rotation=tuple(float(value) for value in rotation.ravel()),
        translation=tuple(float(value) for value in matrix[:3, 3]),
    )


def _camera_tensors(
    camera: SceneCamera,
    *,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
    height = max(2, round(camera.height * width / camera.width))
    scale_x = width / camera.width
    scale_y = height / camera.height
    camera_to_scene = torch.tensor(
        camera.camera_to_scene,
        dtype=torch.float32,
        device=device,
    ).reshape(4, 4)
    intrinsics = torch.tensor(
        camera.intrinsics,
        dtype=torch.float32,
        device=device,
    ).reshape(3, 3)
    intrinsics[0] *= scale_x
    intrinsics[1] *= scale_y
    return camera_to_scene[None], intrinsics[None], height, scale_x, scale_y


def _bbox(mask: torch.Tensor) -> list[int] | None:
    y, x = torch.where(mask)
    if x.numel() == 0:
        return None
    return [
        int(x.min()),
        int(y.min()),
        int(x.max()) + 1,
        int(y.max()) + 1,
    ]


def _render_frame(
    *,
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
    frame_index: int,
    background: GaussianTensorSet,
    pose_assets: tuple[GaussianTensorSet, ...],
    shader: torch.nn.Module,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    instance_alpha_threshold: float,
    alpha_consistency_tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    person_count = int(manifest["person_count"])
    transformed = []
    for person_index in range(person_count):
        instance_id = int(arrays["instance_ids"][person_index])
        pose_index = int(arrays["pose_indices"][frame_index, person_index])
        pose = pose_assets[pose_index]
        pose_with_identity = GaussianTensorSet(
            means=pose.means,
            quats=pose.quats,
            log_scales=pose.log_scales,
            opacity_logits=pose.opacity_logits,
            features=pose.features,
            instance_ids=torch.full_like(pose.instance_ids, instance_id),
            appearance_space_sha256=pose.appearance_space_sha256,
        )
        transform = _similarity_from_matrix(
            arrays["scene_from_asset"][frame_index, person_index]
        )
        transformed.append(transform_gaussians(pose_with_identity, transform))
    composed = compose_gaussians(background, transformed)

    with torch.no_grad():
        rendered_features, alpha, _ = rasterization(
            means=composed.means,
            quats=composed.quats,
            scales=torch.exp(composed.log_scales),
            opacities=torch.sigmoid(composed.opacity_logits),
            colors=composed.features,
            viewmats=torch.linalg.inv(camera_to_scene),
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=None,
            near_plane=0.01,
            far_plane=1.0e10,
            render_mode="RGB+ED",
            packed=False,
            tile_size=16,
            with_ut=True,
            with_eval3d=True,
            nht=True,
            center_ray_mode=False,
            ray_dir_scale=shader.ray_dir_scale,
        )
        rgb, extras = shader(rendered_features)
        if extras is None or extras.shape[-1] != 1:
            raise RuntimeError("NHT RGB+ED did not return one expected-depth channel.")
        depth = extras
        rgb = (rgb[..., :3] + (1.0 - alpha)).clamp(0.0, 1.0)
        one_hot = torch.nn.functional.one_hot(
            composed.instance_ids,
            num_classes=person_count + 1,
        ).to(dtype=composed.features.dtype)
        contribution, aov_alpha, _ = rasterization(
            means=composed.means,
            quats=composed.quats,
            scales=torch.exp(composed.log_scales),
            opacities=torch.sigmoid(composed.opacity_logits),
            colors=one_hot,
            viewmats=torch.linalg.inv(camera_to_scene),
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=None,
            near_plane=0.01,
            far_plane=1.0e10,
            render_mode="RGB",
            packed=False,
            tile_size=16,
            with_ut=True,
            with_eval3d=True,
            nht=False,
        )
        torch.cuda.synchronize()

    contribution_sum_error = float(
        torch.abs(contribution.sum(dim=-1, keepdim=True) - aov_alpha).max()
    )
    nht_alpha_error = float(torch.abs(aov_alpha - alpha).max())
    if contribution_sum_error > alpha_consistency_tolerance:
        raise RuntimeError("PLCS instance contributions do not sum to AOV alpha.")
    if nht_alpha_error > alpha_consistency_tolerance:
        raise RuntimeError("PLCS AOV alpha differs from NHT alpha.")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (rgb, alpha, depth, contribution, aov_alpha)
    ):
        raise RuntimeError(f"PLCS frame {frame_index} contains non-finite values.")

    masks = contribution[..., 1:] >= instance_alpha_threshold
    segmentation = torch.argmax(contribution, dim=-1).to(torch.int32)
    segmentation = torch.where(
        aov_alpha[..., 0] >= instance_alpha_threshold,
        segmentation,
        torch.full_like(segmentation, -1),
    )
    identities = manifest["identities"]
    pose_records = manifest["pose_assets"]
    labels = []
    for person_index in range(person_count):
        identity = identities[person_index]
        pose_index = int(arrays["pose_indices"][frame_index, person_index])
        instance_id = int(arrays["instance_ids"][person_index])
        mask = masks[0, ..., person_index]
        visible_pixels = int(mask.sum())
        labels.append(
            {
                "identity_id": identity["identity_id"],
                "instance_id": instance_id,
                "present": bool(arrays["present"][frame_index, person_index]),
                "pose_index": pose_index,
                "pose_id": pose_records[pose_index]["pose_id"],
                "pose_asset_sha256": pose_records[pose_index]["sha256"],
                "position_court_m": arrays["positions_court_m"][
                    frame_index, person_index
                ].tolist(),
                "velocity_court_mps": arrays["velocities_court_mps"][
                    frame_index, person_index
                ].tolist(),
                "yaw_radians": float(
                    arrays["yaw_radians"][frame_index, person_index]
                ),
                "scene_from_asset": arrays["scene_from_asset"][
                    frame_index, person_index
                ].tolist(),
                "projected_root_uv_render_pixels": [
                    float(arrays["camera_uv"][frame_index, person_index, 0] * scale_x),
                    float(arrays["camera_uv"][frame_index, person_index, 1] * scale_y),
                ],
                "camera_depth": float(
                    arrays["camera_depth"][frame_index, person_index]
                ),
                "geometric_visible": bool(
                    arrays["camera_geometric_visible"][frame_index, person_index]
                ),
                "exact_visible_pixel_count": visible_pixels,
                "exact_bbox_xyxy_exclusive": _bbox(mask),
                "exact_contribution_mass": float(
                    contribution[0, ..., instance_id].sum()
                ),
                "render_visible": visible_pixels > 0,
            }
        )
    arrays_out = {
        "rgb": rgb[0].mul(255).round().to(torch.uint8).cpu().numpy(),
        "alpha": alpha[0, ..., 0].float().cpu().numpy(),
        "depth": depth[0, ..., 0].float().cpu().numpy(),
        "instance_contribution": contribution[0].float().cpu().numpy(),
        "instance_mask": masks[0].cpu().numpy(),
        "instance_segmentation": segmentation[0].cpu().numpy(),
    }
    frame_labels: dict[str, object] = {
        "schema": FRAME_LABEL_SCHEMA,
        "plan_fingerprint": manifest["plan_fingerprint"],
        "scene_id": manifest["scene"]["scene_id"],
        "mode": manifest["mode"],
        "frame_index": frame_index,
        "camera_id": manifest["camera"]["camera_id"],
        "resolution": [width, height],
        "composed_gaussian_count": composed.gaussian_count,
        "renderer_api_call_count": 2,
        "instances": labels,
        "instance_aov": {
            "channel_instance_ids": list(range(person_count + 1)),
            "background_channel": 0,
            "instance_alpha_threshold": instance_alpha_threshold,
            "contribution_sum_vs_aov_alpha_max_abs": contribution_sum_error,
            "aov_alpha_vs_nht_alpha_max_abs": nht_alpha_error,
        },
        "rgb_overlay_used": False,
        "all_finite": True,
    }
    return arrays_out, frame_labels


def _write_frame(
    root: Path,
    *,
    frame_index: int,
    arrays: dict[str, np.ndarray],
    labels: dict[str, object],
) -> dict[str, object]:
    frame_dir = root / "frames" / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True)
    paths = {
        name: frame_dir / ("rgb.png" if name == "rgb" else f"{name}.npy")
        for name in arrays
    }
    Image.fromarray(arrays["rgb"], mode="RGB").save(paths["rgb"])
    for name, array in arrays.items():
        if name != "rgb":
            np.save(paths[name], array, allow_pickle=False)
    labels_path = frame_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    paths["labels"] = labels_path
    return {
        "frame_index": frame_index,
        **{
            name: _relative_file_ref(root, path)
            for name, path in sorted(paths.items())
        },
    }


def _verify_output(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != RENDER_SCHEMA:
        raise ValueError("Unsupported PLCS render manifest.")
    unsigned = dict(manifest)
    declared = unsigned.pop("render_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise ValueError("PLCS render fingerprint differs.")
    for frame in manifest["frames"]:
        for name, reference in frame.items():
            if name == "frame_index":
                continue
            relative = PurePosixPath(reference["relative_path"])
            path = (root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(root)
                or not path.is_file()
                or path.stat().st_size != reference["size_bytes"]
                or _sha256_file(path) != reference["sha256"]
            ):
                raise ValueError(f"PLCS render output changed: {relative}.")
    return {
        "render_fingerprint": declared,
        "frame_count": len(manifest["frames"]),
        "all_people_visible": manifest["visibility"]["all_people_visible"],
        "aov_alpha_vs_nht_alpha_max_abs": manifest["visibility"][
            "aov_alpha_vs_nht_alpha_max_abs"
        ],
    }


def main() -> None:
    args = _parse_args()
    plan_dir = args.plan_dir.resolve()
    composition_path = args.background_composition.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output: {output_dir}")
    if args.width <= 1:
        raise SystemExit("width must be greater than one.")
    if not 0.0 < args.instance_alpha_threshold <= 1.0:
        raise SystemExit("instance-alpha-threshold must lie in (0, 1].")
    if args.aov_alpha_consistency_tolerance < 0.0:
        raise SystemExit("aov-alpha-consistency-tolerance must be non-negative.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    manifest, plan_arrays, camera = _load_plan(plan_dir)
    frame_indices = _parse_frame_indices(
        args.frame_indices,
        int(manifest["frame_count"]),
    )
    if _sha256_file(composition_path) != manifest["background_composition"]["sha256"]:
        raise ValueError("PLCS plan references a different background composition.")
    composition = load_gaussian_scene_manifest(composition_path)
    if (
        composition.composition_fingerprint
        != manifest["background_composition"]["composition_fingerprint"]
    ):
        raise ValueError("PLCS background composition fingerprint differs.")
    background_asset = composition.background
    verify_local_gaussian_asset(background_asset)
    appearance_space = manifest["avatar_nht_manifest"]["appearance_space_sha256"]
    if background_asset.appearance_space_sha256 != appearance_space:
        raise ValueError("PLCS avatar and background appearance spaces differ.")

    gsplat_path = Path(__file__).resolve().parent / "upstream" / "gsplat"
    renderer_commit = _git_head(gsplat_path)
    if renderer_commit != composition.renderer_commit or _git_dirty(gsplat_path):
        raise ValueError("PLCS renderer commit is different or dirty.")

    device = torch.device("cuda:0")
    background = _load_tensor_set(
        background_asset,
        instance_id=0,
        device=device,
    )
    pose_assets = _load_pose_assets(
        manifest,
        appearance_space_sha256=appearance_space,
        device=device,
    )
    if any(asset.feature_dim != background.feature_dim for asset in pose_assets):
        raise ValueError("PLCS pose and background feature dimensions differ.")
    shader, shader_config = _load_shader(
        background_asset.appearance_payload,
        feature_dim=background.feature_dim,
        device=device,
    )
    camera_to_scene, intrinsics, height, scale_x, scale_y = _camera_tensors(
        camera,
        width=args.width,
        device=device,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        frame_records = []
        visibility_counts = {
            int(instance_id): 0 for instance_id in plan_arrays["instance_ids"]
        }
        nht_alpha_errors = []
        contribution_errors = []
        for frame_index in frame_indices:
            arrays_out, labels = _render_frame(
                manifest=manifest,
                arrays=plan_arrays,
                frame_index=frame_index,
                background=background,
                pose_assets=pose_assets,
                shader=shader,
                camera_to_scene=camera_to_scene,
                intrinsics=intrinsics,
                width=args.width,
                height=height,
                scale_x=scale_x,
                scale_y=scale_y,
                instance_alpha_threshold=args.instance_alpha_threshold,
                alpha_consistency_tolerance=args.aov_alpha_consistency_tolerance,
            )
            for instance in labels["instances"]:
                if instance["render_visible"]:
                    visibility_counts[int(instance["instance_id"])] += 1
            nht_alpha_errors.append(
                labels["instance_aov"]["aov_alpha_vs_nht_alpha_max_abs"]
            )
            contribution_errors.append(
                labels["instance_aov"][
                    "contribution_sum_vs_aov_alpha_max_abs"
                ]
            )
            frame_records.append(
                _write_frame(
                    temporary,
                    frame_index=frame_index,
                    arrays=arrays_out,
                    labels=labels,
                )
            )
        all_people_visible = all(
            count == len(frame_indices) for count in visibility_counts.values()
        )
        render_manifest: dict[str, object] = {
            "schema": RENDER_SCHEMA,
            "mode": manifest["mode"],
            "plan": {
                "sha256": _sha256_file(plan_dir / "manifest.json"),
                "plan_fingerprint": manifest["plan_fingerprint"],
            },
            "background_composition": {
                "sha256": _sha256_file(composition_path),
                "composition_fingerprint": composition.composition_fingerprint,
            },
            "camera_id": camera.camera_id,
            "resolution": [args.width, height],
            "frame_indices": list(frame_indices),
            "frames": frame_records,
            "renderer": {
                "backend": "nht-gsplat",
                "commit": renderer_commit,
                "api_calls_per_frame": 2,
                "api_call_count": 2 * len(frame_indices),
                "shader_config": shader_config,
            },
            "visibility": {
                "method": "exact-eval3d-instance-contribution-aov-v1",
                "instance_alpha_threshold": args.instance_alpha_threshold,
                "aov_alpha_consistency_tolerance": (
                    args.aov_alpha_consistency_tolerance
                ),
                "render_visible_frame_counts": {
                    str(key): value for key, value in visibility_counts.items()
                },
                "all_people_visible": all_people_visible,
                "aov_alpha_vs_nht_alpha_max_abs": max(nht_alpha_errors),
                "contribution_sum_vs_aov_alpha_max_abs": max(
                    contribution_errors
                ),
                "exact_per_pixel_instance_mask": True,
            },
            "rgb_overlay_used": False,
            "all_finite": True,
        }
        render_manifest["render_fingerprint"] = _canonical_sha256(render_manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(render_manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(output_dir)
        print(json.dumps(_verify_output(output_dir), indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
