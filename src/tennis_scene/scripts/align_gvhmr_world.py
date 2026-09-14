"""Fit and compare gravity-fixed GVHMR world similarities to saved PLCS tracks."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.submodules.configuration import BundledModelAssetPaths
from src.tennis_scene.motion_alignment.artifacts import (
    load_world_motion,
    transform_smpl_parameters,
)
from src.tennis_scene.motion_alignment.experiment import (
    compare_track,
    match_player,
    sha256,
    write_json,
)
from src.tennis_scene.motion_alignment.similarity import SimilarityConfig
from src.tennis_scene.motion_alignment.visualization import (
    plot_diagnostics,
    render_comparison,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

# Declared before any side effect: every CLI path argument and the role that
# grants write authority for it. ``clip_dir``/``motion_dir`` are repository data,
# ``body_models_dir`` is a licensed external asset, and the output directory is
# caller-owned and may not exist yet.
PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.gvhmr_alignment",
    fields=(
        BoundaryPathField(
            "clip_dir",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "motion_dir",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "asset_repository_root",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "body_models_dir",
            PathRole.EXTERNAL_ASSET,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "output_dir",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument(
        "--motion-dir",
        type=Path,
        required=True,
        help="Directory containing cam*.gvhmr.npz global SMPL archives",
    )
    parser.add_argument("--asset-repository-root", type=Path, required=True)
    parser.add_argument("--body-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma-position", type=float, default=0.5)
    parser.add_argument("--sigma-heading-deg", type=float, default=30.0)
    parser.add_argument("--heading-weight", type=float, default=1.0)
    parser.add_argument("--scale-prior", type=float, default=1.0)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    # Resolve every explicit CLI path before validation. Relative arguments keep
    # resolving against the process CWD, exactly as the previous CLI did; the
    # boundary then enforces the absolute role contract and existence checks.
    asset_root = args.asset_repository_root.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=asset_root,
            data_root=(asset_root / "data").resolve(strict=False),
            checkpoint_root=(asset_root / "ckpt").resolve(strict=False),
            artifact_root=(asset_root / "outputs").resolve(strict=False),
            output_root=output_dir,
            cache_root=(asset_root / ".cache").resolve(strict=False),
            external_asset_root=(asset_root / "third_party").resolve(strict=False),
        )
    )
    paths = PATH_BOUNDARY.validate(
        {
            "clip_dir": args.clip_dir.expanduser().resolve(strict=False),
            "motion_dir": args.motion_dir.expanduser().resolve(strict=False),
            "asset_repository_root": asset_root,
            "body_models_dir": args.body_models_dir.expanduser().resolve(strict=False),
            "output_dir": output_dir,
        },
        resolver=resolver,
    )
    clip = paths.declared("clip_dir").path
    motion_dir = paths.declared("motion_dir").path
    output = paths.declared("output_dir").path
    asset_root = paths.declared("asset_repository_root").path
    body_models_dir = paths.declared("body_models_dir").path
    torch.set_num_threads(args.cpu_threads)
    if output.exists():
        raise FileExistsError(
            f"Use a new output directory to preserve previous results: {output}"
        )
    config = SimilarityConfig(
        sigma_position=args.sigma_position,
        sigma_heading=float(np.deg2rad(args.sigma_heading_deg)),
        heading_weight=args.heading_weight,
        scale_prior=args.scale_prior,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    manifest = json.loads((clip / "clip.json").read_text())
    scene_path = clip / "annotations/tennis_scene/scene.npz"
    metadata_path = scene_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    # Read only the arrays needed by the experiment, excluding the legacy mesh.
    with np.load(scene_path, allow_pickle=False) as archive:
        scene = {
            key: archive[key]
            for key in (
                "player_position",
                "player_yaw",
                "human_kp_2d",
                "human_kp_vis",
                "court_vis",
                "fps",
                "num_frames",
                "width",
                "height",
            )
        }
    frames, fps = int(scene["num_frames"]), float(scene["fps"])
    image_size = (int(scene["width"]), int(scene["height"]))
    if frames != manifest["num_frames"] or not np.isclose(
        fps, manifest["fps"], atol=1e-6
    ):
        raise ValueError("Clip manifest and SceneResult frame timing disagree")
    if metadata.get("dataset_clip_id") != manifest["clip_id"]:
        raise ValueError("SceneResult belongs to another clip")
    camera_ids = metadata["camera_ids"]
    if camera_ids != manifest["camera_ids"]:
        raise ValueError("Scene and clip camera ordering differ")
    sources = sorted(motion_dir.glob("*.gvhmr.npz"))
    if not sources:
        raise FileNotFoundError(
            f"No global SMPL archives found under {motion_dir}; local GVHMR JSON does not contain world transl"
        )
    vendor = asset_root / "src/submodules/vendor/gvhmr"
    assets = BundledModelAssetPaths(
        hmr2_mean_params=vendor / "hmr2/smpl_mean_params.npz",
        smplx_to_smpl=vendor / "body_model/data/smplx2smpl_sparse.pt",
        smpl_coco17_regressor=vendor / "body_model/data/smpl_coco17_J_regressor.pt",
        smplx_verts437=vendor / "body_model/data/smplx_verts437.pt",
        smpl_neutral_joint_regressor=vendor
        / "body_model/data/smpl_neutral_J_regressor.pt",
    )
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    metrics: dict[str, Any] = {
        "schema_version": "gvhmr_plcs_similarity_experiment_v1",
        "clip_id": manifest["clip_id"],
        "fps": fps,
        "frame_count": frames,
        "config": asdict(config),
        "inputs": {
            str(path): sha256(path)
            for path in (clip / "clip.json", scene_path, metadata_path)
        },
        "interpretation": {
            "position_target": "SceneResult player_position: renderer's regressed SMPL pelvis anchor; PLCS training calls raw AMASS transl pelvis, an upstream semantic ambiguity",
            "source_root": "neutral SMPL J_regressor[0] on reconstructed global mesh, not raw transl",
            "confidence": "clipped 2D detection confidence proxy; not calibrated PLCS uncertainty",
            "reprojection": "saved approximate single-plane camera calibration, no distortion correction; detection agreement, not ground-truth 3D accuracy",
            "baseline": "controlled comparison using same world-source articulation, with existing renderer's per-frame PLCS placement rule",
            "contact": "shared source-low-speed ankle proxy, not verified ground contact",
        },
        "players": {},
    }
    arrays_by_player = {}
    for source in sources:
        camera_id = source.name.removesuffix(".gvhmr.npz")
        if camera_id not in camera_ids:
            raise ValueError(f"Source camera is not in this clip: {camera_id}")
        camera = camera_ids.index(camera_id)
        frozen_source = output / "inputs" / source.name
        shutil.copyfile(source, frozen_source)
        with np.load(frozen_source, allow_pickle=False) as raw:
            source_meta = json.loads(str(raw["metadata_json"].item()))
            if not source_meta["source_id"].startswith(
                f"{manifest['clip_id']}:{camera_id}:"
            ):
                raise ValueError("Global source belongs to another clip or camera")
            if raw["observed_mask"].shape != (frames,) or not np.isclose(
                source_meta["native_fps"], fps, atol=1e-6
            ):
                raise ValueError("Global source timing does not match SceneResult")
            player, association_errors = match_player(
                raw["keypoints_2d_px"],
                raw["observed_mask"],
                scene["human_kp_2d"][:, camera],
                image_size,
            )
        if player in arrays_by_player:
            raise ValueError(
                f"Multiple sources map to player {player}; choose one source per player explicitly"
            )
        print(
            f"CPU reconstruction: {camera_id} -> player {player}, {frames} frames",
            flush=True,
        )
        world = load_world_motion(
            frozen_source,
            body_models_dir=body_models_dir,
            bundled_assets=assets,
        )
        player_metrics, player_arrays, transforms = compare_track(
            source_root=world.root_position,
            source_rotation=world.rotation,
            source_joints=world.joints_coco,
            source_confidence=world.confidence,
            observed=world.observed,
            target_position=scene["player_position"][player].astype(np.float64),
            target_yaw=scene["player_yaw"][player].astype(np.float64),
            observations=scene["human_kp_2d"][player].astype(np.float64),
            visibility=scene["human_kp_vis"][player].astype(np.float64),
            court_visibility=scene["court_vis"].astype(np.float64),
            camera_fits=metadata["court_reference"]["camera_fits"],
            fps=fps,
            image_size=image_size,
            config=config,
        )
        player_metrics.update(
            {
                "camera_id": camera_id,
                "source_id": source_meta["source_id"],
                "source_sha256": sha256(frozen_source),
                "association_median_normalized_hip_errors": association_errors,
                "source_confidence_diagnostics": asdict(world.confidence_diagnostics),
            }
        )
        for mode, transform in transforms.items():
            params = transform_smpl_parameters(
                body_pose=world.raw_body_pose,
                betas=world.raw_betas,
                global_orient=world.raw_global_orient,
                transl=world.raw_transl,
                rest_pelvis=world.rest_pelvis,
                scale=transform.scale,
                yaw=transform.yaw,
                translation=transform.translation,
            )
            export_metadata = {
                "schema_version": "scaled_smpl_court_v1",
                "source_id": source_meta["source_id"],
                "coordinate_system": "right_handed_court_z_up_m",
                "fps": fps,
                "render_contract": "world_vertices = body_scale * SMPL(body_pose, betas, global_orient, transl=0).vertices + transl; ordinary SMPL transl alone is insufficient",
            }
            np.savez_compressed(
                output / f"player_{player}_{mode}.npz",
                **params,
                root_position=player_arrays[f"{mode}_position"],
                joints_smpl=transform.apply(world.joints_smpl),
                joints_coco=player_arrays[f"{mode}_joints"],
                rotation=player_arrays[f"{mode}_rotation"],
                observed=world.observed,
                metadata_json=np.asarray(json.dumps(export_metadata)),
            )
        metrics["players"][str(player)] = player_metrics
        arrays_by_player[player] = player_arrays
        print(
            f"Player {player}: scales fixed=1, free={transforms['free'].scale:.4f}",
            flush=True,
        )
    if sorted(arrays_by_player) != list(range(len(scene["player_position"]))):
        raise ValueError("Global sources do not cover every PLCS player")
    arrays = {
        key: np.stack(
            [arrays_by_player[player][key] for player in sorted(arrays_by_player)]
        )
        for key in next(iter(arrays_by_player.values()))
    }
    np.savez_compressed(output / "comparison_arrays.npz", **arrays)
    write_json(output / "metrics.json", metrics)
    plot_diagnostics(output, arrays, fps)
    if args.render_video:
        print("Rendering comparison video", flush=True)
        render_comparison(output, arrays, fps)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
