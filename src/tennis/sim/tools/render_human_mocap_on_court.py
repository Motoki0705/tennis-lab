from __future__ import annotations

import argparse
import math
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.tennis.geometry.court import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
    sample_camera_position_on_fence,
)
from src.tennis.sim.generator import _rotation_matrix_z
from src.tennis.sim.human_assets import HumanAssetLibrary
from src.visualize.tennis_render import render_pose2d_frame
from src.visualize.video_io import write_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render human motion assets (VitPose-17 3D skeletons) on the tennis court "
            "into a single demonstration video."
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("data/human_mocap/processed"),
        help="Root directory containing converted .npz motion clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output video path (e.g. outputs/human_mocap_demo/demo.mp4)",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=6,
        help="Number of human actors to place on the court.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of the clip in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS and target FPS for resampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling assets and placements.",
    )
    return parser.parse_args()


def _sample_player_origins(
    rng: random.Random,
    count: int,
    *,
    margin_x: float = 0.5,
    margin_y: float = 0.5,
    min_separation: float = 1.5,
) -> list[Tensor]:
    anchors: list[Tensor] = []
    attempts = 0
    margin_x = max(0.0, float(margin_x))
    margin_y = max(0.0, float(margin_y))
    min_sep = max(0.1, float(min_separation))
    x_min = X_MIN + margin_x
    x_max = X_MAX - margin_x
    y_min = Y_MIN + margin_y
    y_max = Y_MAX - margin_y
    max_attempts = max(count * 20, 2000)

    while len(anchors) < count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        candidate = torch.tensor([x, y, 0.0], dtype=torch.float32)
        if all(
            torch.linalg.norm(candidate[:2] - other[:2]) >= min_sep for other in anchors
        ):
            anchors.append(candidate)

    if len(anchors) < count:
        msg = "failed to place all actors on the court without collision"
        raise RuntimeError(msg)
    return anchors


def _build_world_trajectories(
    library: HumanAssetLibrary,
    frames_total: int,
    fps: float,
    rng: random.Random,
    num_actors: int,
) -> list[Tensor]:
    samples = [
        library.sample_sequence(frames_total, fps, rng) for _ in range(num_actors)
    ]
    anchors = _sample_player_origins(rng, num_actors)
    worlds: list[Tensor] = []

    for sample, anchor in zip(samples, anchors, strict=True):
        yaw = rng.uniform(-math.pi, math.pi)
        joints = torch.from_numpy(sample.joints).float()  # (T, 17, 3)
        pelvis = torch.from_numpy(sample.pelvis).float()  # (T, 3)
        rot = _rotation_matrix_z(yaw)
        joints_rot = torch.matmul(joints, rot.t())
        pelvis_rot = torch.matmul(pelvis, rot.t()) + anchor.view(1, 3)
        joints_world = joints_rot + pelvis_rot.unsqueeze(1)
        worlds.append(joints_world)

    return worlds


def _iter_frames(
    worlds: list[Tensor],
    fps: float,
    *,
    rng: random.Random,
) -> Iterator[np.ndarray]:
    court3d = court_keypoints_3d()
    side = rng.choice(["near", "far", "left", "right"])
    cam_center = sample_camera_position_on_fence(rng.random(), side)
    cam = make_look_at_camera(cam_center)

    uv_court, mask_court = project_points(cam, court3d)
    court_pts = uv_court.detach().cpu().numpy()
    court_vis = mask_court.to(torch.uint8).detach().cpu().tolist()

    T = int(worlds[0].shape[0]) if worlds else 0
    for t in range(T):
        player_poses: list[np.ndarray] = []
        player_vis: list[list[int]] = []
        for traj in worlds:
            pts3d = traj[t]  # (17, 3)
            uv, mask = project_points(cam, pts3d)
            uv_np = uv.detach().cpu().numpy()
            mask_np = mask.to(torch.uint8).detach().cpu().tolist()
            player_poses.append(uv_np)
            player_vis.append(mask_np)

        frame = render_pose2d_frame(
            width=cam.w,
            height=cam.h,
            court_points=court_pts,
            court_visibility=court_vis,
            player_poses=player_poses,
            player_pose_visibility=player_vis,
            racket_points=None,
            racket_visibility=None,
        )
        yield frame


def main() -> int:
    args = _parse_args()

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))

    frames_total = max(1, int(round(float(args.duration) * float(args.fps))))
    asset_root = args.asset_root

    library = HumanAssetLibrary(
        root=asset_root,
        min_frames=frames_total,
    )

    worlds = _build_world_trajectories(
        library=library,
        frames_total=frames_total,
        fps=float(args.fps),
        rng=rng,
        num_actors=int(args.num_actors),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_video(
        out_path=args.output,
        frames=_iter_frames(worlds, float(args.fps), rng=rng),
        fps=float(args.fps),
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
