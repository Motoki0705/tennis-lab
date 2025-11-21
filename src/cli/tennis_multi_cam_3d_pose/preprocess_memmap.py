"""CLI for converting tennis JSON scenes into npz/memmap arrays.

This script reads the dataset structure produced by ``build_tennis_dataset.py``
and materializes per-scene npz files under ``arrays/<split>`` that contain
fully normalized 2D/3D tensors. At training time, ``TennisSceneWindowDataset``
can load from these npz files with ``use_memmap=true`` to avoid repeated JSON
parsing and Python loops.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tennis.geometry.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)


def _decompose_pose_for_v2(pose_3d: np.ndarray) -> dict[str, np.ndarray]:
    """絶対座標ポーズからv2用要素を分解.

    Args:
        pose_3d (np.ndarray): [T, M, J, 3] 絶対座標ポーズ（正規化済み）

    Returns:
        dict[str, np.ndarray]: v2用GTデータ
            - canonical_pose_gt: [T, M, J, 3] ルート相対座標
            - root_trans_gt: [T, M, 3] ルート位置（x, y, z）
            - root_rot_gt: [T, M, 2] ルート回転（cos, sin）
            - global_pose_gt: [T, M, J, 3] 絶対座標（元データ）

    """
    T, M, J, _ = pose_3d.shape

    # 1. root_trans: ルート関節（腰）の絶対位置 [T, M, 3]
    # 最初の関節（インデックス0）をルートとして使用
    root_trans = pose_3d[:, :, 0, :].copy()  # [T, M, 3]

    # 2. ルート相対座標を計算 [T, M, J, 3]
    pose_rel = pose_3d - root_trans[:, :, None, :]

    # 3. root_rot: 肩ベクトルから向きを計算 [T, M, 2] (cos, sin)
    # 左肩（インデックス11）と右肩（インデックス12）のベクトルから向きを計算
    left_shoulder = pose_3d[:, :, 11, :]  # [T, M, 3]
    right_shoulder = pose_3d[:, :, 12, :]  # [T, M, 3]

    # 肩ベクトルを計算
    shoulder_vector = right_shoulder - left_shoulder  # [T, M, 3]

    # XZ平面での向きを計算（Y軸は無視）
    theta = np.arctan2(shoulder_vector[..., 1], shoulder_vector[..., 0])  # [T, M]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 4. canonical_pose: ルート相対かつ yaw を打ち消した座標 [T, M, J, 3]
    x_rel = pose_rel[..., 0]
    y_rel = pose_rel[..., 1]
    z_rel = pose_rel[..., 2]

    # R(-theta) を適用して yaw 成分を除去
    x_can = cos_theta[..., None] * x_rel + sin_theta[..., None] * y_rel
    y_can = -sin_theta[..., None] * x_rel + cos_theta[..., None] * y_rel
    z_can = z_rel
    canonical_pose = np.stack([x_can, y_can, z_can], axis=-1)

    # 5. root_rot: yaw 角を (cos, sin) で表現
    root_rot = np.stack([cos_theta, sin_theta], axis=-1)  # [T, M, 2]

    # 6. global_pose: 元の絶対座標（そのまま）
    global_pose = pose_3d.copy()

    return {
        "canonical_pose_gt": canonical_pose.astype("float32"),
        "root_trans_gt": root_trans.astype("float32"),  # [T, M, 3]
        "root_rot_gt": root_rot.astype("float32"),  # [T, M, 2]
        "global_pose_gt": global_pose.astype("float32"),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess tennis JSON scenes into npz/memmap arrays"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/tennis_autogen",
        help="Root directory for auto-generated datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset directory under dataset_root",
    )
    parser.add_argument(
        "--max_cameras",
        type=int,
        default=4,
        help="Maximum number of cameras per scene (matches dataset.max_cameras)",
    )
    parser.add_argument(
        "--max_players",
        type=int,
        default=20,
        help="Maximum number of players per frame (matches dataset.max_players)",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=20,
        help="Number of keypoints per player (pose17 + racket3 = 20)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to preprocess",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npz files if they already exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes for scene preprocessing "
            "(0 or 1 for single-process)."
        ),
    )
    return parser.parse_args(argv)


def _normalize_2d(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return points.copy()
    out = points.copy().astype("float32")
    out[..., 0] = (out[..., 0] / float(width)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / float(height)) * 2.0 - 1.0
    return out


def _process_scene_json(
    scene_path: Path,
    max_cameras: int,
    max_players: int,
    num_joints: int,
) -> dict[str, np.ndarray]:
    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)
    frames = scene.get("frames", [])
    if not isinstance(frames, list) or not frames:
        msg = f"Scene has no frames: {scene_path}"
        raise ValueError(msg)
    num_cameras = int(scene.get("num_cameras", 0))
    if num_cameras <= 0:
        msg = f"Scene reports non-positive num_cameras: {scene_path}"
        raise ValueError(msg)
    if num_cameras > max_cameras:
        msg = f"Scene uses {num_cameras} cameras but max_cameras={max_cameras}"
        raise ValueError(msg)

    cameras = scene.get("cameras", [])
    if not isinstance(cameras, list) or len(cameras) != num_cameras:
        msg = f"Invalid cameras metadata in scene: {scene_path}"
        raise ValueError(msg)

    T_total = len(frames)
    T = T_total
    V = max_cameras
    M = max_players
    J = num_joints

    image_sizes: list[tuple[int, int]] = []
    camera_C = np.zeros((V, 3), dtype="float32")
    camera_R = np.zeros((V, 3, 3), dtype="float32")
    camera_intr = np.zeros((V, 3), dtype="float32")
    image_size_arr = np.zeros((V, 2), dtype="int32")
    for idx, cam in enumerate(cameras):
        size = cam.get("image_size", [0, 0])
        w, h = 0, 0
        if isinstance(size, list) and len(size) >= 2:
            w, h = int(size[0]), int(size[1])
        image_sizes.append((w, h))
        image_size_arr[idx, 0] = w
        image_size_arr[idx, 1] = h

        # Require real camera calibration from the simulator JSON.
        if "camera_C" not in cam or "camera_R" not in cam or "camera_intr" not in cam:
            msg = (
                f"Camera entry missing calibration fields in scene: {scene_path} "
                f"(index {idx})"
            )
            raise ValueError(msg)

        cam_C = np.asarray(cam["camera_C"], dtype="float32")
        cam_R = np.asarray(cam["camera_R"], dtype="float32")
        cam_intr = np.asarray(cam["camera_intr"], dtype="float32")
        camera_C[idx, :] = cam_C.reshape(3)
        camera_R[idx, :, :] = cam_R.reshape(3, 3)
        camera_intr[idx, :] = cam_intr.reshape(3)

    keypoints_2d = np.zeros((T, V, M, J, 2), dtype="float32")
    player_mask = np.zeros((T, V, M), dtype=bool)
    pose_3d = np.zeros((T, M, J, 3), dtype="float32")
    exist_3d = np.zeros((T, M), dtype=bool)
    court_2d = np.zeros((V, 20, 2), dtype="float32")

    # Court keypoints: take from frame 0 for each camera.
    first_frame = frames[0]
    for v in range(num_cameras):
        cam_key = f"cam_{v}"
        cam_payload = first_frame.get(cam_key, {})
        court_bundle = cam_payload.get("court_keypoints_2d", {})
        pts = court_bundle.get("points", [])
        if isinstance(pts, list) and len(pts) >= 20:
            pts_np = np.asarray(pts[:20], dtype="float32")
            w, h = image_sizes[v]
            pts_norm = _normalize_2d(pts_np, w, h)
            court_2d[v, :, :] = pts_norm

    # Per-frame 2D/3D.
    for t, frame in enumerate(frames):
        players_3d = frame.get("player_joints_3d", [])
        rackets_3d = frame.get("racket_points_3d", [])
        for v in range(num_cameras):
            cam_key = f"cam_{v}"
            cam_payload = frame.get(cam_key, {})
            player_bundle = cam_payload.get("player_keypoints_2d", {})
            racket_bundle = cam_payload.get("racket_keypoints_2d", {})
            joints = player_bundle.get("joints", [])
            rackets = racket_bundle.get("points", [])
            if not isinstance(joints, list):
                continue
            if not isinstance(rackets, list):
                rackets = [[] for _ in range(len(joints))]
            w, h = image_sizes[v]
            num_players = min(len(joints), M)
            for m in range(num_players):
                pose_pts = joints[m]
                racket_pts = rackets[m] if m < len(rackets) else []
                if not isinstance(pose_pts, list):
                    continue
                pose_np = np.zeros((17, 2), dtype="float32")
                racket_np = np.zeros((3, 2), dtype="float32")
                pose_src = np.asarray(pose_pts, dtype="float32")
                pose_np[: min(17, pose_src.shape[0]), :] = pose_src[
                    : min(17, pose_src.shape[0]), :
                ]
                if isinstance(racket_pts, list):
                    racket_src = np.asarray(racket_pts, dtype="float32")
                    racket_np[: min(3, racket_src.shape[0]), :] = racket_src[
                        : min(3, racket_src.shape[0]), :
                    ]
                combined = np.concatenate([pose_np, racket_np], axis=0)
                combined = _normalize_2d(combined, w, h)
                keypoints_2d[t, v, m, :, :] = combined
                player_mask[t, v, m] = True

        if isinstance(players_3d, list):
            num_players_3d = min(len(players_3d), M)
            if not isinstance(rackets_3d, list):
                rackets_3d = [[] for _ in range(len(players_3d))]
            for m in range(num_players_3d):
                pose3d = players_3d[m]
                racket3d = rackets_3d[m] if m < len(rackets_3d) else []
                if not isinstance(pose3d, list):
                    continue
                pose3d_np = np.zeros((17, 3), dtype="float32")
                racket3d_np = np.zeros((3, 3), dtype="float32")
                pose3d_src = np.asarray(pose3d, dtype="float32")
                pose3d_np[: min(17, pose3d_src.shape[0]), :] = pose3d_src[
                    : min(17, pose3d_src.shape[0]), :
                ]
                if isinstance(racket3d, list):
                    racket3d_src = np.asarray(racket3d, dtype="float32")
                    racket3d_np[: min(3, racket3d_src.shape[0]), :] = racket3d_src[
                        : min(3, racket3d_src.shape[0]), :
                    ]
                combined3d = np.concatenate([pose3d_np, racket3d_np], axis=0)
                combined3d[:, 0] = combined3d[:, 0] / float(HALF_DOUBLES_WIDTH)
                combined3d[:, 1] = combined3d[:, 1] / float(HALF_LENGTH)
                combined3d[:, 2] = combined3d[:, 2] / float(NET_HEIGHT_POST)
                pose_3d[t, m, :, :] = combined3d
                exist_3d[t, m] = True

    # v2用GTデータを生成
    v2_gt_data = _decompose_pose_for_v2(pose_3d)

    return {
        "keypoints_2d": keypoints_2d,
        "player_mask": player_mask,
        "court_2d": court_2d,
        "pose_3d_gt": pose_3d,
        "exist_3d_gt": exist_3d,
        "camera_C": camera_C,
        "camera_R": camera_R,
        "camera_intr": camera_intr,
        "image_size": image_size_arr,
        # v2用GTデータ
        **v2_gt_data,
    }


def _process_single_scene(
    args: tuple[Path, Path, int, int, int, bool],
) -> None:
    """Worker function to preprocess a single scene JSON into an npz file."""
    scene_path, arrays_dir, max_cameras, max_players, num_joints, overwrite = args
    stem = scene_path.stem
    out_path = arrays_dir / f"{stem}.npz"
    if out_path.exists() and not overwrite:
        return
    arrays = _process_scene_json(
        scene_path,
        max_cameras=max_cameras,
        max_players=max_players,
        num_joints=num_joints,
    )
    # Use uncompressed npz to allow mmap_mode="r" in np.load.
    np.savez(out_path, **arrays)


def _process_split(
    dataset_dir: Path,
    split: str,
    max_cameras: int,
    max_players: int,
    num_joints: int,
    overwrite: bool,
    num_workers: int,
) -> None:
    scenes_dir = dataset_dir / "scenes" / split
    arrays_dir = dataset_dir / "arrays" / split
    arrays_dir.mkdir(parents=True, exist_ok=True)
    if not scenes_dir.exists():
        return
    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    if not scene_paths:
        return

    num_workers = int(num_workers)
    if num_workers <= 0 or len(scene_paths) == 1:
        # Single-process preprocessing (original behavior).
        for scene_path in tqdm(
            scene_paths,
            desc=f"Preprocess scenes ({split})",
        ):
            _process_single_scene(
                (
                    scene_path,
                    arrays_dir,
                    max_cameras,
                    max_players,
                    num_joints,
                    overwrite,
                )
            )
        return

    # Multi-process preprocessing using ProcessPoolExecutor.
    tasks = [
        (scene_path, arrays_dir, max_cameras, max_players, num_joints, overwrite)
        for scene_path in scene_paths
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in tqdm(
            executor.map(_process_single_scene, tasks),
            total=len(tasks),
            desc=f"Preprocess scenes ({split})",
        ):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the memmap preprocessing CLI."""
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_root) / args.dataset_name
    if not dataset_dir.exists():
        msg = f"Dataset directory not found: {dataset_dir}"
        raise SystemExit(msg)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    for split in tqdm(
        splits,
        desc="Processing splits",
    ):
        _process_split(
            dataset_dir,
            split,
            max_cameras=int(args.max_cameras),
            max_players=int(args.max_players),
            num_joints=int(args.num_joints),
            overwrite=bool(args.overwrite),
            num_workers=int(getattr(args, "num_workers", 0)),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
