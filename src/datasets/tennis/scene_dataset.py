"""Dataset that loads fixed-length tennis pose windows from JSON scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class _WindowRecord:
    """Metadata for a single temporal window within a scene."""

    scene_path: str
    scene_id: str
    t_start: int
    t_end: int
    num_frames: int
    num_cameras: int
    max_players_in_window: int


class TennisSceneWindowDataset(Dataset):
    r"""Dataset that materializes `[T, V, M, J, 2]` pose windows from scenes.

    The dataset assumes that :mod:`build_tennis_dataset.py` has populated
    ``scenes/<split>`` and ``index/<split>_index.jsonl`` under a common
    ``dataset_root/dataset_name`` directory.

    Args:
        dataset_root (str | Path): Root directory for auto-generated datasets.
        dataset_name (str): Name of the dataset directory under ``dataset_root``.
        split (str): One of ``"train"``, ``"val"``, or ``"test"``.
        window_T (int): Temporal window length in frames.
        max_cameras (int): Maximum number of cameras to materialize.
        max_players (int): Maximum number of players per image.
        num_joints (int): Number of keypoints per player.
        use_memmap (bool): Whether to load from preprocessed npz memmap files
            instead of parsing JSON scenes directly.

    Raises:
        ValueError: If ``window_T``, ``max_cameras``, or ``max_players`` is
            not positive.
        FileNotFoundError: If the index file for the requested split does not
            exist under ``dataset_root/dataset_name/index``.

    """

    def __init__(
        self,
        dataset_root: str | Path,
        dataset_name: str,
        split: str,
        window_T: int,
        max_cameras: int,
        max_players: int,
        num_joints: int = 20,
        use_memmap: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.split = split
        self.window_T = int(window_T)
        self.max_cameras = int(max_cameras)
        self.max_players = int(max_players)
        self.num_joints = int(num_joints)
        self.use_memmap = bool(use_memmap)

        if self.window_T <= 0:
            msg = "window_T must be positive"
            raise ValueError(msg)
        if self.max_cameras <= 0:
            msg = "max_cameras must be positive"
            raise ValueError(msg)
        if self.max_players <= 0:
            msg = "max_players must be positive"
            raise ValueError(msg)

        self.dataset_dir = self.dataset_root / self.dataset_name
        index_path = self.dataset_dir / "index" / f"{self.split}_index.jsonl"
        if not index_path.exists():
            msg = f"Index file not found for split '{self.split}': {index_path}"
            raise FileNotFoundError(msg)
        self.records = self._load_index(index_path)
        self._arrays_cache: dict[str, Any] = {}

    @staticmethod
    def _load_index(index_path: Path) -> list[_WindowRecord]:
        records: list[_WindowRecord] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data: dict[str, Any] = json.loads(line)
                records.append(
                    _WindowRecord(
                        scene_path=str(data["scene_path"]),
                        scene_id=str(data.get("scene_id", "")),
                        t_start=int(data["t_start"]),
                        t_end=int(data["t_end"]),
                        num_frames=int(data["num_frames"]),
                        num_cameras=int(data["num_cameras"]),
                        max_players_in_window=int(data["max_players_in_window"]),
                    )
                )
        return records

    def __len__(self) -> int:
        """Return the number of temporal windows in the dataset."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Load and return a single window sample."""
        if self.use_memmap:
            return self._getitem_memmap(index)
        return self._getitem_from_json(index)

    def _getitem_memmap(self, index: int) -> dict[str, Tensor]:
        rec = self.records[index]
        rel = Path(rec.scene_path)
        stem = rel.stem
        npz_path = self.dataset_dir / "arrays" / self.split / f"{stem}.npz"
        if stem not in self._arrays_cache:
            if not npz_path.exists():
                msg = f"Memmap npz not found for scene '{stem}': {npz_path}"
                raise FileNotFoundError(msg)
            self._arrays_cache[stem] = __import__("numpy").load(  # lazy import
                npz_path, mmap_mode="r"
            )
        arrs = self._arrays_cache[stem]

        T_total = int(arrs["keypoints_2d"].shape[0])
        T = self.window_T
        if rec.t_end > T_total:
            msg = f"Window end {rec.t_end} exceeds scene length {T_total}"
            raise ValueError(msg)

        t_start = rec.t_start
        t_end = rec.t_end
        length = t_end - t_start
        if length > T:
            msg = f"Window length {length} exceeds configured window_T={T}"
            raise ValueError(msg)

        keypoints_2d = torch.zeros(
            (T, self.max_cameras, self.max_players, self.num_joints, 2),
            dtype=torch.float32,
        )
        player_mask = torch.zeros(
            (T, self.max_cameras, self.max_players), dtype=torch.bool
        )
        pose_3d = torch.zeros(
            (T, self.max_players, self.num_joints, 3), dtype=torch.float32
        )
        exist_3d = torch.zeros((T, self.max_players), dtype=torch.bool)
        court_2d = torch.zeros((self.max_cameras, 20, 2), dtype=torch.float32)

        src_key = torch.from_numpy(arrs["keypoints_2d"][t_start:t_end])  # [len,V,M,J,2]
        src_mask = torch.from_numpy(arrs["player_mask"][t_start:t_end])  # [len,V,M]
        src_pose = torch.from_numpy(arrs["pose_3d_gt"][t_start:t_end])  # [len,M,J,3]
        src_exist = torch.from_numpy(arrs["exist_3d_gt"][t_start:t_end])  # [len,M]
        src_court = torch.from_numpy(arrs["court_2d"])  # [V,20,2]

        keypoints_2d[:length, : src_key.shape[1], : src_key.shape[2]] = src_key
        player_mask[:length, : src_mask.shape[1], : src_mask.shape[2]] = src_mask
        pose_3d[:length, : src_pose.shape[1]] = src_pose
        exist_3d[:length, : src_exist.shape[1]] = src_exist
        court_2d[: src_court.shape[0]] = src_court

        return {
            "keypoints_2d": keypoints_2d,
            "player_mask": player_mask,
            "court_2d": court_2d,
            "pose_3d_gt": pose_3d,
            "exist_3d_gt": exist_3d,
            "scene_id": torch.tensor([hash(rec.scene_id)], dtype=torch.long),
            "t_start": torch.tensor([t_start], dtype=torch.long),
            "t_end": torch.tensor([t_end], dtype=torch.long),
        }

    def _getitem_from_json(self, index: int) -> dict[str, Tensor]:
        rec = self.records[index]
        scene_path = self.dataset_dir / rec.scene_path
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
        if num_cameras > self.max_cameras:
            msg = f"Scene uses {num_cameras} cameras but max_cameras={self.max_cameras}"
            raise ValueError(msg)
        if rec.num_frames > self.window_T:
            msg = f"Window length {rec.num_frames} exceeds configured window_T={self.window_T}"
            raise ValueError(msg)

        t_start = rec.t_start
        t_end = rec.t_end
        window_frames = frames[t_start:t_end]
        # Tensor shapes
        T = self.window_T
        V = self.max_cameras
        M = self.max_players
        J = self.num_joints

        keypoints_2d = torch.zeros((T, V, M, J, 2), dtype=torch.float32)
        player_mask = torch.zeros((T, V, M), dtype=torch.bool)
        pose_3d = torch.zeros((T, M, J, 3), dtype=torch.float32)
        exist_3d = torch.zeros((T, M), dtype=torch.bool)
        court_2d = torch.zeros((V, 20, 2), dtype=torch.float32)

        # Camera image sizes for normalization (width, height).
        cameras = scene.get("cameras", [])
        if not isinstance(cameras, list) or len(cameras) != num_cameras:
            msg = f"Invalid cameras metadata in scene: {scene_path}"
            raise ValueError(msg)
        image_sizes: list[tuple[int, int]] = []
        for cam in cameras:
            size = cam.get("image_size", [0, 0])
            if not isinstance(size, list) or len(size) < 2:
                image_sizes.append((0, 0))
            else:
                image_sizes.append((int(size[0]), int(size[1])))

        # Court keypoints are assumed constant across frames; take from the first.
        first_frame = window_frames[0]
        for v in range(num_cameras):
            cam_key = f"cam_{v}"
            cam_payload = first_frame.get(cam_key, {})
            court_bundle = cam_payload.get("court_keypoints_2d", {})
            pts = court_bundle.get("points", [])
            if isinstance(pts, list) and len(pts) >= 20:
                pts_tensor = torch.as_tensor(pts[:20], dtype=torch.float32)
                w, h = image_sizes[v]
                if w > 0 and h > 0:
                    pts_tensor[:, 0] = (pts_tensor[:, 0] / float(w)) * 2.0 - 1.0
                    pts_tensor[:, 1] = (pts_tensor[:, 1] / float(h)) * 2.0 - 1.0
                court_2d[v, :, :] = pts_tensor

        # Populate per-frame player keypoints (2D + 3D).
        for local_t, frame in enumerate(window_frames):
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
                    # Expect 17 pose joints and 3 racket points; truncate/pad otherwise.
                    pose_tensor = torch.zeros((17, 2), dtype=torch.float32)
                    racket_tensor = torch.zeros((3, 2), dtype=torch.float32)
                    pose_src = torch.as_tensor(pose_pts, dtype=torch.float32)
                    pose_tensor[: min(17, pose_src.shape[0]), :] = pose_src[
                        : min(17, pose_src.shape[0]), :
                    ]
                    if isinstance(racket_pts, list):
                        racket_src = torch.as_tensor(racket_pts, dtype=torch.float32)
                        racket_tensor[: min(3, racket_src.shape[0]), :] = racket_src[
                            : min(3, racket_src.shape[0]), :
                        ]
                    combined = torch.cat([pose_tensor, racket_tensor], dim=0)
                    if w > 0 and h > 0:
                        combined[:, 0] = (combined[:, 0] / float(w)) * 2.0 - 1.0
                        combined[:, 1] = (combined[:, 1] / float(h)) * 2.0 - 1.0
                    keypoints_2d[local_t, v, m, :, :] = combined
                    player_mask[local_t, v, m] = True

            # 3D GT for this frame (per player, view-independent).
            if isinstance(players_3d, list):
                num_players_3d = min(len(players_3d), M)
                if not isinstance(rackets_3d, list):
                    rackets_3d = [[] for _ in range(len(players_3d))]
                for m in range(num_players_3d):
                    pose3d = players_3d[m]
                    racket3d = rackets_3d[m] if m < len(rackets_3d) else []
                    if not isinstance(pose3d, list):
                        continue
                    pose3d_tensor = torch.zeros((17, 3), dtype=torch.float32)
                    racket3d_tensor = torch.zeros((3, 3), dtype=torch.float32)
                    pose3d_src = torch.as_tensor(pose3d, dtype=torch.float32)
                    pose3d_tensor[: min(17, pose3d_src.shape[0]), :] = pose3d_src[
                        : min(17, pose3d_src.shape[0]), :
                    ]
                    if isinstance(racket3d, list):
                        racket3d_src = torch.as_tensor(racket3d, dtype=torch.float32)
                        racket3d_tensor[: min(3, racket3d_src.shape[0]), :] = (
                            racket3d_src[: min(3, racket3d_src.shape[0]), :]
                        )
                    combined3d = torch.cat([pose3d_tensor, racket3d_tensor], dim=0)
                    pose_3d[local_t, m, :, :] = combined3d
                    exist_3d[local_t, m] = True

        return {
            "keypoints_2d": keypoints_2d,
            "player_mask": player_mask,
            "court_2d": court_2d,
            "pose_3d_gt": pose_3d,
            "exist_3d_gt": exist_3d,
            "scene_id": torch.tensor([hash(rec.scene_id)], dtype=torch.long),
            "t_start": torch.tensor([t_start], dtype=torch.long),
            "t_end": torch.tensor([t_end], dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = TennisSceneWindowDataset(
        dataset_root="data/tennis_autogen",
        dataset_name="sim_fps60_dur3p0_C4_P1-20_T10",
        split="train",
        window_T=10,
        max_cameras=4,
        max_players=20,
        num_joints=20,
    )

    for i in range(10):
        sample = dataset[i]
        print(sample["keypoints_2d"].shape)
