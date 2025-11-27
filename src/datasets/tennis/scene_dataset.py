"""Dataset that loads fixed-length tennis pose windows from JSON scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.tennis.augment import (
    apply_random_2d_affine,
    sample_camera_indices,
)


@dataclass
class _WindowTensors:
    """Container for pre-allocated window tensors."""
    keypoints_2d: Tensor
    player_mask: Tensor
    pose_3d: Tensor
    exist_3d: Tensor
    court_2d: Tensor
    camera_C: Tensor
    camera_R: Tensor
    camera_intr: Tensor
    image_size: Tensor
    canonical_pose_gt: Tensor
    root_trans_gt: Tensor
    root_rot_gt: Tensor
    global_pose_gt: Tensor


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
        min_cameras (int | None): Minimum number of cameras required.
        augment_2d (bool): Whether to apply 2D augmentation.

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
        min_cameras: int | None = None,
        augment_2d: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.split = split
        self.window_T = int(window_T)
        self.max_cameras = int(max_cameras)
        self.max_players = int(max_players)
        self.num_joints = int(num_joints)
        self.min_cameras: int | None = (
            int(min_cameras) if min_cameras is not None else None
        )
        self.augment_2d = bool(augment_2d)
        self.current_max_cameras = self.max_cameras
        self.current_min_cameras = self.min_cameras

        if self.window_T <= 0:
            msg = "window_T must be positive"
            raise ValueError(msg)
        if self.max_cameras <= 0:
            msg = "max_cameras must be positive"
            raise ValueError(msg)
        if self.max_players <= 0:
            msg = "max_players must be positive"
            raise ValueError(msg)
        if self.min_cameras is not None and self.min_cameras <= 0:
            msg = "min_cameras must be positive if set"
            raise ValueError(msg)
        if self.min_cameras is not None and self.min_cameras > self.max_cameras:
            msg = "min_cameras must be <= max_cameras"
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
        """Get a single window sample."""
        rec = self.records[index]

        arrs, t_start, t_end, length = self._load_scene_arrays(rec)
        tensors, _ = self._materialize_window(arrs, t_start, t_end, length)
        sample = self._build_sample_dict(tensors, rec, t_start, t_end)

        return apply_random_2d_affine(
            sample,
            enabled=self.augment_2d,
            split=self.split,
        )


    def _load_scene_arrays(self, rec: _WindowRecord) -> tuple[Any, int, int, int]:
        """Load memmap npz and compute time window bounds."""
        rel = Path(rec.scene_path)
        stem = rel.stem
        npz_path = self.dataset_dir / "arrays" / self.split / f"{stem}.npz"

        if stem not in self._arrays_cache:
            if not npz_path.exists():
                msg = f"Memmap npz not found for scene '{stem}': {npz_path}"
                raise FileNotFoundError(msg)
            self._arrays_cache[stem] = __import__("numpy").load(
                npz_path, mmap_mode="r"
            )

        arrs = self._arrays_cache[stem]
        t_start, t_end, length = self._compute_window_bounds(rec, arrs)
        return arrs, t_start, t_end, length

    def _compute_window_bounds(
        self, rec: _WindowRecord, arrs: Any
    ) -> tuple[int, int, int]:
        """Compute t_start, t_end, length for the window."""
        T_total = int(arrs["keypoints_2d"].shape[0])
        if rec.t_end > T_total:
            msg = f"Window end {rec.t_end} exceeds scene length {T_total}"
            raise ValueError(msg)

        T = self.window_T
        t_start = rec.t_start
        t_end = min(rec.t_end, t_start + T)
        length = t_end - t_start
        if length <= 0:
            msg = "Window length must be positive after clamping"
            raise ValueError(msg)
        return t_start, t_end, length

    def _sample_cameras_for_scene(self, num_available: int) -> torch.Tensor:
        """Sample camera indices for the scene."""
        return sample_camera_indices(
            num_available=num_available,
            max_cameras=self.current_max_cameras,
            min_cameras=self.current_min_cameras,
        )

    def _alloc_window_tensors(self) -> _WindowTensors:
        """Allocate zero-initialized tensors for the window."""
        T = self.window_T
        V = self.max_cameras
        M = self.max_players
        J = self.num_joints

        return _WindowTensors(
            keypoints_2d=torch.zeros((T, V, M, J, 2), dtype=torch.float32),
            player_mask=torch.zeros((T, V, M), dtype=torch.bool),
            pose_3d=torch.zeros((T, M, J, 3), dtype=torch.float32),
            exist_3d=torch.zeros((T, M), dtype=torch.bool),
            court_2d=torch.zeros((V, 20, 2), dtype=torch.float32),
            camera_C=torch.zeros((V, 3), dtype=torch.float32),
            camera_R=torch.zeros((V, 3, 3), dtype=torch.float32),
            camera_intr=torch.zeros((V, 3), dtype=torch.float32),
            image_size=torch.zeros((V, 2), dtype=torch.int32),
            canonical_pose_gt=torch.zeros((T, M, J, 3), dtype=torch.float32),
            root_trans_gt=torch.zeros((T, M, 3), dtype=torch.float32),
            root_rot_gt=torch.zeros((T, M, 2), dtype=torch.float32),
            global_pose_gt=torch.zeros((T, M, J, 3), dtype=torch.float32),
        )

    def _fill_2d_tensors(
        self,
        tensors: _WindowTensors,
        arrs: Any,
        t_start: int,
        t_end: int,
        length: int,
        cam_indices: torch.Tensor,
    ) -> None:
        """Fill 2D keypoints and player mask tensors."""
        src_key = torch.from_numpy(arrs["keypoints_2d"][t_start:t_end])
        src_mask = torch.from_numpy(arrs["player_mask"][t_start:t_end])

        num_joints_2d = min(self.num_joints, src_key.shape[3])
        k = int(cam_indices.shape[0])

        tensors.keypoints_2d[:length, :k, : src_key.shape[2], :num_joints_2d] = (
            src_key[:, cam_indices, :, :num_joints_2d]
        )
        tensors.player_mask[:length, :k, : src_mask.shape[2]] = src_mask[:, cam_indices]
    
    def _fill_3d_tensors(
        self,
        tensors: _WindowTensors,
        arrs: Any,
        t_start: int,
        t_end: int,
        length: int,
    ) -> None:
        # --- 基本3D (absolute pose + exist) ---
        src_pose = torch.from_numpy(arrs["pose_3d_gt"][t_start:t_end])   # [len,M,J,3]
        src_exist = torch.from_numpy(arrs["exist_3d_gt"][t_start:t_end]) # [len,M]

        num_joints_3d = min(self.num_joints, src_pose.shape[2])

        tensors.pose_3d[:length, : src_pose.shape[1], :num_joints_3d] = (
            src_pose[:, :, :num_joints_3d]
        )
        tensors.exist_3d[:length, : src_exist.shape[1]] = src_exist

        # --- canonical / root / global も「3D系」としてそのまま埋める ---
        canonical_slice = torch.from_numpy(arrs["canonical_pose_gt"][t_start:t_end])
        root_trans_slice = torch.from_numpy(arrs["root_trans_gt"][t_start:t_end])
        root_rot_slice = torch.from_numpy(arrs["root_rot_gt"][t_start:t_end])
        global_slice = torch.from_numpy(arrs["global_pose_gt"][t_start:t_end])

        J = self.num_joints
        num_joints_canon = min(J, canonical_slice.shape[2])
        num_joints_global = min(J, global_slice.shape[2])

        tensors.canonical_pose_gt[
            :length,
            : canonical_slice.shape[1],
            :num_joints_canon,
        ] = canonical_slice[:, :, :num_joints_canon]

        tensors.root_trans_gt[:length, : root_trans_slice.shape[1]] = root_trans_slice
        tensors.root_rot_gt[:length, : root_rot_slice.shape[1]] = root_rot_slice

        tensors.global_pose_gt[
            :length,
            : global_slice.shape[1],
            :num_joints_global,
        ] = global_slice[:, :, :num_joints_global]

    def _fill_camera_tensors(
        self,
        tensors: _WindowTensors,
        arrs: Any,
        cam_indices: torch.Tensor,
    ) -> None:
        """Fill camera-related tensors."""
        if "camera_C" not in arrs or "camera_R" not in arrs:
            msg = "Memmap arrays missing required camera metadata"
            raise KeyError(msg)

        src_court = torch.from_numpy(arrs["court_2d"])
        src_cam_C = torch.from_numpy(arrs["camera_C"]).to(torch.float32)
        src_cam_R = torch.from_numpy(arrs["camera_R"]).to(torch.float32)
        src_cam_intr = torch.from_numpy(arrs["camera_intr"]).to(torch.float32)
        src_image_size = torch.from_numpy(arrs["image_size"]).to(torch.int32)

        k = int(cam_indices.shape[0])

        tensors.court_2d[:k] = src_court[cam_indices]
        tensors.camera_C[:k] = src_cam_C[cam_indices]
        tensors.camera_R[:k] = src_cam_R[cam_indices]
        tensors.camera_intr[:k] = src_cam_intr[cam_indices]
        tensors.image_size[:k] = src_image_size[cam_indices]

    def _materialize_window(
        self,
        arrs: Any,
        t_start: int,
        t_end: int,
        length: int,
    ) -> tuple[_WindowTensors, torch.Tensor]:
        """Materialize the complete window from arrays."""
        tensors = self._alloc_window_tensors()

        # Camera sampling
        V_src = int(arrs["keypoints_2d"].shape[1])
        cam_indices = self._sample_cameras_for_scene(V_src)

        # v1 + v2 系をまとめて埋める
        self._fill_2d_tensors(tensors, arrs, t_start, t_end, length, cam_indices)
        self._fill_3d_tensors(tensors, arrs, t_start, t_end, length)
        self._fill_camera_tensors(tensors, arrs, cam_indices)

        return tensors, cam_indices


    def _build_sample_dict(
        self,
        tensors: _WindowTensors,
        rec: _WindowRecord,
        t_start: int,
        t_end: int,
    ) -> dict[str, Tensor]:
        return {
            "keypoints_2d": tensors.keypoints_2d,
            "player_mask": tensors.player_mask,
            "court_2d": tensors.court_2d,
            "pose_3d_gt": tensors.pose_3d,
            "exist_3d_gt": tensors.exist_3d,
            "camera_C": tensors.camera_C,
            "camera_R": tensors.camera_R,
            "camera_intr": tensors.camera_intr,
            "image_size": tensors.image_size,
            "canonical_pose_gt": tensors.canonical_pose_gt,
            "root_trans_gt": tensors.root_trans_gt,
            "root_rot_gt": tensors.root_rot_gt,
            "global_pose_gt": tensors.global_pose_gt,
            "scene_id": torch.tensor([hash(rec.scene_id)], dtype=torch.long),
            "t_start": torch.tensor([t_start], dtype=torch.long),
            "t_end": torch.tensor([t_end], dtype=torch.long),
        }

    def set_active_camera_bounds(
        self,
        *,
        max_cameras: int | None = None,
        min_cameras: int | None = None,
    ) -> None:
        """Update the sampling bounds without changing tensor shapes."""
        if max_cameras is not None:
            if max_cameras <= 0 or max_cameras > self.max_cameras:
                msg = "max_cameras must be within (0, self.max_cameras] when ``set_active_camera_bounds`` is called"
                raise ValueError(msg)
            self.current_max_cameras = int(max_cameras)
        if min_cameras is not None:
            if min_cameras <= 0 or min_cameras > self.current_max_cameras:
                msg = "min_cameras must satisfy 0 < min_cameras <= current_max_cameras"
                raise ValueError(msg)
            self.current_min_cameras = int(min_cameras)
        # Ensure min does not exceed max after independent updates.
        if (
            self.current_min_cameras is not None
            and self.current_min_cameras > self.current_max_cameras
        ):
            self.current_min_cameras = self.current_max_cameras


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
