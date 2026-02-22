"""Unified dataset for PLCS frame/sequence/single/multiview modes."""

from __future__ import annotations

import random as rng
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, ListConfig
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.common.data.scene_cache import extract_scene_meta_parallel, get_scene_cache
from src.common.dataset.augmentation import augment_keypoints
from src.plcs.data.targets import build_coco17_world_targets
from src.plcs.data.types import PLCSBatch
from src.plcs.generate_dataset.io.scene_loader import load_scene

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDataset(Dataset[dict[str, Tensor]]):
    """Unified PLCS dataset.

    Returns per-sample tensors with camera-time ordering:
    - human_kp: (N, T, 17, 2)
    - court_kp: (N, T, 20, 2)
    - human_vis: (N, T, 17)
    - court_vis: (N, T, 20)
    - human_mask: (N, T)
    - position: (T, 3)
    - rotation: (T, 2)
    """

    def __init__(
        self,
        scene_dir: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
        cache_maxsize: int = 128,
        parallel_workers: int = 8,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        self.config = config or {}
        self.augment = augment
        self.parallel_workers = parallel_workers

        data_cfg = self.config.get("data", {})
        self.mode = str(data_cfg.get("mode", "frame"))
        self.camera_mode = str(data_cfg.get("camera_mode", "random"))

        self.is_multiview = self.mode in {"multiview", "multiview_sequence"}
        self.is_sequence = self.mode in {"sequence", "multiview_sequence"}

        self.seq_stride = int(data_cfg.get("seq_stride", 1))
        self.min_cameras = int(data_cfg.get("min_cameras", 2 if self.is_multiview else 1))

        if "num_views_range" in data_cfg:
            r = data_cfg["num_views_range"]
            self.num_views_range: tuple[int, int] = (int(r[0]), int(r[1]))
        else:
            self.num_views_range = (1, 2)

        if "seq_len_range" in data_cfg:
            r = data_cfg["seq_len_range"]
            self.seq_len_range: tuple[int, int] = (int(r[0]), int(r[1]))
        else:
            self.seq_len_range = (64, 512)

        augmentation_cfg = data_cfg.get("augmentation")
        if not isinstance(augmentation_cfg, (dict, DictConfig)):
            raise ValueError(
                "data.augmentation must be provided with keys "
                "['keypoint_noise_std', 'visibility_drop_prob']."
            )
        self.kp_noise_std = float(augmentation_cfg["keypoint_noise_std"])
        self.visibility_drop_prob = float(augmentation_cfg["visibility_drop_prob"])

        self._scene_cache = get_scene_cache(load_fn=load_scene, maxsize=cache_maxsize)

        scenes_subdir = self.scene_dir / "scenes"
        self.scene_files = sorted(scenes_subdir.glob("scene_*.npz"))
        if not self.scene_files:
            raise ValueError(f"No scene files found in {scenes_subdir}")

        self.scene_paths: list[Path] = []
        self.index: list[tuple[int, int]] = []  # (scene_idx, start_frame)
        self._build_index()

    def _build_index(self) -> None:
        min_seq_for_index = max(1, int(self.seq_len_range[0]))

        scene_metas = extract_scene_meta_parallel(
            self.scene_files,
            max_workers=self.parallel_workers,
        )

        for meta in scene_metas:
            if meta.num_cameras < self.min_cameras:
                continue
            if self.is_sequence and meta.num_frames < min_seq_for_index:
                continue

            self.scene_paths.append(meta.scene_path)
            scene_idx = len(self.scene_paths) - 1

            if self.is_sequence:
                max_start = max(0, meta.num_frames - min_seq_for_index)
                for start in range(0, max_start + 1, max(1, self.seq_stride)):
                    self.index.append((scene_idx, start))
            else:
                for frame_idx in range(meta.num_frames):
                    self.index.append((scene_idx, frame_idx))

        if not self.index:
            raise ValueError(
                "No valid samples were indexed. "
                "Check data.mode, min_cameras, and sequence length settings."
            )

    def __len__(self) -> int:
        return len(self.index)

    def _select_num_views(self, num_cameras: int) -> int:
        if not self.is_multiview:
            if self.camera_mode == "all":
                return num_cameras
            return 1

        min_views, max_views = self.num_views_range
        max_possible = min(max_views, num_cameras)
        min_possible = min(min_views, max_possible)
        return rng.randint(min_possible, max_possible)

    def _select_cameras(self, num_cameras: int) -> list[int]:
        if self.is_multiview:
            num_views = self._select_num_views(num_cameras)
            return rng.sample(range(num_cameras), num_views)

        if self.camera_mode == "random":
            return [rng.randrange(num_cameras)]
        if self.camera_mode in {"all", "first"}:
            return [0]
        cam_idx = int(self.camera_mode)
        if cam_idx < 0 or cam_idx >= num_cameras:
            raise ValueError(
                f"camera_mode={self.camera_mode} is out of range for {num_cameras} cameras"
            )
        return [cam_idx]

    def _select_seq_len(self, num_frames: int, start_frame: int) -> int:
        if not self.is_sequence:
            return 1

        remaining = num_frames - start_frame
        if remaining <= 0:
            return 1

        min_len, max_len = self.seq_len_range
        max_len = min(max_len, remaining)
        min_len = min(min_len, max_len)
        return rng.randint(min_len, max_len)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        scene_idx, start_frame = self.index[idx]
        scene = self._scene_cache.get(self.scene_paths[scene_idx])

        num_cameras = len(scene["cameras"])
        selected_cameras = self._select_cameras(num_cameras)

        num_frames = int(scene.meta["num_frames"])
        seq_len = self._select_seq_len(num_frames, start_frame)
        end_frame = start_frame + seq_len

        human_kp_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        human_vis_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []

        for cam_idx in selected_cameras:
            cam = scene.cameras[cam_idx]

            human_kp = torch.from_numpy(cam.human_kp_uv[start_frame:end_frame].copy()).float()
            court_kp = torch.from_numpy(cam.court_kp_uv[start_frame:end_frame].copy()).float()
            human_vis = torch.from_numpy(cam.human_kp_visible[start_frame:end_frame].copy()).float()
            court_vis = torch.from_numpy(cam.court_kp_visible[start_frame:end_frame].copy()).float()

            if self.augment:
                human_kp, human_vis = augment_keypoints(
                    human_kp,
                    human_vis,
                    self.kp_noise_std,
                    self.visibility_drop_prob,
                )
                court_kp, court_vis = augment_keypoints(
                    court_kp,
                    court_vis,
                    self.kp_noise_std,
                    self.visibility_drop_prob,
                )

            human_kp = human_kp * human_vis.unsqueeze(-1)
            court_kp = court_kp * court_vis.unsqueeze(-1)

            human_kp_list.append(human_kp)
            court_kp_list.append(court_kp)
            human_vis_list.append(human_vis)
            court_vis_list.append(court_vis)

        position = torch.from_numpy(scene.position[start_frame:end_frame].copy()).float()
        rotation = torch.from_numpy(scene.rotation[start_frame:end_frame].copy()).float()

        sample: dict[str, Tensor] = {
            "human_kp": torch.stack(human_kp_list, dim=0),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "human_vis": torch.stack(human_vis_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "human_mask": torch.ones(len(selected_cameras), seq_len, dtype=torch.float32),
            "position": position,
            "rotation": rotation,
        }

        if "human_kp_3d" not in scene:
            scene["human_kp_3d"] = build_coco17_world_targets(scene)
        sample["human_kp_3d"] = torch.from_numpy(
            scene["human_kp_3d"][start_frame:end_frame].copy()
        ).float()

        return sample


def collate_plcs_batch(batch: list[dict[str, Tensor]]) -> PLCSBatch | dict[str, Tensor]:
    """Collate variable-view/variable-length PLCS samples into a padded batch."""
    max_views = max(int(sample["human_kp"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["human_kp"].shape[1]) for sample in batch)

    human_kp_batch = []
    court_kp_batch = []
    human_vis_batch = []
    court_vis_batch = []
    human_mask_batch = []
    position_batch = []
    rotation_batch = []
    human_kp_3d_batch = []

    for sample in batch:
        n_views = int(sample["human_kp"].shape[0])
        seq_len = int(sample["human_kp"].shape[1])
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        human_kp = sample["human_kp"]
        court_kp = sample["court_kp"]
        human_vis = sample["human_vis"]
        court_vis = sample["court_vis"]
        human_mask = sample["human_mask"]
        position = sample["position"]
        rotation = sample["rotation"]
        human_kp_3d = sample.get("human_kp_3d")

        if pad_seq > 0:
            human_kp = torch.cat([human_kp, torch.zeros(n_views, pad_seq, 17, 2)], dim=1)
            court_kp = torch.cat([court_kp, torch.zeros(n_views, pad_seq, 20, 2)], dim=1)
            human_vis = torch.cat([human_vis, torch.zeros(n_views, pad_seq, 17)], dim=1)
            court_vis = torch.cat([court_vis, torch.zeros(n_views, pad_seq, 20)], dim=1)
            human_mask = torch.cat([human_mask, torch.zeros(n_views, pad_seq)], dim=1)
            position = torch.cat([position, torch.zeros(pad_seq, 3)], dim=0)
            rotation = torch.cat([rotation, torch.zeros(pad_seq, 2)], dim=0)
            if human_kp_3d is not None:
                human_kp_3d = torch.cat([human_kp_3d, torch.zeros(pad_seq, 17, 3)], dim=0)

        if pad_views > 0:
            human_kp = torch.cat([human_kp, torch.zeros(pad_views, max_seq_len, 17, 2)], dim=0)
            court_kp = torch.cat([court_kp, torch.zeros(pad_views, max_seq_len, 20, 2)], dim=0)
            human_vis = torch.cat([human_vis, torch.zeros(pad_views, max_seq_len, 17)], dim=0)
            court_vis = torch.cat([court_vis, torch.zeros(pad_views, max_seq_len, 20)], dim=0)
            human_mask = torch.cat([human_mask, torch.zeros(pad_views, max_seq_len)], dim=0)

        human_kp_batch.append(human_kp)
        court_kp_batch.append(court_kp)
        human_vis_batch.append(human_vis)
        court_vis_batch.append(court_vis)
        human_mask_batch.append(human_mask)
        position_batch.append(position)
        rotation_batch.append(rotation)
        if human_kp_3d is not None:
            human_kp_3d_batch.append(human_kp_3d)

    collated: dict[str, Tensor] = {
        "human_kp": torch.stack(human_kp_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "human_vis": torch.stack(human_vis_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "human_mask": torch.stack(human_mask_batch, dim=0),
        "position": torch.stack(position_batch, dim=0),
        "rotation": torch.stack(rotation_batch, dim=0),
    }
    if human_kp_3d_batch:
        collated["human_kp_3d"] = torch.stack(human_kp_3d_batch, dim=0)

    return collated


def adapt_batch_for_model_profile(
    batch: dict[str, Tensor],
    *,
    input_profile: str,
    camera_index: int = 0,
) -> dict[str, Tensor]:
    """Adapt canonical ``(B,N,T,...)`` batch to model-specific input profile."""
    b, n, _t = batch["human_kp"].shape[:3]
    del b
    if camera_index < 0 or camera_index >= n:
        raise ValueError(
            f"camera_index={camera_index} is out of range for batch with N={n} views."
        )

    if input_profile == "multiview":
        return batch

    if input_profile == "frame":
        adapted: dict[str, Tensor] = {
            "human_kp": batch["human_kp"][:, camera_index, 0],
            "court_kp": batch["court_kp"][:, camera_index, 0],
            "human_vis": batch["human_vis"][:, camera_index, 0],
            "court_vis": batch["court_vis"][:, camera_index, 0],
            "human_mask": batch["human_mask"][:, camera_index, 0],
            "position": batch["position"][:, 0],
            "rotation": batch["rotation"][:, 0],
        }
        if "human_kp_3d" in batch:
            adapted["human_kp_3d"] = batch["human_kp_3d"][:, 0]
        return adapted

    if input_profile == "sequence":
        adapted = {
            "human_kp": batch["human_kp"][:, camera_index],
            "court_kp": batch["court_kp"][:, camera_index],
            "human_vis": batch["human_vis"][:, camera_index],
            "court_vis": batch["court_vis"][:, camera_index],
            "human_mask": batch["human_mask"][:, camera_index],
            "position": batch["position"],
            "rotation": batch["rotation"],
        }
        if "human_kp_3d" in batch:
            adapted["human_kp_3d"] = batch["human_kp_3d"]
        return adapted

    raise ValueError(
        "Unknown model input profile: "
        f"{input_profile}. Supported: ['frame', 'sequence', 'multiview']"
    )


def collate_and_adapt_plcs_batch(
    batch: list[dict[str, Tensor]],
    *,
    input_profile: str,
    camera_index: int = 0,
) -> dict[str, Tensor]:
    """Collate canonical PLCS samples and adapt to model-specific input profile."""
    collated = collate_plcs_batch(batch)
    if not isinstance(collated, dict):
        raise TypeError("Expected dict[str, Tensor] from collate_plcs_batch.")
    return adapt_batch_for_model_profile(
        collated,
        input_profile=input_profile,
        camera_index=camera_index,
    )
