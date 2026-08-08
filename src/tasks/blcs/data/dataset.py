"""Dataset and collate/adaptation utilities for BLCS."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSCompactDatasetReader,
    BLCSTrackIndex,
)
from src.tasks.base.data.canonical_dataset import CanonicalDataset
from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.blcs.data.types import BLCSMultiViewBatch, BLCSMultiViewSample
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BallTrajectoryDataset(CanonicalDataset[BLCSMultiViewSample]):
    """Materialize one canonical object interval with every generated camera."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        split: str,
        config: object,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(config=config, augment=augment, rng=rng)
        self.reader = BLCSCompactDatasetReader(Path(dataset_dir))
        self.index: tuple[BLCSTrackIndex, ...] = self.reader.split_tracks(split)
        if not self.index:
            raise ValueError(f"Canonical BLCS split {split!r} is empty.")
        num_court_kp = self.data_config["num_court_kp"]
        if not isinstance(num_court_kp, int):
            raise TypeError("data.num_court_kp must be an integer.")
        self.num_court_kp = num_court_kp
        if not 1 <= self.num_court_kp <= 20:
            raise ValueError("data.num_court_kp must be within [1, 20].")
        augmentation = self.data_config["augmentation"]
        if not isinstance(augmentation, Mapping):
            raise TypeError("data.augmentation must be a mapping.")
        self.augmentation_pipeline = BLCSBallObservationAugmentation(augmentation)
        self._scale = torch.tensor(COURT_COORD_SCALE_XYZ, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> BLCSMultiViewSample:
        track = self.index[item]
        trajectory = self.reader.materialize_all_views(track.trajectory_id)
        window = self.contiguous_window(track.stop_frame - track.start_frame)
        start = track.start_frame + int(window.start or 0)
        stop = track.start_frame + int(window.stop or 0)
        frames = slice(start, stop)
        object_index = track.object_index
        ball_uv = torch.from_numpy(trajectory.ball_uv[:, frames, object_index])
        ball_vis = torch.from_numpy(
            trajectory.ball_visible[:, frames, object_index]
        ).float()
        sequence_length = stop - start
        court_kp = torch.from_numpy(trajectory.court_kp[:, : self.num_court_kp])[
            :, None
        ].expand(-1, sequence_length, -1, -1)
        court_vis = (
            torch.from_numpy(trajectory.court_visible[:, : self.num_court_kp])
            .float()[:, None]
            .expand(-1, sequence_length, -1)
        )
        sample: BLCSMultiViewSample = {
            "ball_uv": ball_uv,
            "ball_vis": ball_vis,
            "ball_mask": torch.ones_like(ball_vis),
            "court_kp": court_kp,
            "court_vis": court_vis,
            "position_3d": torch.from_numpy(
                trajectory.positions_court_m[frames, object_index]
            )
            / self._scale,
            "velocity_3d": torch.from_numpy(
                trajectory.velocities_court_mps[frames, object_index]
            ),
            "seq_len": torch.tensor(sequence_length, dtype=torch.long),
            "camera_R": torch.from_numpy(trajectory.camera_R),
            "camera_C": torch.from_numpy(trajectory.camera_C),
            "camera_f": torch.from_numpy(trajectory.camera_f),
            "camera_cx": torch.from_numpy(trajectory.camera_cx),
            "camera_cy": torch.from_numpy(trajectory.camera_cy),
            "camera_w": torch.from_numpy(trajectory.camera_w),
            "camera_h": torch.from_numpy(trajectory.camera_h),
        }
        if self.augment:
            return self.augmentation_pipeline.forward(sample)
        return sample


def collate_multiview_trajectories(
    batch: list[BLCSMultiViewSample],
) -> BLCSMultiViewBatch:
    """Collate canonical BLCS samples into padded canonical batch tensors."""
    max_views = max(int(sample["ball_uv"].shape[0]) for sample in batch)
    max_seq_len = max(int(sample["seq_len"].item()) for sample in batch)
    has_clean_targets = any(
        "ball_uv_target" in sample and "ball_vis_target" in sample for sample in batch
    )

    ball_uv_batch = []
    ball_vis_batch = []
    ball_uv_target_batch = []
    ball_vis_target_batch = []
    ball_mask_batch = []
    court_kp_batch = []
    court_vis_batch = []
    position_3d_batch = []
    velocity_3d_batch = []
    seq_len_batch = []
    cam_R_batch: list[Tensor] = []
    cam_C_batch: list[Tensor] = []
    cam_f_batch: list[Tensor] = []
    cam_cx_batch: list[Tensor] = []
    cam_cy_batch: list[Tensor] = []
    cam_w_batch: list[Tensor] = []
    cam_h_batch: list[Tensor] = []
    for sample in batch:
        n_views = int(sample["ball_uv"].shape[0])
        seq_len = int(sample["seq_len"].item())
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        ball_uv = sample["ball_uv"]
        ball_vis = sample["ball_vis"]
        ball_uv_target = (
            sample.get("ball_uv_target", ball_uv) if has_clean_targets else None
        )
        ball_vis_target = (
            sample.get("ball_vis_target", ball_vis) if has_clean_targets else None
        )
        ball_mask = sample["ball_mask"]
        court_kp = sample["court_kp"]
        court_vis = sample["court_vis"]
        position_3d = sample["position_3d"]
        velocity_3d = sample["velocity_3d"]
        n_kp = court_kp.shape[-2]

        cam_R = sample["camera_R"]
        cam_C = sample["camera_C"]
        cam_f = sample["camera_f"]
        cam_cx = sample["camera_cx"]
        cam_cy = sample["camera_cy"]
        cam_w = sample["camera_w"]
        cam_h = sample["camera_h"]

        if pad_seq > 0:
            ball_uv = torch.cat([ball_uv, torch.zeros(n_views, pad_seq, 2)], dim=1)
            ball_vis = torch.cat([ball_vis, torch.zeros(n_views, pad_seq)], dim=1)
            if has_clean_targets:
                assert ball_uv_target is not None
                assert ball_vis_target is not None
                ball_uv_target = torch.cat(
                    [ball_uv_target, torch.zeros(n_views, pad_seq, 2)],
                    dim=1,
                )
                ball_vis_target = torch.cat(
                    [ball_vis_target, torch.zeros(n_views, pad_seq)],
                    dim=1,
                )
            ball_mask = torch.cat([ball_mask, torch.zeros(n_views, pad_seq)], dim=1)
            court_kp = torch.cat(
                [court_kp, torch.zeros(n_views, pad_seq, n_kp, 2)], dim=1
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(n_views, pad_seq, n_kp)], dim=1
            )
            position_3d = torch.cat([position_3d, torch.zeros(pad_seq, 3)], dim=0)
            velocity_3d = torch.cat([velocity_3d, torch.zeros(pad_seq, 3)], dim=0)

        if pad_views > 0:
            ball_uv = torch.cat(
                [ball_uv, torch.zeros(pad_views, max_seq_len, 2)], dim=0
            )
            ball_vis = torch.cat([ball_vis, torch.zeros(pad_views, max_seq_len)], dim=0)
            if has_clean_targets:
                assert ball_uv_target is not None
                assert ball_vis_target is not None
                ball_uv_target = torch.cat(
                    [ball_uv_target, torch.zeros(pad_views, max_seq_len, 2)],
                    dim=0,
                )
                ball_vis_target = torch.cat(
                    [ball_vis_target, torch.zeros(pad_views, max_seq_len)],
                    dim=0,
                )
            ball_mask = torch.cat(
                [ball_mask, torch.zeros(pad_views, max_seq_len)], dim=0
            )
            court_kp = torch.cat(
                [court_kp, torch.zeros(pad_views, max_seq_len, n_kp, 2)], dim=0
            )
            court_vis = torch.cat(
                [court_vis, torch.zeros(pad_views, max_seq_len, n_kp)], dim=0
            )
            # Pad camera parameters with zeros for extra views
            cam_R = torch.cat([cam_R, torch.zeros(pad_views, 3, 3)], dim=0)
            cam_C = torch.cat([cam_C, torch.zeros(pad_views, 3)], dim=0)
            cam_f = torch.cat([cam_f, torch.zeros(pad_views)], dim=0)
            cam_cx = torch.cat([cam_cx, torch.zeros(pad_views)], dim=0)
            cam_cy = torch.cat([cam_cy, torch.zeros(pad_views)], dim=0)
            cam_w = torch.cat(
                [cam_w, torch.ones(pad_views)], dim=0
            )  # ones to avoid div-by-zero
            cam_h = torch.cat([cam_h, torch.ones(pad_views)], dim=0)

        ball_uv_batch.append(ball_uv)
        ball_vis_batch.append(ball_vis)
        if has_clean_targets:
            assert ball_uv_target is not None
            assert ball_vis_target is not None
            ball_uv_target_batch.append(ball_uv_target)
            ball_vis_target_batch.append(ball_vis_target)
        ball_mask_batch.append(ball_mask)
        court_kp_batch.append(court_kp)
        court_vis_batch.append(court_vis)
        position_3d_batch.append(position_3d)
        velocity_3d_batch.append(velocity_3d)
        seq_len_batch.append(sample["seq_len"])
        cam_R_batch.append(cam_R)
        cam_C_batch.append(cam_C)
        cam_f_batch.append(cam_f)
        cam_cx_batch.append(cam_cx)
        cam_cy_batch.append(cam_cy)
        cam_w_batch.append(cam_w)
        cam_h_batch.append(cam_h)
    collated = {
        "ball_uv": torch.stack(ball_uv_batch, dim=0),
        "ball_vis": torch.stack(ball_vis_batch, dim=0),
        "ball_mask": torch.stack(ball_mask_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "position_3d": torch.stack(position_3d_batch, dim=0),
        "velocity_3d": torch.stack(velocity_3d_batch, dim=0),
        "seq_len": torch.stack(seq_len_batch, dim=0),
        "camera_R": torch.stack(cam_R_batch, dim=0),
        "camera_C": torch.stack(cam_C_batch, dim=0),
        "camera_f": torch.stack(cam_f_batch, dim=0),
        "camera_cx": torch.stack(cam_cx_batch, dim=0),
        "camera_cy": torch.stack(cam_cy_batch, dim=0),
        "camera_w": torch.stack(cam_w_batch, dim=0),
        "camera_h": torch.stack(cam_h_batch, dim=0),
    }
    if has_clean_targets:
        collated["ball_uv_target"] = torch.stack(ball_uv_target_batch, dim=0)
        collated["ball_vis_target"] = torch.stack(ball_vis_target_batch, dim=0)
    return cast("BLCSMultiViewBatch", collated)
