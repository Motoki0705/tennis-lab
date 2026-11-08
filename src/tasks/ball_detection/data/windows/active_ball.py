"""Native active-ball training windows over an immutable v3-derived snapshot."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, default_collate

from src.tasks.ball_detection.annotation.video import usage_masks
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord, Split
from src.utils.data.verification import SharedFileVerification


def collate_active_ball_windows(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Batch tensor teachers while retaining optional per-frame provenance.

    Native dimensions must already agree; no implicit resize or padding. Plain
    default_collate cannot represent the null position provenance of negatives
    and unknown frames, so metadata stays as one item per input window.
    """
    if not samples or any(s.get("target_semantics") != "active_ball" for s in samples):
        raise ValueError("Batch requires explicit active_ball target_semantics")
    keys = samples[0].keys()
    if any(s.keys() != keys for s in samples):
        raise ValueError(
            "Active-ball samples must have the same teacher/metadata fields"
        )
    shapes = {tuple(s["rgb"].shape) for s in samples}
    if len(shapes) != 1:
        raise ValueError(
            "Batch images must share native shape; resize/padding is not automatic"
        )
    return {
        key: default_collate([s[key] for s in samples])
        if isinstance(samples[0][key], torch.Tensor)
        else [s[key] for s in samples]
        for key in keys
    }


class ActiveBallWindowDataset(Dataset[dict[str, Any]]):
    """Return usable estimates, direct observations and explicit negative masks.

    The dataset consumes native uint8 RGB and endpoint-normalized coordinates,
    like FrozenBallWindowDataset. It requires the active_ball contract rather
    than silently applying a legacy observed-only training or play selector.
    Every requested split frame is covered; no padding, resampling or new
    inference is performed. The caller chooses batching/model/loss and training.
    """

    def __init__(
        self,
        store: BallFrameStore,
        *,
        split: Split,
        window_length: int,
        stride: int | None = None,
        target_semantics: str = "active_ball",
    ) -> None:
        if target_semantics != "active_ball":
            raise ValueError("active_ball reader requires matching target_semantics")
        store.require_target_semantics(target_semantics)
        stride = window_length if stride is None else stride
        if (
            type(window_length) is not int
            or type(stride) is not int
            or not 1 <= stride <= window_length
        ):
            raise ValueError("Window length and stride must cover every native frame")
        if split not in {"train", "val", "test"}:
            raise ValueError("Expected train/val/test split")
        self.store = store
        self.window_length = window_length
        self.windows: list[tuple[ClipRecord, int]] = []
        for clip in store.split_clips(split):
            if clip.frame_count < window_length:
                raise ValueError(
                    f"{clip.clip_id}: clip is shorter than the requested window"
                )
            starts = list(range(0, clip.frame_count - window_length + 1, stride))
            if starts[-1] != clip.frame_count - window_length:
                starts.append(clip.frame_count - window_length)
            self.windows.extend((clip, start) for start in starts)
        if not self.windows:
            raise ValueError(f"No active_ball windows in {split}")
        self._images = SharedFileVerification(
            {store.image_path(clip): clip.media_sha256 for clip, _ in self.windows}
        )
        self._usage = {
            clip.clip_id: usage_masks(store.video_annotations[clip.clip_id])
            for clip in store.split_clips(split)
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip, start = self.windows[index]
        self._images.verify(self.store.image_path(clip))
        stop = start + self.window_length
        annotation = self.store.video_annotations[clip.clip_id]
        frames = annotation.frames[start:stop]
        masks = self._usage[clip.clip_id]
        rows = self.store.clip_rows(clip)[start:stop]
        assert self.store.teacher_masks is not None
        position_valid = self.store.teacher_masks.coordinate[rows] == 1
        heatmap_valid = self.store.teacher_masks.heatmap[rows] != 0
        xy: NDArray[np.float32] = np.zeros((self.window_length, 2), np.float32)
        for i, frame in enumerate(frames):
            if position_valid[i]:
                assert frame.position is not None
                xy[i] = frame.position.x, frame.position.y
        # A one-pixel axis has only coordinate zero and remains zero normalized.
        uv = xy / np.asarray(
            (max(1, clip.width - 1), max(1, clip.height - 1)), np.float32
        )
        pts = self.store.frames["pts"][rows].copy()
        seconds = (
            (pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))
        ).astype(np.float32)
        rgb = np.stack(
            [
                self.store.read_bgr(int(row))[..., ::-1].transpose(2, 0, 1)
                for row in rows
            ]
        )
        self._images.verify(self.store.image_path(clip))
        return dict(
            rgb=torch.from_numpy(rgb),
            timestamps=torch.from_numpy(seconds),
            xy=torch.from_numpy(xy),
            uv=torch.from_numpy(uv),
            position_valid=torch.from_numpy(position_valid),
            heatmap_valid=torch.from_numpy(heatmap_valid),
            position_weight=torch.from_numpy(position_valid.astype(np.float32)),
            heatmap_weight=torch.from_numpy(heatmap_valid.astype(np.float32)),
            direct_observation=torch.tensor(masks.direct_observation[start:stop]),
            estimated=torch.tensor(masks.estimated[start:stop]),
            active_ball_negative=torch.tensor(
                masks.active_ball_negative_candidate[start:stop]
            ),
            state=tuple(f.state for f in frames),
            occlusion=tuple(f.occlusion for f in frames),
            position_provenance=tuple(
                f.position.provenance.model_dump(mode="json") if f.position else None
                for f in frames
            ),
            frame_indices=torch.tensor([f.frame_index for f in frames]),
            pts=torch.from_numpy(pts),
            time_base=clip.time_base,
            native_fps=clip.fps,
            source_size=torch.tensor((clip.source_width, clip.source_height)),
            clip_id=clip.clip_id,
            start=start,
            source=clip.source,
            camera_id=clip.camera_id,
            group_id=clip.group_id,
            recording_id=annotation.source.recording_id,
            split=clip.split,
            target_semantics="active_ball",
            frame_step=1,
            input_kind="mdd_only",
        )
