"""Real 32-frame windows from a frozen, reviewed play proposal manifest."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.play_manifest import clip_evidence
from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.model_io.mdd import luminance_to_mdd, mdd_coefficients
from src.utils.checksum import dual_sha256


class PoseWindowDataset(Dataset[dict[str, Any]]):
    """Only snapshot-approved artifacts are read; later approvals are irrelevant."""

    def __init__(self, manifest: Path, *, split: str, mdd_a: float, mdd_b: float) -> None:
        self.manifest = json.loads(manifest.read_text())
        if self.manifest["schema"] != "ball_play_windows.v1" or self.manifest["config"]["window_length"] != 32:
            raise ValueError("Expected a frozen 32-frame play-window manifest")
        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid split")
        root = Path(self.manifest["ball_store"]["directory"])
        for name, expected in self.manifest["ball_store"]["hashes"].items():
            if dual_sha256(root / name) != expected:
                raise ValueError("Ball snapshot changed")
        self.store = BallFrameStore(root)
        self.poses = Path(self.manifest["pose_directory"])
        self.gain, self.offset = mdd_coefficients(mdd_a, mdd_b)
        self.records = [r for r in self.manifest["clips"] if r["clip"]["split"] == split]
        self.windows = [(i, start) for i, r in enumerate(self.records) for start in r["windows"]]
        if not self.windows:
            raise ValueError(f"No accepted windows in {split}")
        self._pose_cache: dict[str, np.ndarray] = {}
        self._cached_clip = ""
        self._rgb_stats: dict[str, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start = self.windows[index]
        record = self.records[record_index]
        clip = self.store.clip_by_id(record["clip"]["clip_id"])
        shard = self.store.directory / "shards" / shard_name(clip.index)
        stat = shard.stat()
        state = (stat.st_size, stat.st_mtime_ns)
        if clip.clip_id not in self._rgb_stats:
            if dual_sha256(shard) != record["rgb_shard_sha256"]:
                raise ValueError("Frozen RGB shard checksum mismatch")
            self._rgb_stats[clip.clip_id] = state
        elif self._rgb_stats[clip.clip_id] != state:
            raise ValueError("RGB shard changed during dataset use")
        if self._cached_clip != clip.clip_id:
            entry = record["pose"]
            path = self.poses / entry["file"]
            if not path.resolve().is_relative_to(self.poses.resolve()) or dual_sha256(path) != entry["sha256"]:
                raise ValueError("Frozen pose identity mismatch")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            rows = self.store.clip_rows(clip)
            for key in ("frame_index", "pts"):
                if not np.array_equal(arrays[key], self.store.frames[key][rows]):
                    raise ValueError("Pose and image timelines disagree")
            self._pose_cache, self._cached_clip = arrays, clip.clip_id
        arrays = self._pose_cache
        stop = start + 32
        if stop > clip.frame_count or not any(a <= start and stop <= b for a, b in record["play"]):
            raise ValueError("Window leaves the declared play interval")
        gray = np.empty((32, clip.height, clip.width), np.float32)
        uv: NDArray[np.float32] = np.zeros((32, 2), np.float32)
        _, observed, _, _ = clip_evidence(self.store, clip)
        denominator = clip.scale * np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        for t, frame in enumerate(range(start, stop)):
            row = self.store.row_of(clip, frame)
            image = self.store.read_bgr(row).astype(np.float32) / 255
            # Keep the stored resolution; no pre-MDD resize or ImageNet transform.
            gray[t] = .114 * image[..., 0] + .587 * image[..., 1] + .299 * image[..., 2]
            if observed[frame]:
                uv[t] = self.store.instances_of(row).xy[0] / denominator
        points = arrays["keypoints"][start:stop, :, :, :2].astype(np.float32) / denominator
        person_valid = arrays["observed"][start:stop]
        valid = np.broadcast_to(person_valid[..., None], points.shape[:-1]).copy()
        points[~valid] = 0
        if ((uv[observed[start:stop]] < 0) | (uv[observed[start:stop]] > 1)).any():
            raise ValueError("Observed coordinates exceed the source endpoint grid")
        if not np.isfinite(points).all() or not np.isfinite(uv).all():
            raise ValueError("Nonfinite supervised coordinates")
        pts = arrays["pts"][start:stop]
        seconds = ((pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
        mdd = luminance_to_mdd(torch.from_numpy(gray)[None], gain=self.gain, offset=self.offset)[0]
        return dict(mdd=mdd, pose=torch.from_numpy(points), pose_valid=torch.from_numpy(valid),
                    timestamps=torch.from_numpy(seconds), uv=torch.from_numpy(uv),
                    position_valid=torch.from_numpy(observed[start:stop].copy()),
                    clip_id=clip.clip_id, start=start, source_size=torch.tensor((clip.source_width, clip.source_height)))


def collate_pose_windows(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad only people, never time or image resolution."""
    if not samples:
        raise ValueError("Empty batch")
    people = max(s["pose"].shape[1] for s in samples)
    result: dict[str, Any] = {}
    for key in ("pose", "pose_valid"):
        values = []
        for sample in samples:
            x = sample[key]
            padded = x.new_zeros((32, people, *x.shape[2:]))
            padded[:, :x.shape[1]] = x
            values.append(padded)
        result[key] = torch.stack(values)
    for key in ("mdd", "timestamps", "uv", "position_valid", "source_size"):
        result[key] = torch.stack([s[key] for s in samples])
    result["clip_id"], result["start"] = [s["clip_id"] for s in samples], [s["start"] for s in samples]
    return result


def coordinate_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape != target.shape or valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Coordinate loss shape/mask mismatch")
    if not bool(valid.any()):
        raise ValueError("No observed coordinate supervision")
    return torch.nn.functional.smooth_l1_loss(prediction[valid], target[valid], beta=.01)
