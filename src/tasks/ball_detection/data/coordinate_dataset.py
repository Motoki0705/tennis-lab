"""Read aligned 32-image windows at fixed native/half/quarter sampling rates."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.utils.checksum import dual_sha256

from .annotation_states import AnnotationStates, annotation_states
from .coordinate_manifest import COORDINATE_MANIFEST_SCHEMA, clip_semantic_digest
from .store import BallFrameStore, ClipRecord, shard_name
from .temporal_sampling import SampledWindow, TemporalSamplingConfig


class CoordinateWindowDataset(Dataset[dict[str, Any]]):
    """Modality is explicit. MDD-only never opens a pose manifest or artifact."""

    def __init__(self, manifest: Path, *, split: str, requires_pose: bool) -> None:
        self.manifest = json.loads(manifest.read_text())
        schema = self.manifest.get("schema")
        if schema == "ball_play_windows.v1":
            self.input_kind = "mdd_pose"
            self.frame_steps: tuple[int, ...] = (1,)
            for record in self.manifest["clips"]:
                record["windows"] = [dict(start=start, frame_step=1) for start in record["windows"]]
                record["common_evaluation"] = True
        elif schema == COORDINATE_MANIFEST_SCHEMA:
            self.input_kind = self.manifest["input_kind"]
            self.frame_steps = TemporalSamplingConfig(tuple(self.manifest["sampling"]["frame_steps"])).frame_steps
            semantics = self.manifest["semantics"]
            if (semantics["mdd"] != "recompute_after_rgb_subsampling" or semantics["timestamps"] != "real_seconds"
                    or semantics["start_stride"] != "sampled_frames" or semantics["window_length"] != 32):
                raise ValueError("Unsupported coordinate sampling/MDD semantics")
        else:
            raise ValueError("Unsupported frozen coordinate manifest schema")
        expected = "mdd_pose" if requires_pose else "mdd_only"
        if self.input_kind != expected:
            raise ValueError(f"Model requires {expected}, manifest declares {self.input_kind}")
        if self.manifest["config"]["window_length"] != 32 or split not in {"train", "val", "test"}:
            raise ValueError("Expected a declared split and 32 sampled frames")
        self.requires_pose = requires_pose
        root = Path(self.manifest["ball_store"]["directory"])
        for name, expected_digest in self.manifest["ball_store"]["hashes"].items():
            if dual_sha256(root / name) != expected_digest:
                raise ValueError("Ball snapshot changed")
        self.store = BallFrameStore(root)
        if not requires_pose and self.manifest.get("pose_directory") is not None:
            raise ValueError("MDD-only data must not depend on pose artifacts")
        self.pose_directory = Path(self.manifest["pose_directory"]) if requires_pose else None
        self.records = [r for r in self.manifest["clips"] if r["clip"]["split"] == split]
        self.windows = [(i, SampledWindow(**window)) for i, record in enumerate(self.records) for window in record["windows"]]
        if not self.windows:
            raise ValueError(f"No accepted windows in {split}")
        if any(window.frame_step not in self.frame_steps for _, window in self.windows):
            raise ValueError("Window uses an undeclared frame step")
        self._rgb_versions: dict[str, tuple[int, int, int, int]] = {}
        self._states: dict[str, AnnotationStates] = {}
        self._pose_clip = ""
        self._pose: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def _verify_clip(self, record: dict[str, Any], clip: ClipRecord) -> AnnotationStates:
        path = self.store.directory / "shards" / shard_name(clip.index)
        stat = path.stat()
        version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if clip.clip_id not in self._rgb_versions:
            if dual_sha256(path) != record["rgb_shard_sha256"]:
                raise ValueError("Frozen RGB shard checksum mismatch")
            if "semantic_sha256" in record and clip_semantic_digest(self.store, clip) != record["semantic_sha256"]:
                raise ValueError("Frozen clip GT/geometry identity mismatch")
            self._rgb_versions[clip.clip_id] = version
            self._states[clip.clip_id] = annotation_states(self.store, clip)
        elif self._rgb_versions[clip.clip_id] != version:
            raise ValueError("RGB shard changed during dataset use")
        return self._states[clip.clip_id]

    def _read_pose(self, record: dict[str, Any], clip: ClipRecord, indices: NDArray[np.int64]) -> dict[str, torch.Tensor]:
        if self.pose_directory is None:
            raise ValueError("Pose loading is not available for MDD-only windows")
        if self._pose_clip != clip.clip_id:
            entry = record["pose"]
            if entry["pose_status"] != "approved":
                raise ValueError("Frozen pose is not approved")
            path = self.pose_directory / entry["file"]
            if not path.resolve().is_relative_to(self.pose_directory.resolve()) or dual_sha256(path) != entry["sha256"]:
                raise ValueError("Frozen pose identity mismatch")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            rows = self.store.clip_rows(clip)
            for key in ("frame_index", "pts"):
                if not np.array_equal(arrays[key], self.store.frames[key][rows]):
                    raise ValueError("Pose and image timelines disagree")
            self._pose, self._pose_clip = arrays, clip.clip_id
        denominator = clip.scale * np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        points = self._pose["keypoints"][indices, :, :, :2].astype(np.float32) / denominator
        person_valid = self._pose["observed"][indices]
        if person_valid.dtype != np.bool_ or points.shape[0] != 32 or points.shape[-2:] != (17, 2):
            raise ValueError("Invalid frozen pose shapes or observed mask")
        valid = np.broadcast_to(person_valid[..., None], points.shape[:-1]).copy()
        points[~valid] = 0
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite pose coordinates")
        return dict(pose=torch.from_numpy(points), pose_valid=torch.from_numpy(valid))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, window = self.windows[index]
        record = self.records[record_index]
        clip = self.store.clip_by_id(record["clip"]["clip_id"])
        indices = window.indices()
        if indices[-1] >= clip.frame_count or not any(a <= indices[0] and indices[-1] < b for a, b in record["play"]):
            raise ValueError("Sampled window leaves its declared play span")
        states = self._verify_clip(record, clip)
        position_valid = states.supervision[indices].copy()
        cfg = self.manifest["config"]
        if (not states.target[indices].all() or position_valid.sum() < cfg["min_observed_frames"]
                or states.evidence[indices].mean() < cfg["min_presence_fraction"]):
            raise ValueError("Sampled window no longer satisfies declared teacher/evidence requirements")
        rgb = np.empty((32, 3, clip.height, clip.width), np.uint8)
        uv: NDArray[np.float32] = np.zeros((32, 2), np.float32)
        denominator = clip.scale * np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        for position, frame in enumerate(indices):
            row = self.store.row_of(clip, int(frame))
            image = self.store.read_bgr(row)
            # Copy BGR HWC directly into the final RGB CHW uint8 allocation.
            rgb[position] = image[..., ::-1].transpose(2, 0, 1)
            if position_valid[position]:
                uv[position] = states.xy[frame] / denominator
        if not np.isfinite(uv).all() or ((uv[position_valid] < 0) | (uv[position_valid] > 1)).any():
            raise ValueError("Observed coordinates exceed the source endpoint grid")
        rows = self.store.clip_rows(clip)
        pts = self.store.frames["pts"][rows[indices]]
        seconds = ((pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
        sample = dict(rgb=torch.from_numpy(rgb), timestamps=torch.from_numpy(seconds), uv=torch.from_numpy(uv),
                      position_valid=torch.from_numpy(position_valid), frame_indices=torch.from_numpy(indices),
                      frame_step=window.frame_step, input_kind=self.input_kind,
                      common_evaluation=bool(record["common_evaluation"]),
                      clip_id=clip.clip_id, start=window.start, source=clip.source,
                      source_size=torch.tensor((clip.source_width, clip.source_height)))
        if self.requires_pose:
            sample.update(self._read_pose(record, clip, indices))
        return sample


def collate_coordinate_windows(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples or len({s["input_kind"] for s in samples}) != 1:
        raise ValueError("Batch must contain one nonempty coordinate input kind")
    if len({tuple(s["rgb"].shape) for s in samples}) != 1:
        raise ValueError("Batch images must share native shape; resize/padding is not automatic")
    if any(s["rgb"].dtype != torch.uint8 or s["rgb"].ndim != 4 or s["rgb"].shape[:2] != (32, 3) for s in samples):
        raise ValueError("Coordinate samples require RGB uint8 T,3,H,W")
    result: dict[str, Any] = {}
    if samples[0]["input_kind"] == "mdd_pose":
        people = max(sample["pose"].shape[1] for sample in samples)
        for key in ("pose", "pose_valid"):
            values = []
            for sample in samples:
                value = sample[key]
                padded = value.new_zeros((32, people, *value.shape[2:]))
                padded[:, :value.shape[1]] = value
                values.append(padded)
            result[key] = torch.stack(values)
    elif samples[0]["input_kind"] != "mdd_only" or any("pose" in s or "pose_valid" in s for s in samples):
        raise ValueError("MDD-only batches must not contain pose tensors")
    for key in ("rgb", "timestamps", "uv", "position_valid", "source_size", "frame_indices"):
        result[key] = torch.stack([sample[key] for sample in samples])
    for key in ("clip_id", "start", "source", "frame_step", "common_evaluation", "input_kind"):
        result[key] = [sample[key] for sample in samples]
    return result
