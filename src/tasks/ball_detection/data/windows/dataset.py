"""Read frozen legacy windows or native all-frame windows and shared packs."""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, default_collate

from src.tasks.ball_detection.data.annotation_states import (
    AnnotationStates,
    annotation_states,
)
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    resolve_frame_supervision,
)
from src.tasks.ball_detection.data.windows.all_frame_manifest import (
    ALL_FRAME_MANIFEST_SCHEMA,
    validate_all_frame_manifest,
)
from src.tasks.ball_detection.data.windows.all_frames import (
    AllFrameWindowConfig,
    child_starts,
    decode_label_runs,
)
from src.tasks.ball_detection.data.windows.hybrid import (
    HYBRID_MANIFEST_SCHEMA,
    AbsencePolicy,
    negative_mask,
    reviewed_empty_evidence,
)
from src.tasks.ball_detection.data.windows.manifest import (
    COORDINATE_MANIFEST_SCHEMA,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import (
    SampledWindow,
    TemporalSamplingConfig,
)
from src.utils.checksum import dual_sha256
from src.utils.data.file_ranges import FileRangeReader
from src.utils.data.verification import (
    FileVersion,
    SharedFileVerification,
    file_version,
)


class FrozenBallWindowDataset(Dataset[dict[str, Any]]):
    """Modality is explicit. MDD-only never opens a pose manifest or artifact."""

    def __init__(self, manifest: Path, *, split: str, requires_pose: bool, jpeg_decoder: str = "opencv",
                 verify_empty_annotations: bool = False) -> None:
        if jpeg_decoder not in {"opencv", "nvjpeg"}:
            raise ValueError("JPEG decoder must be explicit: opencv or nvjpeg")
        self.jpeg_decoder = jpeg_decoder
        self.verify_empty_annotations = verify_empty_annotations
        self._jpeg_reader = FileRangeReader()
        self.manifest = json.loads(manifest.read_text())
        schema = self.manifest.get("schema")
        self.all_frames = schema == ALL_FRAME_MANIFEST_SCHEMA
        self.window_length = 32
        self.all_frame_config: AllFrameWindowConfig | None = None
        if not self.all_frames and (
            {"mode", "label_kinds", "summaries"} & self.manifest.keys()
            or any({"packs", "common_intervals", "label_runs"} & r.keys() for r in self.manifest.get("clips", []))
        ):
            raise ValueError("Legacy/new manifest schemas cannot be mixed")
        if self.all_frames:
            if (manifest.parent / "PREPARATION_INCOMPLETE").exists():
                raise ValueError("All-frame preparation is incomplete")
            if set(self.manifest.get("config", {})) != set(AllFrameWindowConfig.__dataclass_fields__):
                raise ValueError("All-frame configuration cannot contain legacy schema fields")
            self.all_frame_config = AllFrameWindowConfig(**self.manifest["config"])
            self.window_length = self.all_frame_config.window_length
            self.input_kind = self.manifest["input_kind"]
            self.frame_steps: tuple[int, ...] = (1,)
        elif schema == "ball_play_windows.v1":
            self.input_kind = "mdd_pose"
            self.frame_steps = (1,)
            for record in self.manifest["clips"]:
                record["windows"] = [dict(start=start, frame_step=1) for start in record["windows"]]
                record["common_evaluation"] = True
        elif schema in {COORDINATE_MANIFEST_SCHEMA, HYBRID_MANIFEST_SCHEMA}:
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
        if split not in {"train", "val", "test"}:
            raise ValueError("Expected a declared train/val/test split")
        if not self.all_frames and self.manifest["config"]["window_length"] != 32:
            raise ValueError("Legacy manifests require 32 frames; legacy/new schemas cannot be mixed")
        self.requires_pose = requires_pose
        root = Path(self.manifest["ball_store"]["directory"])
        for name, expected_digest in self.manifest["ball_store"]["hashes"].items():
            if dual_sha256(root / name) != expected_digest:
                raise ValueError("Ball snapshot changed")
        self.store = BallFrameStore(root)
        if self.all_frames:
            validate_all_frame_manifest(self.manifest, self.store)
        self._heatmap_supervision = resolve_frame_supervision(self.store, OBSERVED_ONLY).supervised
        if not requires_pose and self.manifest.get("pose_directory") is not None:
            raise ValueError("MDD-only data must not depend on pose artifacts")
        self.pose_directory = Path(self.manifest["pose_directory"]) if requires_pose else None
        self.records = [r for r in self.manifest["clips"] if r["clip"]["split"] == split]
        self.pack_windows: list[tuple[int, ...]] = []
        self.window_metadata: list[dict[str, Any]] = []
        self.windows: list[tuple[int, SampledWindow]] = []
        if self.all_frames:
            assert self.all_frame_config is not None
            for i, record in enumerate(self.records):
                if self.manifest["mode"] == "train":
                    for pack in record["packs"]:
                        members = []
                        starts = child_starts(pack["start"], self.all_frame_config)
                        for child_index, first in enumerate(starts):
                            members.append(len(self.windows))
                            self.windows.append((i, SampledWindow(first, 1)))
                            self.window_metadata.append(dict(
                                interval_index=pack["interval_index"], pack_index=len(self.pack_windows),
                                pack_id=f"{record['clip']['clip_id']}:{pack['start']}:{self.all_frame_config.pack_length}",
                                pack_start=pack["start"], child_index=child_index, children_per_pack=len(starts),
                            ))
                        self.pack_windows.append(tuple(members))
                else:
                    for window in record["windows"]:
                        self.windows.append((i, SampledWindow(window["start"], 1)))
                        self.window_metadata.append(dict(window, pack_index=None, pack_id=None,
                                                         pack_start=None, child_index=None, children_per_pack=None))
        else:
            self.windows = [(i, SampledWindow(**window)) for i, record in enumerate(self.records) for window in record["windows"]]
        self.absence_policy = AbsencePolicy.from_mapping(self.manifest["absence"]) if schema == HYBRID_MANIFEST_SCHEMA else None
        self.window_sources = [source for r in self.records for source in r["window_sources"]] if self.absence_policy else []
        if self.absence_policy and any(len(r["windows"]) != len(r["window_sources"]) for r in self.records):
            raise ValueError("Hybrid window sources must align with windows")
        if not self.windows:
            raise ValueError(f"No accepted windows in {split}")
        if any(window.frame_step not in self.frame_steps for _, window in self.windows):
            raise ValueError("Window uses an undeclared frame step")
        self._shared_rgb_verification = SharedFileVerification({
            self.store.image_path(self.store.clip_by_id(record["clip"]["clip_id"])): record["rgb_shard_sha256"]
            for record in self.records
        })
        for record in self.records:
            video = self.store.video_frames.get(record["clip"]["clip_id"])
            if video and video.media_sha256 != record["rgb_shard_sha256"]:
                raise ValueError("Frozen video hash differs from its publication contract")
        self._rgb_versions: dict[str, FileVersion] = {}
        self._states: dict[str, AnnotationStates] = {}
        self._reviewed_empty: dict[str, NDArray[np.bool_]] = {}
        self._label_kinds: dict[str, NDArray[np.uint8]] = {}
        self._pose_clip = ""
        self._pose: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def verify_image_files(self, workers: int, *, progress: Callable[[int, int], None] | None = None) -> None:
        self._shared_rgb_verification.verify_all(max(1, workers), progress=progress)

    def _verify_clip(self, record: dict[str, Any], clip: ClipRecord) -> AnnotationStates:
        path = self.store.image_path(clip)
        version = file_version(path)
        if clip.clip_id not in self._rgb_versions:
            version = self._shared_rgb_verification.verify(path)
            if "semantic_sha256" in record and clip_semantic_digest(self.store, clip) != record["semantic_sha256"]:
                raise ValueError("Frozen clip GT/geometry identity mismatch")
            states = annotation_states(self.store, clip)
            if self.all_frames:
                codes = decode_label_runs(record["label_runs"], clip.frame_count)
                self._label_kinds[clip.clip_id] = codes
                empty = codes == 2
            elif self.absence_policy:
                empty = np.zeros(clip.frame_count, dtype=np.bool_)
                indices = record["reviewed_empty_frames"]
                if sorted(set(indices)) != indices or any(type(i) is not int or not 0 <= i < clip.frame_count for i in indices):
                    raise ValueError("Invalid frozen reviewed-empty frame mask")
                empty[indices] = True
                if (empty & ~(states.reviewed & states.target & (states.count == 0))).any():
                    raise ValueError("Frozen reviewed-empty evidence disagrees with normalized labels")
            elif self.verify_empty_annotations:
                empty, _ = reviewed_empty_evidence(self.store, clip, states)
            else:
                # Historical coordinate consumers do not use absence supervision.
                empty = states.reviewed & states.target & (states.count == 0)
            self._rgb_versions[clip.clip_id] = version
            self._states[clip.clip_id] = states
            self._reviewed_empty[clip.clip_id] = empty
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
        if person_valid.dtype != np.bool_ or points.shape[0] != self.window_length or points.shape[-2:] != (17, 2):
            raise ValueError("Invalid frozen pose shapes or observed mask")
        valid = np.broadcast_to(person_valid[..., None], points.shape[:-1]).copy()
        points[~valid] = 0
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite pose coordinates")
        return dict(pose=torch.from_numpy(points), pose_valid=torch.from_numpy(valid))

    def _read_images(self, clip: ClipRecord, indices: NDArray[np.int64]) -> dict[str, Any]:
        images: dict[str, Any] = {}
        payloads = self.store.read_video_window(clip, indices.tolist(), version=self._rgb_versions[clip.clip_id]) \
            if clip.clip_id in self.store.video_frames else None
        if self.jpeg_decoder == "opencv":
            rgb: NDArray[np.uint8] = np.empty((self.window_length, 3, clip.height, clip.width), np.uint8)
            for position, frame in enumerate(indices):
                image = self.store.read_bgr(self.store.row_of(clip, int(frame))) if payloads is None else cv2.imdecode(
                    np.frombuffer(payloads[position], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (clip.height, clip.width, 3):
                    raise ValueError("JPEG failed to decode to its native dimensions")
                rgb[position] = image[..., ::-1].transpose(2, 0, 1)
            images["rgb"] = torch.from_numpy(rgb)
        else:
            rows = self.store.clip_start[clip.index] + indices
            ranges = [(int(self.store.frames["offset"][row]), int(self.store.frames["length"][row])) for row in rows]
            path = self.store.image_path(clip)
            packed = self._jpeg_reader.read(path, ranges, self._rgb_versions[clip.clip_id]) if payloads is None else \
                np.frombuffer(b"".join(payloads), dtype=np.uint8).copy()
            images.update(jpeg=torch.from_numpy(packed),
                          jpeg_lengths=tuple(length for _, length in ranges),
                          image_shape=(self.window_length, 3, clip.height, clip.width))
        return images

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, window = self.windows[index]
        record = self.records[record_index]
        clip = self.store.clip_by_id(record["clip"]["clip_id"])
        indices = window.indices(self.window_length)
        spans = record["common_intervals"] if self.all_frames else record["eligible"] if self.absence_policy else record["play"]
        if indices[-1] >= clip.frame_count or not any(a <= indices[0] and indices[-1] < b for a, b in spans):
            raise ValueError("Sampled window leaves its declared play span or eligible span")
        states = self._verify_clip(record, clip)
        position_valid = states.supervision[indices].copy()
        cfg = self.manifest["config"]
        if self.all_frames:
            codes = self._label_kinds[clip.clip_id][indices]
            position_valid = codes == 1
            heatmap_valid = codes != 0
        elif self.absence_policy:
            negative = negative_mask(states, self.absence_policy, self._reviewed_empty[clip.clip_id])[indices]
            heatmap_valid = position_valid | negative
            source = self.window_sources[index]
            in_play = any(a <= indices[0] and indices[-1] < b for a, b in record["play"])
            intersects_play = any(a <= indices[-1] and indices[0] < b for a, b in record["play"])
            valid_source = (
                source == "play" and in_play and position_valid.sum() >= cfg["min_observed_frames"]
                and states.evidence[indices].mean() >= cfg["min_presence_fraction"]
            ) or (source == "absent" and negative.all()) or (
                source == "nonplay" and not intersects_play and heatmap_valid.any() and not negative.all()
            )
            if not valid_source or not states.target[indices[0]:indices[-1] + 1].all():
                raise ValueError("Hybrid window violates its declared supervision source")
        else:
            if (not states.target[indices].all() or position_valid.sum() < cfg["min_observed_frames"]
                    or states.evidence[indices].mean() < cfg["min_presence_fraction"]):
                raise ValueError("Sampled window no longer satisfies declared teacher/evidence requirements")
            rows = self.store.clip_rows(clip)
            heatmap_valid = self._heatmap_supervision[rows[indices]] & states.target[indices] & (states.count[indices] <= 1)
            heatmap_valid &= (states.count[indices] != 0) | self._reviewed_empty[clip.clip_id][indices]
        images = self._read_images(clip, indices)
        uv: NDArray[np.float32] = np.zeros((self.window_length, 2), np.float32)
        denominator = clip.scale * np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        uv[position_valid] = states.xy[indices[position_valid]] / denominator
        if not np.isfinite(uv).all() or ((uv[position_valid] < 0) | (uv[position_valid] > 1)).any():
            raise ValueError("Observed coordinates exceed the source endpoint grid")
        rows = self.store.clip_rows(clip)
        pts = self.store.frames["pts"][rows[indices]]
        seconds = ((pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
        if not np.isfinite(seconds).all() or (np.diff(seconds) <= 0).any():
            raise ValueError("Sampled timestamps must be finite and strictly increasing")
        sample = dict(**images, timestamps=torch.from_numpy(seconds), uv=torch.from_numpy(uv),
                      position_valid=torch.from_numpy(position_valid),
                      heatmap_valid=torch.from_numpy(heatmap_valid.copy()),
                      frame_indices=torch.from_numpy(indices),
                      frame_step=window.frame_step, input_kind=self.input_kind,
                      common_evaluation=False if self.all_frames else bool(record["common_evaluation"]),
                      clip_id=clip.clip_id, start=window.start, source=clip.source,
                      source_size=torch.tensor((clip.source_width, clip.source_height)))
        if self.all_frames:
            metadata = self.window_metadata[index]
            a, b = record["common_intervals"][metadata["interval_index"]]
            distance = np.minimum(indices - a, b - 1 - indices)
            owner: NDArray[np.bool_] = np.zeros(self.window_length, dtype=np.bool_)
            if self.manifest["mode"] == "evaluation":
                left, right = metadata["owner_range"]
                owner = (indices >= left) & (indices < right)
            native_pts = self.store.frames["pts"][rows]
            edge_seconds = np.minimum(pts - native_pts[a], native_pts[b - 1] - pts).astype(np.float64) * float(Fraction(clip.time_base))
            sample.update(
                manifest_schema=ALL_FRAME_MANIFEST_SCHEMA, manifest_mode=self.manifest["mode"],
                label_kind=torch.from_numpy(codes.copy()), pts=torch.from_numpy(pts.copy()),
                native_fps=str(Fraction(clip.fps)), time_base=clip.time_base,
                interval_start=a, interval_stop=b, edge_distance=torch.from_numpy(distance),
                edge_bin=torch.from_numpy(np.searchsorted([1, 8, 16, 32, 64], distance, side="right")),
                edge_distance_seconds=torch.from_numpy(edge_seconds),
                owner_mask=torch.from_numpy(owner), score_mask=torch.from_numpy(owner & heatmap_valid),
                **{key: metadata[key] for key in ("pack_index", "pack_id", "pack_start", "child_index", "children_per_pack")},
            )
        if self.requires_pose:
            sample.update(self._read_pose(record, clip, indices))
        return sample


def collate_window_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples or len({s["input_kind"] for s in samples}) != 1:
        raise ValueError("Batch must contain one nonempty coordinate input kind")
    all_frames = [s.get("manifest_schema") == ALL_FRAME_MANIFEST_SCHEMA for s in samples]
    if len(set(all_frames)) != 1:
        raise ValueError("Cannot mix legacy and all-frame manifest schemas in a batch")
    encoded = ["jpeg" in s for s in samples]
    if len(set(encoded)) != 1 or any(("rgb" in s) == encoded[i] for i, s in enumerate(samples)):
        raise ValueError("Batch must contain exactly one image representation")
    shapes = [s["image_shape"] if encoded[0] else tuple(s["rgb"].shape) for s in samples]
    if len(set(shapes)) != 1:
        raise ValueError("Batch images must share native shape; resize/padding is not automatic")
    shape = shapes[0]
    if (len(shape) != 4 or shape[0] < 2 or shape[1] != 3
            or (not all_frames[0] and shape[0] != 32)):
        raise ValueError("Expected native T,3,H,W; legacy samples require T=32")
    length = shape[0]
    result: dict[str, Any] = {}
    if encoded[0]:
        for sample in samples:
            lengths = sample["jpeg_lengths"]
            if (sample["jpeg"].dtype != torch.uint8 or sample["jpeg"].ndim != 1
                    or len(lengths) != length or min(lengths) < 1 or sum(lengths) != sample["jpeg"].numel()
                    or sample["image_shape"][:2] != (length, 3)):
                raise ValueError("Invalid packed JPEG window")
        result.update(jpeg=torch.cat([s["jpeg"] for s in samples]),
                      jpeg_lengths=tuple(length for s in samples for length in s["jpeg_lengths"]),
                      image_shape=(len(samples), *shapes[0]))
    else:
        if any(s["rgb"].dtype != torch.uint8 or s["rgb"].ndim != 4 or s["rgb"].shape[:2] != (length, 3) for s in samples):
            raise ValueError("Coordinate samples require RGB uint8 T,3,H,W")
        result["rgb"] = default_collate([s["rgb"] for s in samples])
    if samples[0]["input_kind"] == "mdd_pose":
        people = max(sample["pose"].shape[1] for sample in samples)
        for key in ("pose", "pose_valid"):
            values = []
            for sample in samples:
                value = sample[key]
                padded = value.new_zeros((length, people, *value.shape[2:]))
                padded[:, :value.shape[1]] = value
                values.append(padded)
            result[key] = torch.stack(values)
    elif samples[0]["input_kind"] != "mdd_only" or any("pose" in s or "pose_valid" in s for s in samples):
        raise ValueError("MDD-only batches must not contain pose tensors")
    for key in ("timestamps", "uv", "position_valid", "source_size", "frame_indices"):
        # PyTorch allocates the result directly in shared memory inside workers.
        # A plain stack would be copied again by multiprocessing's tensor sender.
        result[key] = default_collate([sample[key] for sample in samples])
    if any("heatmap_valid" in sample for sample in samples):
        if not all("heatmap_valid" in sample for sample in samples):
            raise ValueError("Cannot mix heatmap and coordinate supervision contracts")
        result["heatmap_valid"] = default_collate([sample["heatmap_valid"] for sample in samples])
    for key in ("clip_id", "start", "source", "frame_step", "common_evaluation", "input_kind"):
        result[key] = [sample[key] for sample in samples]
    if all_frames[0]:
        if len({s["manifest_mode"] for s in samples}) != 1:
            raise ValueError("Cannot mix all-frame train and evaluation windows")
        for key in ("label_kind", "pts", "edge_distance", "edge_bin", "edge_distance_seconds", "owner_mask", "score_mask"):
            result[key] = default_collate([s[key] for s in samples])
        for key in ("manifest_schema", "manifest_mode", "native_fps", "time_base", "interval_start", "interval_stop",
                    "pack_index", "pack_id", "pack_start", "child_index", "children_per_pack"):
            result[key] = [s[key] for s in samples]
    return result


class FrozenBallPackDataset(Dataset[list[dict[str, Any]]]):
    """Index shared packs, preserving all children, including zero-teacher ones."""

    def __init__(self, manifest: Path, *, split: str, jpeg_decoder: str = "opencv") -> None:
        self.window_dataset = FrozenBallWindowDataset(manifest, split=split, requires_pose=False, jpeg_decoder=jpeg_decoder)
        if not self.window_dataset.all_frames or self.window_dataset.manifest["mode"] != "train":
            raise ValueError("Pack datasets require an all-frame training manifest")

    def __len__(self) -> int:
        return len(self.window_dataset.pack_windows)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return [self.window_dataset[i] for i in self.window_dataset.pack_windows[index]]


def collate_pack_samples(packs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Flatten complete packs and retain boundaries for each sampled occurrence."""
    if not packs or any(not p for p in packs):
        raise ValueError("Pack batches must contain nonempty complete packs")
    offsets = [0]
    for pack in packs:
        if (any(s.get("manifest_schema") != ALL_FRAME_MANIFEST_SCHEMA or s.get("manifest_mode") != "train" for s in pack)
                or len({s["pack_id"] for s in pack}) != 1
                or [s["child_index"] for s in pack] != list(range(len(pack)))
                or any(s["children_per_pack"] != len(pack) for s in pack)):
            raise ValueError("Each pack occurrence must contain every declared child in order")
        offsets.append(offsets[-1] + len(pack))
    result = collate_window_samples([s for pack in packs for s in pack])
    result["pack_offsets"] = tuple(offsets)
    return result
