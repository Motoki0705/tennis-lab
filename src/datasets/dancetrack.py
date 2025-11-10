"""DanceTrack dataset + sample/target containers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2

WindowSampler = Callable[[int, int, int], Iterable[tuple[int, int]]]


@dataclass(slots=True)
class TargetFrame:
    """Target annotations for a single timestep within a sequence."""

    center: Tensor
    size: Tensor
    track_ids: Tensor
    confidence: Tensor

    @classmethod
    def empty(cls, device: torch.device | None = None) -> TargetFrame:
        """Return a zero-sized target payload aligned with padded frames."""
        center = torch.zeros((0, 2), dtype=torch.float32, device=device)
        size = torch.zeros((0, 2), dtype=torch.float32, device=device)
        scores = torch.zeros(0, dtype=torch.float32, device=device)
        track_ids = torch.zeros(0, dtype=torch.long, device=device)
        return cls(center=center, size=size, track_ids=track_ids, confidence=scores)


@dataclass(slots=True)
class TrackingSample:
    """Raw dataset sample consumed by the collate function."""

    frames: Tensor
    targets: list[TargetFrame]
    sequence_id: str
    frame_indices: list[int]


@dataclass(slots=True)
class _Annotation:
    frame_index: int
    track_id: int
    bbox_xywh: tuple[float, float, float, float]
    confidence: float


def _to_bbox(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Return a fixed-length bbox tuple, padding/truncating as needed."""
    if len(values) >= 4:
        return (values[0], values[1], values[2], values[3])
    padded = values + (0.0,) * (4 - len(values))
    return (padded[0], padded[1], padded[2], padded[3])


@dataclass(slots=True)
class _SequenceMeta:
    name: str
    frame_paths: list[str]
    annotations: dict[int, list[_Annotation]]
    frame_rate: float
    image_size: tuple[int, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_paths": self.frame_paths,
            "frame_rate": self.frame_rate,
            "image_size": self.image_size,
            "annotations": {
                str(idx): [
                    {
                        "frame_index": ann.frame_index,
                        "track_id": ann.track_id,
                        "bbox_xywh": ann.bbox_xywh,
                        "confidence": ann.confidence,
                    }
                    for ann in anns
                ]
                for idx, anns in self.annotations.items()
            },
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> _SequenceMeta:
        annotations: dict[int, list[_Annotation]] = {}
        for key, items in data.get("annotations", {}).items():
            annotations[int(key)] = [
                _Annotation(
                    frame_index=int(item["frame_index"]),
                    track_id=int(item["track_id"]),
                    bbox_xywh=_to_bbox(tuple(float(x) for x in item["bbox_xywh"])),
                    confidence=float(item["confidence"]),
                )
                for item in items
            ]
        image_size = tuple(int(v) for v in data.get("image_size", (0, 0)))
        return cls(
            name=str(data.get("name")),
            frame_paths=[str(p) for p in data.get("frame_paths", [])],
            annotations=annotations,
            frame_rate=float(data.get("frame_rate", 0.0)),
            image_size=_to_image_size(image_size),
        )


def _to_image_size(values: tuple[int, ...]) -> tuple[int, int]:
    """Return ``(width, height)`` with a graceful fallback for malformed data."""
    if len(values) >= 2:
        return (values[0], values[1])
    padded = values + (0,) * (2 - len(values))
    return (padded[0], padded[1])


def _default_sampler(
    length: int, window: int, stride: int
) -> Iterable[tuple[int, int]]:
    if length <= 0:
        return []
    if length <= window:
        return [(0, length)]
    windows = []
    idx = 0
    while idx + window <= length:
        windows.append((idx, idx + window))
        idx += max(stride, 1)
    if windows[-1][1] < length:
        windows.append((length - window, length))
    return windows


class DancetrackDataset(Dataset[TrackingSample]):
    """Dataset that assembles variable length windows from DanceTrack sequences."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | DictConfig,
        split: str,
        window_sampler: WindowSampler | None = None,
        debug: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _to_dict(cfg)
        self.split = split
        self.window_cfg = self.cfg.get("window", {})
        self.window_size = int(self.window_cfg.get("size", 8))
        self.window_stride = int(self.window_cfg.get("stride", self.window_size))
        if self.window_size <= 0:
            msg = "window.size must be > 0"
            raise ValueError(msg)
        self.root = Path(
            self.cfg.get("root", "third_party/DanceTrack/dancetrack")
        ).expanduser()
        split_map = self.cfg.get("split", {})
        split_dir = split_map.get(split, split)
        self.split_path = self.root / split_dir
        self.sampler = window_sampler or _default_sampler
        cache_cfg = self.cfg.get("cache", {})
        self.cache_enabled = bool(cache_cfg.get("enabled", False))
        cache_path = cache_cfg.get("path")
        if cache_path:
            self.cache_path = Path(cache_path)
            if not self.cache_path.is_absolute():
                self.cache_path = (self.root / self.cache_path).resolve()
        else:
            self.cache_path = self.root / ".dancetrack_meta.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._transform = self._build_transform(self.cfg.get("image", {}))
        self._augment = self._build_augment(self.cfg.get("image", {}))
        self.debug = dict(debug or {})
        self.sequences = self._load_sequences()
        self.windows = self._materialize_windows()
        if self.debug.get("minimal"):
            self.windows = self.windows[: min(4, len(self.windows))]

    def _build_transform(self, image_cfg: Mapping[str, Any]) -> v2.Compose:
        size = int(image_cfg.get("resize", 224))
        mean = image_cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
        std = image_cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225])
        return v2.Compose(
            [
                v2.Resize((size, size)),
                v2.ConvertImageDtype(torch.float32),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def _build_augment(
        self, image_cfg: Mapping[str, Any]
    ) -> v2.RandomHorizontalFlip | None:
        prob = float(image_cfg.get("horizontal_flip_prob", 0.0))
        if prob <= 0:
            return None
        return v2.RandomHorizontalFlip(p=prob)

    def _load_sequences(self) -> list[_SequenceMeta]:
        cached = self._read_cache()
        if cached is not None:
            return cached
        sequences: list[_SequenceMeta] = []
        for seq_dir in sorted(self.split_path.glob("*/seqinfo.ini")):
            seq_root = seq_dir.parent
            seq_name = seq_root.name
            info = _parse_seqinfo(seq_dir)
            frames = sorted((seq_root / info["imDir"]).glob(f"*{info['imExt']}"))
            annotations = _parse_annotations(seq_root / "gt" / "gt.txt")
            rel_paths = [str(frame.relative_to(self.root)) for frame in frames]
            sequences.append(
                _SequenceMeta(
                    name=seq_name,
                    frame_paths=rel_paths,
                    annotations=annotations,
                    frame_rate=float(info["frameRate"]),
                    image_size=(int(info["imWidth"]), int(info["imHeight"])),
                )
            )
        self._write_cache(sequences)
        return sequences

    def _read_cache(self) -> list[_SequenceMeta] | None:
        if not self.cache_enabled or not self.cache_path.exists():
            return None
        with self.cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        split_entries = payload.get(self.split)
        if split_entries is None:
            return None
        return [_SequenceMeta.from_json(entry) for entry in split_entries]

    def _write_cache(self, sequences: Sequence[_SequenceMeta]) -> None:
        if not self.cache_enabled:
            return
        payload = {}
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        payload[self.split] = [seq.to_json() for seq in sequences]
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _materialize_windows(self) -> list[tuple[int, int, int]]:
        windows: list[tuple[int, int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            length = len(seq.frame_paths)
            for start, end in self.sampler(
                length, self.window_size, self.window_stride
            ):
                windows.append((seq_idx, start, end))
        return windows

    def __len__(self) -> int:
        """Return the number of sampled windows in the dataset."""
        return len(self.windows)

    def __getitem__(self, index: int) -> TrackingSample:
        """Load and return a tracked window identified by `index`."""
        seq_idx, start, end = self.windows[index]
        seq = self.sequences[seq_idx]
        frame_ids = list(range(start, end))
        frames: list[Tensor] = []
        targets: list[TargetFrame] = []
        for frame_id in frame_ids:
            frame, target = self._load_frame(seq, frame_id)
            frames.append(frame)
            targets.append(target)
        stacked = torch.stack(frames, dim=0)
        return TrackingSample(
            frames=stacked,
            targets=targets,
            sequence_id=seq.name,
            frame_indices=frame_ids,
        )

    def _load_frame(
        self, seq: _SequenceMeta, frame_id: int
    ) -> tuple[Tensor, TargetFrame]:
        frame_path = self.root / seq.frame_paths[frame_id]
        try:
            image = read_image(str(frame_path))
        except FileNotFoundError:
            H = seq.image_size[1]
            W = seq.image_size[0]
            image = torch.zeros(3, H, W, dtype=torch.uint8)
        anns = seq.annotations.get(frame_id + 1, [])
        boxes_xyxy = self._annotations_to_boxes(anns)
        image_tensor = tv_tensors.Image(image)
        bbox_tensor = tv_tensors.BoundingBoxes(
            boxes_xyxy,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image.shape[-2], image.shape[-1]),
        )
        if self._augment is not None:
            image_tensor, bbox_tensor = self._augment(image_tensor, bbox_tensor)
        image_tensor, bbox_tensor = self._transform(image_tensor, bbox_tensor)
        target = _boxes_to_target(bbox_tensor, anns)
        return torch.as_tensor(image_tensor), target

    def _annotations_to_boxes(self, anns: Sequence[_Annotation]) -> Tensor:
        if not anns:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = torch.tensor(
            [
                [ann.bbox_xywh[0], ann.bbox_xywh[1], ann.bbox_xywh[2], ann.bbox_xywh[3]]
                for ann in anns
            ],
            dtype=torch.float32,
        )
        xyxy = boxes.clone()
        xyxy[:, 2:] = xyxy[:, :2] + xyxy[:, 2:]
        return xyxy


def _boxes_to_target(
    boxes: tv_tensors.BoundingBoxes, anns: Sequence[_Annotation]
) -> TargetFrame:
    tensor = torch.as_tensor(boxes)
    if tensor.numel() == 0:
        return TargetFrame.empty()
    cx = (tensor[:, 0] + tensor[:, 2]) * 0.5
    cy = (tensor[:, 1] + tensor[:, 3]) * 0.5
    size = torch.stack(
        [tensor[:, 2] - tensor[:, 0], tensor[:, 3] - tensor[:, 1]], dim=-1
    )
    center = torch.stack([cx, cy], dim=-1)
    track_ids = torch.tensor([ann.track_id for ann in anns], dtype=torch.long)
    confidence = torch.tensor([ann.confidence for ann in anns], dtype=torch.float32)
    return TargetFrame(
        center=center, size=size, track_ids=track_ids, confidence=confidence
    )


def _parse_seqinfo(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", maxsplit=1)
            info[key] = value
    return info


def _parse_annotations(path: Path) -> dict[int, list[_Annotation]]:
    annotations: dict[int, list[_Annotation]] = {}
    if not path.exists():
        return annotations
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            frame_id, track_id, x, y, w, h, conf, *_rest = line.split(",")
            ann = _Annotation(
                frame_index=int(frame_id),
                track_id=int(track_id),
                bbox_xywh=(float(x), float(y), float(w), float(h)),
                confidence=float(conf),
            )
            annotations.setdefault(ann.frame_index, []).append(ann)
    return annotations


def _to_dict(cfg: Mapping[str, Any] | DictConfig | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
