"""PyTorch dataset and DataModule for the WASB tennis corpus."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.wasb.tennis_format import TennisLabelRow, load_label_csv, make_empty_row

LOGGER = logging.getLogger(__name__)

VisibilityMode = Literal["none", "any_visible", "all_visible"]
Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class SequenceSample:
    """A fixed-length sequence of frames and target annotations."""

    frame_paths: list[Path]
    targets: list[TennisLabelRow]
    match: str
    clip: str


def _list_frames(clip_dir: Path, image_ext: str) -> list[str]:
    ext = image_ext.lower().lstrip(".")
    return sorted(
        [p.name for p in clip_dir.iterdir() if p.suffix.lower().lstrip(".") == ext]
    )


def _load_labels(clip_dir: Path, csv_filename: str) -> dict[str, TennisLabelRow]:
    csv_path = clip_dir / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")
    return {row.file_name: row for row in load_label_csv(csv_path)}


def _should_keep(targets: Sequence[TennisLabelRow], mode: VisibilityMode) -> bool:
    if mode == "none":
        return True
    visibility = [row.visibility > 0 for row in targets]
    if mode == "any_visible":
        return any(visibility)
    if mode == "all_visible":
        return all(visibility)
    raise ValueError(f"Unknown visibility mode: {mode}")


def _resolve_matches(root_dir: Path, matches: Iterable[str]) -> list[str]:
    match_list = list(matches)
    if match_list:
        return match_list
    return sorted([p.name for p in root_dir.iterdir() if p.is_dir()])


def build_sequence_index(
    root_dir: Path,
    matches: Iterable[str],
    frames_in: int,
    frames_out: int,
    step: int,
    visibility_mode: VisibilityMode,
    image_ext: str,
    csv_filename: str,
) -> list[SequenceSample]:
    """Create a list of sequence samples from tennis clips."""
    samples: list[SequenceSample] = []
    match_list = _resolve_matches(root_dir, matches)
    for match in match_list:
        match_dir = root_dir / match
        if not match_dir.exists():
            LOGGER.warning("Match directory missing, skipping: %s", match_dir)
            continue

        for clip_dir in sorted(match_dir.iterdir()):
            if not clip_dir.is_dir():
                continue

            frame_names = _list_frames(clip_dir, image_ext)
            if len(frame_names) < frames_in:
                continue

            labels = _load_labels(clip_dir, csv_filename)
            max_start = len(frame_names) - frames_in

            for start_idx in range(0, max_start + 1, step):
                window = frame_names[start_idx : start_idx + frames_in]
                target_names = window[-frames_out:]
                targets = [
                    labels.get(name, make_empty_row(name)) for name in target_names
                ]
                if not _should_keep(targets, visibility_mode):
                    continue

                frame_paths = [clip_dir / name for name in window]
                samples.append(
                    SequenceSample(
                        frame_paths=frame_paths,
                        targets=targets,
                        match=match,
                        clip=clip_dir.name,
                    )
                )
    LOGGER.info(
        "Indexed %d sequences from %d matches under %s",
        len(samples),
        len(match_list),
        root_dir,
    )
    return samples


class TennisSequenceDataset(Dataset):
    """Sliding-window dataset for WASB tennis training."""

    def __init__(
        self,
        root_dir: str | Path,
        matches: Sequence[str],
        frames_in: int,
        frames_out: int = 1,
        step: int = 1,
        visibility_mode: VisibilityMode = "none",
        image_ext: str = ".jpg",
        csv_filename: str = "Label.csv",
        transform: Transform | None = None,
        resize_hw: tuple[int, int] | None = None,
        heatmap_hw: tuple[int, int] | None = None,
        heatmap_sigma: float | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Root directory containing tennis games (e.g., ``data/tennis``).
            matches: Game directories to include (e.g., ``["game1", "game2"]``).
            frames_in: Number of input frames per sample.
            frames_out: Number of target frames (tail of the window).
            step: Stride for the sliding window.
            visibility_mode: Filter strategy based on target visibility.
            image_ext: Frame file extension to load.
            csv_filename: Annotation file name inside each clip directory.
            transform: Optional transform applied to each PIL image.
            resize_hw: Optional ``(height, width)`` resize before ``ToTensor``.
        """
        if frames_out > frames_in:
            raise ValueError("frames_out cannot exceed frames_in")
        self.root_dir = Path(root_dir)
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.visibility_mode = visibility_mode
        self.image_ext = image_ext
        self.csv_filename = csv_filename
        self.resize_hw = resize_hw
        self.heatmap_hw = tuple(heatmap_hw) if heatmap_hw is not None else None
        self.heatmap_sigma = heatmap_sigma

        self.transform = transform or self._default_transform(resize_hw)
        self.samples = build_sequence_index(
            root_dir=self.root_dir,
            matches=matches,
            frames_in=frames_in,
            frames_out=frames_out,
            step=step,
            visibility_mode=visibility_mode,
            image_ext=image_ext,
            csv_filename=csv_filename,
        )
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root_dir}")

    @staticmethod
    def _default_transform(resize_hw: tuple[int, int] | None) -> Transform:
        ops: list[Callable[[Image.Image], Image.Image] | Transform] = []
        if resize_hw is not None:
            ops.append(transforms.Resize(resize_hw))
        ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str] | str]:
        sample = self.samples[index]
        frames: list[torch.Tensor] = []
        first_w, first_h = 0, 0

        for frame_path in sample.frame_paths:
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                if first_w == 0 and first_h == 0:
                    first_w, first_h = img.size
                img_t = self.transform(img)
                frames.append(img_t)

        frames_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        targets_px = torch.tensor(
            [[t.x, t.y] for t in sample.targets], dtype=torch.float32
        )
        if self.resize_hw is not None:
            resize_h, resize_w = self.resize_hw
            scale_x = resize_w / first_w
            scale_y = resize_h / first_h
            targets_px = targets_px * torch.tensor([scale_x, scale_y])
            final_w, final_h = resize_w, resize_h
        else:
            final_w, final_h = first_w, first_h

        targets_norm = targets_px / torch.tensor([final_w, final_h], dtype=torch.float32)
        visibility = torch.tensor(
            [t.visibility for t in sample.targets], dtype=torch.int64
        )
        scores = torch.tensor([t.score for t in sample.targets], dtype=torch.float32)
        heatmap_hw = self._get_heatmap_hw(final_h, final_w)
        target_heatmaps = self._build_heatmaps(
            targets_norm, visibility, heatmap_hw, sigma=self.heatmap_sigma
        )

        return {
            "frames": frames_tensor,
            "targets_px": targets_px,
            "targets_norm": targets_norm,
            "target_heatmaps": target_heatmaps,
            "visibility": visibility,
            "scores": scores,
            "match": sample.match,
            "clip": sample.clip,
            "frame_paths": [str(p) for p in sample.frame_paths],
        }

    def _get_heatmap_hw(self, final_h: int, final_w: int) -> tuple[int, int]:
        """Resolve heatmap height/width, defaulting to half-resolution."""
        if self.heatmap_hw is not None:
            return self.heatmap_hw

        # Default: half-resolution of the processed frames.
        h = max(int(final_h // 2), 1)
        w = max(int(final_w // 2), 1)
        self.heatmap_hw = (h, w)
        return self.heatmap_hw

    @staticmethod
    def _build_heatmaps(
        targets_norm: torch.Tensor,
        visibility: torch.Tensor,
        heatmap_hw: tuple[int, int],
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Create per-frame heatmaps at the desired resolution."""
        h, w = heatmap_hw
        heatmaps = torch.zeros((targets_norm.shape[0], h, w), dtype=torch.float32)

        if sigma is not None and sigma > 0:
            # Precompute coordinate grids for Gaussian kernels.
            ys = torch.arange(h, dtype=torch.float32).view(h, 1)
            xs = torch.arange(w, dtype=torch.float32).view(1, w)

        for idx, (coord, vis) in enumerate(zip(targets_norm, visibility)):
            if vis <= 0:
                continue
            x = torch.clamp(coord[0], 0.0, 1.0) * max(w - 1, 1)
            y = torch.clamp(coord[1], 0.0, 1.0) * max(h - 1, 1)
            xi = int(torch.round(x).item())
            yi = int(torch.round(y).item())
            xi = max(0, min(xi, w - 1))
            yi = max(0, min(yi, h - 1))
            if sigma is None or sigma <= 0:
                heatmaps[idx, yi, xi] = 1.0
            else:
                gauss = torch.exp(
                    -((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma * sigma)
                )
                heatmaps[idx] = torch.maximum(heatmaps[idx], gauss)

        return heatmaps
