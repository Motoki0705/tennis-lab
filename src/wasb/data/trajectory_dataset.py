from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from src.wasb.tennis_format import TennisLabelRow, load_label_csv, make_empty_row


@dataclass(frozen=True)
class TrajectoryWindow:
    match: str
    clip: str
    labels: list[TennisLabelRow]


def _list_frames(clip_dir: Path, image_ext: str) -> list[str]:
    ext = image_ext.lower().lstrip(".")
    return sorted(
        [p.name for p in clip_dir.iterdir() if p.suffix.lower().lstrip(".") == ext]
    )


def _resolve_matches(root_dir: Path, matches: Iterable[str]) -> list[str]:
    match_list = list(matches)
    if match_list:
        return match_list
    return sorted([p.name for p in root_dir.iterdir() if p.is_dir()])


def _load_clip_labels(
    clip_dir: Path, csv_filename: str, image_ext: str
) -> list[TennisLabelRow]:
    csv_path = clip_dir / csv_filename
    labels_dict = {row.file_name: row for row in load_label_csv(csv_path)}
    frame_names = _list_frames(clip_dir, image_ext)
    if not frame_names:
        return []
    rows: list[TennisLabelRow] = []
    for name in frame_names:
        row = labels_dict.get(name)
        if row is None:
            row = make_empty_row(name)
        rows.append(row)
    return rows


def build_trajectory_windows(
    root_dir: Path,
    matches: Iterable[str],
    sequence_length: int,
    step: int,
    image_ext: str,
    csv_filename: str,
    min_visible_per_window: int,
) -> list[TrajectoryWindow]:
    windows: list[TrajectoryWindow] = []
    match_list = _resolve_matches(root_dir, matches)
    for match in match_list:
        match_dir = root_dir / match
        if not match_dir.exists():
            continue
        for clip_dir in sorted(match_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            labels = _load_clip_labels(clip_dir, csv_filename, image_ext)
            if len(labels) < sequence_length:
                continue
            max_start = len(labels) - sequence_length
            for start in range(0, max_start + 1, step):
                window_labels = labels[start : start + sequence_length]
                visible = sum(1 for r in window_labels if r.visibility > 0)
                if visible < min_visible_per_window:
                    continue
                windows.append(
                    TrajectoryWindow(
                        match=match,
                        clip=clip_dir.name,
                        labels=window_labels,
                    )
                )
    if not windows:
        raise RuntimeError(f"No trajectory windows found under {root_dir}")
    return windows


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        matches: Sequence[str],
        sequence_length: int,
        step: int = 1,
        image_ext: str = ".jpg",
        csv_filename: str = "Label.csv",
        min_visible_per_window: int = 1,
        block_mask_min_len: int = 4,
        block_mask_max_len: int = 7,
        sparse_mask_prob: float = 0.05,
        noise_prob: float = 0.3,
        noise_std_px: float = 3.0,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.step = step
        self.image_ext = image_ext
        self.csv_filename = csv_filename
        self.min_visible_per_window = min_visible_per_window
        self.block_mask_min_len = max(0, block_mask_min_len)
        self.block_mask_max_len = max(self.block_mask_min_len, block_mask_max_len)
        self.sparse_mask_prob = max(0.0, min(1.0, sparse_mask_prob))
        self.noise_prob = max(0.0, min(1.0, noise_prob))
        self.noise_std_px = float(noise_std_px)

        self.windows = build_trajectory_windows(
            root_dir=self.root_dir,
            matches=matches,
            sequence_length=self.sequence_length,
            step=self.step,
            image_ext=self.image_ext,
            csv_filename=self.csv_filename,
            min_visible_per_window=self.min_visible_per_window,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _sample_block_mask(self, valid: torch.Tensor) -> torch.Tensor:
        length_min = self.block_mask_min_len
        length_max = self.block_mask_max_len
        L = valid.shape[0]
        mask = torch.zeros(L, dtype=torch.bool)
        if length_max <= 0 or not valid.any():
            return mask
        length = torch.randint(length_min, length_max + 1, (1,)).item()
        length = max(1, min(length, L))
        max_start = L - length
        if max_start < 0:
            return mask
        for _ in range(5):
            start = 0 if max_start == 0 else torch.randint(0, max_start + 1, (1,)).item()
            idx = torch.arange(start, start + length)
            if valid[idx].any():
                mask[idx] = True
                break
        return mask

    @staticmethod
    def _sample_sparse_mask(candidates: torch.Tensor, prob: float) -> torch.Tensor:
        if prob <= 0.0:
            return torch.zeros_like(candidates, dtype=torch.bool)
        rand = torch.rand_like(candidates, dtype=torch.float32)
        return (rand < prob) & candidates

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        labels = window.labels
        L = len(labels)

        xy = torch.tensor([[r.x, r.y] for r in labels], dtype=torch.float32)
        visibility = torch.tensor([r.visibility for r in labels], dtype=torch.int64)

        valid = visibility > 0

        block_mask = self._sample_block_mask(valid)
        remaining_for_sparse = valid & ~block_mask
        sparse_mask = self._sample_sparse_mask(
            remaining_for_sparse, self.sparse_mask_prob
        )
        remaining_for_noise = valid & ~block_mask & ~sparse_mask
        noise_mask = self._sample_sparse_mask(remaining_for_noise, self.noise_prob)

        xy_input = xy.clone()
        xy_input[block_mask | sparse_mask] = 0.0
        if noise_mask.any() and self.noise_std_px > 0.0:
            noise = torch.randn((L, 2), dtype=torch.float32) * self.noise_std_px
            noise[~noise_mask] = 0.0
            xy_input = xy_input + noise

        loss_mask_block = (block_mask & valid).to(torch.float32)
        loss_mask_sparse = (sparse_mask & valid).to(torch.float32)
        loss_mask_noise = (noise_mask & valid).to(torch.float32)

        scale = torch.tensor([1920.0, 1080.0], dtype=torch.float32)
        xy_input_norm = xy_input / scale
        target_xy_norm = xy / scale

        return {
            "xy_input_norm": xy_input_norm,
            "target_xy_norm": target_xy_norm,
            "loss_mask_block": loss_mask_block,
            "loss_mask_sparse": loss_mask_sparse,
            "loss_mask_noise": loss_mask_noise,
            "orig_visibility": visibility,
        }
