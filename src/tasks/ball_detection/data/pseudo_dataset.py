"""Pseudo-labeled dataset for sequence self-training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.tasks.ball_detection.data.io.annotation_reader import read_label_csv


@dataclass(frozen=True)
class _ClipSequence:
    frame_paths: tuple[Path, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    vis: tuple[float, ...]
    score: tuple[float, ...]


class PseudoBallDataset(Dataset[dict[str, Tensor]]):
    """Dataset that reads pseudo labels generated under output root."""

    def __init__(
        self,
        pseudo_root_dir: str | Path,
        image_size_hw: tuple[int, int] = (288, 512),
        window_size: int = 16,
        window_stride: int = 8,
        min_window_size: int = 4,
    ) -> None:
        self.pseudo_root_dir = Path(pseudo_root_dir)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.min_window_size = int(min_window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if self.min_window_size <= 0:
            raise ValueError("min_window_size must be positive.")

        self.transform = transforms.Compose([transforms.Resize(image_size_hw), transforms.ToTensor()])
        self.clips, self.windows = self._index_samples()
        if not self.windows:
            raise RuntimeError(f"No pseudo-labeled samples found under {self.pseudo_root_dir}")

    def _index_samples(self) -> tuple[list[_ClipSequence], list[tuple[int, int, int]]]:
        clips: list[_ClipSequence] = []
        windows: list[tuple[int, int, int]] = []
        for game_dir in sorted([p for p in self.pseudo_root_dir.iterdir() if p.is_dir()]):
            for clip_dir in sorted([p for p in game_dir.iterdir() if p.is_dir() and p.name.startswith("Clip")]):
                csv_path = clip_dir / "Label.csv"
                if not csv_path.exists():
                    continue

                rows = read_label_csv(csv_path)
                frame_paths: list[Path] = []
                xs: list[float] = []
                ys: list[float] = []
                vis: list[float] = []
                score: list[float] = []
                for row in rows:
                    frame_path = clip_dir / row.file_name
                    if not frame_path.exists():
                        continue
                    frame_paths.append(frame_path)
                    xs.append(float(row.x))
                    ys.append(float(row.y))
                    vis.append(float(row.visibility))
                    score.append(float(row.score))

                length = len(frame_paths)
                if length < self.min_window_size:
                    continue

                clip_idx = len(clips)
                clips.append(
                    _ClipSequence(
                        frame_paths=tuple(frame_paths),
                        xs=tuple(xs),
                        ys=tuple(ys),
                        vis=tuple(vis),
                        score=tuple(score),
                    )
                )

                for start in range(0, length, self.window_stride):
                    end = min(start + self.window_size, length)
                    if end - start < self.min_window_size:
                        continue
                    windows.append((clip_idx, start, end))

        return clips, windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        clip_idx, start, end = self.windows[index]
        clip = self.clips[clip_idx]

        frames: list[Tensor] = []
        target_xy: list[Tensor] = []
        target_vis: list[Tensor] = []
        target_weight: list[Tensor] = []
        for t in range(start, end):
            frame_path = clip.frame_paths[t]
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                frame = self.transform(img)
            frames.append(frame)

            x = clip.xs[t]
            y = clip.ys[t]
            vis_bin = 1.0 if clip.vis[t] > 0 else 0.0
            score = max(float(clip.score[t]), 1e-3)
            target_xy.append(
                torch.tensor(
                    [x / max(w - 1, 1), y / max(h - 1, 1)],
                    dtype=torch.float32,
                )
            )
            target_vis.append(torch.tensor(vis_bin, dtype=torch.float32))
            target_weight.append(torch.tensor(score, dtype=torch.float32))

        seq_len = end - start
        return {
            "frames": torch.stack(frames, dim=0),
            "target_xy": torch.stack(target_xy, dim=0),
            "target_vis": torch.stack(target_vis, dim=0),
            "target_weight": torch.stack(target_weight, dim=0),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }
