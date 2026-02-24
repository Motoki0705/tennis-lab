"""Unlabeled clip dataset used for pseudo-label generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledBallDataset(Dataset[dict[str, object]]):
    """Frame-level dataset without annotations."""

    def __init__(self, root_dir: str | Path, games: list[str], image_size_hw: tuple[int, int] = (288, 512)) -> None:
        self.root_dir = Path(root_dir)
        self.games = games
        self.transform = transforms.Compose([transforms.Resize(image_size_hw), transforms.ToTensor()])
        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(f"No unlabeled samples found under {self.root_dir}")

    def _index_samples(self) -> list[Path]:
        samples: list[Path] = []
        for game in self.games:
            game_dir = self.root_dir / game
            if not game_dir.exists():
                continue
            for clip_dir in sorted([p for p in game_dir.iterdir() if p.is_dir() and p.name.startswith("Clip")]):
                frame_files = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                samples.extend(frame_files)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        frame_path = self.samples[index]
        with Image.open(frame_path) as img:
            frame = self.transform(img.convert("RGB"))
        return {
            "frame": frame,
            "frame_path": str(frame_path),
            "file_name": frame_path.name,
        }
