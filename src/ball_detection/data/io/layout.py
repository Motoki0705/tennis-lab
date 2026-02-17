"""Dataset layout discovery for WASB-style tennis clips."""

from __future__ import annotations

from pathlib import Path

from src.ball_detection.data.type import ClipLayout, FrameRecord


def _iter_clip_dirs(game_dir: Path) -> list[Path]:
    return sorted([p for p in game_dir.iterdir() if p.is_dir() and p.name.startswith("Clip")])


def discover_clip_layouts(root_dir: str | Path) -> list[ClipLayout]:
    """Discover game/clip/frame layout under `data/tennis`-style root.

    Expected layout:
    - root/gameX/ClipY/*.jpg
    - root/gameX/ClipY/Label.csv
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    layouts: list[ClipLayout] = []
    for game_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for clip_dir in _iter_clip_dirs(game_dir):
            frame_files = sorted(
                [p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            frames = tuple(
                FrameRecord(frame_index=i, file_name=f.name, frame_path=f)
                for i, f in enumerate(frame_files)
            )
            label_csv = clip_dir / "Label.csv"
            layouts.append(
                ClipLayout(
                    game_name=game_dir.name,
                    clip_name=clip_dir.name,
                    clip_dir=clip_dir,
                    label_csv=label_csv,
                    frames=frames,
                )
            )
    return layouts
