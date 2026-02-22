"""Dataset layout discovery for WASB clips and video inputs."""

from __future__ import annotations

from pathlib import Path

from src.ball_detection.data.type import ClipLayout, FrameRecord, VideoLayout


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


def _sanitize_game_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return cleaned or "game"


def discover_video_layouts(
    root_dir: str | Path,
    *,
    extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv"),
) -> list[VideoLayout]:
    """Discover input videos under root and map each file to one game name."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    exts = {e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in extensions}
    video_files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    layouts: list[VideoLayout] = []
    used_names: dict[str, int] = {}
    for video_path in video_files:
        base = _sanitize_game_name(video_path.stem)
        next_index = used_names.get(base, 0)
        used_names[base] = next_index + 1
        game_name = base if next_index == 0 else f"{base}_{next_index + 1:02d}"
        layouts.append(
            VideoLayout(
                game_name=game_name,
                video_name=video_path.name,
                video_path=video_path,
            )
        )

    return layouts
