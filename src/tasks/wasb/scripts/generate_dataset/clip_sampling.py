#!/usr/bin/env python
"""Clip sampling workflow utilities for WASB dataset curation.

This script supports the "clip sampling" workflow that was previously part of
`src.tasks.wasb.scripts.generate_game`:

- `generate_samples`: Create preview mp4s per clip under `output_dir/samples/<game>/`
- `apply_clip_selection`: Apply manual selection by keeping only preview files that
  remain in `samples/<game>/`, then reindex `Clip*` directories.

Configuration is managed via Hydra using `src/tasks/wasb/configs/clip_sampling.yaml`.

Usage:
    uv run python -m src.tasks.wasb.scripts.generate_dataset.clip_sampling mode=generate_samples \
      output_dir=data/tennis generate_samples=[game11]

    uv run python -m src.tasks.wasb.scripts.generate_dataset.clip_sampling mode=apply_clip_selection \
      output_dir=data/tennis apply_clip_selection=[game11]

"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.tasks.wasb.tennis_format import load_label_csv


def _resolve_path(path_str: str | None) -> Path | None:
    """Resolve a path from config (relative to current working directory)."""
    if path_str is None:
        return None
    return Path(path_str)


def run_from_config(cfg: DictConfig) -> int:
    """Dispatch execution based on the configuration mode."""
    mode = str(getattr(cfg, "mode", "generate_samples"))
    if mode == "generate_samples":
        return generate_samples(cfg)
    if mode == "apply_clip_selection":
        return apply_clip_selection(cfg)
    raise ValueError(f"Unknown mode: {mode}")


def _iter_clip_dirs(game_dir: Path) -> list[tuple[int, Path]]:
    clips: list[tuple[int, Path]] = []
    if not game_dir.exists():
        return clips
    for child in game_dir.iterdir():
        if child.is_dir() and child.name.startswith("Clip"):
            suffix = child.name[4:]
            if suffix.isdigit():
                clips.append((int(suffix), child))
    clips.sort(key=lambda x: x[0])
    return clips


def _make_clip_preview_video(clip_dir: Path, output_path: Path, fps: int) -> None:
    import cv2

    label_path = clip_dir / "Label.csv"
    if not label_path.exists():
        return
    rows = load_label_csv(label_path)
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r.file_name)

    first_frame_path = clip_dir / f"frame_{rows_sorted[0].file_name}"
    frame = cv2.imread(str(first_frame_path))
    if frame is None:
        return

    height, width = frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for row in rows_sorted:
            frame_path = clip_dir / f"frame_{row.file_name}"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            if row.visibility != 0:
                x = int(round(row.x))
                y = int(round(row.y))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), 2, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def _resolve_games(output_dir: Path, games: list[str]) -> list[str]:
    if len(games) == 1 and games[0].lower() == "all":
        resolved: list[str] = []

        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)

            videos = meta.get("videos", {})
            for status in videos.values():
                game_name = status.get("output_game")
                if isinstance(game_name, str) and game_name not in resolved:
                    resolved.append(game_name)
        else:
            if output_dir.exists():
                for path in sorted(output_dir.iterdir()):
                    if path.is_dir() and path.name.startswith("game"):
                        resolved.append(path.name)

        return resolved

    return games


def _generate_samples_for_game(
    output_dir: Path, game_name: str, fps: int = 50, show_progress: bool = True
) -> None:
    game_dir = output_dir / game_name
    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return

    samples_dir = output_dir / "samples" / game_name
    clips = _iter_clip_dirs(game_dir)

    if show_progress:
        iterator = tqdm(clips, desc=f"{game_name}", unit="clip")
    else:
        iterator = clips

    for index, clip_dir in iterator:
        output_path = samples_dir / f"Clip_{index}.mp4"
        _make_clip_preview_video(clip_dir, output_path, fps)


def generate_samples(cfg: DictConfig) -> int:
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    games_cfg = list(getattr(cfg, "generate_samples", []))
    games = _resolve_games(output_dir, games_cfg)
    if not games:
        print("No games to generate samples for.")
        return 1

    fps = int(getattr(cfg, "fps", 50))
    quiet = bool(getattr(cfg, "quiet", False))
    iterable = games if quiet else tqdm(games, desc="Generating samples", unit="game")

    for game_name in iterable:
        if not quiet:
            print(f"Generating samples for {game_name}...")
        _generate_samples_for_game(
            output_dir,
            game_name,
            fps=fps,
            show_progress=not quiet,
        )

    return 0


def _collect_kept_indices(samples_dir: Path) -> set[int]:
    kept: set[int] = set()
    if not samples_dir.exists():
        return kept

    for path in samples_dir.glob("Clip_*.mp4"):
        stem = path.stem
        if not stem.startswith("Clip_"):
            continue
        suffix = stem[5:]
        if suffix.isdigit():
            kept.add(int(suffix))

    return kept


def _delete_clip_dirs(game_dir: Path, indices: set[int]) -> None:
    for index in sorted(indices):
        clip_dir = game_dir / f"Clip{index}"
        if clip_dir.exists() and clip_dir.is_dir():
            shutil.rmtree(clip_dir)


def _reindex_clip_dirs(game_dir: Path) -> None:
    clips = _iter_clip_dirs(game_dir)
    if not clips:
        return

    temp_paths: list[Path] = []
    for index, path in clips:
        temp_path = path.with_name(f"__tmp_Clip{index}")
        path.rename(temp_path)
        temp_paths.append(temp_path)

    for new_index, temp_path in enumerate(temp_paths, start=1):
        final_path = game_dir / f"Clip{new_index}"
        temp_path.rename(final_path)


def _update_meta_num_clips(output_dir: Path, game_name: str) -> None:
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        return

    with meta_path.open("r") as f:
        meta = json.load(f)

    videos = meta.get("videos", {})
    game_dir = output_dir / game_name
    clip_count = len(_iter_clip_dirs(game_dir))

    updated = False
    for status in videos.values():
        if status.get("output_game") == game_name:
            status["num_clips"] = clip_count
            updated = True

    if updated:
        from datetime import datetime

        meta["updated_at"] = datetime.now().isoformat()
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)


def _apply_clip_selection_for_game(output_dir: Path, game_name: str) -> None:
    game_dir = output_dir / game_name
    samples_dir = output_dir / "samples" / game_name

    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        return

    if not samples_dir.exists():
        print(f"No samples found for {game_name} at {samples_dir}")
        return

    clips = _iter_clip_dirs(game_dir)
    if not clips:
        print(f"No clips found in {game_dir}")
        return

    original_indices = {index for index, _ in clips}
    kept_indices = _collect_kept_indices(samples_dir)

    to_drop = original_indices - kept_indices
    if to_drop:
        print(f"Dropping clips for {game_name}: {sorted(to_drop)}")
        _delete_clip_dirs(game_dir, to_drop)

    _reindex_clip_dirs(game_dir)
    _update_meta_num_clips(output_dir, game_name)
    shutil.rmtree(samples_dir, ignore_errors=True)


def apply_clip_selection(cfg: DictConfig) -> int:
    output_dir_str = getattr(cfg, "output_dir", "data/tennis")
    output_dir = _resolve_path(str(output_dir_str))
    if output_dir is None:
        print("Error: Failed to resolve 'output_dir' path.", file=sys.stderr)
        return 1

    games_cfg = list(getattr(cfg, "apply_clip_selection", []))
    games = _resolve_games(output_dir, games_cfg)
    if not games:
        print("No games to apply clip selection for.")
        return 1

    for game_name in games:
        print(f"Applying clip selection for {game_name}...")
        _apply_clip_selection_for_game(output_dir, game_name)

    return 0


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="clip_sampling",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    exit_code = run_from_config(cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
