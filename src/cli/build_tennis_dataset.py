"""CLI for building tennis pose training datasets from simulator scenes.

本スクリプトは `gen_tennis_pose_scenes.py` を Python API レベルで呼び出し、
train/val/test のシーン JSON と簡易インデックス、およびメタ情報をまとめて生成する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tennis.sim.generator import (
    GenConfig,
    TennisPoseSceneGenerator,
    write_scene_json,
)
from src.tennis.sim.schema import validate_scene_dict


@dataclass(slots=True)
class _SplitConfig:
    r"""Configuration for a single split (train/val/test).

    Attributes:
        name (str): Split name (e.g. ``"train"``, ``"val"``, ``"test"``).
        num_scenes (int): Number of scenes to generate for the split.
        seed (int): Random seed used for this split.

    """

    name: str
    num_scenes: int
    seed: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv (Sequence[str] | None): Optional list of arguments. If ``None``,
            defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Build tennis pose training datasets from simulator scenes"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/tennis_autogen",
        help="Root directory for auto-generated datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "Optional dataset name. If omitted, a name is derived from the "
            "simulator and window parameters."
        ),
    )
    parser.add_argument(
        "--num_scenes_train",
        type=int,
        default=100,
        help="Number of training scenes to generate",
    )
    parser.add_argument(
        "--num_scenes_val",
        type=int,
        default=20,
        help="Number of validation scenes to generate",
    )
    parser.add_argument(
        "--num_scenes_test",
        type=int,
        default=20,
        help="Number of test scenes to generate",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Simulator frames per second",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Simulator duration per scene in seconds",
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=4,
        help="Number of cameras per scene",
    )
    parser.add_argument(
        "--asset_root",
        type=str,
        default="data/raw/3dtennisds",
        help="Path to the 3DTennisDS asset directory",
    )
    parser.add_argument(
        "--min_players",
        type=int,
        default=1,
        help="Minimum number of players per scene",
    )
    parser.add_argument(
        "--max_players",
        type=int,
        default=20,
        help="Maximum number of players per scene",
    )
    parser.add_argument(
        "--window_T",
        type=int,
        default=10,
        help="Temporal window length (frames) for index entries",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=5,
        help="Stride (frames) between temporal windows when building the index",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed; split seeds are derived from this value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing dataset directory",
    )
    return parser.parse_args(argv)


def _auto_dataset_name(args: argparse.Namespace) -> str:
    """Derive a dataset name from simulator and window parameters.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Auto-generated dataset name.

    """
    duration_str = str(args.duration).replace(".", "p")
    return (
        f"sim_fps{int(args.fps)}_dur{duration_str}_C{int(args.num_cameras)}_"
        f"P{int(args.min_players)}-{int(args.max_players)}_T{int(args.window_T)}"
    )


def _ensure_writable_dir(path: Path, overwrite: bool) -> None:
    """Ensure the dataset directory is writable or raise a SystemExit.

    Args:
        path (Path): Target dataset directory.
        overwrite (bool): Whether overwriting existing content is allowed.

    Raises:
        SystemExit: If the directory exists and ``overwrite`` is False.

    Returns:
        None: This function does not return a value.

    """
    if path.exists() and any(path.iterdir()) and not overwrite:
        msg = (
            f"Dataset directory already exists and is not empty: {path}. "
            "Pass --overwrite to rebuild."
        )
        raise SystemExit(msg)
    path.mkdir(parents=True, exist_ok=True)


def _git_commit_hash() -> str | None:
    """Return the current git commit hash if available.

    Returns:
        str | None: Commit hash string, or ``None`` if unavailable.

    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # pragma: no cover - best-effort metadata
        return None
    return out.decode("utf-8").strip() or None


def _generate_split_scenes(
    out_dir: Path,
    split_cfg: _SplitConfig,
    fps: int,
    duration: float,
    num_cameras: int,
    asset_root: str,
    min_players: int,
    max_players: int,
) -> None:
    """Generate JSON scenes for a given split.

    Args:
        out_dir (Path): Root dataset directory.
        split_cfg (_SplitConfig): Split configuration (name, scene count, seed).
        fps (int): Simulator frames per second.
        duration (float): Scene duration in seconds.
        num_cameras (int): Number of cameras per scene.
        asset_root (str): Path to 3DTennisDS asset directory.
        min_players (int): Minimum number of players per scene.
        max_players (int): Maximum number of players per scene.

    Returns:
        None: This function does not return a value.

    """
    split_dir = out_dir / "scenes" / split_cfg.name
    split_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenConfig(
        fps=fps,
        duration_sec=duration,
        num_cameras=num_cameras,
        asset_root=asset_root,
        min_players=min_players,
        max_players=max_players,
        seed=split_cfg.seed,
    )
    gen = TennisPoseSceneGenerator(cfg)
    for i in range(split_cfg.num_scenes):
        scene_id = f"{split_cfg.name}_{i}"
        scene = gen.generate_scene(scene_id=scene_id)
        validate_scene_dict(scene)
        path = split_dir / f"scene_{i:06d}.json"
        write_scene_json(path, scene)


def _build_index_for_split(
    dataset_dir: Path,
    split_name: str,
    window_T: int,
    window_stride: int,
) -> None:
    r"""Build a simple JSONL index for a given split.

    各レコードは 1 つのシーン内の時間ウィンドウを表し、``scene_path``、
    ``t_start``、``t_end`` などのメタデータのみを含む。実データの
    ローディングとテンソル化は Dataset 側の責務とする。

    Args:
        dataset_dir (Path): Dataset root directory.
        split_name (str): Split name (e.g. ``"train"``).
        window_T (int): Temporal window length in frames.
        window_stride (int): Stride between windows in frames.

    Returns:
        None: This function does not return a value.

    """
    scenes_dir = dataset_dir / "scenes" / split_name
    index_dir = dataset_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{split_name}_index.jsonl"

    records: list[Mapping[str, Any]] = []
    for scene_path in sorted(scenes_dir.glob("scene_*.json")):
        with scene_path.open("r", encoding="utf-8") as f:
            scene = json.load(f)
        frames = scene.get("frames", [])
        if not isinstance(frames, list) or not frames:
            continue
        T_total = len(frames)
        num_cameras = int(scene.get("num_cameras", 0))
        scene_id = str(scene.get("scene_id", scene_path.stem))

        t = 0
        while t < T_total:
            t_end = min(t + window_T, T_total)
            window_frames = frames[t:t_end]
            max_players = 0
            for fr in window_frames:
                players = fr.get("player_joints_3d", [])
                if isinstance(players, list):
                    max_players = max(max_players, len(players))
            record = {
                "scene_path": str(scene_path.relative_to(dataset_dir)),
                "scene_id": scene_id,
                "t_start": t,
                "t_end": t_end,
                "num_frames": t_end - t,
                "num_cameras": num_cameras,
                "max_players_in_window": max_players,
            }
            records.append(record)
            if t_end == T_total:
                break
            t += window_stride

    with index_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def _write_meta(
    dataset_dir: Path,
    args: argparse.Namespace,
    splits: Sequence[_SplitConfig],
) -> None:
    """Write meta.json summarizing dataset generation parameters.

    Args:
        dataset_dir (Path): Dataset root directory.
        args (argparse.Namespace): Parsed CLI arguments.
        splits (Sequence[_SplitConfig]): Per-split configurations.

    Returns:
        None: This function does not return a value.

    """
    meta: dict[str, Any] = {
        "fps": int(args.fps),
        "duration_sec": float(args.duration),
        "num_cameras": int(args.num_cameras),
        "asset_root": str(args.asset_root),
        "min_players": int(args.min_players),
        "max_players": int(args.max_players),
        "window_T": int(args.window_T),
        "window_stride": int(args.window_stride),
        "seed": int(args.seed),
        "splits": {s.name: asdict(s) for s in splits},
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
    }
    meta_path = dataset_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a tennis pose training dataset from simulator scenes.

    Args:
        argv (Sequence[str] | None): Optional arguments. If ``None``, defaults
            to ``sys.argv[1:]``.

    Returns:
        int: 0 on success, non-zero on error.

    Raises:
        SystemExit: If invalid arguments are provided.

    """
    args = _parse_args(argv)
    if args.max_players < args.min_players:
        msg = "--max_players must be greater than or equal to --min_players"
        raise SystemExit(msg)
    if args.window_T <= 0:
        raise SystemExit("window_T must be positive")
    if args.window_stride <= 0:
        raise SystemExit("window_stride must be positive")

    dataset_root = Path(args.dataset_root)
    dataset_name = args.dataset_name or _auto_dataset_name(args)
    dataset_dir = dataset_root / dataset_name
    _ensure_writable_dir(dataset_dir, overwrite=bool(args.overwrite))

    # Prepare split configs with deterministic but distinct seeds.
    base_seed = int(args.seed)
    splits = [
        _SplitConfig("train", int(args.num_scenes_train), base_seed),
        _SplitConfig("val", int(args.num_scenes_val), base_seed + 1),
        _SplitConfig("test", int(args.num_scenes_test), base_seed + 2),
    ]

    # Generate scenes per split.
    for split in splits:
        _generate_split_scenes(
            dataset_dir,
            split,
            fps=int(args.fps),
            duration=float(args.duration),
            num_cameras=int(args.num_cameras),
            asset_root=str(args.asset_root),
            min_players=int(args.min_players),
            max_players=int(args.max_players),
        )

    # Build simple JSONL indices.
    for split in splits:
        _build_index_for_split(
            dataset_dir,
            split.name,
            window_T=int(args.window_T),
            window_stride=int(args.window_stride),
        )

    # Write meta information.
    _write_meta(dataset_dir, args, splits)

    sys.stdout.write(
        f"[tennis-dataset] Built dataset '{dataset_name}' under {dataset_root}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
