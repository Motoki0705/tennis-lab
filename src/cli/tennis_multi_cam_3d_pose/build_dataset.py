"""CLI for building tennis pose training datasets from simulator scenes.

本スクリプトは `gen_tennis_pose_scenes.py` を Python API レベルで呼び出し、
train/val/test のシーン JSON と簡易インデックス、およびメタ情報をまとめて生成する。
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

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


_GENERATOR: TennisPoseSceneGenerator | None = None


def _init_generator(gen_cfg_dict: dict[str, Any]) -> None:
    """Initialize worker processes by constructing a shared generator.

    The GenConfig is passed as a plain mapping to avoid OmegaConf dependence
    in worker processes.
    """
    global _GENERATOR
    cfg = GenConfig(**gen_cfg_dict)
    _GENERATOR = TennisPoseSceneGenerator(cfg)


def _worker_generate_scene(args: tuple[int, str, int]) -> tuple[int, dict[str, Any]]:
    """Worker function to generate and validate a single scene.

    Args:
        args (tuple[int, str, int]): Tuple of (scene_index, split_name, base_seed).

    Returns:
        tuple[int, dict[str, Any]]: Tuple of (scene_index, scene_dict).

    Raises:
        RuntimeError: If the worker generator is not initialized.

    """
    idx, split_name, base_seed = args
    if _GENERATOR is None:
        msg = "Worker generator is not initialized"
        raise RuntimeError(msg)
    # Use a per-scene seed derived from the split seed and scene index so that
    # results are deterministic but differ across scenes, independent of
    # scheduling.
    _GENERATOR.reseed(int(base_seed) + int(idx))
    scene_id = f"{split_name}_{idx}"
    scene = _GENERATOR.generate_scene(scene_id=scene_id)
    validate_scene_dict(scene)
    return idx, dict(scene)


def _parse_args(argv: Sequence[str] | None = None) -> SimpleNamespace:
    """Parse CLI arguments.

    Args:
        argv (Sequence[str] | None): Optional list of arguments. If ``None``,
            defaults to ``sys.argv[1:]``.

    Returns:
        SimpleNamespace: Parsed arguments.

    Raises:
        SystemExit: If invalid or missing arguments are provided.

    """
    if argv is None:
        argv = sys.argv[1:]

    cfg_path: str | None = None
    overwrite = False

    tokens = list(argv)
    it = iter(tokens)
    for token in it:
        if token in ("-h", "--help"):
            msg = "Usage: build_tennis_dataset.py --config PATH [--overwrite]\n"
            raise SystemExit(msg)
        if token == "--config":
            try:
                cfg_path = next(it)
            except StopIteration as exc:
                raise SystemExit("Expected path after --config") from exc
        elif token.startswith("--config="):
            cfg_path = token.split("=", 1)[1]
        elif token == "--overwrite":
            overwrite = True
        else:
            if cfg_path is None and not token.startswith("-"):
                cfg_path = token
            else:
                msg = f"Unknown argument: {token}"
                raise SystemExit(msg)

    if cfg_path is None:
        msg = (
            "Missing --config PATH\n"
            "Usage: build_tennis_dataset.py --config PATH [--overwrite]\n"
        )
        raise SystemExit(msg)

    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    if not isinstance(cfg_dict, Mapping):
        msg = f"Config root must be a mapping: {cfg_path}"
        raise SystemExit(msg)

    defaults: dict[str, Any] = {
        "dataset_root": "data/tennis_autogen",
        "dataset_name": None,
        "num_scenes_train": 100,
        "num_scenes_val": 20,
        "num_scenes_test": 20,
        "fps": 60,
        "duration": 3.0,
        "num_cameras": 4,
        "asset_root": "data/raw/3dtennisds",
        "min_players": 1,
        "max_players": 20,
        "window_T": 10,
        "window_stride": 5,
        "seed": 1234,
        "num_workers": 0,
        "overwrite": False,
    }

    merged = {**defaults, **cfg_dict}
    if "overwrite" in cfg_dict:
        merged["overwrite"] = bool(cfg_dict["overwrite"])
    if overwrite:
        merged["overwrite"] = True

    merged["config"] = cfg_path

    return SimpleNamespace(**merged)


def _auto_dataset_name(args: SimpleNamespace) -> str:
    """Derive a dataset name from simulator and window parameters.

    Args:
        args (SimpleNamespace): Parsed CLI arguments.

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
    num_workers: int,
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
        num_workers (int): Number of worker processes for parallel generation.

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

    num_workers = int(num_workers)
    if num_workers <= 0:
        # Single-process generation (original behavior).
        gen = TennisPoseSceneGenerator(cfg)
        for i in tqdm(
            range(split_cfg.num_scenes),
            desc=f"Generating scenes ({split_cfg.name})",
        ):
            scene_id = f"{split_cfg.name}_{i}"
            scene = gen.generate_scene(scene_id=scene_id)
            validate_scene_dict(scene)
            path = split_dir / f"scene_{i:06d}.json"
            write_scene_json(path, scene)
        return

    # Multi-process generation using a shared generator per worker.
    gen_cfg_dict = asdict(cfg)
    tasks = [
        (i, split_cfg.name, int(split_cfg.seed)) for i in range(split_cfg.num_scenes)
    ]
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_generator,
        initargs=(gen_cfg_dict,),
    ) as executor:
        for idx, scene in tqdm(
            executor.map(_worker_generate_scene, tasks),
            total=split_cfg.num_scenes,
            desc=f"Generating scenes ({split_cfg.name})",
        ):
            path = split_dir / f"scene_{idx:06d}.json"
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
    scene_paths = sorted(scenes_dir.glob("scene_*.json"))
    for scene_path in tqdm(
        scene_paths,
        desc=f"Building index ({split_name})",
    ):
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
    args: SimpleNamespace,
    splits: Sequence[_SplitConfig],
) -> None:
    """Write meta.json summarizing dataset generation parameters.

    Args:
        dataset_dir (Path): Dataset root directory.
        args (SimpleNamespace): Parsed CLI arguments.
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
            num_workers=int(getattr(args, "num_workers", 0)),
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
