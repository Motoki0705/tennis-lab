"""Copy small indexes and hardlink immutable JPEG shards for a training plan."""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from src.utils.checksum import dual_sha256

from .store import BallFrameStore, shard_name


def _version(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def snapshot_ball_store(
    source: Path, destination: Path, *, progress: Callable[[int, int], None] | None = None,
) -> tuple[BallFrameStore, dict[str, str]]:
    """No 130 GiB duplication or silent copy fallback on another filesystem.

    Hardlinks isolate paths from later store replacement, not in-place byte
    mutation. Each reader still checks the frozen shard digest on first use.
    """
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Training snapshot must not be inside the source store")
    source_hashes = {name: dual_sha256(source / name) for name in ("metadata.json", "index.npz")}
    store = BallFrameStore(source)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "shards").mkdir()
    for name, expected in source_hashes.items():
        shutil.copyfile(source / name, destination / name)
        if dual_sha256(destination / name) != expected:
            raise ValueError("Source store changed while copying snapshot indexes")
    hashes = {}
    for index, clip in enumerate(store.clips):
        original = source / "shards" / shard_name(clip.index)
        target = destination / "shards" / shard_name(clip.index)
        before = original.stat()
        try:
            os.link(original, target)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise ValueError("Snapshot requires output on the same filesystem as JPEG shards; copying RGB is not automatic") from exc
            raise
        hashes[clip.clip_id] = dual_sha256(target)
        after = original.stat()
        # Creating additional hardlinks changes ctime, not the content version.
        if _version(before) != _version(after) or _version(target.stat()) != _version(after):
            raise ValueError(f"JPEG shard changed while freezing {clip.clip_id}")
        if progress is not None:
            progress(index + 1, len(store.clips))
    if any(dual_sha256(source / name) != expected for name, expected in source_hashes.items()):
        raise ValueError("Source store changed during snapshot preparation")
    return BallFrameStore(destination), hashes
