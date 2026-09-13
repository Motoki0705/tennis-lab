"""Transactional, resumable lossless compaction of a published Court owner."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import BadZipFile

from src.synthetic_data_generation.dataset.court.assembler import validate_court_dataset
from src.synthetic_data_generation.dataset.runtime import directory_size_bytes
from src.synthetic_data_generation.pipeline.locking import scene_write_lock
from src.utils.data.float32_store import SUFFIX, read_float32, write_float32
from src.utils.io import save_json_atomic


def compact(
    source: Path, destination: Path, *, resume: bool = False, workers: int = 4
) -> dict[str, object]:
    """Publish a disjoint copy only after byte equality and semantic validation.

    An interrupted attempt retains its explicitly owned sibling staging directory.
    ``resume=True`` verifies its source manifest fingerprint and every reused array.
    Source files are never linked, rewritten, or removed.
    """
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and destination trees must be disjoint.")
    if destination.exists():
        raise FileExistsError(destination)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in 1..8.")
    staging = destination.with_name(f".{destination.name}.compressing")
    marker = destination.with_name(f".{destination.name}.compression.json")
    started = time.perf_counter()
    with scene_write_lock(source.parents[1]):
        manifest_bytes = (source / "dataset.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        identity = {
            "schema": "court_lossless_compaction_v1",
            "source": str(source),
            "destination": str(destination),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        arrays: set[str] = set()
        for sample in manifest["samples"]:
            for field in ("rgb", "alpha", "depth"):
                relative = sample[field]
                path = source / relative
                if (
                    path.is_symlink()
                    or not path.resolve(strict=True).is_relative_to(source)
                    or path.suffix != ".npy"
                ):
                    raise ValueError(f"Expected contained ordinary NPY: {path}")
                arrays.add(relative)
        if resume:
            if (
                marker.is_symlink()
                or staging.is_symlink()
                or not staging.is_dir()
                or json.loads(marker.read_text()) != identity
            ):
                raise ValueError(
                    "Resume requires an exact owned source/staging identity."
                )
        else:
            if staging.exists() or marker.exists():
                raise FileExistsError(
                    "An unfinished compaction exists; use explicit resume."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("x") as stream:
                json.dump(identity, stream)
            staging.mkdir()
        files = tuple(path for path in source.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Source must not contain symlinks.")
        before = sum(path.stat().st_size for path in files)

        def copy_file(path: Path) -> int:
            relative = path.relative_to(source)
            if relative.as_posix() == "dataset.json":
                return 0
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.as_posix() not in arrays:
                if target.is_symlink():
                    raise ValueError("Staging contains a symbolic link.")
                shutil.copy2(path, target)
                return 0
            value = read_float32(path)
            target = target.with_suffix(SUFFIX)
            if target.is_symlink():
                raise ValueError("Staging contains a symbolic link.")
            if target.exists():
                try:
                    restored = read_float32(target)
                    if (
                        restored.tobytes() == value.tobytes()
                        and restored.shape == value.shape
                    ):
                        return 1
                except (ValueError, OSError, EOFError, BadZipFile):
                    pass  # Explicit resume repairs only this attempt's incomplete output.
                target.unlink()
            write_float32(target, value)
            if read_float32(target).tobytes() != value.tobytes():
                raise ValueError(f"Bitwise verification failed: {path}")
            return 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            reused = sum(executor.map(copy_file, files))
        for sample in manifest["samples"]:
            for field in ("rgb", "alpha", "depth"):
                sample[field] = str(Path(sample[field]).with_suffix(SUFFIX))
        save_json_atomic(manifest, staging / "dataset.json")
        if (source / "dataset.json").read_bytes() != manifest_bytes:
            raise RuntimeError("Source manifest changed during compression.")
        # Generation timings remain historical; publication bytes describe this copy.
        performance_path = staging / "diagnostics/performance.json"
        performance = json.loads(performance_path.read_text())
        for _ in range(8):
            actual = directory_size_bytes(staging)
            performance["metrics"]["published_bytes"] = actual
            performance["metrics"]["dense_reference_bytes"] = actual
            save_json_atomic(performance, performance_path)
            if directory_size_bytes(staging) == actual:
                break
        else:
            raise RuntimeError("Publication byte accounting did not converge.")
        validate_court_dataset(staging)
        after = directory_size_bytes(staging)
        staging.rename(destination)
        marker.unlink()
    return {
        **identity,
        "samples": len(manifest["samples"]),
        "original_bytes": before,
        "compressed_bytes": after,
        "ratio": after / before,
        "reused_verified_arrays": reused,
        "wall_seconds": time.perf_counter() - started,
        "bitwise_verified": True,
    }
