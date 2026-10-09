from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.utils.checksum import dual_sha256
from src.utils.shared_file_verification import SharedFileVerification


def _verify_worker(cache: SharedFileVerification, path: Path, gate: Any, counter: Any) -> None:
    def count_hash(p: Path) -> str:
        with counter.get_lock():
            counter.value += 1
        digest: str = dual_sha256(p)
        return digest

    assert gate.wait(30)
    with patch("src.utils.shared_file_verification.dual_sha256", count_hash):
        for _ in range(3):
            cache.verify(path)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_workers_share_successful_hash_once(tmp_path: Path, method: str) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"verified content" * 1000)
    cache = SharedFileVerification({path: hashlib.sha256(path.read_bytes()).hexdigest()})
    ctx: Any = mp.get_context(method)
    gate, counter = ctx.Event(), ctx.Value("i", 0)
    # Bound production methods remain importable under pytest's importlib mode
    # when a spawned interpreter cannot import the tests namespace.
    workers = [ctx.Process(target=cache.verify, args=(path,)) if method == "spawn"
               else ctx.Process(target=_verify_worker, args=(cache, path, gate, counter)) for _ in range(3)]
    try:
        for worker in workers:
            worker.start()
        gate.set()
        for worker in workers:
            worker.join(40)
            assert worker.exitcode == 0
        if method == "fork":
            assert counter.value == 1
        with patch("src.utils.shared_file_verification.dual_sha256", side_effect=AssertionError("rehash")):
            cache.verify(path)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)


def test_rejects_mutation_even_if_size_and_mtime_are_restored(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"old")
    cache = SharedFileVerification({path: hashlib.sha256(b"old").hexdigest()})
    cache.verify(path)
    stat = path.stat()
    time.sleep(.02)  # Cross the timestamp granularity of the WSL filesystem.
    path.write_bytes(b"new")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        cache.verify(path)


def test_failed_hash_is_never_published(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"wrong")
    cache = SharedFileVerification({path: hashlib.sha256(b"right").hexdigest()})
    with pytest.raises(ValueError, match="checksum"):
        cache.verify(path)
    path.write_bytes(b"right")
    cache.verify(path)


def test_change_between_hash_and_publication_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"original")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = SharedFileVerification({path: digest})

    def mutate(_: Path) -> str:
        path.write_bytes(b"changed!longer")
        return digest

    with patch("src.utils.shared_file_verification.dual_sha256", mutate), pytest.raises(ValueError, match="changed"):
        cache.verify(path)
    with pytest.raises(ValueError, match="checksum"):
        cache.verify(path)


def test_upfront_pool_publishes_to_parent_and_future_workers(tmp_path: Path) -> None:
    files = {}
    for i in range(3):
        path = tmp_path / str(i)
        path.write_bytes(str(i).encode())
        files[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = SharedFileVerification(files)
    cache.verify_all(2)
    with patch("src.utils.shared_file_verification.dual_sha256", side_effect=AssertionError("rehash")):
        for path in files:
            cache.verify(path)


def test_pool_propagates_failure_without_trusting_the_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"wrong")
    cache = SharedFileVerification({path: hashlib.sha256(b"correct").hexdigest()})
    with pytest.raises(ValueError, match="checksum"):
        cache.verify_all(2)
    with pytest.raises(ValueError, match="checksum"):
        cache.verify(path)
