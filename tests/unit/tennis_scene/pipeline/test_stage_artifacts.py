"""Stage replay requires the same inputs/settings and untampered numeric data."""

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.tennis_scene.pipeline.artifacts import PipelineArtifactStore


def test_stage_cache_roundtrip_and_identity_rejection(tmp_path: Path) -> None:
    store = PipelineArtifactStore(tmp_path)
    identity = {"settings": MappingProxyType({"threshold": .5}), "input": "sha"}
    values = np.array([[True, False]])
    store.save("association", identity, {"mask": values, "ids": np.array([[0, -1]], np.int64)})
    loaded = store.load("association", identity)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["mask"], values)
    with pytest.raises(ValueError, match="Stale"):
        store.load("association", {"input": "different"})
    (tmp_path / "association.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load("association", identity)


def test_load_only_cache_never_executes_missing_work(tmp_path: Path) -> None:
    store = PipelineArtifactStore(tmp_path, source="load")
    with pytest.raises(FileNotFoundError):
        store.load("court", {})
    with pytest.raises(RuntimeError, match="load-only"):
        store.save("court", {}, {})
