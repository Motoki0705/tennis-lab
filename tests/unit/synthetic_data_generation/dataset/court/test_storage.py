from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court import storage
from src.synthetic_data_generation.dataset.runtime import directory_size_bytes
from src.utils.io import load_json, save_json_atomic


def source_owner(tmp_path: Path) -> Path:
    root = tmp_path / "source" / "B00" / "datasets" / "court"
    root.mkdir(parents=True)
    sample = {}
    for field in ("rgb", "alpha", "depth"):
        path = root / f"{field}.npy"
        np.save(path, np.ones((3, 4, 3 if field == "rgb" else 1), dtype=np.float32))
        sample[field] = path.name
    save_json_atomic({"samples": [sample]}, root / "dataset.json")
    save_json_atomic(
        {"metrics": {"published_bytes": 1}}, root / "diagnostics/performance.json"
    )
    return root


def test_failed_validation_never_publishes_and_resume_rechecks_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_owner(tmp_path)
    original = (source / "dataset.json").read_bytes()
    destination = tmp_path / "result"

    def fail(path: Path) -> None:
        raise RuntimeError("semantic gate")

    monkeypatch.setattr(storage, "validate_court_dataset", fail)
    with pytest.raises(RuntimeError, match="semantic gate"):
        storage.compact(source, destination, workers=1)
    assert not destination.exists()
    assert (source / "dataset.json").read_bytes() == original
    staging = destination.with_name(".result.compressing")
    # Simulate one interrupted archive; only the explicit attempt is repaired.
    (staging / "rgb.f32.npz").write_bytes(b"interrupted")
    monkeypatch.setattr(storage, "validate_court_dataset", lambda path: None)
    report = storage.compact(source, destination, workers=1, resume=True)
    assert report["reused_verified_arrays"] == 2
    assert not staging.exists()
    assert load_json(destination / "diagnostics/performance.json")["metrics"][
        "published_bytes"
    ] == directory_size_bytes(destination)
    assert (source / "dataset.json").read_bytes() == original


def test_resume_rejects_changed_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_owner(tmp_path)
    destination = tmp_path / "result"

    def fail(path: Path) -> None:
        raise RuntimeError("semantic gate")

    monkeypatch.setattr(storage, "validate_court_dataset", fail)
    with pytest.raises(RuntimeError):
        storage.compact(source, destination)
    (source / "dataset.json").write_text('{"samples": []}')
    with pytest.raises(ValueError, match="identity"):
        storage.compact(source, destination, resume=True)
    assert not destination.exists()


def test_in_place_and_unowned_staging_are_rejected(tmp_path: Path) -> None:
    source = source_owner(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        storage.compact(source, source)
    destination = tmp_path / "result"
    destination.with_name(".result.compressing").mkdir()
    with pytest.raises(FileExistsError):
        storage.compact(source, destination)
