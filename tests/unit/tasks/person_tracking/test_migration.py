"""Explicit migration preserves observations and never overwrites an archive."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.contracts import ScoreOrigin
from src.tasks.person_tracking.migration import migrate_v2_features
from tests.support.tasks.person_tracking.archive import write_v2_features


def test_v2_migration_preserves_arrays_provenance_and_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    before = source.read_bytes()
    destination = tmp_path / "new" / "v3.npz"

    assert migrate_v2_features(source, destination) == destination

    assert source.read_bytes() == before
    with np.load(source, allow_pickle=False) as old, np.load(destination, allow_pickle=False) as new:
        assert set(new.files) == {*old.files, "score_origin"}
        for name in set(old.files) - {"metadata"}:
            np.testing.assert_array_equal(new[name], old[name])
            assert new[name].dtype == old[name].dtype
        metadata = json.loads(new["metadata"].item())
        assert metadata == {
            **json.loads(old["metadata"].item()), "schema": "person_detection_features_v3",
        }
        np.testing.assert_array_equal(new["score_origin"], np.full(2, ScoreOrigin.LEGACY, np.uint8))
    frames, provenance = load_features(destination)
    assert [frame.frame for frame in frames] == [0, 1, 2]
    assert [frame.rows.tolist() for frame in frames] == [[7], [], [11]]
    assert frames[0].score_at(0) == 0.0
    assert frames[2].score_at(0) == 0.75
    assert provenance == metadata["provenance"]
    with pytest.raises(ValueError, match="requires explicit migration"):
        load_features(source)


@pytest.mark.parametrize("destination_kind", ["source", "existing", "symlink", "hardlink"])
def test_v2_migration_never_overwrites_source_or_destination(tmp_path: Path, destination_kind: str) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    destination = tmp_path / "v3.npz"
    if destination_kind == "source":
        destination = source
    elif destination_kind == "symlink":
        destination.symlink_to(source)
    elif destination_kind == "hardlink":
        destination.hardlink_to(source)
    else:
        destination.write_bytes(b"keep existing destination")
    before_source, before_destination = source.read_bytes(), destination.read_bytes()
    with pytest.raises(FileExistsError):
        migrate_v2_features(source, destination)
    assert source.read_bytes() == before_source
    assert destination.read_bytes() == before_destination


@pytest.mark.parametrize("fault", ["v1", "v3", "missing_field", "extra_field", "offsets", "duplicate_row", "scores", "poses"])
def test_v2_migration_rejects_invalid_source_before_writing(tmp_path: Path, fault: str) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if fault in {"v1", "v3"}:
        metadata = json.loads(arrays["metadata"].item())
        metadata["schema"] = f"person_detection_features_{fault}"
        arrays["metadata"] = np.asarray(json.dumps(metadata))
    elif fault == "missing_field":
        del arrays["scores"]
    elif fault == "extra_field":
        arrays["score_origin"] = np.full(2, ScoreOrigin.MODEL, np.uint8)
    elif fault == "offsets":
        arrays["offsets"][-1] = 1
    elif fault == "duplicate_row":
        arrays["rows"][1] = arrays["rows"][0]
    elif fault == "scores":
        arrays["scores"][1] = 2.0
    else:
        arrays["poses"][0, 0, 0] = np.nan
    np.savez_compressed(source, **arrays)
    before = source.read_bytes()
    destination = tmp_path / "new" / "v3.npz"
    with pytest.raises(ValueError):
        migrate_v2_features(source, destination)
    assert not destination.parent.exists()
    assert source.read_bytes() == before


def test_v2_migration_rejects_dangling_destination_symlink(tmp_path: Path) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    destination = tmp_path / "v3.npz"
    target = tmp_path / "new" / "target.npz"
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        migrate_v2_features(source, destination)
    assert destination.is_symlink()
    assert not target.parent.exists()
