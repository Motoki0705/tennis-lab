"""The feature migration CLI has an explicit, inventoried path boundary."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.contracts import ScoreOrigin
from src.tasks.person_tracking.scripts import migrate_features
from src.utils.configuration import PathContractError
from src.utils.configuration.discovery import discover_runtime_boundaries
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES, BoundaryKind
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.person_tracking.archive import write_v2_features


def test_migration_cli_has_one_discoverable_validated_boundary() -> None:
    module = "src.tasks.person_tracking.scripts.migrate_features"
    boundary, = [item for item in EXPECTED_RUNTIME_BOUNDARIES if item.module == module]
    assert boundary.kind == BoundaryKind.ARGPARSE
    assert boundary.executable_module
    assert boundary.validator_key == migrate_features.PATH_BOUNDARY.name
    discovered = discover_runtime_boundaries(PROJECT_ROOT / "src")
    actual, = [item for item in discovered if item.module == module]
    assert actual.callable_name == "main"
    assert not any(item.module == "src.tasks.person_tracking.migration" for item in discovered)


@pytest.mark.parametrize("fault", ["relative_source", "relative_destination", "missing_source", "directory_source", "directory_destination", "dangling_destination"])
def test_migration_cli_validates_paths_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    destination = tmp_path / "new" / "v3.npz"
    if fault == "relative_source":
        source = Path("v2.npz")
    elif fault == "relative_destination":
        destination = Path("v3.npz")
    elif fault == "missing_source":
        source = tmp_path / "missing.npz"
    elif fault == "directory_source":
        source = tmp_path
    elif fault == "directory_destination":
        destination = tmp_path
    else:
        destination = tmp_path / "dangling.npz"
        destination.symlink_to(tmp_path / "new" / "v3.npz")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Migration must not run before path validation")

    monkeypatch.setattr(migrate_features, "migrate_v2_features", forbidden)
    monkeypatch.setattr(sys, "argv", ["migrate_features", str(source), str(destination)])
    expected_error = FileExistsError if fault in {"directory_destination", "dangling_destination"} else PathContractError
    with pytest.raises(expected_error):
        migrate_features.main()
    assert not (tmp_path / "new").exists()


def test_migration_cli_writes_a_separate_v3_and_rejects_reuse(tmp_path: Path) -> None:
    source = tmp_path / "v2.npz"
    write_v2_features(source)
    before = source.read_bytes()
    destination = tmp_path / "new" / "v3.npz"
    command = [sys.executable, "-m", "src.tasks.person_tracking.scripts.migrate_features", str(source), str(destination)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(destination)
    frames, _ = load_features(destination)
    assert [frame.score_origins().tolist() for frame in frames] == [[ScoreOrigin.LEGACY], [], [ScoreOrigin.LEGACY]]
    assert source.read_bytes() == before
    migrated = destination.read_bytes()
    repeated = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
    assert destination.read_bytes() == migrated
    assert source.read_bytes() == before
