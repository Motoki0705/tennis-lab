"""The public migration CLI validates paths before reading or publishing a store."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tasks.player_detection.data.store import PlayerFrameStore
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.scripts import migrate_store
from src.utils.configuration import PathContractError
from src.utils.configuration.discovery import discover_runtime_boundaries
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES, BoundaryKind
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.player_detection.chat_root import make_synthetic_root
from tests.support.tasks.player_detection.store import legacy_store


def test_migration_cli_has_one_discoverable_validated_boundary() -> None:
    module = "src.tasks.player_detection.scripts.migrate_store"
    boundary, = [item for item in EXPECTED_RUNTIME_BOUNDARIES if item.module == module]
    assert boundary.kind == BoundaryKind.ARGPARSE
    assert boundary.executable_module
    assert boundary.validator_key == migrate_store.PATH_BOUNDARY.name
    discovered = discover_runtime_boundaries(PROJECT_ROOT / "src")
    actual, = [item for item in discovered if item.module == module]
    assert actual.callable_name == "main"
    assert not any(item.module == "src.tasks.player_detection.data.migration" for item in discovered)


@pytest.mark.parametrize("fault", ["relative_source", "relative_destination", "missing_source", "file_source"])
def test_migration_cli_rejects_invalid_paths_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "new-parent" / "v2"
    if fault == "relative_source":
        source = Path("source")
    elif fault == "relative_destination":
        destination = Path("new-parent/v2")
    elif fault == "missing_source":
        source = tmp_path / "missing"
    else:
        source = tmp_path / "file"
        source.write_text("not a store")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Migration must not run before path validation")

    monkeypatch.setattr(migrate_store, "migrate_v1_store", forbidden)
    monkeypatch.setattr(sys, "argv", ["migrate_store", str(source), str(destination)])
    with pytest.raises(PathContractError):
        migrate_store.main()
    assert not (tmp_path / "new-parent").exists()


def test_migration_cli(tmp_path: Path) -> None:
    synthetic = make_synthetic_root(tmp_path)
    current = build_dataset(synthetic.config)
    source = legacy_store(current, tmp_path / "old-data" / "legacy-v1")
    destination = tmp_path / "separate-output" / "migrated-v2"
    result = subprocess.run(
        [sys.executable, "-m", "src.tasks.player_detection.scripts.migrate_store", str(source), str(destination)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == str(destination)
    assert PlayerFrameStore(destination).frames["frame_index"].tolist() == [1, 2, 4]
