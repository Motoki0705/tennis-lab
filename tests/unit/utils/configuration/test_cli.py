"""Explicit selection and source-verified validation through nested CLI routes."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.utils.configuration.cli import dispatch_cli
from src.utils.configuration.discovery import discover_runtime_boundaries


def test_dispatch_preserves_arguments_and_restores_argv_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ["train", "--stage", "coordinates", "--output", "/a path/run", "--seed", "42"]
    monkeypatch.setattr(sys, "argv", original)

    def main() -> None:
        assert sys.argv == ["train", "--output", "/a path/run", "--seed", "42"]
        raise ValueError("leaf validation failed")

    with patch("importlib.import_module", return_value=SimpleNamespace(main=main)) as load, pytest.raises(ValueError, match="leaf validation"):
        dispatch_cli("stage", {"coordinates": "src.example"}, description="test")
    load.assert_called_once_with("src.example")
    assert sys.argv is original


def test_invalid_selector_never_imports_or_runs_a_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--stage", "unknown"])
    with patch("importlib.import_module", side_effect=AssertionError("must not run")), pytest.raises(SystemExit) as caught:
        dispatch_cli("stage", {"coordinates": "src.example"}, description="test")
    assert caught.value.code == 2


def fixture_routes(tmp_path: Path, route: str) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "entry.py").write_text(
        "from src.utils.configuration.cli import dispatch_cli\n"
        "def main():\n"
        f"    dispatch_cli('stage', {route}, description='test')\n"
    )
    (source / "nested.py").write_text(
        "from src.utils.configuration.cli import dispatch_cli\n"
        "def main():\n"
        "    dispatch_cli('initialize', {'scratch': 'src.leaf'}, description='test')\n"
    )
    (source / "leaf.py").write_text(
        "from src.utils.configuration import NonHydraPathBoundary\n"
        "PATH_BOUNDARY = NonHydraPathBoundary(name='real-leaf', fields=('declared-path',))\n"
        "def main():\n"
        "    PATH_BOUNDARY.validate(arguments, resolver=resolver)\n"
    )
    return source


def test_nested_route_records_the_actual_leaf_validator(tmp_path: Path) -> None:
    source = fixture_routes(tmp_path, "{'coordinates': 'src.nested'}")
    entry = next(b for b in discover_runtime_boundaries(source) if b.module == "src.entry")
    assert entry.routes == (("stage", "coordinates", "src.nested"),)
    assert entry.delegated_boundaries == (("src.leaf", "main", "real-leaf", "src.utils.configuration.paths.NonHydraPathBoundary.validate"),)
    assert entry.validator_callable == "src.utils.configuration.cli.dispatch_cli"


@pytest.mark.parametrize("route", [
    "{'coordinates': 'src.missing'}",
    "{'coordinates': 'src.entry'}",
    "{'duplicate': 'src.leaf', 'duplicate': 'src.nested'}",
    "dynamic_routes",
    "{}",
])
def test_unresolvable_duplicate_dynamic_and_cyclic_routes_fail_closed(tmp_path: Path, route: str) -> None:
    source = fixture_routes(tmp_path, route)
    entry = next(b for b in discover_runtime_boundaries(source) if b.module == "src.entry")
    assert entry.validator_callable is None
    assert not entry.delegated_boundaries


def test_unvalidated_leaf_invalidates_the_whole_router(tmp_path: Path) -> None:
    source = fixture_routes(tmp_path, "{'good': 'src.nested', 'bad': 'src.unvalidated'}")
    (source / "unvalidated.py").write_text("def main():\n    publish_without_validation()\n")
    entry = next(b for b in discover_runtime_boundaries(source) if b.module == "src.entry")
    assert entry.validator_callable is None
