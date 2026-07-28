"""Tests for provenance discovery inside the independently pinned NHT runtime."""

from __future__ import annotations

from src.synthetic_data_generation.rendering.nht.provenance import (
    repository_from_package_file,
)


def test_repository_from_package_file_finds_editable_checkout(tmp_path) -> None:
    repository = tmp_path / "gsplat"
    (repository / ".git").mkdir(parents=True)
    module_file = repository / "gsplat" / "__init__.py"
    module_file.parent.mkdir()
    module_file.write_text("")

    assert repository_from_package_file(module_file) == repository
