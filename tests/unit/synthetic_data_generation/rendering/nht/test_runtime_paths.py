"""Tests for package-path discovery inside the NHT runtime."""

from __future__ import annotations

from src.synthetic_data_generation.rendering.nht.runtime_paths import (
    package_root_from_file,
)


def test_package_root_from_file_finds_editable_checkout(tmp_path) -> None:
    repository = tmp_path / "gsplat"
    (repository / ".git").mkdir(parents=True)
    module_file = repository / "gsplat" / "__init__.py"
    module_file.parent.mkdir()
    module_file.write_text("")

    assert package_root_from_file(module_file) == repository
