"""Regression tests for the path-driven synthetic-data package layout."""

from __future__ import annotations

from src.utils.paths import PROJECT_ROOT


def test_release_gate_directories_are_absent() -> None:
    package = PROJECT_ROOT / "src/synthetic_data_generation"
    forbidden = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_dir() and path.name in {"reporting", "validation"}
    }

    assert forbidden == set()


def test_executable_modules_live_under_scripts() -> None:
    package = PROJECT_ROOT / "src/synthetic_data_generation"
    executable_modules = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
    }

    assert executable_modules
    assert all(path.startswith("scripts/") for path in executable_modules)


def test_generic_visualization_has_no_cycle_or_domain_paths() -> None:
    generic_visualization = PROJECT_ROOT / "src/synthetic_data_generation/visualization"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in generic_visualization.rglob("*.py")
    ).lower()

    assert "cycle-" not in source
    assert "b00" not in source
    assert "blcs" not in source
    assert "plcs" not in source
