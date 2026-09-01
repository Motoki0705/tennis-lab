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


def test_canonical_visualization_has_no_scene_specific_paths() -> None:
    visualization = PROJECT_ROOT / "src/synthetic_data_generation/visualization"
    modules = {
        path.name for path in visualization.glob("*.py") if path.is_file()
    }
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in visualization.rglob("*.py")
    ).lower()

    assert modules == {
        "__init__.py",
        "configuration.py",
        "contracts.py",
        "court_aabb.py",
        "overlays.py",
        "renderer.py",
        "sources.py",
    }
    assert "cycle-" not in source
    assert "b00" not in source
    assert all(domain in source for domain in ("court", "blcs", "plcs"))
