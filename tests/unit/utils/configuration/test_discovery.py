"""Import-graph regressions for configuration boundary discovery."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import src.utils.configuration.audit as audit
import src.utils.configuration.catalog as catalog
import src.utils.configuration.discovery as discovery
from src.utils.configuration.inventory import (
    EXTERNAL_RUNTIME_MODULES,
    AuditInventory,
    AuditRule,
)
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    "modules",
    [
        ("src.utils.configuration.audit", "src.utils.configuration.catalog"),
        ("src.utils.configuration.catalog", "src.utils.configuration.audit"),
    ],
)
def test_configuration_modules_import_cold_in_either_order(
    modules: tuple[str, str],
) -> None:
    imports = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_boundary_discovery_has_one_canonical_owner() -> None:
    owners: list[Path] = []
    configuration_root = PROJECT_ROOT / "src/utils/configuration"
    for path in configuration_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "discover_runtime_boundaries"
            for node in tree.body
        ):
            owners.append(path)

    assert owners == [configuration_root / "discovery.py"]
    assert discovery.discover_runtime_boundaries.__module__ == discovery.__name__
    assert not hasattr(audit, "discover_runtime_boundaries")
    assert not hasattr(catalog, "discover_runtime_boundaries")


def test_catalog_and_audit_depend_on_lower_discovery_without_a_back_edge() -> None:
    catalog_source = (PROJECT_ROOT / "src/utils/configuration/catalog.py").read_text(
        encoding="utf-8"
    )
    audit_source = (PROJECT_ROOT / "src/utils/configuration/audit.py").read_text(
        encoding="utf-8"
    )
    discovery_source = (
        PROJECT_ROOT / "src/utils/configuration/discovery.py"
    ).read_text(encoding="utf-8")

    assert "import src.utils.configuration.discovery" in catalog_source
    assert "import src.utils.configuration.discovery" in audit_source
    assert "src.utils.configuration.audit" not in catalog_source
    assert "src.utils.configuration.catalog" not in discovery_source
    assert "src.utils.configuration.audit" not in discovery_source


def test_catalog_does_not_publish_legacy_synthetic_boundaries() -> None:
    synthetic = {
        contract.boundary_id
        for contract in catalog.BOUNDARY_CONTRACTS
        if contract.boundary_id.startswith("src.synthetic_data_generation.")
    }

    assert synthetic == {
        "src.synthetic_data_generation.scripts.compact_court_storage:main",
        "src.synthetic_data_generation.scripts.court_scene_assets:main",
        "src.synthetic_data_generation.scripts.generate_court_scale:main",
        "src.synthetic_data_generation.scripts.edit_alignment:main",
        "src.synthetic_data_generation.scripts.generate_publication_visualizations:main",
        "src.synthetic_data_generation.scripts.review_court_dataset:main",
        "src.synthetic_data_generation.scripts.run_appearance_variant:main",
        "src.synthetic_data_generation.scripts.run_scene_pipeline:main",
        "src.synthetic_data_generation.scripts.visualize_dataset:main",
        "src.synthetic_data_generation.scripts.visualize_scene_reveal:main",
    }
    discovery_source = (
        PROJECT_ROOT / "src/utils/configuration/discovery.py"
    ).read_text(encoding="utf-8")
    assert "scripts.alignment.geometry_bridge" not in discovery_source


def test_catalog_keeps_task_local_generation_boundaries_distinct() -> None:
    boundary_ids = {contract.boundary_id for contract in catalog.BOUNDARY_CONTRACTS}

    assert {
        "src.tasks.blcs.scripts.generate_dataset:main",
        "src.tasks.blcs.scripts.preview_augmentation:main",
        "src.tasks.blcs.scripts.visualize:main",
        "src.tasks.plcs.scripts.generate_dataset:main",
        "src.tasks.plcs.scripts.preview_augmentation:main",
        "src.tasks.plcs.scripts.visualize:main",
    } <= boundary_ids


def test_external_runtime_registration_preserves_neighbor_boundary_checks(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    shots = source_root / "tennis_scene/visualization/blender/shots"
    shots.mkdir(parents=True)
    source = (
        "import bpy\n"
        "import argparse\n"
        "def main():\n"
        "    return argparse.ArgumentParser().parse_args()\n"
    )
    for name in ("reveal", "validate_reveal", "unregistered_shot"):
        (shots / f"{name}.py").write_text(source, encoding="utf-8")
    exporter = source_root / "tennis_scene/scripts/export_blender.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text(source.replace("import bpy\n", ""), encoding="utf-8")

    report = audit.inspect_source(source_root, inventory=AuditInventory())

    expected = {
        "src.tennis_scene.visualization.blender.shots.unregistered_shot:main",
        "src.tennis_scene.scripts.export_blender:main",
    }
    assert {
        f"{boundary.module}:{boundary.callable_name}"
        for boundary in report.discovered_boundaries
    } == expected
    assert {issue.boundary_id for issue in report.boundary_issues} == expected
    assert not report.passed


def test_external_runtime_sources_still_reject_configuration_fallbacks(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    shot = source_root / "tennis_scene/visualization/blender/shots/reveal.py"
    shot.parent.mkdir(parents=True)
    shot.write_text(
        "from collections.abc import Mapping\n"
        "def read_config(config: Mapping[str, object]):\n"
        "    return config.get('root', 'fallback')\n"
        "def main():\n"
        "    return read_config({})\n",
        encoding="utf-8",
    )

    report = audit.inspect_source(source_root, inventory=AuditInventory())

    assert not report.discovered_boundaries
    assert AuditRule.GET_WITH_FALLBACK in {finding.rule for finding in report.findings}
    assert not report.passed


def test_external_runtime_registrations_have_sources_reasons_and_no_repo_imports() -> (
    None
):
    for module, reason in EXTERNAL_RUNTIME_MODULES.items():
        assert reason.strip(), module
        path = PROJECT_ROOT / Path(*module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, module
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                name == "src" or name.startswith("src.") for name in modules
            ), module
