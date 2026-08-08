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
from src.utils.configuration import DEFAULT_AUDIT_INVENTORY, MigrationStatus
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


def test_deleted_inventory_module_uses_its_explicit_canonical_authority() -> None:
    historical = tuple(
        record
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.former_module == "src.configuration_validation"
    )
    inventory_source = (
        PROJECT_ROOT / "src/utils/configuration/inventory.py"
    ).read_text(encoding="utf-8")

    assert historical
    assert all(record.status is MigrationStatus.MIGRATED for record in historical)
    assert all(record.domain == "base" for record in historical)
    assert all(
        record.canonical_symbol.startswith("src.tasks.base.configuration.")
        for record in historical
    )
    assert '== "configuration_validation"' not in inventory_source


def test_catalog_does_not_publish_legacy_synthetic_boundaries() -> None:
    synthetic = tuple(
        contract
        for contract in catalog.BOUNDARY_CONTRACTS
        if contract.boundary_id.startswith("src.synthetic_data_generation.")
    )

    assert all(
        contract.boundary_id.startswith(
            "src.synthetic_data_generation.scripts.run_scene_pipeline:"
        )
        for contract in synthetic
    )
    discovery_source = (
        PROJECT_ROOT / "src/utils/configuration/discovery.py"
    ).read_text(encoding="utf-8")
    assert "scripts.alignment.geometry_bridge" not in discovery_source
