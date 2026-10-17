"""Explicit storage ownership and read-only inventory/GC simulation."""

from .catalog import StorageCatalog
from .contracts import InventoryScope, ManagedArtifact, Protection, Retention
from .inventory import inventory
from .planner import plan_gc, save_dry_run

__all__ = [
    "InventoryScope",
    "ManagedArtifact",
    "Protection",
    "Retention",
    "StorageCatalog",
    "inventory",
    "plan_gc",
    "save_dry_run",
]
