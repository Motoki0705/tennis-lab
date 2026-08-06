"""Audit repository-owned configuration and path-resolution contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT_TEXT = str(_REPOSITORY_ROOT)
if _REPOSITORY_ROOT_TEXT in sys.path:
    sys.path.remove(_REPOSITORY_ROOT_TEXT)
sys.path.insert(0, _REPOSITORY_ROOT_TEXT)

from src.utils.configuration.audit import (  # noqa: E402
    AuditOptions,
    run_configuration_audit,
)


def main() -> int:
    """Run the canonical configuration audit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--show-ledger",
        action="store_true",
        help="render every exact migration record as tab-separated data",
    )
    parser.add_argument(
        "--show-contracts",
        action="store_true",
        help="render required/optional/default/precedence authorities",
    )
    parser.add_argument(
        "--show-discovered-boundaries",
        action="store_true",
        help="render source-discovered callable, executable, and validator bindings",
    )
    parser.add_argument(
        "--regenerate-ledger",
        action="store_true",
        help="emit deterministic candidate ledger JSON without changing embedded data",
    )
    parser.add_argument(
        "--write-generated-data",
        action="store_true",
        help=(
            "freeze re-anchored exemptions and ledger metadata in migration_data.py "
            "and exemption_data.py; fails on every newly unclassified construct"
        ),
    )
    parser.add_argument(
        "--source-revision",
        help="immutable revision label required by --write-generated-data",
    )
    arguments = parser.parse_args()
    source_revision = cast(str | None, arguments.source_revision)
    write_generated_data = cast(bool, arguments.write_generated_data)
    if write_generated_data and source_revision is None:
        parser.error("--write-generated-data requires --source-revision")
    return run_configuration_audit(
        cast(Path, arguments.source_root),
        AuditOptions(
            show_ledger=cast(bool, arguments.show_ledger),
            show_contracts=cast(bool, arguments.show_contracts),
            show_discovered_boundaries=cast(
                bool, arguments.show_discovered_boundaries
            ),
            regenerate_ledger=cast(bool, arguments.regenerate_ledger),
            write_generated_data=write_generated_data,
            source_revision=source_revision,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
