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
        "--show-contracts",
        action="store_true",
        help="render required/optional/default/precedence authorities",
    )
    parser.add_argument(
        "--show-discovered-boundaries",
        action="store_true",
        help="render source-discovered callable, executable, and validator bindings",
    )
    arguments = parser.parse_args()
    return run_configuration_audit(
        cast(Path, arguments.source_root),
        AuditOptions(
            show_contracts=cast(bool, arguments.show_contracts),
            show_discovered_boundaries=cast(
                bool, arguments.show_discovered_boundaries
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
