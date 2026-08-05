"""Execute the BLCS strict-configuration negative matrix."""

from __future__ import annotations

from src.tasks.blcs.configuration import run_negative_matrix


def main() -> int:
    """Run every BLCS negative case without starting a runtime workload."""
    run_negative_matrix()
    print("BLCS negative validation matrix: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
