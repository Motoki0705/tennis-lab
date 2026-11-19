"""Run only the first serial repeat using the benchmark in the current checkout."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd() / "knowledge" / "runs" / "run-i1757-video-batch-meiji"))

import benchmark  # noqa: E402


def main() -> int:
    all_attempts = benchmark.attempts
    if benchmark.SPEC["regimes"][0] != "first":
        raise ValueError("Expected the first regime of the pinned benchmark")

    def first_attempt(root: Path) -> list[Path]:
        return all_attempts(root)[:1]

    # The supervisor re-executes the launcher, so the worker uses the same attempt selection.
    with (
        patch.object(benchmark, "attempts", first_attempt),
        patch.object(benchmark, "__file__", __file__),
        patch.dict(benchmark.SPEC, {"conditions": {"serial": benchmark.SPEC["conditions"]["serial"]}}),
    ):
        return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
