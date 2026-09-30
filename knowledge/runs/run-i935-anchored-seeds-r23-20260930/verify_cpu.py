"""Replay the r22 snapshot in an isolated process, preserving historical files."""

import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE.parent / "run-i935-precision-variants-s42-r21-20260930"))
import check_cpu_identity  # noqa: E402

check_cpu_identity.BUNDLE = BUNDLE / "cpu_identity"

if __name__ == "__main__":
    check_cpu_identity.main()
