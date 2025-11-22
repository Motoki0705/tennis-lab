"""Legacy CLI wrapper for Tennis Pose v2/v2.5/v3 training.

This module is kept for backward compatibility only. All logic has been
centralized in :mod:`src.cli.tennis_multi_cam_3d_pose.train`. Any direct
invocation of ``train_v2.py`` simply forwards arguments to ``train.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.cli.tennis_multi_cam_3d_pose.train import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the unified ``train.py`` entrypoint."""
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
